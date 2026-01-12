import re
import time
import unicodedata
from datetime import datetime
from typing import Any, Dict, List

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

TYPE_DELAY_MS = 150
ACTION_DELAY_MS = 200

CURSOR_OVERLAY_SCRIPT = """
(() => {
  if (window.__codexCursorOverlayInitialized) return;
  window.__codexCursorOverlayInitialized = true;

  const init = () => {
    if (window.__codexCursorOverlay) return;
    const cursor = document.createElement('div');
    cursor.id = '__codex_cursor';
    Object.assign(cursor.style, {
      position: 'fixed',
      width: '16px',
      height: '16px',
      border: '2px solid #ff4081',
      borderRadius: '50%',
      background: 'rgba(255, 61, 0, 0.35)',
      boxShadow: '0 0 8px rgba(255, 61, 0, 0.75)',
      pointerEvents: 'none',
      zIndex: '2147483647',
      transform: 'translate(-50%, -50%)',
      transition: 'top 70ms linear, left 70ms linear'
    });
    document.body.appendChild(cursor);
    window.addEventListener(
      'mousemove',
      (event) => {
        cursor.style.left = `${event.clientX}px`;
        cursor.style.top = `${event.clientY}px`;
      },
      true
    );
    window.__codexCursorOverlay = cursor;
  };

  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
"""


def set_type_delay(ms: int) -> None:
    global TYPE_DELAY_MS
    TYPE_DELAY_MS = max(0, ms)


def set_action_delay(ms: int) -> None:
    global ACTION_DELAY_MS
    ACTION_DELAY_MS = max(0, ms)


def pause_after_action(page) -> None:
    if ACTION_DELAY_MS > 0:
        page.wait_for_timeout(ACTION_DELAY_MS)


def move_mouse_to_element(page, target) -> None:
    try:
        box = target.bounding_box()
    except PlaywrightTimeoutError:
        return
    if not box:
        return
    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2
    page.mouse.move(x, y, steps=25)
    pause_after_action(page)


def safe_bbox(target):
    try:
        box = target.bounding_box()
    except PlaywrightTimeoutError:
        return None
    except Exception:
        return None
    if not box:
        return None
    return {
        "x": box.get("x"),
        "y": box.get("y"),
        "width": box.get("width"),
        "height": box.get("height"),
    }


def slow_fill(page, field, text: str) -> None:
    move_mouse_to_element(page, field)
    field.click()
    if TYPE_DELAY_MS <= 0:
        field.fill(text)
    else:
        field.fill("")
        field.type(text, delay=TYPE_DELAY_MS)
    pause_after_action(page)


def type_segment_text(page, field, text: str) -> None:
    move_mouse_to_element(page, field)
    field.click()
    field.fill("")
    delay = max(TYPE_DELAY_MS, 50) if TYPE_DELAY_MS else 50
    field.type(text, delay=delay)
    pause_after_action(page)


def _normalize_text_for_match(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return "".join(ch for ch in s if ch.isalnum() or ch.isspace()).strip()


def find_question_container(page, label: str):
    locator = page.locator("div[role='listitem']")
    count = locator.count()
    target_norm = _normalize_text_for_match(label)
    if not target_norm:
        return None
    for idx in range(count):
        item = locator.nth(idx)
        try:
            text = item.inner_text()
        except PlaywrightTimeoutError:
            continue
        text_norm = _normalize_text_for_match(text)
        if target_norm and target_norm in text_norm:
            return item
    return None


def fill_text_question(page, container, value: Any):
    container.scroll_into_view_if_needed()
    field = container.locator(
        "textarea, input[type='text'], input[type='email'], input[type='url'], input[type='number']"
    )
    if field.count() == 0:
        raise ValueError("No text input found")
    slow_fill(page, field.first, str(value))


def _normalize_option_text(s: str) -> str:
    return _normalize_text_for_match(s)


def select_option(page, container, value: str, role: str) -> bool:
    """
    Try to select an option by:
    1) matching elements with the given role (radio/checkbox) by normalized text,
    2) if that fails, clicking the text node itself inside the question.
    """
    target_norm = _normalize_option_text(value)
    if not target_norm:
        return False

    # First: elements with the given role
    options = container.locator(f"[role='{role}']")
    count = options.count()
    for idx in range(count):
        option = options.nth(idx)
        try:
            label = option.get_attribute("aria-label") or option.inner_text()
        except PlaywrightTimeoutError:
            continue
        if not label:
            continue
        label_norm = _normalize_option_text(label)
        if target_norm in label_norm:
            option.scroll_into_view_if_needed()
            move_mouse_to_element(page, option)
            option.click()
            pause_after_action(page)
            return True

    # Fallback: click the visible text itself
    # (on Google Forms, clicking the label text also toggles the checkbox/radio)
    text_candidates = container.get_by_text(value, exact=False)
    tcount = text_candidates.count()
    for idx in range(tcount):
        candidate = text_candidates.nth(idx)
        try:
            label = candidate.inner_text()
        except PlaywrightTimeoutError:
            continue
        if target_norm not in _normalize_option_text(label):
            continue
        candidate.scroll_into_view_if_needed()
        move_mouse_to_element(page, candidate)
        candidate.click()
        pause_after_action(page)
        return True

    return False


def fill_single_choice_question(page, container, value: Any):
    if not select_option(page, container, str(value), "radio"):
        raise ValueError("Single choice option not found")


def fill_multi_choice_question(page, container, value: Any):
    if isinstance(value, list):
        values = value
    else:
        values = [v.strip() for v in str(value).split(",") if v.strip()]
    for entry in values:
        if not select_option(page, container, str(entry), "checkbox"):
            raise ValueError(f"Multi choice option not found for {entry}")


def find_input_by_keywords(container, keywords: List[str]):
    lowered = [word.lower() for word in keywords]
    inputs = container.locator("input")
    count = inputs.count()
    for idx in range(count):
        field = inputs.nth(idx)
        label = (
            field.get_attribute("aria-label")
            or field.get_attribute("placeholder")
            or ""
        )
        label_lower = label.lower()
        if label and any(word in label_lower for word in lowered):
            return field
    return None


def _detect_yyyy_mm_dd_inputs(container):
    inputs = container.locator("input")
    count = inputs.count()
    year_input = month_input = day_input = None
    for idx in range(count):
        field = inputs.nth(idx)
        label = (
            field.get_attribute("aria-label")
            or field.get_attribute("placeholder")
            or ""
        )
        l = label.lower()
        if not l:
            continue
        if "y" in l and year_input is None:
            year_input = field
        elif "m" in l and month_input is None:
            month_input = field
        elif "d" in l and day_input is None:
            day_input = field
    return year_input, month_input, day_input


def _ordered_text_number_inputs(container):
    loc = container.locator("input[type='text'], input[type='number']")
    count = loc.count()
    items = []
    for idx in range(count):
        field = loc.nth(idx)
        try:
            box = field.bounding_box()
        except PlaywrightTimeoutError:
            continue
        if not box:
            continue
        items.append((box["y"], box["x"], field))
    items.sort(key=lambda t: (t[0], t[1]))
    return [field for _, _, field in items]


def fill_date_question(page, container, value: Any):
    dt = datetime.strptime(str(value), "%Y-%m-%d")

    # 1) Native single date input
    single_input = container.locator("input[type='date']")
    if single_input.count() > 0:
        field = single_input.first
        move_mouse_to_element(page, field)
        field.click()
        field.fill("")
        field.fill(dt.strftime("%Y-%m-%d"))
        pause_after_action(page)
        return

    # 2) Single masked text input: mm/dd/yyyy
    text_inputs = container.locator("input[type='text']")
    if text_inputs.count() == 1:
        field = text_inputs.first
        move_mouse_to_element(page, field)
        field.click()
        try:
            page.keyboard.press("Control+A")
        except Exception:
            pass
        page.keyboard.press("Backspace")
        formatted = dt.strftime("%m/%d/%Y")
        delay = max(TYPE_DELAY_MS, 50) if TYPE_DELAY_MS else 50
        page.keyboard.type(formatted, delay=delay)
        pause_after_action(page)
        return

    # 3) Segmented YYYY / MM / DD by labels/placeholders
    year_input, month_input, day_input = _detect_yyyy_mm_dd_inputs(container)
    if year_input and month_input and day_input:
        type_segment_text(page, year_input, str(dt.year))
        type_segment_text(page, month_input, f"{dt.month}")
        type_segment_text(page, day_input, f"{dt.day}")
        return

    # 4) Segmented by positional order (assume first three are year, month, day)
    ordered_inputs = _ordered_text_number_inputs(container)
    if len(ordered_inputs) >= 3:
        year_input = ordered_inputs[0]
        month_input = ordered_inputs[1]
        day_input = ordered_inputs[2]

        type_segment_text(page, year_input, str(dt.year))
        type_segment_text(page, month_input, f"{dt.month}")
        type_segment_text(page, day_input, f"{dt.day}")
        return

    # 5) Final fallback: generic segmented logic
    segments = [
        (["year", "yy", "yyyy"], str(dt.year)),
        (["month", "mm"], f"{dt.month}"),
        (["day", "dd"], f"{dt.day}"),
    ]
    _fill_segmented_inputs(page, container, segments)


def fill_time_question(page, container, value: Any):
    tm = datetime.strptime(str(value), "%H:%M")
    segments = [(["hour"], str(tm.hour)), (["minute"], f"{tm.minute:02d}")]
    _fill_segmented_inputs(page, container, segments)


def _fill_segmented_inputs(page, container, segments: List[Any]):
    fallback_inputs = container.locator("input[type='text'], input[type='number']")
    fallback_total = fallback_inputs.count()
    fallback_idx = 0
    for keywords, text in segments:
        field = find_input_by_keywords(container, keywords)
        target = field
        if target is None:
            if fallback_idx >= fallback_total:
                raise ValueError("Not enough date/time inputs")
            target = fallback_inputs.nth(fallback_idx)
            fallback_idx += 1
        slow_fill(page, target, text)


def process_entry(page, entry: Dict[str, Any], start_time: float, step_idx: int) -> Dict[str, Any]:
    label = entry.get("label", "")
    widget = entry.get("widget_type", "")
    value = entry.get("value")
    t0 = time.perf_counter() - start_time
    result = {
        "step": step_idx,
        "label": label,
        "widget_type": widget,
        "value": value,
        "success": False,
        "t_start_s": t0,
        "t_end_s": t0,
        "bbox": None,
    }
    if not label or not widget:
        result["t_end_s"] = time.perf_counter() - start_time
        return result
    container = find_question_container(page, label)
    if container is None:
        result["t_end_s"] = time.perf_counter() - start_time
        return result
    result["bbox"] = safe_bbox(container)
    handlers = {
        "short_text": fill_text_question,
        "paragraph_text": fill_text_question,
        "single_choice": fill_single_choice_question,
        "multi_choice": fill_multi_choice_question,
        "date": fill_date_question,
        "time": fill_time_question,
    }
    handler = handlers.get(widget)
    if not handler:
        result["t_end_s"] = time.perf_counter() - start_time
        return result
    try:
        handler(page, container, value)
        result["success"] = True
    except Exception:
        result["success"] = False
    result["t_end_s"] = time.perf_counter() - start_time
    return result


def fill_form(page, answers: List[Dict[str, Any]], start_time: float, pause_seconds: float) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    for i, entry in enumerate(answers):
        actions.append(process_entry(page, entry, start_time, i))
        if pause_seconds > 0:
            page.wait_for_timeout(int(pause_seconds * 1000))
    return actions


def submit_form(page, start_time: float) -> Dict[str, Any]:
    t0 = time.perf_counter() - start_time
    submit_clicked = False
    used_bbox = None
    for pattern in [re.compile("submit", re.I)]:
        try:
            button = page.get_by_role("button", name=pattern)
            move_mouse_to_element(page, button)
            button.click()
            submit_clicked = True
            used_bbox = safe_bbox(button)
            pause_after_action(page)
            break
        except PlaywrightTimeoutError:
            continue
    if not submit_clicked:
        try:
            locator = page.get_by_text("Submit", exact=False)
            move_mouse_to_element(page, locator)
            locator.click()
            submit_clicked = True
            used_bbox = safe_bbox(locator)
            pause_after_action(page)
        except PlaywrightTimeoutError:
            submit_locator = page.locator("div[role='button']").filter(has_text="Submit")
            try:
                button = submit_locator.first
                move_mouse_to_element(page, button)
                button.click()
                submit_clicked = True
                used_bbox = safe_bbox(button)
                pause_after_action(page)
            except PlaywrightTimeoutError:
                t1 = time.perf_counter() - start_time
                return {"success": False, "t_start_s": t0, "t_end_s": t1, "bbox": used_bbox}
    confirmation_texts = [
        "Response recorded",
        "Response has been recorded",
        "Thanks for submitting",
        "Your response has been recorded",
    ]
    for text in confirmation_texts:
        try:
            page.get_by_text(text, exact=False).wait_for(state="visible", timeout=8000)
            t1 = time.perf_counter() - start_time
            return {"success": True, "t_start_s": t0, "t_end_s": t1, "bbox": used_bbox}
        except PlaywrightTimeoutError:
            continue
        except Exception:
            break
    try:
        page.wait_for_url(re.compile(r"formResponse", re.IGNORECASE), timeout=8000)
        t1 = time.perf_counter() - start_time
        return {"success": True, "t_start_s": t0, "t_end_s": t1, "bbox": used_bbox}
    except PlaywrightTimeoutError:
        pass
    except Exception:
        pass
    t1 = time.perf_counter() - start_time
    return {"success": False, "t_start_s": t0, "t_end_s": t1, "bbox": used_bbox}
