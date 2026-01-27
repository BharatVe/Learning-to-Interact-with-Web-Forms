import re
import time
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

TYPE_DELAY_MS = 120
ACTION_DELAY_MS = 180

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


def move_mouse_to_element(page, target, event_logger=None, step_ref: Optional[int] = None, intent: Optional[str] = None) -> None:
    try:
        box = target.bounding_box()
    except PlaywrightTimeoutError:
        return
    if not box:
        return
    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2
    page.mouse.move(x, y, steps=25)
    if event_logger is not None and event_logger.record_hover:
        event_logger.log_event(
            "hover_at",
            args={},
            coords=event_logger.coords_from_values(x, y),
            target=target,
            step_ref=step_ref,
            intent=intent,
            outcome=True,
        )
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


def _get_scroll_state(page) -> Dict[str, Optional[float]]:
    try:
        data = page.evaluate("() => ({scroll_x: window.scrollX, scroll_y: window.scrollY})")
        return {"scroll_x": data.get("scroll_x"), "scroll_y": data.get("scroll_y")}
    except Exception:
        return {"scroll_x": None, "scroll_y": None}


def _capture_target_bbox(target_capture: Optional[Dict[str, Any]], page, target) -> None:
    if target_capture is None:
        return
    target_capture["bbox"] = safe_bbox(target)
    target_capture["scroll_y"] = _get_scroll_state(page).get("scroll_y")


class EventLogger:
    def __init__(self, page, start_time: float, record_hover: bool = True) -> None:
        self.page = page
        self.start_time = start_time
        self.events: List[Dict[str, Any]] = []
        self.record_hover = record_hover
        self.viewport_w: Optional[int] = None
        self.viewport_h: Optional[int] = None
        self._refresh_viewport()

    def _refresh_viewport(self) -> None:
        try:
            viewport = self.page.viewport_size or {}
            self.viewport_w = viewport.get("width")
            self.viewport_h = viewport.get("height")
        except Exception:
            self.viewport_w = None
            self.viewport_h = None

    def _now(self) -> float:
        return time.perf_counter() - self.start_time

    def coords_from_values(self, x_px: Optional[float], y_px: Optional[float]) -> Dict[str, Any]:
        if self.viewport_w is None or self.viewport_h is None:
            self._refresh_viewport()
        coords = {
            "x_px": x_px,
            "y_px": y_px,
            "x_norm": None,
            "y_norm": None,
            "viewport_w": self.viewport_w,
            "viewport_h": self.viewport_h,
        }
        if x_px is not None and self.viewport_w:
            x_norm = int(round((x_px / self.viewport_w) * 999))
            coords["x_norm"] = max(0, min(999, x_norm))
        if y_px is not None and self.viewport_h:
            y_norm = int(round((y_px / self.viewport_h) * 999))
            coords["y_norm"] = max(0, min(999, y_norm))
        return coords

    def coords_from_target(self, target) -> Dict[str, Any]:
        bbox = safe_bbox(target)
        if not bbox:
            return self.coords_from_values(None, None)
        x = bbox["x"] + bbox["width"] / 2
        y = bbox["y"] + bbox["height"] / 2
        return self.coords_from_values(x, y)

    def _grounding(self, target) -> Dict[str, Any]:
        grounding = {
            "target_bbox": safe_bbox(target) if target is not None else None,
            "target_desc": None,
            "role": None,
            "name": None,
            "aria_label": None,
            "placeholder": None,
            "label_text": None,
            "selector": None,
            "tag": None,
            "id": None,
        }
        if target is None:
            return grounding
        handle = None
        try:
            handle = target.element_handle()
        except Exception:
            handle = None
        if handle is None:
            return grounding
        try:
            info = handle.evaluate(
                """(el) => {
                    const tag = el.tagName ? el.tagName.toLowerCase() : null;
                    const role = el.getAttribute('role') || null;
                    const ariaLabel = el.getAttribute('aria-label') || null;
                    const name = el.getAttribute('name') || null;
                    const id = el.id || null;
                    const cls = typeof el.className === 'string' ? el.className : null;
                    const text = (el.innerText || '').trim().slice(0, 120) || null;
                    const placeholder = el.getAttribute('placeholder') || null;
                    const labelledBy = el.getAttribute('aria-labelledby') || null;
                    let labelText = null;
                    if (labelledBy) {
                      const ids = labelledBy.split(/\\s+/).filter(Boolean);
                      labelText = ids.map(id => {
                        const el = document.getElementById(id);
                        return el ? (el.innerText || '').trim() : '';
                      }).join(' ').trim() || null;
                    }
                    return { tag, role, ariaLabel, name, id, cls, text, placeholder, labelText };
                }"""
            )
        except Exception:
            return grounding
        grounding["role"] = info.get("role")
        grounding["aria_label"] = info.get("ariaLabel")
        grounding["placeholder"] = info.get("placeholder")
        grounding["label_text"] = info.get("labelText")
        grounding["tag"] = info.get("tag")
        grounding["id"] = info.get("id")
        grounding["name"] = info.get("name") or info.get("ariaLabel") or info.get("text")
        desc = (
            info.get("ariaLabel")
            or info.get("labelText")
            or info.get("placeholder")
            or info.get("text")
            or info.get("name")
        )
        grounding["target_desc"] = desc
        selector = None
        if info.get("id"):
            selector = f"#{info['id']}"
        elif info.get("name") and info.get("tag"):
            selector = f"{info['tag']}[name='{info['name']}']"
        elif info.get("cls") and info.get("tag"):
            first_class = str(info["cls"]).split()[0]
            selector = f"{info['tag']}.{first_class}"
        elif info.get("tag"):
            selector = info.get("tag")
        grounding["selector"] = selector
        return grounding

    @staticmethod
    def _has_grounding_data(grounding: Dict[str, Any]) -> bool:
        for key, value in grounding.items():
            if value is not None:
                return True
        return False

    def log_event(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        coords: Optional[Dict[str, Any]] = None,
        target=None,
        grounding: Optional[Dict[str, Any]] = None,
        scroll: Optional[Dict[str, Optional[float]]] = None,
        step_ref: Optional[int] = None,
        intent: Optional[str] = None,
        outcome: bool = True,
        error: Optional[str] = None,
    ) -> None:
        if coords is None and target is not None:
            coords = self.coords_from_target(target)
        if scroll is None:
            scroll = _get_scroll_state(self.page)
        if grounding is None and target is not None:
            grounding = self._grounding(target)
        event: Dict[str, Any] = {
            "t_s": self._now(),
            "name": name,
            "args": args or {},
            "scroll_x": scroll.get("scroll_x"),
            "scroll_y": scroll.get("scroll_y"),
            "step_ref": step_ref,
            "outcome": {"success": outcome, "error": error},
        }
        if coords is not None:
            x_px = coords.get("x_px") if isinstance(coords, dict) else None
            y_px = coords.get("y_px") if isinstance(coords, dict) else None
            if x_px is not None or y_px is not None:
                event["coords"] = coords
        if grounding is not None and self._has_grounding_data(grounding):
            event["grounding"] = grounding
        if intent:
            event["intent"] = intent
        self.events.append(event)


def scroll_into_view_with_log(page, target, event_logger=None, step_ref: Optional[int] = None, intent: Optional[str] = None) -> None:
    if event_logger is None:
        target.scroll_into_view_if_needed()
        return
    before = _get_scroll_state(page)
    target.scroll_into_view_if_needed()
    after = _get_scroll_state(page)
    if before and after:
        delta_x = None
        delta_y = None
        if before.get("scroll_x") is not None and after.get("scroll_x") is not None:
            delta_x = after["scroll_x"] - before["scroll_x"]
        if before.get("scroll_y") is not None and after.get("scroll_y") is not None:
            delta_y = after["scroll_y"] - before["scroll_y"]
        if delta_x or delta_y:
            event_logger.log_event(
                "scroll_document",
                args={"delta_x": delta_x, "delta_y": delta_y},
                target=target,
                step_ref=step_ref,
                intent=intent,
                outcome=True,
            )


def _click_with_log(page, target, event_logger=None, step_ref: Optional[int] = None, intent: Optional[str] = None) -> None:
    if event_logger is None:
        target.click()
        return
    coords = event_logger.coords_from_target(target)
    grounding = event_logger._grounding(target)
    scroll_state = _get_scroll_state(page)
    try:
        target.click()
    except Exception as exc:
        event_logger.log_event(
            "click_at",
            args={"button": "left"},
            coords=coords,
            target=target,
            grounding=grounding,
            scroll=scroll_state,
            step_ref=step_ref,
            intent=intent,
            outcome=False,
            error=str(exc),
        )
        raise
    event_logger.log_event(
        "click_at",
        args={"button": "left"},
        coords=coords,
        target=target,
        grounding=grounding,
        scroll=scroll_state,
        step_ref=step_ref,
        intent=intent,
        outcome=True,
    )


def _type_text_with_log(
    page,
    target,
    text: str,
    event_logger=None,
    step_ref: Optional[int] = None,
    intent: Optional[str] = None,
    clear_before_typing: bool = False,
    press_enter: bool = False,
    input_method: str = "type",
    delay: Optional[int] = None,
) -> None:
    if event_logger is None:
        if input_method == "fill":
            target.fill(text)
        elif input_method == "keyboard":
            page.keyboard.type(text, delay=delay)
        else:
            target.type(text, delay=delay)
        return
    coords = event_logger.coords_from_target(target)
    grounding = event_logger._grounding(target)
    scroll_state = _get_scroll_state(page)
    try:
        if input_method == "fill":
            target.fill(text)
        elif input_method == "keyboard":
            page.keyboard.type(text, delay=delay)
        else:
            target.type(text, delay=delay)
    except Exception as exc:
        event_logger.log_event(
            "type_text_at",
            args={
                "text": text,
                "press_enter": press_enter,
                "clear_before_typing": clear_before_typing,
                "input_method": input_method,
            },
            coords=coords,
            target=target,
            grounding=grounding,
            scroll=scroll_state,
            step_ref=step_ref,
            intent=intent,
            outcome=False,
            error=str(exc),
        )
        raise
    event_logger.log_event(
        "type_text_at",
        args={
            "text": text,
            "press_enter": press_enter,
            "clear_before_typing": clear_before_typing,
            "input_method": input_method,
        },
        coords=coords,
        target=target,
        grounding=grounding,
        scroll=scroll_state,
        step_ref=step_ref,
        intent=intent,
        outcome=True,
    )


def _press_key_with_log(
    page,
    keys: str,
    event_logger=None,
    step_ref: Optional[int] = None,
    intent: Optional[str] = None,
    target=None,
) -> None:
    if event_logger is None:
        page.keyboard.press(keys)
        return
    coords = event_logger.coords_from_target(target) if target is not None else None
    grounding = event_logger._grounding(target) if target is not None else None
    scroll_state = _get_scroll_state(page)
    try:
        page.keyboard.press(keys)
    except Exception as exc:
        event_logger.log_event(
            "key_combination",
            args={"keys": keys},
            coords=coords,
            target=target,
            grounding=grounding,
            scroll=scroll_state,
            step_ref=step_ref,
            intent=intent,
            outcome=False,
            error=str(exc),
        )
        raise
    event_logger.log_event(
        "key_combination",
        args={"keys": keys},
        coords=coords,
        target=target,
        grounding=grounding,
        scroll=scroll_state,
        step_ref=step_ref,
        intent=intent,
        outcome=True,
    )


def slow_fill(
    page,
    field,
    text: str,
    event_logger=None,
    step_ref: Optional[int] = None,
    intent: Optional[str] = None,
    target_capture: Optional[Dict[str, Any]] = None,
) -> None:
    move_mouse_to_element(page, field, event_logger=event_logger, step_ref=step_ref, intent=intent)
    _capture_target_bbox(target_capture, page, field)
    _click_with_log(page, field, event_logger=event_logger, step_ref=step_ref, intent=intent)
    if TYPE_DELAY_MS <= 0:
        _type_text_with_log(
            page,
            field,
            text,
            event_logger=event_logger,
            step_ref=step_ref,
            intent=intent,
            clear_before_typing=True,
            input_method="fill",
        )
    else:
        field.fill("")
        _type_text_with_log(
            page,
            field,
            text,
            event_logger=event_logger,
            step_ref=step_ref,
            intent=intent,
            clear_before_typing=True,
            input_method="type",
            delay=TYPE_DELAY_MS,
        )
    pause_after_action(page)


def type_segment_text(
    page,
    field,
    text: str,
    event_logger=None,
    step_ref: Optional[int] = None,
    intent: Optional[str] = None,
    target_capture: Optional[Dict[str, Any]] = None,
) -> None:
    move_mouse_to_element(page, field, event_logger=event_logger, step_ref=step_ref, intent=intent)
    _capture_target_bbox(target_capture, page, field)
    _click_with_log(page, field, event_logger=event_logger, step_ref=step_ref, intent=intent)
    field.fill("")
    delay = max(TYPE_DELAY_MS, 50) if TYPE_DELAY_MS else 50
    _type_text_with_log(
        page,
        field,
        text,
        event_logger=event_logger,
        step_ref=step_ref,
        intent=intent,
        clear_before_typing=True,
        input_method="type",
        delay=delay,
    )
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


def fill_text_question(
    page,
    container,
    value: Any,
    event_logger=None,
    step_ref: Optional[int] = None,
    intent: Optional[str] = None,
    target_capture: Optional[Dict[str, Any]] = None,
):
    scroll_into_view_with_log(page, container, event_logger=event_logger, step_ref=step_ref, intent=intent)
    field = container.locator(
        "textarea, input[type='text'], input[type='email'], input[type='url'], input[type='number']"
    )
    if field.count() == 0:
        raise ValueError("No text input found")
    slow_fill(
        page,
        field.first,
        str(value),
        event_logger=event_logger,
        step_ref=step_ref,
        intent=intent,
        target_capture=target_capture,
    )


def _normalize_option_text(s: str) -> str:
    return _normalize_text_for_match(s)


def select_option(
    page,
    container,
    value: str,
    role: str,
    event_logger=None,
    step_ref: Optional[int] = None,
    intent: Optional[str] = None,
    target_capture: Optional[Dict[str, Any]] = None,
) -> bool:
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
            scroll_into_view_with_log(
                page, option, event_logger=event_logger, step_ref=step_ref, intent=intent
            )
            move_mouse_to_element(page, option, event_logger=event_logger, step_ref=step_ref, intent=intent)
            _capture_target_bbox(target_capture, page, option)
            _click_with_log(page, option, event_logger=event_logger, step_ref=step_ref, intent=intent)
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
        scroll_into_view_with_log(
            page, candidate, event_logger=event_logger, step_ref=step_ref, intent=intent
        )
        move_mouse_to_element(page, candidate, event_logger=event_logger, step_ref=step_ref, intent=intent)
        _capture_target_bbox(target_capture, page, candidate)
        _click_with_log(page, candidate, event_logger=event_logger, step_ref=step_ref, intent=intent)
        pause_after_action(page)
        return True

    return False


def fill_single_choice_question(
    page,
    container,
    value: Any,
    event_logger=None,
    step_ref: Optional[int] = None,
    intent: Optional[str] = None,
    target_capture: Optional[Dict[str, Any]] = None,
):
    if not select_option(
        page,
        container,
        str(value),
        "radio",
        event_logger=event_logger,
        step_ref=step_ref,
        intent=intent,
        target_capture=target_capture,
    ):
        raise ValueError("Single choice option not found")


def fill_multi_choice_question(
    page,
    container,
    value: Any,
    event_logger=None,
    step_ref: Optional[int] = None,
    intent: Optional[str] = None,
    target_capture: Optional[Dict[str, Any]] = None,
):
    if isinstance(value, list):
        values = value
    else:
        values = [v.strip() for v in str(value).split(",") if v.strip()]
    for entry in values:
        if not select_option(
            page,
            container,
            str(entry),
            "checkbox",
            event_logger=event_logger,
            step_ref=step_ref,
            intent=intent,
            target_capture=target_capture,
        ):
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


def _build_intent(label: str, widget: str, value: Any) -> Optional[str]:
    if not label:
        return None
    if widget in {"single_choice", "multi_choice"}:
        if value is None:
            return f"Select {label}"
        if isinstance(value, list):
            joined = ", ".join(str(v) for v in value)
        else:
            joined = str(value)
        return f"Select {joined} for {label}"
    return f"Fill {label}"


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


def fill_date_question(
    page,
    container,
    value: Any,
    event_logger=None,
    step_ref: Optional[int] = None,
    intent: Optional[str] = None,
    target_capture: Optional[Dict[str, Any]] = None,
):
    dt = datetime.strptime(str(value), "%Y-%m-%d")
    scroll_into_view_with_log(page, container, event_logger=event_logger, step_ref=step_ref, intent=intent)

    # 1) Native single date input
    single_input = container.locator("input[type='date']")
    if single_input.count() > 0:
        field = single_input.first
        move_mouse_to_element(page, field, event_logger=event_logger, step_ref=step_ref, intent=intent)
        _capture_target_bbox(target_capture, page, field)
        _click_with_log(page, field, event_logger=event_logger, step_ref=step_ref, intent=intent)
        field.fill("")
        _type_text_with_log(
            page,
            field,
            dt.strftime("%Y-%m-%d"),
            event_logger=event_logger,
            step_ref=step_ref,
            intent=intent,
            clear_before_typing=True,
            input_method="fill",
        )
        pause_after_action(page)
        return

    # 2) Single masked text input: mm/dd/yyyy
    text_inputs = container.locator("input[type='text']")
    if text_inputs.count() == 1:
        field = text_inputs.first
        move_mouse_to_element(page, field, event_logger=event_logger, step_ref=step_ref, intent=intent)
        _capture_target_bbox(target_capture, page, field)
        _click_with_log(page, field, event_logger=event_logger, step_ref=step_ref, intent=intent)
        try:
            _press_key_with_log(
                page, "Control+A", event_logger=event_logger, step_ref=step_ref, intent=intent, target=field
            )
        except Exception:
            pass
        _press_key_with_log(
            page, "Backspace", event_logger=event_logger, step_ref=step_ref, intent=intent, target=field
        )
        formatted = dt.strftime("%m/%d/%Y")
        delay = max(TYPE_DELAY_MS, 50) if TYPE_DELAY_MS else 50
        _type_text_with_log(
            page,
            field,
            formatted,
            event_logger=event_logger,
            step_ref=step_ref,
            intent=intent,
            clear_before_typing=True,
            input_method="keyboard",
            delay=delay,
        )
        pause_after_action(page)
        return

    # 3) Segmented YYYY / MM / DD by labels/placeholders
    year_input, month_input, day_input = _detect_yyyy_mm_dd_inputs(container)
    if year_input and month_input and day_input:
        type_segment_text(
            page,
            year_input,
            str(dt.year),
            event_logger=event_logger,
            step_ref=step_ref,
            intent=intent,
            target_capture=target_capture,
        )
        type_segment_text(
            page,
            month_input,
            f"{dt.month}",
            event_logger=event_logger,
            step_ref=step_ref,
            intent=intent,
            target_capture=target_capture,
        )
        type_segment_text(
            page,
            day_input,
            f"{dt.day}",
            event_logger=event_logger,
            step_ref=step_ref,
            intent=intent,
            target_capture=target_capture,
        )
        return

    # 4) Segmented by positional order (assume first three are year, month, day)
    ordered_inputs = _ordered_text_number_inputs(container)
    if len(ordered_inputs) >= 3:
        year_input = ordered_inputs[0]
        month_input = ordered_inputs[1]
        day_input = ordered_inputs[2]

        type_segment_text(
            page,
            year_input,
            str(dt.year),
            event_logger=event_logger,
            step_ref=step_ref,
            intent=intent,
            target_capture=target_capture,
        )
        type_segment_text(
            page,
            month_input,
            f"{dt.month}",
            event_logger=event_logger,
            step_ref=step_ref,
            intent=intent,
            target_capture=target_capture,
        )
        type_segment_text(
            page,
            day_input,
            f"{dt.day}",
            event_logger=event_logger,
            step_ref=step_ref,
            intent=intent,
            target_capture=target_capture,
        )
        return

    # 5) Final fallback: generic segmented logic
    segments = [
        (["year", "yy", "yyyy"], str(dt.year)),
        (["month", "mm"], f"{dt.month}"),
        (["day", "dd"], f"{dt.day}"),
    ]
    _fill_segmented_inputs(
        page,
        container,
        segments,
        event_logger=event_logger,
        step_ref=step_ref,
        intent=intent,
        target_capture=target_capture,
    )


def fill_time_question(
    page,
    container,
    value: Any,
    event_logger=None,
    step_ref: Optional[int] = None,
    intent: Optional[str] = None,
    target_capture: Optional[Dict[str, Any]] = None,
):
    tm = datetime.strptime(str(value), "%H:%M")
    segments = [(["hour"], str(tm.hour)), (["minute"], f"{tm.minute:02d}")]
    scroll_into_view_with_log(page, container, event_logger=event_logger, step_ref=step_ref, intent=intent)
    _fill_segmented_inputs(
        page,
        container,
        segments,
        event_logger=event_logger,
        step_ref=step_ref,
        intent=intent,
        target_capture=target_capture,
    )


def _fill_segmented_inputs(
    page,
    container,
    segments: List[Any],
    event_logger=None,
    step_ref: Optional[int] = None,
    intent: Optional[str] = None,
    target_capture: Optional[Dict[str, Any]] = None,
):
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
        slow_fill(
            page,
            target,
            text,
            event_logger=event_logger,
            step_ref=step_ref,
            intent=intent,
            target_capture=target_capture,
        )


def process_entry(
    page,
    entry: Dict[str, Any],
    start_time: float,
    step_idx: int,
    event_logger: Optional[EventLogger] = None,
) -> Dict[str, Any]:
    label = entry.get("label", "")
    widget = entry.get("widget_type", "")
    value = entry.get("value")
    t0 = time.perf_counter() - start_time
    intent = _build_intent(label, widget, value)
    result = {
        "step": step_idx,
        "label": label,
        "widget_type": widget,
        "value": value,
        "success": False,
        "t_start_s": t0,
        "t_end_s": t0,
        "bbox": None,
        "intent": intent,
        "scroll_y": None,
        "target_bbox": None,
        "target_scroll_y": None,
    }
    if not label or not widget:
        result["t_end_s"] = time.perf_counter() - start_time
        return result
    container = find_question_container(page, label)
    if container is None:
        result["t_end_s"] = time.perf_counter() - start_time
        return result
    result["bbox"] = safe_bbox(container)
    result["scroll_y"] = _get_scroll_state(page).get("scroll_y")
    target_capture: Dict[str, Any] = {"bbox": None, "scroll_y": None}
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
        handler(
            page,
            container,
            value,
            event_logger=event_logger,
            step_ref=step_idx,
            intent=intent,
            target_capture=target_capture,
        )
        result["success"] = True
    except Exception:
        result["success"] = False
    result["target_bbox"] = target_capture.get("bbox")
    result["target_scroll_y"] = target_capture.get("scroll_y")
    if result["target_scroll_y"] is not None:
        result["scroll_y"] = result["target_scroll_y"]
    result["t_end_s"] = time.perf_counter() - start_time
    return result


def fill_form(
    page,
    answers: List[Dict[str, Any]],
    start_time: float,
    pause_seconds: float,
    event_logger: Optional[EventLogger] = None,
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    for i, entry in enumerate(answers):
        actions.append(process_entry(page, entry, start_time, i, event_logger=event_logger))
        if pause_seconds > 0:
            page.wait_for_timeout(int(pause_seconds * 1000))
    return actions


def submit_form(page, start_time: float, event_logger: Optional[EventLogger] = None) -> Dict[str, Any]:
    t0 = time.perf_counter() - start_time
    submit_clicked = False
    used_bbox = None
    submit_intent = "Submit form"
    info: Dict[str, Any] = {
        "success": False,
        "t_start_s": t0,
        "t_end_s": t0,
        "bbox": None,
        "submit_clicked": False,
        "confirmation_method": None,
        "confirmation_text": None,
        "final_url": None,
    }
    for pattern in [re.compile("submit", re.I)]:
        try:
            button = page.get_by_role("button", name=pattern)
            scroll_into_view_with_log(page, button, event_logger=event_logger, intent=submit_intent)
            move_mouse_to_element(page, button, event_logger=event_logger, intent=submit_intent)
            used_bbox = safe_bbox(button)
            _click_with_log(page, button, event_logger=event_logger, intent=submit_intent)
            submit_clicked = True
            pause_after_action(page)
            break
        except PlaywrightTimeoutError:
            continue
    if not submit_clicked:
        try:
            locator = page.get_by_text("Submit", exact=False)
            scroll_into_view_with_log(page, locator, event_logger=event_logger, intent=submit_intent)
            move_mouse_to_element(page, locator, event_logger=event_logger, intent=submit_intent)
            used_bbox = safe_bbox(locator)
            _click_with_log(page, locator, event_logger=event_logger, intent=submit_intent)
            submit_clicked = True
            pause_after_action(page)
        except PlaywrightTimeoutError:
            submit_locator = page.locator("div[role='button']").filter(has_text="Submit")
            try:
                button = submit_locator.first
                scroll_into_view_with_log(page, button, event_logger=event_logger, intent=submit_intent)
                move_mouse_to_element(page, button, event_logger=event_logger, intent=submit_intent)
                used_bbox = safe_bbox(button)
                _click_with_log(page, button, event_logger=event_logger, intent=submit_intent)
                submit_clicked = True
                pause_after_action(page)
            except PlaywrightTimeoutError:
                t1 = time.perf_counter() - start_time
                info["t_end_s"] = t1
                info["bbox"] = used_bbox
                info["submit_clicked"] = submit_clicked
                info["final_url"] = page.url
                info["error"] = "submit_button_not_found"
                return info
    confirmation_texts = [
        "Response recorded",
        "Response has been recorded",
        "Thanks for submitting",
        "Your response has been recorded",
    ]
    for text in confirmation_texts:
        try:
            page.get_by_text(text, exact=False).wait_for(state="visible", timeout=8000)
            if event_logger is not None:
                event_logger.log_event(
                    "wait_5_seconds",
                    args={"method": "text", "text": text, "timeout_ms": 8000},
                    intent="Wait for submission confirmation",
                    outcome=True,
                )
            t1 = time.perf_counter() - start_time
            info["success"] = True
            info["t_end_s"] = t1
            info["bbox"] = used_bbox
            info["submit_clicked"] = submit_clicked
            info["confirmation_method"] = "text"
            info["confirmation_text"] = text
            info["final_url"] = page.url
            return info
        except PlaywrightTimeoutError:
            if event_logger is not None:
                event_logger.log_event(
                    "wait_5_seconds",
                    args={"method": "text", "text": text, "timeout_ms": 8000},
                    intent="Wait for submission confirmation",
                    outcome=False,
                    error="timeout",
                )
            continue
        except Exception:
            break
    try:
        page.wait_for_url(re.compile(r"formResponse", re.IGNORECASE), timeout=8000)
        if event_logger is not None:
            event_logger.log_event(
                "wait_5_seconds",
                args={"method": "url", "pattern": "formResponse", "timeout_ms": 8000},
                intent="Wait for submission confirmation",
                outcome=True,
            )
            event_logger.log_event(
                "navigate",
                args={"url": page.url},
                intent="Confirm submission navigation",
                outcome=True,
            )
        t1 = time.perf_counter() - start_time
        info["success"] = True
        info["t_end_s"] = t1
        info["bbox"] = used_bbox
        info["submit_clicked"] = submit_clicked
        info["confirmation_method"] = "url"
        info["final_url"] = page.url
        return info
    except PlaywrightTimeoutError:
        if event_logger is not None:
            event_logger.log_event(
                "wait_5_seconds",
                args={"method": "url", "pattern": "formResponse", "timeout_ms": 8000},
                intent="Wait for submission confirmation",
                outcome=False,
                error="timeout",
            )
        pass
    except Exception:
        pass
    t1 = time.perf_counter() - start_time
    info["t_end_s"] = t1
    info["bbox"] = used_bbox
    info["submit_clicked"] = submit_clicked
    info["final_url"] = page.url
    return info
