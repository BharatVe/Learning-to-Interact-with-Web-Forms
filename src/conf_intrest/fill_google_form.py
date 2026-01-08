"""
Automation helper for the Conference Interest (conf_interest) Google Form.
"""

import argparse
import json
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from playwright.sync_api import Playwright, TimeoutError as PlaywrightTimeoutError, sync_playwright

FORM_NAME = "Conference Interest"
FORM_IDENTIFIER = "conf_interest"
FORM_DEFAULT_URL = "https://docs.google.com/forms/d/e/1FAIpQLScdiEfCs_Ld5bCIuVilB_hDhlSQCaRCwcJJHihThWaYherK3g/viewform?usp=publish-editor"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ANSWERS_JSON = SCRIPT_DIR / "answers_conference.json"
DEFAULT_DATASET_ROOT = "data/forms"
DEFAULT_VIDEO_DIR = "videos"
DEFAULT_RUN_ID = None
DEFAULT_TYPE_DELAY = 150
DEFAULT_ACTION_DELAY = 300
DEFAULT_SLOW_MO = 250

TYPE_DELAY_MS = DEFAULT_TYPE_DELAY
ACTION_DELAY_MS = DEFAULT_ACTION_DELAY

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill a Google Form via Playwright")
    parser.add_argument("--form-url", default=FORM_DEFAULT_URL)
    parser.add_argument("--answers-json", default=DEFAULT_ANSWERS_JSON)
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--slow-mo", type=int, default=DEFAULT_SLOW_MO)
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    parser.add_argument("--type-delay", type=int, default=DEFAULT_TYPE_DELAY)
    parser.add_argument("--action-delay", type=int, default=DEFAULT_ACTION_DELAY)
    parser.add_argument("--form-id", default=FORM_IDENTIFIER)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser.parse_args()


def load_run_specifications(path: Path) -> List[Dict[str, Any]]:
    """
    Return run specifications loaded from the provided answers JSON file.
    Each specification contains a `run_suffix` (optional) and an `answers` list.
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return [{"run_suffix": None, "answers": data}]
    if not isinstance(data, dict):
        raise ValueError("answers-json must be a list or a dict describing runs")
    runs = data.get("runs")
    if runs is None:
        runs = data.get("multi_runs")
    if not isinstance(runs, list):
        raise ValueError("answers-json dict must contain a 'runs' list")
    specs: List[Dict[str, Any]] = []
    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"run entry at index {idx} must be an object")
        answers = run.get("answers")
        if not isinstance(answers, list):
            raise ValueError(f"run entry at index {idx} must contain an answers list")
        suffix = run.get("suffix") or run.get("run_suffix")
        specs.append({"run_suffix": suffix, "answers": answers})
    return specs


def resolve_dataset_root(args: argparse.Namespace) -> Path:
    dataset_root_value = args.dataset_root
    dataset_root = Path(dataset_root_value)
    # Maintain backward compatibility with --video-dir if dataset-root not overridden.
    if dataset_root_value == DEFAULT_DATASET_ROOT and args.video_dir != DEFAULT_VIDEO_DIR:
        dataset_root = Path(args.video_dir)
    return dataset_root.resolve()


def build_run_id(
    base_run_id: Optional[str], run_suffix: Optional[str], run_index: int, total_runs: int
) -> str:
    if base_run_id:
        run_id = base_run_id
        if total_runs > 1 and not run_suffix:
            run_id = f"{run_id}_{run_index + 1:02d}"
    else:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        run_id = timestamp
        if total_runs > 1 and not run_suffix:
            run_id = f"{run_id}_{run_index + 1:02d}"
    if run_suffix:
        run_id = f"{run_id}_{run_suffix}"
    return run_id


def create_run_directory(dataset_root: Path, form_id: str, run_id: str) -> Path:
    run_dir_name = f"{form_id}_{run_id}"
    run_dir = dataset_root / form_id / "runs" / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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


def run_single_form(
    playwright: Playwright,
    form_id: str,
    form_url: str,
    answers: List[Dict[str, Any]],
    run_dir: Path,
    run_id: str,
    args: argparse.Namespace,
    answers_source_path: Path,
) -> bool:
    run_dir = Path(run_dir)
    run_identifier = f"{form_id}_{run_id}"
    start_time = time.perf_counter()
    actions: List[Dict[str, Any]] = []
    submit_info: Dict[str, Any] = {"success": False, "t_start_s": None, "t_end_s": None, "bbox": None}
    submitted = False
    video_path: Optional[Path] = None
    answers_instance_path = run_dir / "answers_instance.json"
    answers_instance_path.write_text(json.dumps(answers, indent=2))
    captured_exception: Optional[Exception] = None
    raw_video_path: Optional[Path] = None
    page_video = None
    browser = None
    context = None
    page = None

    try:
        browser = playwright.chromium.launch(headless=False, slow_mo=args.slow_mo)
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            record_video_dir=str(run_dir),
        )
        context.add_init_script(CURSOR_OVERLAY_SCRIPT)
        page = context.new_page()
        page_video = page.video
        page.set_default_timeout(15000)
        page.goto(form_url, wait_until="domcontentloaded")
        page.wait_for_load_state("networkidle")
        page.mouse.move(10, 10)
        for i, entry in enumerate(answers):
            actions.append(process_entry(page, entry, start_time, i))
            if args.pause_seconds > 0:
                page.wait_for_timeout(int(args.pause_seconds * 1000))
        submit_info = submit_form(page, start_time)
        submitted = submit_info["success"]
        if args.pause_seconds > 0:
            page.wait_for_timeout(int(args.pause_seconds * 1000))
    except Exception as exc:
        captured_exception = exc
        if submit_info.get("t_start_s") is None:
            now = time.perf_counter() - start_time
            submit_info["t_start_s"] = now
            submit_info["t_end_s"] = now
            submit_info["error"] = str(exc)
    finally:
        if page:
            try:
                page.close()
            except Exception:
                pass
        if context:
            try:
                context.close()
            except Exception:
                pass
        if browser:
            try:
                browser.close()
            except Exception:
                pass

    if raw_video_path is None and page_video is not None:
        try:
            raw_video_path = Path(page_video.path())
        except Exception:
            raw_video_path = None

    if raw_video_path is None:
        try:
            candidate_videos = sorted(
                run_dir.glob("*.webm"), key=lambda p: p.stat().st_mtime, reverse=True
            )
            raw_video_path = candidate_videos[0] if candidate_videos else None
        except Exception:
            raw_video_path = None

    if raw_video_path and raw_video_path.exists():
        final_video_path = run_dir / f"{run_identifier}.webm"
        if raw_video_path != final_video_path:
            try:
                if final_video_path.exists():
                    final_video_path.unlink()
            except Exception:
                pass
            try:
                raw_video_path.rename(final_video_path)
            except Exception:
                pass
            else:
                video_path = final_video_path
        if video_path is None and raw_video_path.exists():
            video_path = raw_video_path
    elif raw_video_path:
        video_path = raw_video_path

    log = {
        "run_identifier": run_identifier,
        "run_id": run_id,
        "form_name": FORM_NAME,
        "form_id": form_id,
        "form_url": form_url,
        "run_directory": str(run_dir),
        "video_path": str(video_path) if video_path else None,
        "answers_source_path": str(answers_source_path),
        "answers_instance_path": str(answers_instance_path),
        "run_params": {
            "slow_mo": args.slow_mo,
            "type_delay_ms": TYPE_DELAY_MS,
            "action_delay_ms": ACTION_DELAY_MS,
            "pause_seconds": args.pause_seconds,
        },
        "actions": actions,
        "submit": submit_info,
        "submitted": submitted,
    }
    annotations_path = run_dir / "annotations.json"
    annotations_path.write_text(json.dumps(log, indent=2))

    if captured_exception:
        raise captured_exception
    return submitted


def main():
    args = parse_args()
    set_type_delay(args.type_delay)
    set_action_delay(args.action_delay)

    answers_path = Path(args.answers_json).resolve()
    run_specs = load_run_specifications(answers_path)
    dataset_root = resolve_dataset_root(args)
    form_id = args.form_id or FORM_IDENTIFIER
    form_url = args.form_url or FORM_DEFAULT_URL
    total_runs = len(run_specs)

    failed_runs: List[str] = []
    unconfirmed_runs: List[str] = []

    with sync_playwright() as playwright:
        for idx, spec in enumerate(run_specs):
            answers = spec.get("answers", [])
            if not isinstance(answers, list):
                raise ValueError("Each run specification must include an answers list")
            run_suffix = spec.get("run_suffix")
            run_id = build_run_id(args.run_id, run_suffix, idx, total_runs)
            run_dir = create_run_directory(dataset_root, form_id, run_id)
            try:
                submitted = run_single_form(
                    playwright=playwright,
                    form_id=form_id,
                    form_url=form_url,
                    answers=answers,
                    run_dir=run_dir,
                    run_id=run_id,
                    args=args,
                    answers_source_path=answers_path,
                )
            except Exception as exc:
                failed_runs.append(f"{run_id}: {exc}")
                continue
            if not submitted:
                unconfirmed_runs.append(run_id)

    if failed_runs:
        for message in failed_runs:
            print(f"[ERROR] Run failed: {message}", file=sys.stderr)
        sys.exit(1)
    if unconfirmed_runs:
        joined = ", ".join(unconfirmed_runs)
        print(f"[WARN] Submission not confirmed for runs: {joined}", file=sys.stderr)


if __name__ == "__main__":
    main()
