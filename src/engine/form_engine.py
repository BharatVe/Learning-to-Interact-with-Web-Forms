import datetime as dt
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from engine.trace_logger import TraceLogger


CURSOR_OVERLAY_SCRIPT = r"""
(() => {
  if (window.__thesisCursorOverlayInitialized) return;
  window.__thesisCursorOverlayInitialized = true;

  const init = () => {
    if (window.__thesisCursorOverlay) return;
    const cursor = document.createElement('div');
    cursor.id = '__thesis_cursor_overlay';
    Object.assign(cursor.style, {
      position: 'fixed',
      width: '16px',
      height: '16px',
      border: '2px solid #ff2d55',
      borderRadius: '50%',
      background: 'rgba(255,45,85,0.22)',
      boxShadow: '0 0 10px rgba(255,45,85,0.75)',
      pointerEvents: 'none',
      zIndex: '2147483647',
      transform: 'translate(-50%, -50%)',
      transition: 'top 70ms linear, left 70ms linear'
    });
    document.body.appendChild(cursor);

    window.addEventListener('mousemove', (event) => {
      cursor.style.left = `${event.clientX}px`;
      cursor.style.top = `${event.clientY}px`;
    }, true);

    window.__thesisCursorOverlay = cursor;
  };

  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
"""


def _norm_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").lower()
    value = value.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


class FormEngine:
    def __init__(
        self,
        page,
        viewport: Dict[str, int],
        observations_dir: Path,
        trace: TraceLogger,
        timeout_ms: int = 15000,
        type_delay_ms: int = 120,
        action_delay_ms: int = 220,
        take_screenshots: bool = True,
    ) -> None:
        self.page = page
        self.viewport = viewport
        self.observations_dir = observations_dir
        self.trace = trace
        self.timeout_ms = timeout_ms
        self.type_delay_ms = max(0, type_delay_ms)
        self.action_delay_ms = max(0, action_delay_ms)
        self.take_screenshots = take_screenshots
        self.long_text_threshold = 90
        self.long_text_max_delay_ms = 35
        self.handlers = {
            "short_text": self._handle_text,
            "paragraph_text": self._handle_text,
            "single_choice": self._handle_single_choice,
            "multi_choice": self._handle_multi_choice,
            "date": self._handle_date,
            "time": self._handle_time,
        }

    def enable_mouse_overlay(self) -> None:
        self.page.evaluate(CURSOR_OVERLAY_SCRIPT)

    def _coords_norm(self, x: float, y: float) -> Tuple[int, int]:
        width = max(int(self.viewport.get("width", 1)), 1)
        height = max(int(self.viewport.get("height", 1)), 1)
        x_norm = int(round((x / width) * 999))
        y_norm = int(round((y / height) * 999))
        return max(0, min(999, x_norm)), max(0, min(999, y_norm))

    def _get_scroll_state(self) -> Dict[str, Optional[float]]:
        try:
            data = self.page.evaluate("() => ({scroll_x: window.scrollX, scroll_y: window.scrollY})")
            return {"scroll_x": data.get("scroll_x"), "scroll_y": data.get("scroll_y")}
        except Exception:
            return {"scroll_x": None, "scroll_y": None}

    def _event_extra(
        self,
        scroll: Optional[Dict[str, Optional[float]]] = None,
        x_px: Optional[float] = None,
        y_px: Optional[float] = None,
    ) -> Dict[str, Any]:
        scroll = scroll or {}
        return {
            "viewport_w": self.viewport.get("width"),
            "viewport_h": self.viewport.get("height"),
            "scroll_x": scroll.get("scroll_x"),
            "scroll_y": scroll.get("scroll_y"),
            "x_px": x_px,
            "y_px": y_px,
        }

    def _pause(self) -> None:
        if self.action_delay_ms > 0:
            self.page.wait_for_timeout(self.action_delay_ms)

    def _bbox(self, locator) -> Optional[Dict[str, float]]:
        try:
            box = locator.bounding_box()
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

    def _center(self, locator) -> Tuple[Optional[float], Optional[float]]:
        box = self._bbox(locator)
        if not box:
            return None, None
        return float(box["x"] + box["width"] / 2.0), float(box["y"] + box["height"] / 2.0)

    def _screenshot(self, name: str) -> Optional[str]:
        if not self.take_screenshots:
            return None
        self.observations_dir.mkdir(parents=True, exist_ok=True)
        abs_path = (self.observations_dir / name).resolve()
        self.page.screenshot(path=str(abs_path))
        return str(abs_path)

    def get_page_text(self) -> str:
        try:
            return self.page.locator("body").inner_text(timeout=min(self.timeout_ms, 3000))
        except Exception:
            return ""

    def take_observation_screenshot(self, filename: str) -> Optional[str]:
        return self._screenshot(filename)

    def _log_scroll_if_needed(
        self,
        before: Dict[str, Optional[float]],
        after: Dict[str, Optional[float]],
        step_ref: Optional[int],
    ) -> None:
        before_y = before.get("scroll_y")
        after_y = after.get("scroll_y")
        if before_y is None or after_y is None:
            return
        delta = after_y - before_y
        if abs(delta) < 1:
            return
        self.trace.log_event(
            "browser_mouse_wheel",
            {"deltaX": 0, "deltaY": int(round(delta))},
            step_ref=step_ref,
            extra=self._event_extra(after),
        )

    def _scroll_into_view(self, locator, step_ref: Optional[int]) -> None:
        before = self._get_scroll_state()
        locator.scroll_into_view_if_needed(timeout=self.timeout_ms)
        after = self._get_scroll_state()
        self._log_scroll_if_needed(before, after, step_ref)

    def _hover(self, locator, step_ref: Optional[int]) -> None:
        x, y = self._center(locator)
        scroll = self._get_scroll_state()
        try:
            if x is None or y is None:
                raise RuntimeError("target_not_visible")
            x_norm, y_norm = self._coords_norm(x, y)
            self.page.mouse.move(x, y, steps=20)
            self.trace.log_event(
                "browser_mouse_move_xy",
                {"x": x_norm, "y": y_norm},
                step_ref=step_ref,
                extra=self._event_extra(scroll, x_px=x, y_px=y),
            )
            self._pause()
        except Exception as exc:
            self.trace.log_event(
                "browser_mouse_move_xy",
                {"x": None, "y": None},
                step_ref=step_ref,
                ok=False,
                error=str(exc),
                extra=self._event_extra(scroll, x_px=x, y_px=y),
            )
            raise

    def _click(self, locator, step_ref: Optional[int]) -> None:
        x, y = self._center(locator)
        scroll = self._get_scroll_state()
        try:
            if x is not None and y is not None:
                x_norm, y_norm = self._coords_norm(x, y)
                args = {"x": x_norm, "y": y_norm}
            else:
                args = {"x": None, "y": None}
            locator.click(timeout=self.timeout_ms)
            self.trace.log_event(
                "browser_mouse_click_xy",
                args,
                step_ref=step_ref,
                extra=self._event_extra(scroll, x_px=x, y_px=y),
            )
            self._pause()
        except Exception as exc:
            self.trace.log_event(
                "browser_mouse_click_xy",
                {"x": None, "y": None},
                step_ref=step_ref,
                ok=False,
                error=str(exc),
                extra=self._event_extra(scroll, x_px=x, y_px=y),
            )
            raise

    def _type(self, locator, text: str, step_ref: Optional[int], clear_before_typing: bool = True) -> None:
        x, y = self._center(locator)
        scroll = self._get_scroll_state()
        try:
            if clear_before_typing:
                locator.fill("", timeout=self.timeout_ms)
            text_str = str(text)
            text_len = len(text_str)
            effective_delay = self.type_delay_ms
            if text_len > self.long_text_threshold:
                effective_delay = min(self.type_delay_ms, self.long_text_max_delay_ms)
            dynamic_timeout = max(self.timeout_ms, 3000 + (text_len * max(effective_delay, 8)))
            args = {"text": text_str, "slowly": effective_delay > 0, "submit": False}
            locator.type(text_str, delay=effective_delay, timeout=dynamic_timeout)
            self.trace.log_event(
                "browser_type",
                args,
                step_ref=step_ref,
                extra=self._event_extra(scroll, x_px=x, y_px=y),
            )
            self._pause()
        except Exception as exc:
            self.trace.log_event(
                "browser_type",
                {"text": str(text), "slowly": self.type_delay_ms > 0, "submit": False},
                step_ref=step_ref,
                ok=False,
                error=str(exc),
                extra=self._event_extra(scroll, x_px=x, y_px=y),
            )
            raise

    def _fill_direct(self, locator, text: str, step_ref: Optional[int], clear_before_typing: bool = True) -> None:
        x, y = self._center(locator)
        scroll = self._get_scroll_state()
        try:
            if clear_before_typing:
                locator.fill("", timeout=self.timeout_ms)
            locator.fill(str(text), timeout=self.timeout_ms)
            args = {"text": str(text), "slowly": False, "submit": False}
            self.trace.log_event(
                "browser_type",
                args,
                step_ref=step_ref,
                extra=self._event_extra(scroll, x_px=x, y_px=y),
            )
            self._pause()
        except Exception as exc:
            self.trace.log_event(
                "browser_type",
                {"text": str(text), "slowly": False, "submit": False},
                step_ref=step_ref,
                ok=False,
                error=str(exc),
                extra=self._event_extra(scroll, x_px=x, y_px=y),
            )
            raise

    def _find_question_container(self, label: str):
        target = _norm_text(label)
        if not target:
            return None
        items = self.page.locator("div[role='listitem']")
        count = items.count()
        for idx in range(count):
            item = items.nth(idx)
            try:
                text = item.inner_text(timeout=1000)
            except Exception:
                continue
            if target in _norm_text(text):
                return item
        return None

    def _find_button_by_name(self, pattern: re.Pattern[str]):
        candidates = [
            self.page.get_by_role("button", name=pattern),
            self.page.locator("div[role='button'], button").filter(has_text=pattern),
        ]
        for locator in candidates:
            try:
                count = locator.count()
            except Exception:
                continue
            for idx in range(count):
                button = locator.nth(idx)
                try:
                    if button.is_visible():
                        return button
                except Exception:
                    continue
        return None

    def _click_next_page(self, step_ref: Optional[int]) -> bool:
        button = self._find_button_by_name(re.compile(r"^(next|continue|weiter)$", re.I))
        if button is None:
            button = self._find_button_by_name(re.compile(r"next|continue|weiter", re.I))
        if button is None:
            return False
        self._scroll_into_view(button, step_ref)
        self._hover(button, step_ref)
        self._click(button, step_ref)
        self.page.wait_for_timeout(max(self.action_delay_ms, 250))
        return True

    def _find_container_with_pagination(
        self,
        label: str,
        step_ref: Optional[int],
        max_page_hops: int = 4,
    ):
        container = self._find_question_container(label)
        if container is not None:
            return container
        for _ in range(max_page_hops):
            moved = self._click_next_page(step_ref)
            if not moved:
                break
            container = self._find_question_container(label)
            if container is not None:
                return container
        return None

    def _find_input_by_keywords(self, container, keywords: List[str]):
        lowered = [kw.lower() for kw in keywords]
        inputs = container.locator("input")
        count = inputs.count()
        for idx in range(count):
            field = inputs.nth(idx)
            try:
                label = field.get_attribute("aria-label") or field.get_attribute("placeholder") or ""
            except Exception:
                label = ""
            normalized = label.lower()
            if normalized and any(kw in normalized for kw in lowered):
                return field
        return None

    def _ordered_inputs(self, container):
        fields = container.locator("input[type='text'], input[type='number']")
        count = fields.count()
        items: List[Tuple[float, float, Any]] = []
        for idx in range(count):
            field = fields.nth(idx)
            box = self._bbox(field)
            if not box:
                continue
            items.append((float(box["y"]), float(box["x"]), field))
        items.sort(key=lambda entry: (entry[0], entry[1]))
        return [entry[2] for entry in items]

    def _normalize_date(self, value: Any) -> dt.datetime:
        raw = str(value).strip()
        fmts = ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%d-%m-%Y"]
        for fmt in fmts:
            try:
                return dt.datetime.strptime(raw, fmt)
            except ValueError:
                continue
        raise RuntimeError(f"invalid_date_value: {value}")

    def _format_date_for_text_field(self, parsed: dt.datetime, field) -> str:
        hint = ""
        try:
            hint = (
                field.get_attribute("placeholder")
                or field.get_attribute("aria-label")
                or field.get_attribute("name")
                or ""
            ).lower()
        except Exception:
            hint = ""

        sep = "/"
        if "-" in hint:
            sep = "-"
        elif "." in hint:
            sep = "."

        dd_idx = hint.find("dd")
        mm_idx = hint.find("mm")
        yyyy_idx = hint.find("yyyy")
        if yyyy_idx == -1:
            yyyy_idx = hint.find("year")
        if dd_idx == -1:
            dd_idx = hint.find("day")
        if mm_idx == -1:
            mm_idx = hint.find("month")

        # Prefer explicit hint order if present.
        if yyyy_idx != -1 and mm_idx != -1 and dd_idx != -1:
            ordered = sorted(
                [("yyyy", yyyy_idx), ("mm", mm_idx), ("dd", dd_idx)],
                key=lambda item: item[1],
            )
            parts = {
                "yyyy": f"{parsed.year:04d}",
                "mm": f"{parsed.month:02d}",
                "dd": f"{parsed.day:02d}",
            }
            return sep.join(parts[token] for token, _ in ordered)

        if dd_idx != -1 and mm_idx != -1 and dd_idx < mm_idx:
            return f"{parsed.day:02d}{sep}{parsed.month:02d}{sep}{parsed.year:04d}"
        if mm_idx != -1 and dd_idx != -1 and mm_idx < dd_idx:
            return f"{parsed.month:02d}{sep}{parsed.day:02d}{sep}{parsed.year:04d}"

        # Default for plain text mask.
        return f"{parsed.month:02d}{sep}{parsed.day:02d}{sep}{parsed.year:04d}"

    def _normalize_time(self, value: Any) -> dt.datetime:
        raw = str(value).strip().lower().replace(".", "")
        fmts = ["%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p"]
        for fmt in fmts:
            try:
                return dt.datetime.strptime(raw, fmt)
            except ValueError:
                continue
        raise RuntimeError(f"invalid_time_value: {value}")

    def _set_action_target(self, action: Dict[str, Any], target) -> None:
        action["target_bbox"] = self._bbox(target)
        try:
            action["target_role"] = target.get_attribute("role")
            action["target_name"] = target.get_attribute("aria-label") or target.get_attribute("placeholder")
            target_id = target.get_attribute("id")
            action["target_selector"] = f"#{target_id}" if target_id else None
        except Exception:
            pass

    def _read_input_value(self, field) -> str:
        try:
            return field.input_value(timeout=self.timeout_ms)
        except Exception:
            try:
                value = field.evaluate("el => el.value")
                return str(value or "")
            except Exception:
                return ""

    def _required_info(self, container) -> Dict[str, Any]:
        try:
            data = container.evaluate(
                """(el) => {
                  const hasRequiredAttr = !!el.querySelector('[required], [aria-required=\"true\"]');
                  const markerPattern = /required|\\*/i;
                  const hasRequiredMarker = Array.from(el.querySelectorAll('*')).some((node) => {
                    const text = (node.textContent || '').trim();
                    const aria = node.getAttribute ? (node.getAttribute('aria-label') || '') : '';
                    const cls = node.getAttribute ? (node.getAttribute('class') || '') : '';
                    return markerPattern.test(text) || markerPattern.test(aria) || /required/i.test(String(cls));
                  });
                  return {
                    required: hasRequiredAttr || hasRequiredMarker,
                    required_attr: hasRequiredAttr,
                    required_marker: hasRequiredMarker
                  };
                }"""
            )
            if isinstance(data, dict):
                return {
                    "required": bool(data.get("required")),
                    "required_attr": bool(data.get("required_attr")),
                    "required_marker": bool(data.get("required_marker")),
                }
        except Exception:
            pass
        return {"required": None, "required_attr": None, "required_marker": None}

    def _attach_required_info(self, action: Dict[str, Any], container) -> None:
        info = self._required_info(container)
        action["required"] = info.get("required")
        action["required_attr"] = info.get("required_attr")
        action["required_marker"] = info.get("required_marker")

    def _canonical_text_value(self, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    def _assert_input_value(self, field, expected: Any, error_code: str) -> None:
        actual_raw = self._read_input_value(field)
        actual = self._canonical_text_value(actual_raw)
        wanted = self._canonical_text_value(expected)
        if wanted.isdigit() and actual.isdigit():
            if int(wanted) != int(actual):
                raise RuntimeError(f"{error_code}: expected={wanted}, actual={actual_raw}")
            return
        if wanted != actual:
            raise RuntimeError(f"{error_code}: expected={wanted}, actual={actual_raw}")

    def _is_option_selected(self, option) -> bool:
        try:
            selected = option.evaluate(
                """(el) => {
                  const aria = el.getAttribute('aria-checked');
                  if (aria !== null) return aria === 'true';
                  if (el.hasAttribute('checked')) return true;
                  if (el.matches('input[type=checkbox], input[type=radio]')) return !!el.checked;
                  const nestedInput = el.querySelector('input[type=checkbox], input[type=radio]');
                  if (nestedInput) return !!nestedInput.checked;
                  const cls = (el.getAttribute('class') || '').toLowerCase();
                  if (cls.includes('checked') || cls.includes('selected')) return true;
                  return false;
                }"""
            )
            return bool(selected)
        except Exception:
            return False

    def _find_matching_role_options(self, container, value: str, role: str):
        matches: List[Any] = []
        target_norm = _norm_text(value)
        options = container.locator(f"[role='{role}']")
        for idx in range(options.count()):
            option = options.nth(idx)
            try:
                label = option.get_attribute("aria-label") or option.inner_text(timeout=1000)
            except Exception:
                continue
            if target_norm and target_norm in _norm_text(label):
                matches.append(option)
        return matches

    def _assert_option_selected(self, container, value: str, role: str) -> None:
        matches = self._find_matching_role_options(container, value, role)
        if not matches:
            raise RuntimeError(f"option_verify_not_found: {value}")
        if not any(self._is_option_selected(option) for option in matches):
            raise RuntimeError(f"option_not_selected: {value}")

    def _handle_text(self, label: str, value: Any, step_idx: int, action: Dict[str, Any]) -> None:
        container = self._find_container_with_pagination(label, step_idx)
        if container is None:
            raise RuntimeError("container_not_found")
        action["bbox"] = self._bbox(container)
        action["scroll_y"] = self._get_scroll_state().get("scroll_y")
        self._attach_required_info(action, container)

        self._scroll_into_view(container, step_idx)
        field = container.locator(
            "textarea, input[type='text'], input[type='email'], input[type='url'], input[type='number']"
        ).first
        if field.count() == 0:
            raise RuntimeError("input_not_found")

        self._set_action_target(action, field)
        self._hover(field, step_idx)
        self._click(field, step_idx)
        self._type(field, str(value), step_idx, clear_before_typing=True)
        self._assert_input_value(field, str(value), "text_value_mismatch")

    def _select_option(self, container, value: str, role: str, step_idx: int, action: Dict[str, Any]) -> bool:
        target_norm = _norm_text(value)
        options = container.locator(f"[role='{role}']")
        for idx in range(options.count()):
            option = options.nth(idx)
            try:
                label = option.get_attribute("aria-label") or option.inner_text(timeout=1000)
            except Exception:
                continue
            if target_norm and target_norm in _norm_text(label):
                self._scroll_into_view(option, step_idx)
                self._set_action_target(action, option)
                self._hover(option, step_idx)
                self._click(option, step_idx)
                return True

        text_hits = container.get_by_text(value, exact=False)
        for idx in range(text_hits.count()):
            hit = text_hits.nth(idx)
            try:
                label = hit.inner_text(timeout=1000)
            except Exception:
                continue
            if target_norm and target_norm in _norm_text(label):
                self._scroll_into_view(hit, step_idx)
                self._set_action_target(action, hit)
                self._hover(hit, step_idx)
                self._click(hit, step_idx)
                return True
        return False

    def _handle_single_choice(self, label: str, value: Any, step_idx: int, action: Dict[str, Any]) -> None:
        container = self._find_container_with_pagination(label, step_idx)
        if container is None:
            raise RuntimeError("container_not_found")
        action["bbox"] = self._bbox(container)
        action["scroll_y"] = self._get_scroll_state().get("scroll_y")
        self._attach_required_info(action, container)
        self._scroll_into_view(container, step_idx)
        if not self._select_option(container, str(value), "radio", step_idx, action):
            raise RuntimeError("option_not_found")
        self._assert_option_selected(container, str(value), "radio")

    def _handle_multi_choice(self, label: str, value: Any, step_idx: int, action: Dict[str, Any]) -> None:
        container = self._find_container_with_pagination(label, step_idx)
        if container is None:
            raise RuntimeError("container_not_found")
        action["bbox"] = self._bbox(container)
        action["scroll_y"] = self._get_scroll_state().get("scroll_y")
        self._attach_required_info(action, container)
        self._scroll_into_view(container, step_idx)
        values = value if isinstance(value, list) else [v.strip() for v in str(value).split(",") if v.strip()]
        for entry in values:
            if not self._select_option(container, str(entry), "checkbox", step_idx, action):
                raise RuntimeError(f"option_not_found: {entry}")
            self._assert_option_selected(container, str(entry), "checkbox")

    def _handle_date(self, label: str, value: Any, step_idx: int, action: Dict[str, Any]) -> None:
        parsed = self._normalize_date(value)
        container = self._find_container_with_pagination(label, step_idx)
        if container is None:
            raise RuntimeError("container_not_found")
        action["bbox"] = self._bbox(container)
        action["scroll_y"] = self._get_scroll_state().get("scroll_y")
        self._attach_required_info(action, container)
        self._scroll_into_view(container, step_idx)

        native = container.locator("input[type='date']")
        if native.count() > 0:
            field = native.first
            self._set_action_target(action, field)
            self._hover(field, step_idx)
            self._click(field, step_idx)
            expected = parsed.strftime("%Y-%m-%d")
            self._fill_direct(field, expected, step_idx, clear_before_typing=True)
            actual = self._read_input_value(field).strip()
            if actual != expected:
                raise RuntimeError(f"date_value_mismatch: expected={expected}, actual={actual}")
            return

        text_inputs = container.locator("input[type='text']")
        if text_inputs.count() == 1:
            field = text_inputs.first
            self._set_action_target(action, field)
            self._hover(field, step_idx)
            self._click(field, step_idx)
            formatted = self._format_date_for_text_field(parsed, field)
            self._type(field, formatted, step_idx, clear_before_typing=True)
            self._assert_input_value(field, formatted, "date_value_mismatch")
            return

        year_input = self._find_input_by_keywords(container, ["year", "yyyy", "yy"])
        month_input = self._find_input_by_keywords(container, ["month", "mm"])
        day_input = self._find_input_by_keywords(container, ["day", "dd"])

        ordered = self._ordered_inputs(container)
        if not year_input or not month_input or not day_input:
            if len(ordered) < 3:
                raise RuntimeError("date_inputs_not_found")
            year_input, month_input, day_input = ordered[0], ordered[1], ordered[2]

        for field, text in [
            (year_input, str(parsed.year)),
            (month_input, str(parsed.month)),
            (day_input, str(parsed.day)),
        ]:
            self._set_action_target(action, field)
            self._hover(field, step_idx)
            self._click(field, step_idx)
            self._type(field, text, step_idx, clear_before_typing=True)
            self._assert_input_value(field, text, "date_segment_mismatch")

    def _handle_time(self, label: str, value: Any, step_idx: int, action: Dict[str, Any]) -> None:
        parsed = self._normalize_time(value)
        hour_24 = parsed.strftime("%H")
        minute = parsed.strftime("%M")
        hour_12 = parsed.strftime("%I")
        meridiem = parsed.strftime("%p")

        container = self._find_container_with_pagination(label, step_idx)
        if container is None:
            raise RuntimeError("container_not_found")
        action["bbox"] = self._bbox(container)
        action["scroll_y"] = self._get_scroll_state().get("scroll_y")
        self._attach_required_info(action, container)
        self._scroll_into_view(container, step_idx)

        native = container.locator("input[type='time']")
        if native.count() > 0:
            field = native.first
            self._set_action_target(action, field)
            self._hover(field, step_idx)
            self._click(field, step_idx)
            self._type(field, f"{hour_24}:{minute}", step_idx, clear_before_typing=True)
            self._assert_input_value(field, f"{hour_24}:{minute}", "time_value_mismatch")
            return

        hour_input = self._find_input_by_keywords(container, ["hour", "hh", "h"])
        minute_input = self._find_input_by_keywords(container, ["minute", "mm", "m"])

        ordered = self._ordered_inputs(container)
        if not hour_input or not minute_input:
            if len(ordered) < 2:
                raise RuntimeError("time_inputs_not_found")
            hour_input, minute_input = ordered[0], ordered[1]

        am_pm_target = None
        for marker in [meridiem, meridiem.lower()]:
            probe = container.get_by_text(marker, exact=False)
            if probe.count() > 0:
                am_pm_target = probe.first
                break

        hour_text = hour_12 if am_pm_target is not None else hour_24
        for field, text in [(hour_input, hour_text), (minute_input, minute)]:
            self._set_action_target(action, field)
            self._hover(field, step_idx)
            self._click(field, step_idx)
            self._type(field, text, step_idx, clear_before_typing=True)
            self._assert_input_value(field, text, "time_segment_mismatch")

        if am_pm_target is not None:
            self._set_action_target(action, am_pm_target)
            self._hover(am_pm_target, step_idx)
            self._click(am_pm_target, step_idx)

    def _selected_option_labels(self, container, role: str) -> List[str]:
        labels: List[str] = []
        options = container.locator(f"[role='{role}']")
        count = options.count()
        for idx in range(count):
            option = options.nth(idx)
            if not self._is_option_selected(option):
                continue
            try:
                label = option.get_attribute("aria-label") or option.inner_text(timeout=1000) or ""
            except Exception:
                label = ""
            label_text = re.sub(r"\s+", " ", str(label or "")).strip()
            if label_text and label_text not in labels:
                labels.append(label_text)
        return labels

    def _read_date_value(self, container) -> Optional[str]:
        native = container.locator("input[type='date']")
        if native.count() > 0:
            value = self._read_input_value(native.first).strip()
            return value or None

        text_inputs = container.locator("input[type='text']")
        if text_inputs.count() == 1:
            raw = self._read_input_value(text_inputs.first).strip()
            if not raw:
                return None
            try:
                return self._normalize_date(raw).strftime("%Y-%m-%d")
            except Exception:
                return raw

        year_input = self._find_input_by_keywords(container, ["year", "yyyy", "yy"])
        month_input = self._find_input_by_keywords(container, ["month", "mm"])
        day_input = self._find_input_by_keywords(container, ["day", "dd"])
        ordered = self._ordered_inputs(container)
        if (not year_input or not month_input or not day_input) and len(ordered) >= 3:
            year_input, month_input, day_input = ordered[0], ordered[1], ordered[2]
        if not year_input or not month_input or not day_input:
            return None
        year = self._read_input_value(year_input).strip()
        month = self._read_input_value(month_input).strip()
        day = self._read_input_value(day_input).strip()
        if not year or not month or not day:
            return None
        if year.isdigit() and month.isdigit() and day.isdigit():
            return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
        return f"{year}-{month}-{day}"

    def _read_time_value(self, container) -> Optional[str]:
        native = container.locator("input[type='time']")
        if native.count() > 0:
            value = self._read_input_value(native.first).strip()
            return value or None

        hour_input = self._find_input_by_keywords(container, ["hour", "hh", "h"])
        minute_input = self._find_input_by_keywords(container, ["minute", "mm", "m"])
        ordered = self._ordered_inputs(container)
        if (not hour_input or not minute_input) and len(ordered) >= 2:
            hour_input, minute_input = ordered[0], ordered[1]
        if not hour_input or not minute_input:
            return None
        hour = self._read_input_value(hour_input).strip()
        minute = self._read_input_value(minute_input).strip()
        if not hour or not minute:
            return None

        marker_state = None
        for marker_text in ["AM", "PM"]:
            probe = container.get_by_text(marker_text, exact=False)
            if probe.count() == 0:
                continue
            option = probe.first
            try:
                state = option.evaluate(
                    """(el) => {
                      const aria = el.getAttribute('aria-checked') || el.getAttribute('aria-pressed') || el.getAttribute('aria-selected');
                      if (aria !== null) return aria === 'true';
                      const cls = (el.getAttribute('class') || '').toLowerCase();
                      return cls.includes('checked') || cls.includes('selected') || cls.includes('active');
                    }"""
                )
            except Exception:
                state = False
            if state:
                marker_state = marker_text
                break

        if marker_state and hour.isdigit() and minute.isdigit():
            hour_num = int(hour) % 12
            if marker_state == "PM":
                hour_num += 12
            return f"{hour_num:02d}:{int(minute):02d}"
        if hour.isdigit() and minute.isdigit():
            return f"{int(hour):02d}:{int(minute):02d}"
        return f"{hour}:{minute}"

    def verify_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        label = str(entry.get("label") or "")
        widget = str(entry.get("widget_type") or "")
        result: Dict[str, Any] = {
            "label": label,
            "widget_type": widget,
            "verified": False,
            "actual_value": None,
            "detail": None,
        }
        if not label or not widget:
            result["detail"] = "missing_label_or_widget"
            return result

        container = self._find_question_container(label)
        if container is None:
            result["detail"] = "container_not_visible"
            return result

        try:
            if widget in {"short_text", "paragraph_text"}:
                field = container.locator(
                    "textarea, input[type='text'], input[type='email'], input[type='url'], input[type='number']"
                ).first
                if field.count() == 0:
                    raise RuntimeError("input_not_found")
                result["actual_value"] = self._read_input_value(field).strip()
                result["verified"] = True
            elif widget == "single_choice":
                labels = self._selected_option_labels(container, "radio")
                result["actual_value"] = labels[0] if labels else None
                result["verified"] = bool(labels)
                if not labels:
                    result["detail"] = "no_selected_option"
            elif widget == "multi_choice":
                labels = self._selected_option_labels(container, "checkbox")
                result["actual_value"] = labels
                result["verified"] = True
            elif widget == "date":
                result["actual_value"] = self._read_date_value(container)
                result["verified"] = result["actual_value"] is not None
                if result["actual_value"] is None:
                    result["detail"] = "date_value_unavailable"
            elif widget == "time":
                result["actual_value"] = self._read_time_value(container)
                result["verified"] = result["actual_value"] is not None
                if result["actual_value"] is None:
                    result["detail"] = "time_value_unavailable"
            else:
                result["detail"] = "unsupported_widget"
        except Exception as exc:
            result["detail"] = f"{type(exc).__name__}: {exc}"
            result["verified"] = False
        return result

    def fill_step(self, entry: Dict[str, Any], step_idx: int) -> Tuple[Dict[str, Any], Optional[str]]:
        label = entry.get("label", "")
        widget = entry.get("widget_type", "")
        value = entry.get("value")

        if label and widget in {"single_choice", "multi_choice"}:
            value_text = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
            intent = f"Select {value_text} for {label}"
        elif label:
            intent = f"Fill {label}"
        else:
            intent = None

        action: Dict[str, Any] = {
            "step": step_idx,
            "label": label,
            "widget_type": widget,
            "value": value,
            "intent": intent,
            "success": False,
            "t_start_s": self.trace.now(),
            "t_end_s": self.trace.now(),
            "bbox": None,
            "target_bbox": None,
            "pre_screenshot": None,
            "post_screenshot": None,
            "target_role": None,
            "target_name": None,
            "target_selector": None,
            "scroll_y": None,
            "required": None,
            "required_attr": None,
            "required_marker": None,
            "error": None,
        }

        error: Optional[str] = None
        action["pre_screenshot"] = self._screenshot(f"step_{step_idx:04d}_pre.png")
        try:
            if not label or not widget:
                raise RuntimeError("missing_label_or_widget")
            handler = self.handlers.get(widget)
            if handler is None:
                raise RuntimeError("unsupported_widget")
            handler(label, value, step_idx, action)
            action["success"] = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            action["error"] = error
            action["success"] = False
        finally:
            action["post_screenshot"] = self._screenshot(f"step_{step_idx:04d}_post.png")
            action["t_end_s"] = self.trace.now()

        return action, error

    def _find_submit_button_with_pagination(self, max_page_hops: int = 4):
        button = self._find_button_by_name(re.compile(r"submit", re.I))
        if button is not None:
            return button, 0
        pagination_hops = 0
        for _ in range(max_page_hops):
            moved = self._click_next_page(None)
            if not moved:
                break
            pagination_hops += 1
            button = self._find_button_by_name(re.compile(r"submit", re.I))
            if button is not None:
                return button, pagination_hops
        return None, pagination_hops

    def submit(self) -> Tuple[Dict[str, Any], Optional[str]]:
        error: Optional[str] = None
        info: Dict[str, Any] = {
            "success": False,
            "t_start_s": self.trace.now(),
            "t_end_s": self.trace.now(),
            "bbox": None,
            "submit_clicked": False,
            "confirmation_method": None,
            "final_url": None,
            "pagination_hops": 0,
            "submit_label": None,
            "pre_screenshot": self._screenshot("submit_pre.png"),
            "post_screenshot": None,
        }

        try:
            button, pagination_hops = self._find_submit_button_with_pagination()
            info["pagination_hops"] = pagination_hops
            if button is None:
                raise RuntimeError("submit_button_not_found")

            self._scroll_into_view(button, None)
            info["bbox"] = self._bbox(button)
            try:
                info["submit_label"] = (button.inner_text(timeout=1000) or "").strip() or button.get_attribute("aria-label")
            except Exception:
                try:
                    info["submit_label"] = button.get_attribute("aria-label")
                except Exception:
                    info["submit_label"] = None
            self._hover(button, None)
            self._click(button, None)
            info["submit_clicked"] = True

            confirmation_texts = [
                "Response recorded",
                "Response has been recorded",
                "Thanks for submitting",
                "Your response has been recorded",
            ]
            for text_value in confirmation_texts:
                try:
                    self.page.get_by_text(text_value, exact=False).wait_for(state="visible", timeout=8000)
                    info["success"] = True
                    info["confirmation_method"] = "text"
                    info["final_url"] = self.page.url
                    break
                except Exception:
                    continue

            if not info["success"]:
                try:
                    self.page.wait_for_url(re.compile(r"formResponse", re.IGNORECASE), timeout=8000)
                    info["success"] = True
                    info["confirmation_method"] = "url"
                except Exception:
                    pass

            if not info["success"]:
                try:
                    body_text = (self.page.locator("body").inner_text(timeout=2000) or "").lower()
                    indicators = [
                        "response recorded",
                        "response has been recorded",
                        "your response has been recorded",
                        "response submitted",
                        "submit another response",
                        "edit your response",
                        "thanks for submitting",
                        "thank you for submitting",
                        "thank you",
                    ]
                    if any(token in body_text for token in indicators):
                        info["success"] = True
                        info["confirmation_method"] = "heuristic_text"
                except Exception:
                    pass

            if not info["success"]:
                try:
                    submit_still_visible = self._find_button_by_name(re.compile(r"submit", re.I))
                    question_count = self.page.locator("div[role='listitem']").count()
                    if submit_still_visible is None and question_count == 0:
                        info["success"] = True
                        info["confirmation_method"] = "heuristic_post_submit_state"
                except Exception:
                    pass

            info["final_url"] = self.page.url
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            info["post_screenshot"] = self._screenshot("submit_post.png")
            info["t_end_s"] = self.trace.now()

        return info, error
