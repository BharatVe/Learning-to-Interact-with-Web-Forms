import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from engine.trace_logger import TraceLogger


class FormEngine:
    def __init__(
        self,
        page,
        viewport: Dict[str, int],
        observations_dir: Path,
        trace: TraceLogger,
        timeout_ms: int = 15000,
        type_delay_ms: int = 60,
        action_delay_ms: int = 120,
    ) -> None:
        self.page = page
        self.viewport = viewport
        self.observations_dir = observations_dir
        self.trace = trace
        self.timeout_ms = timeout_ms
        self.type_delay_ms = max(0, type_delay_ms)
        self.action_delay_ms = max(0, action_delay_ms)
        self.handlers = {
            "short_text": self._handle_text,
            "paragraph_text": self._handle_text,
            "single_choice": self._handle_single_choice,
            "multi_choice": self._handle_multi_choice,
            "date": self._handle_date,
            "time": self._handle_time,
        }

    def _coords_norm(self, x: float, y: float) -> Tuple[int, int]:
        w = max(int(self.viewport.get("width", 1)), 1)
        h = max(int(self.viewport.get("height", 1)), 1)
        x_norm = int(round((x / w) * 999))
        y_norm = int(round((y / h) * 999))
        return max(0, min(999, x_norm)), max(0, min(999, y_norm))

    def _get_scroll_state(self) -> Dict[str, Optional[float]]:
        try:
            return self.page.evaluate("() => ({scroll_x: window.scrollX, scroll_y: window.scrollY})")
        except Exception:
            return {"scroll_x": None, "scroll_y": None}

    def _event_extra(
        self,
        scroll_state: Optional[Dict[str, Optional[float]]] = None,
        x_px: Optional[float] = None,
        y_px: Optional[float] = None,
    ) -> Dict[str, Any]:
        scroll_state = scroll_state or {}
        return {
            "viewport_w": self.viewport.get("width"),
            "viewport_h": self.viewport.get("height"),
            "scroll_x": scroll_state.get("scroll_x"),
            "scroll_y": scroll_state.get("scroll_y"),
            "x_px": x_px,
            "y_px": y_px,
        }

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
        direction = "down" if delta > 0 else "up"
        self.trace.log_event(
            "scroll_document",
            {"direction": direction},
            step_ref=step_ref,
            extra=self._event_extra(after),
        )

    def _pause_after_action(self) -> None:
        if self.action_delay_ms > 0:
            self.page.wait_for_timeout(self.action_delay_ms)

    def _screenshot(self, name: str) -> str:
        rel_path = f"observations/{name}"
        abs_path = self.observations_dir / name
        self.observations_dir.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(abs_path))
        return rel_path

    def _hover_at(self, x: float, y: float, step_ref: Optional[int]) -> None:
        scroll_state = self._get_scroll_state()
        try:
            self.page.mouse.move(x, y, steps=20)
            x_norm, y_norm = self._coords_norm(x, y)
            self.trace.log_event(
                "hover_at",
                {"x": x_norm, "y": y_norm},
                step_ref=step_ref,
                extra=self._event_extra(scroll_state, x_px=x, y_px=y),
            )
            self._pause_after_action()
        except Exception as exc:
            self.trace.log_event(
                "hover_at",
                {"x": None, "y": None},
                step_ref=step_ref,
                ok=False,
                error=str(exc),
                extra=self._event_extra(scroll_state, x_px=x, y_px=y),
            )
            raise

    def _click_at(self, x: float, y: float, step_ref: Optional[int]) -> None:
        scroll_state = self._get_scroll_state()
        try:
            self.page.mouse.click(x, y)
            x_norm, y_norm = self._coords_norm(x, y)
            self.trace.log_event(
                "click_at",
                {"x": x_norm, "y": y_norm},
                step_ref=step_ref,
                extra=self._event_extra(scroll_state, x_px=x, y_px=y),
            )
            self._pause_after_action()
        except Exception as exc:
            self.trace.log_event(
                "click_at",
                {"x": None, "y": None},
                step_ref=step_ref,
                ok=False,
                error=str(exc),
                extra=self._event_extra(scroll_state, x_px=x, y_px=y),
            )
            raise

    def _type_text_at(self, x: float, y: float, text: str, step_ref: Optional[int]) -> None:
        scroll_state = self._get_scroll_state()
        try:
            self.page.keyboard.type(text, delay=self.type_delay_ms)
            x_norm, y_norm = self._coords_norm(x, y)
            self.trace.log_event(
                "type_text_at",
                {"x": x_norm, "y": y_norm, "text": text, "clear_before_typing": False},
                step_ref=step_ref,
                extra=self._event_extra(scroll_state, x_px=x, y_px=y),
            )
            self._pause_after_action()
        except Exception as exc:
            self.trace.log_event(
                "type_text_at",
                {"x": None, "y": None, "text": text, "clear_before_typing": False},
                step_ref=step_ref,
                ok=False,
                error=str(exc),
                extra=self._event_extra(scroll_state, x_px=x, y_px=y),
            )
            raise

    def _press_keys(self, keys: str, step_ref: Optional[int]) -> None:
        scroll_state = self._get_scroll_state()
        try:
            self.page.keyboard.press(keys)
            self.trace.log_event(
                "key_combination",
                {"keys": keys},
                step_ref=step_ref,
                extra=self._event_extra(scroll_state),
            )
            self._pause_after_action()
        except Exception as exc:
            self.trace.log_event(
                "key_combination",
                {"keys": keys},
                step_ref=step_ref,
                ok=False,
                error=str(exc),
                extra=self._event_extra(scroll_state),
            )
            raise

    def _locate_input(self, label: str, step_ref: Optional[int]) -> Dict[str, Any]:
        before = self._get_scroll_state()
        script = r"""
(label) => {
  const norm = (s) => (s || '').toLowerCase()
    .replace(/[’‘]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  const items = Array.from(document.querySelectorAll("div[role='listitem']"));
  const container = items.find(item => norm(item.innerText).includes(norm(label)));
  if (!container) return { found: false };
  const field = container.querySelector("textarea, input[type='text'], input[type='email'], input[type='url'], input[type='number']");
  if (!field) return { found: false };
  field.scrollIntoView({block: 'center'});
  const rect = field.getBoundingClientRect();
  const containerRect = container.getBoundingClientRect();
  return {
    found: true,
    container_bbox: { x: containerRect.x, y: containerRect.y, width: containerRect.width, height: containerRect.height },
    target_bbox: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
    center: { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 },
    role: field.getAttribute('role') || null,
    name: field.getAttribute('aria-label') || field.getAttribute('placeholder') || null,
    selector: field.id ? ('#' + field.id) : null,
    scroll_x: window.scrollX,
    scroll_y: window.scrollY
  };
}
"""
        data = self.page.evaluate(script, label)
        after = self._get_scroll_state()
        self._log_scroll_if_needed(before, after, step_ref)
        if not data or not data.get("found"):
            raise RuntimeError("input_not_found")
        return data

    def _locate_option(self, label: str, value: str, role: str, step_ref: Optional[int]) -> Dict[str, Any]:
        before = self._get_scroll_state()
        script = r"""
(label, value, role) => {
  const norm = (s) => (s || '').toLowerCase()
    .replace(/[’‘]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  const items = Array.from(document.querySelectorAll("div[role='listitem']"));
  const container = items.find(item => norm(item.innerText).includes(norm(label)));
  if (!container) return { found: false };
  const options = Array.from(container.querySelectorAll("[role='" + role + "']"));
  const target = options.find(opt => norm(opt.getAttribute('aria-label') || opt.innerText).includes(norm(value)));
  if (!target) return { found: false };
  target.scrollIntoView({block: 'center'});
  const rect = target.getBoundingClientRect();
  const containerRect = container.getBoundingClientRect();
  return {
    found: true,
    container_bbox: { x: containerRect.x, y: containerRect.y, width: containerRect.width, height: containerRect.height },
    target_bbox: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
    center: { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 },
    role: target.getAttribute('role') || null,
    name: target.getAttribute('aria-label') || target.innerText || null,
    selector: target.id ? ('#' + target.id) : null,
    scroll_x: window.scrollX,
    scroll_y: window.scrollY
  };
}
"""
        data = self.page.evaluate(script, label, value, role)
        after = self._get_scroll_state()
        self._log_scroll_if_needed(before, after, step_ref)
        if not data or not data.get("found"):
            raise RuntimeError("option_not_found")
        return data

    def _locate_date_inputs(self, label: str, step_ref: Optional[int]) -> Dict[str, Any]:
        before = self._get_scroll_state()
        script = r"""
(label) => {
  const norm = (s) => (s || '').toLowerCase()
    .replace(/[’‘]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  const items = Array.from(document.querySelectorAll("div[role='listitem']"));
  const container = items.find(item => norm(item.innerText).includes(norm(label)));
  if (!container) return { found: false };
  container.scrollIntoView({block: 'center'});
  const containerRect = container.getBoundingClientRect();
  const dateInput = container.querySelector("input[type='date']");
  if (dateInput) {
    const rect = dateInput.getBoundingClientRect();
    return {
      found: true,
      mode: 'date',
      container_bbox: { x: containerRect.x, y: containerRect.y, width: containerRect.width, height: containerRect.height },
      targets: [{ x: rect.x, y: rect.y, width: rect.width, height: rect.height }],
      selector: dateInput.id ? ('#' + dateInput.id) : null,
      scroll_x: window.scrollX,
      scroll_y: window.scrollY
    };
  }
  const inputs = Array.from(container.querySelectorAll("input[type='text'], input[type='number']"));
  if (!inputs.length) return { found: false };
  const targets = inputs.map(el => {
    const r = el.getBoundingClientRect();
    return { x: r.x, y: r.y, width: r.width, height: r.height };
  });
  return {
    found: true,
    mode: 'segments',
    container_bbox: { x: containerRect.x, y: containerRect.y, width: containerRect.width, height: containerRect.height },
    targets,
    scroll_x: window.scrollX,
    scroll_y: window.scrollY
  };
}
"""
        data = self.page.evaluate(script, label)
        after = self._get_scroll_state()
        self._log_scroll_if_needed(before, after, step_ref)
        if not data or not data.get("found"):
            raise RuntimeError("date_inputs_not_found")
        return data

    def _locate_time_inputs(self, label: str, step_ref: Optional[int]) -> Dict[str, Any]:
        return self._locate_date_inputs(label, step_ref)

    def _locate_submit_button(self) -> Dict[str, Any]:
        before = self._get_scroll_state()
        script = """
() => {
  const buttons = Array.from(document.querySelectorAll("div[role='button'], button"));
  const btn = buttons.find(el => /submit/i.test(el.innerText || el.getAttribute('aria-label') || ''));
  if (!btn) return { found: false };
  btn.scrollIntoView({block: 'center'});
  const rect = btn.getBoundingClientRect();
  return {
    found: true,
    bbox: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
    center: { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 },
    scroll_x: window.scrollX,
    scroll_y: window.scrollY
  };
}
"""
        data = self.page.evaluate(script)
        after = self._get_scroll_state()
        self._log_scroll_if_needed(before, after, None)
        return data if data else {"found": False}

    def _handle_text(self, label: str, value: Any, step_idx: int, action: Dict[str, Any]) -> None:
        data = self._locate_input(label, step_idx)
        action["bbox"] = data.get("container_bbox")
        action["target_bbox"] = data.get("target_bbox")
        action["scroll_y"] = data.get("scroll_y")
        action["target_role"] = data.get("role")
        action["target_name"] = data.get("name")
        action["target_selector"] = data.get("selector")
        center = data["center"]
        self._hover_at(center["x"], center["y"], step_idx)
        self._click_at(center["x"], center["y"], step_idx)
        self._type_text_at(center["x"], center["y"], str(value), step_idx)

    def _handle_single_choice(self, label: str, value: Any, step_idx: int, action: Dict[str, Any]) -> None:
        data = self._locate_option(label, str(value), "radio", step_idx)
        action["bbox"] = data.get("container_bbox")
        action["target_bbox"] = data.get("target_bbox")
        action["scroll_y"] = data.get("scroll_y")
        action["target_role"] = data.get("role")
        action["target_name"] = data.get("name")
        action["target_selector"] = data.get("selector")
        center = data["center"]
        self._hover_at(center["x"], center["y"], step_idx)
        self._click_at(center["x"], center["y"], step_idx)

    def _handle_multi_choice(self, label: str, value: Any, step_idx: int, action: Dict[str, Any]) -> None:
        values = value if isinstance(value, list) else [v.strip() for v in str(value).split(",") if v.strip()]
        for entry_value in values:
            data = self._locate_option(label, str(entry_value), "checkbox", step_idx)
            action["bbox"] = data.get("container_bbox")
            action["target_bbox"] = data.get("target_bbox")
            action["scroll_y"] = data.get("scroll_y")
            action["target_role"] = data.get("role")
            action["target_name"] = data.get("name")
            action["target_selector"] = data.get("selector")
            center = data["center"]
            self._hover_at(center["x"], center["y"], step_idx)
            self._click_at(center["x"], center["y"], step_idx)

    def _handle_date(self, label: str, value: Any, step_idx: int, action: Dict[str, Any]) -> None:
        data = self._locate_date_inputs(label, step_idx)
        action["bbox"] = data.get("container_bbox")
        action["scroll_y"] = data.get("scroll_y")
        targets = data.get("targets", [])
        if data.get("mode") == "date" and targets:
            rect = targets[0]
            center = {"x": rect["x"] + rect["width"] / 2, "y": rect["y"] + rect["height"] / 2}
            action["target_bbox"] = rect
            self._hover_at(center["x"], center["y"], step_idx)
            self._click_at(center["x"], center["y"], step_idx)
            self._type_text_at(center["x"], center["y"], str(value), step_idx)
            return
        parts = str(value).split("-")
        if len(targets) >= 3 and len(parts) == 3:
            for idx, part in enumerate(parts):
                rect = targets[idx]
                center = {"x": rect["x"] + rect["width"] / 2, "y": rect["y"] + rect["height"] / 2}
                action["target_bbox"] = rect
                self._hover_at(center["x"], center["y"], step_idx)
                self._click_at(center["x"], center["y"], step_idx)
                self._type_text_at(center["x"], center["y"], part, step_idx)
                if idx < 2:
                    self._press_keys("Tab", step_idx)

    def _handle_time(self, label: str, value: Any, step_idx: int, action: Dict[str, Any]) -> None:
        data = self._locate_time_inputs(label, step_idx)
        action["bbox"] = data.get("container_bbox")
        action["scroll_y"] = data.get("scroll_y")
        targets = data.get("targets", [])
        parts = str(value).split(":")
        for idx, part in enumerate(parts):
            if idx >= len(targets):
                break
            rect = targets[idx]
            center = {"x": rect["x"] + rect["width"] / 2, "y": rect["y"] + rect["height"] / 2}
            action["target_bbox"] = rect
            self._hover_at(center["x"], center["y"], step_idx)
            self._click_at(center["x"], center["y"], step_idx)
            self._type_text_at(center["x"], center["y"], part, step_idx)
            if idx < len(parts) - 1:
                self._press_keys("Tab", step_idx)

    def fill_step(self, entry: Dict[str, Any], step_idx: int) -> Tuple[Dict[str, Any], Optional[str]]:
        label = entry.get("label", "")
        widget = entry.get("widget_type", "")
        value = entry.get("value")
        intent = None
        if label:
            if widget in {"single_choice", "multi_choice"}:
                if isinstance(value, list):
                    joined = ", ".join(str(v) for v in value)
                else:
                    joined = str(value)
                intent = f"Select {joined} for {label}"
            else:
                intent = f"Fill {label}"

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
            "error": None,
        }

        error: Optional[str] = None
        action["pre_screenshot"] = self._screenshot(f"step_{step_idx:04d}_pre.png")
        try:
            if not label or not widget:
                raise RuntimeError("missing_label_or_widget")
            handler = self.handlers.get(widget)
            if not handler:
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

    def submit(self) -> Tuple[Dict[str, Any], Optional[str]]:
        error: Optional[str] = None
        pre_path = self._screenshot("submit_pre.png")
        info: Dict[str, Any] = {
            "success": False,
            "t_start_s": self.trace.now(),
            "t_end_s": self.trace.now(),
            "bbox": None,
            "submit_clicked": False,
            "confirmation_method": None,
            "final_url": None,
            "pre_screenshot": pre_path,
            "post_screenshot": None,
        }
        try:
            locate = self._locate_submit_button()
            if locate.get("found"):
                info["bbox"] = locate.get("bbox")
                center = locate.get("center")
                if center:
                    self._hover_at(center["x"], center["y"], None)
                    self._click_at(center["x"], center["y"], None)
                    info["submit_clicked"] = True
            if info["submit_clicked"]:
                try:
                    self.page.wait_for_url("**/formResponse**", timeout=self.timeout_ms)
                    info["success"] = True
                    info["confirmation_method"] = "url"
                except Exception:
                    try:
                        confirmed = self.page.evaluate(
                            "() => /response has been recorded|thanks|thank you/i.test(document.body.innerText || '')"
                        )
                        if confirmed:
                            info["success"] = True
                            info["confirmation_method"] = "text"
                    except Exception:
                        pass
            info["final_url"] = self.page.url
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            info["post_screenshot"] = self._screenshot("submit_post.png")
            info["t_end_s"] = self.trace.now()

        return info, error
