import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from engine.mcp_trace_client import MCPClient
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

JSON_MARKER = "THESIS_JSON:"


def _extract_marked_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = payload.get("text")
    if not isinstance(text, str):
        return {}
    match = re.search(rf"{JSON_MARKER}(\{{.*\}})", text, flags=re.S)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(1))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class MCPBrowserEngine:
    def __init__(
        self,
        mcp_client: MCPClient,
        trace: TraceLogger,
        observations_dir: Path,
        timeout_ms: int,
        type_delay_ms: int,
        action_delay_ms: int,
        take_screenshots: bool,
    ) -> None:
        self.mcp = mcp_client
        self.trace = trace
        self.observations_dir = observations_dir
        self.timeout_ms = max(1000, timeout_ms)
        self.type_delay_ms = max(0, type_delay_ms)
        self.action_delay_ms = max(0, action_delay_ms)
        self.take_screenshots = take_screenshots

    def _call_tool(
        self,
        name: str,
        args: Dict[str, Any],
        step_ref: Optional[int],
        trace_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        logged_args = trace_args if trace_args is not None else args
        try:
            payload = self.mcp.call_tool(name, args)
            self.trace.log_event(
                name,
                logged_args,
                step_ref=step_ref,
                extra={"backend": "mcp_server"},
            )
            return payload
        except Exception as exc:
            self.trace.log_event(
                name,
                logged_args,
                step_ref=step_ref,
                ok=False,
                error=str(exc),
                extra={"backend": "mcp_server"},
            )
            raise

    def _run_code(self, code: str, purpose: str, step_ref: Optional[int]) -> Dict[str, Any]:
        payload = self._call_tool(
            "browser_run_code",
            {"code": code},
            step_ref=step_ref,
            trace_args={"purpose": purpose},
        )
        parsed = _extract_marked_json(payload)
        return parsed if parsed else {}

    def _wait_seconds(self, seconds: float, step_ref: Optional[int]) -> None:
        if seconds <= 0:
            return
        self._call_tool(
            "browser_wait_for",
            {"time": float(seconds)},
            step_ref=step_ref,
        )

    def wait_seconds(self, seconds: float, step_ref: Optional[int]) -> None:
        self._wait_seconds(seconds, step_ref)

    def _screenshot(self, filename: str, step_ref: Optional[int]) -> Optional[str]:
        if not self.take_screenshots:
            return None
        self.observations_dir.mkdir(parents=True, exist_ok=True)
        rel_path = f"observations/{filename}"
        self._call_tool(
            "browser_take_screenshot",
            {"type": "png", "filename": rel_path},
            step_ref=step_ref,
        )
        return rel_path

    def navigate(self, form_url: str) -> Dict[str, Any]:
        self._call_tool("browser_navigate", {"url": form_url}, step_ref=None)
        self._wait_seconds(max(self.action_delay_ms, 300) / 1000.0, step_ref=None)
        env_code = f"""
async (page) => {{
  const env = {{
    devicePixelRatio: await page.evaluate(() => window.devicePixelRatio || null),
    userAgent: await page.evaluate(() => navigator.userAgent || null),
    locale: await page.evaluate(() => navigator.language || null),
    timezone: await page.evaluate(() => Intl.DateTimeFormat().resolvedOptions().timeZone || null),
    url: page.url()
  }};
  return "{JSON_MARKER}" + JSON.stringify(env);
}}
"""
        return self._run_code(env_code, purpose="collect_env", step_ref=None)

    def enable_mouse_overlay(self) -> None:
        code = f"""
async (page) => {{
  await page.evaluate({json.dumps(CURSOR_OVERLAY_SCRIPT)});
  return "{JSON_MARKER}" + JSON.stringify({{"ok": true}});
}}
"""
        self._run_code(code, purpose="enable_mouse_overlay", step_ref=None)

    def _build_fill_step_code(self, entry: Dict[str, Any]) -> str:
        payload = json.dumps(
            {
                "label": entry.get("label", ""),
                "widgetType": entry.get("widget_type", ""),
                "value": entry.get("value"),
                "timeoutMs": self.timeout_ms,
                "typeDelayMs": self.type_delay_ms,
                "actionDelayMs": self.action_delay_ms,
            },
            ensure_ascii=True,
        )
        script = r"""
async (page) => {
  const step = __STEP_JSON__;

  const norm = (value) => String(value || "")
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[’‘]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  const canonical = (value) => String(value ?? "").replace(/\s+/g, " ").trim();
  const longTypeTimeout = (text) => Math.max(step.timeoutMs, 3000 + String(text).length * Math.max(step.typeDelayMs, 8));

  const bbox = async (locator) => {
    const box = await locator.boundingBox().catch(() => null);
    if (!box) return null;
    return { x: box.x, y: box.y, width: box.width, height: box.height };
  };

  const getTargetMeta = async (locator) => {
    const role = await locator.getAttribute("role").catch(() => null);
    const ariaLabel = await locator.getAttribute("aria-label").catch(() => null);
    const placeholder = await locator.getAttribute("placeholder").catch(() => null);
    const id = await locator.getAttribute("id").catch(() => null);
    return {
      role: role || null,
      name: ariaLabel || placeholder || null,
      selector: id ? `#${id}` : null
    };
  };

  const requiredInfo = async (container) => {
    return await container.evaluate((el) => {
      const hasRequiredAttr = !!el.querySelector("[required], [aria-required='true']");
      const markerPattern = /required|\*/i;
      const hasRequiredMarker = Array.from(el.querySelectorAll("*")).some((node) => {
        const text = (node.textContent || "").trim();
        const aria = node.getAttribute ? (node.getAttribute("aria-label") || "") : "";
        const cls = node.getAttribute ? (node.getAttribute("class") || "") : "";
        return markerPattern.test(text) || markerPattern.test(aria) || /required/i.test(String(cls));
      });
      return {
        required: hasRequiredAttr || hasRequiredMarker,
        required_attr: hasRequiredAttr,
        required_marker: hasRequiredMarker
      };
    }).catch(() => ({ required: null, required_attr: null, required_marker: null }));
  };

  const findQuestionContainer = async (label) => {
    const target = norm(label);
    const items = page.locator("div[role='listitem']");
    const count = await items.count();
    for (let i = 0; i < count; i++) {
      const item = items.nth(i);
      const text = await item.innerText({ timeout: 1200 }).catch(() => "");
      if (target && norm(text).includes(target)) return item;
    }
    return null;
  };

  const findVisibleButton = async (pattern) => {
    const candidates = page.locator("button, div[role='button']");
    const count = await candidates.count();
    for (let i = 0; i < count; i++) {
      const btn = candidates.nth(i);
      const visible = await btn.isVisible().catch(() => false);
      if (!visible) continue;
      const text = await btn.innerText({ timeout: 800 }).catch(() => "");
      if (pattern.test(text || "")) return btn;
      const aria = await btn.getAttribute("aria-label").catch(() => "");
      if (pattern.test(aria || "")) return btn;
    }
    return null;
  };

  const findContainerWithPagination = async (label, maxHops = 4) => {
    let container = await findQuestionContainer(label);
    if (container) return container;
    for (let hop = 0; hop < maxHops; hop++) {
      const nextButton = await findVisibleButton(/^(next|continue|weiter)$/i) || await findVisibleButton(/next|continue|weiter/i);
      if (!nextButton) break;
      await nextButton.scrollIntoViewIfNeeded().catch(() => {});
      await nextButton.click({ timeout: step.timeoutMs });
      await page.waitForTimeout(Math.max(step.actionDelayMs, 250));
      container = await findQuestionContainer(label);
      if (container) return container;
    }
    return null;
  };

  const findInputByKeywords = async (container, keywords) => {
    const inputs = container.locator("input");
    const count = await inputs.count();
    const lowered = keywords.map((k) => String(k).toLowerCase());
    for (let i = 0; i < count; i++) {
      const field = inputs.nth(i);
      const aria = await field.getAttribute("aria-label").catch(() => "");
      const ph = await field.getAttribute("placeholder").catch(() => "");
      const raw = `${aria || ""} ${ph || ""}`.toLowerCase();
      if (raw && lowered.some((k) => raw.includes(k))) return field;
    }
    return null;
  };

  const orderedInputs = async (container) => {
    const locator = container.locator("input[type='text'], input[type='number']");
    const count = await locator.count();
    const withBoxes = [];
    for (let i = 0; i < count; i++) {
      const field = locator.nth(i);
      const box = await field.boundingBox().catch(() => null);
      if (!box) continue;
      withBoxes.push({ field, y: box.y, x: box.x });
    }
    withBoxes.sort((a, b) => (a.y - b.y) || (a.x - b.x));
    return withBoxes.map((item) => item.field);
  };

  const typeInto = async (field, text) => {
    await field.scrollIntoViewIfNeeded().catch(() => {});
    await field.click({ timeout: step.timeoutMs });
    await field.fill("", { timeout: step.timeoutMs });
    await field.type(String(text), {
      delay: step.typeDelayMs,
      timeout: longTypeTimeout(text)
    });
  };

  const readInputValue = async (field) => {
    return await field.inputValue({ timeout: step.timeoutMs })
      .catch(async () => await field.evaluate((el) => String(el.value || "")).catch(() => ""));
  };

  const verifyInput = async (field, expected, code) => {
    const actual = canonical(await readInputValue(field));
    const wanted = canonical(expected);
    if (/^\d+$/.test(wanted) && /^\d+$/.test(actual)) {
      if (parseInt(wanted, 10) !== parseInt(actual, 10)) {
        throw new Error(`${code}: expected=${wanted}, actual=${actual}`);
      }
      return;
    }
    if (wanted !== actual) {
      throw new Error(`${code}: expected=${wanted}, actual=${actual}`);
    }
  };

  const isSelected = async (option) => {
    return await option.evaluate((el) => {
      const aria = el.getAttribute("aria-checked");
      if (aria !== null) return aria === "true";
      if (el.hasAttribute("checked")) return true;
      if (el.matches("input[type=checkbox], input[type=radio]")) return !!el.checked;
      const nested = el.querySelector("input[type=checkbox], input[type=radio]");
      if (nested) return !!nested.checked;
      const cls = (el.getAttribute("class") || "").toLowerCase();
      return cls.includes("checked") || cls.includes("selected");
    }).catch(() => false);
  };

  const findRoleOptions = async (container, value, role) => {
    const out = [];
    const target = norm(value);
    const options = container.locator(`[role='${role}']`);
    const count = await options.count();
    for (let i = 0; i < count; i++) {
      const option = options.nth(i);
      const aria = await option.getAttribute("aria-label").catch(() => "");
      const text = aria || await option.innerText({ timeout: 800 }).catch(() => "");
      if (target && norm(text).includes(target)) out.push(option);
    }
    return out;
  };

  const selectOption = async (container, value, role) => {
    const options = await findRoleOptions(container, value, role);
    for (const option of options) {
      await option.scrollIntoViewIfNeeded().catch(() => {});
      await option.click({ timeout: step.timeoutMs });
      return option;
    }
    const hits = container.getByText(String(value), { exact: false });
    const count = await hits.count();
    for (let i = 0; i < count; i++) {
      const hit = hits.nth(i);
      const text = await hit.innerText({ timeout: 800 }).catch(() => "");
      if (norm(text).includes(norm(value))) {
        await hit.scrollIntoViewIfNeeded().catch(() => {});
        await hit.click({ timeout: step.timeoutMs });
        return hit;
      }
    }
    return null;
  };

  const assertRoleSelected = async (container, value, role) => {
    const options = await findRoleOptions(container, value, role);
    if (!options.length) throw new Error(`option_verify_not_found: ${value}`);
    for (const option of options) {
      if (await isSelected(option)) return;
    }
    throw new Error(`option_not_selected: ${value}`);
  };

  const parseDateParts = (raw) => {
    const text = String(raw || "").trim();
    let m = text.match(/^(\d{4})[-\/](\d{1,2})[-\/](\d{1,2})$/);
    if (m) return { y: Number(m[1]), mo: Number(m[2]), d: Number(m[3]) };
    m = text.match(/^(\d{1,2})[-\/](\d{1,2})[-\/](\d{4})$/);
    if (m) return { y: Number(m[3]), mo: Number(m[1]), d: Number(m[2]) };
    throw new Error(`invalid_date_value: ${raw}`);
  };

  const parseTimeParts = (raw) => {
    const text = String(raw || "").trim().toLowerCase();
    let m = text.match(/^(\d{1,2}):(\d{2})$/);
    if (m) return { h24: Number(m[1]), min: Number(m[2]) };
    m = text.match(/^(\d{1,2}):(\d{2})\s*([ap]m)$/);
    if (m) {
      let hour = Number(m[1]) % 12;
      if (m[3] === "pm") hour += 12;
      return { h24: hour, min: Number(m[2]) };
    }
    throw new Error(`invalid_time_value: ${raw}`);
  };

  const container = await findContainerWithPagination(step.label, 4);
  if (!container) throw new Error("container_not_found");

  const result = {
    bbox: await bbox(container),
    target_bbox: null,
    target_role: null,
    target_name: null,
    target_selector: null,
    required: null,
    required_attr: null,
    required_marker: null,
  };
  Object.assign(result, await requiredInfo(container));

  const widget = String(step.widgetType || "");
  if (widget === "short_text" || widget === "paragraph_text") {
    const field = container.locator("textarea, input[type='text'], input[type='email'], input[type='url'], input[type='number']").first();
    if (await field.count() === 0) throw new Error("input_not_found");
    await typeInto(field, String(step.value ?? ""));
    await verifyInput(field, String(step.value ?? ""), "text_value_mismatch");
    result.target_bbox = await bbox(field);
    Object.assign(result, await getTargetMeta(field));
  } else if (widget === "single_choice") {
    const option = await selectOption(container, String(step.value ?? ""), "radio");
    if (!option) throw new Error("option_not_found");
    await assertRoleSelected(container, String(step.value ?? ""), "radio");
    result.target_bbox = await bbox(option);
    Object.assign(result, await getTargetMeta(option));
  } else if (widget === "multi_choice") {
    const values = Array.isArray(step.value) ? step.value : String(step.value || "").split(",").map(v => v.trim()).filter(Boolean);
    for (const value of values) {
      const option = await selectOption(container, String(value), "checkbox");
      if (!option) throw new Error(`option_not_found: ${value}`);
      await assertRoleSelected(container, String(value), "checkbox");
      result.target_bbox = await bbox(option);
      Object.assign(result, await getTargetMeta(option));
    }
  } else if (widget === "date") {
    const date = parseDateParts(step.value);
    const native = container.locator("input[type='date']");
    if (await native.count() > 0) {
      const field = native.first();
      const expected = `${date.y.toString().padStart(4, "0")}-${date.mo.toString().padStart(2, "0")}-${date.d.toString().padStart(2, "0")}`;
      await field.click({ timeout: step.timeoutMs });
      await field.fill(expected, { timeout: step.timeoutMs });
      await verifyInput(field, expected, "date_value_mismatch");
      result.target_bbox = await bbox(field);
      Object.assign(result, await getTargetMeta(field));
    } else {
      const textInputs = container.locator("input[type='text']");
      if (await textInputs.count() === 1) {
        const field = textInputs.first();
        const formatted = `${date.mo.toString().padStart(2, "0")}/${date.d.toString().padStart(2, "0")}/${date.y.toString().padStart(4, "0")}`;
        await typeInto(field, formatted);
        await verifyInput(field, formatted, "date_value_mismatch");
        result.target_bbox = await bbox(field);
        Object.assign(result, await getTargetMeta(field));
      } else {
        let yearInput = await findInputByKeywords(container, ["year", "yyyy", "yy"]);
        let monthInput = await findInputByKeywords(container, ["month", "mm"]);
        let dayInput = await findInputByKeywords(container, ["day", "dd"]);
        if (!yearInput || !monthInput || !dayInput) {
          const ordered = await orderedInputs(container);
          if (ordered.length < 3) throw new Error("date_inputs_not_found");
          yearInput = ordered[0];
          monthInput = ordered[1];
          dayInput = ordered[2];
        }
        await typeInto(yearInput, String(date.y));
        await verifyInput(yearInput, String(date.y), "date_segment_mismatch");
        await typeInto(monthInput, String(date.mo));
        await verifyInput(monthInput, String(date.mo), "date_segment_mismatch");
        await typeInto(dayInput, String(date.d));
        await verifyInput(dayInput, String(date.d), "date_segment_mismatch");
        result.target_bbox = await bbox(dayInput);
        Object.assign(result, await getTargetMeta(dayInput));
      }
    }
  } else if (widget === "time") {
    const parsed = parseTimeParts(step.value);
    const h24 = String(parsed.h24).padStart(2, "0");
    const minute = String(parsed.min).padStart(2, "0");
    const h12 = String((((parsed.h24 + 11) % 12) + 1)).padStart(2, "0");
    const meridiem = parsed.h24 >= 12 ? "PM" : "AM";

    const native = container.locator("input[type='time']");
    if (await native.count() > 0) {
      const field = native.first();
      const expected = `${h24}:${minute}`;
      await typeInto(field, expected);
      await verifyInput(field, expected, "time_value_mismatch");
      result.target_bbox = await bbox(field);
      Object.assign(result, await getTargetMeta(field));
    } else {
      let hourInput = await findInputByKeywords(container, ["hour", "hh", "h"]);
      let minuteInput = await findInputByKeywords(container, ["minute", "mm", "m"]);
      if (!hourInput || !minuteInput) {
        const ordered = await orderedInputs(container);
        if (ordered.length < 2) throw new Error("time_inputs_not_found");
        hourInput = ordered[0];
        minuteInput = ordered[1];
      }
      const marker = container.getByText(meridiem, { exact: false });
      const hasMeridiem = (await marker.count()) > 0;
      const hourText = hasMeridiem ? h12 : h24;
      await typeInto(hourInput, hourText);
      await verifyInput(hourInput, hourText, "time_segment_mismatch");
      await typeInto(minuteInput, minute);
      await verifyInput(minuteInput, minute, "time_segment_mismatch");
      if (hasMeridiem) {
        await marker.first().click({ timeout: step.timeoutMs });
      }
      result.target_bbox = await bbox(minuteInput);
      Object.assign(result, await getTargetMeta(minuteInput));
    }
  } else {
    throw new Error("unsupported_widget");
  }

  return "__MARKER__" + JSON.stringify(result);
}
"""
        return script.replace("__STEP_JSON__", payload).replace("__MARKER__", JSON_MARKER)

    def _build_submit_code(self) -> str:
        payload = json.dumps(
            {
                "timeoutMs": self.timeout_ms,
                "actionDelayMs": self.action_delay_ms,
            },
            ensure_ascii=True,
        )
        script = r"""
async (page) => {
  const cfg = __CFG_JSON__;
  const bbox = async (locator) => {
    const box = await locator.boundingBox().catch(() => null);
    if (!box) return null;
    return { x: box.x, y: box.y, width: box.width, height: box.height };
  };

  const findVisibleButton = async (pattern) => {
    const candidates = page.locator("button, div[role='button']");
    const count = await candidates.count();
    for (let i = 0; i < count; i++) {
      const btn = candidates.nth(i);
      const visible = await btn.isVisible().catch(() => false);
      if (!visible) continue;
      const text = await btn.innerText({ timeout: 800 }).catch(() => "");
      const aria = await btn.getAttribute("aria-label").catch(() => "");
      if (pattern.test(text || "") || pattern.test(aria || "")) return btn;
    }
    return null;
  };

  const findSubmitWithPagination = async (maxHops = 4) => {
    let submit = await findVisibleButton(/submit/i);
    if (submit) return submit;
    for (let hop = 0; hop < maxHops; hop++) {
      const nextButton = await findVisibleButton(/^(next|continue|weiter)$/i) || await findVisibleButton(/next|continue|weiter/i);
      if (!nextButton) break;
      await nextButton.click({ timeout: cfg.timeoutMs });
      await page.waitForTimeout(Math.max(cfg.actionDelayMs, 250));
      submit = await findVisibleButton(/submit/i);
      if (submit) return submit;
    }
    return null;
  };

  const info = {
    success: false,
    submit_clicked: false,
    confirmation_method: null,
    final_url: page.url(),
    bbox: null
  };

  const button = await findSubmitWithPagination(4);
  if (!button) throw new Error("submit_button_not_found");
  await button.scrollIntoViewIfNeeded().catch(() => {});
  info.bbox = await bbox(button);
  await button.click({ timeout: cfg.timeoutMs });
  info.submit_clicked = true;

  const confirmations = [
    "Response recorded",
    "Response has been recorded",
    "Thanks for submitting",
    "Your response has been recorded"
  ];
  for (const text of confirmations) {
    try {
      await page.getByText(text, { exact: false }).first().waitFor({ state: "visible", timeout: 8000 });
      info.success = true;
      info.confirmation_method = "text";
      info.final_url = page.url();
      break;
    } catch (e) {
    }
  }

  if (!info.success) {
    try {
      await page.waitForURL(/formResponse/i, { timeout: 8000 });
      info.success = true;
      info.confirmation_method = "url";
      info.final_url = page.url();
    } catch (e) {
    }
  }

  info.final_url = page.url();
  return "__MARKER__" + JSON.stringify(info);
}
"""
        return script.replace("__CFG_JSON__", payload).replace("__MARKER__", JSON_MARKER)

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
        action["pre_screenshot"] = self._screenshot(f"step_{step_idx:04d}_pre.png", step_idx)
        try:
            if not label or not widget:
                raise RuntimeError("missing_label_or_widget")
            code = self._build_fill_step_code(entry)
            result = self._run_code(code, purpose="fill_step", step_ref=step_idx)
            action["success"] = True
            for key in [
                "bbox",
                "target_bbox",
                "target_role",
                "target_name",
                "target_selector",
                "required",
                "required_attr",
                "required_marker",
            ]:
                if key in result:
                    action[key] = result.get(key)
            self._wait_seconds(self.action_delay_ms / 1000.0, step_idx)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            action["error"] = error
            action["success"] = False
        finally:
            action["post_screenshot"] = self._screenshot(f"step_{step_idx:04d}_post.png", step_idx)
            action["t_end_s"] = self.trace.now()
        return action, error

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
            "pre_screenshot": self._screenshot("submit_pre.png", None),
            "post_screenshot": None,
        }

        try:
            result = self._run_code(self._build_submit_code(), purpose="submit", step_ref=None)
            info["success"] = bool(result.get("success"))
            info["bbox"] = result.get("bbox")
            info["submit_clicked"] = bool(result.get("submit_clicked"))
            info["confirmation_method"] = result.get("confirmation_method")
            info["final_url"] = result.get("final_url")
            self._wait_seconds(self.action_delay_ms / 1000.0, None)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            info["post_screenshot"] = self._screenshot("submit_post.png", None)
            info["t_end_s"] = self.trace.now()
        return info, error

    def close(self) -> None:
        try:
            self._call_tool("browser_close", {}, step_ref=None)
        except Exception:
            pass
