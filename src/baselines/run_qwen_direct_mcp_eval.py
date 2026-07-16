import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines import run_baseline_eval as rbe  # noqa: E402
from baselines.model_registry import get_model_by_id  # noqa: E402
from engine.browser_language import force_english_google_forms_url  # noqa: E402
from engine.mcp_browser_engine import MCPBrowserEngine  # noqa: E402
from engine.mcp_trace_client import MCPClient  # noqa: E402
from engine.runner import _default_mcp_server_command, iter_run_specs, load_form_spec, resolve_answers_path  # noqa: E402
from engine.trace_logger import TraceLogger  # noqa: E402

DEFAULT_CONFIG = "configs/baselines/track_baseline_models.json"
DEFAULT_ANSWERS_ROOT = "data/answers"
DEFAULT_DATASET_ROOT = "data/model_baselines"
DEFAULT_EXPERIMENT_ID = "baseline_qwen_direct_mcp_v1"
DEFAULT_TIMEOUT_S = 900
DEFAULT_MAX_STEPS = 24
DEFAULT_API_TIMEOUT_S = 120
DEFAULT_TEXT_MAX_NEW_TOKENS = 1024
DEFAULT_VLM_MAX_NEW_TOKENS = 1024
DEFAULT_BROWSER_MCP_TIMEOUT_MS = 180000
SCHEMA_VERSION = "baseline_eval.v5"
SUMMARY_SCHEMA_VERSION = "baseline_summary.v5"
RAW_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(?P<payload>.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
DIRECT_MCP_MODEL_TOOLS = {
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_fill_form",
    "browser_select_option",
    "browser_check",
    "browser_uncheck",
    "browser_wait_for",
    "browser_press_key",
}
DIRECT_MCP_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "browser_snapshot": {"type": "object", "properties": {}, "additionalProperties": False},
    "browser_click": {
        "type": "object",
        "properties": {"ref": {"type": "string", "description": "Element reference from browser_snapshot."}},
        "required": ["ref"],
        "additionalProperties": False,
    },
    "browser_type": {
        "type": "object",
        "properties": {
            "ref": {"type": "string", "description": "Editable element reference from browser_snapshot."},
            "text": {"type": "string"},
            "submit": {"type": "boolean"},
            "slowly": {"type": "boolean"},
        },
        "required": ["ref", "text"],
        "additionalProperties": False,
    },
    "browser_fill_form": {
        "type": "object",
        "properties": {
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"ref": {"type": "string"}, "value": {}},
                    "required": ["ref", "value"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["fields"],
        "additionalProperties": False,
    },
    "browser_select_option": {
        "type": "object",
        "properties": {"ref": {"type": "string"}, "values": {"type": "array", "items": {"type": "string"}}},
        "required": ["ref", "values"],
        "additionalProperties": False,
    },
    "browser_check": {
        "type": "object",
        "properties": {"ref": {"type": "string"}},
        "required": ["ref"],
        "additionalProperties": False,
    },
    "browser_uncheck": {
        "type": "object",
        "properties": {"ref": {"type": "string"}},
        "required": ["ref"],
        "additionalProperties": False,
    },
    "browser_wait_for": {
        "type": "object",
        "properties": {"time": {"type": "number", "minimum": 0}},
        "required": ["time"],
        "additionalProperties": False,
    },
    "browser_press_key": {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
        "additionalProperties": False,
    },
}


def _log_progress(message: str) -> None:
    print(f"[INFO] direct_mcp_{message}", flush=True)


def _http_post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url=url, data=body, method="POST")
    for key, value in headers.items():
        request.add_header(key, value)
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=max(1, int(timeout_s))) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        raise RuntimeError(f"openai_compat_http_error:{exc.code}:{raw}") from exc
    except Exception as exc:
        raise RuntimeError(f"openai_compat_request_failed:{exc}") from exc
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"openai_compat_invalid_json:{exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("openai_compat_response_not_object")
    return parsed


def _parse_openai_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("openai_compat_missing_choices")
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content")
    text_parts: List[str] = []
    if isinstance(content, str):
        text_parts.append(content.strip())
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"].strip())
    tool_calls = message.get("tool_calls")
    parsed_tool_calls: List[Dict[str, Any]] = []
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = function.get("name")
            arguments = function.get("arguments")
            if not isinstance(name, str) or not name.strip():
                continue
            if isinstance(arguments, str):
                try:
                    parsed_arguments = json.loads(arguments)
                except Exception:
                    parsed_arguments = {"_raw_arguments": arguments}
            elif isinstance(arguments, dict):
                parsed_arguments = arguments
            else:
                parsed_arguments = {}
            parsed_tool_calls.append(
                {
                    "id": str(call.get("id") or f"tool_{len(parsed_tool_calls) + 1}"),
                    "name": name.strip(),
                    "arguments": parsed_arguments,
                    "raw_arguments": arguments,
                }
            )
    tool_call_transport = "native_tool_calls" if parsed_tool_calls else "none"
    if not parsed_tool_calls:
        text = "\n".join([part for part in text_parts if part]).strip()
        parsed_tool_calls.extend(_parse_raw_mcp_tool_calls(text))
        if parsed_tool_calls:
            tool_call_transport = "text_tool_call_fallback"
    return {
        "text": "\n".join([part for part in text_parts if part]).strip(),
        "tool_calls": parsed_tool_calls,
        "tool_call_transport": tool_call_transport,
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage") if isinstance(payload.get("usage"), dict) else {},
    }


def _parse_raw_mcp_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Recover direct-MCP tool calls returned as text by imperfect OpenAI-compatible parsers."""
    out: List[Dict[str, Any]] = []
    raw = str(text or "")
    for match in RAW_TOOL_CALL_RE.finditer(raw):
        payload = _parse_text_tool_call_payload(match.group("payload"))
        if payload is None:
            continue
        name = payload.get("name")
        arguments = payload.get("arguments")
        if not isinstance(name, str) or not name.strip():
            function = payload.get("function") if isinstance(payload.get("function"), dict) else {}
            name = function.get("name")
            arguments = function.get("arguments", arguments)
        if not isinstance(name, str) or not name.strip():
            continue
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except Exception:
                parsed_arguments = {"_raw_arguments": arguments}
        elif isinstance(arguments, dict):
            parsed_arguments = arguments
        else:
            parsed_arguments = {}
        out.append(
            {
                "id": f"raw_tool_{len(out) + 1}",
                "name": name.strip(),
                "arguments": parsed_arguments,
                "raw_arguments": arguments,
                "source": "assistant_text_tool_call",
            }
        )
    return out


def _parse_text_tool_call_payload(payload_text: str) -> Optional[Dict[str, Any]]:
    raw = str(payload_text or "").strip()
    if not raw:
        return None
    candidates = [raw]
    if not raw.startswith("{"):
        candidates.append("{" + raw.strip().strip(",") + "}")
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _mcp_tools_to_openai_tools(tool_defs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for tool in tool_defs:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name") or "").strip()
        if not name or name not in DIRECT_MCP_MODEL_TOOLS:
            continue
        description = str(tool.get("description") or "").strip()
        schema = DIRECT_MCP_TOOL_SCHEMAS.get(name) or tool.get("inputSchema") or tool.get("input_schema") or {"type": "object", "properties": {}}
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": schema,
                },
            }
        )
    return [tool for tool in out if str(((tool.get("function") or {}).get("name")) or "") in DIRECT_MCP_MODEL_TOOLS]


def _filter_tools_for_visible_controls(openai_tools: List[Dict[str, Any]], control_contract: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Hide tools that are documented but incompatible with the current Google Forms controls."""
    always_visible = {
        "browser_snapshot",
        "browser_click",
        "browser_type",
        "browser_fill_form",
        "browser_check",
        "browser_uncheck",
        "browser_wait_for",
        "browser_press_key",
    }
    visible_names = set(always_visible)
    if any("browser_select_option" in (control.get("valid_mcp_tools") or []) for control in control_contract if isinstance(control, dict)):
        visible_names.add("browser_select_option")
    return [tool for tool in openai_tools if str(((tool.get("function") or {}).get("name")) or "") in visible_names]


def _filter_control_contract_tools(control_contract: List[Dict[str, Any]], visible_tool_names: set[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for control in control_contract:
        if not isinstance(control, dict):
            continue
        item = dict(control)
        item["valid_mcp_tools"] = [
            tool for tool in (control.get("valid_mcp_tools") or []) if isinstance(tool, str) and tool in visible_tool_names
        ]
        out.append(item)
    return out


def _tool_use_guidance(visible_tool_names: set[str]) -> str:
    lines = [
        "Use the official Playwright MCP ref-first workflow: inspect browser_snapshot refs, then call tools with those refs.",
    ]
    if {"browser_type", "browser_fill_form"}.issubset(visible_tool_names):
        lines.append("For editable fields use browser_type or browser_fill_form with refs.")
    elif "browser_type" in visible_tool_names:
        lines.append("For editable fields use browser_type with refs.")
    elif "browser_fill_form" in visible_tool_names:
        lines.append("For editable fields use browser_fill_form with refs.")
    if "browser_check" in visible_tool_names:
        lines.append("For checkboxes and radio buttons prefer browser_click with refs; browser_check is also available when the visible ref supports it.")
    else:
        lines.append("For checkboxes and radio buttons use browser_click with refs.")
    if "browser_select_option" in visible_tool_names:
        lines.append(
            "For real HTML select elements use browser_select_option; for custom Google Forms dropdowns, click the visible ref, inspect the next snapshot, then choose the visible option ref."
        )
    else:
        lines.append("For custom Google Forms dropdowns, click the visible ref, inspect the next snapshot, then choose the visible option ref.")
    return "\n".join(lines)


def _coerce_ref(value: Any) -> Optional[str]:
    ref = str(value or "").strip()
    return ref if ref else None


def _normalize_tool_arguments(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Keep model calls aligned with the public Playwright MCP schemas."""
    raw = arguments if isinstance(arguments, dict) else {}
    if name in {"browser_click", "browser_check", "browser_uncheck"}:
        ref = _coerce_ref(raw.get("ref"))
        return {"ref": ref} if ref else {}
    if name == "browser_type":
        out: Dict[str, Any] = {}
        ref = _coerce_ref(raw.get("ref"))
        if ref:
            out["ref"] = ref
        if "text" in raw:
            out["text"] = str(raw.get("text") or "")
        if "submit" in raw:
            out["submit"] = bool(raw.get("submit"))
        if "slowly" in raw:
            out["slowly"] = bool(raw.get("slowly"))
        return out
    if name == "browser_fill_form":
        fields = []
        for field in raw.get("fields") if isinstance(raw.get("fields"), list) else []:
            if not isinstance(field, dict):
                continue
            ref = _coerce_ref(field.get("ref"))
            if not ref or "value" not in field:
                continue
            fields.append({"ref": ref, "value": field.get("value")})
        return {"fields": fields}
    if name == "browser_select_option":
        out = {}
        ref = _coerce_ref(raw.get("ref"))
        if ref:
            out["ref"] = ref
        values = raw.get("values")
        if isinstance(values, list):
            out["values"] = [str(value) for value in values]
        elif values is not None:
            out["values"] = [str(values)]
        return out
    if name == "browser_wait_for":
        try:
            return {"time": max(0.0, float(raw.get("time")))}
        except Exception:
            return {}
    if name == "browser_press_key":
        key = str(raw.get("key") or "").strip()
        return {"key": key} if key else {}
    return dict(raw)


def _missing_required_tool_fields(name: str, arguments: Dict[str, Any]) -> List[str]:
    schema = DIRECT_MCP_TOOL_SCHEMAS.get(name) or {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    missing = []
    for field in required:
        if field not in arguments or arguments.get(field) in (None, ""):
            missing.append(str(field))
    if name == "browser_fill_form" and not arguments.get("fields"):
        missing.append("fields")
    return missing


def _ref_dom_info(engine: MCPBrowserEngine, ref: str, step_ref: int) -> Dict[str, Any]:
    code = f"""
async (page) => {{
  const ref = {json.dumps(ref)};
  const locator = page.locator(`aria-ref=${{ref}}`);
  const count = await locator.count().catch(() => 0);
  if (!count) {{
    const out = "THESIS_JSON:" + JSON.stringify({{found: false, ref}});
    console.log(out);
    return out;
  }}
  const info = await locator.first().evaluate((el) => {{
    const tag = String(el.tagName || "").toLowerCase();
    const type = String(el.getAttribute("type") || "").toLowerCase();
    const role = String(el.getAttribute("role") || "");
    const aria = String(el.getAttribute("aria-label") || "");
    return {{found: true, tag, type, role, ariaLabel: aria}};
  }}).catch((error) => ({{found: false, ref, error: String(error)}}));
  const out = "THESIS_JSON:" + JSON.stringify({{ref, ...info}});
  console.log(out);
  return out;
}}
"""
    return engine._run_code(code, purpose="mcp_ref_dom_info", step_ref=step_ref)


def _validate_tool_call_for_execution(
    *,
    engine: MCPBrowserEngine,
    name: str,
    arguments: Dict[str, Any],
    step_idx: int,
) -> Optional[Dict[str, Any]]:
    missing = _missing_required_tool_fields(name, arguments)
    if missing:
        return {"error": "invalid_mcp_tool_arguments", "tool": name, "missing": missing}
    if name == "browser_select_option":
        ref = _coerce_ref(arguments.get("ref"))
        if not ref:
            return {"error": "invalid_mcp_tool_arguments", "tool": name, "missing": ["ref"]}
        info = _ref_dom_info(engine, ref, step_idx)
        if str(info.get("tag") or "").lower() != "select":
            return {
                "error": "incompatible_mcp_tool_for_ref",
                "tool": name,
                "ref": ref,
                "required_tag": "select",
                "actual_tag": info.get("tag"),
                "actual_type": info.get("type"),
                "actual_role": info.get("role"),
                "detail": "browser_select_option is only valid for HTML select elements.",
            }
    return None


def _tool_result_to_message_content(payload: Dict[str, Any], limit_chars: int = 12000) -> str:
    text = json.dumps(payload, ensure_ascii=True, default=str)
    if len(text) <= limit_chars:
        return text
    return text[:limit_chars].rstrip() + "...(truncated)"


def _observation_prompt(
    form_url: str,
    remaining_answers: List[Dict[str, Any]],
    page_text: str,
    url: str,
    step_idx: int,
    accessibility_snapshot: str = "",
    control_contract: Optional[List[Dict[str, Any]]] = None,
    model_visible_tool_names: Optional[List[str]] = None,
    fill_only_done: bool = False,
) -> str:
    answers_json = json.dumps(remaining_answers, ensure_ascii=True, indent=2)
    page_excerpt = page_text[:8000]
    snapshot_excerpt = str(accessibility_snapshot or "")[:12000]
    snapshot_block = ""
    if snapshot_excerpt:
        snapshot_block = (
            "Latest Playwright MCP accessibility snapshot:\n"
            f"{snapshot_excerpt}\n\n"
            "When a Playwright MCP tool argument requires an element reference, use refs from the snapshot rather than invented CSS selectors.\n\n"
        )
    contract_block = ""
    if control_contract:
        contract_excerpt = json.dumps(control_contract[:80], ensure_ascii=True, indent=2)[:12000]
        contract_block = (
            "Visible Google Forms control/tool compatibility metadata:\n"
            f"{contract_excerpt}\n\n"
            "This metadata describes which documented Playwright MCP tools are compatible with each visible ref. "
            "It does not choose answers or prescribe an action sequence.\n\n"
        )
    visible_tools = ", ".join(sorted(set(model_visible_tool_names or DIRECT_MCP_MODEL_TOOLS)))
    visible_tool_set = set(model_visible_tool_names or DIRECT_MCP_MODEL_TOOLS)
    tool_guidance = _tool_use_guidance(visible_tool_set)
    if fill_only_done:
        terminal_guidance = (
            "Fill-only terminal condition: enter all target answers, never click Submit, and reply with plain text DONE only when the visible form values match the target answers.\n"
            "If a Submit button is visible, do not click it in this condition. Use DONE for completion after filling.\n"
            "Match answer values exactly; do not reformat dates, times, emails, or option text.\n"
            "If a click does not change the form after one retry, use a different tool/action or scroll; do not repeat the same click.\n"
        )
    else:
        terminal_guidance = (
            "Benchmark terminal condition: DONE means you have observed a form submission confirmation page, not merely that fields appear filled.\n"
            "Before submitting, double-check the visible form state against the target answers as well as the current observation allows.\n"
            "Do not use DONE to say the form is ready to submit. If the form appears correct and you intend to submit, call a Playwright MCP tool on the visible Submit button ref.\n"
        )
    return (
        "You are controlling a browser only through Playwright MCP tools.\n"
        "Use the provided Playwright MCP tools directly. Do not invent custom actions or JSON schemas.\n"
        "Do not output pyautogui, mouse coordinates, scripts, plans, or native computer-use actions in this condition.\n"
        "If your serving backend cannot emit native tool_calls, output exactly one textual fallback call as "
        "<tool_call>{\"name\":\"browser_click\",\"arguments\":{\"ref\":\"...\"}}</tool_call> using one provided tool name.\n"
        f"{tool_guidance}\n"
        f"Use only these Playwright MCP tool names in this step: {visible_tools}.\n"
        f"{terminal_guidance}"
        "When you believe that terminal condition is met, reply with plain text DONE and a brief note.\n"
        "If you cannot make further safe progress, reply with plain text STOP and a brief reason.\n\n"
        f"Form URL: {form_url}\n"
        f"Current URL: {url}\n"
        f"Step index: {step_idx}\n"
        "Target answers:\n"
        f"{answers_json}\n\n"
        f"{snapshot_block}"
        f"{contract_block}"
        "Current page text excerpt:\n"
        f"{page_excerpt}\n"
    )


def _done_text(text: str) -> bool:
    raw = str(text or "").strip().lower()
    if raw.startswith("done") or raw.startswith("stop"):
        return True
    return bool(re.search(r"\b(done|stop)\b", raw))


def _load_run_answers(answers_path: Path, run_index: int) -> List[Dict[str, Any]]:
    for idx, run_spec in enumerate(iter_run_specs(answers_path), start=1):
        if idx == run_index:
            answers = run_spec.get("answers", [])
            if not isinstance(answers, list):
                raise ValueError(f"Run {run_index} answers must be a list")
            return answers
    raise IndexError(f"Run index out of range: {run_index} for {answers_path}")


def _detect_submit_success(engine: MCPBrowserEngine, step_ref: Optional[int]) -> Tuple[bool, Dict[str, Any]]:
    code = """
async (page) => {
  const text = (await page.locator("body").innerText({ timeout: 3000 }).catch(() => "") || "");
  const url = page.url();
  const normalized = text.toLowerCase();
  const success = normalized.includes("response has been recorded")
    || normalized.includes("your response has been recorded")
    || normalized.includes("ihre antwort wurde gesendet")
    || normalized.includes("antwort wurde gesendet")
    || normalized.includes("antwort wurde erfasst")
    || normalized.includes("deine antwort wurde aufgezeichnet");
  const out = "THESIS_JSON:" + JSON.stringify({ success, text, url });
  console.log(out);
  return out;
}
"""
    payload = engine._run_code(code, purpose="detect_submit_success", step_ref=step_ref)
    return bool(payload.get("success")), {"page_text": str(payload.get("text") or ""), "url": str(payload.get("url") or "")}


def _take_accessibility_snapshot(mcp: MCPClient, trace: TraceLogger, step_ref: int) -> str:
    args: Dict[str, Any] = {}
    try:
        result = mcp.call_tool("browser_snapshot", args)
        trace.log_event("browser_snapshot", args, step_ref=step_ref, extra={"backend": "mcp_server", "observation": True})
        return _tool_result_to_message_content(result, limit_chars=12000)
    except Exception as exc:
        trace.log_event("browser_snapshot", args, step_ref=step_ref, ok=False, error=str(exc), extra={"backend": "mcp_server", "observation": True})
        return json.dumps({"error": str(exc)}, ensure_ascii=True)


def _valid_tools_for_control(info: Dict[str, Any]) -> List[str]:
    tag = str(info.get("tag") or "").lower()
    role = str(info.get("role") or "").lower()
    input_type = str(info.get("type") or "").lower()
    tools: List[str] = []
    if tag == "select":
        tools.append("browser_select_option")
    if tag in {"input", "textarea"} or role in {"textbox", "combobox"}:
        tools.extend(["browser_type", "browser_click"])
    if role in {"radio", "checkbox", "button"} or tag == "button":
        tools.append("browser_click")
    if role in {"radio", "checkbox"}:
        tools.append("browser_check")
    if input_type in {"checkbox"}:
        tools.extend(["browser_check", "browser_uncheck", "browser_click"])
    if input_type in {"radio"}:
        tools.append("browser_click")
    out: List[str] = []
    for tool in tools:
        if tool in DIRECT_MCP_MODEL_TOOLS and tool not in out:
            out.append(tool)
    return out


def _control_by_ref(control_contract: List[Dict[str, Any]], ref: Optional[str]) -> Optional[Dict[str, Any]]:
    if not ref:
        return None
    for control in control_contract:
        if isinstance(control, dict) and str(control.get("ref") or "") == ref:
            return control
    return None


def _control_looks_like_submit(control: Optional[Dict[str, Any]], arguments: Optional[Dict[str, Any]] = None) -> bool:
    values = []
    if isinstance(control, dict):
        values.extend([control.get("label"), control.get("aria_label"), control.get("role"), control.get("tag")])
    if isinstance(arguments, dict):
        values.extend([arguments.get("ref"), arguments.get("element"), arguments.get("label")])
    text = " ".join(str(value or "") for value in values).lower()
    return any(token in text for token in ["submit", "send", "senden", "absenden"])


def _verification_snapshot(engine: MCPBrowserEngine, question_states: List[Dict[str, Any]], step_ref: int) -> Dict[str, Any]:
    correct = 0
    attempted = 0
    total = len(question_states)
    details: List[Dict[str, Any]] = []
    for idx, question_state in enumerate(question_states):
        try:
            verification = engine.verify_entry(question_state, step_ref + idx)
        except Exception as exc:
            verification = {"verified": False, "actual_value": None, "detail": str(exc)}
        verified_correct = bool(verification.get("verified")) and rbe._value_matches(question_state.get("value"), verification.get("actual_value"))
        actual_value = verification.get("actual_value")
        is_attempted = bool(verification.get("verified")) or actual_value not in (None, "", [])
        correct += int(verified_correct)
        attempted += int(is_attempted)
        details.append(
            {
                "question_id": question_state.get("question_id"),
                "label": question_state.get("label"),
                "verified": bool(verification.get("verified")),
                "verified_correct": verified_correct,
                "attempted": is_attempted,
                "actual_value": actual_value,
            }
        )
    return {"correct": correct, "attempted": attempted, "total": total, "details": details}


def _collect_control_contract(engine: MCPBrowserEngine, step_ref: int) -> List[Dict[str, Any]]:
    code = """
async (page) => {
  return await page.evaluate(() => {
  const selectors = [
    "input",
    "textarea",
    "select",
    "button",
    "[role='button']",
    "[role='radio']",
    "[role='checkbox']",
    "[role='combobox']",
    "[contenteditable='true']"
  ];
  const clean = (value) => String(value || "").replace(/\\s+/g, " ").trim().slice(0, 160);
  const labelFor = (node) => {
    const aria = clean(node.getAttribute("aria-label"));
    if (aria) return aria;
    const listitem = node.closest("div[role='listitem']");
    if (listitem) return clean(listitem.innerText);
    return clean(node.innerText || node.value || node.placeholder || "");
  };
  const nodes = Array.from(document.querySelectorAll(selectors.join(",")));
  const out = [];
  const seen = new Set();
  for (const node of nodes) {
    const rect = node.getBoundingClientRect();
    if (!rect || rect.width < 2 || rect.height < 2 || rect.bottom < 0 || rect.right < 0 || rect.top > innerHeight || rect.left > innerWidth) continue;
    const style = getComputedStyle(node);
    if (!style || style.visibility === "hidden" || style.display === "none") continue;
    const ref = node.getAttribute("aria-ref");
    if (!ref || seen.has(ref)) continue;
    seen.add(ref);
    out.push({
      ref,
      label: labelFor(node),
      tag: String(node.tagName || "").toLowerCase(),
      type: String(node.getAttribute("type") || "").toLowerCase(),
      role: String(node.getAttribute("role") || ""),
      aria_label: clean(node.getAttribute("aria-label")),
      value: clean(node.value || node.getAttribute("aria-checked") || ""),
      checked: node.checked === true || node.getAttribute("aria-checked") === "true",
      disabled: node.disabled === true || node.getAttribute("aria-disabled") === "true"
    });
  }
  const result = "THESIS_JSON:" + JSON.stringify({controls: out});
  console.log(result);
  return result;
  });
}
"""
    payload = engine._run_code(code, purpose="mcp_control_contract", step_ref=step_ref)
    controls = payload.get("controls") if isinstance(payload.get("controls"), list) else []
    out: List[Dict[str, Any]] = []
    for item in controls:
        if not isinstance(item, dict):
            continue
        normalized = {
            "ref": item.get("ref"),
            "label": item.get("label"),
            "tag": item.get("tag"),
            "type": item.get("type"),
            "role": item.get("role"),
            "aria_label": item.get("aria_label"),
            "checked": item.get("checked"),
            "disabled": item.get("disabled"),
        }
        normalized["valid_mcp_tools"] = _valid_tools_for_control(normalized)
        out.append(normalized)
    return out


def _build_messages(
    *,
    model_kind: str,
    observation_text: str,
    screenshot_path: Optional[Path],
    history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a benchmarked browser agent. Use the Playwright MCP tools directly. "
                "Do not emit any custom action schema. Only use provided tools or return DONE/STOP."
            ),
        }
    ]
    messages.extend(history)
    if model_kind in {"vlm", "computer_use_agent"}:
        if screenshot_path is None or not screenshot_path.exists():
            raise RuntimeError(f"{model_kind}_observation_requires_screenshot")
        import base64

        data_url = "data:image/png;base64," + base64.b64encode(screenshot_path.read_bytes()).decode("ascii")
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": observation_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": observation_text})
    return messages


def _compact_history(history: List[Dict[str, Any]], max_turns: int) -> List[Dict[str, Any]]:
    """Keep complete recent assistant/tool turns without splitting tool-call pairs."""
    limit = max(0, int(max_turns))
    if limit == 0:
        return list(history)
    assistant_starts = [idx for idx, message in enumerate(history) if message.get("role") == "assistant"]
    if len(assistant_starts) <= limit:
        return list(history)
    start = assistant_starts[-limit]
    omitted_turns = len(assistant_starts) - limit
    return [
        {
            "role": "user",
            "content": (
                f"Context note: {omitted_turns} earlier browser-action turn(s) were omitted to stay within the model context window. "
                "Treat the current observation, screenshot, remaining answers, and recent tool results as authoritative."
            ),
        },
        *history[start:],
    ]


def _assistant_history_message(model_reply: Dict[str, Any]) -> Dict[str, Any]:
    tool_calls = model_reply.get("tool_calls") or []
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": "" if tool_calls else (model_reply.get("text") or ""),
    }
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=True)},
            }
            for call in tool_calls
        ]
    return message


def _text_tool_call_payload(call: Dict[str, Any]) -> str:
    return "<tool_call>" + json.dumps({"name": call.get("name"), "arguments": call.get("arguments") or {}}, ensure_ascii=True) + "</tool_call>"


def _assistant_history_message_for_transport(model_reply: Dict[str, Any], *, native_tool_call_history: bool) -> Dict[str, Any]:
    if native_tool_call_history:
        return _assistant_history_message(model_reply)
    text = str(model_reply.get("text") or "").strip()
    if not text and (model_reply.get("tool_calls") or []):
        text = "\n".join(_text_tool_call_payload(call) for call in model_reply.get("tool_calls") or [])
    return {"role": "assistant", "content": text}


def _tool_result_history_message(content: str, *, native_tool_call_history: bool, tool_call_id: str) -> Dict[str, Any]:
    if native_tool_call_history:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}
    return {"role": "user", "content": f"Tool result for {tool_call_id}:\n{content}"}


def _call_model(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    native_tool_calls: bool,
    max_new_tokens: int,
    timeout_s: int,
) -> Dict[str, Any]:
    started = time.perf_counter()
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": int(max_new_tokens),
        "messages": messages,
    }
    if native_tool_calls:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    response = _http_post_json(
        url=base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        payload=payload,
        timeout_s=timeout_s,
    )
    parsed = _parse_openai_response(response)
    parsed["duration_s"] = round(time.perf_counter() - started, 3)
    return parsed


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct Playwright MCP evaluation for one model/form/run.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["text_llm", "vlm", "computer_use_agent"], required=True)
    parser.add_argument("--form-id", required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--answers-root", default=DEFAULT_ANSWERS_ROOT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--trial-id")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--api-timeout-s", type=int, default=DEFAULT_API_TIMEOUT_S)
    parser.add_argument("--browser-mcp-timeout-ms", type=int, default=DEFAULT_BROWSER_MCP_TIMEOUT_MS)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--history-turns",
        type=int,
        default=0,
        help="Retain only this many complete recent assistant/tool turns; 0 keeps full history.",
    )
    parser.add_argument("--browser-mcp-cmd")
    parser.add_argument("--fill-only-done", action="store_true", default=False)
    parser.add_argument("--run-label")
    parser.add_argument("--retention-window", type=int, default=rbe.DEFAULT_RETENTION_WINDOW)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    run_label = rbe._make_run_label(args.run_label)
    run_started_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    max_new_tokens = int(
        args.max_new_tokens
        if args.max_new_tokens is not None
        else (DEFAULT_VLM_MAX_NEW_TOKENS if args.model_kind in {"vlm", "computer_use_agent"} else DEFAULT_TEXT_MAX_NEW_TOKENS)
    )

    config_path = (ROOT_DIR / args.config).resolve()
    model_cfg = get_model_by_id(config_path, args.model_id)
    if str(model_cfg.get("provider") or "") != "openai_compat":
        raise ValueError(f"run_qwen_direct_mcp_eval expects provider=openai_compat: {args.model_id}")
    if str(model_cfg.get("track") or "") != "direct_mcp_tool_use":
        raise ValueError(f"run_qwen_direct_mcp_eval expects track=direct_mcp_tool_use: {args.model_id}")

    base_url = str(os.environ.get("OPENAI_BASE_URL") or model_cfg.get("openai_base_url") or "").rstrip("/")
    model_name = str(os.environ.get("OPENAI_MODEL") or model_cfg.get("openai_model") or model_cfg.get("served_model_name") or "").strip()
    api_key = str(os.environ.get("OPENAI_API_KEY") or model_cfg.get("openai_api_key") or "EMPTY")
    if not base_url or not model_name:
        raise RuntimeError("direct_mcp_openai_compat_missing_base_url_or_model")
    native_tool_calls_env = str(os.environ.get("DIRECT_MCP_NATIVE_TOOL_CALLS") or "").strip().lower()
    if native_tool_calls_env:
        native_tool_calls_enabled = native_tool_calls_env in {"1", "true", "yes", "on"}
    else:
        native_tool_calls_enabled = args.model_kind != "computer_use_agent"
    task_mode = "fill_only_done" if args.fill_only_done else "fill_and_submit"

    _log_progress(
        "trial_start "
        f"model_id={args.model_id} model_kind={args.model_kind} form_id={args.form_id} "
        f"run_index={args.run_index} experiment_id={args.experiment_id} "
        f"api_model={model_name} base_url={base_url} native_tool_calls={native_tool_calls_enabled} task_mode={task_mode}"
    )

    form_spec = load_form_spec(args.form_id, ROOT_DIR / "src" / "forms")
    form_url = force_english_google_forms_url(str(form_spec.get("form_url") or form_spec.get("url") or ""))
    if not form_url:
        raise ValueError(f"Missing form_url in spec for {args.form_id}")

    answers_path = resolve_answers_path(argparse.Namespace(answers_root=args.answers_root, answers_file="runs.json"), args.form_id)
    answers = _load_run_answers(answers_path, args.run_index)
    answer_run_id = f"run_{args.run_index:04d}"
    trial_id = args.trial_id or rbe._make_trial_id()
    paths = rbe._build_trial_paths(args, args.model_id, args.form_id, answer_run_id, trial_id)
    question_states = rbe._build_question_states(answers)
    rbe._write_json(paths["answers_path"], question_states)
    rbe._touch(paths["model_io_path"])
    rbe._touch(paths["step_inputs_path"])

    start_time = time.perf_counter()
    trace = TraceLogger(path=paths["trace_path"], start_time=start_time, validate_mcp_actions=False, strict_mcp_validation=False, mcp_client=None)
    tool_call_count = 0
    tool_error_count = 0
    tool_call_transport_counts: Dict[str, int] = {}
    accessibility_snapshot_count = 0
    inference_roundtrip_s: Optional[float] = None
    terminal_screenshot_path: Optional[str] = None
    history: List[Dict[str, Any]] = []
    stop_reason: Optional[str] = None
    failure_category: Optional[str] = None
    failure_detail: Optional[str] = None
    submit_success = False
    success = False
    invalid_tool_call_count = 0
    invalid_tool_signature_counts: Dict[str, int] = {}
    last_invalid_tool_payload: Optional[Dict[str, Any]] = None
    submit_attempts: List[Dict[str, Any]] = []

    required_tools = ["browser_navigate", "browser_run_code", "browser_wait_for", "browser_close", "browser_snapshot"]
    if args.model_kind in {"vlm", "computer_use_agent"}:
        required_tools.append("browser_take_screenshot")
    command: Any = args.browser_mcp_cmd or _default_mcp_server_command(args, paths)

    mcp = None
    engine = None
    try:
        mcp = MCPClient(
            command=command,
            timeout_ms=args.browser_mcp_timeout_ms,
            required_tools=required_tools,
            env={
                key: value
                for key, value in {
                    "PLAYWRIGHT_BROWSERS_PATH": os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""),
                    "LD_LIBRARY_PATH": os.environ.get("NODE_LD_LIBRARY_PATH_FOR_MCP", ""),
                }.items()
                if value
            },
        )
        engine = MCPBrowserEngine(
            mcp_client=mcp,
            trace=trace,
            observations_dir=paths["observations_dir"],
            timeout_ms=15000,
            type_delay_ms=120,
            action_delay_ms=220,
            take_screenshots=True,
        )
        env = engine.navigate(form_url)
        tool_defs = mcp.get_tool_definitions()
        openai_tools = _mcp_tools_to_openai_tools(tool_defs)
        if not openai_tools:
            raise RuntimeError("playwright_mcp_exposed_no_tools")
        _log_progress(
            "model_visible_tool_contract "
            f"model_id={args.model_id} tools={','.join(sorted(str(((tool.get('function') or {}).get('name')) or '') for tool in openai_tools))}"
        )

        for step_idx in range(args.max_steps):
            elapsed = time.perf_counter() - start_time
            if elapsed >= args.timeout_s:
                stop_reason = "timeout"
                failure_category = "timeout"
                failure_detail = f"timeout after {args.timeout_s}s"
                break

            page_text = engine.get_page_text(step_idx)
            snapshot_text = _take_accessibility_snapshot(mcp, trace, step_idx)
            accessibility_snapshot_count += 1
            control_contract = _collect_control_contract(engine, step_idx)
            current_url = str((env or {}).get("url") or form_url)
            screenshot_path: Optional[Path] = None
            if args.model_kind in {"vlm", "computer_use_agent"}:
                screenshot = engine.take_observation_screenshot(f"step_{step_idx:04d}_vlm.png", step_ref=step_idx)
                screenshot_path = Path(screenshot) if screenshot else None
            remaining_answers = rbe._serialize_remaining_answers(question_states)
            step_tools = _filter_tools_for_visible_controls(openai_tools, control_contract)
            step_tool_names = {str(((tool.get("function") or {}).get("name")) or "") for tool in step_tools}
            control_contract = _filter_control_contract_tools(control_contract, step_tool_names)
            observation_text = _observation_prompt(
                form_url=form_url,
                remaining_answers=remaining_answers,
                page_text=page_text,
                url=current_url,
                step_idx=step_idx,
                accessibility_snapshot=snapshot_text,
                control_contract=control_contract,
                model_visible_tool_names=sorted(step_tool_names),
                fill_only_done=bool(args.fill_only_done),
            )
            compacted_history = _compact_history(history, args.history_turns)
            messages = _build_messages(
                model_kind=args.model_kind,
                observation_text=observation_text,
                screenshot_path=screenshot_path,
                history=compacted_history,
            )
            step_input = {
                "phase": "step_input",
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "step_index": step_idx,
                "tool_protocol": "mcp",
                "mcp_server": "playwright",
                "remaining_answers": remaining_answers,
                "accessibility_snapshot_excerpt": snapshot_text[:4000],
                "control_contract": control_contract,
                "page_text_excerpt": page_text[:4000],
                "screenshot_path": str(screenshot_path) if screenshot_path else None,
                "prompt_char_count": len(observation_text),
                "prompt_token_estimate": max(1, (len(observation_text) + 3) // 4),
                "history_message_count": len(history),
                "retained_history_message_count": len(compacted_history),
                "history_turn_limit": args.history_turns,
                "model_visible_tools": sorted(step_tool_names),
            }
            rbe._append_jsonl(paths["step_inputs_path"], step_input)
            _log_progress(
                "step_start "
                f"model_id={args.model_id} form_id={args.form_id} run_index={args.run_index} "
                f"step_index={step_idx} visible_tools={','.join(sorted(step_tool_names))}"
            )

            model_reply = _call_model(
                base_url=base_url,
                api_key=api_key,
                model=model_name,
                messages=messages,
                tools=step_tools,
                native_tool_calls=native_tool_calls_enabled,
                max_new_tokens=max_new_tokens,
                timeout_s=args.api_timeout_s,
            )
            inference_roundtrip_s = model_reply.get("duration_s")
            transport = str(model_reply.get("tool_call_transport") or "none")
            tool_call_transport_counts[transport] = tool_call_transport_counts.get(transport, 0) + 1
            rbe._append_jsonl(
                paths["model_io_path"],
                {
                    "phase": "step",
                    "step_index": step_idx,
                    "tool_protocol": "mcp",
                    "mcp_server": "playwright",
                    "assistant_text": model_reply.get("text"),
                    "tool_calls": model_reply.get("tool_calls"),
                    "tool_call_transport": model_reply.get("tool_call_transport"),
                    "model_inference": {"duration_s": model_reply.get("duration_s"), "usage": model_reply.get("usage")},
                },
            )
            history.append(_assistant_history_message_for_transport(model_reply, native_tool_call_history=native_tool_calls_enabled))
            tool_calls = model_reply.get("tool_calls") or []
            if not tool_calls:
                if _done_text(model_reply.get("text") or ""):
                    stop_reason = "done"
                    break
                stop_reason = "model_no_tool_calls"
                failure_category = "model_no_tool_calls"
                failure_detail = "assistant returned no tool calls and no DONE/STOP text"
                break
            _log_progress(
                "step_tool_calls "
                f"model_id={args.model_id} form_id={args.form_id} run_index={args.run_index} "
                f"step_index={step_idx} count={len(tool_calls)} transport={transport}"
            )

            for call in tool_calls:
                tool_call_count += 1
                name = str(call.get("name") or "").strip()
                raw_arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
                arguments = _normalize_tool_arguments(name, raw_arguments)
                control = _control_by_ref(control_contract, _coerce_ref(arguments.get("ref")))
                is_submit_attempt = name == "browser_click" and _control_looks_like_submit(control, arguments)
                submit_attempt: Optional[Dict[str, Any]] = None
                pre_click_snapshot: Optional[Dict[str, Any]] = None
                if name == "browser_click" and engine is not None:
                    pre_click_snapshot = _verification_snapshot(engine, question_states, step_ref=(step_idx * 1000) + 100)
                if is_submit_attempt and engine is not None and pre_click_snapshot is not None:
                    submit_attempt = {
                        "step_index": step_idx,
                        "tool_call_index": tool_call_count,
                        "tool": name,
                        "arguments": arguments,
                        "ref": arguments.get("ref"),
                        "control": control,
                        "pre_submit_verified_correctness": pre_click_snapshot["correct"],
                        "pre_submit_attempted_count": pre_click_snapshot["attempted"],
                        "pre_submit_question_total": pre_click_snapshot["total"],
                    }
                if args.fill_only_done and is_submit_attempt:
                    tool_error_count += 1
                    blocked_payload = {
                        "error": "submit_disabled_in_fill_only_done",
                        "detail": "This evaluation condition requires filling fields and returning DONE without clicking Submit.",
                    }
                    if submit_attempt is not None:
                        submit_attempt.update(
                            {
                                "tool_call_ok": False,
                                "tool_error": blocked_payload["error"],
                                "post_submit_success": False,
                                "blocked_by_harness": True,
                            }
                        )
                        submit_attempts.append(submit_attempt)
                    trace.log_event(name, arguments, step_ref=step_idx, ok=False, error=json.dumps(blocked_payload, ensure_ascii=True), extra={"backend": "fill_only_done_guard"})
                    history.append(
                        _tool_result_history_message(
                            json.dumps(blocked_payload, ensure_ascii=True),
                            native_tool_call_history=native_tool_calls_enabled,
                            tool_call_id=call["id"],
                        )
                    )
                    rbe._append_jsonl(
                        paths["model_io_path"],
                        {
                            "phase": "tool_error",
                            "step_index": step_idx,
                            "tool_protocol": "mcp",
                            "mcp_server": "playwright",
                            "tool_name": name,
                            "tool_arguments": arguments,
                            "error": blocked_payload,
                            "blocked_by_harness": True,
                        },
                    )
                    if pre_click_snapshot is not None and pre_click_snapshot["correct"] >= pre_click_snapshot["total"]:
                        stop_reason = "filled_without_submit"
                        success = True
                        submit_success = False
                        failure_category = None
                        failure_detail = None
                        break
                    continue
                if name not in step_tool_names or name not in DIRECT_MCP_MODEL_TOOLS or name not in mcp.available_tools:
                    tool_error_count += 1
                    error_payload = {
                        "error": f"unsupported_or_hidden_tool:{name}",
                        "available_tools": sorted(set(mcp.available_tools).intersection(step_tool_names)),
                    }
                    history.append(
                        _tool_result_history_message(
                            json.dumps(error_payload, ensure_ascii=True),
                            native_tool_call_history=native_tool_calls_enabled,
                            tool_call_id=call["id"],
                        )
                    )
                    rbe._append_jsonl(
                        paths["model_io_path"],
                        {
                            "phase": "tool_error",
                            "step_index": step_idx,
                            "tool_protocol": "mcp",
                            "mcp_server": "playwright",
                            "tool_name": name,
                            "tool_arguments": arguments,
                            "error": error_payload["error"],
                        },
                    )
                    continue
                validation_error = _validate_tool_call_for_execution(
                    engine=engine,
                    name=name,
                    arguments=arguments,
                    step_idx=step_idx,
                )
                if validation_error:
                    tool_error_count += 1
                    invalid_tool_call_count += 1
                    last_invalid_tool_payload = validation_error
                    signature = json.dumps(
                        {"tool": name, "arguments": arguments, "error": validation_error.get("error")},
                        sort_keys=True,
                        ensure_ascii=True,
                    )
                    invalid_tool_signature_counts[signature] = invalid_tool_signature_counts.get(signature, 0) + 1
                    trace.log_event(name, arguments, step_ref=step_idx, ok=False, error=json.dumps(validation_error, ensure_ascii=True), extra={"backend": "mcp_contract_guard"})
                    history.append(
                        _tool_result_history_message(
                            json.dumps(validation_error, ensure_ascii=True),
                            native_tool_call_history=native_tool_calls_enabled,
                            tool_call_id=call["id"],
                        )
                    )
                    rbe._append_jsonl(
                        paths["model_io_path"],
                        {
                            "phase": "tool_error",
                            "step_index": step_idx,
                            "tool_protocol": "mcp",
                            "mcp_server": "playwright",
                            "tool_name": name,
                            "tool_arguments": arguments,
                            "raw_tool_arguments": raw_arguments,
                            "error": validation_error,
                            "repeat_same_invalid_tool_count": invalid_tool_signature_counts[signature],
                        },
                    )
                    if invalid_tool_signature_counts[signature] >= 4:
                        stop_reason = "repeat_invalid_tool_call"
                        failure_category = "invalid_mcp_tool_loop"
                        failure_detail = json.dumps(validation_error, ensure_ascii=True)
                        break
                    continue
                try:
                    result = mcp.call_tool(name, arguments)
                    trace.log_event(name, arguments, step_ref=step_idx, extra={"backend": "mcp_server"})
                    history.append(
                        _tool_result_history_message(
                            _tool_result_to_message_content(result),
                            native_tool_call_history=native_tool_calls_enabled,
                            tool_call_id=call["id"],
                        )
                    )
                    post_submit_success = False
                    post_submit_probe: Dict[str, Any] = {}
                    if name == "browser_click" and engine is not None:
                        post_submit_success, post_submit_probe = _detect_submit_success(engine, step_ref=step_idx)
                        if post_submit_success and submit_attempt is None and pre_click_snapshot is not None:
                            submit_attempt = {
                                "step_index": step_idx,
                                "tool_call_index": tool_call_count,
                                "tool": name,
                                "arguments": arguments,
                                "ref": arguments.get("ref"),
                                "control": control,
                                "submit_detected_by": "post_click_confirmation_probe",
                                "pre_submit_verified_correctness": pre_click_snapshot["correct"],
                                "pre_submit_attempted_count": pre_click_snapshot["attempted"],
                                "pre_submit_question_total": pre_click_snapshot["total"],
                            }
                    if submit_attempt is not None and engine is not None:
                        post_snapshot = _verification_snapshot(engine, question_states, step_ref=(step_idx * 1000) + 500)
                        submit_attempt.update(
                            {
                                "tool_call_ok": True,
                                "post_submit_success": bool(post_submit_success),
                                "post_submit_url": post_submit_probe.get("url"),
                                "post_submit_page_text_excerpt": str(post_submit_probe.get("page_text") or "")[:1000],
                                "post_submit_verified_correctness": post_snapshot["correct"],
                                "post_submit_attempted_count": post_snapshot["attempted"],
                                "post_submit_question_total": post_snapshot["total"],
                                "submitted_while_incomplete": post_snapshot["correct"] < post_snapshot["total"],
                            }
                        )
                        submit_attempts.append(submit_attempt)
                        rbe._append_jsonl(
                            paths["model_io_path"],
                            {
                                "phase": "submit_attempt",
                                "step_index": step_idx,
                                "tool_protocol": "mcp",
                                "mcp_server": "playwright",
                                **submit_attempt,
                            },
                        )
                    if args.fill_only_done and engine is not None and name != "browser_snapshot":
                        fill_snapshot = _verification_snapshot(engine, question_states, step_ref=(step_idx * 1000) + 700)
                        if fill_snapshot["correct"] >= fill_snapshot["total"]:
                            stop_reason = "filled_without_submit"
                            success = True
                            submit_success = False
                            failure_category = None
                            failure_detail = None
                            break
                except Exception as exc:
                    tool_error_count += 1
                    trace.log_event(name, arguments, step_ref=step_idx, ok=False, error=str(exc), extra={"backend": "mcp_server"})
                    history.append(
                        _tool_result_history_message(
                            json.dumps({"error": str(exc)}, ensure_ascii=True),
                            native_tool_call_history=native_tool_calls_enabled,
                            tool_call_id=call["id"],
                        )
                    )
                    if submit_attempt is not None:
                        submit_attempt.update({"tool_call_ok": False, "tool_error": str(exc), "post_submit_success": False})
                        submit_attempts.append(submit_attempt)
                        rbe._append_jsonl(
                            paths["model_io_path"],
                            {
                                "phase": "submit_attempt",
                                "step_index": step_idx,
                                "tool_protocol": "mcp",
                                "mcp_server": "playwright",
                                **submit_attempt,
                            },
                        )
            if stop_reason is not None:
                break

        if stop_reason is None:
            stop_reason = "max_steps_exceeded"
            failure_category = "max_steps_exceeded"
            failure_detail = f"max_steps={args.max_steps}"

        if engine is not None and not args.fill_only_done:
            submit_success, submit_probe = _detect_submit_success(engine, step_ref=args.max_steps)
            success = submit_success
            if stop_reason == "done" and not submit_success:
                stop_reason = "premature_done_without_submit"
                failure_category = "premature_done_without_submit"
                failure_detail = "assistant returned DONE/STOP before submission success was detected"
            env = dict(env or {})
            env["submit_probe"] = submit_probe
            terminal_screenshot_path = engine.take_observation_screenshot("final.png" if success else "error.png", step_ref=None)
        elif engine is not None:
            submit_success = False
            env = dict(env or {})
            env["submit_probe"] = {"skipped": True, "reason": "fill_only_done"}
            terminal_screenshot_path = engine.take_observation_screenshot("error.png", step_ref=None)

        for idx, question_state in enumerate(question_states):
            try:
                verification = engine.verify_entry(question_state, args.max_steps + idx if engine is not None else idx) if engine is not None else {}
            except Exception as exc:
                verification = {"verified": False, "actual_value": None, "detail": str(exc)}
            previous_correct = bool(question_state.get("verified_correct"))
            current_verified = bool(verification.get("verified"))
            current_correct = current_verified and rbe._value_matches(question_state.get("value"), verification.get("actual_value"))
            preserve_previous = bool(args.fill_only_done) and previous_correct and not current_verified
            question_state["last_verification"] = verification
            if current_verified or not preserve_previous:
                question_state["actual_value"] = verification.get("actual_value")
            question_state["verified"] = current_verified or preserve_previous
            question_state["verified_correct"] = current_correct or preserve_previous
            actual_value = question_state.get("actual_value")
            question_state["attempted"] = bool(question_state.get("verified")) or actual_value not in (None, "", [])
            question_state["attempted_correct"] = bool(question_state["verified_correct"])
            question_state["final_status"] = "correct_verified" if question_state["verified_correct"] else "failed"
        if args.fill_only_done:
            verified_correct = sum(1 for state in question_states if bool(state.get("verified_correct")))
            question_total = len(question_states)
            submit_success = False
            success = question_total > 0 and verified_correct >= question_total
            if success:
                failure_category = None
                failure_detail = None
                if stop_reason in {None, "done", "max_steps_exceeded"}:
                    stop_reason = "filled_without_submit" if stop_reason != "done" else "done"
            elif stop_reason == "done":
                stop_reason = "done_incomplete_fill_only"
                failure_category = "done_incomplete_fill_only"
                failure_detail = f"verified_correctness={verified_correct}/{question_total}"
            elif stop_reason == "filled_without_submit":
                stop_reason = "fill_only_verification_mismatch"
                failure_category = "fill_only_verification_mismatch"
                failure_detail = f"verified_correctness={verified_correct}/{question_total}"

    except Exception as exc:
        stop_reason = "environment_error"
        failure_category = "environment_error"
        failure_detail = str(exc)
        success = False
        submit_success = False
    finally:
        if engine is not None:
            engine.close()
        elif mcp is not None:
            mcp.close()
        trace_summary = trace.summary()
        trace.close()

    metrics = rbe._calculate_metrics(question_states)
    duration_s = round(time.perf_counter() - start_time, 3)
    submit_attempt_count = len(submit_attempts)
    successful_submit_attempt_count = sum(1 for attempt in submit_attempts if attempt.get("post_submit_success"))
    failed_submit_attempt_count = submit_attempt_count - successful_submit_attempt_count
    submitted_while_incomplete_count = sum(1 for attempt in submit_attempts if attempt.get("submitted_while_incomplete"))
    first_submit_step = submit_attempts[0].get("step_index") if submit_attempts else None
    pre_first_submit_verified_correctness = submit_attempts[0].get("pre_submit_verified_correctness") if submit_attempts else None
    pre_successful_submit_verified_correctness = next(
        (attempt.get("pre_submit_verified_correctness") for attempt in submit_attempts if attempt.get("post_submit_success")),
        None,
    )
    premature_done_without_submit = stop_reason == "premature_done_without_submit"
    nonempty_transports = {key: value for key, value in tool_call_transport_counts.items() if key != "none" and value > 0}
    if len(nonempty_transports) == 1:
        tool_call_transport = next(iter(nonempty_transports))
    elif len(nonempty_transports) > 1:
        tool_call_transport = "mixed"
    else:
        tool_call_transport = "none"
    done_step: Optional[int] = None
    if premature_done_without_submit:
        try:
            lines = paths["model_io_path"].read_text(encoding="utf-8").splitlines()
            for line in reversed(lines):
                item = json.loads(line)
                if item.get("phase") == "step" and _done_text(item.get("assistant_text") or ""):
                    done_step = item.get("step_index")
                    break
        except Exception:
            done_step = None
    reference_metrics = rbe._resolve_reference_efficiency(
        form_id=args.form_id,
        answer_run_id=answer_run_id,
        model_duration_s=duration_s,
        model_trace_path=paths["trace_path"],
        model_action_count=tool_call_count,
        prefer_model_action_count=True,
    )
    annotations: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": args.experiment_id,
        "trial_id": trial_id,
        "run_label": run_label,
        "run_started_utc": run_started_utc,
        "run_completed_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_id": args.model_id,
        "model_kind": args.model_kind,
        "track": "direct_mcp_tool_use",
        "provider": model_cfg.get("provider"),
        "tool_protocol": "mcp",
        "mcp_server": "playwright",
        "tool_contract": "playwright_mcp_documented_forms_interaction_v1",
        "task_mode": task_mode,
        "native_tool_calls_enabled": native_tool_calls_enabled,
        "tool_call_transport": tool_call_transport,
        "tool_call_transport_counts": tool_call_transport_counts,
        "model_visible_tools": [tool["function"]["name"] for tool in openai_tools] if "openai_tools" in locals() else [],
        "serving_mode": "openai_compat_persistent",
        "server_backend": model_cfg.get("server_backend"),
        "form_id": args.form_id,
        "answer_run_id": answer_run_id,
        "success": bool(success),
        "submit_success": bool(submit_success),
        "stop_reason": stop_reason,
        "failure_category": failure_category,
        "failure_detail": failure_detail,
        "tool_call_count": tool_call_count,
        "action_count": reference_metrics.get("trace_action_count"),
        "trace_action_count": reference_metrics.get("trace_action_count"),
        "trace_action_count_source": reference_metrics.get("trace_action_count_source"),
        "tool_error_count": tool_error_count,
        "invalid_tool_call_count": invalid_tool_call_count,
        "last_invalid_tool_payload": last_invalid_tool_payload,
        "submit_attempt_count": submit_attempt_count,
        "successful_submit_attempt_count": successful_submit_attempt_count,
        "failed_submit_attempt_count": failed_submit_attempt_count,
        "submitted_while_incomplete_count": submitted_while_incomplete_count,
        "first_submit_step": first_submit_step,
        "pre_first_submit_verified_correctness": pre_first_submit_verified_correctness,
        "pre_successful_submit_verified_correctness": pre_successful_submit_verified_correctness,
        "premature_done_without_submit": premature_done_without_submit,
        "done_step": done_step,
        "submit_attempts": submit_attempts,
        "accessibility_snapshot_count": accessibility_snapshot_count,
        "inference_roundtrip_s": inference_roundtrip_s,
        "duration_s": duration_s,
        **reference_metrics,
        "trace": trace_summary,
        "artifacts": rbe._artifact_payload(paths),
        "questions": question_states,
    }
    if terminal_screenshot_path:
        if success:
            annotations["artifacts"]["final_screenshot_path"] = terminal_screenshot_path
        else:
            annotations["artifacts"]["error_screenshot_path"] = terminal_screenshot_path
    annotations.update(metrics)

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment_id": args.experiment_id,
        "trial_id": trial_id,
        "run_label": run_label,
        "run_started_utc": run_started_utc,
        "run_completed_utc": annotations["run_completed_utc"],
        "model_id": args.model_id,
        "model_kind": args.model_kind,
        "track": "direct_mcp_tool_use",
        "provider": model_cfg.get("provider"),
        "tool_protocol": "mcp",
        "mcp_server": "playwright",
        "tool_contract": "playwright_mcp_documented_forms_interaction_v1",
        "task_mode": task_mode,
        "native_tool_calls_enabled": native_tool_calls_enabled,
        "tool_call_transport": tool_call_transport,
        "tool_call_transport_counts": tool_call_transport_counts,
        "server_backend": model_cfg.get("server_backend"),
        "form_id": args.form_id,
        "answer_run_id": answer_run_id,
        "success": bool(success),
        "submit_success": bool(submit_success),
        "stop_reason": stop_reason,
        "failure_category": failure_category,
        "failure_detail": failure_detail,
        "question_total": annotations["question_total"],
        "question_correctness": annotations["question_correctness"],
        "attempted_count": annotations["attempted_count"],
        "attempted_correctness": annotations["attempted_correctness"],
        "verified_count": annotations["verified_count"],
        "verified_correctness": annotations["verified_correctness"],
        "action_count": annotations.get("action_count"),
        "tool_call_count": tool_call_count,
        "trace_action_count": annotations.get("trace_action_count"),
        "trace_action_count_source": annotations.get("trace_action_count_source"),
        "tool_error_count": tool_error_count,
        "invalid_tool_call_count": invalid_tool_call_count,
        "last_invalid_tool_payload": last_invalid_tool_payload,
        "submit_attempt_count": submit_attempt_count,
        "successful_submit_attempt_count": successful_submit_attempt_count,
        "failed_submit_attempt_count": failed_submit_attempt_count,
        "submitted_while_incomplete_count": submitted_while_incomplete_count,
        "first_submit_step": first_submit_step,
        "pre_first_submit_verified_correctness": pre_first_submit_verified_correctness,
        "pre_successful_submit_verified_correctness": pre_successful_submit_verified_correctness,
        "premature_done_without_submit": premature_done_without_submit,
        "done_step": done_step,
        "accessibility_snapshot_count": accessibility_snapshot_count,
        "inference_roundtrip_s": inference_roundtrip_s,
        "duration_s": duration_s,
        "reference_available": annotations.get("reference_available"),
        "reference_run_path": annotations.get("reference_run_path"),
        "reference_trace_path": annotations.get("reference_trace_path"),
        "reference_video_path": annotations.get("reference_video_path"),
        "reference_action_count": annotations.get("reference_action_count"),
        "reference_duration_s": annotations.get("reference_duration_s"),
        "action_overhead_ratio": annotations.get("action_overhead_ratio"),
        "time_overhead_ratio": annotations.get("time_overhead_ratio"),
        "action_count_delta": annotations.get("action_count_delta"),
        "duration_delta_s": annotations.get("duration_delta_s"),
        "artifacts": annotations["artifacts"],
    }

    rbe._write_json(paths["annotations_path"], annotations)
    rbe._write_json(paths["summary_path"], summary)
    _log_progress(
        "trial_complete "
        f"model_id={args.model_id} form_id={args.form_id} run_index={args.run_index} "
        f"success={bool(success)} submit_success={bool(submit_success)} stop_reason={stop_reason} "
        f"tool_calls={tool_call_count} invalid_tool_calls={invalid_tool_call_count} duration_s={duration_s}"
    )
    manifest_entry = {
        "experiment_id": args.experiment_id,
        "trial_id": trial_id,
        "run_label": run_label,
        "run_started_utc": run_started_utc,
        "run_completed_utc": annotations["run_completed_utc"],
        "model_id": args.model_id,
        "model_kind": args.model_kind,
        "track": "direct_mcp_tool_use",
        "provider": model_cfg.get("provider"),
        "tool_protocol": "mcp",
        "mcp_server": "playwright",
        "tool_contract": "playwright_mcp_documented_forms_interaction_v1",
        "task_mode": task_mode,
        "native_tool_calls_enabled": native_tool_calls_enabled,
        "tool_call_transport": tool_call_transport,
        "tool_call_transport_counts": tool_call_transport_counts,
        "server_backend": model_cfg.get("server_backend"),
        "form_id": args.form_id,
        "answer_run_id": answer_run_id,
        "success": bool(success),
        "submit_success": bool(submit_success),
        "stop_reason": stop_reason,
        "failure_category": failure_category,
        "failure_detail": failure_detail,
        "tool_call_count": tool_call_count,
        "action_count": annotations.get("action_count"),
        "trace_action_count": annotations.get("trace_action_count"),
        "trace_action_count_source": annotations.get("trace_action_count_source"),
        "tool_error_count": tool_error_count,
        "invalid_tool_call_count": invalid_tool_call_count,
        "submit_attempt_count": submit_attempt_count,
        "successful_submit_attempt_count": successful_submit_attempt_count,
        "failed_submit_attempt_count": failed_submit_attempt_count,
        "submitted_while_incomplete_count": submitted_while_incomplete_count,
        "first_submit_step": first_submit_step,
        "premature_done_without_submit": premature_done_without_submit,
        "accessibility_snapshot_count": accessibility_snapshot_count,
        "inference_roundtrip_s": inference_roundtrip_s,
        "reference_available": annotations.get("reference_available"),
        "reference_action_count": annotations.get("reference_action_count"),
        "reference_duration_s": annotations.get("reference_duration_s"),
        "action_overhead_ratio": annotations.get("action_overhead_ratio"),
        "time_overhead_ratio": annotations.get("time_overhead_ratio"),
        "summary_path": str(paths["summary_path"]),
        "annotations_path": str(paths["annotations_path"]),
        "trace_path": str(paths["trace_path"]),
        "model_io_path": str(paths["model_io_path"]),
        "step_inputs_path": str(paths["step_inputs_path"]),
        "video_path": annotations["artifacts"].get("video_path"),
    }
    rbe._append_jsonl(paths["manifest_path"], manifest_entry)
    print(f"[INFO] wrote direct-mcp summary: {paths['summary_path']}")
    print(f"[INFO] wrote direct-mcp annotations: {paths['annotations_path']}")
    print(f"[INFO] wrote direct-mcp manifest: {paths['manifest_path']}")
    print(f"[INFO] stop_reason: {stop_reason}")
    print(f"[INFO] success: {success}")
    print(f"[INFO] submit_success: {submit_success}")
    print(f"[INFO] verified_correctness: {annotations['verified_correctness']}/{annotations['question_total']}")
    return 0 if not failure_category else 1


def _default_mcp_server_command(args: argparse.Namespace, paths: Dict[str, Path]) -> List[str]:
    return rbe._default_browser_mcp_command(
        rbe.DEFAULT_VIEWPORT_WIDTH,
        rbe.DEFAULT_VIEWPORT_HEIGHT,
        paths["artifact_dir"],
        bool(args.headless),
        int(args.browser_mcp_timeout_ms),
    )


if __name__ == "__main__":
    raise SystemExit(main())
