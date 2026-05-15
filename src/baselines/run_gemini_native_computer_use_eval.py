import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines import run_baseline_eval as rbe  # noqa: E402
from baselines.action_schema import parse_action, validate_low_level_action  # noqa: E402
from baselines.model_registry import get_model_by_id  # noqa: E402
from baselines.prompt_builders import compact_page_text  # noqa: E402
from engine.browser_language import force_english_google_forms_url  # noqa: E402
from engine.runner import iter_run_specs, load_form_spec, resolve_answers_path  # noqa: E402
from engine.trace_logger import TraceLogger  # noqa: E402

DEFAULT_ANSWERS_ROOT = "data/answers"
DEFAULT_DATASET_ROOT = "data/model_baselines"
DEFAULT_CONFIG = "configs/baselines/track_baseline_models.json"
DEFAULT_EXPERIMENT_ID = "track_baseline_gemini_v1"
DEFAULT_MAX_STEPS = 48
DEFAULT_TIMEOUT_S = 3600
DEFAULT_API_TIMEOUT_S = 180
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_PROMPT_MODE = "gemini_native_computer_use_minimal_v1"
DEFAULT_INTERACTION_PROTOCOL = "human_ui_v1"
DEFAULT_OBSERVATION_MODE = "vision_coords"
DEFAULT_SCORING_MODE = "soft_quality_v1"
DEFAULT_GEMINI_MODEL = "gemini-2.5-computer-use-preview-10-2025"
DEFAULT_MIN_REQUEST_INTERVAL_S = 8.0
DEFAULT_MAX_INFER_RETRIES = 2
DEFAULT_MAX_RETRY_DELAY_S = 75.0
SCHEMA_VERSION = "baseline_eval.v3"
SUMMARY_SCHEMA_VERSION = "baseline_summary.v3"


def _load_run_answers(answers_path: Path, run_index: int) -> List[Dict[str, Any]]:
    for idx, run_spec in enumerate(iter_run_specs(answers_path), start=1):
        if idx == run_index:
            answers = run_spec.get("answers", [])
            if not isinstance(answers, list):
                raise ValueError(f"Run {run_index} answers must be a list")
            return answers
    raise IndexError(f"Run index out of range: {run_index} for {answers_path}")


def _http_post_json(url: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url=url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=max(1, int(timeout_s))) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        raise RuntimeError(f"gemini_http_error:{exc.code}:{raw}") from exc
    except Exception as exc:
        raise RuntimeError(f"gemini_request_failed:{exc}") from exc

    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"gemini_invalid_json:{exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("gemini_response_not_object")
    return parsed


def _safe_to_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    for meth in ("to_dict", "model_dump", "dict"):
        fn = getattr(value, meth, None)
        if callable(fn):
            try:
                payload = fn()
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
    try:
        raw = json.loads(json.dumps(value, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        raw = {}
    return raw if isinstance(raw, dict) else {}


def _tool_declarations() -> List[Dict[str, Any]]:
    return [
        {
            "functionDeclarations": [
                {
                    "name": "submit",
                    "description": "Submit the current form when all answers are filled.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string", "description": "Optional short rationale for telemetry only."},
                        },
                    },
                },
                {
                    "name": "done",
                    "description": "Stop when task is complete or no further progress is possible.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string", "description": "Optional short rationale for telemetry only."},
                        },
                    },
                },
                {
                    "name": "require_confirmation",
                    "description": "Optional model safety request. Executor auto-allows and logs this event.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string"},
                            "action": {
                                "type": "object",
                                "description": "Optional proposed low-level action payload.",
                            },
                        },
                    },
                },
            ]
        }
    ]


def _build_goal_prompt(
    form_url: str,
    remaining_answers: List[Dict[str, Any]],
    last_result: Dict[str, Any],
    interaction_map: List[Dict[str, Any]],
    page_text: str,
    observation_mode: str,
) -> str:
    validation_feedback = rbe._normalize_validation_feedback(last_result)
    recovery_hint = str(validation_feedback.get("hint") or "").strip()
    page_text_block = ""
    if observation_mode != "vision_coords":
        page_text_block = f"Visible page text excerpt:\n{compact_page_text(page_text, max_chars=3000)}\n\n"
    recovery_block = f"Recovery hint:\n{recovery_hint}\n\n" if recovery_hint else ""
    return (
        "Task: Fill the web form and submit successfully.\n"
        "You are using the Gemini Computer Use tool in a browser environment.\n"
        "Use the Computer Use tool for clicking, typing, scrolling, hovering, and key combinations.\n"
        "The target form is already open, so do not navigate away unless recovery truly requires it.\n"
        "You may make mistakes and recover; continue progressing toward completion.\n"
        "Prefer function calls. Custom functions available to you are submit, done, and require_confirmation.\n"
        "Interaction rules:\n"
        "- Click the intended field before typing unless focus is clearly already on that field.\n"
        "- If typing leaves a field empty or unchanged, click again, clear with a key combination, and retype.\n"
        "- For checkbox or radio choices, click the exact mapped option control and verify the state change.\n"
        "- For split date or time fields, click the intended subfield and use key combinations when needed.\n"
        "- Do not repeat the same action on the same target more than twice without changing strategy.\n"
        "- Use submit only when Remaining answers is empty.\n"
        "- Use done only when no more safe progress is possible.\n\n"
        f"Current URL:\n{form_url}\n\n"
        f"Remaining answers:\n{json.dumps(remaining_answers, indent=2, ensure_ascii=True)}\n\n"
        f"Interaction map (coords normalized [0,999]):\n{json.dumps(interaction_map or [], indent=2, ensure_ascii=True)}\n\n"
        f"Validation feedback:\n{json.dumps(validation_feedback or {}, indent=2, ensure_ascii=True)}\n\n"
        f"Last action result:\n{json.dumps(last_result or {}, indent=2, ensure_ascii=True)}\n\n"
        f"{recovery_block}"
        f"{page_text_block}"
    )


def _image_part_for_path(path_value: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return None
    try:
        raw = path.read_bytes()
    except Exception:
        return None
    if not raw:
        return None
    encoded = base64.b64encode(raw).decode("ascii")
    return {
        "inlineData": {
            "mimeType": "image/png",
            "data": encoded,
        }
    }


def _extract_function_calls(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    direct_calls = payload.get("function_calls")
    if isinstance(direct_calls, list):
        for item in direct_calls:
            if isinstance(item, dict):
                calls.append(item)

    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            content = cand.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                fn_call = part.get("functionCall") or part.get("function_call")
                if isinstance(fn_call, dict):
                    calls.append(fn_call)
    return calls


def _extract_text(payload: Dict[str, Any]) -> str:
    texts: List[str] = []
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            content = cand.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
    if texts:
        return "\n".join(texts).strip()

    text = payload.get("text")
    if isinstance(text, str):
        return text.strip()
    return ""


def _normalize_int(value: Any, *, field: str) -> int:
    try:
        as_int = int(value)
    except Exception as exc:
        raise ValueError(f"invalid_{field}: {exc}") from exc
    return as_int


def _normalize_action_from_function_call(call: Dict[str, Any]) -> Dict[str, Any]:
    name = str(call.get("name") or "").strip()
    args = call.get("args")
    if not isinstance(args, dict):
        args = {}

    action: Dict[str, Any] = {"action": name}
    reason = args.get("reason")
    if isinstance(reason, str) and reason.strip():
        action["reason"] = reason.strip()

    if name in {"move_mouse", "click_mouse"}:
        action["target"] = {
            "x": _normalize_int(args.get("x"), field="x"),
            "y": _normalize_int(args.get("y"), field="y"),
        }
    elif name == "hover_at":
        action["action"] = "move_mouse"
        action["target"] = {
            "x": _normalize_int(args.get("x"), field="x"),
            "y": _normalize_int(args.get("y"), field="y"),
        }
    elif name == "click_at":
        action["action"] = "click_mouse"
        action["target"] = {
            "x": _normalize_int(args.get("x"), field="x"),
            "y": _normalize_int(args.get("y"), field="y"),
        }
    elif name == "type_text_at":
        action["action"] = "type_text"
        action["target"] = {
            "x": _normalize_int(args.get("x"), field="x"),
            "y": _normalize_int(args.get("y"), field="y"),
        }
        value = args.get("text") if args.get("text") is not None else args.get("value")
        action["value"] = str(value or "")
        action["press_enter"] = bool(args.get("press_enter"))
        action["clear_before_typing"] = bool(args.get("clear_before_typing", True))
    elif name == "scroll_document":
        action["action"] = "scroll"
        delta = args.get("delta")
        if delta is None:
            delta = args.get("delta_y")
        if delta is None:
            delta = args.get("scroll_delta_y")
        if delta is None:
            direction = str(args.get("direction") or "").strip().lower()
            delta = -700 if direction == "up" else 700
        action["delta"] = _normalize_int(delta, field="delta")
    elif name == "wait_5_seconds":
        action["action"] = "wait"
        action["delta"] = 5000
    elif name == "key_combination":
        action["action"] = "press_key"
        keys = args.get("keys")
        if isinstance(keys, list):
            action["value"] = "+".join(str(item).strip() for item in keys if str(item).strip())
        elif isinstance(keys, str):
            action["value"] = keys.strip()
        else:
            value = args.get("value") if args.get("value") is not None else args.get("key")
            action["value"] = str(value or "")
    elif name == "type_text":
        value = args.get("value") if args.get("value") is not None else args.get("text")
        action["value"] = str(value or "")
    elif name == "press_key":
        value = args.get("value") if args.get("value") is not None else args.get("key")
        action["value"] = str(value or "")
    elif name in {"scroll", "wait"}:
        action["delta"] = _normalize_int(args.get("delta"), field="delta")
    elif name in {"submit", "done", "require_confirmation"}:
        pass
    else:
        raise ValueError(f"unsupported_function_call:{name}")
    return action


def _extract_retry_delay_s(error_text: str) -> Optional[float]:
    text = str(error_text or "")
    match = re.search(r'"retryDelay"\s*:\s*"(?P<secs>\d+)s"', text)
    if match:
        try:
            return float(match.group("secs"))
        except Exception:
            return None
    match = re.search(r"retry in\s+(?P<secs>\d+(?:\.\d+)?)s", text, flags=re.IGNORECASE)
    if match:
        try:
            return float(match.group("secs"))
        except Exception:
            return None
    return None


def _is_quota_error(error_text: str) -> bool:
    text = str(error_text or "").lower()
    return "gemini_http_error:429" in text or "resource_exhausted" in text or "quota exceeded" in text


def _normalize_action_from_text(raw_text: str) -> Dict[str, Any]:
    parsed = parse_action(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError("text_action_not_object")
    action = dict(parsed)

    target = action.get("target") if isinstance(action.get("target"), dict) else {}
    if "x" in action and "x" not in target:
        target["x"] = action.get("x")
    if "y" in action and "y" not in target:
        target["y"] = action.get("y")
    if target:
        action["target"] = target

    if action.get("action") == "type_text" and "value" not in action and isinstance(action.get("text"), str):
        action["value"] = action.get("text")
    if action.get("action") == "press_key" and "value" not in action and isinstance(action.get("key"), str):
        action["value"] = action.get("key")

    return action


def _resolve_action_and_safety(
    function_calls: List[Dict[str, Any]],
    raw_text: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    safety_events: List[Dict[str, Any]] = []
    if function_calls:
        first = function_calls[0]
        action = _normalize_action_from_function_call(first)
        source_meta = {
            "source": "function_call",
            "function_name": str(first.get("name") or ""),
            "function_args": first.get("args") if isinstance(first.get("args"), dict) else {},
        }
        first_args = source_meta.get("function_args") if isinstance(source_meta.get("function_args"), dict) else {}
        safety_decision = first_args.get("safety_decision") if isinstance(first_args, dict) else None
        if isinstance(safety_decision, dict) and str(safety_decision.get("decision") or "").strip().lower() == "require_confirmation":
            safety_events.append(
                {
                    "source": "function_call",
                    "reason": str(safety_decision.get("explanation") or action.get("reason") or "require_confirmation"),
                    "decision": "auto_allow",
                }
            )
        if action.get("action") == "require_confirmation":
            reason = str(action.get("reason") or "require_confirmation")
            args = first_args
            proposed = args.get("action") if isinstance(args, dict) else None
            safety_events.append(
                {
                    "source": "function_call",
                    "reason": reason,
                    "decision": "auto_allow",
                }
            )
            if isinstance(proposed, dict):
                action = dict(proposed)
            else:
                action = {"action": "wait", "delta": 500, "reason": "auto_allow_confirmation_without_action"}
        return action, safety_events, source_meta

    action = _normalize_action_from_text(raw_text)
    source_meta = {
        "source": "text_json",
    }
    if bool(action.pop("require_confirmation", False)):
        reason = str(action.get("reason") or "require_confirmation")
        safety_events.append(
            {
                "source": "text_json",
                "reason": reason,
                "decision": "auto_allow",
            }
        )
    proposed_action = action.pop("proposed_action", None)
    if isinstance(proposed_action, dict):
        action = proposed_action
    return action, safety_events, source_meta


class GeminiNativeAdapter:
    def __init__(self, model_cfg: Dict[str, Any], api_timeout_s: int) -> None:
        self.model_cfg = dict(model_cfg)
        self.api_timeout_s = max(1, int(api_timeout_s))
        self.api_key = str(os.environ.get("GEMINI_API_KEY") or "").strip()
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY missing")
        self.model = str(os.environ.get("GEMINI_MODEL") or model_cfg.get("gemini_model") or DEFAULT_GEMINI_MODEL).strip()
        if not self.model:
            raise RuntimeError("gemini_model missing")
        self.min_request_interval_s = max(
            0.0,
            float(os.environ.get("GEMINI_MIN_REQUEST_INTERVAL_S") or model_cfg.get("gemini_min_request_interval_s") or DEFAULT_MIN_REQUEST_INTERVAL_S),
        )
        self.max_infer_retries = max(
            0,
            int(os.environ.get("GEMINI_MAX_INFER_RETRIES") or model_cfg.get("gemini_max_infer_retries") or DEFAULT_MAX_INFER_RETRIES),
        )
        self.max_retry_delay_s = max(
            1.0,
            float(os.environ.get("GEMINI_MAX_RETRY_DELAY_S") or model_cfg.get("gemini_max_retry_delay_s") or DEFAULT_MAX_RETRY_DELAY_S),
        )
        self._last_request_started_at = 0.0

        try:
            from google import genai  # noqa: F401

            self._genai = genai
            self._client = genai.Client(api_key=self.api_key)
            self._sdk_ready = True
            self._sdk_error: Optional[str] = None
        except Exception as exc:
            self._genai = None
            self._client = None
            self._sdk_ready = False
            self._sdk_error = str(exc)

    def _supported_exclusions(self) -> List[str]:
        return [
            "open_web_browser",
            "search",
            "navigate",
            "go_back",
            "go_forward",
            "drag_and_drop",
            "scroll_at",
        ]

    def _pace_requests(self) -> None:
        if self.min_request_interval_s <= 0:
            return
        now = time.perf_counter()
        if self._last_request_started_at > 0:
            wait_s = self.min_request_interval_s - (now - self._last_request_started_at)
            if wait_s > 0:
                time.sleep(wait_s)
        self._last_request_started_at = time.perf_counter()

    def _build_sdk_config(self, max_new_tokens: int) -> Any:
        if not self._sdk_ready or self._genai is None:
            raise RuntimeError("google_genai_sdk_unavailable")
        types_mod = self._genai.types
        return types_mod.GenerateContentConfig(
            temperature=0,
            max_output_tokens=int(max_new_tokens),
            tools=[
                types_mod.Tool(
                    computer_use=types_mod.ComputerUse(
                        environment=types_mod.Environment.ENVIRONMENT_BROWSER,
                        excluded_predefined_functions=self._supported_exclusions(),
                    )
                ),
                types_mod.Tool(function_declarations=_tool_declarations()[0]["functionDeclarations"]),
            ],
        )

    def _build_rest_payload(self, contents: List[Dict[str, Any]], max_new_tokens: int) -> Dict[str, Any]:
        return {
            "contents": contents,
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": int(max_new_tokens),
            },
            "tools": [
                {
                    "computerUse": {
                        "environment": "ENVIRONMENT_BROWSER",
                        "excludedPredefinedFunctions": self._supported_exclusions(),
                    }
                },
                {
                    "functionDeclarations": _tool_declarations()[0]["functionDeclarations"],
                },
            ],
        }

    def _generate_via_sdk(self, contents: List[Dict[str, Any]], max_new_tokens: int) -> Dict[str, Any]:
        if not self._sdk_ready or self._client is None:
            raise RuntimeError(f"google_genai_sdk_unavailable:{self._sdk_error}")

        started = time.perf_counter()
        self._pace_requests()
        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self._build_sdk_config(max_new_tokens),
        )
        payload = _safe_to_dict(response)
        payload.setdefault(
            "_meta",
            {
                "transport": "google_genai_sdk",
                "duration_s": round(time.perf_counter() - started, 3),
            },
        )
        return payload

    def _generate_via_rest(self, contents: List[Dict[str, Any]], max_new_tokens: int) -> Dict[str, Any]:
        started = time.perf_counter()
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={urllib.parse.quote(self.api_key)}"
        )
        payload = self._build_rest_payload(contents, max_new_tokens)
        self._pace_requests()
        response = _http_post_json(url=url, payload=payload, timeout_s=self.api_timeout_s)
        response.setdefault(
            "_meta",
            {
                "transport": "gemini_rest",
                "duration_s": round(time.perf_counter() - started, 3),
            },
        )
        return response

    def _generate_via_rest_json_fallback(self, contents: List[Dict[str, Any]], max_new_tokens: int) -> Dict[str, Any]:
        started = time.perf_counter()
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={urllib.parse.quote(self.api_key)}"
        )
        latest_user = None
        for item in reversed(contents):
            if isinstance(item, dict) and str(item.get("role") or "") == "user":
                latest_user = item
                break
        if latest_user is None:
            latest_user = {"role": "user", "parts": [{"text": "Return one JSON action object."}]}
        payload = {
            "contents": [latest_user],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": int(max_new_tokens),
                "responseMimeType": "application/json",
            },
        }
        response = _http_post_json(url=url, payload=payload, timeout_s=self.api_timeout_s)
        response.setdefault(
            "_meta",
            {
                "transport": "gemini_rest_json_fallback",
                "duration_s": round(time.perf_counter() - started, 3),
            },
        )
        return response

    def infer(self, contents: List[Dict[str, Any]], max_new_tokens: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        started = time.perf_counter()
        errors: List[str] = []

        retry_count = 0
        while True:
            try:
                payload = self._generate_via_sdk(contents, max_new_tokens=max_new_tokens)
                meta = {
                    "provider": "gemini_native",
                    "provider_model": self.model,
                    "duration_s": round(time.perf_counter() - started, 3),
                    "transport": str((payload.get("_meta") or {}).get("transport") or "google_genai_sdk"),
                    "sdk_fallback": False,
                    "retry_count": retry_count,
                }
                return payload, meta
            except Exception as exc:
                err_text = str(exc)
                errors.append(f"sdk:{err_text}")
                if retry_count < self.max_infer_retries and _is_quota_error(err_text):
                    retry_delay_s = _extract_retry_delay_s(err_text) or min(10.0 * (retry_count + 1), self.max_retry_delay_s)
                    time.sleep(max(1.0, min(retry_delay_s, self.max_retry_delay_s)))
                    retry_count += 1
                    continue
                if _is_quota_error(err_text):
                    raise RuntimeError(f"gemini_quota_exhausted:{err_text}") from exc
                break
        try:
            payload = self._generate_via_rest(contents, max_new_tokens=max_new_tokens)
            meta = {
                "provider": "gemini_native",
                "provider_model": self.model,
                "duration_s": round(time.perf_counter() - started, 3),
                "transport": str((payload.get("_meta") or {}).get("transport") or "gemini_rest"),
                "sdk_fallback": True,
                "sdk_errors": errors,
            }
            return payload, meta
        except Exception as exc:
            errors.append(f"rest:{exc}")
            if _is_quota_error(str(exc)):
                raise RuntimeError(f"gemini_quota_exhausted:{exc}") from exc

        payload = self._generate_via_rest_json_fallback(contents, max_new_tokens=max_new_tokens)
        meta = {
            "provider": "gemini_native",
            "provider_model": self.model,
            "duration_s": round(time.perf_counter() - started, 3),
            "transport": str((payload.get("_meta") or {}).get("transport") or "gemini_rest_json_fallback"),
            "sdk_fallback": True,
            "sdk_errors": errors,
        }
        return payload, meta


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run native Gemini computer-use baseline evaluation for one model/form/run.")
    parser.add_argument("--model-id", default="computer_use_gemini_native")
    parser.add_argument("--form-id", required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--answers-root", default=DEFAULT_ANSWERS_ROOT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--trial-id")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--api-timeout-s", type=int, default=DEFAULT_API_TIMEOUT_S)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--execution-backend", choices=["local", "mcp_server"], default="mcp_server")
    parser.add_argument("--browser-mcp-cmd")
    parser.add_argument("--browser-mcp-timeout-ms", type=int, default=120000)
    parser.add_argument("--browser-init-retries", type=int, default=2)
    parser.add_argument("--browser-init-retry-delay-s", type=float, default=1.5)
    parser.add_argument("--invalid-action-budget", type=int, default=0)
    parser.add_argument("--viewport-width", type=int, default=1440)
    parser.add_argument("--viewport-height", type=int, default=900)
    parser.add_argument("--interaction-protocol", choices=["legacy_semantic_v1", "human_ui_v1"], default=DEFAULT_INTERACTION_PROTOCOL)
    parser.add_argument("--observation-mode", choices=["vision_coords", "vision_coords_text"], default=DEFAULT_OBSERVATION_MODE)
    parser.add_argument("--scoring-mode", choices=["soft_quality_v1", "legacy_binary_v1"], default=DEFAULT_SCORING_MODE)
    parser.add_argument("--disable-action-coercion", action="store_true", default=False)
    parser.add_argument("--retention-window", type=int, default=rbe.DEFAULT_RETENTION_WINDOW)
    parser.add_argument("--run-label")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    args.retention_window = max(0, int(args.retention_window))
    run_label = rbe._make_run_label(args.run_label)
    run_started_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    config_path = (ROOT_DIR / args.config).resolve()
    model_cfg = get_model_by_id(config_path, args.model_id)
    if str(model_cfg.get("provider") or "") != "gemini_native":
        raise ValueError(f"run_gemini_native_computer_use_eval expects provider=gemini_native: {args.model_id}")

    adapter = GeminiNativeAdapter(model_cfg=model_cfg, api_timeout_s=args.api_timeout_s)

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

    trace = TraceLogger(
        path=paths["trace_path"],
        start_time=start_time,
        validate_mcp_actions=True,
        strict_mcp_validation=True,
        mcp_client=None,
    )

    annotations: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": args.experiment_id,
        "trial_id": trial_id,
        "run_label": run_label,
        "run_started_utc": run_started_utc,
        "model_id": args.model_id,
        "model_kind": "computer_use_agent",
        "track": "direct_api_tool_use",
        "provider": model_cfg.get("provider"),
        "api_provider": "gemini_native",
        "form_id": args.form_id,
        "answer_run_id": answer_run_id,
        "form_url": form_url,
        "execution_backend": args.execution_backend,
        "prompt_mode": DEFAULT_PROMPT_MODE,
        "interaction_protocol": str(args.interaction_protocol),
        "observation_mode": str(args.observation_mode),
        "scoring_mode": str(args.scoring_mode),
        "success": False,
        "submit_success": False,
        "stop_reason": None,
        "failure_category": None,
        "failure_detail": None,
        "question_total": len(question_states),
        "question_correctness": 0,
        "attempted_count": 0,
        "attempted_correctness": 0,
        "verified_count": 0,
        "verified_correctness": 0,
        "action_count": 0,
        "invalid_actions": 0,
        "duration_s": None,
        "model_driven_execution": False,
        "autonomy_step_rate": 0.0,
        "action_diversity": 0.0,
        "loop_ratio": 0.0,
        "correction_count": 0,
        "composite_score": 0.0,
        "safety_require_confirmation_count": 0,
        "safety_auto_allowed_count": 0,
        "safety_decision_log": [],
        "model": {"provider": model_cfg.get("provider"), "hf_repo": model_cfg.get("hf_repo")},
        "input_contract": {
            "provides_form_spec": False,
            "provides_dom_dump_upfront": False,
            "provides_answers": True,
            "provides_labels": True,
            "provides_widget_types": True,
            "provides_values": True,
        },
        "run_params": {
            "headless": bool(args.headless),
            "timeout_s": args.timeout_s,
            "max_steps": args.max_steps,
            "max_new_tokens": args.max_new_tokens,
            "invalid_action_budget": args.invalid_action_budget,
            "viewport": {"width": args.viewport_width, "height": args.viewport_height},
            "browser_mcp_cmd": args.browser_mcp_cmd,
            "browser_mcp_timeout_ms": args.browser_mcp_timeout_ms,
            "browser_init_retries": int(args.browser_init_retries),
            "browser_init_retry_delay_s": float(args.browser_init_retry_delay_s),
            "api_timeout_s": args.api_timeout_s,
            "disable_action_coercion": bool(args.disable_action_coercion),
            "retention_window": int(args.retention_window),
            "run_label": run_label,
            "gemini_model": adapter.model,
            "interaction_protocol": str(args.interaction_protocol),
            "observation_mode": str(args.observation_mode),
            "scoring_mode": str(args.scoring_mode),
        },
        "artifacts": rbe._artifact_payload(paths),
        "trace": {},
        "environment": {},
        "steps": [],
        "questions": question_states,
        "failure_events": [],
        "soft_violations": [],
    }

    rbe._append_jsonl(
        paths["model_io_path"],
        {
            "phase": "setup",
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "trial_id": trial_id,
            "run_label": run_label,
            "model_id": args.model_id,
            "form_id": args.form_id,
            "answer_run_id": answer_run_id,
            "status": "started",
            "api_provider": "gemini_native",
            "gemini_model": adapter.model,
        },
    )

    execution_session = None
    last_result: Dict[str, Any] = {}
    terminal_screenshot_path: Optional[str] = None
    observation_cache: Dict[int, Dict[str, Any]] = {}
    conversation: List[Dict[str, Any]] = []

    try:
        execution_session = rbe._make_execution_session(args, paths, trace)
        annotations["environment"] = execution_session.start(form_url) or {}
        observation_cache[0] = execution_session.observe(0)

        initial_observation = observation_cache[0] if isinstance(observation_cache[0], dict) else {}
        initial_screenshot = str(initial_observation.get("screenshot_path") or "").strip()
        initial_interaction_map = initial_observation.get("interaction_map") if isinstance(initial_observation.get("interaction_map"), list) else []
        interaction_map_guard_failed = False
        if args.interaction_protocol == "human_ui_v1" and args.execution_backend == "mcp_server" and initial_screenshot and len(initial_interaction_map) == 0:
            try:
                observation_cache[0] = execution_session.observe(0)
                initial_observation = observation_cache[0] if isinstance(observation_cache[0], dict) else {}
                initial_interaction_map = (
                    initial_observation.get("interaction_map")
                    if isinstance(initial_observation.get("interaction_map"), list)
                    else []
                )
            except Exception:
                initial_interaction_map = []
        if args.interaction_protocol == "human_ui_v1" and args.execution_backend == "mcp_server" and initial_screenshot and len(initial_interaction_map) == 0:
            interaction_map_guard_failed = True

        rbe._append_jsonl(
            paths["step_inputs_path"],
            {
                "phase": "startup",
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "step_index": 0,
                "screenshot_path": initial_screenshot or None,
                "interaction_map_count": len(initial_interaction_map),
                "interaction_map_guard_failed": interaction_map_guard_failed,
            },
        )
        if interaction_map_guard_failed:
            raise RuntimeError("browser_mcp_preflight_failed: interaction_map_empty_under_human_ui_v1")

        for step_idx in range(args.max_steps):
            elapsed = time.perf_counter() - start_time
            if elapsed >= args.timeout_s:
                annotations["stop_reason"] = "timeout"
                rbe._set_failure(annotations, "timeout", f"timeout after {args.timeout_s}s", step_idx)
                break

            remaining_answers = rbe._serialize_remaining_answers(question_states)
            observation = observation_cache.pop(step_idx, None)
            if observation is None:
                observation = execution_session.observe(step_idx)
            if not isinstance(observation, dict):
                observation = {}
            page_text = str(observation.get("page_text") or "")
            screenshot_path = str(observation.get("screenshot_path") or "") or None
            raw_interaction_map = observation.get("interaction_map") if isinstance(observation.get("interaction_map"), list) else []
            interaction_map = rbe._enrich_interaction_map(raw_interaction_map, remaining_answers)

            prompt = _build_goal_prompt(
                form_url=form_url,
                remaining_answers=remaining_answers,
                last_result=last_result,
                interaction_map=interaction_map,
                page_text=page_text,
                observation_mode=args.observation_mode,
            )

            step_input_record = {
                "phase": "step_input",
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "step_index": step_idx,
                "form_url": form_url,
                "remaining_answers": remaining_answers,
                "last_result": dict(last_result or {}),
                "page_text_excerpt": compact_page_text(page_text, max_chars=3000),
                "screenshot_path": screenshot_path,
                "interaction_map": interaction_map,
                "interaction_map_count": len(interaction_map),
                "behavior_nudge": None,
                "prompt_hash": rbe._prompt_hash(prompt),
            }
            rbe._append_jsonl(paths["step_inputs_path"], step_input_record)

            user_parts: List[Dict[str, Any]] = [{"text": prompt}]
            image_part = _image_part_for_path(screenshot_path)
            if image_part is not None:
                user_parts.append(image_part)
            step_request_contents = conversation + [{"role": "user", "parts": user_parts}]

            infer_started = time.perf_counter()
            try:
                api_payload, infer_meta = adapter.infer(step_request_contents, max_new_tokens=args.max_new_tokens)
                infer_meta["duration_s"] = round(time.perf_counter() - infer_started, 3)
            except Exception as exc:
                infer_error = f"model_inference_failed: {exc}"
                failure_category = "quota_exhausted" if "gemini_quota_exhausted:" in str(exc) else "model_inference_failed"
                step_record = {
                    "step_index": step_idx,
                    "elapsed_s": round(time.perf_counter() - start_time, 3),
                    "prompt_mode": DEFAULT_PROMPT_MODE,
                    "remaining_answers_before": len(remaining_answers),
                    "page_text_excerpt": page_text[:2000],
                    "screenshot_path": screenshot_path,
                    "raw_model_output": None,
                    "action": None,
                    "warnings": [],
                    "status": "failed",
                    "error": infer_error,
                    "matched_question_id": None,
                    "target_match": None,
                    "execution": None,
                    "verification": None,
                    "progress_made": False,
                    "interaction_map_count": len(interaction_map),
                    "model_inference": {"attempts": [{"attempt": 1, "error": str(exc)}]},
                }
                annotations["steps"].append(step_record)
                rbe._set_failure(annotations, failure_category, str(exc), step_idx)
                annotations["stop_reason"] = failure_category
                break

            raw_text = _extract_text(api_payload)
            function_calls = _extract_function_calls(api_payload)
            payload_excerpt = json.dumps(api_payload, ensure_ascii=True)[:4000]

            step_record: Dict[str, Any] = {
                "step_index": step_idx,
                "elapsed_s": round(time.perf_counter() - start_time, 3),
                "prompt_mode": DEFAULT_PROMPT_MODE,
                "remaining_answers_before": len(remaining_answers),
                "page_text_excerpt": page_text[:2000],
                "screenshot_path": screenshot_path,
                "raw_model_output": raw_text or payload_excerpt,
                "action": None,
                "warnings": [],
                "status": None,
                "error": None,
                "matched_question_id": None,
                "target_match": None,
                "execution": None,
                "verification": None,
                "progress_made": False,
                "interaction_map_count": len(interaction_map),
                "model_inference": {"attempts": [{"attempt": 1, **infer_meta}]},
            }
            io_record: Dict[str, Any] = {
                "phase": "step",
                "step_index": step_idx,
                "elapsed_s": round(time.perf_counter() - start_time, 3),
                "prompt_mode": DEFAULT_PROMPT_MODE,
                "prompt": prompt,
                "remaining_answers": remaining_answers,
                "screenshot_path": screenshot_path,
                "interaction_map": interaction_map,
                "raw_model_output": raw_text,
                "api_response_excerpt": payload_excerpt,
                "parsed_action": None,
                "warnings": [],
                "error": None,
                "matched_question_id": None,
                "target_match": None,
                "execution": None,
                "verification": None,
                "model_inference": step_record["model_inference"],
            }

            try:
                action_candidate, safety_events, source_meta = _resolve_action_and_safety(function_calls, raw_text)
                for event in safety_events:
                    annotations["safety_require_confirmation_count"] = int(annotations.get("safety_require_confirmation_count") or 0) + 1
                    if str(event.get("decision") or "") == "auto_allow":
                        annotations["safety_auto_allowed_count"] = int(annotations.get("safety_auto_allowed_count") or 0) + 1
                    log_entry = {
                        "step_index": step_idx,
                        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                        "source": event.get("source"),
                        "reason": event.get("reason"),
                        "decision": event.get("decision"),
                    }
                    annotations.setdefault("safety_decision_log", []).append(log_entry)

                action, warnings = validate_low_level_action(action_candidate)
                step_record["action"] = action
                step_record["warnings"] = warnings
                step_record["model_output_source"] = source_meta
                io_record["parsed_action"] = action
                io_record["warnings"] = warnings
                io_record["model_output_source"] = source_meta
            except Exception as exc:
                annotations["invalid_actions"] += 1
                message = f"model_output_invalid: {exc}"
                step_record["status"] = "failed"
                step_record["error"] = message
                io_record["error"] = message
                io_record["parse_error"] = str(exc)
                rbe._record_soft_violation(annotations, "model_output_invalid", str(exc), step_idx)
                annotations["steps"].append(step_record)
                rbe._append_jsonl(paths["model_io_path"], io_record)
                last_result = {"status": "failed", "error": message, "remaining_answers": len(remaining_answers)}
                if rbe._invalid_action_budget_exhausted(annotations["invalid_actions"], args.invalid_action_budget):
                    annotations["stop_reason"] = "model_output_invalid"
                    break
                continue

            action_name = str(action.get("action") or "")
            annotations["action_count"] += 1

            if action_name == "submit" and len(remaining_answers) > 0:
                detail = json.dumps(
                    {
                        "remaining_question_ids": [str(item.get("question_id") or "") for item in remaining_answers],
                        "remaining_count": len(remaining_answers),
                    },
                    ensure_ascii=True,
                )
                step_record["status"] = "failed"
                step_record["error"] = "premature_submit_with_remaining_answers"
                io_record["error"] = "premature_submit_with_remaining_answers"
                rbe._record_soft_violation(annotations, "premature_submit", detail, step_idx)
                annotations["steps"].append(step_record)
                rbe._append_jsonl(paths["model_io_path"], io_record)
                last_result = {
                    "status": step_record["status"],
                    "error": step_record["error"],
                    "remaining_answers": len(rbe._serialize_remaining_answers(question_states)),
                }
                continue

            target = action.get("target") if isinstance(action.get("target"), dict) else {}
            matched_idx, question_state, match_debug = rbe._match_question_state(question_states, target)
            step_record["target_match"] = match_debug
            io_record["target_match"] = match_debug
            if question_state is not None and matched_idx is not None:
                step_record["matched_question_id"] = question_state.get("question_id")
                io_record["matched_question_id"] = question_state.get("question_id")

            execution_payload: Dict[str, Any] = {}
            exec_err: Optional[str] = None
            try:
                if action_name == "move_mouse":
                    execution_payload = execution_session.execute_move_mouse(int(target.get("x")), int(target.get("y")), step_idx)
                    step_record["status"] = "moved"
                    step_record["progress_made"] = True
                elif action_name == "click_mouse":
                    execution_payload = execution_session.execute_click_mouse(int(target.get("x")), int(target.get("y")), step_idx)
                    step_record["status"] = "clicked"
                    step_record["progress_made"] = True
                    if question_state is not None:
                        question_state["attempted"] = True
                elif action_name == "type_text":
                    if isinstance(target, dict) and "x" in target and "y" in target:
                        execution_session.execute_click_mouse(int(target.get("x")), int(target.get("y")), step_idx)
                        execution_session.execute_wait(0.2, step_idx)
                    if bool(action.get("clear_before_typing", False)):
                        execution_session.execute_press_key("Control+A", step_idx)
                        execution_session.execute_press_key("Backspace", step_idx)
                    execution_payload = execution_session.execute_type_text(str(action.get("value") or ""), step_idx)
                    if bool(action.get("press_enter")):
                        execution_session.execute_press_key("Enter", step_idx)
                    step_record["status"] = "typed"
                    step_record["progress_made"] = True
                    if question_state is not None:
                        question_state["attempted"] = True
                        question_state["attempted_correct"] = rbe._value_matches(question_state.get("value"), action.get("value"))
                elif action_name == "wait":
                    wait_seconds = max(0.25, float(action.get("delta") or 1000) / 1000.0)
                    execution_session.execute_wait(wait_seconds, step_idx)
                    execution_payload = {"status": "waited", "seconds": wait_seconds}
                    step_record["status"] = "waited"
                elif action_name == "scroll":
                    delta = int(action.get("delta") or 600)
                    execution_session.execute_scroll(delta, step_idx)
                    execution_payload = {"status": "scrolled", "delta": delta}
                    step_record["status"] = "scrolled"
                    step_record["progress_made"] = True
                elif action_name == "press_key":
                    key = str(action.get("value") or "Tab")
                    execution_session.execute_press_key(key, step_idx)
                    execution_payload = {"status": "pressed_key", "key": key}
                    step_record["status"] = "pressed_key"
                    step_record["progress_made"] = True
                elif action_name == "submit":
                    submit_info, submit_err = execution_session.submit()
                    execution_payload = submit_info
                    if submit_err:
                        step_record["status"] = "failed"
                        step_record["error"] = f"submission_failed: {submit_err}"
                        rbe._record_soft_violation(annotations, "submission_failed", submit_err, step_idx)
                    elif submit_info.get("success"):
                        step_record["status"] = "submitted"
                        step_record["progress_made"] = True
                        annotations["success"] = True
                        annotations["submit_success"] = True
                        annotations["stop_reason"] = "submitted"
                    else:
                        step_record["status"] = "failed"
                        step_record["error"] = "submission_failed: not confirmed"
                        rbe._record_soft_violation(
                            annotations,
                            "submission_failed",
                            json.dumps(submit_info, ensure_ascii=True),
                            step_idx,
                        )
                elif action_name == "done":
                    execution_payload = {"status": "done"}
                    step_record["status"] = "done"
                    annotations["stop_reason"] = "done"
            except Exception as exc:
                exec_err = str(exc)

            step_record["execution"] = execution_payload
            io_record["execution"] = execution_payload
            if exec_err:
                step_record["status"] = "failed"
                step_record["error"] = f"widget_interaction_failed: {exec_err}"
                rbe._record_soft_violation(annotations, "widget_interaction_failed", exec_err, step_idx)

            if step_record.get("status") not in {"submitted", "done", "failed"}:
                verification_rows: List[Dict[str, Any]] = []
                for state in question_states:
                    result = execution_session.verify_entry(state, step_idx)
                    state["last_verification"] = result
                    state["actual_value"] = result.get("actual_value")
                    state["verified"] = bool(result.get("verified"))
                    state["verified_correct"] = bool(result.get("verified")) and rbe._value_matches(
                        state.get("value"), result.get("actual_value")
                    )
                    if state["verified"]:
                        state["attempted"] = True
                    if state["verified_correct"]:
                        state["attempted_correct"] = True
                        state["final_status"] = "correct_verified"
                    elif state.get("attempted_correct"):
                        state["final_status"] = "correct_attempted_only"
                    elif state.get("attempted"):
                        state["final_status"] = "failed"
                    verification_rows.append(
                        {
                            "question_id": state.get("question_id"),
                            "verified": state.get("verified"),
                            "verified_correct": state.get("verified_correct"),
                            "detail": result.get("detail"),
                        }
                    )
                io_record["verification"] = verification_rows

                if question_state is not None and action_name in {"type_text", "click_mouse"}:
                    step_verification = question_state.get("last_verification") or {}
                    step_record["verification"] = step_verification
                    step_record["expected_label"] = question_state.get("label")
                    step_record["expected_value"] = question_state.get("value")
                    step_record["executed_value"] = action.get("value")
                    if question_state.get("verified_correct"):
                        step_record["status"] = "filled"
                        step_record["progress_made"] = True
                    elif question_state.get("verified"):
                        step_record["status"] = "filled_unverified"
                        step_record["error"] = "verification_failed"
                        detail = json.dumps(
                            {
                                "expected": question_state.get("value"),
                                "actual": step_verification.get("actual_value"),
                            },
                            ensure_ascii=True,
                        )
                        rbe._record_soft_violation(annotations, "verification_failed", detail, step_idx)
                    elif action_name == "type_text" and step_record.get("status") == "typed":
                        step_record["status"] = "filled_unverified"
                        step_record["error"] = f"verification_failed: {step_verification.get('detail')}"
                        rbe._record_soft_violation(
                            annotations,
                            "verification_failed",
                            str(step_verification.get("detail")),
                            step_idx,
                        )

            next_observation = None
            if step_record.get("status") not in {"failed", "done"} and annotations.get("stop_reason") not in {"submitted", "done"}:
                try:
                    next_observation = execution_session.observe(step_idx + 1)
                    if isinstance(next_observation, dict):
                        observation_cache[step_idx + 1] = next_observation
                except Exception:
                    next_observation = None

            if function_calls:
                first_call = function_calls[0]
                call_name = str(first_call.get("name") or "")
                call_args = first_call.get("args") if isinstance(first_call.get("args"), dict) else {}
                response_payload: Dict[str, Any] = {
                    "status": step_record.get("status"),
                    "error": step_record.get("error"),
                    "execution": execution_payload,
                }
                response_parts: List[Dict[str, Any]] = []
                if isinstance(next_observation, dict):
                    current_url = str(next_observation.get("url") or form_url).strip()
                    if current_url:
                        response_payload["url"] = current_url
                    fr_image = _image_part_for_path(str(next_observation.get("screenshot_path") or ""))
                    if fr_image is not None:
                        response_parts.append({"inlineData": fr_image["inlineData"]})
                conversation.append({"role": "model", "parts": [{"functionCall": {"name": call_name, "args": call_args}}]})
                conversation.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": call_name,
                                    "response": response_payload,
                                    "parts": response_parts,
                                }
                            }
                        ],
                    }
                )
            elif raw_text:
                conversation.append({"role": "model", "parts": [{"text": raw_text[:4000]}]})

            if len(conversation) > 12:
                conversation = conversation[-12:]

            annotations["steps"].append(step_record)
            rbe._append_jsonl(paths["model_io_path"], io_record)
            last_result = {
                "status": step_record["status"],
                "error": step_record["error"],
                "remaining_answers": len(rbe._serialize_remaining_answers(question_states)),
            }

            if annotations["stop_reason"] in {"submitted", "submission_failed", "model_output_invalid", "model_inference_failed", "done"}:
                break

        if annotations["stop_reason"] is None:
            annotations["stop_reason"] = "max_steps_exceeded"
            rbe._set_failure(annotations, "max_steps_exceeded", f"max_steps={args.max_steps}")
    except Exception as exc:
        annotations["stop_reason"] = "environment_error"
        category = rbe._classify_environment_error(str(exc))
        rbe._set_failure(annotations, category, str(exc))
        annotations["run_error"] = str(exc)
        if execution_session is not None:
            try:
                terminal_screenshot_path = execution_session.capture_terminal_screenshot("error.png")
            except Exception:
                terminal_screenshot_path = None
    finally:
        if execution_session is not None and annotations.get("success"):
            try:
                terminal_screenshot_path = execution_session.capture_terminal_screenshot("final.png")
            except Exception:
                terminal_screenshot_path = terminal_screenshot_path
        if execution_session is not None:
            execution_session.close()
        trace_summary = trace.summary()
        trace.close()

    video_path = rbe._finalize_trial_video(paths["artifact_dir"], paths["video_path"])
    if video_path is not None and video_path.exists():
        annotations["artifacts"]["video_path"] = str(video_path)
    if terminal_screenshot_path:
        if annotations.get("success"):
            annotations["artifacts"]["final_screenshot_path"] = terminal_screenshot_path
        else:
            annotations["artifacts"]["error_screenshot_path"] = terminal_screenshot_path

    metrics = rbe._calculate_metrics(question_states)
    annotations.update(metrics)
    if str(args.scoring_mode or "") == "soft_quality_v1":
        soft_metrics = rbe._calculate_soft_quality_metrics(annotations.get("steps", []), annotations, bool(annotations.get("submit_success")))
        annotations.update(soft_metrics)
    annotations["duration_s"] = round(time.perf_counter() - start_time, 3)
    annotations.update(
        rbe._resolve_reference_efficiency(
            form_id=args.form_id,
            answer_run_id=answer_run_id,
            model_duration_s=annotations["duration_s"],
            model_trace_path=paths["trace_path"],
            model_action_count=annotations.get("action_count"),
        )
    )
    annotations["trace"] = trace_summary
    run_completed_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    annotations["run_completed_utc"] = run_completed_utc

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment_id": args.experiment_id,
        "trial_id": trial_id,
        "run_label": run_label,
        "run_started_utc": run_started_utc,
        "run_completed_utc": run_completed_utc,
        "model_id": args.model_id,
        "model_kind": "computer_use_agent",
        "track": annotations.get("track"),
        "provider": annotations.get("provider"),
        "api_provider": annotations.get("api_provider"),
        "form_id": args.form_id,
        "answer_run_id": answer_run_id,
        "execution_backend": args.execution_backend,
        "prompt_mode": DEFAULT_PROMPT_MODE,
        "interaction_protocol": str(args.interaction_protocol),
        "observation_mode": str(args.observation_mode),
        "scoring_mode": str(args.scoring_mode),
        "success": bool(annotations["success"]),
        "submit_success": bool(annotations["submit_success"]),
        "stop_reason": annotations["stop_reason"],
        "failure_category": annotations["failure_category"],
        "failure_detail": annotations["failure_detail"],
        "question_total": annotations["question_total"],
        "question_correctness": annotations["question_correctness"],
        "attempted_count": annotations["attempted_count"],
        "attempted_correctness": annotations["attempted_correctness"],
        "verified_count": annotations["verified_count"],
        "verified_correctness": annotations["verified_correctness"],
        "action_count": annotations["action_count"],
        "trace_action_count": annotations.get("trace_action_count"),
        "trace_action_count_source": annotations.get("trace_action_count_source"),
        "invalid_actions": annotations["invalid_actions"],
        "duration_s": annotations["duration_s"],
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
        "model_driven_execution": annotations.get("model_driven_execution"),
        "autonomy_step_rate": annotations.get("autonomy_step_rate"),
        "action_diversity": annotations.get("action_diversity"),
        "loop_ratio": annotations.get("loop_ratio"),
        "correction_count": annotations.get("correction_count"),
        "composite_score": annotations.get("composite_score"),
        "safety_require_confirmation_count": int(annotations.get("safety_require_confirmation_count") or 0),
        "safety_auto_allowed_count": int(annotations.get("safety_auto_allowed_count") or 0),
        "artifacts": annotations["artifacts"],
    }

    rbe._append_jsonl(
        paths["model_io_path"],
        {
            "phase": "terminal",
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "stop_reason": summary["stop_reason"],
            "failure_category": summary["failure_category"],
            "failure_detail": summary["failure_detail"],
            "success": summary["success"],
            "submit_success": summary["submit_success"],
            "attempted_correctness": summary["attempted_correctness"],
            "verified_correctness": summary["verified_correctness"],
            "trace_action_count": summary.get("trace_action_count"),
            "reference_available": summary.get("reference_available"),
            "reference_action_count": summary.get("reference_action_count"),
            "reference_duration_s": summary.get("reference_duration_s"),
            "action_overhead_ratio": summary.get("action_overhead_ratio"),
            "time_overhead_ratio": summary.get("time_overhead_ratio"),
            "safety_require_confirmation_count": summary["safety_require_confirmation_count"],
            "safety_auto_allowed_count": summary["safety_auto_allowed_count"],
        },
    )

    rbe._write_json(paths["annotations_path"], annotations)
    rbe._write_json(paths["summary_path"], summary)

    manifest_entry = {
        "experiment_id": args.experiment_id,
        "trial_id": trial_id,
        "run_label": run_label,
        "run_started_utc": run_started_utc,
        "run_completed_utc": run_completed_utc,
        "model_id": args.model_id,
        "model_kind": "computer_use_agent",
        "track": annotations.get("track"),
        "provider": annotations.get("provider"),
        "api_provider": annotations.get("api_provider"),
        "form_id": args.form_id,
        "answer_run_id": answer_run_id,
        "success": summary["success"],
        "submit_success": summary["submit_success"],
        "stop_reason": summary["stop_reason"],
        "failure_category": summary["failure_category"],
        "failure_detail": summary["failure_detail"],
        "trace_action_count": summary.get("trace_action_count"),
        "trace_action_count_source": summary.get("trace_action_count_source"),
        "reference_available": summary.get("reference_available"),
        "reference_action_count": summary.get("reference_action_count"),
        "reference_duration_s": summary.get("reference_duration_s"),
        "action_overhead_ratio": summary.get("action_overhead_ratio"),
        "time_overhead_ratio": summary.get("time_overhead_ratio"),
        "summary_path": str(paths["summary_path"]),
        "annotations_path": str(paths["annotations_path"]),
        "trace_path": str(paths["trace_path"]),
        "model_io_path": str(paths["model_io_path"]),
        "step_inputs_path": str(paths["step_inputs_path"]),
        "video_path": annotations["artifacts"]["video_path"],
        "artifact_dir": str(paths["artifact_dir"]),
    }
    rbe._append_jsonl(paths["manifest_path"], manifest_entry)
    rbe._update_experiment_indexes(
        experiment_root=paths["experiment_root"],
        manifest_entry=manifest_entry,
        run_label=run_label,
        retention_window=args.retention_window,
    )

    print(f"[INFO] wrote gemini baseline summary: {paths['summary_path']}")
    print(f"[INFO] wrote gemini baseline annotations: {paths['annotations_path']}")
    print(f"[INFO] wrote gemini baseline manifest: {paths['manifest_path']}")
    print(f"[INFO] provider: gemini_native")
    print(f"[INFO] gemini_model: {adapter.model}")
    print(f"[INFO] stop_reason: {summary['stop_reason']}")
    print(f"[INFO] success: {summary['success']}")
    print(f"[INFO] submit_success: {summary['submit_success']}")
    print(f"[INFO] attempted_correctness: {summary['attempted_correctness']}/{summary['question_total']}")
    print(f"[INFO] verified_correctness: {summary['verified_correctness']}/{summary['question_total']}")
    print(
        "[INFO] safety: "
        f"require_confirmation={summary['safety_require_confirmation_count']} "
        f"auto_allowed={summary['safety_auto_allowed_count']}"
    )
    if summary["success"]:
        return 0
    if summary["failure_category"] == "quota_exhausted":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
