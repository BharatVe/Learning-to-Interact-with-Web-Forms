import argparse
import base64
import json
import os
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
from baselines.action_schema import validate_low_level_action  # noqa: E402
from baselines.model_registry import get_model_by_id  # noqa: E402
from engine.browser_language import force_english_google_forms_url  # noqa: E402
from engine.runner import iter_run_specs, load_form_spec, resolve_answers_path  # noqa: E402
from engine.trace_logger import TraceLogger  # noqa: E402

DEFAULT_ANSWERS_ROOT = "data/answers"
DEFAULT_DATASET_ROOT = "data/model_baselines"
DEFAULT_CONFIG = "configs/baselines/track_baseline_models.json"
DEFAULT_EXPERIMENT_ID = "gemini_35_flash_lowcost_token_pilot_v1"
DEFAULT_MAX_STEPS = 48
DEFAULT_TIMEOUT_S = 3600
DEFAULT_API_TIMEOUT_S = 180
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_MAX_INFER_RETRIES = 2
DEFAULT_RETRY_DELAY_S = 20.0
DEFAULT_RETRY_BACKOFF = 2.0
DEFAULT_RETRY_MAX_DELAY_S = 240.0
DEFAULT_INTERACTION_PROTOCOL = "human_ui_v1"
DEFAULT_OBSERVATION_MODE = "vision_coords"
DEFAULT_SCORING_MODE = "soft_quality_v1"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_KEY_FILE = ".secrets/gemini_api_key"
PROMPT_MODE = "gemini_35_flash_lowcost_balanced_v1"
SCHEMA_VERSION = "baseline_eval.v3"
SUMMARY_SCHEMA_VERSION = "baseline_summary.v3"


def _task_mode(fill_only: bool) -> str:
    return "fill_only_done" if fill_only else "fill_and_submit"


def _load_run_answers(answers_path: Path, run_index: int) -> List[Dict[str, Any]]:
    for idx, run_spec in enumerate(iter_run_specs(answers_path), start=1):
        if idx == run_index:
            answers = run_spec.get("answers", [])
            if not isinstance(answers, list):
                raise ValueError(f"Run {run_index} answers must be a list")
            return answers
    raise IndexError(f"Run index out of range: {run_index} for {answers_path}")


def _read_api_key() -> Tuple[str, str]:
    env_key = str(os.environ.get("GEMINI_API_KEY") or "").strip()
    if env_key:
        return env_key, "env:GEMINI_API_KEY"
    key_file = Path(os.environ.get("GEMINI_API_KEY_FILE") or ROOT_DIR / DEFAULT_KEY_FILE)
    if not key_file.is_absolute():
        key_file = ROOT_DIR / key_file
    try:
        value = key_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"GEMINI_API_KEY missing and key file not found: {key_file}. "
            "Create it with mode 600 or set GEMINI_API_KEY_FILE."
        ) from exc
    if not value:
        raise RuntimeError(f"Gemini key file is empty: {key_file}")
    return value, f"file:{key_file}"


def _http_post_json(url: str, payload: Dict[str, Any], api_key: str, timeout_s: int) -> Dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(url=url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("x-goog-api-key", api_key)
    try:
        with urllib.request.urlopen(request, timeout=max(1, int(timeout_s))) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        raise RuntimeError(f"gemini_interactions_http_error:{exc.code}:{raw}") from exc
    except Exception as exc:
        raise RuntimeError(f"gemini_interactions_request_failed:{exc}") from exc
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"gemini_interactions_invalid_json:{exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("gemini_interactions_response_not_object")
    return parsed


def _is_transient_provider_error(message: str) -> bool:
    text = str(message or "").lower()
    transient_markers = [
        "gemini_interactions_http_error:429",
        "gemini_interactions_http_error:500",
        "gemini_interactions_http_error:503",
        "gemini_interactions_http_error:504",
        "high demand",
        "temporarily overloaded",
        "temporarily unavailable",
        "deadline_exceeded",
        "deadline exceeded",
        "timed out",
        "timeout",
        "read operation timed out",
    ]
    return any(marker in text for marker in transient_markers)


def _provider_failure_category(message: str) -> str:
    return "provider_capacity_error" if _is_transient_provider_error(message) else "model_inference_failed"


def _image_data_for_path(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return None
    raw = path.read_bytes()
    if not raw:
        return None
    return base64.b64encode(raw).decode("ascii")


def _compact_answers(question_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for state in question_states:
        if state.get("verified_correct"):
            continue
        rows.append(
            {
                "id": state.get("question_id"),
                "label": state.get("label"),
                "type": state.get("widget_type"),
                "value": state.get("value"),
            }
        )
    return rows


def _compact_controls(interaction_map: List[Dict[str, Any]], limit: int = 40) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in interaction_map[: max(0, int(limit))]:
        if not isinstance(item, dict):
            continue
        row: Dict[str, Any] = {}
        for src, dst in (("ref", "ref"), ("role", "role"), ("type", "type"), ("text", "text"), ("label", "label")):
            value = item.get(src)
            if isinstance(value, str) and value.strip():
                row[dst] = value.strip()[:80]
        for key in ("x", "y"):
            if key in item:
                row[key] = item.get(key)
        qid = item.get("question_id_guess") or item.get("question_id")
        if isinstance(qid, str) and qid.strip():
            row["qid"] = qid.strip()
        if row:
            rows.append(row)
    return rows


def _last_status(last_result: Dict[str, Any]) -> str:
    if not last_result:
        return ""
    status = str(last_result.get("status") or "").strip()
    error = str(last_result.get("error") or "").strip()
    if error:
        return f"{status or 'failed'}:{error[:80]}"
    return status[:80]


def _build_prompt(
    question_states: List[Dict[str, Any]],
    last_result: Dict[str, Any],
    interaction_map: List[Dict[str, Any]],
    include_controls: bool,
    fill_only: bool,
) -> str:
    answers = json.dumps(_compact_answers(question_states), ensure_ascii=True, separators=(",", ":"))
    task = (
        "Fill this Google Form but do not submit it. Use browser computer-use actions: click, type, scroll, wait. "
        "Match answers exactly; do not reformat dates, times, emails, or option text. When all answers are entered, output done. Never press Submit. "
        "If a click does not change the form after one retry, use a different action or scroll; do not repeat the same click."
        if fill_only
        else "Fill this Google Form and submit it. Use browser computer-use actions: click, type, scroll, wait. Match answers exactly. Submit only after all answers are entered."
    )
    parts = [
        task,
        f"Answers:{answers}",
    ]
    last = _last_status(last_result)
    if last:
        parts.append(f"Last:{last}")
    if include_controls:
        controls = json.dumps(_compact_controls(interaction_map), ensure_ascii=True, separators=(",", ":"))
        parts.append(f"Controls:{controls}")
    return "\n".join(parts)


def _normalize_usage(payload: Dict[str, Any]) -> Dict[str, int]:
    usage = payload.get("usageMetadata") or payload.get("usage_metadata") or payload.get("usage")
    if not isinstance(usage, dict):
        for item in _extract_items(payload):
            candidate = item.get("usageMetadata") or item.get("usage_metadata") or item.get("usage")
            if isinstance(candidate, dict):
                usage = candidate
                break
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = (
        usage.get("input_tokens")
        or usage.get("promptTokenCount")
        or usage.get("prompt_token_count")
        or usage.get("inputTokenCount")
        or 0
    )
    output_tokens = (
        usage.get("output_tokens")
        or usage.get("candidatesTokenCount")
        or usage.get("candidates_token_count")
        or usage.get("outputTokenCount")
        or 0
    )
    total_tokens = usage.get("total_tokens") or usage.get("totalTokenCount") or usage.get("total_token_count") or 0
    try:
        input_i = int(input_tokens or 0)
    except Exception:
        input_i = 0
    try:
        output_i = int(output_tokens or 0)
    except Exception:
        output_i = 0
    try:
        total_i = int(total_tokens or 0)
    except Exception:
        total_i = 0
    if total_i <= 0:
        total_i = input_i + output_i
    return {"input_tokens": input_i, "output_tokens": output_i, "total_tokens": total_i}


def _cost(usage: Dict[str, int], pricing: Dict[str, Any]) -> Dict[str, Any]:
    input_rate = float(pricing.get("input_usd_per_1m_tokens") or 1.5)
    output_rate = float(pricing.get("output_usd_per_1m_tokens") or 9.0)
    eur_per_usd = float(pricing.get("eur_per_usd") or 0.93)
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or 0)
    if input_tokens <= 0 and output_tokens <= 0 and total_tokens > 0:
        input_tokens = total_tokens
    usd = (input_tokens / 1_000_000.0) * input_rate
    usd += (output_tokens / 1_000_000.0) * output_rate
    return {
        "currency": "USD",
        "estimated_usd": round(usd, 6),
        "estimated_eur": round(usd * eur_per_usd, 6),
        "cost_basis": "input_output_tokens" if int(usage.get("input_tokens") or 0) or int(usage.get("output_tokens") or 0) else "total_tokens_as_input_lower_bound",
        "pricing": {
            "input_usd_per_1m_tokens": input_rate,
            "output_usd_per_1m_tokens": output_rate,
            "eur_per_usd": eur_per_usd,
        },
    }


def _extract_items(value: Any) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    if isinstance(value, dict):
        found.append(value)
        for child in value.values():
            found.extend(_extract_items(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_extract_items(child))
    return found


def _done_text(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return text == "done" or text.startswith("done ") or text.startswith("done.") or text.startswith("done:")


def _extract_done_text(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for item in _extract_items(payload):
        for key in ("text", "content", "output_text"):
            value = item.get(key)
            if isinstance(value, str) and _done_text(value):
                return {"source": "text", "name": "done", "args": {"text": value}}
    return None


def _extract_action(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    for item in _extract_items(payload):
        item_type = str(item.get("type") or item.get("kind") or "").strip().lower()
        name = str(item.get("name") or item.get("action") or item.get("function") or "").strip()
        if item_type not in {"computer_call", "function_call", "tool_call"} and not name:
            continue
        args = item.get("arguments") or item.get("args") or item.get("input")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}
        if not name:
            name = str(args.get("action") or args.get("name") or item_type).strip()
        normalized = _normalize_action(name, args)
        return normalized, {"source": item_type or "recursive", "name": name, "args": args}
    done_meta = _extract_done_text(payload)
    if done_meta:
        return {"action": "done"}, done_meta
    raise ValueError("gemini_response_missing_computer_action")


def _coord(args: Dict[str, Any], key: str) -> Optional[int]:
    value = args.get(key)
    if value is None:
        loc = args.get("coordinate") or args.get("coordinates") or args.get("location") or args.get("target")
        if isinstance(loc, dict):
            value = loc.get(key)
        elif isinstance(loc, list) and len(loc) >= 2:
            value = loc[0 if key == "x" else 1]
    if value is None:
        return None
    return max(0, min(999, int(round(float(value)))))


def _normalize_action(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    raw = str(name or "").strip().lower()
    action: Dict[str, Any] = {"action": raw}
    x = _coord(args, "x")
    y = _coord(args, "y")
    if raw in {"click", "click_at", "click_mouse", "double_click"}:
        if x is None or y is None:
            raise ValueError("click_missing_coordinates")
        return {"action": "click_mouse", "target": {"x": x, "y": y}}
    if raw in {"type", "type_text", "type_text_at"}:
        text = args.get("text") if args.get("text") is not None else args.get("value")
        action = {"action": "type_text", "value": str(text or ""), "clear_before_typing": bool(args.get("clear_before_typing", True))}
        if x is not None and y is not None:
            action["target"] = {"x": x, "y": y}
        return action
    if raw in {"scroll", "scroll_document"}:
        delta = args.get("delta") or args.get("delta_y") or args.get("scroll_delta_y")
        if delta is None:
            direction = str(args.get("direction") or "").lower()
            delta = -700 if direction == "up" else 700
        return {"action": "scroll", "delta": int(float(delta))}
    if raw in {"wait", "wait_5_seconds"}:
        seconds = args.get("seconds")
        delta = args.get("delta")
        if delta is None:
            delta = int(float(seconds) * 1000) if seconds is not None else 1000
        return {"action": "wait", "delta": int(float(delta))}
    if raw in {"key_combination", "press_key"}:
        keys = args.get("keys")
        if isinstance(keys, list):
            value = "+".join(str(item).strip() for item in keys if str(item).strip())
        else:
            value = args.get("value") or args.get("key") or keys or ""
        return {"action": "press_key", "value": str(value)}
    if raw in {"submit", "done"}:
        return {"action": raw}
    if raw in {"navigate", "open_web_browser", "go_back", "go_forward"}:
        return {"action": "wait", "delta": 500, "reason": f"ignored_navigation_action:{raw}"}
    raise ValueError(f"unsupported_gemini_action:{raw}")


class GeminiLowCostAdapter:
    def __init__(self, model_cfg: Dict[str, Any], api_timeout_s: int) -> None:
        self.model_cfg = dict(model_cfg)
        self.api_timeout_s = max(1, int(api_timeout_s))
        self.api_key, self.api_key_source = _read_api_key()
        self.model = str(os.environ.get("GEMINI_MODEL") or model_cfg.get("gemini_model") or DEFAULT_GEMINI_MODEL).strip()
        if not self.model:
            raise RuntimeError("gemini_model missing")
        self.endpoint = str(model_cfg.get("interactions_endpoint") or "https://generativelanguage.googleapis.com/v1beta/interactions").rstrip("/")
        self.max_infer_retries = max(
            0,
            int(os.environ.get("GEMINI_MAX_INFER_RETRIES") or model_cfg.get("gemini_max_infer_retries") or DEFAULT_MAX_INFER_RETRIES),
        )
        self.retry_delay_s = max(
            1.0,
            float(os.environ.get("GEMINI_RETRY_DELAY_S") or model_cfg.get("gemini_retry_delay_s") or DEFAULT_RETRY_DELAY_S),
        )
        self.retry_backoff = max(
            1.0,
            float(os.environ.get("GEMINI_RETRY_BACKOFF") or model_cfg.get("gemini_retry_backoff") or DEFAULT_RETRY_BACKOFF),
        )
        self.retry_max_delay_s = max(
            self.retry_delay_s,
            float(os.environ.get("GEMINI_RETRY_MAX_DELAY_S") or model_cfg.get("gemini_retry_max_delay_s") or DEFAULT_RETRY_MAX_DELAY_S),
        )

    def infer(self, prompt: str, screenshot_path: Optional[str], max_new_tokens: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        image_data = _image_data_for_path(screenshot_path)
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_data:
            content.append({"type": "image", "mime_type": "image/png", "data": image_data})
        payload = {
            "model": self.model,
            "input": content,
            "tools": [{"type": "computer_use", "environment": "browser"}],
        }
        started = time.perf_counter()
        url = f"{self.endpoint}?key={urllib.parse.quote(self.api_key)}"
        errors: List[str] = []
        for attempt_idx in range(self.max_infer_retries + 1):
            try:
                response = _http_post_json(url=url, payload=payload, api_key=self.api_key, timeout_s=self.api_timeout_s)
                break
            except RuntimeError as exc:
                err = str(exc)
                errors.append(err)
                transient = _is_transient_provider_error(err)
                if not transient or attempt_idx >= self.max_infer_retries:
                    raise
                delay_s = min(self.retry_delay_s * (self.retry_backoff ** attempt_idx), self.retry_max_delay_s)
                time.sleep(delay_s)
        usage = _normalize_usage(response)
        meta = {
            "provider": "gemini_low_cost",
            "provider_model": self.model,
            "transport": "gemini_interactions_rest",
            "duration_s": round(time.perf_counter() - started, 3),
            "retry_count": len(errors),
            "retry_errors": errors,
            "usage": usage,
            "retry_policy": {
                "max_infer_retries": self.max_infer_retries,
                "retry_delay_s": self.retry_delay_s,
                "retry_backoff": self.retry_backoff,
                "retry_max_delay_s": self.retry_max_delay_s,
            },
        }
        return response, meta


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run low-token Gemini 3.5 Flash computer-use evaluation.")
    parser.add_argument("--model-id", default="computer_use_gemini_35_flash_lowcost")
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
    parser.add_argument("--interaction-protocol", choices=["human_ui_v1"], default=DEFAULT_INTERACTION_PROTOCOL)
    parser.add_argument("--observation-mode", choices=["vision_coords"], default=DEFAULT_OBSERVATION_MODE)
    parser.add_argument("--scoring-mode", choices=["soft_quality_v1", "legacy_binary_v1"], default=DEFAULT_SCORING_MODE)
    parser.add_argument("--include-controls", action="store_true", default=False)
    parser.add_argument("--fill-only", action="store_true", default=False)
    parser.add_argument("--disable-action-coercion", action="store_true", default=False)
    parser.add_argument("--retention-window", type=int, default=rbe.DEFAULT_RETENTION_WINDOW)
    parser.add_argument("--run-label")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    run_label = rbe._make_run_label(args.run_label)
    run_started_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    config_path = (ROOT_DIR / args.config).resolve()
    model_cfg = get_model_by_id(config_path, args.model_id)
    if str(model_cfg.get("provider") or "") != "gemini_low_cost":
        raise ValueError(f"run_gemini_low_cost_eval expects provider=gemini_low_cost: {args.model_id}")
    adapter = GeminiLowCostAdapter(model_cfg=model_cfg, api_timeout_s=args.api_timeout_s)

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
    trace = TraceLogger(path=paths["trace_path"], start_time=start_time, validate_mcp_actions=True, strict_mcp_validation=True, mcp_client=None)

    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    annotations: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": args.experiment_id,
        "trial_id": trial_id,
        "run_label": run_label,
        "run_started_utc": run_started_utc,
        "model_id": args.model_id,
        "model_kind": "computer_use_agent",
        "track": model_cfg.get("track"),
        "provider": model_cfg.get("provider"),
        "api_provider": "gemini_low_cost",
        "form_id": args.form_id,
        "answer_run_id": answer_run_id,
        "form_url": form_url,
        "execution_backend": args.execution_backend,
        "prompt_mode": PROMPT_MODE,
        "task_mode": _task_mode(bool(args.fill_only)),
        "interaction_protocol": args.interaction_protocol,
        "observation_mode": args.observation_mode,
        "scoring_mode": args.scoring_mode,
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
        "model": {"provider": model_cfg.get("provider"), "gemini_model": adapter.model},
        "run_params": {
            "headless": bool(args.headless),
            "timeout_s": args.timeout_s,
            "max_steps": args.max_steps,
            "max_new_tokens": args.max_new_tokens,
            "browser_mcp_timeout_ms": args.browser_mcp_timeout_ms,
            "include_controls": bool(args.include_controls),
            "fill_only": bool(args.fill_only),
            "gemini_model": adapter.model,
            "api_key_source": adapter.api_key_source.split(":", 1)[0],
        },
        "artifacts": rbe._artifact_payload(paths),
        "trace": {},
        "environment": {},
        "steps": [],
        "questions": question_states,
        "failure_events": [],
        "soft_violations": [],
        "token_usage": dict(usage_total),
    }

    execution_session = None
    last_result: Dict[str, Any] = {}
    terminal_screenshot_path: Optional[str] = None
    observation_cache: Dict[int, Dict[str, Any]] = {}
    try:
        execution_session = rbe._make_execution_session(args, paths, trace)
        annotations["environment"] = execution_session.start(form_url) or {}
        observation_cache[0] = execution_session.observe(0)
        for step_idx in range(args.max_steps):
            if time.perf_counter() - start_time >= args.timeout_s:
                annotations["stop_reason"] = "timeout"
                rbe._set_failure(annotations, "timeout", f"timeout after {args.timeout_s}s", step_idx)
                break

            remaining_answers = _compact_answers(question_states)
            if not remaining_answers:
                if args.fill_only:
                    annotations["submit_success"] = False
                    annotations["success"] = True
                    annotations["stop_reason"] = "filled_without_submit"
                    break
                submit_info, submit_err = execution_session.submit()
                annotations["submit_success"] = bool(submit_info.get("success")) and not submit_err
                annotations["success"] = bool(annotations["submit_success"])
                annotations["stop_reason"] = "submitted" if annotations["success"] else "submission_failed"
                if submit_err:
                    rbe._set_failure(annotations, "submission_failed", submit_err, step_idx)
                break

            observation = observation_cache.pop(step_idx, None) or execution_session.observe(step_idx)
            if not isinstance(observation, dict):
                observation = {}
            screenshot_path = str(observation.get("screenshot_path") or "") or None
            raw_interaction_map = observation.get("interaction_map") if isinstance(observation.get("interaction_map"), list) else []
            interaction_map = rbe._enrich_interaction_map(raw_interaction_map, rbe._serialize_remaining_answers(question_states))
            prompt = _build_prompt(question_states, last_result, interaction_map, bool(args.include_controls), bool(args.fill_only))
            rbe._append_jsonl(
                paths["step_inputs_path"],
                {
                    "phase": "step_input",
                    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                    "step_index": step_idx,
                    "remaining_answer_count": len(remaining_answers),
                    "screenshot_path": screenshot_path,
                    "include_controls": bool(args.include_controls),
                    "interaction_map_count": len(interaction_map),
                    "prompt_hash": rbe._prompt_hash(prompt),
                    "prompt_char_count": len(prompt),
                },
            )

            try:
                api_payload, infer_meta = adapter.infer(prompt, screenshot_path, max_new_tokens=args.max_new_tokens)
                usage = infer_meta.get("usage") if isinstance(infer_meta.get("usage"), dict) else {}
                for key in usage_total:
                    usage_total[key] += int(usage.get(key) or 0)
                action_candidate, source_meta = _extract_action(api_payload)
                action, warnings = validate_low_level_action(action_candidate)
            except Exception as exc:
                annotations["invalid_actions"] += 1
                failure_category = _provider_failure_category(str(exc))
                step_record = {
                    "step_index": step_idx,
                    "status": "failed",
                    "error": f"model_inference_or_action_failed:{exc}",
                    "prompt_mode": PROMPT_MODE,
                    "remaining_answers_before": len(remaining_answers),
                    "screenshot_path": screenshot_path,
                    "model_inference": {"attempts": [{"attempt": 1, "error": str(exc)}]},
                }
                annotations["steps"].append(step_record)
                rbe._append_jsonl(paths["model_io_path"], {"phase": "step", "step_index": step_idx, "prompt": prompt, "error": str(exc)})
                rbe._set_failure(annotations, failure_category, str(exc), step_idx)
                annotations["stop_reason"] = failure_category
                break

            action_name = str(action.get("action") or "")
            annotations["action_count"] += 1
            step_record = {
                "step_index": step_idx,
                "elapsed_s": round(time.perf_counter() - start_time, 3),
                "prompt_mode": PROMPT_MODE,
                "remaining_answers_before": len(remaining_answers),
                "screenshot_path": screenshot_path,
                "action": action,
                "warnings": warnings,
                "status": None,
                "error": None,
                "execution": None,
                "verification": None,
                "progress_made": False,
                "model_output_source": source_meta,
                "model_inference": {"attempts": [{"attempt": 1, **infer_meta}]},
            }
            io_record = {
                "phase": "step",
                "step_index": step_idx,
                "prompt_mode": PROMPT_MODE,
                "prompt": prompt,
                "screenshot_path": screenshot_path,
                "parsed_action": action,
                "model_output_source": source_meta,
                "model_inference": step_record["model_inference"],
            }

            exec_err: Optional[str] = None
            execution_payload: Dict[str, Any] = {}
            try:
                target = action.get("target") if isinstance(action.get("target"), dict) else {}
                if action_name == "click_mouse":
                    execution_payload = execution_session.execute_click_mouse(int(target.get("x")), int(target.get("y")), step_idx)
                    step_record["status"] = "clicked"
                    step_record["progress_made"] = True
                elif action_name == "type_text":
                    if "x" in target and "y" in target:
                        execution_session.execute_click_mouse(int(target.get("x")), int(target.get("y")), step_idx)
                        execution_session.execute_wait(0.2, step_idx)
                    if bool(action.get("clear_before_typing", False)):
                        execution_session.execute_press_key("Control+A", step_idx)
                        execution_session.execute_press_key("Backspace", step_idx)
                    execution_payload = execution_session.execute_type_text(str(action.get("value") or ""), step_idx)
                    step_record["status"] = "typed"
                    step_record["progress_made"] = True
                elif action_name == "scroll":
                    delta = int(action.get("delta") or 700)
                    execution_session.execute_scroll(delta, step_idx)
                    execution_payload = {"status": "scrolled", "delta": delta}
                    step_record["status"] = "scrolled"
                    step_record["progress_made"] = True
                elif action_name == "wait":
                    wait_seconds = max(0.25, float(action.get("delta") or 1000) / 1000.0)
                    execution_session.execute_wait(wait_seconds, step_idx)
                    execution_payload = {"status": "waited", "seconds": wait_seconds}
                    step_record["status"] = "waited"
                elif action_name == "press_key":
                    key = str(action.get("value") or "Tab")
                    execution_session.execute_press_key(key, step_idx)
                    execution_payload = {"status": "pressed_key", "key": key}
                    step_record["status"] = "pressed_key"
                    step_record["progress_made"] = True
                elif action_name == "submit":
                    if args.fill_only:
                        execution_payload = {"status": "blocked_by_harness", "reason": "fill_only_no_submit"}
                        step_record["status"] = "blocked_submit_fill_only"
                        step_record["error"] = "submit_disabled_in_fill_only_done"
                        rbe._record_soft_violation(annotations, "submit_disabled_in_fill_only_done", "model requested submit", step_idx)
                    else:
                        submit_info, submit_err = execution_session.submit()
                        execution_payload = submit_info
                        annotations["submit_success"] = bool(submit_info.get("success")) and not submit_err
                        annotations["success"] = bool(annotations["submit_success"])
                        annotations["stop_reason"] = "submitted" if annotations["success"] else "submission_failed"
                        step_record["status"] = annotations["stop_reason"]
                        if submit_err:
                            step_record["error"] = submit_err
                elif action_name == "done":
                    execution_payload = {"status": "done"}
                    annotations["stop_reason"] = "done"
                    step_record["status"] = "done"
                else:
                    raise RuntimeError(f"unsupported_action:{action_name}")
            except Exception as exc:
                exec_err = str(exc)
            if exec_err:
                step_record["status"] = "failed"
                step_record["error"] = f"widget_interaction_failed:{exec_err}"
                rbe._record_soft_violation(annotations, "widget_interaction_failed", exec_err, step_idx)
            step_record["execution"] = execution_payload
            io_record["execution"] = execution_payload

            verification_rows: List[Dict[str, Any]] = []
            if step_record.get("status") not in {"submitted", "failed"}:
                for state in question_states:
                    result = execution_session.verify_entry(state, step_idx)
                    previous_correct = bool(state.get("verified_correct"))
                    current_verified = bool(result.get("verified"))
                    current_correct = current_verified and rbe._value_matches(state.get("value"), result.get("actual_value"))
                    preserve_previous = bool(args.fill_only) and previous_correct and not current_verified
                    state["last_verification"] = result
                    if current_verified or not preserve_previous:
                        state["actual_value"] = result.get("actual_value")
                    state["verified"] = current_verified or preserve_previous
                    state["verified_correct"] = current_correct or preserve_previous
                    if state["verified"]:
                        state["attempted"] = True
                    if state["verified_correct"]:
                        state["attempted_correct"] = True
                        state["final_status"] = "correct_verified"
                    verification_rows.append({"question_id": state.get("question_id"), "verified": state.get("verified"), "verified_correct": state.get("verified_correct")})
                if args.fill_only:
                    remaining_after_step = _compact_answers(question_states)
                    if not remaining_after_step:
                        annotations["submit_success"] = False
                        annotations["success"] = True
                        annotations["stop_reason"] = "filled_without_submit" if action_name != "done" else "done"
                    elif action_name == "done":
                        annotations["submit_success"] = False
                        annotations["success"] = False
                        annotations["stop_reason"] = "done_incomplete_fill_only"
                        rbe._set_failure(
                            annotations,
                            "done_incomplete_fill_only",
                            f"remaining_answers={len(remaining_after_step)}",
                            step_idx,
                        )
            step_record["verification"] = verification_rows
            io_record["verification"] = verification_rows
            annotations["steps"].append(step_record)
            rbe._append_jsonl(paths["model_io_path"], io_record)
            last_result = {"status": step_record.get("status"), "error": step_record.get("error")}
            if annotations.get("stop_reason"):
                break
            try:
                observation_cache[step_idx + 1] = execution_session.observe(step_idx + 1)
            except Exception:
                pass

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
                pass
        if execution_session is not None:
            execution_session.close()
        trace_summary = trace.summary()
        trace.close()

    video_path = rbe._finalize_trial_video(paths["artifact_dir"], paths["video_path"])
    if video_path is not None and video_path.exists():
        annotations["artifacts"]["video_path"] = str(video_path)
    if terminal_screenshot_path:
        annotations["artifacts"]["final_screenshot_path" if annotations.get("success") else "error_screenshot_path"] = terminal_screenshot_path

    metrics = rbe._calculate_metrics(question_states)
    annotations.update(metrics)
    if args.scoring_mode == "soft_quality_v1":
        annotations.update(rbe._calculate_soft_quality_metrics(annotations.get("steps", []), annotations, bool(annotations.get("submit_success"))))
    annotations["duration_s"] = round(time.perf_counter() - start_time, 3)
    annotations["token_usage"] = dict(usage_total)
    annotations["cost_estimate"] = _cost(usage_total, model_cfg.get("pricing") if isinstance(model_cfg.get("pricing"), dict) else {})
    annotations["projected_30_run_cost"] = {
        "estimated_usd": round(float(annotations["cost_estimate"]["estimated_usd"]) * 30, 6),
        "estimated_eur": round(float(annotations["cost_estimate"]["estimated_eur"]) * 30, 6),
    }
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
        "task_mode": annotations.get("task_mode"),
        "execution_backend": args.execution_backend,
        "prompt_mode": PROMPT_MODE,
        "success": bool(annotations["success"]),
        "submit_success": bool(annotations["submit_success"]),
        "stop_reason": annotations["stop_reason"],
        "failure_category": annotations["failure_category"],
        "failure_detail": annotations["failure_detail"],
        "question_total": annotations["question_total"],
        "attempted_correctness": annotations["attempted_correctness"],
        "verified_correctness": annotations["verified_correctness"],
        "action_count": annotations["action_count"],
        "invalid_actions": annotations["invalid_actions"],
        "duration_s": annotations["duration_s"],
        "token_usage": annotations["token_usage"],
        "cost_estimate": annotations["cost_estimate"],
        "projected_30_run_cost": annotations["projected_30_run_cost"],
        "artifacts": annotations["artifacts"],
    }
    rbe._append_jsonl(
        paths["model_io_path"],
        {
            "phase": "terminal",
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "stop_reason": summary["stop_reason"],
            "success": summary["success"],
            "submit_success": summary["submit_success"],
            "token_usage": summary["token_usage"],
            "cost_estimate": summary["cost_estimate"],
            "projected_30_run_cost": summary["projected_30_run_cost"],
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
        "task_mode": annotations.get("task_mode"),
        "success": summary["success"],
        "submit_success": summary["submit_success"],
        "stop_reason": summary["stop_reason"],
        "failure_category": summary["failure_category"],
        "failure_detail": summary["failure_detail"],
        "token_usage": summary["token_usage"],
        "cost_estimate": summary["cost_estimate"],
        "projected_30_run_cost": summary["projected_30_run_cost"],
        "summary_path": str(paths["summary_path"]),
        "annotations_path": str(paths["annotations_path"]),
        "trace_path": str(paths["trace_path"]),
        "model_io_path": str(paths["model_io_path"]),
        "step_inputs_path": str(paths["step_inputs_path"]),
        "video_path": annotations["artifacts"]["video_path"],
        "artifact_dir": str(paths["artifact_dir"]),
    }
    rbe._append_jsonl(paths["manifest_path"], manifest_entry)
    rbe._update_experiment_indexes(paths["experiment_root"], manifest_entry, run_label, args.retention_window)

    print(f"[INFO] wrote gemini low-cost summary: {paths['summary_path']}")
    print(f"[INFO] provider: gemini_low_cost")
    print(f"[INFO] gemini_model: {adapter.model}")
    print(f"[INFO] stop_reason: {summary['stop_reason']}")
    print(f"[INFO] success: {summary['success']}")
    print(f"[INFO] submit_success: {summary['submit_success']}")
    print(f"[INFO] token_usage: {json.dumps(summary['token_usage'], sort_keys=True)}")
    print(f"[INFO] cost_estimate: {json.dumps(summary['cost_estimate'], sort_keys=True)}")
    print(f"[INFO] projected_30_run_cost: {json.dumps(summary['projected_30_run_cost'], sort_keys=True)}")
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
