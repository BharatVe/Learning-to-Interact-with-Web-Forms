import argparse
import json
import os
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
from baselines.action_schema import parse_action, validate_action  # noqa: E402
from baselines.model_registry import get_model_by_id  # noqa: E402
from baselines.prompt_builders import build_text_prompt, compact_page_text  # noqa: E402
from engine.browser_language import force_english_google_forms_url  # noqa: E402
from engine.runner import iter_run_specs, load_form_spec, resolve_answers_path  # noqa: E402
from engine.trace_logger import TraceLogger  # noqa: E402

DEFAULT_ANSWERS_ROOT = "data/answers"
DEFAULT_DATASET_ROOT = "data/model_baselines"
DEFAULT_CONFIG = "configs/baselines/minimal_models.json"
DEFAULT_EXPERIMENT_ID = "baseline_direct_api_v1"
DEFAULT_MAX_STEPS = 15
DEFAULT_TIMEOUT_S = 300
DEFAULT_API_TIMEOUT_S = 60
DEFAULT_MAX_NEW_TOKENS = 192
DEFAULT_PROVIDER = "auto"
DEFAULT_PROMPT_MODE = "answers_labels_types_values"
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


def _extract_openai_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("openai_response_missing_choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError("openai_response_missing_message")
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "\n".join(parts).strip()
    raise RuntimeError("openai_response_missing_text")


def _extract_anthropic_text(payload: Dict[str, Any]) -> str:
    content = payload.get("content")
    if not isinstance(content, list):
        raise RuntimeError("anthropic_response_missing_content")
    parts: List[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
    if not parts:
        raise RuntimeError("anthropic_response_missing_text")
    return "\n".join(parts).strip()


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
        raise RuntimeError(f"api_http_error:{exc.code}:{raw}") from exc
    except Exception as exc:
        raise RuntimeError(f"api_request_failed:{exc}") from exc

    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"api_invalid_json:{exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("api_response_not_object")
    return parsed


def _select_provider(provider_arg: str) -> str:
    value = str(provider_arg or "").strip().lower()
    if value in {"openai", "anthropic"}:
        return value
    if value != "auto":
        raise ValueError(f"Unsupported provider: {provider_arg}")
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    raise RuntimeError("provider_auto_detect_failed: missing OPENAI_API_KEY and ANTHROPIC_API_KEY")


class DirectAPIAdapter:
    def __init__(self, provider: str, model_cfg: Dict[str, Any], api_timeout_s: int) -> None:
        self.provider = _select_provider(provider)
        self.model_cfg = dict(model_cfg)
        self.api_timeout_s = max(1, int(api_timeout_s))

    def infer(self, prompt: str, max_new_tokens: int) -> Tuple[str, Dict[str, Any]]:
        started = time.perf_counter()
        if self.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY missing")
            model = str(os.environ.get("OPENAI_MODEL") or self.model_cfg.get("openai_model") or "gpt-4.1-mini")
            payload = {
                "model": model,
                "temperature": 0,
                "max_tokens": int(max_new_tokens),
                "messages": [
                    {"role": "system", "content": "Return exactly one JSON action object and nothing else."},
                    {"role": "user", "content": prompt},
                ],
            }
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            response = _http_post_json(
                url=f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                payload=payload,
                timeout_s=self.api_timeout_s,
            )
            text = _extract_openai_text(response)
            meta = {"provider": "openai", "provider_model": model, "duration_s": round(time.perf_counter() - started, 3)}
            return text, meta

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY missing")
        model = str(os.environ.get("ANTHROPIC_MODEL") or self.model_cfg.get("anthropic_model") or "claude-3-5-sonnet-latest")
        payload = {
            "model": model,
            "max_tokens": int(max_new_tokens),
            "temperature": 0,
            "system": "Return exactly one JSON action object and nothing else.",
            "messages": [{"role": "user", "content": prompt}],
        }
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        response = _http_post_json(
            url=f"{base_url}/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            payload=payload,
            timeout_s=self.api_timeout_s,
        )
        text = _extract_anthropic_text(response)
        meta = {"provider": "anthropic", "provider_model": model, "duration_s": round(time.perf_counter() - started, 3)}
        return text, meta


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct API baseline evaluation for one model/form/run.")
    parser.add_argument("--model-id", default="computer_use_mcp_api")
    parser.add_argument("--form-id", required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--provider", choices=["auto", "openai", "anthropic"], default=DEFAULT_PROVIDER)
    parser.add_argument("--api-timeout-s", type=int, default=DEFAULT_API_TIMEOUT_S)
    parser.add_argument("--answers-root", default=DEFAULT_ANSWERS_ROOT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--trial-id")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--execution-backend", choices=["local", "mcp_server"], default="mcp_server")
    parser.add_argument("--browser-mcp-cmd")
    parser.add_argument("--browser-mcp-timeout-ms", type=int, default=120000)
    parser.add_argument("--invalid-action-budget", type=int, default=0)
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--viewport-height", type=int, default=720)
    parser.add_argument("--disable-action-coercion", action="store_true", default=False)
    parser.add_argument("--compact-page-text-max-chars", type=int, default=rbe.DEFAULT_COMPACT_PAGE_TEXT_MAX_CHARS)
    parser.add_argument("--retention-window", type=int, default=rbe.DEFAULT_RETENTION_WINDOW)
    parser.add_argument("--run-label")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    args.compact_page_text_max_chars = max(500, int(args.compact_page_text_max_chars))
    args.retention_window = max(0, int(args.retention_window))
    run_label = rbe._make_run_label(args.run_label)
    run_started_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    config_path = (ROOT_DIR / args.config).resolve()
    model_cfg = get_model_by_id(config_path, args.model_id)
    if model_cfg.get("provider") != "api_over_mcp":
        raise ValueError(f"run_direct_api_eval expects provider=api_over_mcp: {args.model_id}")

    adapter = DirectAPIAdapter(provider=args.provider, model_cfg=model_cfg, api_timeout_s=args.api_timeout_s)
    resolved_provider = adapter.provider

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
        "api_provider": resolved_provider,
        "form_id": args.form_id,
        "answer_run_id": answer_run_id,
        "form_url": form_url,
        "execution_backend": args.execution_backend,
        "prompt_mode": DEFAULT_PROMPT_MODE,
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
            "api_timeout_s": args.api_timeout_s,
            "provider": args.provider,
            "disable_action_coercion": bool(args.disable_action_coercion),
            "compact_page_text_max_chars": int(args.compact_page_text_max_chars),
            "retention_window": int(args.retention_window),
            "run_label": run_label,
        },
        "artifacts": rbe._artifact_payload(paths),
        "trace": {},
        "environment": {},
        "steps": [],
        "questions": question_states,
        "failure_events": [],
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
            "api_provider": resolved_provider,
        },
    )

    execution_session = None
    last_result: Dict[str, Any] = {}
    terminal_screenshot_path: Optional[str] = None
    observation_cache: Dict[int, Dict[str, Any]] = {}

    try:
        execution_session = rbe._make_execution_session(args, paths, trace)
        annotations["environment"] = execution_session.start(form_url) or {}
        observation_cache[0] = execution_session.observe(0)

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
            page_text = str(observation.get("page_text") or "")
            screenshot_path = observation.get("screenshot_path")

            prompt = build_text_prompt(
                form_url,
                remaining_answers,
                page_text,
                last_result,
                compact_page_text_max_chars=args.compact_page_text_max_chars,
            )

            step_input_record = {
                "phase": "step_input",
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "step_index": step_idx,
                "form_url": form_url,
                "remaining_answers": remaining_answers,
                "last_result": dict(last_result or {}),
                "page_text_excerpt": compact_page_text(page_text, max_chars=int(args.compact_page_text_max_chars)),
                "screenshot_path": screenshot_path,
                "behavior_nudge": None,
                "prompt_hash": rbe._prompt_hash(prompt),
            }
            rbe._append_jsonl(paths["step_inputs_path"], step_input_record)
            infer_started = time.perf_counter()
            try:
                raw_output, infer_meta = adapter.infer(prompt, max_new_tokens=args.max_new_tokens)
                infer_meta["duration_s"] = round(time.perf_counter() - infer_started, 3)
            except Exception as exc:
                infer_error = f"model_inference_failed: {exc}"
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
                    "model_inference": {"attempts": [{"attempt": 1, "error": str(exc)}]},
                }
                annotations["steps"].append(step_record)
                rbe._set_failure(annotations, "model_inference_failed", str(exc), step_idx)
                annotations["stop_reason"] = "model_inference_failed"
                break

            step_record: Dict[str, Any] = {
                "step_index": step_idx,
                "elapsed_s": round(time.perf_counter() - start_time, 3),
                "prompt_mode": DEFAULT_PROMPT_MODE,
                "remaining_answers_before": len(remaining_answers),
                "page_text_excerpt": page_text[:2000],
                "screenshot_path": screenshot_path,
                "raw_model_output": raw_output,
                "action": None,
                "warnings": [],
                "status": None,
                "error": None,
                "matched_question_id": None,
                "target_match": None,
                "execution": None,
                "verification": None,
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
                "raw_model_output": raw_output,
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
                parsed = parse_action(raw_output)
                action, warnings = validate_action(parsed)
                step_record["action"] = action
                step_record["warnings"] = warnings
                io_record["parsed_action"] = action
                io_record["warnings"] = warnings
            except Exception as exc:
                annotations["invalid_actions"] += 1
                message = f"model_output_invalid: {exc}"
                step_record["status"] = "failed"
                step_record["error"] = message
                io_record["error"] = message
                rbe._set_failure(annotations, "model_output_invalid", str(exc), step_idx)
                annotations["steps"].append(step_record)
                rbe._append_jsonl(paths["model_io_path"], io_record)
                last_result = {"status": "failed", "error": message, "remaining_answers": len(remaining_answers)}
                if rbe._invalid_action_budget_exhausted(annotations["invalid_actions"], args.invalid_action_budget):
                    annotations["stop_reason"] = "model_output_invalid"
                    break
                continue

            action_name = action["action"]
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
                rbe._set_failure(annotations, "premature_submit", detail, step_idx)
                annotations["steps"].append(step_record)
                rbe._append_jsonl(paths["model_io_path"], io_record)
                last_result = {
                    "status": step_record["status"],
                    "error": step_record["error"],
                    "remaining_answers": len(rbe._serialize_remaining_answers(question_states)),
                }
                continue

            if action_name == "wait":
                wait_seconds = max(0.25, float(action.get("delta") or 1000) / 1000.0)
                execution_session.execute_wait(wait_seconds, step_idx)
                step_record["status"] = "waited"
            elif action_name == "scroll":
                delta = int(action.get("delta") or 600)
                execution_session.execute_scroll(delta, step_idx)
                step_record["status"] = "scrolled"
            elif action_name == "press_key":
                key = str(action.get("value") or "Tab")
                execution_session.execute_press_key(key, step_idx)
                step_record["status"] = "pressed_key"
            elif action_name == "submit":
                submit_info, submit_err = execution_session.submit()
                step_record["execution"] = submit_info
                if submit_err:
                    step_record["status"] = "failed"
                    step_record["error"] = f"submission_failed: {submit_err}"
                    annotations["stop_reason"] = "submission_failed"
                    rbe._set_failure(annotations, "submission_failed", submit_err, step_idx)
                elif submit_info.get("success"):
                    step_record["status"] = "submitted"
                    annotations["success"] = True
                    annotations["submit_success"] = True
                    annotations["stop_reason"] = "submitted"
                else:
                    step_record["status"] = "failed"
                    step_record["error"] = "submission_failed: not confirmed"
                    annotations["stop_reason"] = "submission_failed"
                    rbe._set_failure(annotations, "submission_failed", json.dumps(submit_info, ensure_ascii=True), step_idx)
            elif action_name == "done":
                step_record["status"] = "done"
                annotations["stop_reason"] = "done"
            else:
                matched_idx, question_state, match_debug = rbe._match_question_state(question_states, action.get("target", {}))
                step_record["target_match"] = match_debug
                io_record["target_match"] = match_debug
                if question_state is None or matched_idx is None:
                    step_record["status"] = "failed"
                    step_record["error"] = "target_not_found"
                    rbe._set_failure(annotations, "target_not_found", json.dumps(match_debug, ensure_ascii=True), step_idx)
                else:
                    if not bool(args.disable_action_coercion):
                        action, warnings2 = rbe._coerce_action_for_widget(action, question_state)
                        if warnings2:
                            step_record["warnings"] = list(dict.fromkeys(step_record["warnings"] + warnings2))
                            io_record["warnings"] = list(dict.fromkeys(io_record["warnings"] + warnings2))
                    resolved_action_name = action.get("action")
                    if not rbe._action_supported_for_widget(str(resolved_action_name), str(question_state.get("widget_type") or "")):
                        step_record["status"] = "failed"
                        step_record["error"] = f"widget_interaction_failed: incompatible_action_for_widget:{question_state.get('widget_type')}"
                        rbe._set_failure(annotations, "widget_interaction_failed", f"incompatible_action_for_widget:{question_state.get('widget_type')}", step_idx)
                        annotations["steps"].append(step_record)
                        rbe._append_jsonl(paths["model_io_path"], io_record)
                        last_result = {"status": step_record["status"], "error": step_record["error"], "remaining_answers": len(rbe._serialize_remaining_answers(question_states))}
                        continue
                    exec_entry = rbe._build_entry_from_action(action, question_state)
                    question_state["attempted"] = True
                    step_record["matched_question_id"] = question_state.get("question_id")
                    action_result, exec_err = execution_session.execute_fill(exec_entry, step_idx)
                    verification_result = execution_session.verify_entry(question_state, step_idx)
                    question_state["last_execution"] = action_result
                    question_state["last_verification"] = verification_result
                    question_state["actual_value"] = verification_result.get("actual_value")
                    question_state["attempted_correct"] = rbe._value_matches(question_state.get("value"), exec_entry.get("value"))
                    question_state["verified"] = bool(verification_result.get("verified"))
                    question_state["verified_correct"] = bool(verification_result.get("verified")) and rbe._value_matches(
                        question_state.get("value"), verification_result.get("actual_value")
                    )
                    if question_state["verified_correct"]:
                        question_state["final_status"] = "correct_verified"
                    elif question_state["attempted_correct"]:
                        question_state["final_status"] = "correct_attempted_only"
                    else:
                        question_state["final_status"] = "failed"

                    step_record["execution"] = action_result
                    step_record["verification"] = verification_result
                    step_record["expected_label"] = question_state.get("label")
                    step_record["expected_value"] = question_state.get("value")
                    step_record["executed_value"] = exec_entry.get("value")

                    if exec_err:
                        step_record["status"] = "failed"
                        step_record["error"] = f"widget_interaction_failed: {exec_err}"
                        rbe._set_failure(annotations, "widget_interaction_failed", exec_err, step_idx)
                    elif question_state["verified"] and not question_state["verified_correct"]:
                        step_record["status"] = "filled_unverified"
                        step_record["error"] = "verification_failed"
                        rbe._set_failure(
                            annotations,
                            "verification_failed",
                            json.dumps({"expected": question_state.get("value"), "actual": verification_result.get("actual_value")}, ensure_ascii=True),
                            step_idx,
                        )
                    elif not question_state["verified"]:
                        step_record["status"] = "filled_unverified"
                        step_record["error"] = f"verification_failed: {verification_result.get('detail')}"
                        rbe._set_failure(annotations, "verification_failed", str(verification_result.get("detail")), step_idx)
                    else:
                        step_record["status"] = "filled"

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

    print(f"[INFO] wrote direct baseline summary: {paths['summary_path']}")
    print(f"[INFO] wrote direct baseline annotations: {paths['annotations_path']}")
    print(f"[INFO] wrote direct baseline manifest: {paths['manifest_path']}")
    print(f"[INFO] provider: {resolved_provider}")
    print(f"[INFO] stop_reason: {summary['stop_reason']}")
    print(f"[INFO] success: {summary['success']}")
    print(f"[INFO] submit_success: {summary['submit_success']}")
    print(f"[INFO] attempted_correctness: {summary['attempted_correctness']}/{summary['question_total']}")
    print(f"[INFO] verified_correctness: {summary['verified_correctness']}/{summary['question_total']}")
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
