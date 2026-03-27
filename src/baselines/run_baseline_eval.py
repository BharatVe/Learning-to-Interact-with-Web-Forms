import argparse
import hashlib
import json
import os
import re
import signal
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines.action_schema import parse_action, validate_action  # noqa: E402
from baselines.model_registry import get_model_by_id  # noqa: E402
from baselines.prompt_builders import (  # noqa: E402
    CONTEXT_PACKAGE_VERSION,
    build_text_prompt,
    build_vlm_prompt,
    compact_page_text,
    selected_canonical_fewshot_ids,
)
from engine.form_engine import FormEngine  # noqa: E402
from engine.mcp_browser_engine import MCPBrowserEngine  # noqa: E402
from engine.mcp_trace_client import MCPClient, MCPTraceClient  # noqa: E402
from engine.runner import (  # noqa: E402
    _default_mcp_server_command,
    iter_run_specs,
    load_form_spec,
    resolve_answers_path,
)
from engine.trace_logger import TraceLogger  # noqa: E402

DEFAULT_ANSWERS_ROOT = "data/answers"
DEFAULT_DATASET_ROOT = "data/model_baselines"
DEFAULT_MAX_STEPS = 36
DEFAULT_TIMEOUT_S = 1800
DEFAULT_INVALID_ACTION_BUDGET = 0
DEFAULT_MAX_NEW_TOKENS = 224
DEFAULT_EXECUTION_BACKEND = "mcp_server"
DEFAULT_PROMPT_MODE = "answers_labels_types_values"
DEFAULT_PROMPT_PROFILE = "detailed_v1"
DEFAULT_HISTORY_WINDOW = 4
DEFAULT_FEWSHOT_COUNT = 3
DEFAULT_EXPERIMENT_ID = "baseline_mcp_v1"
DEFAULT_BROWSER_MCP_TIMEOUT_MS = 120000
DEFAULT_API_TIMEOUT_S = 120
DEFAULT_INFERENCE_BACKEND = "auto"
DEFAULT_BROWSER_INIT_RETRIES = 2
DEFAULT_BROWSER_INIT_RETRY_DELAY_S = 1.5
DEFAULT_VIEWPORT_WIDTH = 1280
DEFAULT_VIEWPORT_HEIGHT = 720
DEFAULT_TRACE_MCP_TIMEOUT_MS = 5000
DEFAULT_STEP_SOFT_TIMEOUT_S = 60.0
DEFAULT_STEP_RETRY_MAX_NEW_TOKENS = 160
DEFAULT_IDLE_STEP_THRESHOLD = 4
DEFAULT_IDLE_NUDGE_MAX = 3
DEFAULT_COMPACT_PAGE_TEXT_MAX_CHARS = 5000
DEFAULT_RETENTION_WINDOW = 5
DEFAULT_BROWSER_SNAPSHOT_MODE = "none"
DEFAULT_TRACK = "mediated"
BASELINE_EVAL_SCHEMA_VERSION = "baseline_eval.v3"
BASELINE_SUMMARY_SCHEMA_VERSION = "baseline_summary.v3"

TEXT_WIDGET_ACTIONS = {
    "short_text": {"type"},
    "paragraph_text": {"type"},
    "date": {"type"},
    "time": {"type"},
}
CHOICE_WIDGET_ACTIONS = {
    "single_choice": {"select_option", "click"},
    "multi_choice": {"select_option", "click"},
}
ALLOWED_WIDGET_ACTIONS = {**TEXT_WIDGET_ACTIONS, **CHOICE_WIDGET_ACTIONS}


def _norm_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_run_answers(answers_path: Path, run_index: int) -> List[Dict[str, Any]]:
    for idx, run_spec in enumerate(iter_run_specs(answers_path), start=1):
        if idx == run_index:
            answers = run_spec.get("answers", [])
            if not isinstance(answers, list):
                raise ValueError(f"Run {run_index} answers must be a list")
            return answers
    raise IndexError(f"Run index out of range: {run_index} for {answers_path}")


def _select_inference_backend(model_cfg: Dict[str, Any], requested_backend: str) -> str:
    provider = str(model_cfg.get("provider") or "").strip().lower()
    requested = str(requested_backend or DEFAULT_INFERENCE_BACKEND).strip().lower()
    supported = {"local_hf", "openai_compat"}
    if provider not in supported:
        raise ValueError(f"Unsupported provider for run_baseline_eval: {provider}")
    if requested == "auto":
        return provider
    if requested not in supported:
        raise ValueError(f"Unsupported inference backend: {requested_backend}")
    if requested != provider:
        raise ValueError(
            f"inference_backend_mismatch: provider={provider} requested={requested}. "
            "Use --inference-backend auto or align model provider."
        )
    return requested


def _make_adapter(
    model_cfg: Dict[str, Any],
    model_kind: str,
    max_new_tokens: int,
    inference_backend: str,
    api_timeout_s: int,
):
    if inference_backend == "openai_compat":
        from baselines.model_adapters.openai_compat import OpenAICompatAdapter

        return OpenAICompatAdapter(
            model_cfg=model_cfg,
            model_kind=model_kind,
            max_new_tokens=max_new_tokens,
            api_timeout_s=api_timeout_s,
        )

    model_dir = ROOT_DIR / "models" / str(model_cfg["id"])
    if model_kind == "text_llm":
        from baselines.model_adapters.local_text import LocalTextAdapter

        return LocalTextAdapter(model_dir=model_dir, max_new_tokens=max_new_tokens)
    if model_kind == "vlm":
        from baselines.model_adapters.local_vlm import LocalVLMAdapter

        return LocalVLMAdapter(model_dir=model_dir, max_new_tokens=max_new_tokens)
    raise ValueError(f"Unsupported model kind for local baseline: {model_kind}")


class _SoftTimeoutError(TimeoutError):
    pass


def _track_name(model_cfg: Dict[str, Any]) -> str:
    track = str(model_cfg.get("track") or "").strip()
    return track or DEFAULT_TRACK


def _requires_gpu(model_cfg: Dict[str, Any], require_gpu_flag: bool) -> bool:
    return bool(require_gpu_flag or model_cfg.get("requires_gpu"))


def _ensure_gpu_available(require_gpu: bool) -> None:
    if not require_gpu:
        return
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"gpu_check_failed: {exc}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("gpu_required_for_model")


def _ensure_model_runtime_compat(model_cfg: Dict[str, Any]) -> None:
    if str(model_cfg.get("provider") or "") != "local_hf":
        return
    model_id = str(model_cfg.get("id") or "").strip()
    if not model_id:
        return
    model_dir = ROOT_DIR / "models" / model_id
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise RuntimeError(f"model_runtime_check_failed: missing config.json for {model_id} at {model_dir}")
    try:
        from transformers import AutoConfig
    except Exception as exc:
        raise RuntimeError(f"model_runtime_check_failed: transformers_import_error: {exc}") from exc
    try:
        AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "model_runtime_incompatible: "
            f"{model_id} cannot be loaded by current transformers runtime ({exc}). "
            "Upgrade transformers to a version that supports this architecture."
        ) from exc


def _with_soft_timeout(timeout_s: float, fn):
    if timeout_s <= 0:
        return fn()

    def _raise_timeout(signum, frame):
        _ = signum
        _ = frame
        raise _SoftTimeoutError(f"model_step_soft_timeout:{timeout_s}s")

    prev_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev_handler)


def _is_oom_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return (
        "out of memory" in text
        or "cuda out of memory" in text
        or "cudnn_status_alloc_failed" in text
        or "hip out of memory" in text
    )


def _infer_with_retry(
    adapter: Any,
    args: argparse.Namespace,
    model_kind: str,
    prompt: str,
    image_path: Optional[Path],
) -> Tuple[str, Dict[str, Any]]:
    attempts: List[Dict[str, Any]] = []
    token_budgets = [int(args.max_new_tokens)]
    retry_budget = int(args.step_retry_max_new_tokens or 0)
    if retry_budget > 0 and retry_budget != token_budgets[0]:
        token_budgets.append(retry_budget)

    last_exc: Optional[Exception] = None
    for attempt_idx, token_budget in enumerate(token_budgets, start=1):
        started = time.perf_counter()
        timed_out = False
        err_msg = None
        try:
            if model_kind == "text_llm":
                output = _with_soft_timeout(
                    float(args.step_soft_timeout_s),
                    lambda: adapter.infer(prompt, max_new_tokens_override=token_budget),
                )
            else:
                assert image_path is not None
                output = _with_soft_timeout(
                    float(args.step_soft_timeout_s),
                    lambda: adapter.infer(prompt, image_path, max_new_tokens_override=token_budget),
                )
            attempts.append(
                {
                    "attempt": attempt_idx,
                    "max_new_tokens": token_budget,
                    "duration_s": round(time.perf_counter() - started, 3),
                    "timed_out": False,
                    "oom": False,
                    "error": None,
                }
            )
            return str(output), {"attempts": attempts, "retried": attempt_idx > 1}
        except Exception as exc:
            last_exc = exc
            timed_out = isinstance(exc, _SoftTimeoutError)
            oom = _is_oom_error(exc)
            err_msg = str(exc)
            if oom:
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                next_budget = max(32, token_budget // 2)
                while next_budget < token_budget:
                    if next_budget not in token_budgets:
                        token_budgets.append(next_budget)
                        break
                    if next_budget <= 32:
                        break
                    next_budget = max(32, next_budget // 2)
            attempts.append(
                {
                    "attempt": attempt_idx,
                    "max_new_tokens": token_budget,
                    "duration_s": round(time.perf_counter() - started, 3),
                    "timed_out": timed_out,
                    "oom": oom,
                    "error": err_msg,
                }
            )
            if not (timed_out or oom) or attempt_idx == len(token_budgets):
                break

    raise RuntimeError(
        json.dumps({"message": "model_inference_failed", "attempts": attempts}, ensure_ascii=True)
    ) from last_exc


def _extract_inference_attempts(exc: Exception) -> List[Dict[str, Any]]:
    raw = str(exc or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    if not isinstance(parsed, dict):
        return []
    attempts = parsed.get("attempts")
    return attempts if isinstance(attempts, list) else []


def _build_idle_recovery_nudge(
    idle_streak: int,
    remaining_answers: List[Dict[str, Any]],
    nudge_index: int,
    nudge_max: int,
    validation_feedback: Optional[Dict[str, Any]] = None,
) -> str:
    remaining_ids = [
        str(item.get("question_id") or "").strip()
        for item in remaining_answers
        if str(item.get("question_id") or "").strip()
    ]
    feedback = dict(validation_feedback or {})
    feedback_text = ""
    if feedback:
        feedback_text = f" Last feedback: {json.dumps(feedback, ensure_ascii=True)}"
    return (
        f"No measurable progress for {idle_streak} consecutive steps. "
        f"Recovery attempt {nudge_index}/{nudge_max}. "
        "Choose ONE remaining question_id that is still unanswered, then execute a concrete UI interaction now. "
        "Do not repeat a previously failed target, and do not output wait/scroll/submit unless strictly necessary. "
        f"Prioritize unanswered IDs: {json.dumps(remaining_ids, ensure_ascii=True)}"
        f"{feedback_text}"
    )


def _apply_action_policy(
    action: Dict[str, Any],
    question_state: Dict[str, Any],
    disable_action_coercion: bool,
) -> Tuple[Dict[str, Any], List[str]]:
    if disable_action_coercion:
        return dict(action), []
    return _coerce_action_for_widget(action, question_state)


def _default_browser_mcp_command(
    viewport_width: int,
    viewport_height: int,
    artifact_dir: Path,
    headless: bool,
    browser_mcp_timeout_ms: int,
) -> List[str]:
    mcp_bin = shutil.which("playwright-mcp")
    timeout_ms = max(15000, int(browser_mcp_timeout_ms))
    command = ([mcp_bin] if mcp_bin else ["npx", "-y", "@playwright/mcp@latest"]) + [
        "--browser",
        "chromium",
        "--isolated",
        "--host",
        "127.0.0.1",
        "--output-dir",
        str(artifact_dir),
        "--save-video",
        f"{viewport_width}x{viewport_height}",
        "--viewport-size",
        f"{viewport_width},{viewport_height}",
        "--snapshot-mode",
        DEFAULT_BROWSER_SNAPSHOT_MODE,
        "--timeout-action",
        str(timeout_ms),
        "--timeout-navigation",
        str(max(60000, timeout_ms)),
    ]
    if headless:
        command.append("--headless")
    return command


def _action_supported_for_widget(action_name: str, widget_type: str) -> bool:
    if action_name in {"wait", "scroll", "press_key", "submit", "done"}:
        return True
    return action_name in ALLOWED_WIDGET_ACTIONS.get(widget_type, set())


def _make_trial_id() -> str:
    return "trial_" + datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def _sanitize_job_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(value or "").strip())
    return cleaned or "na"


def _make_run_label(run_label_override: Optional[str] = None) -> str:
    candidate = str(run_label_override or "").strip()
    if candidate:
        return candidate
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    job_id = _sanitize_job_id(os.environ.get("SLURM_JOB_ID") or "na")
    return f"{timestamp}_job{job_id}"


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(str(prompt or "").encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_trial_sort_timestamp(summary_path: Path, trial_dir: Path) -> float:
    payload = _load_json(summary_path)
    if isinstance(payload, dict):
        ts = payload.get("run_completed_utc") or payload.get("run_started_utc")
        if isinstance(ts, str) and ts.strip():
            raw = ts.strip()
            try:
                if raw.endswith("Z"):
                    raw = raw[:-1] + "+00:00"
                return datetime.fromisoformat(raw).timestamp()
            except Exception:
                pass
    try:
        return summary_path.stat().st_mtime
    except Exception:
        return trial_dir.stat().st_mtime


def _collect_trial_dirs(run_root: Path) -> List[Path]:
    if not run_root.exists():
        return []
    return [path for path in run_root.iterdir() if path.is_dir()]


def _apply_retention_window(
    experiment_root: Path,
    model_id: str,
    form_id: str,
    answer_run_id: str,
    retention_window: int,
) -> List[Dict[str, Any]]:
    keep_n = max(0, int(retention_window))
    if keep_n <= 0:
        return []

    run_root = experiment_root / model_id / form_id / answer_run_id
    trial_dirs = _collect_trial_dirs(run_root)
    if len(trial_dirs) <= keep_n:
        return []

    def _sort_key(path: Path) -> Tuple[float, str]:
        summary_path = path / "summary.json"
        return (_extract_trial_sort_timestamp(summary_path, path), path.name)

    sorted_dirs = sorted(trial_dirs, key=_sort_key, reverse=True)
    to_archive = sorted_dirs[keep_n:]
    archive_root = experiment_root / "_archive" / model_id / form_id / answer_run_id
    archived: List[Dict[str, Any]] = []

    for trial_dir in to_archive:
        destination = archive_root / trial_dir.name
        if destination.exists():
            destination = archive_root / f"{trial_dir.name}_{int(time.time())}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(trial_dir), str(destination))
        archived.append(
            {
                "trial_id": trial_dir.name,
                "from_artifact_dir": str(trial_dir),
                "to_artifact_dir": str(destination),
                "archived": True,
            }
        )

    return archived


def _collect_latest_entries(experiment_root: Path) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}

    canonical_glob = experiment_root.glob("*/*/run_*/*/summary.json")
    for summary_path in canonical_glob:
        try:
            rel_parts = summary_path.relative_to(experiment_root).parts
        except Exception:
            continue
        if len(rel_parts) < 5:
            continue
        model_id, form_id, answer_run_id, trial_id = rel_parts[0], rel_parts[1], rel_parts[2], rel_parts[3]
        key = f"{model_id}|{form_id}|{answer_run_id}"
        ts = _extract_trial_sort_timestamp(summary_path, summary_path.parent)
        payload = _load_json(summary_path)
        run_label = payload.get("run_label") if isinstance(payload, dict) else None
        row = {
            "model_id": model_id,
            "form_id": form_id,
            "answer_run_id": answer_run_id,
            "trial_id": trial_id,
            "summary_path": str(summary_path),
            "artifact_dir": str(summary_path.parent),
            "run_label": run_label,
            "archived": False,
            "updated_ts": ts,
        }
        prev = latest.get(key)
        if prev is None or float(row["updated_ts"]) >= float(prev.get("updated_ts") or 0):
            latest[key] = row

    for row in latest.values():
        row.pop("updated_ts", None)
    return latest


def _update_experiment_indexes(
    experiment_root: Path,
    manifest_entry: Dict[str, Any],
    run_label: str,
    retention_window: int,
) -> None:
    index_root = experiment_root / "_index"
    by_job_root = index_root / "by_job"
    index_root.mkdir(parents=True, exist_ok=True)
    by_job_root.mkdir(parents=True, exist_ok=True)

    job_id = _sanitize_job_id(os.environ.get("SLURM_JOB_ID") or "na")
    event_now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    run_row = {
        "event": "trial_recorded",
        "event_time_utc": event_now,
        "job_id": job_id,
        "run_label": run_label,
        "experiment_id": manifest_entry.get("experiment_id"),
        "model_id": manifest_entry.get("model_id"),
        "model_kind": manifest_entry.get("model_kind"),
        "form_id": manifest_entry.get("form_id"),
        "answer_run_id": manifest_entry.get("answer_run_id"),
        "trial_id": manifest_entry.get("trial_id"),
        "track": manifest_entry.get("track"),
        "provider": manifest_entry.get("provider"),
        "success": bool(manifest_entry.get("success")),
        "submit_success": bool(manifest_entry.get("submit_success")),
        "stop_reason": manifest_entry.get("stop_reason"),
        "failure_category": manifest_entry.get("failure_category"),
        "summary_path": manifest_entry.get("summary_path"),
        "artifact_dir": manifest_entry.get("artifact_dir"),
        "archived": False,
    }
    runs_path = index_root / "runs.jsonl"
    _append_jsonl(runs_path, run_row)

    archived_rows = _apply_retention_window(
        experiment_root=experiment_root,
        model_id=str(manifest_entry.get("model_id") or ""),
        form_id=str(manifest_entry.get("form_id") or ""),
        answer_run_id=str(manifest_entry.get("answer_run_id") or ""),
        retention_window=retention_window,
    )
    for archived in archived_rows:
        archive_event = {
            "event": "trial_archived",
            "event_time_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "job_id": job_id,
            "run_label": run_label,
            "experiment_id": manifest_entry.get("experiment_id"),
            "model_id": manifest_entry.get("model_id"),
            "form_id": manifest_entry.get("form_id"),
            "answer_run_id": manifest_entry.get("answer_run_id"),
            "trial_id": archived.get("trial_id"),
            "from_artifact_dir": archived.get("from_artifact_dir"),
            "to_artifact_dir": archived.get("to_artifact_dir"),
            "archived": True,
        }
        _append_jsonl(runs_path, archive_event)

    latest_payload = _collect_latest_entries(experiment_root)
    _write_json(index_root / "latest.json", latest_payload)

    run_rows: List[Dict[str, Any]] = []
    if runs_path.exists():
        for line in runs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except Exception:
                continue
            if not isinstance(parsed, dict):
                continue
            if parsed.get("event") != "trial_recorded":
                continue
            if str(parsed.get("job_id") or "") == job_id:
                run_rows.append(parsed)

    _write_json(
        by_job_root / f"job_{job_id}.json",
        {"job_id": job_id, "trial_count": len(run_rows), "trials": run_rows},
    )


def _value_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, list):
        expected_items = sorted(_norm_text(item) for item in expected)
        if isinstance(actual, list):
            actual_items = sorted(_norm_text(item) for item in actual)
        else:
            actual_items = sorted(_norm_text(item) for item in str(actual).split(",") if item.strip())
        return expected_items == actual_items
    return _norm_text(expected) == _norm_text(actual)


def _build_entry_from_action(action: Dict[str, Any], expected_entry: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(expected_entry)
    if "value" in action and action.get("value") is not None:
        resolved["value"] = action.get("value")
    return resolved


def _classify_environment_error(message: str) -> str:
    text = str(message or "").lower()
    if "browser_take_screenshot" in text or "screenshot" in text:
        return "screenshot_write_failed"
    return "environment_error"


def _build_question_states(answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    states: List[Dict[str, Any]] = []
    for idx, answer in enumerate(answers, start=1):
        state = dict(answer)
        state["question_id"] = f"q_{idx:03d}"
        state["attempted"] = False
        state["attempted_correct"] = False
        state["verified"] = False
        state["verified_correct"] = False
        state["actual_value"] = None
        state["final_status"] = "not_attempted"
        state["last_execution"] = None
        state["last_verification"] = None
        states.append(state)
    return states


def _serialize_remaining_answers(question_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    remaining: List[Dict[str, Any]] = []
    for state in question_states:
        if state.get("verified_correct"):
            continue
        remaining.append(
            {
                "question_id": state.get("question_id"),
                "label": state.get("label"),
                "widget_type": state.get("widget_type"),
                "value": state.get("value"),
            }
        )
    return remaining


def _serialize_visible_field_map(question_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    visible: List[Dict[str, Any]] = []
    for state in question_states:
        if state.get("verified_correct"):
            continue
        visible.append(
            {
                "question_id": state.get("question_id"),
                "label": state.get("label"),
                "widget_type": state.get("widget_type"),
                "expected_value": state.get("value"),
            }
        )
    return visible


def _recent_history_from_steps(steps: List[Dict[str, Any]], window: int) -> List[Dict[str, Any]]:
    keep = max(0, int(window))
    if keep <= 0:
        return []
    rows = steps[-keep:]
    history: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        history.append(
            {
                "step_index": row.get("step_index"),
                "action": (row.get("action") or {}).get("action") if isinstance(row.get("action"), dict) else None,
                "target_question_id": ((row.get("action") or {}).get("target") or {}).get("question_id")
                if isinstance((row.get("action") or {}).get("target"), dict)
                else None,
                "status": row.get("status"),
                "error": row.get("error"),
                "progress_made": row.get("progress_made"),
                "remaining_answers_before": row.get("remaining_answers_before"),
            }
        )
    return history


def _normalize_validation_feedback(last_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = dict(last_result or {})
    error = str(payload.get("error") or "").strip()
    if not error:
        return {
            "status": payload.get("status"),
            "category": "ok",
            "message": None,
            "hint": None,
        }

    error_l = error.lower()
    category = "execution_error"
    hint = "Choose a different remaining target.question_id and execute one concrete action."
    if "model_output_invalid" in error_l:
        category = "model_output_invalid"
        hint = "Return one valid JSON object that follows the output schema exactly."
    elif "target_not_found" in error_l:
        category = "target_not_found"
        hint = "Use one of the allowed target.question_id values from remaining answers."
    elif "incompatible_action_for_widget" in error_l:
        category = "action_widget_mismatch"
        hint = "Choose a widget-compatible action type for the target question."
    elif "verification_failed" in error_l:
        category = "verification_failed"
        hint = "Retry with exact expected value formatting and verify field focus/selection."
    elif "premature_submit" in error_l:
        category = "premature_submit"
        hint = "Do not submit until remaining answers is empty."

    return {
        "status": payload.get("status"),
        "category": category,
        "message": error,
        "hint": hint,
    }


def _available_open_labels(question_states: List[Dict[str, Any]]) -> List[str]:
    return [str(state.get("label") or "") for state in question_states if not state.get("verified_correct")]


def _match_question_state(
    question_states: List[Dict[str, Any]],
    target: Dict[str, Any],
) -> Tuple[Optional[int], Optional[Dict[str, Any]], Dict[str, Any]]:
    target = dict(target or {})
    question_id = str(target.get("question_id") or "").strip()
    alias_fields = [target.get("label"), target.get("text"), target.get("selector_hint")]
    raw_aliases = [str(item).strip() for item in alias_fields if isinstance(item, str) and item.strip()]
    normalized_aliases = [_norm_text(item) for item in raw_aliases]

    debug = {
        "target": target,
        "target_question_id": question_id or None,
        "target_candidates": raw_aliases,
        "available_question_ids": [
            str(state.get("question_id") or "")
            for state in question_states
            if not state.get("verified_correct")
        ],
        "available_labels": _available_open_labels(question_states),
        "match_strategy": None,
    }

    open_states = [(idx, state) for idx, state in enumerate(question_states) if not state.get("verified_correct")]
    if question_id:
        for idx, state in open_states:
            if str(state.get("question_id") or "").strip() == question_id:
                debug["match_strategy"] = "question_id"
                return idx, state, debug

    if normalized_aliases:
        for idx, state in open_states:
            label_norm = _norm_text(state.get("label", ""))
            if any(candidate == label_norm for candidate in normalized_aliases):
                debug["match_strategy"] = "exact_label"
                return idx, state, debug
        for idx, state in open_states:
            label_norm = _norm_text(state.get("label", ""))
            if any(candidate in label_norm or label_norm in candidate for candidate in normalized_aliases):
                debug["match_strategy"] = "alias_substring"
                return idx, state, debug

    return None, None, debug


def _coerce_action_for_widget(
    action: Dict[str, Any],
    question_state: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    widget_type = str(question_state.get("widget_type") or "")
    action_name = str(action.get("action") or "")
    coerced = dict(action)
    warnings: List[str] = []

    if widget_type in TEXT_WIDGET_ACTIONS and action_name in {"select_option", "click"}:
        coerced["action"] = "type"
        warnings.append(f"coerced_action:{action_name}->type")
    elif widget_type in CHOICE_WIDGET_ACTIONS and action_name == "type":
        coerced["action"] = "select_option"
        warnings.append("coerced_action:type->select_option")

    return coerced, warnings


def _invalid_action_budget_exhausted(invalid_actions: int, invalid_action_budget: int) -> bool:
    return invalid_action_budget > 0 and invalid_actions >= invalid_action_budget


def _calculate_metrics(question_states: List[Dict[str, Any]]) -> Dict[str, int]:
    attempted_count = sum(1 for item in question_states if item.get("attempted"))
    attempted_correctness = sum(1 for item in question_states if item.get("attempted_correct"))
    verified_count = sum(1 for item in question_states if item.get("verified"))
    verified_correctness = sum(1 for item in question_states if item.get("verified_correct"))
    return {
        "question_total": len(question_states),
        "attempted_count": attempted_count,
        "attempted_correctness": attempted_correctness,
        "verified_count": verified_count,
        "verified_correctness": verified_correctness,
        "question_correctness": verified_correctness,
    }


def _set_failure(annotations: Dict[str, Any], category: str, detail: str, step_index: Optional[int] = None) -> None:
    annotations["failure_category"] = category
    annotations["failure_detail"] = detail
    event: Dict[str, Any] = {"type": category, "detail": detail}
    if step_index is not None:
        event["step_index"] = step_index
    annotations.setdefault("failure_events", []).append(event)


def _finalize_trial_video(artifact_dir: Path, final_video_path: Path) -> Optional[Path]:
    if final_video_path.exists():
        return final_video_path
    try:
        candidate_videos = sorted(
            [path for path in artifact_dir.rglob("*.webm") if path != final_video_path],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception:
        candidate_videos = []
    if not candidate_videos:
        return None
    raw_video_path = candidate_videos[0]
    if final_video_path.exists():
        final_video_path.unlink()
    try:
        raw_video_path.rename(final_video_path)
        return final_video_path
    except Exception:
        return raw_video_path


class ExecutionSessionBase:
    def __init__(
        self,
        artifact_dir: Path,
        observations_dir: Path,
        trace: TraceLogger,
        headless: bool,
        viewport_width: int,
        viewport_height: int,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.observations_dir = observations_dir
        self.trace = trace
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.engine = None

    def _observation_path(self, step_idx: int) -> Path:
        return self.observations_dir / f"step_{step_idx:04d}.png"

    def _capture_screenshot(self, path: Path, step_ref: Optional[int]) -> Optional[str]:
        raise NotImplementedError

    def _get_page_text(self, step_idx: int) -> str:
        raise NotImplementedError

    def observe(self, step_idx: int) -> Dict[str, Any]:
        screenshot_path = self._capture_screenshot(self._observation_path(step_idx), step_idx)
        return {
            "page_text": self._get_page_text(step_idx),
            "screenshot_path": screenshot_path,
        }

    def capture_terminal_screenshot(self, filename: str) -> Optional[str]:
        return self._capture_screenshot((self.artifact_dir / filename).resolve(), step_ref=None)


class LocalExecutionSession(ExecutionSessionBase):
    def __init__(
        self,
        artifact_dir: Path,
        observations_dir: Path,
        trace: TraceLogger,
        headless: bool,
        viewport_width: int,
        viewport_height: int,
    ) -> None:
        super().__init__(
            artifact_dir=artifact_dir,
            observations_dir=observations_dir,
            trace=trace,
            headless=headless,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
        )
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    def start(self, form_url: str) -> Dict[str, Any]:
        from playwright.sync_api import sync_playwright

        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless, slow_mo=0)
        self.context = self.browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
            record_video_dir=str(self.artifact_dir),
            record_video_size={"width": self.viewport_width, "height": self.viewport_height},
        )
        self.page = self.context.new_page()
        self.page.set_default_timeout(15000)
        self.page.goto(form_url, wait_until="load", timeout=15000)
        self.engine = FormEngine(
            page=self.page,
            viewport={"width": self.viewport_width, "height": self.viewport_height},
            observations_dir=self.observations_dir,
            trace=self.trace,
            timeout_ms=15000,
            type_delay_ms=120,
            action_delay_ms=220,
            take_screenshots=True,
        )
        self.engine.enable_mouse_overlay()
        return self.page.evaluate(
            "() => ({devicePixelRatio: window.devicePixelRatio || null, userAgent: navigator.userAgent || null, locale: navigator.language || null, timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || null, url: window.location.href})"
        )

    def _capture_screenshot(self, path: Path, step_ref: Optional[int]) -> Optional[str]:
        _ = step_ref
        path.parent.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(path))
        return str(path)

    def _get_page_text(self, step_idx: int) -> str:
        _ = step_idx
        return self.engine.get_page_text()

    def execute_fill(self, entry: Dict[str, Any], step_idx: int) -> Tuple[Dict[str, Any], Optional[str]]:
        return self.engine.fill_step(entry, step_idx)

    def verify_entry(self, entry: Dict[str, Any], step_idx: int) -> Dict[str, Any]:
        _ = step_idx
        return self.engine.verify_entry(entry)

    def execute_wait(self, seconds: float, step_idx: int) -> None:
        _ = step_idx
        self.page.wait_for_timeout(max(int(seconds * 1000), 0))

    def execute_scroll(self, delta: int, step_idx: int) -> None:
        self.page.mouse.wheel(0, delta)
        self.trace.log_event("browser_mouse_wheel", {"deltaX": 0, "deltaY": delta}, step_ref=step_idx)

    def execute_press_key(self, key: str, step_idx: int) -> None:
        self.page.keyboard.press(key)
        self.trace.log_event("browser_press_key", {"key": key}, step_ref=step_idx)

    def submit(self) -> Tuple[Dict[str, Any], Optional[str]]:
        return self.engine.submit()

    def close(self) -> None:
        try:
            if self.page is not None:
                self.page.close()
        except Exception:
            pass
        try:
            if self.context is not None:
                self.context.close()
        except Exception:
            pass
        try:
            if self.browser is not None:
                self.browser.close()
        except Exception:
            pass
        try:
            if self.playwright is not None:
                self.playwright.stop()
        except Exception:
            pass


class MCPExecutionSession(ExecutionSessionBase):
    def __init__(
        self,
        artifact_dir: Path,
        observations_dir: Path,
        trace: TraceLogger,
        headless: bool,
        viewport_width: int,
        viewport_height: int,
        browser_mcp_cmd: Optional[str],
        browser_mcp_timeout_ms: int,
        browser_init_retries: int,
        browser_init_retry_delay_s: float,
    ) -> None:
        super().__init__(
            artifact_dir=artifact_dir,
            observations_dir=observations_dir,
            trace=trace,
            headless=headless,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
        )
        self.browser_mcp_cmd = browser_mcp_cmd
        self.browser_mcp_timeout_ms = browser_mcp_timeout_ms
        self.browser_init_retries = max(0, int(browser_init_retries))
        self.browser_init_retry_delay_s = max(0.0, float(browser_init_retry_delay_s))
        self.browser_mcp = None
        self.engine = None

    def _close_runtime(self) -> None:
        try:
            if self.engine is not None:
                self.engine.close()
        except Exception:
            pass
        self.engine = None
        try:
            if self.browser_mcp is not None:
                self.browser_mcp.close()
        except Exception:
            pass
        self.browser_mcp = None

    def start(self, form_url: str) -> Dict[str, Any]:
        required_tools = [
            "browser_navigate",
            "browser_run_code",
            "browser_wait_for",
            "browser_take_screenshot",
            "browser_close",
        ]
        command: Any = self.browser_mcp_cmd or _default_browser_mcp_command(
            self.viewport_width,
            self.viewport_height,
            self.artifact_dir,
            self.headless,
            self.browser_mcp_timeout_ms,
        )
        mcp_env = {"PLAYWRIGHT_BROWSERS_PATH": os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")}
        max_attempts = max(1, self.browser_init_retries + 1)
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                self.browser_mcp = MCPClient(
                    command=command,
                    timeout_ms=self.browser_mcp_timeout_ms,
                    required_tools=required_tools,
                    env=mcp_env,
                )
                self.engine = MCPBrowserEngine(
                    mcp_client=self.browser_mcp,
                    trace=self.trace,
                    observations_dir=self.observations_dir,
                    timeout_ms=15000,
                    type_delay_ms=120,
                    action_delay_ms=220,
                    take_screenshots=True,
                )
                env = self.engine.navigate(form_url)
                self.engine.enable_mouse_overlay()
                startup_screenshot = self.engine.take_observation_screenshot("startup_probe.png", step_ref=None)
                startup_text = self.engine.get_page_text(step_ref=None)
                if not startup_screenshot and not str(startup_text or "").strip():
                    raise RuntimeError("browser_mcp_startup_probe_failed")
                if isinstance(env, dict):
                    env = dict(env)
                    env["startup_probe"] = {
                        "screenshot_path": startup_screenshot,
                        "has_page_text": bool(str(startup_text or "").strip()),
                    }
                return env
            except Exception as exc:
                last_exc = exc
                raw = str(exc).lower()
                retryable = attempt < max_attempts and (
                    "snapshotforai" in raw
                    or "timeout" in raw
                    or "timed out" in raw
                )
                self._close_runtime()
                if retryable:
                    time.sleep(self.browser_init_retry_delay_s)
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("mcp_start_failed")

    def _capture_screenshot(self, path: Path, step_ref: Optional[int]) -> Optional[str]:
        return self.engine.capture_screenshot(path, step_ref)

    def _get_page_text(self, step_idx: int) -> str:
        return self.engine.get_page_text(step_idx)

    def execute_fill(self, entry: Dict[str, Any], step_idx: int) -> Tuple[Dict[str, Any], Optional[str]]:
        return self.engine.fill_step(entry, step_idx)

    def verify_entry(self, entry: Dict[str, Any], step_idx: int) -> Dict[str, Any]:
        return self.engine.verify_entry(entry, step_idx)

    def execute_wait(self, seconds: float, step_idx: int) -> None:
        self.engine.wait_seconds(seconds, step_ref=step_idx)

    def execute_scroll(self, delta: int, step_idx: int) -> None:
        code = f"""
async (page) => {{
  await page.mouse.wheel(0, {int(delta)});
  const out = \"THESIS_JSON:\" + JSON.stringify({{\"ok\": true}});
  console.log(out);
  return out;
}}
"""
        self.engine._run_code(code, purpose="scroll", step_ref=step_idx)

    def execute_press_key(self, key: str, step_idx: int) -> None:
        code = f"""
async (page) => {{
  await page.keyboard.press({json.dumps(str(key))});
  const out = \"THESIS_JSON:\" + JSON.stringify({{\"ok\": true}});
  console.log(out);
  return out;
}}
"""
        self.engine._run_code(code, purpose="press_key", step_ref=step_idx)

    def submit(self) -> Tuple[Dict[str, Any], Optional[str]]:
        return self.engine.submit()

    def close(self) -> None:
        self._close_runtime()


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model baseline evaluation for one model/form/run.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["text_llm", "vlm"], required=True)
    parser.add_argument("--form-id", required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--answers-root", default=DEFAULT_ANSWERS_ROOT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--logs-root", default="logs/baseline_eval")
    parser.add_argument("--config", default="configs/baselines/minimal_models.json")
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--trial-id")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--execution-backend", choices=["local", "mcp_server"], default=DEFAULT_EXECUTION_BACKEND)
    parser.add_argument(
        "--inference-backend",
        choices=["auto", "local_hf", "openai_compat"],
        default=DEFAULT_INFERENCE_BACKEND,
    )
    parser.add_argument("--api-timeout-s", type=int, default=DEFAULT_API_TIMEOUT_S)
    parser.add_argument("--browser-mcp-cmd")
    parser.add_argument("--browser-mcp-timeout-ms", type=int, default=DEFAULT_BROWSER_MCP_TIMEOUT_MS)
    parser.add_argument("--browser-init-retries", type=int, default=DEFAULT_BROWSER_INIT_RETRIES)
    parser.add_argument("--browser-init-retry-delay-s", type=float, default=DEFAULT_BROWSER_INIT_RETRY_DELAY_S)
    parser.add_argument("--invalid-action-budget", type=int, default=DEFAULT_INVALID_ACTION_BUDGET)
    parser.add_argument("--viewport-width", type=int, default=DEFAULT_VIEWPORT_WIDTH)
    parser.add_argument("--viewport-height", type=int, default=DEFAULT_VIEWPORT_HEIGHT)
    parser.add_argument("--require-gpu", action="store_true", default=False)
    parser.add_argument("--disable-action-coercion", dest="disable_action_coercion", action="store_true", default=True)
    parser.add_argument("--enable-action-coercion", dest="disable_action_coercion", action="store_false")
    parser.add_argument("--step-soft-timeout-s", type=float, default=DEFAULT_STEP_SOFT_TIMEOUT_S)
    parser.add_argument("--step-retry-max-new-tokens", type=int, default=DEFAULT_STEP_RETRY_MAX_NEW_TOKENS)
    parser.add_argument("--idle-step-threshold", type=int, default=DEFAULT_IDLE_STEP_THRESHOLD)
    parser.add_argument("--idle-nudge-max", type=int, default=DEFAULT_IDLE_NUDGE_MAX)
    parser.add_argument("--compact-page-text-max-chars", type=int, default=DEFAULT_COMPACT_PAGE_TEXT_MAX_CHARS)
    parser.add_argument(
        "--prompt-profile",
        choices=["legacy", "detailed_v1", "runtime_safe_v1"],
        default=DEFAULT_PROMPT_PROFILE,
    )
    parser.add_argument("--history-window", type=int, default=DEFAULT_HISTORY_WINDOW)
    parser.add_argument("--fewshot-enabled", dest="fewshot_enabled", action="store_true", default=True)
    parser.add_argument("--no-fewshot-enabled", dest="fewshot_enabled", action="store_false")
    parser.add_argument("--fewshot-count", type=int, default=DEFAULT_FEWSHOT_COUNT)
    parser.add_argument("--retention-window", type=int, default=DEFAULT_RETENTION_WINDOW)
    parser.add_argument("--run-label")
    return parser.parse_args(argv)


def _build_trial_paths(args: argparse.Namespace, model_id: str, form_id: str, answer_run_id: str, trial_id: str) -> Dict[str, Path]:
    experiment_root = (ROOT_DIR / args.dataset_root / args.experiment_id).resolve()
    artifact_dir = experiment_root / model_id / form_id / answer_run_id / trial_id
    observations_dir = artifact_dir / "observations"
    paths = {
        "experiment_root": experiment_root,
        "artifact_dir": artifact_dir,
        "observations_dir": observations_dir,
        "trace_path": artifact_dir / "tool_trace.jsonl",
        "answers_path": artifact_dir / "answers_instance.json",
        "annotations_path": artifact_dir / "annotations.json",
        "summary_path": artifact_dir / "summary.json",
        "model_io_path": artifact_dir / "model_io.jsonl",
        "step_inputs_path": artifact_dir / "step_inputs.jsonl",
        "manifest_path": experiment_root / "manifest.jsonl",
        "video_path": artifact_dir / f"{form_id}_{trial_id}.webm",
        "final_screenshot_path": artifact_dir / "final.png",
        "error_screenshot_path": artifact_dir / "error.png",
    }
    artifact_dir.mkdir(parents=True, exist_ok=False)
    observations_dir.mkdir(parents=True, exist_ok=True)
    return paths


def _artifact_payload(paths: Dict[str, Path]) -> Dict[str, Optional[str]]:
    return {
        "artifact_dir": str(paths["artifact_dir"]),
        "observations_dir": str(paths["observations_dir"]),
        "trace_path": str(paths["trace_path"]),
        "answers_path": str(paths["answers_path"]),
        "annotations_path": str(paths["annotations_path"]),
        "summary_path": str(paths["summary_path"]),
        "model_io_path": str(paths["model_io_path"]),
        "step_inputs_path": str(paths["step_inputs_path"]),
        "video_path": None,
        "final_screenshot_path": None,
        "error_screenshot_path": None,
    }


def _make_execution_session(args: argparse.Namespace, paths: Dict[str, Path], trace: TraceLogger):
    if args.execution_backend == "local":
        return LocalExecutionSession(
            artifact_dir=paths["artifact_dir"],
            observations_dir=paths["observations_dir"],
            trace=trace,
            headless=args.headless,
            viewport_width=args.viewport_width,
            viewport_height=args.viewport_height,
        )
    return MCPExecutionSession(
        artifact_dir=paths["artifact_dir"],
        observations_dir=paths["observations_dir"],
        trace=trace,
        headless=args.headless,
        viewport_width=args.viewport_width,
        viewport_height=args.viewport_height,
        browser_mcp_cmd=args.browser_mcp_cmd,
        browser_mcp_timeout_ms=args.browser_mcp_timeout_ms,
        browser_init_retries=args.browser_init_retries,
        browser_init_retry_delay_s=args.browser_init_retry_delay_s,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    args.idle_step_threshold = max(0, int(args.idle_step_threshold))
    args.idle_nudge_max = max(0, int(args.idle_nudge_max))
    args.compact_page_text_max_chars = max(500, int(args.compact_page_text_max_chars))
    args.history_window = max(0, int(args.history_window))
    args.fewshot_count = max(0, int(args.fewshot_count))
    args.retention_window = max(0, int(args.retention_window))
    args.api_timeout_s = max(1, int(args.api_timeout_s))
    args.browser_init_retries = max(0, int(args.browser_init_retries))
    args.browser_init_retry_delay_s = max(0.0, float(args.browser_init_retry_delay_s))
    args.prompt_profile = str(args.prompt_profile or DEFAULT_PROMPT_PROFILE).strip().lower()
    run_label = _make_run_label(args.run_label)
    run_started_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    config_path = (ROOT_DIR / args.config).resolve()
    model_cfg = get_model_by_id(config_path, args.model_id)
    if model_cfg.get("kind") != args.model_kind:
        raise ValueError(f"Model kind mismatch for {args.model_id}: expected {model_cfg.get('kind')}, got {args.model_kind}")
    provider = str(model_cfg.get("provider") or "").strip()
    if provider not in {"local_hf", "openai_compat"}:
        raise ValueError(f"Unsupported provider for run_baseline_eval: {args.model_id} provider={provider}")
    resolved_inference_backend = _select_inference_backend(model_cfg, args.inference_backend)
    args.inference_backend = resolved_inference_backend
    if resolved_inference_backend == "local_hf":
        _ensure_gpu_available(_requires_gpu(model_cfg, bool(args.require_gpu)))
        _ensure_model_runtime_compat(model_cfg)

    form_spec = load_form_spec(args.form_id, ROOT_DIR / "src" / "forms")
    form_url = str(form_spec.get("form_url") or form_spec.get("url") or "")
    if not form_url:
        raise ValueError(f"Missing form_url in spec for {args.form_id}")

    answers_path = resolve_answers_path(argparse.Namespace(answers_root=args.answers_root, answers_file="runs.json"), args.form_id)
    answers = _load_run_answers(answers_path, args.run_index)
    answer_run_id = f"run_{args.run_index:04d}"
    trial_id = args.trial_id or _make_trial_id()
    paths = _build_trial_paths(args, args.model_id, args.form_id, answer_run_id, trial_id)

    question_states = _build_question_states(answers)
    _write_json(paths["answers_path"], question_states)
    _touch(paths["model_io_path"])
    _touch(paths["step_inputs_path"])

    adapter = _make_adapter(
        model_cfg=model_cfg,
        model_kind=args.model_kind,
        max_new_tokens=args.max_new_tokens,
        inference_backend=resolved_inference_backend,
        api_timeout_s=args.api_timeout_s,
    )
    start_time = time.perf_counter()

    trace_mcp_client = None
    try:
        trace_mcp_client = MCPTraceClient(
            command=_default_mcp_server_command(),
            tool_name="record_action",
            timeout_ms=DEFAULT_TRACE_MCP_TIMEOUT_MS,
        )
    except Exception:
        trace_mcp_client = None

    trace = TraceLogger(
        path=paths["trace_path"],
        start_time=start_time,
        validate_mcp_actions=True,
        strict_mcp_validation=True,
        mcp_client=trace_mcp_client,
    )

    annotations: Dict[str, Any] = {
        "schema_version": BASELINE_EVAL_SCHEMA_VERSION,
        "experiment_id": args.experiment_id,
        "trial_id": trial_id,
        "run_label": run_label,
        "run_started_utc": run_started_utc,
        "model_id": args.model_id,
        "model_kind": args.model_kind,
        "track": _track_name(model_cfg),
        "provider": model_cfg.get("provider"),
        "inference_backend": args.inference_backend,
        "is_fallback_model": bool(model_cfg.get("is_fallback")),
        "fallback_for": model_cfg.get("fallback_for"),
        "form_id": args.form_id,
        "answer_run_id": answer_run_id,
        "form_url": form_url,
        "execution_backend": args.execution_backend,
        "prompt_mode": DEFAULT_PROMPT_MODE,
        "prompt_profile": args.prompt_profile,
        "context_package_version": CONTEXT_PACKAGE_VERSION,
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
        "idle_reprompts": 0,
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
            "inference_backend": args.inference_backend,
            "api_timeout_s": int(args.api_timeout_s),
            "require_gpu": bool(args.require_gpu),
            "disable_action_coercion": bool(args.disable_action_coercion),
            "step_soft_timeout_s": float(args.step_soft_timeout_s),
            "step_retry_max_new_tokens": int(args.step_retry_max_new_tokens),
            "idle_step_threshold": int(args.idle_step_threshold),
            "idle_nudge_max": int(args.idle_nudge_max),
            "compact_page_text_max_chars": int(args.compact_page_text_max_chars),
            "prompt_profile": args.prompt_profile,
            "history_window": int(args.history_window),
            "fewshot_enabled": bool(args.fewshot_enabled),
            "fewshot_count": int(args.fewshot_count),
            "retention_window": int(args.retention_window),
            "run_label": run_label,
        },
        "artifacts": _artifact_payload(paths),
        "trace": {},
        "environment": {},
        "steps": [],
        "questions": question_states,
        "failure_events": [],
    }

    _append_jsonl(
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
            "prompt_profile": args.prompt_profile,
            "browser_mcp_timeout_ms": args.browser_mcp_timeout_ms,
            "inference_backend": args.inference_backend,
            "api_timeout_s": int(args.api_timeout_s),
        },
    )

    execution_session = None
    last_result: Dict[str, Any] = {}
    terminal_screenshot_path: Optional[str] = None
    observation_cache: Dict[int, Dict[str, Any]] = {}
    idle_streak = 0
    idle_nudge_count = 0
    try:
        execution_session = _make_execution_session(args, paths, trace)
        annotations["environment"] = execution_session.start(form_url) or {}
        observation_cache[0] = execution_session.observe(0)
        initial_page_text = str(observation_cache[0].get("page_text") or "")
        initial_screenshot = observation_cache[0].get("screenshot_path")
        preflight_ok = bool(initial_page_text.strip()) or bool(initial_screenshot)
        annotations.setdefault("environment", {})["browser_preflight"] = {
            "ok": preflight_ok,
            "browser_mcp_timeout_ms": args.browser_mcp_timeout_ms,
            "has_page_text": bool(initial_page_text.strip()),
            "has_screenshot": bool(initial_screenshot),
        }
        _append_jsonl(
            paths["model_io_path"],
            {
                "phase": "backend_sanity",
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "status": "ok" if preflight_ok else "failed",
                "prompt_profile": args.prompt_profile,
                "browser_mcp_timeout_ms": args.browser_mcp_timeout_ms,
                "preflight_ok": preflight_ok,
                "screenshot_path": initial_screenshot,
                "page_text_excerpt": initial_page_text[:500],
            },
        )
        if not preflight_ok:
            raise RuntimeError("browser_mcp_preflight_failed: initial observation missing screenshot and page text")

        for step_idx in range(args.max_steps):
            elapsed = time.perf_counter() - start_time
            if elapsed >= args.timeout_s:
                annotations["stop_reason"] = "timeout"
                _set_failure(annotations, "timeout", f"timeout after {args.timeout_s}s", step_idx)
                break

            remaining_answers = _serialize_remaining_answers(question_states)
            visible_field_map = _serialize_visible_field_map(question_states)
            recent_history = _recent_history_from_steps(annotations.get("steps", []), args.history_window)
            validation_feedback = _normalize_validation_feedback(last_result)
            fewshot_ids = selected_canonical_fewshot_ids(enabled=bool(args.fewshot_enabled), count=int(args.fewshot_count))
            observation = observation_cache.pop(step_idx, None)
            if observation is None:
                observation = execution_session.observe(step_idx)
            page_text = str(observation.get("page_text") or "")
            screenshot_path = observation.get("screenshot_path")

            image_path = Path(screenshot_path) if screenshot_path else paths["observations_dir"] / f"step_{step_idx:04d}.png"
            behavior_nudge: Optional[str] = None
            if (
                remaining_answers
                and args.idle_step_threshold > 0
                and idle_streak >= args.idle_step_threshold
                and idle_nudge_count < args.idle_nudge_max
            ):
                idle_nudge_count += 1
                annotations["idle_reprompts"] = int(annotations.get("idle_reprompts", 0)) + 1
                behavior_nudge = _build_idle_recovery_nudge(
                    idle_streak=idle_streak,
                    remaining_answers=remaining_answers,
                    nudge_index=idle_nudge_count,
                    nudge_max=args.idle_nudge_max,
                    validation_feedback=validation_feedback,
                )

            if args.model_kind == "text_llm":
                prompt = build_text_prompt(
                    form_url,
                    remaining_answers,
                    page_text,
                    last_result,
                    behavior_nudge=behavior_nudge,
                    compact_page_text_max_chars=args.compact_page_text_max_chars,
                    prompt_profile=args.prompt_profile,
                    visible_field_map=visible_field_map,
                    recent_history=recent_history,
                    validation_feedback=validation_feedback,
                    fewshot_enabled=bool(args.fewshot_enabled),
                    fewshot_count=int(args.fewshot_count),
                )
            else:
                prompt = build_vlm_prompt(
                    form_url,
                    remaining_answers,
                    page_text,
                    last_result,
                    image_path,
                    behavior_nudge=behavior_nudge,
                    compact_page_text_max_chars=args.compact_page_text_max_chars,
                    prompt_profile=args.prompt_profile,
                    visible_field_map=visible_field_map,
                    recent_history=recent_history,
                    validation_feedback=validation_feedback,
                    fewshot_enabled=bool(args.fewshot_enabled),
                    fewshot_count=int(args.fewshot_count),
                )

            step_input_record = {
                "phase": "step_input",
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "step_index": step_idx,
                "form_url": form_url,
                "prompt_profile": args.prompt_profile,
                "context_package_version": CONTEXT_PACKAGE_VERSION,
                "remaining_answers": remaining_answers,
                "visible_field_map": visible_field_map,
                "recent_history": recent_history,
                "validation_feedback": validation_feedback,
                "fewshot_ids": fewshot_ids,
                "last_result": dict(last_result or {}),
                "page_text_excerpt": compact_page_text(page_text, max_chars=int(args.compact_page_text_max_chars)),
                "screenshot_path": screenshot_path,
                "behavior_nudge": behavior_nudge,
                "prompt_hash": _prompt_hash(prompt),
            }
            _append_jsonl(paths["step_inputs_path"], step_input_record)

            try:
                raw_output, infer_meta = _infer_with_retry(
                    adapter=adapter,
                    args=args,
                    model_kind=args.model_kind,
                    prompt=prompt,
                    image_path=image_path if args.model_kind != "text_llm" else None,
                )
            except Exception as exc:
                infer_error = f"model_inference_failed: {exc}"
                attempts = _extract_inference_attempts(exc)
                step_record = {
                    "step_index": step_idx,
                    "elapsed_s": round(time.perf_counter() - start_time, 3),
                    "prompt_mode": DEFAULT_PROMPT_MODE,
                    "prompt_profile": args.prompt_profile,
                    "remaining_answers_before": len(remaining_answers),
                    "page_text_excerpt": page_text[:2000],
                    "screenshot_path": screenshot_path,
                    "idle_streak_before": idle_streak,
                    "behavior_nudge": behavior_nudge,
                    "raw_model_output": None,
                    "action": None,
                    "warnings": [],
                    "status": "failed",
                    "error": infer_error,
                    "matched_question_id": None,
                    "target_match": None,
                    "execution": None,
                    "verification": None,
                    "model_inference": {"attempts": attempts},
                }
                io_record = {
                    "phase": "step",
                    "step_index": step_idx,
                    "elapsed_s": round(time.perf_counter() - start_time, 3),
                    "prompt_mode": DEFAULT_PROMPT_MODE,
                    "prompt_profile": args.prompt_profile,
                    "prompt": prompt,
                    "remaining_answers": remaining_answers,
                    "screenshot_path": screenshot_path,
                    "idle_streak_before": idle_streak,
                    "behavior_nudge": behavior_nudge,
                    "raw_model_output": None,
                    "parsed_action": None,
                    "warnings": [],
                    "error": infer_error,
                    "matched_question_id": None,
                    "target_match": None,
                    "execution": None,
                    "verification": None,
                    "model_inference": {"attempts": attempts},
                }
                annotations["steps"].append(step_record)
                _append_jsonl(paths["model_io_path"], io_record)
                annotations["stop_reason"] = "model_inference_failed"
                _set_failure(annotations, "model_inference_failed", str(exc), step_idx)
                break

            step_record: Dict[str, Any] = {
                "step_index": step_idx,
                "elapsed_s": round(time.perf_counter() - start_time, 3),
                "prompt_mode": DEFAULT_PROMPT_MODE,
                "prompt_profile": args.prompt_profile,
                "remaining_answers_before": len(remaining_answers),
                "page_text_excerpt": page_text[:2000],
                "screenshot_path": screenshot_path,
                "idle_streak_before": idle_streak,
                "behavior_nudge": behavior_nudge,
                "raw_model_output": raw_output,
                "action": None,
                "warnings": [],
                "status": None,
                "error": None,
                "matched_question_id": None,
                "target_match": None,
                "execution": None,
                "verification": None,
                "model_inference": infer_meta,
            }
            io_record: Dict[str, Any] = {
                "phase": "step",
                "step_index": step_idx,
                "elapsed_s": round(time.perf_counter() - start_time, 3),
                "prompt_mode": DEFAULT_PROMPT_MODE,
                "prompt_profile": args.prompt_profile,
                "prompt": prompt,
                "remaining_answers": remaining_answers,
                "screenshot_path": screenshot_path,
                "idle_streak_before": idle_streak,
                "behavior_nudge": behavior_nudge,
                "raw_model_output": raw_output,
                "parsed_action": None,
                "warnings": [],
                "error": None,
                "matched_question_id": None,
                "target_match": None,
                "execution": None,
                "verification": None,
                "model_inference": infer_meta,
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
                io_record["parse_error"] = str(exc)
                io_record["raw_model_output_len"] = len(str(raw_output or ""))
                _set_failure(annotations, "model_output_invalid", str(exc), step_idx)
                step_record["progress_made"] = False
                io_record["progress_made"] = False
                idle_streak += 1
                annotations["steps"].append(step_record)
                _append_jsonl(paths["model_io_path"], io_record)
                last_result = {"status": "failed", "error": message, "remaining_answers": len(remaining_answers)}
                if _invalid_action_budget_exhausted(annotations["invalid_actions"], args.invalid_action_budget):
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
                _set_failure(annotations, "premature_submit", detail, step_idx)
                step_record["progress_made"] = False
                io_record["progress_made"] = False
                idle_streak += 1
                annotations["steps"].append(step_record)
                _append_jsonl(paths["model_io_path"], io_record)
                last_result = {
                    "status": step_record["status"],
                    "error": step_record["error"],
                    "remaining_answers": len(_serialize_remaining_answers(question_states)),
                }
                continue

            if action_name == "wait":
                wait_seconds = max(0.25, float(action.get("delta") or 1000) / 1000.0)
                execution_session.execute_wait(wait_seconds, step_idx)
                step_record["status"] = "waited"
                io_record["execution"] = {"status": "waited", "seconds": wait_seconds}
            elif action_name == "scroll":
                delta = int(action.get("delta") or 600)
                execution_session.execute_scroll(delta, step_idx)
                step_record["status"] = "scrolled"
                io_record["execution"] = {"status": "scrolled", "delta": delta}
            elif action_name == "press_key":
                key = str(action.get("value") or "Tab")
                execution_session.execute_press_key(key, step_idx)
                step_record["status"] = "pressed_key"
                io_record["execution"] = {"status": "pressed_key", "key": key}
            elif action_name == "submit":
                submit_info, submit_err = execution_session.submit()
                step_record["execution"] = submit_info
                io_record["execution"] = submit_info
                if submit_err:
                    step_record["status"] = "failed"
                    step_record["error"] = f"submission_failed: {submit_err}"
                    annotations["stop_reason"] = "submission_failed"
                    _set_failure(annotations, "submission_failed", submit_err, step_idx)
                elif submit_info.get("success"):
                    step_record["status"] = "submitted"
                    annotations["success"] = True
                    annotations["submit_success"] = True
                    annotations["stop_reason"] = "submitted"
                else:
                    step_record["status"] = "failed"
                    step_record["error"] = "submission_failed: not confirmed"
                    annotations["stop_reason"] = "submission_failed"
                    _set_failure(annotations, "submission_failed", json.dumps(submit_info, ensure_ascii=True), step_idx)
            elif action_name == "done":
                step_record["status"] = "done"
                annotations["stop_reason"] = "done"
                io_record["execution"] = {"status": "done"}
            else:
                matched_idx, question_state, match_debug = _match_question_state(question_states, action.get("target", {}))
                step_record["target_match"] = match_debug
                io_record["target_match"] = match_debug
                if question_state is None or matched_idx is None:
                    detail = json.dumps(match_debug, ensure_ascii=True)
                    step_record["status"] = "failed"
                    step_record["error"] = "target_not_found"
                    _set_failure(annotations, "target_not_found", detail, step_idx)
                else:
                    action, coercion_warnings = _apply_action_policy(action, question_state, bool(args.disable_action_coercion))
                    if coercion_warnings:
                        step_record["warnings"] = list(dict.fromkeys(step_record["warnings"] + coercion_warnings))
                        io_record["warnings"] = list(dict.fromkeys(io_record["warnings"] + coercion_warnings))
                        step_record["action"] = action
                        io_record["parsed_action"] = action
                    resolved_action_name = action.get("action")
                    if not _action_supported_for_widget(str(resolved_action_name), str(question_state.get("widget_type") or "")):
                        step_record["status"] = "failed"
                        step_record["error"] = f"widget_interaction_failed: incompatible_action_for_widget:{question_state.get('widget_type')}"
                        _set_failure(annotations, "widget_interaction_failed", f"incompatible_action_for_widget:{question_state.get('widget_type')}", step_idx)
                        step_record["progress_made"] = False
                        io_record["progress_made"] = False
                        idle_streak += 1
                        annotations["steps"].append(step_record)
                        _append_jsonl(paths["model_io_path"], io_record)
                        last_result = {
                            "status": step_record["status"],
                            "error": step_record["error"],
                            "remaining_answers": len(_serialize_remaining_answers(question_states)),
                        }
                        continue
                    exec_entry = _build_entry_from_action(action, question_state)
                    question_state["attempted"] = True
                    step_record["matched_question_id"] = question_state.get("question_id")
                    io_record["matched_question_id"] = question_state.get("question_id")
                    action_result, exec_err = execution_session.execute_fill(exec_entry, step_idx)
                    verification_result = execution_session.verify_entry(question_state, step_idx)
                    question_state["last_execution"] = action_result
                    question_state["last_verification"] = verification_result
                    question_state["actual_value"] = verification_result.get("actual_value")
                    question_state["attempted_correct"] = _value_matches(question_state.get("value"), exec_entry.get("value"))
                    question_state["verified"] = bool(verification_result.get("verified"))
                    question_state["verified_correct"] = bool(verification_result.get("verified")) and _value_matches(
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
                    io_record["execution"] = action_result
                    io_record["verification"] = verification_result

                    if exec_err:
                        step_record["status"] = "failed"
                        step_record["error"] = f"widget_interaction_failed: {exec_err}"
                        _set_failure(annotations, "widget_interaction_failed", exec_err, step_idx)
                    elif question_state["verified"] and not question_state["verified_correct"]:
                        step_record["status"] = "filled_unverified"
                        step_record["error"] = "verification_failed"
                        _set_failure(
                            annotations,
                            "verification_failed",
                            json.dumps({"expected": question_state.get("value"), "actual": verification_result.get("actual_value")}, ensure_ascii=True),
                            step_idx,
                        )
                    elif not question_state["verified"]:
                        step_record["status"] = "filled_unverified"
                        step_record["error"] = f"verification_failed: {verification_result.get('detail')}"
                        _set_failure(annotations, "verification_failed", str(verification_result.get("detail")), step_idx)
                    else:
                        step_record["status"] = "filled"

            remaining_after = len(_serialize_remaining_answers(question_states))
            progress_made = remaining_after < len(remaining_answers) or step_record.get("status") in {"submitted", "done"}
            step_record["progress_made"] = bool(progress_made)
            io_record["progress_made"] = bool(progress_made)
            if progress_made:
                idle_streak = 0
            else:
                idle_streak += 1

            annotations["steps"].append(step_record)
            _append_jsonl(paths["model_io_path"], io_record)
            last_result = {
                "status": step_record["status"],
                "error": step_record["error"],
                "remaining_answers": len(_serialize_remaining_answers(question_states)),
            }

            if annotations["stop_reason"] in {"submitted", "submission_failed", "model_output_invalid", "model_inference_failed", "done"}:
                break

        if annotations["stop_reason"] is None:
            annotations["stop_reason"] = "max_steps_exceeded"
            _set_failure(annotations, "max_steps_exceeded", f"max_steps={args.max_steps}")
    except Exception as exc:
        annotations["stop_reason"] = "environment_error"
        category = _classify_environment_error(str(exc))
        _set_failure(annotations, category, str(exc))
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

    video_path = _finalize_trial_video(paths["artifact_dir"], paths["video_path"])
    if video_path is not None and video_path.exists():
        annotations["artifacts"]["video_path"] = str(video_path)
    if terminal_screenshot_path:
        if annotations.get("success"):
            annotations["artifacts"]["final_screenshot_path"] = terminal_screenshot_path
        else:
            annotations["artifacts"]["error_screenshot_path"] = terminal_screenshot_path

    metrics = _calculate_metrics(question_states)
    annotations.update(metrics)
    annotations["duration_s"] = round(time.perf_counter() - start_time, 3)
    annotations["trace"] = trace_summary
    run_completed_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    annotations["run_completed_utc"] = run_completed_utc

    summary = {
        "schema_version": BASELINE_SUMMARY_SCHEMA_VERSION,
        "experiment_id": args.experiment_id,
        "trial_id": trial_id,
        "run_label": run_label,
        "run_started_utc": run_started_utc,
        "run_completed_utc": run_completed_utc,
        "model_id": args.model_id,
        "model_kind": args.model_kind,
        "track": annotations.get("track"),
        "provider": annotations.get("provider"),
        "inference_backend": annotations.get("inference_backend"),
        "is_fallback_model": bool(annotations.get("is_fallback_model")),
        "fallback_for": annotations.get("fallback_for"),
        "form_id": args.form_id,
        "answer_run_id": answer_run_id,
        "execution_backend": args.execution_backend,
        "prompt_mode": DEFAULT_PROMPT_MODE,
        "prompt_profile": annotations.get("prompt_profile"),
        "context_package_version": annotations.get("context_package_version"),
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
        "invalid_actions": annotations["invalid_actions"],
        "idle_reprompts": annotations.get("idle_reprompts", 0),
        "duration_s": annotations["duration_s"],
        "artifacts": annotations["artifacts"],
    }

    _append_jsonl(
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
            "idle_reprompts": summary.get("idle_reprompts", 0),
        },
    )

    _write_json(paths["annotations_path"], annotations)
    _write_json(paths["summary_path"], summary)

    manifest_entry = {
        "experiment_id": args.experiment_id,
        "trial_id": trial_id,
        "run_label": run_label,
        "run_started_utc": run_started_utc,
        "run_completed_utc": run_completed_utc,
        "model_id": args.model_id,
        "model_kind": args.model_kind,
        "track": annotations.get("track"),
        "provider": annotations.get("provider"),
        "inference_backend": annotations.get("inference_backend"),
        "is_fallback_model": bool(annotations.get("is_fallback_model")),
        "fallback_for": annotations.get("fallback_for"),
        "prompt_profile": annotations.get("prompt_profile"),
        "context_package_version": annotations.get("context_package_version"),
        "form_id": args.form_id,
        "answer_run_id": answer_run_id,
        "success": summary["success"],
        "submit_success": summary["submit_success"],
        "stop_reason": summary["stop_reason"],
        "failure_category": summary["failure_category"],
        "failure_detail": summary["failure_detail"],
        "summary_path": str(paths["summary_path"]),
        "annotations_path": str(paths["annotations_path"]),
        "trace_path": str(paths["trace_path"]),
        "model_io_path": str(paths["model_io_path"]),
        "step_inputs_path": str(paths["step_inputs_path"]),
        "video_path": annotations["artifacts"]["video_path"],
        "artifact_dir": str(paths["artifact_dir"]),
    }
    _append_jsonl(paths["manifest_path"], manifest_entry)
    _update_experiment_indexes(
        experiment_root=paths["experiment_root"],
        manifest_entry=manifest_entry,
        run_label=run_label,
        retention_window=args.retention_window,
    )

    print(f"[INFO] wrote baseline summary: {paths['summary_path']}")
    print(f"[INFO] wrote baseline annotations: {paths['annotations_path']}")
    print(f"[INFO] wrote baseline manifest: {paths['manifest_path']}")
    print(f"[INFO] stop_reason: {summary['stop_reason']}")
    print(f"[INFO] prompt_profile: {summary.get('prompt_profile')}")
    print(f"[INFO] inference_backend: {summary.get('inference_backend')}")
    print(f"[INFO] browser_mcp_timeout_ms: {annotations.get('run_params', {}).get('browser_mcp_timeout_ms')}")
    print(f"[INFO] failure_category: {summary['failure_category']}")
    print(f"[INFO] success: {summary['success']}")
    print(f"[INFO] submit_success: {summary['submit_success']}")
    print(f"[INFO] attempted_correctness: {summary['attempted_correctness']}/{summary['question_total']}")
    print(f"[INFO] verified_correctness: {summary['verified_correctness']}/{summary['question_total']}")
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
