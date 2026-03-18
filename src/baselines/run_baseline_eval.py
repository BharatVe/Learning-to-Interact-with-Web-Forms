import argparse
import json
import os
import re
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
from baselines.model_adapters.local_text import LocalTextAdapter  # noqa: E402
from baselines.model_adapters.local_vlm import LocalVLMAdapter  # noqa: E402
from baselines.model_registry import get_model_by_id  # noqa: E402
from baselines.prompt_builders import build_text_prompt, build_vlm_prompt  # noqa: E402
from engine.form_engine import FormEngine  # noqa: E402
from engine.runner import iter_run_specs, load_form_spec, resolve_answers_path  # noqa: E402
from engine.trace_logger import TraceLogger  # noqa: E402


DEFAULT_ANSWERS_ROOT = "data/answers"
DEFAULT_DATASET_ROOT = "data/baseline_eval"
DEFAULT_MAX_STEPS = 60
DEFAULT_TIMEOUT_S = 180
DEFAULT_INVALID_ACTION_BUDGET = 3
DEFAULT_LOGS_ROOT = "logs/baseline_eval"
DEFAULT_SCREENSHOT_NAME = "model_step_{step:04d}.png"


def _norm_text(value: str) -> str:
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


def _get_page_text(page) -> str:
    try:
        return page.locator("body").inner_text(timeout=3000)
    except Exception:
        return ""


def _match_remaining_entry(remaining: List[Dict[str, Any]], target: Dict[str, Any]) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    labels = [target.get("label"), target.get("text"), target.get("selector_hint")]
    candidates = [_norm_text(item) for item in labels if isinstance(item, str) and item.strip()]
    if not candidates:
        return None, None

    for idx, entry in enumerate(remaining):
        label_norm = _norm_text(entry.get("label", ""))
        if any(candidate == label_norm for candidate in candidates):
            return idx, entry
    for idx, entry in enumerate(remaining):
        label_norm = _norm_text(entry.get("label", ""))
        if any(candidate in label_norm or label_norm in candidate for candidate in candidates):
            return idx, entry
    return None, None


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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_adapter(model_cfg: Dict[str, Any], max_new_tokens: int):
    model_dir = ROOT_DIR / "models" / str(model_cfg["id"])
    kind = model_cfg.get("kind")
    if kind == "text_llm":
        return LocalTextAdapter(model_dir=model_dir, max_new_tokens=max_new_tokens)
    if kind == "vlm":
        return LocalVLMAdapter(model_dir=model_dir, max_new_tokens=max_new_tokens)
    raise ValueError(f"Unsupported model kind for local baseline: {kind}")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local baseline evaluation for one model/form/run.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["text_llm", "vlm"], required=True)
    parser.add_argument("--form-id", required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--answers-root", default=DEFAULT_ANSWERS_ROOT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--logs-root", default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--config", default="configs/baselines/minimal_models.json")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--headless", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    config_path = (ROOT_DIR / args.config).resolve()
    model_cfg = get_model_by_id(config_path, args.model_id)
    if model_cfg.get("kind") != args.model_kind:
        raise ValueError(f"Model kind mismatch for {args.model_id}: expected {model_cfg.get('kind')}, got {args.model_kind}")
    if model_cfg.get("provider") != "local_hf":
        raise ValueError(f"Only local_hf models are supported in this runner: {args.model_id}")

    form_spec = load_form_spec(args.form_id, ROOT_DIR / "src" / "forms")
    form_url = str(form_spec.get("form_url") or form_spec.get("url") or "")
    if not form_url:
        raise ValueError(f"Missing form_url in spec for {args.form_id}")

    answers_path = resolve_answers_path(argparse.Namespace(answers_root=args.answers_root, answers_file="runs.json"), args.form_id)
    answers = _load_run_answers(answers_path, args.run_index)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_name = f"run_{args.run_index:04d}"
    artifact_dir = (ROOT_DIR / args.dataset_root / args.model_id / args.form_id / f"{run_name}_{timestamp}").resolve()
    obs_dir = artifact_dir / "observations"
    trace_path = artifact_dir / "tool_trace.jsonl"
    result_path = (ROOT_DIR / args.logs_root / f"{timestamp}_{args.model_id}_{args.form_id}_{run_name}.json").resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    obs_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "timestamp_utc": timestamp,
        "model_id": args.model_id,
        "model_kind": args.model_kind,
        "prompt_mode": "baseline1_single_action",
        "form_id": args.form_id,
        "run_index": args.run_index,
        "form_url": form_url,
        "success": False,
        "stop_reason": None,
        "invalid_actions": 0,
        "action_count": 0,
        "question_correctness": 0,
        "question_total": len(answers),
        "steps": [],
        "artifacts": {
            "artifact_dir": str(artifact_dir),
            "trace_path": str(trace_path),
            "video_path": None,
            "answers_path": str(answers_path),
        },
        "environment": {
            "headless": bool(args.headless),
            "timeout_s": args.timeout_s,
            "max_steps": args.max_steps,
        },
        "duration_s": None,
    }

    adapter = _make_adapter(model_cfg, args.max_new_tokens)
    remaining_answers = [dict(item) for item in answers]
    last_result: Dict[str, Any] = {}
    start_time = time.perf_counter()
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT_DIR / ".playwright-browsers"))

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        result["stop_reason"] = "environment_error"
        result["error"] = str(exc)
        _write_json(result_path, result)
        return 1

    trace = TraceLogger(path=trace_path, start_time=start_time, validate_mcp_actions=True, strict_mcp_validation=True)
    browser = None
    context = None
    page = None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=args.headless, slow_mo=0)
            context = browser.new_context(
                viewport={"width": 1280, "height": 720},
                record_video_dir=str(artifact_dir),
                record_video_size={"width": 1280, "height": 720},
            )
            page = context.new_page()
            page.set_default_timeout(15000)
            page.goto(form_url, wait_until="load", timeout=15000)
            engine = FormEngine(
                page=page,
                viewport={"width": 1280, "height": 720},
                observations_dir=obs_dir,
                trace=trace,
                timeout_ms=15000,
                type_delay_ms=120,
                action_delay_ms=220,
                take_screenshots=True,
            )

            for step_idx in range(args.max_steps):
                elapsed = time.perf_counter() - start_time
                if elapsed >= args.timeout_s:
                    result["stop_reason"] = "timeout"
                    break
                if not remaining_answers:
                    break

                page_text = _get_page_text(page)
                screenshot_path = obs_dir / DEFAULT_SCREENSHOT_NAME.format(step=step_idx)
                page.screenshot(path=str(screenshot_path))

                if args.model_kind == "text_llm":
                    prompt = build_text_prompt(form_url, remaining_answers, page_text, last_result)
                    print(f"[INFO] Model step {step_idx}: generating text action")
                    raw_output = adapter.infer(prompt)
                else:
                    prompt = build_vlm_prompt(form_url, remaining_answers, page_text, last_result, screenshot_path)
                    print(f"[INFO] Model step {step_idx}: generating VLM action")
                    raw_output = adapter.infer(prompt, screenshot_path)

                step_record: Dict[str, Any] = {
                    "step_index": step_idx,
                    "raw_model_output": raw_output,
                    "prompt_excerpt": prompt[:2000],
                    "screenshot_path": str(screenshot_path),
                    "status": None,
                    "action": None,
                    "warnings": [],
                    "error": None,
                }

                try:
                    parsed = parse_action(raw_output)
                    action, warnings = validate_action(parsed)
                    print(f"[INFO] Model step {step_idx}: action={action['action']}")
                    step_record["action"] = action
                    step_record["warnings"] = warnings
                except Exception as exc:
                    result["invalid_actions"] += 1
                    step_record["status"] = "invalid"
                    step_record["error"] = f"model_output_invalid: {exc}"
                    result["steps"].append(step_record)
                    last_result = {"status": "invalid", "error": step_record["error"]}
                    if result["invalid_actions"] >= DEFAULT_INVALID_ACTION_BUDGET:
                        result["stop_reason"] = "invalid_action_budget"
                        break
                    continue

                action_name = action["action"]
                if action_name == "wait":
                    wait_ms = max(250, int(action.get("delta") or 1000))
                    page.wait_for_timeout(wait_ms)
                    step_record["status"] = "waited"
                elif action_name == "scroll":
                    delta = int(action.get("delta") or 600)
                    page.mouse.wheel(0, delta)
                    trace.log_event("browser_mouse_wheel", {"deltaX": 0, "deltaY": delta}, step_ref=step_idx)
                    step_record["status"] = "scrolled"
                elif action_name == "press_key":
                    key = str(action.get("value") or "Tab")
                    page.keyboard.press(key)
                    trace.log_event("browser_press_key", {"key": key}, step_ref=step_idx)
                    step_record["status"] = "pressed_key"
                elif action_name == "submit":
                    submit_info, submit_err = engine.submit()
                    step_record["submit"] = submit_info
                    if submit_err:
                        step_record["status"] = "failed"
                        step_record["error"] = f"submission_failed: {submit_err}"
                        result["stop_reason"] = "submission_failed"
                    elif submit_info.get("success"):
                        step_record["status"] = "submitted"
                        result["success"] = True
                        result["stop_reason"] = "submitted"
                    else:
                        step_record["status"] = "failed"
                        step_record["error"] = "submission_failed: not confirmed"
                        result["stop_reason"] = "submission_failed"
                        result["steps"].append(step_record)
                        break
                elif action_name == "done":
                    step_record["status"] = "done"
                    result["stop_reason"] = "done"
                    result["steps"].append(step_record)
                    break
                else:
                    remaining_idx, expected_entry = _match_remaining_entry(remaining_answers, action.get("target", {}))
                    if expected_entry is None or remaining_idx is None:
                        step_record["status"] = "failed"
                        step_record["error"] = "target_not_found"
                    else:
                        exec_entry = _build_entry_from_action(action, expected_entry)
                        action_result, err = engine.fill_step(exec_entry, step_idx)
                        step_record["executor_action"] = action_result
                        step_record["expected_label"] = expected_entry.get("label")
                        step_record["expected_value"] = expected_entry.get("value")
                        step_record["executed_value"] = exec_entry.get("value")
                        if err:
                            step_record["status"] = "failed"
                            step_record["error"] = f"widget_interaction_failed: {err}"
                        else:
                            step_record["status"] = "filled"
                            if _value_matches(expected_entry.get("value"), exec_entry.get("value")):
                                result["question_correctness"] += 1
                            remaining_answers.pop(remaining_idx)

                result["action_count"] += 1
                result["steps"].append(step_record)
                last_result = {
                    "status": step_record["status"],
                    "error": step_record["error"],
                    "remaining_answers": len(remaining_answers),
                }

                if result["stop_reason"] in {"submitted", "submission_failed", "invalid_action_budget", "done"}:
                    break

            if result["stop_reason"] is None:
                if not remaining_answers:
                    submit_info, submit_err = engine.submit()
                    result["steps"].append(
                        {
                            "step_index": len(result["steps"]),
                            "status": "submitted" if submit_info.get("success") and not submit_err else "failed",
                            "submit": submit_info,
                            "error": None if not submit_err else f"submission_failed: {submit_err}",
                        }
                    )
                    if submit_info.get("success") and not submit_err:
                        result["success"] = True
                        result["stop_reason"] = "submitted"
                    else:
                        result["stop_reason"] = "submission_failed"
                else:
                    result["stop_reason"] = "max_steps_exceeded"
    except Exception as exc:
        result["stop_reason"] = "environment_error"
        result["error"] = str(exc)
    finally:
        try:
            if page is not None:
                page.close()
        except Exception:
            pass
        try:
            if context is not None:
                context.close()
        except Exception:
            pass
        try:
            if browser is not None:
                browser.close()
        except Exception:
            pass
        trace.close()

    try:
        videos = sorted(artifact_dir.rglob("*.webm"), key=lambda item: item.stat().st_mtime, reverse=True)
    except Exception:
        videos = []
    if videos:
        result["artifacts"]["video_path"] = str(videos[0])

    result["duration_s"] = round(time.perf_counter() - start_time, 3)
    _write_json(result_path, result)
    print(f"[INFO] wrote baseline eval result: {result_path}")
    print(f"[INFO] stop_reason: {result['stop_reason']}")
    print(f"[INFO] success: {result['success']}")
    print(f"[INFO] question_correctness: {result['question_correctness']}/{result['question_total']}")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
