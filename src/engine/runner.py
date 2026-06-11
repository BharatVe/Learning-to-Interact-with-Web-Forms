import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from engine.form_engine import FormEngine  # noqa: E402
from engine.browser_language import english_context_options, force_english_google_forms_url, write_playwright_mcp_english_config  # noqa: E402
from engine.mcp_browser_engine import MCPBrowserEngine  # noqa: E402
from engine.mcp_trace_client import MCPClient  # noqa: E402
from engine.mcp_trace_client import MCPTraceClient  # noqa: E402
from engine.trace_contract import supported_actions  # noqa: E402
from engine.trace_logger import TraceLogger  # noqa: E402

DEFAULT_DATASET_ROOT = "data/forms"
DEFAULT_SLOW_MO = 200
DEFAULT_TIMEOUT_MS = 15000
DEFAULT_TYPE_DELAY_MS = 120
DEFAULT_ACTION_DELAY_MS = 220
DEFAULT_POST_SUBMIT_DELAY_SECONDS = 0.0
DEFAULT_ANSWERS_ROOT = "data/answers"
DEFAULT_ANSWERS_FILE = "runs.json"
DEFAULT_TRACE_MODE = "mcp"
DEFAULT_MCP_TOOL_NAME = "record_action"
DEFAULT_MCP_TIMEOUT_MS = 5000
DEFAULT_BROWSER_MCP_TIMEOUT_MS = 120000
DEFAULT_MCP_BROWSER_INSTALL_TIMEOUT_SECONDS = 600

RUN_DIR_PATTERN = re.compile(r"run_(\d{4})$")


def _default_mcp_server_command() -> List[str]:
    return [sys.executable, str(SRC_DIR / "engine" / "mcp_trace_server.py")]


def _default_browser_mcp_command(args: argparse.Namespace, run_dir: Path) -> List[str]:
    explicit_executable = os.environ.get("PLAYWRIGHT_MCP_CHROMIUM_EXECUTABLE", "").strip()
    python_playwright_executable = "" if explicit_executable else _detect_python_playwright_chromium_executable()
    mcp_bin = shutil.which("playwright-mcp")
    timeout_ms = max(15000, int(getattr(args, "browser_mcp_timeout_ms", DEFAULT_BROWSER_MCP_TIMEOUT_MS)))
    command = (
        [mcp_bin]
        if mcp_bin
        else ["npx", "-y", "@playwright/mcp@latest"]
    ) + [
        "--config",
        str(write_playwright_mcp_english_config(run_dir)),
        "--browser",
        "chromium",
        "--isolated",
        "--host",
        "127.0.0.1",
        "--output-dir",
        str(run_dir),
        "--save-video",
        f"{args.viewport_width}x{args.viewport_height}",
        "--viewport-size",
        f"{args.viewport_width},{args.viewport_height}",
        "--snapshot-mode",
        "none",
        "--timeout-action",
        str(timeout_ms),
        "--timeout-navigation",
        str(max(60000, timeout_ms)),
    ]
    if explicit_executable:
        command.extend(["--executable-path", explicit_executable])
    elif python_playwright_executable:
        command.extend(["--executable-path", python_playwright_executable])
    if args.headless:
        command.append("--headless")
    if _is_wsl():
        command.append("--no-sandbox")
    return command


def _default_playwright_browsers_path() -> str:
    env_value = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if env_value:
        return env_value
    return str(ROOT_DIR / ".playwright-browsers-node")


def _is_wsl() -> bool:
    try:
        data = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False
    return "microsoft" in data or "wsl" in data


def _detect_python_playwright_chromium_executable() -> Optional[str]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return None
    try:
        with sync_playwright() as p:
            path = p.chromium.executable_path
    except Exception:
        return None
    if not path:
        return None
    path_str = str(path)
    return path_str if Path(path_str).exists() else None


def _has_any_chromium_cache() -> bool:
    return _find_cached_chromium_executable() is not None


def _find_cached_chromium_executable() -> Optional[Path]:
    cache_dir = Path(_default_playwright_browsers_path())
    if not cache_dir.exists():
        return None
    for entry in cache_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith("chromium-"):
            continue
        candidates = [
            entry / "chrome-linux" / "chrome",
            entry / "chrome-linux64" / "chrome",
            entry / "chrome-linux" / "headless_shell",
            entry / "chrome-linux64" / "headless_shell",
        ]
        for candidate in candidates:
            if candidate.exists() and os.access(candidate, os.X_OK):
                return candidate
    return None


def ensure_playwright_mcp_package(timeout_seconds: int) -> None:
    if shutil.which("playwright-mcp"):
        return
    cmd = ["npx", "-y", "@playwright/mcp@latest", "--version"]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=max(30, timeout_seconds),
    )
    if proc.returncode == 0:
        return
    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    details = stderr or stdout or "no output"
    raise RuntimeError(
        "Failed to resolve official Playwright MCP package via npx. "
        f"Command: {' '.join(cmd)}. Details: {details}"
    )


def ensure_node_playwright_browser(browser_name: str, timeout_seconds: int) -> None:
    if browser_name != "chromium":
        return
    cmd = ["npx", "-y", "playwright@latest", "install", browser_name]
    env = dict(os.environ)
    env.setdefault("PLAYWRIGHT_BROWSERS_PATH", _default_playwright_browsers_path())
    print(f"[INFO] Installing Node Playwright browser for MCP: {browser_name}")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=max(60, timeout_seconds),
        env=env,
    )
    if proc.returncode == 0:
        return
    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    details = stderr or stdout or "no output"
    if _has_any_chromium_cache():
        print(
            "[WARN] Node Playwright browser install failed, but existing Chromium cache was found. "
            f"Proceeding with cached browser. Details: {details}",
            file=sys.stderr,
        )
        return
    raise RuntimeError(
        f"Failed to install Node Playwright browser '{browser_name}' for MCP. "
        f"Command: {' '.join(cmd)}. PLAYWRIGHT_BROWSERS_PATH={env.get('PLAYWRIGHT_BROWSERS_PATH')}. "
        f"Details: {details}"
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate form datasets via Playwright")
    parser.add_argument("--form-id")
    parser.add_argument("--all-forms", action="store_true")
    parser.add_argument("--smoke-test-all-forms", action="store_true")
    parser.add_argument("--form-url")
    parser.add_argument("--answers-root", default=DEFAULT_ANSWERS_ROOT)
    parser.add_argument("--answers-file", default=DEFAULT_ANSWERS_FILE)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--specs-root")
    parser.add_argument("--num-runs", type=int)
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-existing-video", action="store_true")
    parser.add_argument("--overwrite-missing-video", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument(
        "--continue-on-run-error",
        action="store_true",
        default=False,
        help="Continue all-form/batch generation after an individual run fails; failed runs write failure_manifest.json.",
    )
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--slow-mo", type=int, default=DEFAULT_SLOW_MO)
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--viewport-height", type=int, default=720)
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--type-delay-ms", type=int, default=DEFAULT_TYPE_DELAY_MS)
    parser.add_argument("--action-delay-ms", type=int, default=DEFAULT_ACTION_DELAY_MS)
    parser.add_argument(
        "--post-submit-delay-seconds",
        type=float,
        default=DEFAULT_POST_SUBMIT_DELAY_SECONDS,
    )
    parser.add_argument("--screenshots", action="store_true", default=False)
    parser.add_argument("--no-mouse-overlay", action="store_true", default=False)
    parser.add_argument("--interaction-mode", choices=["local", "mcp_server"], default="local")
    parser.add_argument("--trace-mode", choices=["mcp", "local"], default=DEFAULT_TRACE_MODE)
    parser.add_argument("--mcp-server-cmd")
    parser.add_argument("--mcp-tool-name", default=DEFAULT_MCP_TOOL_NAME)
    parser.add_argument("--mcp-timeout-ms", type=int, default=DEFAULT_MCP_TIMEOUT_MS)
    parser.add_argument("--browser-mcp-cmd")
    parser.add_argument("--browser-mcp-timeout-ms", type=int, default=DEFAULT_BROWSER_MCP_TIMEOUT_MS)
    parser.add_argument(
        "--no-mcp-browser-install",
        action="store_true",
        default=False,
        help="Disable automatic Node Playwright browser install preflight for mcp_server mode.",
    )
    parser.add_argument(
        "--mcp-browser-install-timeout-s",
        type=int,
        default=DEFAULT_MCP_BROWSER_INSTALL_TIMEOUT_SECONDS,
    )
    parser.add_argument("--no-mcp-verify-trace", action="store_true", default=False)
    parser.add_argument("--no-mcp-strict", action="store_true", default=False)
    args = parser.parse_args(argv)
    if args.form_id and args.all_forms:
        parser.error("Use either --form-id or --all-forms, not both")
    if args.form_id and args.smoke_test_all_forms:
        parser.error("Use either --form-id or --smoke-test-all-forms, not both")
    if not args.form_id and not args.all_forms and not args.smoke_test_all_forms:
        parser.error("Either --form-id, --all-forms, or --smoke-test-all-forms is required")
    if (args.all_forms or args.smoke_test_all_forms) and args.form_url:
        parser.error("--form-url is only supported with --form-id")
    if args.smoke_test_all_forms and args.num_runs is not None:
        parser.error("--num-runs is not supported with --smoke-test-all-forms (it always runs 1)")
    if args.num_runs is not None and args.num_runs < 1:
        parser.error("--num-runs must be >= 1")
    if args.timeout_ms < 1000:
        parser.error("--timeout-ms must be >= 1000")
    if args.type_delay_ms < 0:
        parser.error("--type-delay-ms must be >= 0")
    if args.action_delay_ms < 0:
        parser.error("--action-delay-ms must be >= 0")
    if args.post_submit_delay_seconds < 0:
        parser.error("--post-submit-delay-seconds must be >= 0")
    if args.mcp_timeout_ms < 100:
        parser.error("--mcp-timeout-ms must be >= 100")
    if args.browser_mcp_timeout_ms < 100:
        parser.error("--browser-mcp-timeout-ms must be >= 100")
    if args.mcp_browser_install_timeout_s < 30:
        parser.error("--mcp-browser-install-timeout-s must be >= 30")
    return args


def load_form_spec(form_id: str, specs_root: Path) -> Dict[str, Any]:
    spec_path = specs_root / form_id / "spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")
    data = json.loads(spec_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Spec file must contain an object: {spec_path}")
    return data


def discover_form_ids(specs_root: Path) -> List[str]:
    if not specs_root.exists():
        raise FileNotFoundError(f"Specs root not found: {specs_root}")
    form_ids: List[str] = []
    for entry in specs_root.iterdir():
        if entry.is_dir() and (entry / "spec.json").exists():
            form_ids.append(entry.name)
    return sorted(form_ids)


def resolve_answers_path(args: argparse.Namespace, form_id: str) -> Path:
    form_answers_dir = (Path(args.answers_root) / form_id).resolve()
    if not form_answers_dir.exists() or not form_answers_dir.is_dir():
        raise FileNotFoundError(
            f"Answers directory not found for '{form_id}': {form_answers_dir}. "
            f"Expected convention: {args.answers_root}/<form_id>/"
        )

    candidates: List[str] = []
    for name in [args.answers_file, "runs.json", "runs.jsonl", "runs.ndjson"]:
        if name not in candidates:
            candidates.append(name)

    for name in candidates:
        candidate = form_answers_dir / name
        if candidate.exists() and candidate.is_file():
            return candidate

    available = sorted(
        p.name for p in form_answers_dir.iterdir() if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".ndjson"}
    )
    expected = ", ".join(candidates)
    found = ", ".join(available) if available else "(none)"
    raise FileNotFoundError(
        f"No valid answers file found for '{form_id}' in {form_answers_dir}. "
        f"Tried: {expected}. Found: {found}"
    )


def normalize_run_spec(run_obj: Any, index_label: str) -> Dict[str, Any]:
    if isinstance(run_obj, list):
        answers = run_obj
        metadata = {}
    elif isinstance(run_obj, dict):
        if "answers" in run_obj:
            answers = run_obj.get("answers")
            metadata = {k: v for k, v in run_obj.items() if k != "answers"}
        else:
            raise ValueError(f"Run {index_label} missing answers list")
    else:
        raise ValueError(f"Run {index_label} must be a list or object")
    if not isinstance(answers, list):
        raise ValueError(f"Run {index_label} answers must be a list")
    return {"answers": answers, "metadata": metadata}


def iter_json_runs(path: Path) -> Iterator[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        yield normalize_run_spec(data, "0")
        return
    if not isinstance(data, dict):
        raise ValueError("answers file must be a list or an object containing runs")
    runs = data.get("runs")
    if runs is None:
        runs = data.get("multi_runs")
    if runs is None and "answers" in data:
        yield normalize_run_spec(data, "0")
        return
    if not isinstance(runs, list):
        raise ValueError("runs must be a list in answers file")
    for idx, run in enumerate(runs):
        yield normalize_run_spec(run, str(idx))


def iter_jsonl_runs(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            run_obj = json.loads(stripped)
            yield normalize_run_spec(run_obj, f"line {line_number}")


def iter_run_specs(path: Path) -> Iterator[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return iter_jsonl_runs(path)
    return iter_json_runs(path)


def existing_run_indices(runs_dir: Path) -> List[int]:
    if not runs_dir.exists():
        return []
    indices: List[int] = []
    for entry in runs_dir.iterdir():
        if not entry.is_dir():
            continue
        match = RUN_DIR_PATTERN.match(entry.name)
        if match:
            indices.append(int(match.group(1)))
    return indices


def next_available_index(existing: set, start_index: int) -> int:
    index = max(1, start_index)
    while index in existing:
        index += 1
    return index


def ensure_run_dir(
    runs_dir: Path,
    run_index: int,
    skip_existing: bool,
    existing_indices: set,
    skip_existing_video: bool,
    overwrite_missing_video: bool,
    overwrite_existing: bool,
) -> Tuple[Optional[Path], str, int]:
    run_name = f"run_{run_index:04d}"
    run_dir = runs_dir / run_name
    if run_dir.exists():
        existing_indices.add(run_index)
        if overwrite_existing:
            shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir, run_name, run_index
        has_video = any(run_dir.rglob("*.webm"))
        if skip_existing_video and has_video:
            return None, run_name, run_index
        if overwrite_missing_video and not has_video:
            shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir, run_name, run_index
        if skip_existing:
            return ensure_run_dir(
                runs_dir,
                run_index + 1,
                skip_existing,
                existing_indices,
                skip_existing_video,
                overwrite_missing_video,
                overwrite_existing,
            )
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    existing_indices.add(run_index)
    return run_dir, run_name, run_index


def finalize_video(run_dir: Path, form_id: str, run_name: str) -> Optional[Path]:
    try:
        candidate_videos = sorted(
            run_dir.rglob("*.webm"), key=lambda p: p.stat().st_mtime, reverse=True
        )
    except Exception:
        candidate_videos = []
    if not candidate_videos:
        return None
    raw_video_path = candidate_videos[0]
    final_video_path = run_dir / f"{form_id}_{run_name}.webm"
    if raw_video_path == final_video_path:
        return raw_video_path
    if final_video_path.exists():
        final_video_path.unlink()
    try:
        raw_video_path.rename(final_video_path)
        return final_video_path
    except Exception:
        return raw_video_path


def _load_trace_records(trace_path: Path) -> List[Dict[str, Any]]:
    if not trace_path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSON in tool trace at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise RuntimeError(f"Invalid trace record type at line {line_number}: {type(record).__name__}")
            records.append(record)
    return records


def _count_events_by_name(trace_records: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in trace_records:
        name = record.get("name")
        if isinstance(name, str) and name:
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _step_event_counts(trace_records: List[Dict[str, Any]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for record in trace_records:
        step_ref = record.get("step_ref")
        if isinstance(step_ref, int):
            counts[step_ref] = counts.get(step_ref, 0) + 1
    return dict(sorted(counts.items()))


def _build_information_layers(annotations: Dict[str, Any], trace_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    actions = annotations.get("actions")
    action_list = actions if isinstance(actions, list) else []
    step_counts = _step_event_counts(trace_records)
    events_by_name = _count_events_by_name(trace_records)

    level_1_steps: List[Dict[str, Any]] = []
    for index, action in enumerate(action_list):
        if not isinstance(action, dict):
            level_1_steps.append({"step": index, "error": "invalid_action_record"})
            continue
        step_value = action.get("step")
        step_idx = step_value if isinstance(step_value, int) else index
        level_1_steps.append(
            {
                "step": step_idx,
                "label": action.get("label"),
                "widget_type": action.get("widget_type"),
                "intent": action.get("intent"),
                "success": bool(action.get("success")),
                "error": action.get("error"),
                "required": action.get("required"),
                "target_role": action.get("target_role"),
                "target_selector": action.get("target_selector"),
                "metadata_status": action.get("metadata_status"),
                "metadata_missing_keys": action.get("metadata_missing_keys"),
                "t_start_s": action.get("t_start_s"),
                "t_end_s": action.get("t_end_s"),
                "trace_event_count": step_counts.get(step_idx, 0),
            }
        )

    return {
        "level_0_run": {
            "form_id": annotations.get("form_id"),
            "run_name": annotations.get("run_name"),
            "form_url": annotations.get("form_url"),
            "submitted": bool(annotations.get("submitted")),
            "failure_reason": annotations.get("failure_reason"),
            "total_steps": len(action_list),
            "trace_event_count": len(trace_records),
            "video_path": annotations.get("video_path"),
        },
        "level_1_steps": level_1_steps,
        "level_2_trace": {
            "trace_path": (annotations.get("trace") or {}).get("tool_trace_path"),
            "event_count": len(trace_records),
            "events_by_name": events_by_name,
            "step_event_counts": step_counts,
            "trace_actions": sorted(events_by_name.keys()),
        },
    }


def _validate_annotation_trace_consistency(
    annotations: Dict[str, Any],
    answers: List[Dict[str, Any]],
    trace_records: List[Dict[str, Any]],
    trace_summary: Dict[str, Any],
    require_submit_success: bool = True,
) -> None:
    problems: List[str] = []

    actions = annotations.get("actions")
    if not isinstance(actions, list):
        problems.append("annotations.actions must be a list")
        actions = []

    if len(actions) != len(answers):
        problems.append(f"actions count ({len(actions)}) does not match answers count ({len(answers)})")

    for idx, action in enumerate(actions):
        if not isinstance(action, dict):
            problems.append(f"actions[{idx}] must be an object")
            continue
        if action.get("step") != idx:
            problems.append(f"actions[{idx}].step must equal {idx}")
        success = bool(action.get("success"))
        error = action.get("error")
        if success and error not in (None, ""):
            problems.append(f"actions[{idx}] is marked success=true but contains error")
        if (not success) and error in (None, ""):
            problems.append(f"actions[{idx}] is marked success=false but error is empty")

    if not trace_records:
        problems.append("tool trace must contain at least one event")
    else:
        allowed_actions = set(supported_actions())
        for line_number, record in enumerate(trace_records, start=1):
            name = record.get("name")
            if not isinstance(name, str) or not name.strip():
                problems.append(f"tool_trace line {line_number} has missing/invalid action name")
                continue
            if name not in allowed_actions:
                problems.append(f"tool_trace line {line_number} uses unsupported action '{name}'")

    step_counts = _step_event_counts(trace_records)
    for idx in range(len(actions)):
        if step_counts.get(idx, 0) < 1:
            problems.append(f"no trace event found for step_ref={idx}")

    summary_count = trace_summary.get("event_count")
    if isinstance(summary_count, int) and summary_count != len(trace_records):
        problems.append(
            f"trace_summary.event_count ({summary_count}) does not match trace lines ({len(trace_records)})"
        )

    submit = annotations.get("submit")
    if not isinstance(submit, dict):
        problems.append("annotations.submit must be an object")
        submit = {}
    submitted = bool(annotations.get("submitted"))
    submit_success = bool(submit.get("success"))
    if submitted != submit_success:
        problems.append("annotations.submitted must match annotations.submit.success")
    if submit_success and submit.get("submit_clicked") is False:
        problems.append("submit.success=true but submit_clicked=false")

    failure_reason = annotations.get("failure_reason")
    if submitted and failure_reason is not None:
        problems.append("failure_reason must be null when submitted=true")
    if (not submitted) and not failure_reason:
        problems.append("failure_reason must be non-null when submitted=false")
    if require_submit_success and not submitted:
        problems.append("submission confirmation is required but submitted=false")

    if problems:
        details = "\n- ".join(problems)
        raise RuntimeError(f"Annotation/trace consistency check failed:\n- {details}")


def validate_run_artifacts(
    run_dir: Path,
    video_path: Optional[Path],
    answers_path: Path,
    annotations_path: Path,
    trace_path: Path,
    trace_summary: Dict[str, Any],
    observations_dir: Path,
    screenshots_required: bool,
    run_error: Optional[Exception] = None,
    execution_errors: Optional[List[str]] = None,
) -> None:
    missing: List[str] = []
    if not answers_path.exists():
        missing.append("answers_instance.json")
    if not annotations_path.exists():
        missing.append("annotations.json")
    if not trace_path.exists() or trace_path.stat().st_size == 0:
        missing.append("tool_trace.jsonl (missing or empty)")
    if trace_summary.get("event_count", 0) < 1:
        missing.append("tool_trace.jsonl (no events)")
    if trace_summary.get("strict_mcp_validation") and trace_summary.get("validation_error_count", 0) > 0:
        missing.append("tool_trace.jsonl (MCP validation errors)")

    if screenshots_required:
        if not observations_dir.exists():
            missing.append("observations/")
        else:
            submit_pre = observations_dir / "submit_pre.png"
            submit_post = observations_dir / "submit_post.png"
            if not submit_pre.exists():
                missing.append("observations/submit_pre.png")
            if not submit_post.exists():
                missing.append("observations/submit_post.png")

    if video_path is None or not video_path.exists() or video_path.stat().st_size == 0:
        missing.append("final video (.webm)")
    if missing:
        joined = ", ".join(missing)
        detail_parts: List[str] = []
        if run_error is not None:
            detail_parts.append(f"run_error={run_error}")
        if execution_errors:
            detail_parts.append(f"errors={' ; '.join(execution_errors)}")
        details = f" Details: {' | '.join(detail_parts)}" if detail_parts else ""
        raise RuntimeError(f"Missing required artifacts in {run_dir}: {joined}.{details}")


def _artifact_presence(run_dir: Path) -> Dict[str, Any]:
    trace_path = run_dir / "tool_trace.jsonl"
    annotations_path = run_dir / "annotations.json"
    answers_path = run_dir / "answers_instance.json"
    videos = sorted(str(path) for path in run_dir.glob("*.webm"))
    return {
        "answers_instance_json": answers_path.exists() and answers_path.stat().st_size > 0,
        "annotations_json": annotations_path.exists() and annotations_path.stat().st_size > 0,
        "tool_trace_jsonl": trace_path.exists() and trace_path.stat().st_size > 0,
        "webm_count": len(videos),
        "webm_files": videos,
    }


def write_failure_manifest(
    run_dir: Path,
    *,
    form_id: str,
    run_name: str,
    run_label: str,
    error: BaseException,
) -> None:
    payload = {
        "schema_version": "1.0",
        "form_id": form_id,
        "run_name": run_name,
        "run_label": run_label,
        "error_type": type(error).__name__,
        "message": str(error),
        "artifacts": _artifact_presence(run_dir),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_dir / "failure_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _playwright_import_error() -> RuntimeError:
    return RuntimeError(
        "Playwright is not installed. Run:\n"
        "  python -m playwright install chromium\n"
        "  python -m playwright install --with-deps chromium  # Linux"
    )


def _playwright_browser_error(exc: Exception) -> RuntimeError:
    msg = str(exc)
    lowered = msg.lower()
    if "sandbox_host_linux.cc" in lowered or "operation not permitted" in lowered:
        return RuntimeError(
            "Playwright browser launch failed due to sandbox/container restrictions. "
            "If running in a restricted environment, execute outside sandbox restrictions, "
            "or run on a host where Chromium sandbox operations are allowed. "
            f"Original error: {msg}"
        )
    return RuntimeError(
        "Playwright browser install appears missing. Run:\n"
        "  python -m playwright install chromium\n"
        "  python -m playwright install --with-deps chromium  # Linux\n"
        f"Original error: {msg}"
    )


def _mcp_browser_error(exc: Exception) -> RuntimeError:
    message = str(exc)
    lowered = message.lower()
    if "listen eperm" in lowered or "operation not permitted" in lowered:
        return RuntimeError(
            "Failed to initialize official Playwright MCP browser server due to local network/socket restrictions. "
            "Use `--interaction-mode local` in restricted environments, or run MCP browser mode on a machine "
            "that allows local listener startup. "
            f"Original error: {message}"
        )
    return RuntimeError(
        "Failed to initialize official Playwright MCP browser server. Ensure Node.js + npx are available, "
        "or pass --browser-mcp-cmd with a working server command. "
        "For offline/reliable startup, install MCP once with `npm i -g @playwright/mcp`. "
        "If MCP reports missing browser binaries, run `npx playwright install chromium` (or `npx playwright install chrome`). "
        f"Original error: {message}"
    )


def _new_annotations(
    form_id: str,
    run_name: str,
    run_label: str,
    form_url: str,
    args: argparse.Namespace,
    trace: TraceLogger,
) -> Dict[str, Any]:
    return {
        "schema_version": "4.0",
        "form_id": form_id,
        "run_name": run_name,
        "run_label": run_label,
        "form_url": form_url,
        "viewport": {"width": args.viewport_width, "height": args.viewport_height},
        "device_pixel_ratio": None,
        "user_agent": None,
        "locale": None,
        "timezone": None,
        "video_path": None,
        "run_params": {
            "headless": bool(args.headless),
            "slow_mo": args.slow_mo,
            "timeout_ms": args.timeout_ms,
            "type_delay_ms": args.type_delay_ms,
            "action_delay_ms": args.action_delay_ms,
            "post_submit_delay_seconds": args.post_submit_delay_seconds,
            "screenshots": bool(args.screenshots),
            "mouse_overlay": not bool(args.no_mouse_overlay),
            "interaction_mode": args.interaction_mode,
            "trace_mode": args.trace_mode,
            "mcp_server_cmd": args.mcp_server_cmd or "bundled_default",
            "mcp_tool_name": args.mcp_tool_name,
            "mcp_timeout_ms": args.mcp_timeout_ms,
            "browser_mcp_cmd": args.browser_mcp_cmd or "default_playwright_mcp",
            "browser_mcp_timeout_ms": args.browser_mcp_timeout_ms,
            "mcp_verify_trace": not bool(args.no_mcp_verify_trace),
            "mcp_strict": not bool(args.no_mcp_strict),
        },
        "actions": [],
        "submit": {
            "success": False,
            "success_inferred": False,
            "t_start_s": trace.now(),
            "t_end_s": trace.now(),
            "bbox": None,
            "submit_clicked": False,
            "confirmation_method": None,
            "final_url": None,
            "pre_screenshot": None,
            "post_screenshot": None,
            "metadata_status": "unknown",
            "metadata_missing_keys": [],
        },
        "trace": {
            "tool_trace_path": "tool_trace.jsonl",
            "screenshot_dir": "observations" if args.screenshots else None,
            "mcp": None,
        },
        "submitted": False,
        "failure_reason": None,
    }


def _run_single_local(
    annotations: Dict[str, Any],
    answers: List[Dict[str, Any]],
    form_url: str,
    run_dir: Path,
    observations_dir: Path,
    args: argparse.Namespace,
    trace: TraceLogger,
) -> Tuple[List[str], Optional[Exception]]:
    errors: List[str] = []
    run_error: Optional[Exception] = None
    page = None
    context = None
    browser = None

    try:
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            raise _playwright_import_error() from exc

        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=args.headless, slow_mo=args.slow_mo)
                context = browser.new_context(
                    viewport={"width": args.viewport_width, "height": args.viewport_height},
                    record_video_dir=str(run_dir),
                    record_video_size={"width": args.viewport_width, "height": args.viewport_height},
                    **english_context_options(),
                )
                page = context.new_page()
                page.set_default_timeout(args.timeout_ms)
                page.goto(form_url, wait_until="load", timeout=args.timeout_ms)
                env = page.evaluate(
                    "() => ({devicePixelRatio: window.devicePixelRatio || null, userAgent: navigator.userAgent || null, locale: navigator.language || null, timezone: (Intl.DateTimeFormat().resolvedOptions().timeZone || null)})"
                )
                if isinstance(env, dict):
                    annotations["device_pixel_ratio"] = env.get("devicePixelRatio")
                    annotations["user_agent"] = env.get("userAgent")
                    annotations["locale"] = env.get("locale")
                    annotations["timezone"] = env.get("timezone")

                engine = FormEngine(
                    page=page,
                    viewport={"width": args.viewport_width, "height": args.viewport_height},
                    observations_dir=observations_dir,
                    trace=trace,
                    timeout_ms=args.timeout_ms,
                    type_delay_ms=args.type_delay_ms,
                    action_delay_ms=args.action_delay_ms,
                    take_screenshots=args.screenshots,
                )
                if not args.no_mouse_overlay:
                    engine.enable_mouse_overlay()

                for idx, entry in enumerate(answers):
                    label = entry.get("label") if isinstance(entry, dict) else None
                    if label:
                        print(f"[INFO] Filling step {idx}: {label}")
                    else:
                        print(f"[INFO] Filling step {idx}")
                    action, err = engine.fill_step(entry, idx)
                    annotations["actions"].append(action)
                    if err:
                        errors.append(f"step {idx}: {err}")

                print("[INFO] Submitting form")
                submit_info, submit_err = engine.submit()
                annotations["submit"] = submit_info
                if submit_err:
                    errors.append(f"submit: {submit_err}")
                elif not submit_info.get("success"):
                    errors.append("submit_not_confirmed")
                elif submit_info.get("success") and args.post_submit_delay_seconds > 0:
                    page.wait_for_timeout(int(args.post_submit_delay_seconds * 1000))
            except PlaywrightError as exc:
                raise _playwright_browser_error(exc) from exc
            except PlaywrightTimeoutError as exc:
                run_error = exc
                errors.append(f"timeout: {exc}")
            except Exception as exc:
                run_error = exc
                errors.append(f"run_error: {exc}")
            finally:
                if page is not None:
                    try:
                        page.close()
                    except BaseException:
                        pass
                if context is not None:
                    try:
                        context.close()
                    except BaseException:
                        pass
                if browser is not None:
                    try:
                        browser.close()
                    except BaseException:
                        pass
    except Exception as exc:
        run_error = exc
        if not any(str(exc) in item for item in errors):
            errors.append(f"run_error: {exc}")
    return errors, run_error


def _run_single_mcp_server(
    annotations: Dict[str, Any],
    answers: List[Dict[str, Any]],
    form_url: str,
    run_dir: Path,
    observations_dir: Path,
    args: argparse.Namespace,
    trace: TraceLogger,
) -> Tuple[List[str], Optional[Exception]]:
    errors: List[str] = []
    run_error: Optional[Exception] = None
    browser_mcp: Optional[MCPClient] = None
    engine: Optional[MCPBrowserEngine] = None

    required_tools = ["browser_navigate", "browser_run_code", "browser_wait_for", "browser_close"]
    if args.screenshots:
        required_tools.append("browser_take_screenshot")

    mcp_command: Any = args.browser_mcp_cmd or _default_browser_mcp_command(args, run_dir)
    mcp_env = {"PLAYWRIGHT_BROWSERS_PATH": _default_playwright_browsers_path()}
    annotations["run_params"]["browser_mcp_cmd"] = mcp_command
    annotations["run_params"]["browser_mcp_env"] = mcp_env
    try:
        timeout_ms = args.browser_mcp_timeout_ms
        last_init_error: Optional[Exception] = None
        for attempt in [1, 2]:
            try:
                browser_mcp = MCPClient(
                    command=mcp_command,
                    timeout_ms=timeout_ms,
                    required_tools=required_tools,
                    env=mcp_env,
                )
                break
            except Exception as exc:
                last_init_error = exc
                if attempt == 1 and "timed out" in str(exc).lower():
                    timeout_ms = max(timeout_ms * 2, 180000)
                    print(
                        f"[WARN] MCP browser init timed out, retrying once with timeout_ms={timeout_ms}",
                        file=sys.stderr,
                    )
                    continue
                raise
        if browser_mcp is None and last_init_error is not None:
            raise last_init_error
        engine = MCPBrowserEngine(
            mcp_client=browser_mcp,
            trace=trace,
            observations_dir=observations_dir,
            timeout_ms=args.timeout_ms,
            type_delay_ms=args.type_delay_ms,
            action_delay_ms=args.action_delay_ms,
            take_screenshots=args.screenshots,
        )
        env = engine.navigate(form_url)
        if isinstance(env, dict):
            annotations["device_pixel_ratio"] = env.get("devicePixelRatio")
            annotations["user_agent"] = env.get("userAgent")
            annotations["locale"] = env.get("locale")
            annotations["timezone"] = env.get("timezone")

        if not args.no_mouse_overlay:
            engine.enable_mouse_overlay()

        for idx, entry in enumerate(answers):
            label = entry.get("label") if isinstance(entry, dict) else None
            if label:
                print(f"[INFO] Filling step {idx}: {label}")
            else:
                print(f"[INFO] Filling step {idx}")
            action, err = engine.fill_step(entry, idx)
            annotations["actions"].append(action)
            if err:
                errors.append(f"step {idx}: {err}")

        print("[INFO] Submitting form")
        submit_info, submit_err = engine.submit()
        annotations["submit"] = submit_info
        if submit_err:
            errors.append(f"submit: {submit_err}")
        elif not submit_info.get("success"):
            errors.append("submit_not_confirmed")
        elif submit_info.get("success") and args.post_submit_delay_seconds > 0:
            engine.wait_seconds(args.post_submit_delay_seconds, step_ref=None)
    except Exception as exc:
        mapped = _mcp_browser_error(exc)
        run_error = mapped
        errors.append(f"run_error: {mapped}")
    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass
        if browser_mcp is not None:
            try:
                browser_mcp.close()
            except Exception:
                pass
    return errors, run_error


def run_single(
    form_id: str,
    form_url: str,
    answers: List[Dict[str, Any]],
    run_dir: Path,
    run_name: str,
    run_label: str,
    args: argparse.Namespace,
    run_metadata: Dict[str, Any],
) -> None:
    answers_path = run_dir / "answers_instance.json"
    answers_path.write_text(json.dumps(answers, indent=2))

    observations_dir = run_dir / "observations"
    if args.screenshots:
        observations_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "tool_trace.jsonl"

    start_time = time.perf_counter()
    trace_mcp_client = None
    if args.trace_mode == "mcp":
        trace_command: Any = args.mcp_server_cmd or _default_mcp_server_command()
        try:
            trace_mcp_client = MCPTraceClient(
                command=trace_command,
                tool_name=args.mcp_tool_name,
                timeout_ms=args.mcp_timeout_ms,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize MCP trace server. "
                "Use --trace-mode local to bypass MCP tracing. "
                f"Details: {exc}"
            ) from exc

    trace = TraceLogger(
        trace_path,
        start_time,
        validate_mcp_actions=not args.no_mcp_verify_trace,
        strict_mcp_validation=not args.no_mcp_strict,
        mcp_client=trace_mcp_client,
    )

    annotations = _new_annotations(form_id, run_name, run_label, form_url, args, trace)
    try:
        if args.interaction_mode == "mcp_server":
            errors, run_error = _run_single_mcp_server(
                annotations=annotations,
                answers=answers,
                form_url=form_url,
                run_dir=run_dir,
                observations_dir=observations_dir,
                args=args,
                trace=trace,
            )
        else:
            errors, run_error = _run_single_local(
                annotations=annotations,
                answers=answers,
                form_url=form_url,
                run_dir=run_dir,
                observations_dir=observations_dir,
                args=args,
                trace=trace,
            )
    finally:
        trace.close()

    trace_summary = trace.summary()
    annotations["trace"]["mcp"] = trace_summary
    trace_records = _load_trace_records(trace_path)

    video_path = finalize_video(run_dir, form_id, run_name)
    annotations["video_path"] = str(video_path) if video_path else None

    submitted = bool(annotations.get("submit", {}).get("success"))
    annotations["submitted"] = submitted
    failure_reason = None
    if not submitted:
        if annotations.get("submit", {}).get("submit_clicked") is False:
            failure_reason = "submit_not_clicked"
        else:
            failure_reason = "confirmation_not_detected"
    if run_error is not None and not submitted:
        failure_reason = "run_error"
    annotations["failure_reason"] = failure_reason
    if run_error is not None:
        annotations["run_error"] = str(run_error)

    if run_metadata:
        annotations["run_metadata"] = run_metadata

    annotations["information_layers"] = _build_information_layers(annotations, trace_records)

    annotations_path = run_dir / "annotations.json"
    annotations_path.write_text(json.dumps(annotations, indent=2))

    if run_error is None and not errors:
        _validate_annotation_trace_consistency(
            annotations=annotations,
            answers=answers,
            trace_records=trace_records,
            trace_summary=trace_summary,
            require_submit_success=True,
        )

        validate_run_artifacts(
            run_dir=run_dir,
            video_path=video_path,
            answers_path=answers_path,
            annotations_path=annotations_path,
            trace_path=trace_path,
            trace_summary=trace_summary,
            observations_dir=observations_dir,
            screenshots_required=bool(args.screenshots),
            run_error=run_error,
            execution_errors=errors,
        )

    screenshot_count = len(list(observations_dir.glob("*.png"))) if observations_dir.exists() else 0
    print("[INFO] Run complete")
    print(f"[INFO] run_dir: {run_dir}")
    print(f"[INFO] video: {video_path}")
    print(f"[INFO] screenshots: {screenshot_count}")
    print(f"[INFO] trace: {trace_path}")
    print(f"[INFO] trace_events: {trace_summary.get('event_count')}")
    print(f"[INFO] trace_mode: {trace_summary.get('mode')}")
    print(f"[INFO] interaction_mode: {args.interaction_mode}")

    if run_error is not None:
        raise RuntimeError(f"Run failed: {run_error}")
    if errors:
        joined = "; ".join(errors)
        raise RuntimeError(f"Run completed with errors: {joined}")


def run_for_form(
    args: argparse.Namespace,
    specs_root: Path,
    dataset_root: Path,
    form_id: str,
    num_runs_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    form_spec = load_form_spec(form_id, specs_root)
    form_url = force_english_google_forms_url(args.form_url or form_spec.get("form_url"))
    if not form_url:
        raise ValueError(f"Form URL not provided and not found in spec for form_id={form_id}")

    answers_path = resolve_answers_path(args, form_id)
    run_specs_iter = iter_run_specs(answers_path)

    runs_dir = dataset_root / form_id / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    existing_indices = set(existing_run_indices(runs_dir))
    start_index = args.start_index or 1
    if start_index < 1:
        raise ValueError("--start-index must be >= 1")
    skip_existing = args.skip_existing or args.resume
    if args.resume and args.start_index is None:
        start_index = next_available_index(existing_indices, 1)

    generated = 0
    current_index = start_index
    failures: List[Dict[str, Any]] = []

    print(f"[INFO] form_id={form_id}, answers={answers_path}")
    for spec in run_specs_iter:
        if num_runs_limit is not None and generated >= num_runs_limit:
            break
        if args.num_runs is not None and generated >= args.num_runs:
            break
        if skip_existing:
            current_index = next_available_index(existing_indices, current_index)
        run_dir, run_name, run_index = ensure_run_dir(
            runs_dir,
            current_index,
            skip_existing,
            existing_indices,
            args.skip_existing_video,
            args.overwrite_missing_video,
            args.overwrite_existing,
        )
        if run_dir is None:
            print(f"[INFO] Skipping existing video for {run_name}.")
            current_index = run_index + 1
            continue

        run_label = f"{form_id}_{run_name}"
        answers = spec.get("answers", [])
        if not isinstance(answers, list):
            raise ValueError("Run answers must be a list")
        run_metadata = spec.get("metadata", {})

        try:
            run_single(
                form_id=form_id,
                form_url=form_url,
                answers=answers,
                run_dir=run_dir,
                run_name=run_name,
                run_label=run_label,
                args=args,
                run_metadata=run_metadata,
            )
        except Exception as exc:
            write_failure_manifest(
                run_dir,
                form_id=form_id,
                run_name=run_name,
                run_label=run_label,
                error=exc,
            )
            failure = {
                "form_id": form_id,
                "run_name": run_name,
                "run_index": run_index,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "failure_manifest_path": str(run_dir / "failure_manifest.json"),
            }
            failures.append(failure)
            print(
                f"[WARN] reference_run_failed form_id={form_id} run_name={run_name} error={exc}",
                file=sys.stderr,
            )
            if not args.continue_on_run_error:
                raise
        generated += 1
        current_index = run_index + 1
    return failures


def main(argv: Optional[List[str]] = None) -> bool:
    args = parse_args(argv)
    specs_root = Path(args.specs_root) if args.specs_root else (SRC_DIR / "forms")
    dataset_root = Path(args.dataset_root).resolve()

    if os.environ.get("PLAYWRIGHT_SKIP_FFMPEG_INSTALL"):
        print(
            "[WARN] PLAYWRIGHT_SKIP_FFMPEG_INSTALL is set; video recording may fail.",
            file=sys.stderr,
        )

    if args.interaction_mode == "mcp_server" and not args.no_mcp_browser_install:
        if args.browser_mcp_cmd:
            print(
                "[WARN] Skipping MCP browser auto-install because --browser-mcp-cmd was provided. "
                "Ensure that custom MCP server command has browser binaries available.",
                file=sys.stderr,
            )
        else:
            try:
                python_executable = _detect_python_playwright_chromium_executable()
                if python_executable:
                    print(
                        f"[INFO] Using Python Playwright Chromium executable for MCP: {python_executable}",
                        file=sys.stderr,
                    )
                try:
                    ensure_playwright_mcp_package(timeout_seconds=args.mcp_browser_install_timeout_s)
                except Exception as preflight_exc:
                    print(
                        "[WARN] MCP package preflight failed; continuing with cached/local resolution. "
                        f"Details: {preflight_exc}",
                        file=sys.stderr,
                    )
                if not python_executable:
                    ensure_node_playwright_browser(
                        browser_name="chromium",
                        timeout_seconds=args.mcp_browser_install_timeout_s,
                    )
            except Exception as exc:
                raise RuntimeError(
                    "MCP browser preflight failed before run start. "
                    "You can retry, run `npx playwright install chromium`, or bypass this with --no-mcp-browser-install. "
                    f"Details: {exc}"
                ) from exc

    if args.smoke_test_all_forms:
        form_ids = discover_form_ids(specs_root)
        if not form_ids:
            raise ValueError(f"No forms found in specs root: {specs_root}")

        failures: List[Tuple[str, str]] = []
        for form_id in form_ids:
            print(f"[SMOKE] Running one test run for form_id={form_id}")
            try:
                form_failures = run_for_form(args, specs_root, dataset_root, form_id, num_runs_limit=1)
                if form_failures:
                    raise RuntimeError(form_failures[0]["message"])
                print(f"[SMOKE] PASS form_id={form_id}")
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                failures.append((form_id, message))
                print(f"[SMOKE] FAIL form_id={form_id}: {message}", file=sys.stderr)

        passed = len(form_ids) - len(failures)
        print(f"[SMOKE] Summary: passed={passed}, failed={len(failures)}, total={len(form_ids)}")
        if failures:
            failed_forms = ", ".join(form_id for form_id, _ in failures)
            raise RuntimeError(f"Smoke test failures in forms: {failed_forms}")
        return True

    form_ids = discover_form_ids(specs_root) if args.all_forms else [str(args.form_id)]
    if not form_ids:
        raise ValueError(f"No forms found in specs root: {specs_root}")

    failures: List[Dict[str, Any]] = []
    for form_id in form_ids:
        failures.extend(run_for_form(args, specs_root, dataset_root, form_id))

    if failures:
        summary_path = dataset_root / "reference_generation_failures.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps({"failures": failures}, indent=2), encoding="utf-8")
        print(
            f"[WARN] reference_generation_failures={len(failures)} summary={summary_path}",
            file=sys.stderr,
        )

    return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
