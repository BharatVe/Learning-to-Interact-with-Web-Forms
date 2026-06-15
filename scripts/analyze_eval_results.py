#!/usr/bin/env python3
"""Build lightweight evaluation analysis tables and SVG plots.

The script intentionally uses only the Python standard library so it works in
the cluster virtualenv without pandas/matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from analyze_reference_dataset import trace_stats


DEFAULT_DATASET_ROOT = Path("data/model_baselines")
DEFAULT_OUTPUT_DIR = Path("docs/eval_results/analysis")
DEFAULT_ANSWERS_ROOT = Path("data/answers")
DEFAULT_FORMS_ROOT = Path("src/forms")
DEFAULT_REFERENCE_ROOT = Path("data/forms")
TARGET_RUN_COUNT = 6
TARGET_TRIAL_COUNT = 300
TARGET_RUN_IDS = tuple(f"run_{idx:04d}" for idx in range(1, TARGET_RUN_COUNT + 1))
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260609
QWEN_MODEL_IDS = {
    "text_qwen3_30b_a3b_instruct_2507",
    "vlm_qwen3_vl_30b_a3b_instruct",
}
OPENCUA_NATIVE_MODEL_ID = "computer_use_opencua_32b"
OPENCUA_DIRECT_MCP_MODEL_ID = "computer_use_opencua_32b_direct_mcp"
TARGET_MODEL_IDS = {
    *QWEN_MODEL_IDS,
    OPENCUA_NATIVE_MODEL_ID,
    OPENCUA_DIRECT_MCP_MODEL_ID,
}
THESIS_MODEL_ORDER = (
    "text_qwen3_30b_a3b_instruct_2507",
    "vlm_qwen3_vl_30b_a3b_instruct",
    OPENCUA_NATIVE_MODEL_ID,
    OPENCUA_DIRECT_MCP_MODEL_ID,
)
THESIS_MODEL_LABELS = {
    "text_qwen3_30b_a3b_instruct_2507": "Qwen Text",
    "vlm_qwen3_vl_30b_a3b_instruct": "Qwen VLM",
    OPENCUA_NATIVE_MODEL_ID: "OpenCUA Native",
    OPENCUA_DIRECT_MCP_MODEL_ID: "OpenCUA MCP",
}
THESIS_COLORS = {
    "Qwen Text": "#2563eb",
    "Qwen VLM": "#059669",
    "OpenCUA Native": "#dc2626",
    "OpenCUA MCP": "#7c3aed",
}
QWEN_EXPERIMENT = "qwen_direct_mcp_english_stepcap128_5form_20260515"
OPENCUA_CONTROL_EXPERIMENT = "opencua_control_guidance_30form_20260526"
OPENCUA_REMAINING_EXPERIMENT = "opencua_control_guidance_remaining20_20260527"
OPENCUA_LOOP_EXPERIMENT = "opencua_loopdetector_30form_retry_20260526"
OPENCUA_TOPDOWN_EXPERIMENT = "opencua_topdown_prompt_20form_20260519"
QWEN_EXPERIMENT_PREFIX = "qwen_direct_mcp_"
OPENCUA_CONTROL_EXPERIMENT_PREFIX = "opencua_control_guidance"
OPENCUA_DIRECT_MCP_EXPERIMENT_PREFIX = "opencua_direct_mcp_tools"
QWEN_RECENT_BATCH_DATE = "2026-05-27"
EXPECTED_EXPERIMENT_TRIALS = {
    "qwen_direct_mcp_all50_run2_20260528": 100,
    "opencua_control_guidance_all50_run2_20260528": 50,
}
QWEN_PREVIOUS_BATCH_FORMS = {
    "research_interest",
    "room_booking",
    "scholarship_interest",
    "seminar_proposal",
    "software_access",
    "sports_tournament",
    "study_group_match",
    "survey_consent",
    "technical_support",
    "thesis_meeting",
}


@dataclass(frozen=True)
class Trial:
    experiment_id: str
    model_id: str
    form_id: str
    answer_run_id: str
    trial_id: str
    success: bool
    submit_success: bool
    stop_reason: str
    failure_category: str
    question_total: int
    scored_correctness: int
    scored_source: str
    pre_successful_submit_correctness: Optional[int]
    pre_first_submit_correctness: Optional[int]
    verified_correctness: int
    attempted_correctness: int
    submit_attempt_count: int
    successful_submit_attempt_count: int
    failed_submit_attempt_count: int
    submitted_while_incomplete_count: int
    action_count: int
    duration_s: float
    reference_available: bool
    reference_action_count: Optional[int]
    reference_duration_s: Optional[float]
    action_overhead_ratio: Optional[float]
    time_overhead_ratio: Optional[float]
    action_count_delta: Optional[int]
    duration_delta_s: Optional[float]
    reference_run_path: str
    reference_trace_path: str
    reference_video_path: str
    wasted_interaction_rate: Optional[float]
    run_completed_utc: str
    summary_path: Path


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        value_float = float(value)
    except Exception:
        return None
    if math.isnan(value_float) or math.isinf(value_float):
        return None
    return value_float


def _score(summary: Dict[str, Any]) -> Tuple[int, str]:
    if summary.get("pre_successful_submit_verified_correctness") is not None:
        return _as_int(summary.get("pre_successful_submit_verified_correctness")), "pre_successful_submit"
    if summary.get("pre_first_submit_verified_correctness") is not None:
        return _as_int(summary.get("pre_first_submit_verified_correctness")), "pre_first_submit"
    return _as_int(summary.get("verified_correctness")), "final_verification"


def _count_valid_trace_events(path: Path) -> Optional[int]:
    stats = trace_stats(path)
    if not stats.get("trace_exists"):
        return None
    if not stats.get("trace_valid"):
        return None
    return int(stats.get("action_count") or 0)


def _max_trace_time_s(trace_path: Path) -> Optional[float]:
    stats = trace_stats(trace_path)
    value = _optional_float(stats.get("last_event_time_s"))
    return value


def _derive_reference_duration_s(reference_annotations: Dict[str, Any], reference_trace_path: Path) -> Optional[float]:
    stats = trace_stats(reference_trace_path)
    value = _optional_float(stats.get("duration_s"))
    return round(value, 6) if value is not None else None


def _reference_run_paths(form_id: str, answer_run_id: str, reference_root: Path = DEFAULT_REFERENCE_ROOT) -> Dict[str, Path]:
    run_root = reference_root / form_id / "runs" / answer_run_id
    return {
        "run_root": run_root,
        "annotations_path": run_root / "annotations.json",
        "trace_path": run_root / "tool_trace.jsonl",
    }


def _reference_metrics(form_id: str, answer_run_id: str, reference_root: Path = DEFAULT_REFERENCE_ROOT) -> Dict[str, Any]:
    paths = _reference_run_paths(form_id, answer_run_id, reference_root)
    annotations = _read_json(paths["annotations_path"]) if paths["annotations_path"].exists() else {}
    trace_path = paths["trace_path"]
    run_root = paths["run_root"]
    webm_candidates = sorted(run_root.glob("*.webm")) if run_root.exists() else []
    raw_video = annotations.get("video_path") if isinstance(annotations, dict) else None
    video_path = str(raw_video).strip() if isinstance(raw_video, str) and raw_video.strip() else ""
    raw_video_exists = bool(video_path and Path(video_path).exists() and Path(video_path).stat().st_size > 0)
    if (not raw_video_exists) and webm_candidates:
        video_path = str(webm_candidates[0])
        raw_video_exists = bool(webm_candidates[0].stat().st_size > 0)
    action_count = _count_valid_trace_events(trace_path)
    duration_s = _derive_reference_duration_s(annotations, trace_path) if trace_path.exists() else None
    run_params = annotations.get("run_params") if isinstance(annotations, dict) else {}
    submit = annotations.get("submit") if isinstance(annotations, dict) else {}
    raw_available = bool(run_root.exists() and paths["annotations_path"].exists() and trace_path.exists())
    submit_success = bool(submit.get("success")) if isinstance(submit, dict) else False
    usable = bool(raw_available and raw_video_exists and action_count is not None and action_count > 0 and duration_s is not None and duration_s > 0 and submit_success)
    return {
        "reference_available": usable,
        "reference_artifacts_present": raw_available,
        "reference_run_path": str(run_root),
        "reference_trace_path": str(trace_path),
        "reference_video_path": video_path,
        "reference_video_available": raw_video_exists,
        "reference_action_count": action_count,
        "reference_duration_s": duration_s,
        "reference_interaction_mode": str(run_params.get("interaction_mode") or "") if isinstance(run_params, dict) else "",
        "reference_trace_mode": str(run_params.get("trace_mode") or "") if isinstance(run_params, dict) else "",
        "reference_submit_success": submit_success,
    }


def _resolve_efficiency_for_trial(summary: Dict[str, Any], form_id: str, answer_run_id: str) -> Dict[str, Any]:
    reference = _reference_metrics(form_id, answer_run_id)
    action_count = _optional_int(summary.get("action_count") or summary.get("trace_action_count"))
    duration_s = _optional_float(summary.get("duration_s"))
    reference_action_count = _optional_int(summary.get("reference_action_count"))
    if reference["reference_available"] and reference["reference_action_count"] is not None:
        reference_action_count = int(reference["reference_action_count"])
    reference_duration_s = _optional_float(summary.get("reference_duration_s"))
    if reference["reference_available"] and reference["reference_duration_s"] is not None:
        reference_duration_s = float(reference["reference_duration_s"])

    action_overhead_ratio = _optional_float(summary.get("action_overhead_ratio"))
    action_count_delta = _optional_int(summary.get("action_count_delta"))
    if reference.get("reference_artifacts_present") and not reference.get("reference_available"):
        reference_action_count = None
        reference_duration_s = None
        action_overhead_ratio = None
        action_count_delta = None
    if action_count is not None and reference_action_count is not None and reference_action_count > 0:
        action_overhead_ratio = round(float(action_count) / float(reference_action_count), 6)
        action_count_delta = int(action_count) - int(reference_action_count)

    time_overhead_ratio = _optional_float(summary.get("time_overhead_ratio"))
    duration_delta_s = _optional_float(summary.get("duration_delta_s"))
    if reference.get("reference_artifacts_present") and not reference.get("reference_available"):
        time_overhead_ratio = None
        duration_delta_s = None
    if duration_s is not None and reference_duration_s is not None and reference_duration_s > 0:
        time_overhead_ratio = round(float(duration_s) / float(reference_duration_s), 6)
        duration_delta_s = round(float(duration_s) - float(reference_duration_s), 6)

    return {
        **reference,
        "reference_action_count": reference_action_count,
        "reference_duration_s": reference_duration_s,
        "action_overhead_ratio": action_overhead_ratio,
        "time_overhead_ratio": time_overhead_ratio,
        "action_count_delta": action_count_delta,
        "duration_delta_s": duration_delta_s,
    }


def _wasted_interaction_rate(path: Path) -> Optional[float]:
    annotations = _read_json(_annotation_path(path))
    steps = annotations.get("steps")
    if not isinstance(steps, list):
        return None
    action_rows = [row for row in steps if isinstance(row, dict)]
    if not action_rows:
        return None
    wasted = 0
    for row in action_rows:
        status = str(row.get("status") or "").lower()
        stall_type = str(row.get("stall_type") or "").lower()
        repeated = _as_int(row.get("repeat_same_signature_count") or row.get("repeat_same_target_count") or row.get("repeat_same_action_count"))
        if status in {"failed", "filled_unverified"} or not bool(row.get("progress_made")) or stall_type or repeated >= 3:
            wasted += 1
    return round(wasted / len(action_rows), 6)


def _trial_from_summary(path: Path, dataset_root: Path) -> Optional[Trial]:
    summary = _read_json(path)
    if not summary:
        return None
    try:
        parts = path.relative_to(dataset_root).parts
    except ValueError:
        parts = path.parts
    experiment_id = str(summary.get("experiment_id") or parts[0])
    model_id = str(summary.get("model_id") or parts[1])
    form_id = str(summary.get("form_id") or parts[2])
    answer_run_id = str(summary.get("answer_run_id") or parts[3])
    trial_id = str(summary.get("trial_id") or parts[4])
    scored, source = _score(summary)
    efficiency = _resolve_efficiency_for_trial(summary, form_id, answer_run_id)
    return Trial(
        experiment_id=experiment_id,
        model_id=model_id,
        form_id=form_id,
        answer_run_id=answer_run_id,
        trial_id=trial_id,
        success=_as_bool(summary.get("success")),
        submit_success=_as_bool(summary.get("submit_success")),
        stop_reason=str(summary.get("stop_reason") or ""),
        failure_category=str(summary.get("failure_category") or ""),
        question_total=_as_int(summary.get("question_total")),
        scored_correctness=scored,
        scored_source=source,
        pre_successful_submit_correctness=(
            None
            if summary.get("pre_successful_submit_verified_correctness") is None
            else _as_int(summary.get("pre_successful_submit_verified_correctness"))
        ),
        pre_first_submit_correctness=(
            None
            if summary.get("pre_first_submit_verified_correctness") is None
            else _as_int(summary.get("pre_first_submit_verified_correctness"))
        ),
        verified_correctness=_as_int(summary.get("verified_correctness")),
        attempted_correctness=_as_int(summary.get("attempted_correctness")),
        submit_attempt_count=_as_int(summary.get("submit_attempt_count")),
        successful_submit_attempt_count=_as_int(summary.get("successful_submit_attempt_count")),
        failed_submit_attempt_count=_as_int(summary.get("failed_submit_attempt_count")),
        submitted_while_incomplete_count=_as_int(summary.get("submitted_while_incomplete_count")),
        action_count=_as_int(summary.get("action_count") or summary.get("trace_action_count")),
        duration_s=_as_float(summary.get("duration_s")),
        reference_available=bool(efficiency.get("reference_available")),
        reference_action_count=_optional_int(efficiency.get("reference_action_count")),
        reference_duration_s=_optional_float(efficiency.get("reference_duration_s")),
        action_overhead_ratio=_optional_float(efficiency.get("action_overhead_ratio")),
        time_overhead_ratio=_optional_float(efficiency.get("time_overhead_ratio")),
        action_count_delta=_optional_int(efficiency.get("action_count_delta")),
        duration_delta_s=_optional_float(efficiency.get("duration_delta_s")),
        reference_run_path=str(efficiency.get("reference_run_path") or ""),
        reference_trace_path=str(efficiency.get("reference_trace_path") or ""),
        reference_video_path=str(efficiency.get("reference_video_path") or ""),
        wasted_interaction_rate=_wasted_interaction_rate(path),
        run_completed_utc=str(summary.get("run_completed_utc") or ""),
        summary_path=path,
    )


def load_trials(dataset_root: Path) -> List[Trial]:
    trials = []
    for path in sorted(dataset_root.glob("**/summary.json")):
        trial = _trial_from_summary(path, dataset_root)
        if trial is not None:
            trials.append(trial)
    return trials


def _is_qwen_current_trial(trial: Trial) -> bool:
    return trial.experiment_id.startswith(QWEN_EXPERIMENT_PREFIX) and trial.model_id in QWEN_MODEL_IDS


def _is_opencua_control_trial(trial: Trial) -> bool:
    return (
        trial.model_id == OPENCUA_NATIVE_MODEL_ID
        and trial.experiment_id.startswith(OPENCUA_CONTROL_EXPERIMENT_PREFIX)
    )


def _is_opencua_direct_mcp_trial(trial: Trial) -> bool:
    return (
        trial.model_id == OPENCUA_DIRECT_MCP_MODEL_ID
        and trial.experiment_id.startswith(OPENCUA_DIRECT_MCP_EXPERIMENT_PREFIX)
    )


def _completed_experiment_ids(trials: Sequence[Trial]) -> set[str]:
    counts = Counter(t.experiment_id for t in trials)
    completed = set(counts)
    for experiment_id in list(completed):
        expected = _expected_trials_for_experiment(experiment_id)
        if expected is not None and counts.get(experiment_id, 0) < expected:
            completed.discard(experiment_id)
    return completed


def _expected_trials_for_experiment(experiment_id: str) -> Optional[int]:
    if experiment_id in EXPECTED_EXPERIMENT_TRIALS:
        return EXPECTED_EXPERIMENT_TRIALS[experiment_id]
    if re.match(r"^qwen_direct_mcp_.*all50_run\d+_\d{8}$", experiment_id):
        return 100
    if re.match(r"^opencua_control_guidance_.*all50_run\d+_\d{8}$", experiment_id):
        return 50
    if re.match(r"^opencua_direct_mcp_tools_.*all50_run\d+_\d{8}$", experiment_id):
        return 50
    return None


def _cohorts(trials: Sequence[Trial]) -> Dict[str, List[Trial]]:
    completed_experiments = _completed_experiment_ids(trials)
    opencua_control_all = [t for t in trials if _is_opencua_control_trial(t)]
    opencua_direct_mcp_all = [t for t in trials if _is_opencua_direct_mcp_trial(t)]
    qwen_recent = [
        t
        for t in trials
        if _is_qwen_current_trial(t) and t.run_completed_utc.startswith(QWEN_RECENT_BATCH_DATE)
    ]
    return {
        "qwen_latest_completed": qwen_recent,
        "qwen_all_completed": [
            t for t in trials if _is_qwen_current_trial(t) and t.experiment_id in completed_experiments
        ],
        "opencua_control_guidance_completed": [
            t for t in opencua_control_all if t.experiment_id in completed_experiments
        ],
        "opencua_direct_mcp_completed": [
            t for t in opencua_direct_mcp_all if t.experiment_id in completed_experiments
        ],
        "opencua_control_guidance_30": [
            t for t in trials if t.experiment_id == OPENCUA_CONTROL_EXPERIMENT
        ],
        "opencua_control_guidance_remaining20": [
            t for t in trials if t.experiment_id == OPENCUA_REMAINING_EXPERIMENT
        ],
        "opencua_loopdetector_partial": [
            t for t in trials if t.experiment_id == OPENCUA_LOOP_EXPERIMENT
        ],
        "opencua_topdown_20": [t for t in trials if t.experiment_id == OPENCUA_TOPDOWN_EXPERIMENT],
    }


def experiment_coverage_rows(trials: Sequence[Trial]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Trial]] = defaultdict(list)
    for trial in trials:
        grouped[trial.experiment_id].append(trial)
    rows = []
    for experiment_id, items in sorted(grouped.items()):
        expected = _expected_trials_for_experiment(experiment_id)
        observed = len(items)
        if expected is None:
            status = "historical_or_open_scope"
        elif observed >= expected:
            status = "complete"
        else:
            status = "in_progress_or_partial"
        rows.append(
            {
                "experiment_id": experiment_id,
                "status": status,
                "observed_trials": observed,
                "expected_trials": "" if expected is None else expected,
                "models": ", ".join(sorted({t.model_id for t in items})),
                "forms": len({t.form_id for t in items}),
                "runs": ", ".join(sorted({t.answer_run_id for t in items})),
                "first_completed_utc": min((t.run_completed_utc for t in items if t.run_completed_utc), default=""),
                "last_completed_utc": max((t.run_completed_utc for t in items if t.run_completed_utc), default=""),
            }
        )
    return rows


def _all_form_ids(forms_root: Path = DEFAULT_FORMS_ROOT) -> List[str]:
    return sorted(entry.name for entry in forms_root.iterdir() if entry.is_dir() and (entry / "spec.json").exists())


def target_coverage_rows(trials: Sequence[Trial]) -> List[Dict[str, Any]]:
    rows = []
    form_ids = _all_form_ids()
    by_model: Dict[str, set[Tuple[str, str]]] = defaultdict(set)
    raw_counts: Dict[Tuple[str, str, str], int] = Counter()
    for trial in trials:
        if trial.model_id not in TARGET_MODEL_IDS:
            continue
        key = (trial.form_id, trial.answer_run_id)
        by_model[trial.model_id].add(key)
        raw_counts[(trial.model_id, trial.form_id, trial.answer_run_id)] += 1
    for model_id in sorted(TARGET_MODEL_IDS):
        expected_pairs = {(form_id, run_id) for form_id in form_ids for run_id in TARGET_RUN_IDS}
        observed_pairs = by_model.get(model_id, set())
        missing_pairs = sorted(expected_pairs - observed_pairs)
        duplicate_pair_count = sum(
            1
            for form_id, run_id in expected_pairs
            if raw_counts.get((model_id, form_id, run_id), 0) > 1
        )
        rows.append(
            {
                "model_id": model_id,
                "target_forms": len(form_ids),
                "target_runs_per_form": TARGET_RUN_COUNT,
                "target_form_runs": len(expected_pairs),
                "observed_unique_form_runs": len(observed_pairs & expected_pairs),
                "missing_form_runs": len(missing_pairs),
                "duplicate_target_form_runs": duplicate_pair_count,
                "forms_complete_target_runs": sum(
                    1
                    for form_id in form_ids
                    if all((form_id, run_id) in observed_pairs for run_id in TARGET_RUN_IDS)
                ),
                "forms_with_any_run": len({form_id for form_id, _run_id in observed_pairs}),
                "missing_examples": "; ".join(f"{form}:{run}" for form, run in missing_pairs[:20]),
            }
        )
    return rows


def _safe_rate(num: float, den: float) -> float:
    return num / den if den else 0.0


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return sorted_values[int(pos)]
    return sorted_values[low] * (high - pos) + sorted_values[high] * (pos - low)


def _fmt_optional(value: Any, digits: int = 3) -> str:
    numeric = _optional_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.{digits}f}"


def _model_label(model_id: str) -> str:
    return THESIS_MODEL_LABELS.get(model_id, model_id[:24])


def _dominant_stop_reason(row: Dict[str, Any]) -> str:
    stop_reasons = row.get("stop_reasons") or Counter()
    if isinstance(stop_reasons, Counter):
        return stop_reasons.most_common(1)[0][0] if stop_reasons else ""
    if isinstance(stop_reasons, dict):
        return max(stop_reasons.items(), key=lambda item: int(item[1]))[0] if stop_reasons else ""
    return ""


def _target_row_by_model(target_coverage: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("model_id") or ""): row for row in target_coverage}


def _efficiency_row_by_scope(efficiency_summary: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("analysis_scope") or ""): row
        for row in efficiency_summary
        if row.get("efficiency_subset") == "all_trials"
    }


def thesis_model_summary_rows(
    model_rows: Sequence[Dict[str, Any]],
    efficiency_summary: Sequence[Dict[str, Any]],
    target_coverage: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_model = {str(row.get("model_id") or ""): row for row in model_rows}
    by_scope = _efficiency_row_by_scope(efficiency_summary)
    targets = _target_row_by_model(target_coverage)
    rows: List[Dict[str, Any]] = []
    for model_id in THESIS_MODEL_ORDER:
        row = by_model.get(model_id)
        if not row:
            continue
        efficiency = by_scope.get(str(row.get("analysis_scope") or ""), {})
        target = targets.get(model_id, {})
        target_form_runs = _as_int(target.get("target_form_runs"))
        observed_unique = _as_int(target.get("observed_unique_form_runs"))
        trials = _as_int(row.get("trials"))
        reference_trials = _as_int(efficiency.get("reference_available_trials") or row.get("reference_available_trials"))
        rows.append(
            {
                "display_label": _model_label(model_id),
                "model_id": model_id,
                "interface_condition": row.get("interface_condition", ""),
                "analysis_scope": row.get("analysis_scope", ""),
                "trials": trials,
                "forms": row.get("forms", 0),
                "target_form_runs": target_form_runs,
                "observed_unique_form_runs": observed_unique,
                "target_coverage_rate": round(_safe_rate(observed_unique, target_form_runs), 6),
                "submit_rate": round(float(row.get("submit_rate") or 0.0), 6),
                "exact_success_rate": round(float(row.get("exact_success_rate") or 0.0), 6),
                "scored_accuracy": round(float(row.get("scored_accuracy") or 0.0), 6),
                "median_action_count": efficiency.get("median_model_action_count", ""),
                "median_duration_s": efficiency.get("median_model_duration_s", ""),
                "reference_available_trials": reference_trials,
                "reference_coverage_rate": round(_safe_rate(reference_trials, trials), 6),
                "median_action_overhead": efficiency.get("median_action_overhead_ratio", ""),
                "median_time_overhead": efficiency.get("median_time_overhead_ratio", ""),
                "dominant_stop_reason": _dominant_stop_reason(row),
            }
        )
    return rows


def _bootstrap_ci(values: Sequence[float], *, samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed + len(values))
    means: List[float] = []
    values_list = list(values)
    for _idx in range(samples):
        draw = [values_list[rng.randrange(len(values_list))] for _item in values_list]
        means.append(sum(draw) / len(draw))
    return _quantile(means, 0.025), _quantile(means, 0.975)


def _trial_exact_success(trial: Trial) -> bool:
    return bool(trial.submit_success and trial.question_total > 0 and trial.scored_correctness == trial.question_total)


def _trial_accuracy(trial: Trial) -> float:
    return _safe_rate(trial.scored_correctness, trial.question_total)


def aggregate(label: str, trials: Sequence[Trial]) -> Dict[str, Any]:
    total_questions = sum(t.question_total for t in trials)
    scored = sum(t.scored_correctness for t in trials)
    pre_success_trials = [t for t in trials if t.pre_successful_submit_correctness is not None]
    pre_success_total = sum(t.pre_successful_submit_correctness or 0 for t in pre_success_trials)
    pre_success_questions = sum(t.question_total for t in pre_success_trials)
    model_ids = sorted({t.model_id for t in trials})
    kinds = sorted({str(_read_json(t.summary_path).get("model_kind") or "") for t in trials})
    tracks = sorted({str(_read_json(t.summary_path).get("track") or "") for t in trials})
    interface_conditions = sorted({_interface_condition(t) for t in trials})
    exact_successes = sum(1 for t in trials if _trial_exact_success(t))
    action_overheads = [float(t.action_overhead_ratio) for t in trials if t.action_overhead_ratio is not None]
    time_overheads = [float(t.time_overhead_ratio) for t in trials if t.time_overhead_ratio is not None]
    wasted_rates = [float(t.wasted_interaction_rate) for t in trials if t.wasted_interaction_rate is not None]
    actions_per_correct = [
        float(t.action_count) / float(t.scored_correctness)
        for t in trials
        if t.action_count > 0 and t.scored_correctness > 0
    ]
    correct_per_min = [
        float(t.scored_correctness) / (float(t.duration_s) / 60.0)
        for t in trials
        if t.duration_s > 0 and t.scored_correctness > 0
    ]
    return {
        "analysis_scope": label,
        "model_id": ", ".join(model_ids),
        "model_kind": ", ".join(kind for kind in kinds if kind),
        "track": ", ".join(track for track in tracks if track),
        "interface_condition": ", ".join(item for item in interface_conditions if item),
        "trials": len(trials),
        "forms": len({t.form_id for t in trials}),
        "question_total": total_questions,
        "successes": sum(1 for t in trials if t.success),
        "exact_successes": exact_successes,
        "submit_successes": sum(1 for t in trials if t.submit_success),
        "success_rate": _safe_rate(sum(1 for t in trials if t.success), len(trials)),
        "exact_success_rate": _safe_rate(exact_successes, len(trials)),
        "submit_rate": _safe_rate(sum(1 for t in trials if t.submit_success), len(trials)),
        "scored_correctness": scored,
        "scored_accuracy": _safe_rate(scored, total_questions),
        "mean_trial_accuracy": _mean([_trial_accuracy(t) for t in trials]) or 0.0,
        "pre_submit_answered": pre_success_total,
        "pre_submit_questions": pre_success_questions,
        "pre_submit_accuracy": _safe_rate(pre_success_total, pre_success_questions),
        "submit_attempt_count": sum(t.submit_attempt_count for t in trials),
        "submitted_while_incomplete_count": sum(t.submitted_while_incomplete_count for t in trials),
        "avg_action_count": _safe_rate(sum(t.action_count for t in trials), len(trials)),
        "avg_duration_s": _safe_rate(sum(t.duration_s for t in trials), len(trials)),
        "reference_available_trials": sum(1 for t in trials if t.reference_available),
        "reference_availability_rate": _safe_rate(sum(1 for t in trials if t.reference_available), len(trials)),
        "median_reference_action_count": _median([float(t.reference_action_count) for t in trials if t.reference_action_count is not None]),
        "median_reference_duration_s": _median([float(t.reference_duration_s) for t in trials if t.reference_duration_s is not None]),
        "median_action_overhead_ratio": _median(action_overheads),
        "p25_action_overhead_ratio": _quantile(action_overheads, 0.25),
        "p75_action_overhead_ratio": _quantile(action_overheads, 0.75),
        "median_time_overhead_ratio": _median(time_overheads),
        "p25_time_overhead_ratio": _quantile(time_overheads, 0.25),
        "p75_time_overhead_ratio": _quantile(time_overheads, 0.75),
        "mean_wasted_interaction_rate": _mean(wasted_rates),
        "median_actions_per_correct_answer": _median(actions_per_correct),
        "median_correct_answers_per_min": _median(correct_per_min),
        "stop_reasons": Counter(t.stop_reason for t in trials),
    }


def aggregate_by_model(label: str, trials: Sequence[Trial]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Trial]] = defaultdict(list)
    for trial in trials:
        grouped[trial.model_id].append(trial)
    return [aggregate(f"{label}:{model}", items) for model, items in sorted(grouped.items())]


def _interface_condition(trial: Trial) -> str:
    if trial.model_id in QWEN_MODEL_IDS:
        return "qwen_direct_mcp_tools"
    if trial.model_id == OPENCUA_NATIVE_MODEL_ID:
        return "opencua_native_screenshot"
    if trial.model_id == OPENCUA_DIRECT_MCP_MODEL_ID:
        return "opencua_direct_mcp_tools"
    return str(_read_json(trial.summary_path).get("track") or "")


def aggregate_by_form_run(label: str, trials: Sequence[Trial]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Trial]] = defaultdict(list)
    for trial in trials:
        grouped[(trial.model_id, trial.form_id, trial.answer_run_id)].append(trial)
    rows = []
    for (model_id, form_id, answer_run_id), items in sorted(grouped.items()):
        row = aggregate(label, items)
        row["model_id"] = model_id
        row["form_id"] = form_id
        row["answer_run_id"] = answer_run_id
        rows.append(row)
    return rows


def trials_by_model_scope(label: str, trials: Sequence[Trial]) -> Dict[str, Sequence[Trial]]:
    grouped: Dict[str, List[Trial]] = defaultdict(list)
    for trial in trials:
        grouped[trial.model_id].append(trial)
    return {f"{label}:{model_id}": items for model_id, items in grouped.items()}


def failure_summary_rows(cohorts: Dict[str, List[Trial]]) -> List[Dict[str, Any]]:
    rows = []
    for cohort, trials in cohorts.items():
        if not trials:
            continue
        grouped: Dict[Tuple[str, str], List[Trial]] = defaultdict(list)
        for trial in trials:
            grouped[(trial.model_id, trial.stop_reason)].append(trial)
        for (model_id, stop_reason), items in sorted(grouped.items()):
            rows.append(
                {
                    "model_id": model_id,
                    "model_kind": str(_read_json(items[0].summary_path).get("model_kind") or ""),
                    "track": str(_read_json(items[0].summary_path).get("track") or ""),
                    "stop_reason": stop_reason,
                    "analysis_scope": cohort,
                    "trials": len(items),
                    "share": _safe_rate(len(items), len(trials)),
                    "submit_successes": sum(1 for item in items if item.submit_success),
                    "successes": sum(1 for item in items if item.success),
                    "avg_action_count": _safe_rate(sum(item.action_count for item in items), len(items)),
                    "scored_correctness": sum(item.scored_correctness for item in items),
                    "question_total": sum(item.question_total for item in items),
                    "scored_accuracy": _safe_rate(
                        sum(item.scored_correctness for item in items),
                        sum(item.question_total for item in items),
                    ),
                }
            )
    return rows


def _annotation_path(summary_path: Path) -> Path:
    return summary_path.with_name("annotations.json")


def _model_io_path(summary_path: Path) -> Path:
    return summary_path.with_name("model_io.jsonl")


def _answers_instance_path(summary_path: Path) -> Path:
    return summary_path.with_name("answers_instance.json")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return rows
    for line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            data = json.loads(text)
        except Exception:
            continue
        if isinstance(data, dict):
            rows.append(data)
    return rows


def _terminal_decision_type(text: Any) -> str:
    normalized = str(text or "").strip().lower()
    if normalized.startswith("done"):
        return "done"
    if normalized.startswith("stop"):
        return "stop"
    return ""


def _model_io_action_records(trial: Trial) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in _read_jsonl(_model_io_path(trial.summary_path)):
        if str(row.get("phase") or "") != "step":
            continue
        tool_calls = row.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                name = str(call.get("name") or "").strip()
                if not name:
                    continue
                records.append(
                    {
                        "action_type": name,
                        "action_group": "mcp_tool_call",
                        "is_executable_action": True,
                        "source": "model_io",
                    }
                )
            continue
        terminal_action = _terminal_decision_type(row.get("assistant_text"))
        if terminal_action:
            records.append(
                {
                    "action_type": terminal_action,
                    "action_group": "terminal_decision",
                    "is_executable_action": False,
                    "source": "model_io",
                }
            )
    return records


def _annotation_action_records(trial: Trial) -> List[Dict[str, Any]]:
    annotations = _read_json(_annotation_path(trial.summary_path))
    steps = annotations.get("steps")
    if not isinstance(steps, list):
        return []
    records: List[Dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        action_name = ""
        if isinstance(action, dict):
            action_name = str(action.get("action") or action.get("type") or action.get("name") or "").strip()
        elif isinstance(action, str):
            action_name = action.strip()
        if not action_name:
            terminal_action = _terminal_decision_type(step.get("raw_output") or step.get("assistant_text"))
            if terminal_action:
                records.append(
                    {
                        "action_type": terminal_action,
                        "action_group": "terminal_decision",
                        "is_executable_action": False,
                        "source": "annotations",
                    }
                )
            continue
        records.append(
            {
                "action_type": action_name,
                "action_group": "native_gui_action",
                "is_executable_action": action_name not in {"done", "stop"},
                "source": "annotations",
            }
        )
    return records


def _trial_model_action_records(trial: Trial) -> List[Dict[str, Any]]:
    records = _model_io_action_records(trial)
    if records:
        return records
    records = _annotation_action_records(trial)
    if records:
        return records
    if trial.action_count > 0:
        return [
            {
                "action_type": "unknown_model_action",
                "action_group": "fallback_summary_count",
                "is_executable_action": True,
                "source": "summary_action_count",
            }
            for _idx in range(trial.action_count)
        ]
    return []


def model_action_metadata_rows(
    trials_by_scope: Dict[str, Sequence[Trial]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    scope_stats: Dict[str, Dict[str, Any]] = {}
    all_action_types: set[str] = set()
    trial_rows: List[Dict[str, Any]] = []

    for scope, trials in sorted(trials_by_scope.items()):
        trials = list(trials)
        if not trials:
            continue
        model_ids = sorted({trial.model_id for trial in trials})
        model_id = model_ids[0] if len(model_ids) == 1 else ", ".join(model_ids)
        display_label = _model_label(model_id)
        type_counts: Counter[str] = Counter()
        executable_type_counts: Counter[str] = Counter()
        groups_by_type: Dict[str, str] = {}
        executable_totals: List[int] = []
        terminal_totals: List[int] = []
        source_counts: Counter[str] = Counter()

        for trial in trials:
            records = _trial_model_action_records(trial)
            action_counts = Counter(str(record["action_type"]) for record in records)
            trial_sources = Counter(str(record.get("source") or "") for record in records)
            executable_count = sum(1 for record in records if bool(record.get("is_executable_action")))
            terminal_count = sum(1 for record in records if not bool(record.get("is_executable_action")))
            executable_totals.append(executable_count)
            terminal_totals.append(terminal_count)
            type_counts.update(action_counts)
            for record in records:
                action_type = str(record["action_type"])
                all_action_types.add(action_type)
                groups_by_type.setdefault(action_type, str(record.get("action_group") or ""))
                source_counts.update([str(record.get("source") or "")])
                if bool(record.get("is_executable_action")):
                    executable_type_counts.update([action_type])
            trial_rows.append(
                {
                    "display_label": display_label,
                    "model_id": model_id,
                    "analysis_scope": scope,
                    "experiment_id": trial.experiment_id,
                    "form_id": trial.form_id,
                    "answer_run_id": trial.answer_run_id,
                    "trial_id": trial.trial_id,
                    "executable_action_count": executable_count,
                    "terminal_decision_count": terminal_count,
                    "action_counts_json": json.dumps(dict(sorted(action_counts.items())), sort_keys=True),
                    "action_source": ", ".join(k for k, v in trial_sources.items() if k and v),
                    "summary_path": str(trial.summary_path),
                }
            )

        scope_stats[scope] = {
            "display_label": display_label,
            "model_id": model_id,
            "analysis_scope": scope,
            "trials": len(trials),
            "type_counts": type_counts,
            "executable_type_counts": executable_type_counts,
            "groups_by_type": groups_by_type,
            "executable_totals": executable_totals,
            "terminal_totals": terminal_totals,
            "source_counts": source_counts,
        }

    action_type_rows: List[Dict[str, Any]] = []
    for scope, stats in sorted(scope_stats.items()):
        executable_total = sum(stats["executable_type_counts"].values())
        for action_type, count in stats["type_counts"].most_common():
            executable_count = int(stats["executable_type_counts"].get(action_type, 0))
            action_type_rows.append(
                {
                    "display_label": stats["display_label"],
                    "model_id": stats["model_id"],
                    "analysis_scope": scope,
                    "trials": stats["trials"],
                    "action_type": action_type,
                    "action_group": stats["groups_by_type"].get(action_type, ""),
                    "count": int(count),
                    "executable_count": executable_count,
                    "mean_per_trial": round(float(count) / float(stats["trials"]), 6) if stats["trials"] else 0.0,
                    "share_of_executable_actions": round(_safe_rate(executable_count, executable_total), 6),
                }
            )

    matrix_columns = sorted(all_action_types)
    matrix_rows: List[Dict[str, Any]] = []
    for scope, stats in sorted(scope_stats.items()):
        executable_total = sum(stats["executable_totals"])
        terminal_total = sum(stats["terminal_totals"])
        row: Dict[str, Any] = {
            "display_label": stats["display_label"],
            "model_id": stats["model_id"],
            "analysis_scope": scope,
            "trials": stats["trials"],
            "total_executable_actions": executable_total,
            "mean_executable_actions_per_trial": round(_safe_rate(executable_total, stats["trials"]), 6),
            "median_executable_actions_per_trial": _median([float(value) for value in stats["executable_totals"]]),
            "total_terminal_decisions": terminal_total,
            "mean_terminal_decisions_per_trial": round(_safe_rate(terminal_total, stats["trials"]), 6),
            "action_sources": ", ".join(k for k, v in stats["source_counts"].items() if k and v),
            "top_action_types": "; ".join(
                f"{action_type}:{count}" for action_type, count in stats["executable_type_counts"].most_common(6)
            ),
        }
        for action_type in matrix_columns:
            row[action_type] = int(stats["type_counts"].get(action_type, 0))
        matrix_rows.append(row)

    return action_type_rows, matrix_rows, trial_rows


def _answer_run_index(answer_run_id: str) -> Optional[int]:
    try:
        return int(str(answer_run_id).split("_", 1)[1])
    except Exception:
        return None


def _iter_answer_runs(path: Path) -> List[Dict[str, Any]]:
    data = _read_json(path)
    runs = data.get("runs") if isinstance(data, dict) else None
    if runs is None and isinstance(data, dict) and isinstance(data.get("answers"), list):
        runs = [data]
    if runs is None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if isinstance(raw, list):
            runs = raw
    if not isinstance(runs, list):
        return []
    normalized = []
    for item in runs:
        if isinstance(item, dict) and isinstance(item.get("answers"), list):
            normalized.append(item)
        elif isinstance(item, list):
            normalized.append({"answers": item})
    return normalized


def _target_value(answer: Dict[str, Any]) -> Any:
    if "target_value" in answer:
        return answer.get("target_value")
    return answer.get("value")


def _answer_validation(trial: Trial) -> Dict[str, Any]:
    source_path = DEFAULT_ANSWERS_ROOT / trial.form_id / "runs.json"
    answer_index = _answer_run_index(trial.answer_run_id)
    expected_answers: List[Dict[str, Any]] = []
    status = "ok"
    detail = ""
    if answer_index is None:
        status = "bad_answer_run_id"
        detail = trial.answer_run_id
    elif not source_path.exists():
        status = "missing_source_answers"
        detail = str(source_path)
    else:
        runs = _iter_answer_runs(source_path)
        if not (1 <= answer_index <= len(runs)):
            status = "run_index_out_of_range"
            detail = f"index={answer_index}, available={len(runs)}"
        else:
            expected = runs[answer_index - 1].get("answers")
            if isinstance(expected, list):
                expected_answers = [item for item in expected if isinstance(item, dict)]
            else:
                status = "source_answers_not_list"
    recorded_path = _answers_instance_path(trial.summary_path)
    recorded_answers: List[Dict[str, Any]] = []
    if status == "ok":
        try:
            recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
        except Exception as exc:
            status = "missing_or_invalid_recorded_answers"
            detail = str(exc)
            recorded = []
        if isinstance(recorded, list):
            recorded_answers = [item for item in recorded if isinstance(item, dict)]
        elif status == "ok":
            status = "recorded_answers_not_list"
    if status == "ok" and len(expected_answers) != len(recorded_answers):
        status = "answer_count_mismatch"
        detail = f"expected={len(expected_answers)}, recorded={len(recorded_answers)}"
    if status == "ok":
        for idx, (expected, recorded) in enumerate(zip(expected_answers, recorded_answers), start=1):
            expected_qid = str(expected.get("question_id") or expected.get("id") or "")
            recorded_qid = str(recorded.get("question_id") or recorded.get("id") or "")
            if expected_qid and recorded_qid and expected_qid != recorded_qid:
                status = "question_id_mismatch"
                detail = f"question={idx}, expected={expected_qid}, recorded={recorded_qid}"
                break
            if expected.get("value") != _target_value(recorded):
                status = "answer_value_mismatch"
                detail = f"question={idx}"
                break
    return {
        "answers_source_path": _rel(source_path),
        "answers_source_index": "" if answer_index is None else answer_index,
        "answers_instance_path": _rel(recorded_path),
        "expected_answer_count": len(expected_answers),
        "recorded_answer_count": len(recorded_answers),
        "answer_set_status": status,
        "answer_set_detail": detail,
    }


def _metadata_path_status(trial: Trial) -> str:
    summary = _read_json(trial.summary_path)
    try:
        parts = trial.summary_path.relative_to(DEFAULT_DATASET_ROOT).parts
    except ValueError:
        return "path_outside_dataset_root"
    checks = {
        "experiment_id": parts[0] if len(parts) > 0 else "",
        "model_id": parts[1] if len(parts) > 1 else "",
        "form_id": parts[2] if len(parts) > 2 else "",
        "answer_run_id": parts[3] if len(parts) > 3 else "",
        "trial_id": parts[4] if len(parts) > 4 else "",
    }
    mismatches = [
        key
        for key, expected in checks.items()
        if summary.get(key) is not None and str(summary.get(key)) != expected
    ]
    return "ok" if not mismatches else "mismatch:" + ",".join(mismatches)


def question_type_rows(trials: Sequence[Trial], cohort: str, reliable_only: bool = False) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for trial in trials:
        # For Qwen submitted trials, final question-level verification often runs
        # on the confirmation page. Keep those out of reliable type-level tables.
        if reliable_only and trial.submit_success:
            continue
        annotations = _read_json(_annotation_path(trial.summary_path))
        questions = annotations.get("questions")
        if not isinstance(questions, list):
            continue
        reliability = "final_verification"
        if trial.submit_success:
            reliability = "post_submit_final_page_caveat"
        for question in questions:
            if not isinstance(question, dict):
                continue
            widget_type = str(question.get("widget_type") or "unknown")
            key = (trial.model_id, widget_type)
            row = grouped.setdefault(
                key,
                {
                    "model_id": trial.model_id,
                    "model_kind": str(_read_json(trial.summary_path).get("model_kind") or ""),
                    "track": str(_read_json(trial.summary_path).get("track") or ""),
                    "widget_type": widget_type,
                    "analysis_scope": cohort,
                    "questions": 0,
                    "attempted": 0,
                    "attempted_correct": 0,
                    "verified": 0,
                    "verified_correct": 0,
                    "failed": 0,
                    "not_attempted": 0,
                    "container_not_visible": 0,
                    "reliability": reliability,
                },
            )
            row["questions"] += 1
            attempted = bool(question.get("attempted"))
            attempted_correct = bool(question.get("attempted_correct"))
            verified = bool(question.get("verified"))
            verified_correct = bool(question.get("verified_correct"))
            row["attempted"] += int(attempted)
            row["attempted_correct"] += int(attempted_correct)
            row["verified"] += int(verified)
            row["verified_correct"] += int(verified_correct)
            final_status = str(question.get("final_status") or "")
            if final_status not in {"correct_verified", "correct_attempted"}:
                row["failed"] += 1
            if final_status == "not_attempted":
                row["not_attempted"] += 1
            detail = ""
            last_verification = question.get("last_verification")
            if isinstance(last_verification, dict):
                detail = str(last_verification.get("detail") or "")
            if detail == "container_not_visible":
                row["container_not_visible"] += 1
            if row["reliability"] != reliability:
                row["reliability"] = "mixed"
    rows = []
    for row in grouped.values():
        total = row["questions"]
        row["attempt_rate"] = _safe_rate(row["attempted"], total)
        row["verified_accuracy"] = _safe_rate(row["verified_correct"], total)
        row["failure_rate"] = _safe_rate(row["failed"], total)
        rows.append(row)
    return sorted(rows, key=lambda item: (item["model_id"], item["widget_type"], item["analysis_scope"]))


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def canonical_trial_rows(trials: Sequence[Trial]) -> List[Dict[str, Any]]:
    rows = []
    for trial in sorted(trials, key=lambda item: (item.model_id, item.form_id, item.answer_run_id, item.trial_id)):
        answer_validation = _answer_validation(trial)
        if trial.experiment_id == QWEN_EXPERIMENT:
            analysis_family = "qwen_direct_mcp"
        elif trial.model_id == OPENCUA_NATIVE_MODEL_ID:
            analysis_family = "opencua_native"
        elif trial.model_id == OPENCUA_DIRECT_MCP_MODEL_ID:
            analysis_family = "opencua_direct_mcp"
        else:
            analysis_family = "other"
        rows.append(
            {
                "model_id": trial.model_id,
                "model_kind": _read_json(trial.summary_path).get("model_kind", ""),
                "track": _read_json(trial.summary_path).get("track", ""),
                "interface_condition": _interface_condition(trial),
                "form_id": trial.form_id,
                "answer_run_id": trial.answer_run_id,
                "trial_id": trial.trial_id,
                "analysis_family": analysis_family,
                "experiment_id": trial.experiment_id,
                "success": trial.success,
                "submit_success": trial.submit_success,
                "stop_reason": trial.stop_reason,
                "failure_category": trial.failure_category,
                "question_total": trial.question_total,
                "scored_correctness": trial.scored_correctness,
                "scored_accuracy": _safe_rate(trial.scored_correctness, trial.question_total),
                "exact_success": _trial_exact_success(trial),
                "scored_source": trial.scored_source,
                "verified_correctness": trial.verified_correctness,
                "pre_successful_submit_correctness": "" if trial.pre_successful_submit_correctness is None else trial.pre_successful_submit_correctness,
                "pre_first_submit_correctness": "" if trial.pre_first_submit_correctness is None else trial.pre_first_submit_correctness,
                "submit_attempt_count": trial.submit_attempt_count,
                "action_count": trial.action_count,
                "duration_s": trial.duration_s,
                "reference_available": trial.reference_available,
                "reference_action_count": "" if trial.reference_action_count is None else trial.reference_action_count,
                "reference_duration_s": "" if trial.reference_duration_s is None else trial.reference_duration_s,
                "action_overhead_ratio": "" if trial.action_overhead_ratio is None else trial.action_overhead_ratio,
                "time_overhead_ratio": "" if trial.time_overhead_ratio is None else trial.time_overhead_ratio,
                "action_count_delta": "" if trial.action_count_delta is None else trial.action_count_delta,
                "duration_delta_s": "" if trial.duration_delta_s is None else trial.duration_delta_s,
                "actions_per_correct_answer": "" if not trial.scored_correctness else round(float(trial.action_count) / float(trial.scored_correctness), 6),
                "correct_answers_per_min": "" if not trial.duration_s else round(float(trial.scored_correctness) / (float(trial.duration_s) / 60.0), 6),
                "wasted_interaction_rate": "" if trial.wasted_interaction_rate is None else trial.wasted_interaction_rate,
                "reference_warning": "" if trial.reference_available else "missing_reference_run",
                "run_completed_utc": trial.run_completed_utc,
                "metadata_path_status": _metadata_path_status(trial),
                **answer_validation,
                "summary_path": _rel(trial.summary_path),
                "annotations_path": _rel(_annotation_path(trial.summary_path)),
                "reference_run_path": _rel(Path(trial.reference_run_path)) if trial.reference_run_path else "",
                "reference_trace_path": _rel(Path(trial.reference_trace_path)) if trial.reference_trace_path else "",
                "reference_video_path": trial.reference_video_path,
            }
        )
    return rows


def answer_validation_rows(trials: Sequence[Trial]) -> List[Dict[str, Any]]:
    rows = []
    for trial in sorted(trials, key=lambda item: (item.model_id, item.form_id, item.answer_run_id, item.trial_id)):
        rows.append(
            {
                "model_id": trial.model_id,
                "form_id": trial.form_id,
                "answer_run_id": trial.answer_run_id,
                "trial_id": trial.trial_id,
                "experiment_id": trial.experiment_id,
                "metadata_path_status": _metadata_path_status(trial),
                **_answer_validation(trial),
                "summary_path": _rel(trial.summary_path),
            }
        )
    return rows


def reference_coverage_rows(forms_root: Path = DEFAULT_FORMS_ROOT) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for form_id in _all_form_ids(forms_root):
        for run_id in TARGET_RUN_IDS:
            metrics = _reference_metrics(form_id, run_id)
            rows.append(
                {
                    "form_id": form_id,
                    "answer_run_id": run_id,
                    "reference_available": metrics["reference_available"],
                    "reference_artifacts_present": metrics["reference_artifacts_present"],
                    "reference_video_available": metrics["reference_video_available"],
                    "reference_action_count": "" if metrics["reference_action_count"] is None else metrics["reference_action_count"],
                    "reference_duration_s": "" if metrics["reference_duration_s"] is None else metrics["reference_duration_s"],
                    "reference_interaction_mode": metrics["reference_interaction_mode"],
                    "reference_trace_mode": metrics["reference_trace_mode"],
                    "reference_submit_success": metrics["reference_submit_success"],
                    "reference_warning": "" if metrics["reference_available"] else "missing_or_unusable_reference",
                    "reference_run_path": _rel(Path(metrics["reference_run_path"])),
                    "reference_trace_path": _rel(Path(metrics["reference_trace_path"])),
                    "reference_video_path": metrics["reference_video_path"],
                }
            )
    return rows


def _ci_columns(prefix: str, values: Sequence[float]) -> Dict[str, Any]:
    low, high = _bootstrap_ci(values)
    return {
        f"{prefix}_mean": "" if _mean(values) is None else round(float(_mean(values) or 0.0), 6),
        f"{prefix}_ci95_low": "" if low is None else round(float(low), 6),
        f"{prefix}_ci95_high": "" if high is None else round(float(high), 6),
    }


def efficiency_summary_rows(model_rows: Sequence[Dict[str, Any]], trials_by_scope: Dict[str, Sequence[Trial]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for base_row in model_rows:
        scope = str(base_row.get("analysis_scope") or "")
        trials = list(trials_by_scope.get(scope, []))
        if not trials:
            continue
        for subset_label, subset_trials in [
            ("all_trials", trials),
            ("submitted_trials", [trial for trial in trials if trial.submit_success]),
            ("exact_success_trials", [trial for trial in trials if _trial_exact_success(trial)]),
        ]:
            action_overheads = [float(t.action_overhead_ratio) for t in subset_trials if t.action_overhead_ratio is not None]
            time_overheads = [float(t.time_overhead_ratio) for t in subset_trials if t.time_overhead_ratio is not None]
            submit_values = [1.0 if t.submit_success else 0.0 for t in subset_trials]
            accuracy_values = [_trial_accuracy(t) for t in subset_trials]
            exact_values = [1.0 if _trial_exact_success(t) else 0.0 for t in subset_trials]
            wasted_rates = [float(t.wasted_interaction_rate) for t in subset_trials if t.wasted_interaction_rate is not None]
            rows.append(
                {
                    "model_id": base_row.get("model_id", ""),
                    "model_kind": base_row.get("model_kind", ""),
                    "track": base_row.get("track", ""),
                    "interface_condition": base_row.get("interface_condition", ""),
                    "analysis_scope": scope,
                    "efficiency_subset": subset_label,
                    "trials": len(subset_trials),
                    "reference_available_trials": sum(1 for t in subset_trials if t.reference_available),
                    "reference_availability_rate": _safe_rate(sum(1 for t in subset_trials if t.reference_available), len(subset_trials)),
                    "exact_success_rate": _safe_rate(sum(1 for t in subset_trials if _trial_exact_success(t)), len(subset_trials)),
                    "submit_rate": _safe_rate(sum(1 for t in subset_trials if t.submit_success), len(subset_trials)),
                    "scored_accuracy": _safe_rate(sum(t.scored_correctness for t in subset_trials), sum(t.question_total for t in subset_trials)),
                    "median_model_action_count": _median([float(t.action_count) for t in subset_trials if t.action_count]),
                    "median_reference_action_count": _median([float(t.reference_action_count) for t in subset_trials if t.reference_action_count is not None]),
                    "median_model_duration_s": _median([float(t.duration_s) for t in subset_trials if t.duration_s]),
                    "median_reference_duration_s": _median([float(t.reference_duration_s) for t in subset_trials if t.reference_duration_s is not None]),
                    "median_action_overhead_ratio": _median(action_overheads),
                    "p25_action_overhead_ratio": _quantile(action_overheads, 0.25),
                    "p75_action_overhead_ratio": _quantile(action_overheads, 0.75),
                    "median_time_overhead_ratio": _median(time_overheads),
                    "p25_time_overhead_ratio": _quantile(time_overheads, 0.25),
                    "p75_time_overhead_ratio": _quantile(time_overheads, 0.75),
                    "mean_wasted_interaction_rate": _mean(wasted_rates),
                    "reference_warning": "" if all(t.reference_available for t in subset_trials) else "some_trials_missing_reference",
                    **_ci_columns("submit_rate", submit_values),
                    **_ci_columns("exact_success_rate", exact_values),
                    **_ci_columns("scored_accuracy", accuracy_values),
                    **_ci_columns("action_overhead_ratio", action_overheads),
                    **_ci_columns("time_overhead_ratio", time_overheads),
                }
            )
    return rows


def paired_efficiency_rows(trials: Sequence[Trial]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, Trial]] = defaultdict(dict)
    for trial in trials:
        if trial.answer_run_id not in TARGET_RUN_IDS:
            continue
        grouped[(trial.form_id, trial.answer_run_id)][trial.model_id] = trial
    pairs = [
        ("qwen_vlm_minus_text", "vlm_qwen3_vl_30b_a3b_instruct", "text_qwen3_30b_a3b_instruct_2507"),
        ("opencua_direct_mcp_minus_native", OPENCUA_DIRECT_MCP_MODEL_ID, OPENCUA_NATIVE_MODEL_ID),
    ]
    rows: List[Dict[str, Any]] = []
    for (form_id, answer_run_id), by_model in sorted(grouped.items()):
        for comparison, treatment_id, baseline_id in pairs:
            treatment = by_model.get(treatment_id)
            baseline = by_model.get(baseline_id)
            if treatment is None or baseline is None:
                continue
            rows.append(
                {
                    "comparison": comparison,
                    "form_id": form_id,
                    "answer_run_id": answer_run_id,
                    "treatment_model_id": treatment_id,
                    "baseline_model_id": baseline_id,
                    "treatment_submit_success": treatment.submit_success,
                    "baseline_submit_success": baseline.submit_success,
                    "submit_success_delta": int(treatment.submit_success) - int(baseline.submit_success),
                    "treatment_exact_success": _trial_exact_success(treatment),
                    "baseline_exact_success": _trial_exact_success(baseline),
                    "exact_success_delta": int(_trial_exact_success(treatment)) - int(_trial_exact_success(baseline)),
                    "treatment_scored_accuracy": _trial_accuracy(treatment),
                    "baseline_scored_accuracy": _trial_accuracy(baseline),
                    "scored_accuracy_delta": _trial_accuracy(treatment) - _trial_accuracy(baseline),
                    "treatment_action_overhead_ratio": "" if treatment.action_overhead_ratio is None else treatment.action_overhead_ratio,
                    "baseline_action_overhead_ratio": "" if baseline.action_overhead_ratio is None else baseline.action_overhead_ratio,
                    "action_overhead_delta": "" if treatment.action_overhead_ratio is None or baseline.action_overhead_ratio is None else treatment.action_overhead_ratio - baseline.action_overhead_ratio,
                    "treatment_time_overhead_ratio": "" if treatment.time_overhead_ratio is None else treatment.time_overhead_ratio,
                    "baseline_time_overhead_ratio": "" if baseline.time_overhead_ratio is None else baseline.time_overhead_ratio,
                    "time_overhead_delta": "" if treatment.time_overhead_ratio is None or baseline.time_overhead_ratio is None else treatment.time_overhead_ratio - baseline.time_overhead_ratio,
                    "reference_available_both": bool(treatment.reference_available and baseline.reference_available),
                }
            )
    return rows


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt_float(value: float) -> str:
    return f"{value:.1f}"


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_id",
        "model_kind",
        "track",
        "interface_condition",
        "form_id",
        "answer_run_id",
        "analysis_scope",
        "trials",
        "forms",
        "successes",
        "exact_successes",
        "submit_successes",
        "success_rate",
        "exact_success_rate",
        "submit_rate",
        "scored_correctness",
        "question_total",
        "scored_accuracy",
        "mean_trial_accuracy",
        "pre_submit_answered",
        "pre_submit_questions",
        "pre_submit_accuracy",
        "submit_attempt_count",
        "submitted_while_incomplete_count",
        "avg_action_count",
        "avg_duration_s",
        "reference_available_trials",
        "reference_availability_rate",
        "median_reference_action_count",
        "median_reference_duration_s",
        "median_action_overhead_ratio",
        "p25_action_overhead_ratio",
        "p75_action_overhead_ratio",
        "median_time_overhead_ratio",
        "p25_time_overhead_ratio",
        "p75_time_overhead_ratio",
        "mean_wasted_interaction_rate",
        "median_actions_per_correct_answer",
        "median_correct_answers_per_min",
        "stop_reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["stop_reasons"] = json.dumps(dict(row.get("stop_reasons") or {}), sort_keys=True)
            writer.writerow(out)


def _write_question_type_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_id",
        "model_kind",
        "track",
        "widget_type",
        "analysis_scope",
        "questions",
        "attempted",
        "attempted_correct",
        "verified",
        "verified_correct",
        "failed",
        "not_attempted",
        "container_not_visible",
        "attempt_rate",
        "verified_accuracy",
        "failure_rate",
        "reliability",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_dict_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#202124}.title{font-size:17px;font-weight:700}.axis{fill:#5f6368}.small{font-size:10px;fill:#5f6368}.label{font-size:12px;font-weight:700}.legend{font-size:11px;fill:#3c4043}</style>',
        '<rect width="100%" height="100%" fill="#fff"/>',
    ]


def _truncate_label(value: Any, limit: int = 28) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "..."


def _svg_label(x: float, y: float, value: Any, *, css_class: str = "", limit: int = 32) -> str:
    text = _truncate_label(value, limit)
    class_attr = f' class="{css_class}"' if css_class else ""
    title = "" if text == str(value or "") else f"<title>{html.escape(str(value or ''))}</title>"
    return f'<text{class_attr} x="{x:.1f}" y="{y:.1f}">{html.escape(text)}{title}</text>'


def _pct_label(value: Any) -> str:
    numeric = _optional_float(value)
    return "n/a" if numeric is None else _pct(numeric)


def _value_label(value: Any, suffix: str = "", digits: int = 1) -> str:
    numeric = _optional_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.{digits}f}{suffix}"


def _scale_cap(values: Sequence[float], minimum: float = 1.0, headroom: float = 1.15) -> float:
    max_value = max(values, default=minimum)
    return max(minimum, max_value * headroom)


def _clamped_bar_width(value: float, cap: float, chart_w: float) -> Tuple[float, bool]:
    if cap <= 0:
        return 0.0, False
    overflow = value > cap
    width = min(max(value, 0.0), cap) / cap * chart_w
    return width, overflow


def write_thesis_effectiveness_overview(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    rows = [row for row in rows if _as_int(row.get("trials"))]
    width = 1080
    row_h = 82
    top = 88
    left = 190
    chart_w = 650
    value_x = 865
    height = top + len(rows) * row_h + 86
    metrics = [
        ("submit_rate", "Submit", "#2563eb"),
        ("exact_success_rate", "Exact", "#16a34a"),
        ("scored_accuracy", "Score", "#f59e0b"),
    ]
    lines = _svg_header(width, height)
    lines.append('<text class="title" x="24" y="30">Thesis Effectiveness Overview</text>')
    lines.append('<text class="axis" x="24" y="52">One averaged result per model/interface condition</text>')
    for idx, (_key, label, color) in enumerate(metrics):
        x = left + idx * 105
        lines.append(f'<rect x="{x}" y="68" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text class="legend" x="{x + 18}" y="79">{label}</text>')
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        full_label = row.get("display_label") or row.get("model_id") or ""
        lines.append(_svg_label(24, y + 16, full_label, css_class="label", limit=22))
        lines.append(f'<text class="small" x="24" y="{y + 34}">n={_as_int(row.get("trials"))}, forms={_as_int(row.get("forms"))}</text>')
        for offset, (key, label, color) in enumerate(metrics):
            bar_y = y + offset * 20
            value = max(0.0, min(1.0, float(row.get(key) or 0.0)))
            lines.append(f'<rect x="{left}" y="{bar_y}" width="{chart_w}" height="13" fill="#eef1f4"/>')
            lines.append(f'<rect x="{left}" y="{bar_y}" width="{value * chart_w:.1f}" height="13" fill="{color}"/>')
            lines.append(f'<text class="small" x="{left - 54}" y="{bar_y + 11}">{label}</text>')
            lines.append(f'<text class="small" x="{value_x}" y="{bar_y + 11}">{_pct(value)}</text>')
    axis_y = top + len(rows) * row_h + 20
    lines.append(f'<line x1="{left}" y1="{axis_y}" x2="{left + chart_w}" y2="{axis_y}" stroke="#dadce0"/>')
    for tick in range(0, 6):
        x = left + chart_w * tick / 5
        lines.append(f'<line x1="{x:.1f}" y1="{axis_y}" x2="{x:.1f}" y2="{axis_y + 5}" stroke="#dadce0"/>')
        lines.append(f'<text class="small" x="{x - 10:.1f}" y="{axis_y + 20}">{tick * 20}%</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_thesis_efficiency_overview(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    rows = [row for row in rows if _as_int(row.get("trials"))]
    width = 1120
    row_h = 102
    top = 98
    left = 205
    chart_w = 620
    value_x = 850
    height = top + len(rows) * row_h + 92
    metrics = [
        ("median_action_overhead", "Action x", "#7c3aed", "x", 2),
        ("median_time_overhead", "Time x", "#0f766e", "x", 2),
        ("median_action_count", "Actions", "#2563eb", "", 1),
        ("median_duration_s", "Seconds", "#f97316", "s", 1),
    ]
    numeric_by_key = {
        key: [float(v) for v in (_optional_float(row.get(key)) for row in rows) if v is not None]
        for key, *_rest in metrics
    }
    caps = {key: _scale_cap(values, minimum=1.0) for key, values in numeric_by_key.items()}
    lines = _svg_header(width, height)
    lines.append('<text class="title" x="24" y="30">Thesis Efficiency Overview</text>')
    lines.append('<text class="axis" x="24" y="52">Median costs; bars are independently scaled by metric and labels show actual values</text>')
    legend_x = left
    for idx, (_key, label, color, _suffix, _digits) in enumerate(metrics):
        x = legend_x + idx * 112
        lines.append(f'<rect x="{x}" y="70" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text class="legend" x="{x + 18}" y="81">{label}</text>')
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        label = row.get("display_label") or row.get("model_id") or ""
        lines.append(_svg_label(24, y + 18, label, css_class="label", limit=22))
        lines.append(f'<text class="small" x="24" y="{y + 36}">refs={_as_int(row.get("reference_available_trials"))}/{_as_int(row.get("trials"))}</text>')
        for offset, (key, metric_label, color, suffix, digits) in enumerate(metrics):
            bar_y = y + offset * 20
            numeric = _optional_float(row.get(key))
            lines.append(f'<text class="small" x="{left - 64}" y="{bar_y + 11}">{metric_label}</text>')
            lines.append(f'<rect x="{left}" y="{bar_y}" width="{chart_w}" height="13" fill="#eef1f4"/>')
            if numeric is None:
                lines.append(f'<text class="small" x="{value_x}" y="{bar_y + 11}">n/a</text>')
                continue
            bar_w, overflow = _clamped_bar_width(float(numeric), caps[key], chart_w)
            lines.append(f'<rect x="{left}" y="{bar_y}" width="{max(1.0, bar_w):.1f}" height="13" fill="{color}"/>')
            suffix_text = "+" if overflow else ""
            lines.append(f'<text class="small" x="{value_x}" y="{bar_y + 11}">{numeric:.{digits}f}{suffix}{suffix_text}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_thesis_failure_mix(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    rows = [row for row in rows if _as_int(row.get("trials"))]
    reasons = sorted({reason for row in rows for reason in (row.get("stop_reasons") or Counter())})
    if not reasons:
        reasons = ["unknown"]
    palette = ["#2563eb", "#dc2626", "#f59e0b", "#16a34a", "#7c3aed", "#0891b2", "#64748b", "#f97316"]
    color = {reason: palette[idx % len(palette)] for idx, reason in enumerate(reasons)}
    width = 1120
    row_h = 62
    top = 82
    left = 205
    chart_w = 660
    value_x = 890
    legend_cols = 2
    legend_rows = math.ceil(len(reasons) / legend_cols)
    height = top + len(rows) * row_h + 42 + legend_rows * 20
    lines = _svg_header(width, height)
    lines.append('<text class="title" x="24" y="30">Thesis Failure Mix</text>')
    lines.append('<text class="axis" x="24" y="52">Trial stop reasons by model; each row sums to 100%</text>')
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        label = row.get("display_label") or row.get("model_id") or ""
        total = max(1, _as_int(row.get("trials")))
        stop_reasons = row.get("stop_reasons") or Counter()
        lines.append(_svg_label(24, y + 16, label, css_class="label", limit=22))
        lines.append(f'<text class="small" x="24" y="{y + 34}">n={total}</text>')
        x = left
        for reason in reasons:
            count = int(stop_reasons.get(reason, 0)) if hasattr(stop_reasons, "get") else 0
            if not count:
                continue
            w = chart_w * count / total
            lines.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="20" fill="{color[reason]}"/>')
            x += w
        lines.append(f'<rect x="{left}" y="{y}" width="{chart_w}" height="20" fill="none" stroke="#dadce0"/>')
        lines.append(f'<text class="small" x="{value_x}" y="{y + 15}">{html.escape(_truncate_label(_dominant_stop_reason(row), 26))}</text>')
    legend_y = top + len(rows) * row_h + 22
    for idx, reason in enumerate(reasons):
        col = idx % legend_cols
        row_idx = idx // legend_cols
        x = 24 + col * 430
        y = legend_y + row_idx * 20
        lines.append(f'<rect x="{x}" y="{y - 10}" width="10" height="10" fill="{color[reason]}"/>')
        lines.append(_svg_label(x + 18, y, reason, css_class="legend", limit=42))
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_thesis_efficiency_frontier(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    plot_rows = [row for row in rows if _optional_float(row.get("median_action_overhead")) is not None]
    width = 980
    height = 600
    left = 92
    top = 80
    chart_w = 690
    chart_h = 390
    x_values = [float(row["median_action_overhead"]) for row in plot_rows]
    x_cap = _scale_cap(x_values, minimum=1.0, headroom=1.2)
    lines = _svg_header(width, height)
    lines.append('<text class="title" x="24" y="30">Thesis Efficiency Frontier</text>')
    lines.append('<text class="axis" x="24" y="52">Higher exact success and lower action overhead is better</text>')
    lines.append(f'<rect x="{left}" y="{top}" width="{chart_w}" height="{chart_h}" fill="#fafafa" stroke="#dadce0"/>')
    for tick in range(0, 6):
        y = top + chart_h - tick * chart_h / 5
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#eef1f4"/>')
        lines.append(f'<text class="small" x="{left - 52}" y="{y + 4:.1f}">{tick * 20}%</text>')
    for tick in range(0, 5):
        x_value = x_cap * tick / 4
        x = left + chart_w * tick / 4
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + chart_h}" stroke="#f1f3f4"/>')
        lines.append(f'<text class="small" x="{x - 10:.1f}" y="{top + chart_h + 20}">{x_value:.1f}</text>')
    for row in plot_rows:
        x_value = float(row["median_action_overhead"])
        y_value = max(0.0, min(1.0, float(row.get("exact_success_rate") or 0.0)))
        x = left + min(x_value, x_cap) / x_cap * chart_w
        y = top + chart_h - y_value * chart_h
        label = str(row.get("display_label") or row.get("model_id") or "")
        color = THESIS_COLORS.get(label, "#2563eb")
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}"/>')
        label_x = min(x + 10, width - 150)
        label_y = max(top + 14, min(y + 4, top + chart_h - 8))
        lines.append(_svg_label(label_x, label_y, label, css_class="small", limit=22))
    lines.append(f'<text class="axis" x="{left + chart_w / 2 - 78:.1f}" y="{top + chart_h + 48}">Median action overhead ratio</text>')
    lines.append(f'<text class="axis" x="24" y="{top - 18}">Exact success rate</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_thesis_reference_coverage(rows: Sequence[Dict[str, Any]], reference_coverage: Sequence[Dict[str, Any]], path: Path) -> None:
    width = 1080
    row_h = 54
    top = 92
    left = 205
    chart_w = 650
    value_x = 880
    usable_refs = sum(1 for row in reference_coverage if str(row.get("reference_available")).lower() == "true" or row.get("reference_available") is True)
    total_refs = len(reference_coverage)
    plot_rows: List[Dict[str, Any]] = [
        {
            "display_label": "Overall Reference",
            "coverage_rate": _safe_rate(usable_refs, total_refs),
            "numerator": usable_refs,
            "denominator": total_refs,
            "color": "#475569",
        }
    ]
    for row in rows:
        plot_rows.append(
            {
                "display_label": row.get("display_label", ""),
                "coverage_rate": row.get("reference_coverage_rate", 0.0),
                "numerator": row.get("reference_available_trials", 0),
                "denominator": row.get("trials", 0),
                "color": THESIS_COLORS.get(str(row.get("display_label") or ""), "#2563eb"),
            }
        )
    height = top + len(plot_rows) * row_h + 58
    lines = _svg_header(width, height)
    lines.append('<text class="title" x="24" y="30">Thesis Reference Coverage</text>')
    lines.append('<text class="axis" x="24" y="52">Usable scripted references for efficiency comparisons</text>')
    for idx, row in enumerate(plot_rows):
        y = top + idx * row_h
        value = max(0.0, min(1.0, float(row.get("coverage_rate") or 0.0)))
        lines.append(_svg_label(24, y + 16, row.get("display_label", ""), css_class="label", limit=24))
        lines.append(f'<rect x="{left}" y="{y}" width="{chart_w}" height="18" fill="#eef1f4"/>')
        lines.append(f'<rect x="{left}" y="{y}" width="{value * chart_w:.1f}" height="18" fill="{row["color"]}"/>')
        lines.append(f'<text class="small" x="{value_x}" y="{y + 14}">{row.get("numerator", 0)}/{row.get("denominator", 0)} ({_pct(value)})</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_form_accuracy(rows: Sequence[Dict[str, Any]], path: Path, title: str) -> None:
    rows = [row for row in rows if row["trials"]]
    width = 1080
    height = 95 + len(rows) * 22
    left = 330
    chart_w = 620
    lines = _svg_header(width, height)
    lines.append(f'<text class="title" x="24" y="30">{html.escape(title)}</text>')
    lines.append('<text class="axis" x="24" y="52">Scored correctness per form/model</text>')
    for idx, row in enumerate(rows):
        y = 78 + idx * 22
        name = f"{row.get('model_id', '')}:{row.get('form_id', '')}:{row.get('answer_run_id', '')}"
        name = html.escape(str(name))
        value = float(row["scored_accuracy"])
        lines.append(f'<text class="small" x="24" y="{y + 11}">{name}</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{chart_w}" height="12" fill="#eef1f4"/>')
        lines.append(f'<rect x="{left}" y="{y}" width="{max(1, value * chart_w):.1f}" height="12" fill="#1a73e8"/>')
        lines.append(
            f'<text class="small" x="{left + chart_w + 8}" y="{y + 11}">{row["scored_correctness"]}/{row["question_total"]} ({_pct(value)})</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _thesis_effectiveness_md(rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "| Model | Trials | Forms | Target Coverage | Submit Rate | Exact Success | Scored Accuracy | Dominant Stop |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('display_label', '')} | {row.get('trials', 0)} | {row.get('forms', 0)} | "
            f"{row.get('observed_unique_form_runs', 0)}/{row.get('target_form_runs', 0)} ({_pct(row.get('target_coverage_rate', 0.0))}) | "
            f"{_pct(row.get('submit_rate', 0.0))} | {_pct(row.get('exact_success_rate', 0.0))} | "
            f"{_pct(row.get('scored_accuracy', 0.0))} | `{row.get('dominant_stop_reason', '')}` |"
        )
    return lines


def _thesis_efficiency_md(rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "| Model | Ref Coverage | Median Actions | Median Duration | Median Action Overhead | Median Time Overhead |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('display_label', '')} | "
            f"{row.get('reference_available_trials', 0)}/{row.get('trials', 0)} ({_pct(row.get('reference_coverage_rate', 0.0))}) | "
            f"{_fmt_optional(row.get('median_action_count'), 1)} | {_fmt_optional(row.get('median_duration_s'), 1)}s | "
            f"{_fmt_optional(row.get('median_action_overhead'), 2)}x | {_fmt_optional(row.get('median_time_overhead'), 2)}x |"
        )
    return lines


def _thesis_action_metadata_md(rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "| Model | Trials | Executable Actions | Median / Trial | Top Action Types |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('display_label', '')} | {row.get('trials', 0)} | "
            f"{row.get('total_executable_actions', 0)} | "
            f"{_fmt_optional(row.get('median_executable_actions_per_trial'), 1)} | "
            f"`{row.get('top_action_types', '')}` |"
        )
    return lines


def write_markdown(
    output_dir: Path,
    thesis_rows: Sequence[Dict[str, Any]],
    action_matrix_rows: Sequence[Dict[str, Any]],
    canonical_rows: Sequence[Dict[str, Any]],
    answer_validation: Sequence[Dict[str, Any]],
    experiment_coverage: Sequence[Dict[str, Any]],
    reference_coverage: Sequence[Dict[str, Any]],
    efficiency_summary: Sequence[Dict[str, Any]],
    paired_efficiency: Sequence[Dict[str, Any]],
    include_diagnostic_plots: bool,
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    answer_status = Counter(str(row.get("answer_set_status") or "") for row in answer_validation)
    metadata_status = Counter(str(row.get("metadata_path_status") or "") for row in answer_validation)
    partial_experiments = [row for row in experiment_coverage if row.get("status") == "in_progress_or_partial"]
    reference_available = sum(1 for row in reference_coverage if str(row.get("reference_available")).lower() == "true" or row.get("reference_available") is True)
    reference_video_available = sum(1 for row in reference_coverage if str(row.get("reference_video_available")).lower() == "true" or row.get("reference_video_available") is True)
    total_references = len(reference_coverage)
    model_reference_rates = [float(row.get("reference_coverage_rate") or 0.0) for row in thesis_rows]
    incomplete_reference_warning = bool(total_references and reference_available < total_references) or any(rate < 0.95 for rate in model_reference_rates)
    lines = [
        "# Evaluation Analysis",
        "",
        f"Last updated: {now}",
        "",
        "Generated by `scripts/analyze_eval_results.py` from `data/model_baselines/**/summary.json`.",
        "",
        "## Thesis Model Effectiveness",
        "",
        *_thesis_effectiveness_md(thesis_rows),
        "",
        "## Thesis Model Efficiency",
        "",
        "Efficiency values compare each model trial with the scripted Playwright reference sharing the same `form_id` and `answer_run_id`.",
        "",
        *_thesis_efficiency_md(thesis_rows),
        "",
        "## Model Action Metadata",
        "",
        "Action counts come from model-issued `model_io.jsonl` tool calls for MCP runs and `annotations.steps` actions for native GUI runs. Automatic trace-only observations such as browser screenshots are excluded unless they appear as explicit model tool calls.",
        "",
        *_thesis_action_metadata_md(action_matrix_rows),
        "",
        "## Reference Coverage Warning",
        "",
        (
            f"Efficiency results are provisional: `{reference_available}/{total_references}` reference form-runs are currently usable, "
            f"and model-level reference coverage is incomplete."
            if incomplete_reference_warning
            else f"Reference coverage is complete for the current target: `{reference_available}/{total_references}` usable reference form-runs."
        ),
        "",
        "## Thesis Plots",
        "",
        "- [Effectiveness overview](plots/thesis_effectiveness_overview.svg)",
        "- [Efficiency overview](plots/thesis_efficiency_overview.svg)",
        "- [Failure mix](plots/thesis_failure_mix.svg)",
        "- [Efficiency frontier](plots/thesis_efficiency_frontier.svg)",
        "- [Reference coverage](plots/thesis_reference_coverage.svg)",
        "",
        "## Notes",
        "",
        "- Thesis-primary outputs exclude latest-batch duplicate rows and keep one averaged row per primary model/interface condition.",
        "- `submitted_while_incomplete_count` stays out of headline metrics until its pre-submit logic is fixed.",
        "- For submitted Qwen trials, use `scored_correctness` rather than final page verification.",
        "",
        "## Partial Running Batches",
        "",
    ]
    if partial_experiments:
        lines.extend(
            [
                "| Experiment | Observed | Expected | Models | Forms | Runs |",
                "|---|---:|---:|---|---:|---|",
            ]
        )
        for row in partial_experiments:
            lines.append(
                f"| `{row['experiment_id']}` | {row['observed_trials']} | {row['expected_trials']} | "
                f"`{row['models']}` | {row['forms']} | `{row['runs']}` |"
            )
    else:
        lines.append("No known fixed-size submitted batch is currently partial in the discovered summaries.")
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"- Canonical trial rows: `{len(canonical_rows)}` in [canonical_trials.csv](canonical_trials.csv).",
            f"- Thesis model summary: [thesis_model_summary.csv](thesis_model_summary.csv).",
            f"- Answer-set validation: [answer_validation.csv](answer_validation.csv); statuses: {', '.join(f'{k}: {v}' for k, v in answer_status.most_common())}.",
            f"- Metadata/path validation: {', '.join(f'{k}: {v}' for k, v in metadata_status.most_common())}.",
            f"- Experiment coverage: [experiment_coverage.csv](experiment_coverage.csv).",
            f"- Target coverage: [target_coverage.csv](target_coverage.csv). Goal: `{TARGET_RUN_COUNT}` answer runs per form / `{TARGET_TRIAL_COUNT}` unique form-runs per model.",
            f"- Reference coverage: [reference_coverage.csv](reference_coverage.csv); complete references: `{reference_available}`, videos: `{reference_video_available}`.",
            f"- Efficiency summary: [efficiency_summary.csv](efficiency_summary.csv); rows: `{len(efficiency_summary)}`.",
            f"- Paired efficiency comparison: [paired_efficiency_comparison.csv](paired_efficiency_comparison.csv); rows: `{len(paired_efficiency)}`.",
            "- Model action matrix: [model_action_matrix.csv](model_action_matrix.csv).",
            "- Model action type summary: [model_action_type_summary.csv](model_action_type_summary.csv).",
            "- Model trial action counts: [model_action_trial_counts.csv](model_action_trial_counts.csv).",
            "- Model summary: [model_summary.csv](model_summary.csv).",
            "- Form summary: [form_summary.csv](form_summary.csv).",
            "- Failure summary: [failure_summary.csv](failure_summary.csv).",
            "- Question-type summary: [question_type_summary.csv](question_type_summary.csv).",
            "- Legacy cohort summary: [cohort_summary.csv](cohort_summary.csv).",
            "- Legacy per-form summary: [per_form_summary.csv](per_form_summary.csv).",
        ]
    )
    if include_diagnostic_plots:
        lines.extend(
            [
                "",
                "## Diagnostic Plots",
                "",
                "- [OpenCUA form accuracy](plots/diagnostic_opencua_forms.svg)",
                "- [Qwen latest-batch form accuracy](plots/diagnostic_qwen_latest_forms.svg)",
            ]
        )
    (output_dir / "latest_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze model evaluation summaries and write CSV/Markdown/SVG outputs.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--include-diagnostic-plots",
        action="store_true",
        help="Also generate granular per-form diagnostic SVGs. Default output is thesis model-level plots only.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    trials = load_trials(dataset_root)
    cohorts = _cohorts(trials)
    cohort_rows = [aggregate(label, items) for label, items in cohorts.items() if items]
    canonical_rows = canonical_trial_rows(trials)
    answer_validation = answer_validation_rows(trials)
    experiment_coverage = experiment_coverage_rows(trials)
    target_coverage = target_coverage_rows(trials)
    qwen_model_rows = aggregate_by_model("qwen_all_completed", cohorts["qwen_all_completed"])
    opencua_model_rows = aggregate_by_model("opencua_control_guidance_completed", cohorts["opencua_control_guidance_completed"])
    opencua_direct_mcp_model_rows = aggregate_by_model("opencua_direct_mcp_completed", cohorts["opencua_direct_mcp_completed"])
    model_rows = qwen_model_rows + opencua_model_rows + opencua_direct_mcp_model_rows
    latest_qwen_model_rows = aggregate_by_model("qwen_latest_completed", cohorts["qwen_latest_completed"])
    opencua_form_rows = aggregate_by_form_run("opencua_control_guidance_completed", cohorts["opencua_control_guidance_completed"])
    opencua_direct_mcp_form_rows = aggregate_by_form_run("opencua_direct_mcp_completed", cohorts["opencua_direct_mcp_completed"])
    qwen_latest_form_rows = aggregate_by_form_run("qwen_latest_completed", cohorts["qwen_latest_completed"])
    question_type_summary = (
        question_type_rows(cohorts["opencua_control_guidance_completed"], "opencua_control_guidance_completed")
        + question_type_rows(cohorts["opencua_direct_mcp_completed"], "opencua_direct_mcp_completed")
        + question_type_rows(cohorts["qwen_latest_completed"], "qwen_latest_partial_only", reliable_only=True)
    )
    failure_rows = failure_summary_rows(cohorts)
    reference_coverage = reference_coverage_rows()
    trials_by_scope: Dict[str, Sequence[Trial]] = {}
    trials_by_scope.update(trials_by_model_scope("qwen_all_completed", cohorts["qwen_all_completed"]))
    trials_by_scope.update(trials_by_model_scope("opencua_control_guidance_completed", cohorts["opencua_control_guidance_completed"]))
    trials_by_scope.update(trials_by_model_scope("opencua_direct_mcp_completed", cohorts["opencua_direct_mcp_completed"]))
    trials_by_scope.update(trials_by_model_scope("qwen_latest_completed", cohorts["qwen_latest_completed"]))
    primary_trials_by_scope: Dict[str, Sequence[Trial]] = {}
    primary_trials_by_scope.update(trials_by_model_scope("qwen_all_completed", cohorts["qwen_all_completed"]))
    primary_trials_by_scope.update(trials_by_model_scope("opencua_control_guidance_completed", cohorts["opencua_control_guidance_completed"]))
    primary_trials_by_scope.update(trials_by_model_scope("opencua_direct_mcp_completed", cohorts["opencua_direct_mcp_completed"]))
    action_type_rows, action_matrix_rows, action_trial_rows = model_action_metadata_rows(primary_trials_by_scope)
    action_matrix_rows = sorted(
        action_matrix_rows,
        key=lambda row: THESIS_MODEL_ORDER.index(str(row.get("model_id") or ""))
        if str(row.get("model_id") or "") in THESIS_MODEL_ORDER
        else len(THESIS_MODEL_ORDER),
    )
    efficiency_summary = efficiency_summary_rows(model_rows + latest_qwen_model_rows, trials_by_scope)
    thesis_rows = thesis_model_summary_rows(model_rows, efficiency_summary, target_coverage)
    thesis_plot_model_rows = [
        {**row, "display_label": _model_label(str(row.get("model_id") or ""))}
        for row in model_rows
        if str(row.get("model_id") or "") in THESIS_MODEL_LABELS
    ]
    paired_efficiency = paired_efficiency_rows(
        cohorts["qwen_all_completed"]
        + cohorts["opencua_control_guidance_completed"]
        + cohorts["opencua_direct_mcp_completed"]
    )

    _write_dict_csv(canonical_rows, output_dir / "canonical_trials.csv")
    _write_dict_csv(answer_validation, output_dir / "answer_validation.csv")
    _write_dict_csv(experiment_coverage, output_dir / "experiment_coverage.csv")
    _write_dict_csv(target_coverage, output_dir / "target_coverage.csv")
    _write_dict_csv(reference_coverage, output_dir / "reference_coverage.csv")
    _write_csv(cohort_rows + qwen_model_rows + latest_qwen_model_rows, output_dir / "cohort_summary.csv")
    _write_csv(model_rows + latest_qwen_model_rows, output_dir / "model_summary.csv")
    _write_dict_csv(thesis_rows, output_dir / "thesis_model_summary.csv")
    _write_dict_csv(efficiency_summary, output_dir / "efficiency_summary.csv")
    _write_dict_csv(paired_efficiency, output_dir / "paired_efficiency_comparison.csv")
    _write_dict_csv(action_matrix_rows, output_dir / "model_action_matrix.csv")
    _write_dict_csv(action_type_rows, output_dir / "model_action_type_summary.csv")
    _write_dict_csv(action_trial_rows, output_dir / "model_action_trial_counts.csv")
    _write_csv(opencua_form_rows + opencua_direct_mcp_form_rows + qwen_latest_form_rows, output_dir / "form_summary.csv")
    _write_csv(opencua_form_rows + opencua_direct_mcp_form_rows + qwen_latest_form_rows, output_dir / "per_form_summary.csv")
    _write_question_type_csv(question_type_summary, output_dir / "question_type_summary.csv")
    _write_dict_csv(failure_rows, output_dir / "failure_summary.csv")
    write_thesis_effectiveness_overview(thesis_rows, plots_dir / "thesis_effectiveness_overview.svg")
    write_thesis_efficiency_overview(thesis_rows, plots_dir / "thesis_efficiency_overview.svg")
    write_thesis_failure_mix(thesis_plot_model_rows, plots_dir / "thesis_failure_mix.svg")
    write_thesis_efficiency_frontier(thesis_rows, plots_dir / "thesis_efficiency_frontier.svg")
    write_thesis_reference_coverage(thesis_rows, reference_coverage, plots_dir / "thesis_reference_coverage.svg")
    if args.include_diagnostic_plots:
        write_form_accuracy(
            opencua_form_rows + opencua_direct_mcp_form_rows,
            plots_dir / "diagnostic_opencua_forms.svg",
            "OpenCUA Form Accuracy Diagnostics",
        )
        write_form_accuracy(
            qwen_latest_form_rows,
            plots_dir / "diagnostic_qwen_latest_forms.svg",
            "Qwen Latest-Batch Form Accuracy Diagnostics",
        )
    write_markdown(
        output_dir,
        thesis_rows,
        action_matrix_rows,
        canonical_rows,
        answer_validation,
        experiment_coverage,
        reference_coverage,
        efficiency_summary,
        paired_efficiency,
        args.include_diagnostic_plots,
    )
    print(f"[INFO] wrote analysis to {output_dir}")
    print(f"[INFO] cohorts={len(cohort_rows)} total_trials={len(trials)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
