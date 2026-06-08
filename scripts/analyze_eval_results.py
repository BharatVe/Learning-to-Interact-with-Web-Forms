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
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_DATASET_ROOT = Path("data/model_baselines")
DEFAULT_OUTPUT_DIR = Path("docs/eval_results/analysis")
DEFAULT_ANSWERS_ROOT = Path("data/answers")
DEFAULT_FORMS_ROOT = Path("src/forms")
TARGET_RUN_COUNT = 6
TARGET_TRIAL_COUNT = 300
TARGET_RUN_IDS = tuple(f"run_{idx:04d}" for idx in range(1, TARGET_RUN_COUNT + 1))
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


def _score(summary: Dict[str, Any]) -> Tuple[int, str]:
    if summary.get("pre_successful_submit_verified_correctness") is not None:
        return _as_int(summary.get("pre_successful_submit_verified_correctness")), "pre_successful_submit"
    if summary.get("pre_first_submit_verified_correctness") is not None:
        return _as_int(summary.get("pre_first_submit_verified_correctness")), "pre_first_submit"
    return _as_int(summary.get("verified_correctness")), "final_verification"


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
        "submit_successes": sum(1 for t in trials if t.submit_success),
        "success_rate": _safe_rate(sum(1 for t in trials if t.success), len(trials)),
        "submit_rate": _safe_rate(sum(1 for t in trials if t.submit_success), len(trials)),
        "scored_correctness": scored,
        "scored_accuracy": _safe_rate(scored, total_questions),
        "pre_submit_answered": pre_success_total,
        "pre_submit_questions": pre_success_questions,
        "pre_submit_accuracy": _safe_rate(pre_success_total, pre_success_questions),
        "submit_attempt_count": sum(t.submit_attempt_count for t in trials),
        "submitted_while_incomplete_count": sum(t.submitted_while_incomplete_count for t in trials),
        "avg_action_count": _safe_rate(sum(t.action_count for t in trials), len(trials)),
        "avg_duration_s": _safe_rate(sum(t.duration_s for t in trials), len(trials)),
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


def _answers_instance_path(summary_path: Path) -> Path:
    return summary_path.with_name("answers_instance.json")


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
                "scored_source": trial.scored_source,
                "verified_correctness": trial.verified_correctness,
                "pre_successful_submit_correctness": "" if trial.pre_successful_submit_correctness is None else trial.pre_successful_submit_correctness,
                "pre_first_submit_correctness": "" if trial.pre_first_submit_correctness is None else trial.pre_first_submit_correctness,
                "submit_attempt_count": trial.submit_attempt_count,
                "submitted_while_incomplete_count": trial.submitted_while_incomplete_count,
                "action_count": trial.action_count,
                "duration_s": trial.duration_s,
                "run_completed_utc": trial.run_completed_utc,
                "metadata_path_status": _metadata_path_status(trial),
                **answer_validation,
                "summary_path": _rel(trial.summary_path),
                "annotations_path": _rel(_annotation_path(trial.summary_path)),
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
        "submit_successes",
        "success_rate",
        "submit_rate",
        "scored_correctness",
        "question_total",
        "scored_accuracy",
        "pre_submit_answered",
        "pre_submit_questions",
        "pre_submit_accuracy",
        "submit_attempt_count",
        "submitted_while_incomplete_count",
        "avg_action_count",
        "avg_duration_s",
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
        '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#202124}.title{font-size:17px;font-weight:700}.axis{fill:#5f6368}.small{font-size:10px;fill:#5f6368}</style>',
        '<rect width="100%" height="100%" fill="#fff"/>',
    ]


def write_metric_bars(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    rows = [row for row in rows if row["trials"]]
    width = 980
    height = 110 + len(rows) * 88
    left = 255
    chart_w = 620
    colors = {"submit_rate": "#1a73e8", "success_rate": "#34a853", "scored_accuracy": "#fbbc04"}
    labels = {"submit_rate": "submit", "success_rate": "success", "scored_accuracy": "score"}
    lines = _svg_header(width, height)
    lines.append('<text class="title" x="24" y="30">Model Performance</text>')
    lines.append('<text class="axis" x="24" y="52">Submit success, strict success, and scored correctness accuracy</text>')
    for idx, row in enumerate(rows):
        y = 82 + idx * 88
        label = html.escape(_row_label(row))
        lines.append(f'<text x="24" y="{y + 16}">{label}</text>')
        for offset, key in enumerate(["submit_rate", "success_rate", "scored_accuracy"]):
            bar_y = y + offset * 20
            value = float(row[key])
            bar_w = max(1, value * chart_w)
            lines.append(f'<rect x="{left}" y="{bar_y}" width="{chart_w}" height="13" fill="#eef1f4"/>')
            lines.append(f'<rect x="{left}" y="{bar_y}" width="{bar_w:.1f}" height="13" fill="{colors[key]}"/>')
            lines.append(f'<text class="small" x="{left - 54}" y="{bar_y + 11}">{labels[key]}</text>')
            lines.append(f'<text class="small" x="{left + chart_w + 8}" y="{bar_y + 11}">{_pct(value)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stop_reason_bars(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    rows = [row for row in rows if row["trials"]]
    reasons = sorted({reason for row in rows for reason in row["stop_reasons"]})
    palette = ["#1a73e8", "#ea4335", "#fbbc04", "#34a853", "#9334e6", "#00acc1", "#ff7043"]
    color = {reason: palette[idx % len(palette)] for idx, reason in enumerate(reasons)}
    width = 980
    height = 115 + len(rows) * 58 + max(1, len(reasons)) * 18
    left = 255
    chart_w = 620
    lines = _svg_header(width, height)
    lines.append('<text class="title" x="24" y="30">Stop Reason Mix</text>')
    lines.append('<text class="axis" x="24" y="52">Share of trials ending in each stop reason</text>')
    for idx, row in enumerate(rows):
        y = 82 + idx * 58
        lines.append(f'<text x="24" y="{y + 15}">{html.escape(_row_label(row))}</text>')
        x = left
        total = max(1, int(row["trials"]))
        for reason in reasons:
            count = int(row["stop_reasons"].get(reason, 0))
            if not count:
                continue
            w = chart_w * count / total
            lines.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="18" fill="{color[reason]}"/>')
            if w > 28:
                lines.append(f'<text class="small" x="{x + 4:.1f}" y="{y + 13}" fill="#fff">{count}</text>')
            x += w
        lines.append(f'<rect x="{left}" y="{y}" width="{chart_w}" height="18" fill="none" stroke="#dadce0"/>')
    legend_y = 90 + len(rows) * 58
    for idx, reason in enumerate(reasons):
        y = legend_y + idx * 18
        lines.append(f'<rect x="24" y="{y - 10}" width="10" height="10" fill="{color[reason]}"/>')
        lines.append(f'<text class="small" x="42" y="{y}">{html.escape(reason)}</text>')
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


def _row_label(row: Dict[str, Any]) -> str:
    model = str(row.get("model_id") or "")
    scope = str(row.get("analysis_scope") or "")
    if ":" in scope:
        scope = scope.split(":", 1)[0]
    form = str(row.get("form_id") or "")
    run = str(row.get("answer_run_id") or "")
    parts = [part for part in [model, form, run, scope] if part]
    return " / ".join(parts)


def _md_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "| Model | Kind | Track | Scope | Forms | Trials | Questions | Submit | Success | Scored Answers | Pre-submit Answers | Incomplete Submits | Avg Actions/Form | Main Stop Reasons |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        reasons = ", ".join(f"{k}: {v}" for k, v in row["stop_reasons"].most_common())
        pre_submit = "n/a"
        if row.get("pre_submit_questions"):
            pre_submit = f"{row['pre_submit_answered']}/{row['pre_submit_questions']} ({_pct(row['pre_submit_accuracy'])})"
        lines.append(
            f"| `{row.get('model_id', '')}` | `{row.get('model_kind', '')}` | `{row.get('track', '')}` | `{row.get('analysis_scope', '')}` | "
            f"{row['forms']} | {row['trials']} | {row['question_total']} | "
            f"{row['submit_successes']} ({_pct(row['submit_rate'])}) | {row['successes']} ({_pct(row['success_rate'])}) | "
            f"{row['scored_correctness']}/{row['question_total']} ({_pct(row['scored_accuracy'])}) | {pre_submit} | "
            f"{row.get('submitted_while_incomplete_count', 0)} | {_fmt_float(row['avg_action_count'])} | {reasons} |"
        )
    return lines


def _question_type_md(rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "| Model | Kind | Track | Question Type | Scope | Questions | Attempted | Verified Correct | Failed/Unanswered | Notes |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        notes = []
        if row.get("not_attempted"):
            notes.append(f"not attempted: {row['not_attempted']}")
        if row.get("container_not_visible"):
            notes.append(f"container not visible: {row['container_not_visible']}")
        reliability = str(row.get("reliability") or "")
        if reliability != "final_verification":
            notes.append(reliability)
        lines.append(
            f"| `{row['model_id']}` | `{row.get('model_kind', '')}` | `{row.get('track', '')}` | `{row['widget_type']}` | `{row.get('analysis_scope', '')}` | {row['questions']} | "
            f"{row['attempted']} ({_pct(row['attempt_rate'])}) | {row['verified_correct']} ({_pct(row['verified_accuracy'])}) | "
            f"{row['failed']} ({_pct(row['failure_rate'])}) | {', '.join(notes)} |"
        )
    return lines


def write_markdown(
    output_dir: Path,
    cohort_rows: Sequence[Dict[str, Any]],
    model_rows: Sequence[Dict[str, Any]],
    latest_qwen_model_rows: Sequence[Dict[str, Any]],
    opencua_form_rows: Sequence[Dict[str, Any]],
    qwen_latest_form_rows: Sequence[Dict[str, Any]],
    question_type_summary: Sequence[Dict[str, Any]],
    canonical_rows: Sequence[Dict[str, Any]],
    answer_validation: Sequence[Dict[str, Any]],
    experiment_coverage: Sequence[Dict[str, Any]],
    target_coverage: Sequence[Dict[str, Any]],
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    answer_status = Counter(str(row.get("answer_set_status") or "") for row in answer_validation)
    metadata_status = Counter(str(row.get("metadata_path_status") or "") for row in answer_validation)
    partial_experiments = [row for row in experiment_coverage if row.get("status") == "in_progress_or_partial"]
    lines = [
        "# Evaluation Analysis",
        "",
        f"Last updated: {now}",
        "",
        "Generated by `scripts/analyze_eval_results.py` from `data/model_baselines/**/summary.json`.",
        "",
        "## Headline",
        "",
        "- Raw artifacts remain under `data/model_baselines/**`; this report is the thesis-facing index organized by model, form, run, and result.",
        "- `canonical_trials.csv` indexes every discovered summary, including rows from jobs that are still running. Main model summaries exclude known incomplete fixed-size batches until their expected trial count is reached.",
        "- Qwen direct-MCP is now the only track with regular form submissions. VLM has the higher cumulative submit rate, while text Qwen remains limited by premature `DONE`.",
        "- OpenCUA control-guidance completed 50 forms but did not submit any form. The model still fails mainly on native Google Forms controls.",
        "- For Qwen submitted trials, use scored correctness, not final `verified_correctness`, because final verification can happen on the Google Forms confirmation page.",
        "- Qwen job `2228977` added 8 new summaries; `workshop_signup` was skipped because both models already had completed summaries.",
        "",
        "## Current Data Index",
        "",
        f"- Canonical trial rows: `{len(canonical_rows)}` in [canonical_trials.csv](canonical_trials.csv).",
        f"- Answer-set validation rows: `{len(answer_validation)}` in [answer_validation.csv](answer_validation.csv); statuses: {', '.join(f'{k}: {v}' for k, v in answer_status.most_common())}.",
        f"- Summary metadata/path validation: {', '.join(f'{k}: {v}' for k, v in metadata_status.most_common())}.",
        f"- Experiment coverage rows: `{len(experiment_coverage)}` in [experiment_coverage.csv](experiment_coverage.csv).",
        f"- Target coverage rows: `{len(target_coverage)}` in [target_coverage.csv](target_coverage.csv). Goal: `{TARGET_RUN_COUNT}` answer runs per form / `{TARGET_TRIAL_COUNT}` unique form-runs for each model.",
        "- Primary summary tables: [model_summary.csv](model_summary.csv), [form_summary.csv](form_summary.csv), [question_type_summary.csv](question_type_summary.csv), [failure_summary.csv](failure_summary.csv).",
        "- Legacy tracker files remain in `docs/eval_results/metrics.csv` and `docs/eval_results/metrics.jsonl`.",
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
            "## Target Coverage",
            "",
            "| Model | Target Form-Runs | Observed Unique | Missing | Duplicate Target Pairs | Complete Forms | Forms With Any Run | Missing Examples |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in target_coverage:
        lines.append(
            f"| `{row['model_id']}` | {row['target_form_runs']} | {row['observed_unique_form_runs']} | "
            f"{row['missing_form_runs']} | {row['duplicate_target_form_runs']} | {row['forms_complete_target_runs']} | "
            f"{row['forms_with_any_run']} | `{row['missing_examples']}` |"
        )
    lines.extend(
        [
            "",
            "## Model Summary",
            "",
            *_md_table(model_rows),
            "",
            "## Latest Qwen Batch By Model",
            "",
            *_md_table(latest_qwen_model_rows),
            "",
            "## Latest Qwen Batch By Form And Run",
            "",
            "| Model | Kind | Track | Form | Run | Submit | Success | Scored Answers | Submit Attempts | Incomplete Submits | Actions | Stop |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in qwen_latest_form_rows:
        reasons = ", ".join(f"{k}: {v}" for k, v in row["stop_reasons"].most_common())
        lines.append(
            f"| `{row['model_id']}` | `{row.get('model_kind', '')}` | `{row.get('track', '')}` | `{row['form_id']}` | `{row.get('answer_run_id', '')}` | {row['submit_successes']}/{row['trials']} | "
            f"{row['successes']}/{row['trials']} | {row['scored_correctness']}/{row['question_total']} ({_pct(row['scored_accuracy'])}) | "
            f"{row.get('submit_attempt_count', 0)} | {row.get('submitted_while_incomplete_count', 0)} | "
            f"{_fmt_float(row['avg_action_count'])} | {reasons} |"
        )
    lines.extend(
        [
            "",
            "## OpenCUA Control-Guidance By Form And Run, All 50 Forms",
            "",
            "| Model | Kind | Track | Form | Run | Submit | Success | Questions | Answered Correctly | Actions | Stop |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in opencua_form_rows:
        reasons = ", ".join(f"{k}: {v}" for k, v in row["stop_reasons"].most_common())
        lines.append(
            f"| `{row['model_id']}` | `{row.get('model_kind', '')}` | `{row.get('track', '')}` | `{row['form_id']}` | `{row.get('answer_run_id', '')}` | {row['submit_successes']}/{row['trials']} | {row['successes']}/{row['trials']} | "
            f"{row['question_total']} | {row['scored_correctness']}/{row['question_total']} ({_pct(row['scored_accuracy'])}) | "
            f"{_fmt_float(row['avg_action_count'])} | {reasons} |"
        )
    lines.extend(
        [
            "",
            "## Question-Type Success And Failure",
            "",
            "For submitted Qwen trials, final per-question verification can occur on the confirmation page. The Qwen rows below therefore use only non-submitted partial-state trials for reliable type-level diagnosis. OpenCUA did not submit, so its rows are direct final-state verification.",
            "",
            *_question_type_md(question_type_summary),
            "",
            "## Plots",
            "",
            "- [Model performance](plots/model_overview.svg)",
            "- [Stop reason mix](plots/stop_reasons.svg)",
            "- [OpenCUA control-guidance form accuracy](plots/opencua_control_forms.svg)",
            "- [Qwen latest-batch form accuracy](plots/qwen_latest_forms.svg)",
            "",
            "## Generated Tables",
            "",
            "- [Canonical trial CSV](canonical_trials.csv)",
            "- [Answer validation CSV](answer_validation.csv)",
            "- [Experiment coverage CSV](experiment_coverage.csv)",
            "- [Target coverage CSV](target_coverage.csv)",
            "- [Model summary CSV](model_summary.csv)",
            "- [Form summary CSV](form_summary.csv)",
            "- [Failure summary CSV](failure_summary.csv)",
            "- [Question-type summary CSV](question_type_summary.csv)",
            "- [Legacy cohort summary CSV](cohort_summary.csv)",
            "- [Legacy per-form summary CSV](per_form_summary.csv)",
        ]
    )
    (output_dir / "latest_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze model evaluation summaries and write CSV/Markdown/SVG outputs.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
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

    _write_dict_csv(canonical_rows, output_dir / "canonical_trials.csv")
    _write_dict_csv(answer_validation, output_dir / "answer_validation.csv")
    _write_dict_csv(experiment_coverage, output_dir / "experiment_coverage.csv")
    _write_dict_csv(target_coverage, output_dir / "target_coverage.csv")
    _write_csv(cohort_rows + qwen_model_rows + latest_qwen_model_rows, output_dir / "cohort_summary.csv")
    _write_csv(model_rows + latest_qwen_model_rows, output_dir / "model_summary.csv")
    _write_csv(opencua_form_rows + opencua_direct_mcp_form_rows + qwen_latest_form_rows, output_dir / "form_summary.csv")
    _write_csv(opencua_form_rows + opencua_direct_mcp_form_rows + qwen_latest_form_rows, output_dir / "per_form_summary.csv")
    _write_question_type_csv(question_type_summary, output_dir / "question_type_summary.csv")
    _write_dict_csv(failure_rows, output_dir / "failure_summary.csv")
    write_metric_bars(model_rows + latest_qwen_model_rows, plots_dir / "model_overview.svg")
    write_stop_reason_bars(model_rows + latest_qwen_model_rows, plots_dir / "stop_reasons.svg")
    write_form_accuracy(opencua_form_rows, plots_dir / "opencua_control_forms.svg", "OpenCUA Control-Guidance Form Accuracy")
    write_form_accuracy(qwen_latest_form_rows, plots_dir / "qwen_latest_forms.svg", "Qwen Latest-Batch Form Accuracy")
    write_markdown(
        output_dir,
        cohort_rows,
        model_rows,
        latest_qwen_model_rows,
        opencua_form_rows + opencua_direct_mcp_form_rows,
        qwen_latest_form_rows,
        question_type_summary,
        canonical_rows,
        answer_validation,
        experiment_coverage,
        target_coverage,
    )
    print(f"[INFO] wrote analysis to {output_dir}")
    print(f"[INFO] cohorts={len(cohort_rows)} total_trials={len(trials)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
