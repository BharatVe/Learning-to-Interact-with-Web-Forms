#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Set


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _load_summary(path_value: str) -> Dict[str, Any]:
    path = Path(path_value)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_rate(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _mean(values: Iterable[float]) -> float:
    xs = list(values)
    if not xs:
        return 0.0
    return float(sum(xs)) / float(len(xs))


def _to_trial_row(summary: Dict[str, Any], manifest_row: Dict[str, Any]) -> Dict[str, Any]:
    question_total = int(summary.get("question_total") or 0)
    verified_correctness = int(summary.get("verified_correctness") or 0)
    attempted_correctness = int(summary.get("attempted_correctness") or 0)
    model_id = str(summary.get("model_id") or manifest_row.get("model_id") or "")
    model_kind = str(summary.get("model_kind") or manifest_row.get("model_kind") or "")
    duration_s = summary.get("duration_s")
    try:
        duration_value = float(duration_s) if duration_s is not None else None
    except Exception:
        duration_value = None
    composite_raw = summary.get("composite_score")
    try:
        composite_value = float(composite_raw) if composite_raw is not None else None
    except Exception:
        composite_value = None

    return {
        "model_id": model_id,
        "model_kind": model_kind,
        "track": str(summary.get("track") or manifest_row.get("track") or ""),
        "success": bool(summary.get("success")),
        "submit_success": bool(summary.get("submit_success")),
        "question_total": question_total,
        "verified_correctness": verified_correctness,
        "attempted_correctness": attempted_correctness,
        "verified_rate": _safe_rate(verified_correctness, question_total),
        "attempted_rate": _safe_rate(attempted_correctness, question_total),
        "duration_s": duration_value,
        "composite_score": composite_value,
        "trace_action_count": summary.get("trace_action_count"),
        "reference_available": bool(summary.get("reference_available")),
        "reference_action_count": summary.get("reference_action_count"),
        "reference_duration_s": summary.get("reference_duration_s"),
        "action_overhead_ratio": summary.get("action_overhead_ratio"),
        "time_overhead_ratio": summary.get("time_overhead_ratio"),
        "failure_category": str(summary.get("failure_category") or ""),
        "form_id": str(summary.get("form_id") or manifest_row.get("form_id") or ""),
        "answer_run_id": str(summary.get("answer_run_id") or manifest_row.get("answer_run_id") or ""),
        "safety_require_confirmation_count": int(summary.get("safety_require_confirmation_count") or 0),
        "safety_auto_allowed_count": int(summary.get("safety_auto_allowed_count") or 0),
    }


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    runs = len(rows)
    success_count = sum(1 for row in rows if bool(row.get("success")))
    submit_success_count = sum(1 for row in rows if bool(row.get("submit_success")))
    failures = [row for row in rows if not bool(row.get("success"))]

    question_total = sum(int(row.get("question_total") or 0) for row in rows)
    verified_correctness = sum(int(row.get("verified_correctness") or 0) for row in rows)
    attempted_correctness = sum(int(row.get("attempted_correctness") or 0) for row in rows)
    safety_require_confirmation_count = sum(int(row.get("safety_require_confirmation_count") or 0) for row in rows)
    safety_auto_allowed_count = sum(int(row.get("safety_auto_allowed_count") or 0) for row in rows)

    durations = [float(row["duration_s"]) for row in rows if row.get("duration_s") is not None]
    composite_scores = [float(row["composite_score"]) for row in rows if row.get("composite_score") is not None]
    trace_action_counts = [float(row["trace_action_count"]) for row in rows if row.get("trace_action_count") is not None]
    reference_action_counts = [float(row["reference_action_count"]) for row in rows if row.get("reference_action_count") is not None]
    reference_durations = [float(row["reference_duration_s"]) for row in rows if row.get("reference_duration_s") is not None]
    action_overheads = [float(row["action_overhead_ratio"]) for row in rows if row.get("action_overhead_ratio") is not None]
    time_overheads = [float(row["time_overhead_ratio"]) for row in rows if row.get("time_overhead_ratio") is not None]
    reference_available_count = sum(1 for row in rows if bool(row.get("reference_available")))

    failure_counter = Counter(
        str(row.get("failure_category") or "unknown")
        for row in failures
        if str(row.get("failure_category") or "").strip()
    )
    if failures and not failure_counter:
        failure_counter["unknown"] = len(failures)

    result: Dict[str, Any] = {
        "runs": runs,
        "success_count": success_count,
        "submit_success_count": submit_success_count,
        "failure_count": len(failures),
        "success_rate": round(_safe_rate(success_count, runs), 6),
        "submit_success_rate": round(_safe_rate(submit_success_count, runs), 6),
        "failure_rate": round(_safe_rate(len(failures), runs), 6),
        "verified_correctness": verified_correctness,
        "attempted_correctness": attempted_correctness,
        "question_total": question_total,
        "verified_correctness_rate": round(_safe_rate(verified_correctness, question_total), 6),
        "attempted_correctness_rate": round(_safe_rate(attempted_correctness, question_total), 6),
        "safety_require_confirmation_count": safety_require_confirmation_count,
        "safety_auto_allowed_count": safety_auto_allowed_count,
        "reference_available_count": reference_available_count,
        "mean_trace_action_count": round(_mean(trace_action_counts), 6) if trace_action_counts else None,
        "median_trace_action_count": round(float(median(trace_action_counts)), 6) if trace_action_counts else None,
        "mean_reference_action_count": round(_mean(reference_action_counts), 6) if reference_action_counts else None,
        "median_reference_action_count": round(float(median(reference_action_counts)), 6) if reference_action_counts else None,
        "mean_reference_duration_s": round(_mean(reference_durations), 6) if reference_durations else None,
        "median_reference_duration_s": round(float(median(reference_durations)), 6) if reference_durations else None,
        "mean_action_overhead_ratio": round(_mean(action_overheads), 6) if action_overheads else None,
        "median_action_overhead_ratio": round(float(median(action_overheads)), 6) if action_overheads else None,
        "mean_time_overhead_ratio": round(_mean(time_overheads), 6) if time_overheads else None,
        "median_time_overhead_ratio": round(float(median(time_overheads)), 6) if time_overheads else None,
        "mean_duration_s": round(_mean(durations), 6) if durations else None,
        "median_duration_s": round(float(median(durations)), 6) if durations else None,
        "mean_composite_score": round(_mean(composite_scores), 6) if composite_scores else None,
        "failure_categories": dict(sorted(failure_counter.items(), key=lambda item: item[0])),
    }
    return result


def _load_trials(dataset_root: Path, experiment_id: str) -> List[Dict[str, Any]]:
    manifest_path = dataset_root / experiment_id / "manifest.jsonl"
    manifest_rows = _load_jsonl(manifest_path)
    out: List[Dict[str, Any]] = []
    for row in manifest_rows:
        summary_path = str(row.get("summary_path") or "").strip()
        if not summary_path:
            continue
        summary = _load_summary(summary_path)
        if not summary:
            continue
        out.append(_to_trial_row(summary, row))
    return out


def _expected_trials(
    expected_forms: int,
    expected_runs_per_form: int,
    model_count: int,
    fallback_trials: List[Dict[str, Any]],
) -> Optional[int]:
    if expected_forms > 0 and expected_runs_per_form > 0 and model_count > 0:
        return int(expected_forms) * int(expected_runs_per_form) * int(model_count)
    forms = {str(row.get("form_id") or "") for row in fallback_trials if str(row.get("form_id") or "")}
    runs = {str(row.get("answer_run_id") or "") for row in fallback_trials if str(row.get("answer_run_id") or "")}
    if not forms or not runs or model_count <= 0:
        return None
    return len(forms) * len(runs) * model_count


def _load_model_ids_from_config(config_path: Optional[str], repo_root: Path) -> Set[str]:
    if not config_path:
        return set()
    path = Path(config_path)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    models = payload.get("models")
    if not isinstance(models, list):
        return set()
    out: Set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("id") or "").strip()
        if model_id:
            out.add(model_id)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize canonical benchmark families (direct-MCP Qwen vs native computer-use).")
    parser.add_argument("--dataset-root", default="data/model_baselines")
    parser.add_argument("--family-a-experiment-id", required=True)
    parser.add_argument("--family-b-experiment-id", required=True)
    parser.add_argument("--config-path", default="")
    parser.add_argument("--expected-model-count", type=int, default=0)
    parser.add_argument("--expected-forms", type=int, default=0)
    parser.add_argument("--expected-runs-per-form", type=int, default=0)
    parser.add_argument("--output", default="logs/track_baseline_summary.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = (repo_root / args.dataset_root).resolve()
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    family_a_trials = _load_trials(dataset_root, args.family_a_experiment_id)
    family_b_trials = _load_trials(dataset_root, args.family_b_experiment_id)
    all_trials = family_a_trials + family_b_trials

    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_kind: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in all_trials:
        by_model[str(row.get("model_id") or "")].append(row)
        by_kind[str(row.get("model_kind") or "")].append(row)

    per_model = {}
    for model_id, rows in sorted(by_model.items(), key=lambda item: item[0]):
        model_kind = str(rows[0].get("model_kind") or "") if rows else ""
        track = str(rows[0].get("track") or "") if rows else ""
        per_model[model_id] = {
            "model_kind": model_kind,
            "track": track,
            **_aggregate(rows),
        }

    per_track = {}
    for model_kind, rows in sorted(by_kind.items(), key=lambda item: item[0]):
        per_track[model_kind] = _aggregate(rows)

    configured_model_ids = _load_model_ids_from_config(args.config_path, repo_root=repo_root)
    observed_model_count = len([mid for mid in by_model.keys() if mid])
    if int(args.expected_model_count) > 0:
        expected_model_count = int(args.expected_model_count)
        model_count_source = "cli"
    elif configured_model_ids:
        expected_model_count = len(configured_model_ids)
        model_count_source = "config_path"
    else:
        expected_model_count = observed_model_count
        model_count_source = "observed"

    expected_total = _expected_trials(
        expected_forms=int(args.expected_forms),
        expected_runs_per_form=int(args.expected_runs_per_form),
        model_count=expected_model_count,
        fallback_trials=all_trials,
    )
    included_total = len(all_trials)
    missing_total = max(0, int(expected_total) - included_total) if expected_total is not None else None

    report = {
        "family_a_experiment_id": args.family_a_experiment_id,
        "family_b_experiment_id": args.family_b_experiment_id,
        "paths": {
            "dataset_root": str(dataset_root),
            "family_a_manifest": str(dataset_root / args.family_a_experiment_id / "manifest.jsonl"),
            "family_b_manifest": str(dataset_root / args.family_b_experiment_id / "manifest.jsonl"),
        },
        "trial_accounting": {
            "expected_total": expected_total,
            "included_total": included_total,
            "missing_total": missing_total,
            "expected_model_count": expected_model_count,
            "observed_model_count": observed_model_count,
            "model_count_source": model_count_source,
            "forms_seen": sorted({str(row.get("form_id") or "") for row in all_trials if str(row.get("form_id") or "")}),
            "runs_seen": sorted({str(row.get("answer_run_id") or "") for row in all_trials if str(row.get("answer_run_id") or "")}),
        },
        "per_track": per_track,
        "per_model": per_model,
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[INFO] wrote track baseline summary: {output_path}")
    print(f"[INFO] included_trials={included_total}")
    if expected_total is not None:
        print(f"[INFO] expected_trials={expected_total} missing_trials={missing_total}")
    for kind, stats in per_track.items():
        print(
            f"[INFO] {kind}: runs={stats.get('runs')} "
            f"success_rate={stats.get('success_rate')} "
            f"verified_rate={stats.get('verified_correctness_rate')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
