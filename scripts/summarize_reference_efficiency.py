#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _mean(values: Iterable[float]) -> float:
    xs = list(values)
    return float(sum(xs)) / float(len(xs)) if xs else 0.0


def _trial_row(summary: Dict[str, Any], manifest_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "experiment_id": str(manifest_row.get("experiment_id") or summary.get("experiment_id") or ""),
        "model_id": str(summary.get("model_id") or manifest_row.get("model_id") or ""),
        "model_kind": str(summary.get("model_kind") or manifest_row.get("model_kind") or ""),
        "track": str(summary.get("track") or manifest_row.get("track") or ""),
        "form_id": str(summary.get("form_id") or manifest_row.get("form_id") or ""),
        "answer_run_id": str(summary.get("answer_run_id") or manifest_row.get("answer_run_id") or ""),
        "trial_id": str(summary.get("trial_id") or manifest_row.get("trial_id") or ""),
        "success": bool(summary.get("success")),
        "submit_success": bool(summary.get("submit_success")),
        "failure_category": str(summary.get("failure_category") or ""),
        "question_total": int(summary.get("question_total") or 0),
        "attempted_correctness": int(summary.get("attempted_correctness") or 0),
        "verified_correctness": int(summary.get("verified_correctness") or 0),
        "duration_s": summary.get("duration_s"),
        "action_count": summary.get("action_count"),
        "trace_action_count": summary.get("trace_action_count"),
        "reference_available": bool(summary.get("reference_available")),
        "reference_action_count": summary.get("reference_action_count"),
        "reference_duration_s": summary.get("reference_duration_s"),
        "action_overhead_ratio": summary.get("action_overhead_ratio"),
        "time_overhead_ratio": summary.get("time_overhead_ratio"),
        "action_count_delta": summary.get("action_count_delta"),
        "duration_delta_s": summary.get("duration_delta_s"),
        "summary_path": str(manifest_row.get("summary_path") or ""),
        "trace_path": str(manifest_row.get("trace_path") or ""),
        "reference_trace_path": str(summary.get("reference_trace_path") or ""),
        "reference_run_path": str(summary.get("reference_run_path") or ""),
    }


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    action_overheads = [float(row["action_overhead_ratio"]) for row in rows if row.get("action_overhead_ratio") is not None]
    time_overheads = [float(row["time_overhead_ratio"]) for row in rows if row.get("time_overhead_ratio") is not None]
    trace_action_counts = [float(row["trace_action_count"]) for row in rows if row.get("trace_action_count") is not None]
    raw_action_counts = [float(row["action_count"]) for row in rows if row.get("action_count") is not None]
    durations = [float(row["duration_s"]) for row in rows if row.get("duration_s") is not None]
    reference_actions = [float(row["reference_action_count"]) for row in rows if row.get("reference_action_count") is not None]
    reference_durations = [float(row["reference_duration_s"]) for row in rows if row.get("reference_duration_s") is not None]
    failure_categories: Dict[str, int] = {}
    for row in rows:
        category = str(row.get("failure_category") or "").strip()
        if category:
            failure_categories[category] = failure_categories.get(category, 0) + 1
    return {
        "runs": len(rows),
        "success_rate": round(sum(1 for row in rows if bool(row.get("success"))) / len(rows), 6) if rows else 0.0,
        "submit_success_rate": round(sum(1 for row in rows if bool(row.get("submit_success"))) / len(rows), 6) if rows else 0.0,
        "reference_available_count": sum(1 for row in rows if bool(row.get("reference_available"))),
        "mean_action_count": round(_mean(raw_action_counts), 6) if raw_action_counts else None,
        "median_action_count": round(float(median(raw_action_counts)), 6) if raw_action_counts else None,
        "mean_trace_action_count": round(_mean(trace_action_counts), 6) if trace_action_counts else None,
        "median_trace_action_count": round(float(median(trace_action_counts)), 6) if trace_action_counts else None,
        "mean_duration_s": round(_mean(durations), 6) if durations else None,
        "median_duration_s": round(float(median(durations)), 6) if durations else None,
        "mean_reference_action_count": round(_mean(reference_actions), 6) if reference_actions else None,
        "median_reference_action_count": round(float(median(reference_actions)), 6) if reference_actions else None,
        "mean_reference_duration_s": round(_mean(reference_durations), 6) if reference_durations else None,
        "median_reference_duration_s": round(float(median(reference_durations)), 6) if reference_durations else None,
        "mean_action_overhead_ratio": round(_mean(action_overheads), 6) if action_overheads else None,
        "median_action_overhead_ratio": round(float(median(action_overheads)), 6) if action_overheads else None,
        "mean_time_overhead_ratio": round(_mean(time_overheads), 6) if time_overheads else None,
        "median_time_overhead_ratio": round(float(median(time_overheads)), 6) if time_overheads else None,
        "failure_categories": dict(sorted(failure_categories.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize model-vs-reference efficiency against scripted Playwright runs.")
    parser.add_argument("--dataset-root", default="data/model_baselines")
    parser.add_argument("--experiment-id", action="append", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--form-id")
    parser.add_argument("--output", default="logs/reference_efficiency_summary.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = (repo_root / args.dataset_root).resolve()
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for experiment_id in args.experiment_id:
        manifest_path = dataset_root / experiment_id / "manifest.jsonl"
        for manifest_row in _load_jsonl(manifest_path):
            summary_path_raw = str(manifest_row.get("summary_path") or "").strip()
            if not summary_path_raw:
                continue
            summary = _load_json(Path(summary_path_raw))
            if not summary:
                continue
            row = _trial_row(summary, manifest_row)
            if args.model_id and row["model_id"] != args.model_id:
                continue
            if args.form_id and row["form_id"] != args.form_id:
                continue
            rows.append(row)

    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_kind: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_form: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["model_id"]].append(row)
        by_kind[row["model_kind"]].append(row)
        by_form[row["form_id"]].append(row)

    report = {
        "experiment_ids": list(args.experiment_id),
        "filters": {"model_id": args.model_id or None, "form_id": args.form_id or None},
        "trial_count": len(rows),
        "per_trial": rows,
        "per_model": {key: _aggregate(value) for key, value in sorted(by_model.items())},
        "per_model_kind": {key: _aggregate(value) for key, value in sorted(by_kind.items())},
        "per_form": {key: _aggregate(value) for key, value in sorted(by_form.items())},
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[INFO] wrote reference efficiency summary: {output_path}")
    print(f"[INFO] included_trials={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
