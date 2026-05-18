#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_DATASET_ROOT = Path("data/model_baselines")
DEFAULT_OUTPUT_DIR = Path("docs/eval_results")

FIELDS = [
    "experiment_id",
    "model_id",
    "model_kind",
    "track",
    "form_id",
    "answer_run_id",
    "trial_id",
    "success",
    "submit_success",
    "stop_reason",
    "failure_category",
    "question_total",
    "verified_correctness",
    "attempted_correctness",
    "pre_successful_submit_verified_correctness",
    "pre_first_submit_verified_correctness",
    "scored_correctness",
    "scored_correctness_source",
    "metric_warning",
    "submit_attempt_count",
    "action_count",
    "duration_s",
    "reference_action_count",
    "reference_duration_s",
    "action_overhead_ratio",
    "time_overhead_ratio",
    "run_completed_utc",
    "summary_path",
]


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def _row_from_summary(path: Path, dataset_root: Path) -> Dict[str, Any]:
    summary = _read_json(path)
    rel_parts = path.relative_to(dataset_root).parts
    experiment_id = str(summary.get("experiment_id") or rel_parts[0])
    model_id = str(summary.get("model_id") or rel_parts[1])
    form_id = str(summary.get("form_id") or rel_parts[2])
    answer_run_id = str(summary.get("answer_run_id") or rel_parts[3])
    trial_id = str(summary.get("trial_id") or rel_parts[4])
    pre_success = summary.get("pre_successful_submit_verified_correctness")
    pre_first = summary.get("pre_first_submit_verified_correctness")
    verified = summary.get("verified_correctness")
    submit_success = bool(summary.get("submit_success"))
    if pre_success is not None:
        scored_correctness = pre_success
        scored_correctness_source = "pre_successful_submit"
    elif pre_first is not None:
        scored_correctness = pre_first
        scored_correctness_source = "pre_first_submit"
    else:
        scored_correctness = verified
        scored_correctness_source = "final_verification"
    metric_warning = ""
    if submit_success and pre_success is None and pre_first is None:
        metric_warning = "submitted_without_pre_submit_snapshot; final verification may run on confirmation page"
    row = {
        "experiment_id": experiment_id,
        "model_id": model_id,
        "model_kind": summary.get("model_kind"),
        "track": summary.get("track"),
        "form_id": form_id,
        "answer_run_id": answer_run_id,
        "trial_id": trial_id,
        "success": summary.get("success"),
        "submit_success": summary.get("submit_success"),
        "stop_reason": summary.get("stop_reason"),
        "failure_category": summary.get("failure_category"),
        "question_total": summary.get("question_total"),
        "verified_correctness": summary.get("verified_correctness"),
        "attempted_correctness": summary.get("attempted_correctness"),
        "pre_successful_submit_verified_correctness": summary.get("pre_successful_submit_verified_correctness"),
        "pre_first_submit_verified_correctness": summary.get("pre_first_submit_verified_correctness"),
        "scored_correctness": scored_correctness,
        "scored_correctness_source": scored_correctness_source,
        "metric_warning": metric_warning,
        "submit_attempt_count": summary.get("submit_attempt_count"),
        "action_count": summary.get("action_count") or summary.get("trace_action_count"),
        "duration_s": summary.get("duration_s"),
        "reference_action_count": summary.get("reference_action_count"),
        "reference_duration_s": summary.get("reference_duration_s"),
        "action_overhead_ratio": summary.get("action_overhead_ratio"),
        "time_overhead_ratio": summary.get("time_overhead_ratio"),
        "run_completed_utc": summary.get("run_completed_utc"),
        "summary_path": _rel(path),
    }
    return {key: "" if row.get(key) is None else row.get(key) for key in FIELDS}


def _iter_rows(dataset_root: Path, experiment_ids: Sequence[str]) -> Iterable[Dict[str, Any]]:
    wanted = {str(item).strip() for item in experiment_ids if str(item).strip()}
    for path in sorted(dataset_root.glob("**/summary.json")):
        experiment_id = path.relative_to(dataset_root).parts[0]
        if wanted and experiment_id not in wanted:
            continue
        yield _row_from_summary(path, dataset_root)


def _sort_key(row: Dict[str, Any]) -> tuple:
    return (
        str(row.get("experiment_id") or ""),
        str(row.get("model_id") or ""),
        str(row.get("form_id") or ""),
        str(row.get("answer_run_id") or ""),
        str(row.get("trial_id") or ""),
    )


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("experiment_id") or "")
        grouped.setdefault(key, []).append(row)
    out: Dict[str, Dict[str, Any]] = {}
    for key, items in sorted(grouped.items()):
        submits = [item for item in items if str(item.get("submit_success")).lower() == "true"]
        successes = [item for item in items if str(item.get("success")).lower() == "true"]
        out[key] = {
            "trials": len(items),
            "successes": len(successes),
            "submit_successes": len(submits),
            "models": sorted({str(item.get("model_id") or "") for item in items if item.get("model_id")}),
            "forms": sorted({str(item.get("form_id") or "") for item in items if item.get("form_id")}),
        }
    return out


def _write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    aggregates = _aggregate(rows)
    lines = [
        "# Evaluation Results Tracker",
        "",
        f"Last updated: {now}",
        "",
        "This file is generated by `scripts/update_eval_results_tracker.py` from `data/model_baselines/**/summary.json`.",
        "",
        "## Experiment Summary",
        "",
        "| Experiment | Trials | Success | Submit Success | Models | Forms |",
        "|---|---:|---:|---:|---|---|",
    ]
    for exp, item in aggregates.items():
        lines.append(
            "| {exp} | {trials} | {successes} | {submit_successes} | {models} | {forms} |".format(
                exp=exp,
                trials=item["trials"],
                successes=item["successes"],
                submit_successes=item["submit_successes"],
                models=", ".join(item["models"]),
                forms=", ".join(item["forms"]),
            )
        )
    lines.extend(
        [
            "",
            "## Trial Rows",
            "",
            "| Experiment | Model | Form | Run | Trial | Submit | Verified | Scored | Source | Stop | Warning | Summary |",
            "|---|---|---|---|---|---:|---:|---:|---|---|---|---|",
        ]
    )
    for row in rows:
        verified = f"{row.get('verified_correctness')}/{row.get('question_total')}"
        scored = f"{row.get('scored_correctness')}/{row.get('question_total')}"
        summary = str(row.get("summary_path") or "")
        lines.append(
            f"| {row.get('experiment_id')} | {row.get('model_id')} | {row.get('form_id')} | "
            f"{row.get('answer_run_id')} | {row.get('trial_id')} | {row.get('submit_success')} | "
            f"{verified} | {scored} | {row.get('scored_correctness_source')} | {row.get('stop_reason')} | "
            f"{row.get('metric_warning')} | `{summary}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a consolidated evaluation results tracker.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--experiment-id",
        action="append",
        default=[],
        help="Experiment to include. Can be passed multiple times. Defaults to all summaries present under dataset-root.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    rows = sorted(_iter_rows(dataset_root, args.experiment_id), key=_sort_key)
    _write_csv(rows, output_dir / "metrics.csv")
    _write_jsonl(rows, output_dir / "metrics.jsonl")
    _write_markdown(rows, output_dir / "README.md")
    print(f"[INFO] wrote {len(rows)} rows to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
