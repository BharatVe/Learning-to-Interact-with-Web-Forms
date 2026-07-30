#!/usr/bin/env python3
"""Create strict paired tables for the same-model FormFactory-style interface study."""

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/baselines/formfactory_qwen3vl_comparison_analysis.json"
EXPORT_ROOT = ROOT / "data/model_baseline_exports"
DOC_PATH = ROOT / "docs/eval_results/FORMFACTORY_QWEN3VL_RESULTS.md"
PREFIX = "formfactory_qwen3vl_"
MODEL_ACTIONS = {
    "browser_click", "browser_type", "browser_fill_form", "browser_select_option",
    "browser_check", "browser_uncheck", "browser_press_key", "browser_mouse_move_xy",
    "browser_mouse_click_xy", "browser_mouse_wheel", "browser_wait_for",
}


def load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def form_ids() -> List[str]:
    return sorted(path.parent.name for path in (ROOT / "src/forms").glob("*/spec.json"))


def summary_paths(condition: Dict[str, Any], answer_run_id: str) -> List[Path]:
    source = condition["source"]
    if source["kind"] == "experiment":
        base = ROOT / "data/model_baselines" / source["experiment_id"] / source["model_id"]
        return sorted(base.glob(f"*/{answer_run_id}/trial_*/summary.json")) if base.exists() else []
    if source["kind"] == "export_csv":
        paths = []
        with resolve(source["path"]).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("model_id") == source["model_id"] and row.get("answer_run_id") == answer_run_id:
                    paths.append(resolve(row["source_summary"]))
        return paths
    raise ValueError(f"Unsupported source kind: {source.get('kind')}")


def audit_map(path: Path, model: Optional[str]) -> Dict[Tuple[str, str], str]:
    if not model:
        return {}
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") == model:
                result[(row["form_id"], row["question_id"])] = row["audit_status"]
    return result


def scored_fields(summary: Dict[str, Any], annotations: Dict[str, Any]) -> List[Dict[str, Any]]:
    questions = [row for row in annotations.get("questions", []) if isinstance(row, dict)]
    by_id = {str(row.get("question_id")): row for row in questions}
    pre = summary.get("pre_successful_submit_field_states")
    if not isinstance(pre, list):
        pre = annotations.get("pre_successful_submit_field_states")
    if isinstance(pre, list) and pre:
        rows = []
        for raw in pre:
            if not isinstance(raw, dict):
                continue
            row = dict(raw)
            row["widget_type"] = by_id.get(str(row.get("question_id")), {}).get("widget_type")
            rows.append(row)
        return rows
    return questions


def actions(summary: Dict[str, Any], annotations: Dict[str, Any]) -> Dict[str, Optional[int]]:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    if not artifacts and isinstance(annotations.get("artifacts"), dict):
        artifacts = annotations["artifacts"]
    raw_path = artifacts.get("trace_path")
    path = Path(raw_path) if isinstance(raw_path, str) else None
    if path is not None and not path.is_absolute():
        path = ROOT / path
    raw_events = 0
    normalized = 0
    if path and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except Exception:
                continue
            if not isinstance(event, dict) or not event.get("name"):
                continue
            raw_events += 1
            name = str(event["name"])
            if name not in MODEL_ACTIONS:
                continue
            if name == "browser_fill_form":
                arguments = event.get("args") if isinstance(event.get("args"), dict) else {}
                fields = arguments.get("fields") if isinstance(arguments.get("fields"), list) else []
                normalized += max(1, len(fields))
            else:
                normalized += 1
    decisions = summary.get("tool_call_count")
    if decisions is None:
        decisions = summary.get("action_count")
    return {
        "raw_trace_events": raw_events if path and path.exists() else None,
        "model_action_decisions": int(decisions) if decisions is not None else None,
        "normalized_model_actions": normalized if path and path.exists() else None,
    }


def make_trial(condition: Dict[str, Any], path: Path, audits: Dict[Tuple[str, str], str]) -> Dict[str, Any]:
    summary = load_json(path)
    annotations_path = path.with_name("annotations.json")
    annotations = load_json(annotations_path) if annotations_path.exists() else {}
    fields = scored_fields(summary, annotations)
    current_form = str(summary.get("form_id") or path.parents[3].name)
    non_dropdown = [row for row in fields if row.get("widget_type") != "dropdown"]
    dropdown = [row for row in fields if row.get("widget_type") == "dropdown"]
    nd_correct = sum(bool(row.get("verified_correct")) for row in non_dropdown)
    if audits:
        dd_confirmed = sum(audits.get((current_form, str(row.get("question_id")))) == "confirmed_correct" for row in dropdown)
        dd_unresolved = sum(audits.get((current_form, str(row.get("question_id")))) == "unresolved_excerpt_gap" for row in dropdown)
    else:
        dd_confirmed = sum(bool(row.get("verified_correct")) for row in dropdown)
        dd_unresolved = 0
    total = len(fields) or int(summary.get("question_total") or 0)
    lower = nd_correct + dd_confirmed
    upper = lower + dd_unresolved
    return {
        "condition_id": condition["id"], "condition_label": condition["label"],
        "interface": condition["interface"],
        "task_mode": summary.get("task_mode") or condition["task_mode"],
        "form_id": current_form, "answer_run_id": summary.get("answer_run_id"),
        "experiment_id": summary.get("experiment_id"), "trial_id": summary.get("trial_id"),
        "success": bool(summary.get("success")), "submit_success": bool(summary.get("submit_success")),
        "question_total": total, "non_dropdown_total": len(non_dropdown),
        "non_dropdown_correct": nd_correct, "dropdown_total": len(dropdown),
        "dropdown_confirmed_correct": dd_confirmed, "dropdown_unresolved": dd_unresolved,
        "correct_lower": lower, "correct_upper": upper,
        "full_fill_lower": total > 0 and lower == total,
        "full_fill_upper": total > 0 and upper == total,
        "duration_s": summary.get("duration_s"), "stop_reason": summary.get("stop_reason"),
        "failure_category": summary.get("failure_category"),
        "source_summary": str(path.relative_to(ROOT)), **actions(summary, annotations),
    }


def percentile(values: List[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * p)]


def bootstrap_ratio(rows: List[Dict[str, Any]], num: str, den: Optional[str], samples: int, rng: random.Random) -> Tuple[float, float]:
    estimates = []
    for _ in range(samples):
        draw = [rows[rng.randrange(len(rows))] for _ in rows]
        numerator = sum(float(row[num]) for row in draw)
        denominator = sum(float(row[den]) for row in draw) if den else len(draw)
        estimates.append(numerator / denominator if denominator else 0.0)
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def aggregate(condition: Dict[str, Any], rows: List[Dict[str, Any]], samples: int, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    nd_ci = bootstrap_ratio(rows, "non_dropdown_correct", "non_dropdown_total", samples, rng)
    fill_ci = bootstrap_ratio(rows, "full_fill_lower", None, samples, rng)
    submit_ci = bootstrap_ratio(rows, "submit_success", None, samples, rng)
    total = sum(row["question_total"] for row in rows)
    nd_total = sum(row["non_dropdown_total"] for row in rows)
    durations = [float(row["duration_s"]) for row in rows if row.get("duration_s") is not None]
    decisions = [float(row["model_action_decisions"]) for row in rows if row.get("model_action_decisions") is not None]
    normalized = [float(row["normalized_model_actions"]) for row in rows if row.get("normalized_model_actions") is not None]
    return {
        "condition_id": condition["id"], "condition_label": condition["label"],
        "task_mode": condition["task_mode"], "trials": len(rows), "fields": total,
        "non_dropdown_field_accuracy": sum(row["non_dropdown_correct"] for row in rows) / nd_total,
        "non_dropdown_accuracy_ci_low": nd_ci[0], "non_dropdown_accuracy_ci_high": nd_ci[1],
        "all_field_accuracy_lower": sum(row["correct_lower"] for row in rows) / total,
        "all_field_accuracy_upper": sum(row["correct_upper"] for row in rows) / total,
        "full_fill_rate_lower": mean(row["full_fill_lower"] for row in rows),
        "full_fill_rate_upper": mean(row["full_fill_upper"] for row in rows),
        "full_fill_lower_ci_low": fill_ci[0], "full_fill_lower_ci_high": fill_ci[1],
        "submit_success_rate": mean(row["submit_success"] for row in rows),
        "submit_success_ci_low": submit_ci[0], "submit_success_ci_high": submit_ci[1],
        "median_duration_s": median(durations) if durations else None,
        "median_model_action_decisions": median(decisions) if decisions else None,
        "median_normalized_model_actions": median(normalized) if normalized else None,
    }


def paired(spec: Dict[str, Any], cohorts: Dict[str, List[Dict[str, Any]]], samples: int, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    left = {row["form_id"]: row for row in cohorts.get(spec["left"], [])}
    right = {row["form_id"]: row for row in cohorts.get(spec["right"], [])}
    rows = []
    for current_form in sorted(set(left).intersection(right)):
        a, b = left[current_form], right[current_form]
        rows.append({
            "comparison_id": spec["id"], "form_id": current_form,
            "non_dropdown_accuracy_diff": a["non_dropdown_correct"] / a["non_dropdown_total"] - b["non_dropdown_correct"] / b["non_dropdown_total"],
            "full_fill_lower_diff": int(a["full_fill_lower"]) - int(b["full_fill_lower"]),
            "submit_success_diff": int(a["submit_success"]) - int(b["submit_success"]),
            "duration_diff_s": float(a["duration_s"]) - float(b["duration_s"]),
        })
    summary = []
    if rows:
        rng = random.Random(seed)
        record = {"comparison_id": spec["id"], "paired_forms": len(rows)}
        for metric in ("non_dropdown_accuracy_diff", "full_fill_lower_diff", "submit_success_diff", "duration_diff_s"):
            draws = [mean(rows[rng.randrange(len(rows))][metric] for _ in rows) for _ in range(samples)]
            record[f"mean_{metric}"] = mean(row[metric] for row in rows)
            record[f"{metric}_ci_low"] = percentile(draws, 0.025)
            record[f"{metric}_ci_high"] = percentile(draws, 0.975)
        summary.append(record)
    return rows, summary


def write_csv(export_root: Path, name: str, rows: List[Dict[str, Any]]) -> None:
    path = export_root / f"{PREFIX}{name}.csv"
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--export-root", type=Path, default=EXPORT_ROOT)
    parser.add_argument("--doc-path", type=Path, default=DOC_PATH)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    export_root = args.export_root.resolve()
    doc_path = args.doc_path.resolve()
    export_root.mkdir(parents=True, exist_ok=True)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    config = load_json(args.config.resolve())
    expected_forms = form_ids()
    if len(expected_forms) != int(config["expected_form_count"]):
        raise ValueError("The registered form set changed; update the frozen analysis cohort deliberately.")
    audit_path = resolve(config["dropdown_audit_csv"])
    cohorts: Dict[str, List[Dict[str, Any]]] = {}
    statuses, problems = [], []
    for condition in config["conditions"]:
        audits = audit_map(audit_path, condition.get("dropdown_audit_model"))
        rows = [make_trial(condition, path, audits) for path in summary_paths(condition, config["answer_run_id"])]
        counts = {form: sum(row["form_id"] == form for row in rows) for form in expected_forms}
        missing = [form for form, count in counts.items() if count == 0]
        duplicates = [form for form, count in counts.items() if count > 1]
        for row in rows:
            if row["task_mode"] != condition["task_mode"]:
                problems.append(f"{condition['id']}: task-mode mismatch in {row['source_summary']}")
        if missing:
            problems.append(f"{condition['id']}: missing {len(missing)} forms")
        if duplicates:
            problems.append(f"{condition['id']}: duplicate cells {duplicates}")
        cohorts[condition["id"]] = sorted(rows, key=lambda row: row["form_id"])
        statuses.append({
            "condition_id": condition["id"], "task_mode": condition["task_mode"],
            "trials_found": len(rows), "forms_found": sum(count > 0 for count in counts.values()),
            "forms_expected": len(expected_forms), "complete": not missing and not duplicates,
        })
    complete = {row["condition_id"] for row in statuses if row["complete"]}
    trial_rows = [row for condition in config["conditions"] for row in cohorts[condition["id"]]]
    aggregate_rows = [aggregate(condition, cohorts[condition["id"]], int(config["bootstrap_samples"]), int(config["bootstrap_seed"])) for condition in config["conditions"] if condition["id"] in complete]
    paired_rows, paired_summaries = [], []
    for spec in config["paired_comparisons"]:
        if spec["left"] not in complete or spec["right"] not in complete:
            continue
        rows, summaries = paired(spec, cohorts, int(config["bootstrap_samples"]), int(config["bootstrap_seed"]))
        paired_rows.extend(rows)
        paired_summaries.extend(summaries)
    write_csv(export_root, "cohort_status", statuses)
    write_csv(export_root, "trial_metrics", trial_rows)
    write_csv(export_root, "aggregate_metrics", aggregate_rows)
    write_csv(export_root, "paired_form_differences", paired_rows)
    write_csv(export_root, "paired_summary", paired_summaries)
    manifest = {
        "analysis_id": config["analysis_id"], "model_family": config["model_family"],
        "answer_run_id": config["answer_run_id"], "form_ids": expected_forms,
        "conditions": statuses, "problems": problems,
        "metric_policy": {
            "primary": "non-dropdown field accuracy with 10,000 form-level bootstrap samples",
            "dropdown": "current corrected verifier in all four newly run conditions",
            "submission": "pre-successful-submit field state when available; final state otherwise",
            "actions": "secondary only; excludes setup, observation, harness verification, synchronization, and close; expands browser_fill_form by field count",
        },
    }
    (export_root / f"{PREFIX}analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    def pct(value: Any) -> str:
        return f"{100.0 * float(value):.1f}%"

    doc_lines = [
        "# Qwen3-VL FormFactory-style vs direct-MCP comparison",
        "",
        "This analysis uses only the explicit cohorts in `configs/baselines/formfactory_qwen3vl_comparison_analysis.json`; it never scans unrelated legacy experiments.",
        "",
        "The primary endpoint is non-dropdown field accuracy with a form-level bootstrap confidence interval. All four cohorts use the current corrected verifier. Submission-enabled correctness is captured immediately before successful submission. Action counts are secondary because coordinate UI primitives and semantic MCP calls have different granularity.",
        "",
        f"Current completion: **{len(complete)}/{len(config['conditions'])} cohorts**.",
        "",
    ]
    if problems:
        doc_lines.extend([
            "> **Interim analysis:** the incomplete cohort is excluded from aggregate and paired tables; no partial paired result is published.",
            "",
            "Incomplete checks:",
            "",
            *[f"- {problem}" for problem in problems],
            "",
        ])
    doc_lines.extend([
        "## Complete-cohort results",
        "",
        "| Condition | Trials | Primary accuracy (95% CI) | All-field accuracy | Full fills | Submissions |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in aggregate_rows:
        doc_lines.append(
            f"| {row['condition_label']} | {row['trials']} | "
            f"{pct(row['non_dropdown_field_accuracy'])} "
            f"({pct(row['non_dropdown_accuracy_ci_low'])}–{pct(row['non_dropdown_accuracy_ci_high'])}) | "
            f"{pct(row['all_field_accuracy_lower'])} | "
            f"{pct(row['full_fill_rate_lower'])} | {pct(row['submit_success_rate'])} |"
        )
    doc_lines.extend(["", "## Complete paired comparisons", ""])
    if paired_summaries:
        doc_lines.extend([
            "Differences are visual minus direct MCP; negative values favour direct MCP.",
            "",
            "| Comparison | Forms | Primary difference (95% CI) | Full-fill difference | Submission difference |",
            "|---|---:|---:|---:|---:|",
        ])
        for row in paired_summaries:
            doc_lines.append(
                f"| {row['comparison_id']} | {row['paired_forms']} | "
                f"{pct(row['mean_non_dropdown_accuracy_diff'])} "
                f"({pct(row['non_dropdown_accuracy_diff_ci_low'])}–{pct(row['non_dropdown_accuracy_diff_ci_high'])}) | "
                f"{pct(row['mean_full_fill_lower_diff'])} | {pct(row['mean_submit_success_diff'])} |"
            )
    else:
        doc_lines.append("No registered paired comparison has two complete 50-form cohorts yet.")
    doc_lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- This is a FormFactory-style adaptation, not a reproduction of the FormFactory study.",
        "- Each condition has one deterministic run per form; confidence intervals quantify variation across forms, not decoding randomness.",
        "- Correctness is the primary comparison. Semantic MCP actions and primitive visual actions do not have equivalent granularity.",
        "",
        "Machine-readable outputs are the `formfactory_qwen3vl_*` files under `data/model_baseline_exports`.",
        "",
    ])
    doc_path.write_text("\n".join(doc_lines), encoding="utf-8")
    print(json.dumps({"conditions": statuses, "problems": problems}, indent=2))
    return 2 if args.strict and problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
