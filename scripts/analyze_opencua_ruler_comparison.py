#!/usr/bin/env python3
"""Build and validate the matched OpenCUA pixel-ruler comparison bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASELINE_EXPERIMENT = "formfactory_style_opencua_fill_only_50_r2_step32_20260728"
RULER_EXPERIMENT = "formfactory_style_opencua_ruler_fill_only_50_r2_step32_20260807"
MODEL_ID = "computer_use_opencua_32b"
ANSWER_RUN_ID = "run_0002"
BOOTSTRAP_SEED = 20260808
BOOTSTRAP_RESAMPLES = 20_000
OUTPUT_REL = Path("evaluation_additions/opencua_ruler_comparison")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_value_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8"))


def csv_text(columns: list[str], rows: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def load_cohort(root: Path, experiment_id: str) -> dict[str, dict[str, Any]]:
    cohort: dict[str, dict[str, Any]] = {}
    experiment_root = root / "data/model_baselines" / experiment_id
    for annotations_path in sorted(experiment_root.glob("**/annotations.json")):
        annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
        if annotations.get("model_id") != MODEL_ID or annotations.get("answer_run_id") != ANSWER_RUN_ID:
            continue
        form_id = str(annotations["form_id"])
        if form_id in cohort:
            raise RuntimeError(f"duplicate form in {experiment_id}: {form_id}")
        summary_path = annotations_path.with_name("summary.json")
        if not summary_path.is_file():
            raise RuntimeError(f"missing summary for {annotations_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cohort[form_id] = {
            "annotations": annotations,
            "summary": summary,
            "annotations_path": annotations_path,
            "summary_path": summary_path,
            "annotations_sha256": sha256_file(annotations_path),
            "summary_sha256": sha256_file(summary_path),
        }
    return cohort


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "not_recorded"


def percentile(sorted_values: list[float], probability: float) -> float:
    index = min(len(sorted_values) - 1, max(0, int(probability * len(sorted_values))))
    return sorted_values[index]


def bootstrap_ci(values: list[float], seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    size = len(values)
    samples = [
        sum(values[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    samples.sort()
    return percentile(samples, 0.025), percentile(samples, 0.975)


def exact_mcnemar_p(baseline_wrong_ruler_correct: int, baseline_correct_ruler_wrong: int) -> float:
    discordant = baseline_wrong_ruler_correct + baseline_correct_ruler_wrong
    if discordant == 0:
        return 1.0
    smaller = min(baseline_wrong_ruler_correct, baseline_correct_ruler_wrong)
    lower_tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * lower_tail)


def pct(numerator: float, denominator: float = 1.0) -> float:
    return round(100.0 * numerator / denominator, 6)


def metric_row(condition: str, rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    fields = sum(row["question_total"] for row in rows)
    correct = sum(row[f"{prefix}_correct"] for row in rows)
    non_fields = sum(row["non_dropdown_total"] for row in rows)
    non_correct = sum(row[f"{prefix}_non_dropdown_correct"] for row in rows)
    actions = [row[f"{prefix}_action_count"] for row in rows]
    durations = [row[f"{prefix}_duration_s"] for row in rows]
    stops = Counter(row[f"{prefix}_stop_reason"] for row in rows)
    return {
        "condition": condition,
        "trials": len(rows),
        "fields": fields,
        "correct_fields": correct,
        "all_field_accuracy_pct": pct(correct, fields),
        "non_dropdown_fields": non_fields,
        "correct_non_dropdown_fields": non_correct,
        "non_dropdown_accuracy_pct": pct(non_correct, non_fields),
        "full_fills": sum(bool(row[f"{prefix}_full_fill"]) for row in rows),
        "full_fill_rate_pct": pct(sum(bool(row[f"{prefix}_full_fill"]) for row in rows), len(rows)),
        "mean_actions": round(statistics.mean(actions), 6),
        "median_actions": round(statistics.median(actions), 6),
        "mean_duration_s": round(statistics.mean(durations), 6),
        "median_duration_s": round(statistics.median(durations), 6),
        "max_steps_exceeded": stops["max_steps_exceeded"],
        "repeated_action_loop": stops["repeated_action_loop"],
    }


def build_bundle(root: Path) -> dict[str, str]:
    baseline = load_cohort(root, BASELINE_EXPERIMENT)
    ruler = load_cohort(root, RULER_EXPERIMENT)
    if len(baseline) != 50 or len(ruler) != 50 or set(baseline) != set(ruler):
        raise RuntimeError("comparison requires the same 50 forms in both cohorts")

    form_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    widget_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    position_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    transitions: Counter[str] = Counter()

    for form_id in sorted(baseline):
        base = baseline[form_id]
        ruled = ruler[form_id]
        for key in ("model_id", "track", "task_mode", "answer_run_id"):
            if base["summary"].get(key) != ruled["summary"].get(key):
                raise ValueError(f"Protocol mismatch for {form_id}: {key}")
        if base["summary"].get("task_mode") != "fill_only_done":
            raise ValueError(f"Unexpected task mode for {form_id}")
        ignored_params = {"base_url", "run_label", "ruler_overlay", "ruler_config"}
        base_params = {
            key: value
            for key, value in base["annotations"]["run_params"].items()
            if key not in ignored_params
        }
        ruler_params = {
            key: value
            for key, value in ruled["annotations"]["run_params"].items()
            if key not in ignored_params
        }
        if base_params != ruler_params:
            raise ValueError(f"Run-parameter mismatch for {form_id}")
        for key in ("locale", "timezone", "devicePixelRatio", "userAgent", "url"):
            if base["annotations"]["environment"].get(key) != ruled["annotations"]["environment"].get(key):
                raise ValueError(f"Browser-environment mismatch for {form_id}: {key}")
        base_questions = {item["question_id"]: item for item in base["annotations"]["questions"]}
        ruler_questions = {item["question_id"]: item for item in ruled["annotations"]["questions"]}
        if set(base_questions) != set(ruler_questions):
            raise RuntimeError(f"question mismatch for {form_id}")
        ordered_ids = [item["question_id"] for item in base["annotations"]["questions"]]
        base_correct = ruler_correct = 0
        base_non = ruler_non = non_total = 0
        for index, question_id in enumerate(ordered_ids, start=1):
            base_question = base_questions[question_id]
            ruler_question = ruler_questions[question_id]
            if base_question.get("widget_type") != ruler_question.get("widget_type"):
                raise RuntimeError(f"widget mismatch for {form_id}/{question_id}")
            if base_question.get("value") != ruler_question.get("value"):
                raise RuntimeError(f"answer mismatch for {form_id}/{question_id}")
            base_ok = bool(base_question.get("verified_correct"))
            ruler_ok = bool(ruler_question.get("verified_correct"))
            base_correct += base_ok
            ruler_correct += ruler_ok
            transition = (
                "both_correct" if base_ok and ruler_ok else
                "baseline_correct_ruler_wrong" if base_ok else
                "baseline_wrong_ruler_correct" if ruler_ok else
                "both_wrong"
            )
            transitions[transition] += 1
            widget_type = str(base_question.get("widget_type") or "unknown")
            widget_counts[widget_type][0] += int(base_ok)
            widget_counts[widget_type][1] += int(ruler_ok)
            widget_counts[widget_type][2] += 1
            question_total = len(ordered_ids)
            position_bucket = (
                "first" if index <= math.ceil(question_total / 3) else
                "middle" if index <= math.ceil(2 * question_total / 3) else
                "last"
            )
            position_counts[position_bucket][0] += int(base_ok)
            position_counts[position_bucket][1] += int(ruler_ok)
            position_counts[position_bucket][2] += 1
            if widget_type != "dropdown":
                base_non += base_ok
                ruler_non += ruler_ok
                non_total += 1
            field_rows.append({
                "form_id": form_id,
                "answer_run_id": ANSWER_RUN_ID,
                "question_id": question_id,
                "question_index": index,
                "position_bucket": position_bucket,
                "widget_type": widget_type,
                "expected_value_sha256": stable_value_hash(base_question.get("value")),
                "baseline_actual_value_sha256": stable_value_hash(base_question.get("actual_value")),
                "ruler_actual_value_sha256": stable_value_hash(ruler_question.get("actual_value")),
                "baseline_verified_correct": str(base_ok).lower(),
                "ruler_verified_correct": str(ruler_ok).lower(),
                "transition": transition,
                "baseline_annotations_sha256": base["annotations_sha256"],
                "ruler_annotations_sha256": ruled["annotations_sha256"],
            })

        question_total = len(ordered_ids)
        base_summary = base["summary"]
        ruler_summary = ruled["summary"]
        form_rows.append({
            "form_id": form_id,
            "answer_run_id": ANSWER_RUN_ID,
            "question_total": question_total,
            "non_dropdown_total": non_total,
            "baseline_correct": base_correct,
            "ruler_correct": ruler_correct,
            "baseline_accuracy_pct": pct(base_correct, question_total),
            "ruler_accuracy_pct": pct(ruler_correct, question_total),
            "accuracy_difference_pp": pct(ruler_correct - base_correct, question_total),
            "baseline_non_dropdown_correct": base_non,
            "ruler_non_dropdown_correct": ruler_non,
            "baseline_non_dropdown_accuracy_pct": pct(base_non, non_total),
            "ruler_non_dropdown_accuracy_pct": pct(ruler_non, non_total),
            "non_dropdown_difference_pp": pct(ruler_non - base_non, non_total),
            "baseline_full_fill": base_correct == question_total,
            "ruler_full_fill": ruler_correct == question_total,
            "baseline_action_count": int(base_summary["action_count"]),
            "ruler_action_count": int(ruler_summary["action_count"]),
            "action_difference": int(ruler_summary["action_count"]) - int(base_summary["action_count"]),
            "baseline_duration_s": float(base_summary["duration_s"]),
            "ruler_duration_s": float(ruler_summary["duration_s"]),
            "duration_difference_s": float(ruler_summary["duration_s"]) - float(base_summary["duration_s"]),
            "baseline_stop_reason": base_summary["stop_reason"],
            "ruler_stop_reason": ruler_summary["stop_reason"],
            "baseline_trial_id": base_summary["trial_id"],
            "ruler_trial_id": ruler_summary["trial_id"],
            "baseline_summary_path": base["summary_path"].relative_to(root).as_posix(),
            "ruler_summary_path": ruled["summary_path"].relative_to(root).as_posix(),
            "baseline_annotations_sha256": base["annotations_sha256"],
            "ruler_annotations_sha256": ruled["annotations_sha256"],
        })
        if int(base["summary"]["verified_correctness"]) != base_correct:
            raise ValueError(f"Baseline summary/annotation mismatch for {form_id}")
        if int(ruled["summary"]["verified_correctness"]) != ruler_correct:
            raise ValueError(f"Ruler summary/annotation mismatch for {form_id}")

    form_differences = [row["accuracy_difference_pp"] / 100.0 for row in form_rows]
    non_differences = [row["non_dropdown_difference_pp"] / 100.0 for row in form_rows]
    form_ci = bootstrap_ci(form_differences, BOOTSTRAP_SEED)
    non_ci = bootstrap_ci(non_differences, BOOTSTRAP_SEED + 1)
    summary_rows = [
        metric_row("OpenCUA visual, no ruler", form_rows, "baseline"),
        metric_row("OpenCUA visual, pixel ruler", form_rows, "ruler"),
    ]
    widget_rows = [{
        "widget_type": widget,
        "fields": counts[2],
        "baseline_correct": counts[0],
        "ruler_correct": counts[1],
        "baseline_accuracy_pct": pct(counts[0], counts[2]),
        "ruler_accuracy_pct": pct(counts[1], counts[2]),
        "difference_pp": pct(counts[1] - counts[0], counts[2]),
    } for widget, counts in sorted(widget_counts.items())]
    position_rows = [{
        "position_bucket": bucket,
        "fields": position_counts[bucket][2],
        "baseline_correct": position_counts[bucket][0],
        "ruler_correct": position_counts[bucket][1],
        "baseline_accuracy_pct": pct(position_counts[bucket][0], position_counts[bucket][2]),
        "ruler_accuracy_pct": pct(position_counts[bucket][1], position_counts[bucket][2]),
        "difference_pp": pct(position_counts[bucket][1] - position_counts[bucket][0], position_counts[bucket][2]),
    } for bucket in ("first", "middle", "last")]
    robustness = {
        "comparison": "pixel ruler minus no ruler",
        "analysis_grain": "50 matched forms; 409 matched fields",
        "all_field_micro_difference_pp": round(summary_rows[1]["all_field_accuracy_pct"] - summary_rows[0]["all_field_accuracy_pct"], 6),
        "mean_paired_form_difference_pp": round(statistics.mean(row["accuracy_difference_pp"] for row in form_rows), 6),
        "mean_paired_form_difference_bootstrap_95_ci_pp": [round(pct(form_ci[0]), 6), round(pct(form_ci[1]), 6)],
        "non_dropdown_micro_difference_pp": round(summary_rows[1]["non_dropdown_accuracy_pct"] - summary_rows[0]["non_dropdown_accuracy_pct"], 6),
        "mean_paired_non_dropdown_difference_pp": round(statistics.mean(row["non_dropdown_difference_pp"] for row in form_rows), 6),
        "mean_paired_non_dropdown_difference_bootstrap_95_ci_pp": [round(pct(non_ci[0]), 6), round(pct(non_ci[1]), 6)],
        "bootstrap": {"unit": "form", "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "percentile_interval": [0.025, 0.975]},
        "form_outcomes": {
            "ruler_better": sum(row["accuracy_difference_pp"] > 0 for row in form_rows),
            "tie": sum(row["accuracy_difference_pp"] == 0 for row in form_rows),
            "ruler_worse": sum(row["accuracy_difference_pp"] < 0 for row in form_rows),
        },
        "field_transitions": dict(sorted(transitions.items())),
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(
            transitions["baseline_wrong_ruler_correct"], transitions["baseline_correct_ruler_wrong"]
        ),
        "claim_boundary": "Descriptive matched comparison only; one run per model-form setting does not estimate run-to-run variability or establish a causal ruler effect.",
    }

    common_parameters = {
        key: ruler[next(iter(ruler))]["annotations"]["run_params"].get(key)
        for key in (
            "max_steps", "max_new_tokens", "model_visible_history_window", "history_images",
            "timeout_s", "api_timeout_s", "browser_mcp_timeout_ms", "viewport", "headless",
            "coordinate_space", "coordinate_transform", "interface_profile", "task_mode",
            "include_symbolic_support", "invalid_action_budget", "disable_action_coercion",
        )
    }
    protocol = {
        "matched_identity": {"model_id": MODEL_ID, "answer_run_id": ANSWER_RUN_ID, "forms": 50},
        "common_recorded_parameters": common_parameters,
        "intended_difference": {
            "baseline_ruler_overlay": False,
            "ruler_ruler_overlay": True,
            "ruler_config": {"tick_spacing_px": 100, "major_tick_spacing_px": 500, "band_px": 36},
            "ruler_prompt_instruction": "Use labeled pixel rulers on the top and left edges as references for absolute click coordinates.",
        },
        "operational_differences": ["run date", "run label", "local model-server port"],
        "unverified_comparability": ["source commit was not recorded for either evaluation batch"],
        "model_visible_verifier_feedback": False,
    }

    paired_form_columns = list(form_rows[0])
    paired_field_columns = list(field_rows[0])
    summary_columns = list(summary_rows[0])
    widget_columns = list(widget_rows[0])
    position_columns = list(position_rows[0])
    outputs: dict[str, str] = {
        "comparison_summary.csv": csv_text(summary_columns, summary_rows),
        "paired_forms.csv": csv_text(paired_form_columns, form_rows),
        "paired_fields.csv": csv_text(paired_field_columns, field_rows),
        "widget_summary.csv": csv_text(widget_columns, widget_rows),
        "position_summary.csv": csv_text(position_columns, position_rows),
        "robustness.json": json.dumps(robustness, indent=2) + "\n",
        "protocol_comparison.json": json.dumps(protocol, indent=2) + "\n",
    }

    completion_time = max(item["summary"]["run_completed_utc"] for item in ruler.values())
    all_delta = robustness["all_field_micro_difference_pp"]
    ci_low, ci_high = robustness["mean_paired_form_difference_bootstrap_95_ci_pp"]
    better = robustness["form_outcomes"]["ruler_better"]
    tied = robustness["form_outcomes"]["tie"]
    worse = robustness["form_outcomes"]["ruler_worse"]
    largest_changes = sorted(form_rows, key=lambda row: abs(row["accuracy_difference_pp"]), reverse=True)[:12]
    accuracy_chart = [
        {"condition": row["condition"], "scope": scope, "accuracy_pct": row[field], "correct_fields": row[correct], "fields": row[total]}
        for row in summary_rows
        for scope, field, correct, total in (
            ("All fields", "all_field_accuracy_pct", "correct_fields", "fields"),
            ("Non-dropdown fields", "non_dropdown_accuracy_pct", "correct_non_dropdown_fields", "non_dropdown_fields"),
        )
    ]
    widget_chart = [
        {"widget_type": row["widget_type"], "condition": condition, "accuracy_pct": row[field], "correct_fields": row[correct], "fields": row["fields"]}
        for row in widget_rows
        for condition, field, correct in (
            ("No ruler", "baseline_accuracy_pct", "baseline_correct"),
            ("Pixel ruler", "ruler_accuracy_pct", "ruler_correct"),
        )
    ]
    report_source_sql = (
        "SELECT * FROM comparison_summary; "
        "SELECT form_id, question_total, baseline_accuracy_pct, ruler_accuracy_pct, accuracy_difference_pp "
        "FROM paired_forms ORDER BY ABS(accuracy_difference_pp) DESC LIMIT 12; "
        "SELECT widget_type, fields, baseline_correct, ruler_correct, baseline_accuracy_pct, ruler_accuracy_pct "
        "FROM widget_summary;"
    )
    source_labels = [
        {
            "id": "matched_annotations",
            "label": "Matched OpenCUA run-2 summaries and field annotations for the ruler and no-ruler 50-form cohorts.",
            "query": {"sql": report_source_sql},
        },
        {"id": "job_accounting", "label": "Slurm output and model-server logs for completed job 2314476."},
    ]
    detailed_sources = [
        {
            "id": "matched_annotations",
            "query": {
                "engine": "local_files",
                "language": "python",
                "description": "Matched form- and field-level comparison generated by scripts/analyze_opencua_ruler_comparison.py.",
                "sql": report_source_sql,
                "tables_used": [
                    f"data/model_baselines/{BASELINE_EXPERIMENT}/**/summary.json",
                    f"data/model_baselines/{BASELINE_EXPERIMENT}/**/annotations.json",
                    f"data/model_baselines/{RULER_EXPERIMENT}/**/summary.json",
                    f"data/model_baselines/{RULER_EXPERIMENT}/**/annotations.json",
                ],
                "filters": ["model_id = computer_use_opencua_32b", "answer_run_id = run_0002", "50 forms present in both cohorts"],
                "metric_definitions": {
                    "all_field_accuracy": "verified-correct target fields divided by all target fields",
                    "non_dropdown_accuracy": "verified-correct non-dropdown target fields divided by non-dropdown target fields",
                    "full_fill": "every target field in the form is verified correct",
                    "paired_form_difference": "ruler form accuracy minus no-ruler form accuracy in percentage points",
                },
                "executed_at": completion_time,
            },
        },
        {
            "id": "job_accounting",
            "query": {
                "engine": "local_files",
                "language": "text",
                "description": "Job completion, recorded run parameters, GPU allocation, and vLLM server configuration.",
                "tables_used": ["logs/slurm/opencua-direct-2314476.out", "logs/slurm/opencua-vllm-2314476.log"],
                "filters": ["Slurm job 2314476"],
                "executed_at": completion_time,
            },
        },
    ]
    aggregate_table_columns = [
        {"field": "condition", "label": "Condition", "type": "text"},
        {"field": "trials", "label": "Forms", "type": "number"},
        {"field": "correct_fields", "label": "Correct fields", "type": "number"},
        {"field": "fields", "label": "Fields", "type": "number"},
        {"field": "all_field_accuracy_pct", "label": "All-field accuracy (%)", "type": "number"},
        {"field": "non_dropdown_accuracy_pct", "label": "Non-dropdown accuracy (%)", "type": "number"},
        {"field": "full_fills", "label": "Full fills", "type": "number"},
        {"field": "mean_actions", "label": "Mean actions", "type": "number"},
        {"field": "mean_duration_s", "label": "Mean duration (s)", "type": "number"},
    ]
    largest_change_rows = [{
        "form_id": row["form_id"],
        "baseline_accuracy_pct": row["baseline_accuracy_pct"],
        "ruler_accuracy_pct": row["ruler_accuracy_pct"],
        "accuracy_difference_pp": row["accuracy_difference_pp"],
        "question_total": row["question_total"],
    } for row in largest_changes]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "OpenCUA Pixel-Ruler Evaluation",
            "description": "Matched 50-form technical comparison of FormFactory-style OpenCUA with and without model-visible pixel rulers.",
            "generatedAt": completion_time,
            "cards": [],
            "charts": [
                {
                    "id": "accuracy_comparison_chart",
                    "title": "Verified field accuracy by evaluation condition",
                    "subtitle": "Matched run-2 forms; 409 total fields and 384 non-dropdown fields.",
                    "type": "bar",
                    "dataset": "accuracy_chart",
                    "sourceId": "matched_annotations",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "scope", "type": "nominal", "label": "Scoring scope"},
                        "y": {"field": "accuracy_pct", "type": "quantitative", "label": "Verified accuracy (%)"},
                        "color": {"field": "condition", "type": "nominal", "label": "Condition"},
                        "tooltip": [
                            {"field": "correct_fields", "type": "quantitative", "label": "Correct fields"},
                            {"field": "fields", "type": "quantitative", "label": "Fields"},
                        ],
                    },
                },
                {
                    "id": "widget_comparison_chart",
                    "title": "Verified accuracy by widget type",
                    "subtitle": "Matched fields; descriptive widget cut without multiple-comparison adjustment.",
                    "type": "bar",
                    "dataset": "widget_chart",
                    "sourceId": "matched_annotations",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "widget_type", "type": "nominal", "label": "Widget type"},
                        "y": {"field": "accuracy_pct", "type": "quantitative", "label": "Verified accuracy (%)"},
                        "color": {"field": "condition", "type": "nominal", "label": "Condition"},
                        "tooltip": [
                            {"field": "correct_fields", "type": "quantitative", "label": "Correct fields"},
                            {"field": "fields", "type": "quantitative", "label": "Fields"},
                        ],
                    },
                },
            ],
            "tables": [
                {
                    "id": "aggregate_table",
                    "title": "Aggregate comparison",
                    "subtitle": "Two matched 50-form cohorts using the same model, answer run, and recorded limits.",
                    "dataset": "aggregate",
                    "sourceId": "matched_annotations",
                    "defaultSort": {"field": "all_field_accuracy_pct", "direction": "desc"},
                    "columns": aggregate_table_columns,
                },
                {
                    "id": "largest_changes_table",
                    "title": "Largest absolute form-level changes",
                    "subtitle": "Twelve forms with the largest absolute difference; percentage points are ruler minus no ruler.",
                    "dataset": "largest_changes",
                    "sourceId": "matched_annotations",
                    "defaultSort": {"field": "accuracy_difference_pp", "direction": "asc"},
                    "columns": [
                        {"field": "form_id", "label": "Form", "type": "text"},
                        {"field": "question_total", "label": "Fields", "type": "number"},
                        {"field": "baseline_accuracy_pct", "label": "No ruler (%)", "type": "number"},
                        {"field": "ruler_accuracy_pct", "label": "Pixel ruler (%)", "type": "number"},
                        {"field": "accuracy_difference_pp", "label": "Difference (pp)", "type": "number", "movement": True},
                    ],
                },
            ],
            "sources": source_labels,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# OpenCUA Pixel-Ruler Evaluation"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "matched_annotations",
                    "body": (
                        "## Technical summary\n\n"
                        f"The pixel ruler produced **no observed aggregate benefit**. Verified all-field accuracy changed from **{summary_rows[0]['all_field_accuracy_pct']:.1f}%** "
                        f"without the ruler to **{summary_rows[1]['all_field_accuracy_pct']:.1f}%** with it (**{all_delta:+.1f} percentage points**). "
                        f"The mean paired form difference was **{robustness['mean_paired_form_difference_pp']:+.1f} points** with a form-bootstrap 95% interval of **{ci_low:+.1f} to {ci_high:+.1f} points**.\n\n"
                        f"The ruler was better on **{better} forms**, tied on **{tied}**, and worse on **{worse}**. Both conditions produced **0 full fills**, and nearly all trials reached the 32-step cap. "
                        "These results support a descriptive statement of no observed ruler benefit, not a causal or run-to-run variability claim."
                    ),
                },
                {
                    "id": "aggregate_story",
                    "type": "markdown",
                    "sourceId": "matched_annotations",
                    "body": (
                        "## Accuracy remained effectively unchanged\n\n"
                        f"Across **409 matched fields**, the ruler condition verified **{summary_rows[1]['correct_fields']}** correct fields versus **{summary_rows[0]['correct_fields']}** without it. "
                        f"After excluding dropdowns, accuracy changed from **{summary_rows[0]['non_dropdown_accuracy_pct']:.1f}%** to **{summary_rows[1]['non_dropdown_accuracy_pct']:.1f}%**. "
                        "The comparison chart starts at zero and uses the same scale for both conditions."
                    ),
                },
                {"id": "accuracy_chart_block", "type": "chart", "chartId": "accuracy_comparison_chart"},
                {"id": "aggregate_table_block", "type": "table", "tableId": "aggregate_table"},
                {
                    "id": "paired_story",
                    "type": "markdown",
                    "sourceId": "matched_annotations",
                    "body": (
                        "## Gains and regressions mostly offset each other\n\n"
                        f"At field level, **{transitions['baseline_wrong_ruler_correct']}** baseline misses became correct with the ruler, while **{transitions['baseline_correct_ruler_wrong']}** baseline-correct fields became wrong. "
                        f"The exact paired McNemar test gives **p = {robustness['mcnemar_exact_two_sided_p']:.3f}**. This inferential check does not show evidence of a systematic directional change. "
                        "The largest individual gain and loss should be treated as diagnostic examples rather than independently tested effects."
                    ),
                },
                {"id": "largest_changes_table_block", "type": "table", "tableId": "largest_changes_table"},
                {
                    "id": "widget_story",
                    "type": "markdown",
                    "sourceId": "matched_annotations",
                    "body": (
                        "## Short-text gains did not generalize across controls\n\n"
                        "Short-text accuracy increased, but single-choice, multi-choice, and dropdown accuracy declined; date and time fields remained at zero in both conditions. "
                        "Because this cut spans several widget types with unequal denominators and no multiple-comparison adjustment, it is descriptive only."
                    ),
                },
                {"id": "widget_chart_block", "type": "chart", "chartId": "widget_comparison_chart"},
                {
                    "id": "scope_definitions",
                    "type": "markdown",
                    "sourceId": "matched_annotations",
                    "body": (
                        "## Scope and metric definitions\n\n"
                        "The unit of pairing is one Google Form using `run_0002`. The cohorts contain the same **50 forms**, **409 target fields**, expected values, question identifiers, and widget types. "
                        "A field is correct only when the saved final verifier marks it correct. All-field accuracy is verified-correct fields divided by all target fields; non-dropdown accuracy applies the same rule after excluding dropdown fields. "
                        "A full fill requires every target field in a form to be verified correct."
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": "matched_annotations",
                    "body": (
                        "## Matched comparison and robustness checks\n\n"
                        "The analysis joins cohorts on form, answer run, and question identifier; asserts identical expected answers and widget types; and rejects duplicates or incomplete cohorts. "
                        f"The uncertainty interval resamples the **50 paired forms** with replacement for **{BOOTSTRAP_RESAMPLES:,} iterations** using seed **{BOOTSTRAP_SEED}**. "
                        "A two-sided exact McNemar test uses the discordant paired field outcomes as a secondary robustness check. Micro-averaged field accuracy and the mean paired form difference are both reported to expose weighting sensitivity."
                    ),
                },
                {
                    "id": "protocol",
                    "type": "markdown",
                    "sourceId": "job_accounting",
                    "body": (
                        "## Recorded protocol was matched except for the ruler intervention\n\n"
                        "Both cohorts used `xlangai/OpenCUA-32B`, the FormFactory-style visual interface, 96 output tokens, 32 steps, two model-visible history turns, one history image, a 1440×900 headless Chrome viewport, and the same timeout and coordinate-transform settings. "
                        "The new condition overlaid 100-pixel ticks, 500-pixel major ticks, and a 36-pixel top/left band, plus one prompt instruction to use those rulers for absolute coordinates. "
                        "Operational differences were run date, run label, and local server port."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Limitations and claim boundary\n\n"
                        "- There is one evaluation run per form-condition setting, so this bundle does not estimate run-to-run model variability.\n"
                        "- Source commits were not recorded in the original cohort metadata; matching recorded parameters cannot rule out every unrecorded code difference between dates.\n"
                        "- Bootstrap intervals quantify variation across forms, not decoding randomness.\n"
                        "- Widget and form-level cuts are exploratory and are not adjusted for multiple comparisons.\n"
                        "- The result is specific to OpenCUA, these 50 forms, run 2 answers, and this ruler design."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Recommended next steps\n\n"
                        "1. Keep the paper claim descriptive: **the tested ruler did not improve the matched OpenCUA evaluation**.\n"
                        "2. Do not collect more ruler runs unless this negative result is important enough to justify repeated independent seeds.\n"
                        "3. If diagnosing the failure mechanism remains important, inspect the existing paired field and action traces before proposing another visual aid; do not infer causality from isolated form changes.\n"
                        "4. Add this batch to release settings and provenance only if the ruler result will remain in the paper."
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "- Would repeated independent runs preserve the near-zero aggregate difference?\n"
                        "- Are coordinate errors actually the limiting mechanism, or does the dominant failure arise from scrolling, retention, and step-budget exhaustion?\n"
                        "- Would a controlled design that changes only ruler visibility within the same source commit alter the conclusion?"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generated_at": completion_time,
            "datasets": {
                "aggregate": summary_rows,
                "accuracy_chart": accuracy_chart,
                "widget_chart": widget_chart,
                "largest_changes": largest_change_rows,
            },
            "access_issues": [],
        },
        "sources": detailed_sources,
    }
    outputs["artifact.json"] = json.dumps(artifact, indent=2) + "\n"

    readme = f"""# OpenCUA pixel-ruler comparison

This directory is a compact, GitHub-trackable evidence bundle for the matched 50-form OpenCUA ruler evaluation. Open `report.html` in any current browser for the primary technical report.

## Result

The tested pixel ruler did not improve aggregate verified correctness: all-field accuracy changed from **{summary_rows[0]['all_field_accuracy_pct']:.1f}%** to **{summary_rows[1]['all_field_accuracy_pct']:.1f}%** ({all_delta:+.1f} percentage points). The paired form bootstrap interval is **{ci_low:+.1f} to {ci_high:+.1f} points**. This supports a descriptive negative result, not a causal or variability claim.

## Files

- `report.html`: self-contained portable report for local viewing.
- `artifact.json`: canonical source for the portable report.
- `comparison_summary.csv`: condition-level metrics.
- `paired_forms.csv`: one matched row per form.
- `paired_fields.csv`: one matched row per field; values are hashed to avoid exposing answer contents.
- `widget_summary.csv` and `position_summary.csv`: exploratory breakdowns.
- `robustness.json`: bootstrap, paired transitions, and exact McNemar result.
- `protocol_comparison.json`: recorded matched settings, intervention, and comparability caveats.
- `source_manifest.json`: source and output hashes, row counts, keys, and commands.

## Regenerate and validate

```bash
python3 scripts/analyze_opencua_ruler_comparison.py
python3 scripts/analyze_opencua_ruler_comparison.py --check
```

The data exporter uses only the Python standard library. `report.html` is already packaged and does not need Python, Node.js, a local server, or network access to open.

## Report validation

The portable-report packager passed artifact validation, packaging, and structural verification. Automated Chromium rendering was not completed in this environment: automatic browser discovery was unavailable, and an explicit attempt with the experiment Chromium timed out after 11.2 seconds. Open `report.html` locally for the remaining visual inspection before publication.

## Paper-safe wording

> In a matched descriptive comparison across 50 forms, adding labeled pixel rulers to the OpenCUA screenshots did not improve verified field accuracy (38.1% with rulers versus 39.1% without). The mean paired form difference was {robustness['mean_paired_form_difference_pp']:+.1f} percentage points (form-bootstrap 95% interval {ci_low:+.1f} to {ci_high:+.1f}). Because each form-condition setting was evaluated once and source commits were not recorded in the original cohort metadata, this result does not establish a causal ruler effect or run-to-run model variability.
"""
    outputs["README.md"] = readme

    source_manifest = {
        "analysis_commit": git_commit(root),
        "analysis_date": "2026-08-08",
        "slurm_job": {"job_id": "2314476", "state": "COMPLETED", "elapsed": "07:21:55", "exit_code": "0:0"},
        "source_cohorts": [
            {
                "experiment_id": experiment,
                "manifest_path": f"data/model_baselines/{experiment}/manifest.jsonl",
                "manifest_sha256": sha256_file(root / f"data/model_baselines/{experiment}/manifest.jsonl"),
                "manifest_rows": 50,
                "primary_key": ["model_id", "form_id", "answer_run_id", "trial_id"],
            }
            for experiment in (BASELINE_EXPERIMENT, RULER_EXPERIMENT)
        ],
        "generated_files": [],
        "generation_command": "python3 scripts/analyze_opencua_ruler_comparison.py",
        "validation_commands": [
            "python3 scripts/analyze_opencua_ruler_comparison.py --check",
            "node <data-analytics-plugin>/skills/build-report/scripts/deliver_portable_artifact.mjs --input evaluation_additions/opencua_ruler_comparison/artifact.json --output evaluation_additions/opencua_ruler_comparison/report.html",
        ],
        "report_validation": {
            "artifact_validation": "passed",
            "packaging": "passed",
            "structural_verification": "passed",
            "browser_render_verification": "not_completed",
            "reason": "Experiment Chromium timed out after 11.2 seconds; local visual inspection remains.",
        },
    }
    for name, content in sorted(outputs.items()):
        if name == "source_manifest.json":
            continue
        row_count: int | str = "not_applicable"
        if name.endswith(".csv"):
            row_count = max(0, len(content.splitlines()) - 1)
        source_manifest["generated_files"].append({"path": (OUTPUT_REL / name).as_posix(), "sha256": sha256_bytes(content.encode("utf-8")), "row_count": row_count})
    outputs["source_manifest.json"] = json.dumps(source_manifest, indent=2) + "\n"
    return outputs


def validate_csv_keys(output_dir: Path) -> None:
    keys = {
        "comparison_summary.csv": ["condition"],
        "paired_forms.csv": ["form_id", "answer_run_id"],
        "paired_fields.csv": ["form_id", "answer_run_id", "question_id"],
        "widget_summary.csv": ["widget_type"],
        "position_summary.csv": ["position_bucket"],
    }
    expected_rows = {"comparison_summary.csv": 2, "paired_forms.csv": 50, "paired_fields.csv": 409, "widget_summary.csv": 7, "position_summary.csv": 3}
    for name, columns in keys.items():
        with (output_dir / name).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_rows[name]:
            raise RuntimeError(f"unexpected row count for {name}: {len(rows)}")
        values = [tuple(row[column] for column in columns) for row in rows]
        if len(values) != len(set(values)) or any(not all(value) for value in values):
            raise RuntimeError(f"invalid primary key for {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = root / OUTPUT_REL
    outputs = build_bundle(root)
    if args.check:
        stale = [
            name for name, content in outputs.items()
            if not (output_dir / name).is_file() or (output_dir / name).read_text(encoding="utf-8") != content
        ]
        if stale:
            raise SystemExit(f"missing or stale comparison outputs: {', '.join(stale)}")
        validate_csv_keys(output_dir)
        print("validated OpenCUA ruler bundle: 50 forms, 409 fields")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (output_dir / name).write_text(content, encoding="utf-8")
    validate_csv_keys(output_dir)
    print("wrote OpenCUA ruler bundle: 50 forms, 409 fields")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
