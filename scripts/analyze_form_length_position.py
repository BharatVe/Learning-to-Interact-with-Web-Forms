#!/usr/bin/env python3
"""Analyze field accuracy by form length and absolute question position.

The analysis intentionally uses an explicit cohort manifest. It never scans for
or substitutes unrelated historical experiments. Non-dropdown fields are the
primary population because the canonical and later visual runs used different
versions of the dropdown verifier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


DEFAULT_CONFIG = "configs/baselines/form_length_position_analysis.json"
DEFAULT_OUTPUT_DIR = "docs/eval_results/form_length_position_analysis"
LENGTH_BUCKETS = ("6", "7-8", "9-10", "11-12")


@dataclass(frozen=True)
class FieldOutcome:
    condition_id: str
    condition_label: str
    interface: str
    form_id: str
    question_id: str
    question_index: int
    question_total: int
    widget_type: str
    verified_correct: bool
    source_path: str


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def pct(numerator: float, denominator: float) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "big")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    columns = list(fields or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_csv_condition(project_root: Path, condition: dict[str, Any]) -> list[FieldOutcome]:
    source = condition["source"]
    source_path = project_root / source["path"]
    model_filter = str(source["model_filter"])
    outcomes: list[FieldOutcome] = []
    with source_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") != model_filter:
                continue
            outcomes.append(
                FieldOutcome(
                    condition_id=str(condition["id"]),
                    condition_label=str(condition["label"]),
                    interface=str(condition["interface"]),
                    form_id=str(row["form_id"]),
                    question_id=str(row["question_id"]),
                    question_index=int(row["question_index"]),
                    question_total=int(row["question_total"]),
                    widget_type=str(row.get("widget_type") or "unknown"),
                    verified_correct=parse_bool(row.get("verified_correct")),
                    source_path=str(source["path"]),
                )
            )
    return outcomes


def load_experiment_condition(
    project_root: Path,
    condition: dict[str, Any],
    answer_run_id: str,
) -> list[FieldOutcome]:
    source = condition["source"]
    base = (
        project_root
        / "data"
        / "model_baselines"
        / str(source["experiment_id"])
        / str(source["model_id"])
    )
    pattern = f"*/{answer_run_id}/trial_*/annotations.json"
    annotation_paths = sorted(base.glob(pattern))
    outcomes: list[FieldOutcome] = []
    for path in annotation_paths:
        annotation = read_json(path)
        if annotation.get("answer_run_id") != answer_run_id:
            raise ValueError(f"Unexpected answer run in {path}: {annotation.get('answer_run_id')}")
        form_id = str(annotation.get("form_id") or "")
        questions = annotation.get("questions")
        if not form_id or not isinstance(questions, list):
            raise ValueError(f"Malformed annotations: {path}")
        for index, question in enumerate(questions, start=1):
            outcomes.append(
                FieldOutcome(
                    condition_id=str(condition["id"]),
                    condition_label=str(condition["label"]),
                    interface=str(condition["interface"]),
                    form_id=form_id,
                    question_id=str(question.get("question_id") or ""),
                    question_index=index,
                    question_total=len(questions),
                    widget_type=str(question.get("widget_type") or "unknown"),
                    verified_correct=bool(question.get("verified_correct")),
                    source_path=str(path.relative_to(project_root)),
                )
            )
    return outcomes


def load_conditions(project_root: Path, config: dict[str, Any]) -> dict[str, list[FieldOutcome]]:
    answer_run_id = str(config["answer_run_id"])
    loaded: dict[str, list[FieldOutcome]] = {}
    for condition in config["conditions"]:
        source_kind = condition["source"]["kind"]
        if source_kind == "field_outcomes_csv":
            rows = load_csv_condition(project_root, condition)
        elif source_kind == "experiment_annotations":
            rows = load_experiment_condition(project_root, condition, answer_run_id)
        else:
            raise ValueError(f"Unsupported source kind: {source_kind}")
        loaded[str(condition["id"])] = rows
    return loaded


def validate_cohorts(
    cohorts: dict[str, list[FieldOutcome]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_forms = int(config["expected_form_count"])
    expected_fields = int(config["expected_field_count"])
    expected_non_dropdown = int(config["expected_non_dropdown_field_count"])
    reference_keys: set[tuple[str, str]] | None = None
    reference_shape: dict[tuple[str, str], tuple[int, int, str]] | None = None
    audit_rows: list[dict[str, Any]] = []

    for condition in config["conditions"]:
        condition_id = str(condition["id"])
        rows = cohorts.get(condition_id, [])
        forms = {row.form_id for row in rows}
        keys = [(row.form_id, row.question_id) for row in rows]
        unique_keys = set(keys)
        non_dropdown = [row for row in rows if row.widget_type != "dropdown"]
        shape = {
            (row.form_id, row.question_id): (row.question_index, row.question_total, row.widget_type)
            for row in rows
        }

        problems: list[str] = []
        if len(forms) != expected_forms:
            problems.append(f"forms={len(forms)} expected={expected_forms}")
        if len(rows) != expected_fields:
            problems.append(f"fields={len(rows)} expected={expected_fields}")
        if len(unique_keys) != len(rows):
            problems.append(f"duplicate_keys={len(rows) - len(unique_keys)}")
        if len(non_dropdown) != expected_non_dropdown:
            problems.append(
                f"non_dropdown_fields={len(non_dropdown)} expected={expected_non_dropdown}"
            )
        if any(not row.question_id for row in rows):
            problems.append("blank_question_id")
        if any(row.question_index < 1 or row.question_index > row.question_total for row in rows):
            problems.append("invalid_question_position")

        if reference_keys is None:
            reference_keys = unique_keys
            reference_shape = shape
        else:
            if unique_keys != reference_keys:
                problems.append(
                    f"question_universe_mismatch missing={len(reference_keys - unique_keys)} "
                    f"extra={len(unique_keys - reference_keys)}"
                )
            elif shape != reference_shape:
                problems.append("question_shape_or_widget_mismatch")

        audit_rows.append(
            {
                "condition_id": condition_id,
                "condition_label": condition["label"],
                "interface": condition["interface"],
                "forms": len(forms),
                "fields": len(rows),
                "unique_form_question_keys": len(unique_keys),
                "non_dropdown_fields": len(non_dropdown),
                "non_dropdown_correct": sum(row.verified_correct for row in non_dropdown),
                "non_dropdown_accuracy_pct": pct(
                    sum(row.verified_correct for row in non_dropdown), len(non_dropdown)
                ),
                "status": "valid" if not problems else "invalid",
                "problems": "; ".join(problems),
            }
        )
        if problems:
            raise ValueError(f"Invalid cohort {condition_id}: {'; '.join(problems)}")

    return audit_rows


def length_bucket(question_total: int) -> str:
    if question_total == 6:
        return "6"
    if 7 <= question_total <= 8:
        return "7-8"
    if 9 <= question_total <= 10:
        return "9-10"
    if 11 <= question_total <= 12:
        return "11-12"
    raise ValueError(f"Unexpected form length: {question_total}")


def form_level_bootstrap(
    rows: Sequence[FieldOutcome],
    samples: int,
    seed: int,
    metric: Callable[[list[FieldOutcome]], float],
) -> tuple[float, float]:
    by_form: dict[str, list[FieldOutcome]] = defaultdict(list)
    for row in rows:
        by_form[row.form_id].append(row)
    form_ids = sorted(by_form)
    if not form_ids:
        return 0.0, 0.0
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        sampled: list[FieldOutcome] = []
        for form_id in rng.choices(form_ids, k=len(form_ids)):
            sampled.extend(by_form[form_id])
        estimates.append(metric(sampled))
    return round(percentile(estimates, 0.025), 2), round(percentile(estimates, 0.975), 2)


def micro_accuracy(rows: Sequence[FieldOutcome]) -> float:
    return 100.0 * sum(row.verified_correct for row in rows) / len(rows) if rows else 0.0


def macro_form_accuracy(rows: Sequence[FieldOutcome]) -> float:
    by_form: dict[str, list[FieldOutcome]] = defaultdict(list)
    for row in rows:
        by_form[row.form_id].append(row)
    if not by_form:
        return 0.0
    return 100.0 * statistics.mean(
        sum(row.verified_correct for row in form_rows) / len(form_rows)
        for form_rows in by_form.values()
    )


def form_length_distribution(reference_rows: Sequence[FieldOutcome]) -> list[dict[str, Any]]:
    form_lengths: dict[str, int] = {}
    for row in reference_rows:
        previous = form_lengths.setdefault(row.form_id, row.question_total)
        if previous != row.question_total:
            raise ValueError(f"Inconsistent question total for {row.form_id}")
    counts = Counter(form_lengths.values())
    return [
        {
            "question_count": question_count,
            "form_count": counts[question_count],
            "share_of_50_forms_pct": pct(counts[question_count], len(form_lengths)),
        }
        for question_count in sorted(counts)
    ]


def position_table(
    cohorts: dict[str, list[FieldOutcome]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    samples = int(config["bootstrap_samples"])
    base_seed = int(config["bootstrap_seed"])
    rows_out: list[dict[str, Any]] = []
    for condition in config["conditions"]:
        condition_id = str(condition["id"])
        primary = [row for row in cohorts[condition_id] if row.widget_type != "dropdown"]
        for position in range(1, max(row.question_total for row in primary) + 1):
            selected = [row for row in primary if row.question_index == position]
            if not selected:
                continue
            ci_low, ci_high = form_level_bootstrap(
                selected,
                samples,
                stable_seed(base_seed, condition_id, "position", str(position)),
                micro_accuracy,
            )
            correct = sum(row.verified_correct for row in selected)
            rows_out.append(
                {
                    "condition_id": condition_id,
                    "condition_label": condition["label"],
                    "interface": condition["interface"],
                    "question_position": position,
                    "eligible_forms": len({row.form_id for row in selected}),
                    "target_fields": len(selected),
                    "correct_fields": correct,
                    "accuracy_pct": pct(correct, len(selected)),
                    "ci_95_low_pct": ci_low,
                    "ci_95_high_pct": ci_high,
                    "low_sample_flag": len(selected) < 10,
                }
            )
    return rows_out


def form_length_table(
    cohorts: dict[str, list[FieldOutcome]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    samples = int(config["bootstrap_samples"])
    base_seed = int(config["bootstrap_seed"])
    rows_out: list[dict[str, Any]] = []
    for condition in config["conditions"]:
        condition_id = str(condition["id"])
        primary = [row for row in cohorts[condition_id] if row.widget_type != "dropdown"]
        for bucket in LENGTH_BUCKETS:
            selected = [row for row in primary if length_bucket(row.question_total) == bucket]
            forms = len({row.form_id for row in selected})
            micro_low, micro_high = form_level_bootstrap(
                selected,
                samples,
                stable_seed(base_seed, condition_id, "length_micro", bucket),
                micro_accuracy,
            )
            macro_low, macro_high = form_level_bootstrap(
                selected,
                samples,
                stable_seed(base_seed, condition_id, "length_macro", bucket),
                macro_form_accuracy,
            )
            correct = sum(row.verified_correct for row in selected)
            rows_out.append(
                {
                    "condition_id": condition_id,
                    "condition_label": condition["label"],
                    "interface": condition["interface"],
                    "length_bucket": bucket,
                    "forms": forms,
                    "target_fields": len(selected),
                    "correct_fields": correct,
                    "micro_accuracy_pct": pct(correct, len(selected)),
                    "micro_ci_95_low_pct": micro_low,
                    "micro_ci_95_high_pct": micro_high,
                    "macro_form_accuracy_pct": round(macro_form_accuracy(selected), 2),
                    "macro_ci_95_low_pct": macro_low,
                    "macro_ci_95_high_pct": macro_high,
                    "low_sample_flag": forms < 5,
                }
            )
    return rows_out


def first_last_three_table(
    cohorts: dict[str, list[FieldOutcome]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    samples = int(config["bootstrap_samples"])
    base_seed = int(config["bootstrap_seed"])
    rows_out: list[dict[str, Any]] = []
    for condition in config["conditions"]:
        condition_id = str(condition["id"])
        primary = [row for row in cohorts[condition_id] if row.widget_type != "dropdown"]
        by_form: dict[str, list[FieldOutcome]] = defaultdict(list)
        for row in primary:
            by_form[row.form_id].append(row)
        form_scores: dict[str, tuple[float, float, int, int, int, int]] = {}
        for form_id, form_rows in by_form.items():
            early = [row for row in form_rows if row.question_index <= 3]
            late = [row for row in form_rows if row.question_index > row.question_total - 3]
            if not early or not late:
                raise ValueError(f"Missing early/late fields for {condition_id}/{form_id}")
            form_scores[form_id] = (
                micro_accuracy(early),
                micro_accuracy(late),
                sum(row.verified_correct for row in early),
                len(early),
                sum(row.verified_correct for row in late),
                len(late),
            )
        form_ids = sorted(form_scores)
        rng = random.Random(stable_seed(base_seed, condition_id, "first_last_three"))
        boot_differences: list[float] = []
        for _ in range(samples):
            sample_ids = rng.choices(form_ids, k=len(form_ids))
            early_macro = statistics.mean(form_scores[form_id][0] for form_id in sample_ids)
            late_macro = statistics.mean(form_scores[form_id][1] for form_id in sample_ids)
            boot_differences.append(late_macro - early_macro)

        early_macro = statistics.mean(score[0] for score in form_scores.values())
        late_macro = statistics.mean(score[1] for score in form_scores.values())
        early_correct = sum(score[2] for score in form_scores.values())
        early_targets = sum(score[3] for score in form_scores.values())
        late_correct = sum(score[4] for score in form_scores.values())
        late_targets = sum(score[5] for score in form_scores.values())
        rows_out.append(
            {
                "condition_id": condition_id,
                "condition_label": condition["label"],
                "interface": condition["interface"],
                "forms": len(form_ids),
                "early_correct_fields": early_correct,
                "early_target_fields": early_targets,
                "early_micro_accuracy_pct": pct(early_correct, early_targets),
                "early_macro_form_accuracy_pct": round(early_macro, 2),
                "late_correct_fields": late_correct,
                "late_target_fields": late_targets,
                "late_micro_accuracy_pct": pct(late_correct, late_targets),
                "late_macro_form_accuracy_pct": round(late_macro, 2),
                "late_minus_early_macro_pp": round(late_macro - early_macro, 2),
                "difference_ci_95_low_pp": round(percentile(boot_differences, 0.025), 2),
                "difference_ci_95_high_pp": round(percentile(boot_differences, 0.975), 2),
            }
        )
    return rows_out


def lookup(rows: Iterable[dict[str, Any]], condition_id: str, key: str, value: Any) -> dict[str, Any]:
    return next(row for row in rows if row["condition_id"] == condition_id and row[key] == value)


def build_figure(
    output_path: Path,
    config: dict[str, Any],
    distribution_rows: list[dict[str, Any]],
    position_rows: list[dict[str, Any]],
    length_rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "gemini_35_flash": "#0072B2",
        "opencua_direct_mcp": "#E69F00",
        "qwen3_text": "#009E73",
        "qwen3_vl_direct_mcp": "#CC79A7",
        "opencua_formfactory_style": "#D55E00",
        "qwen3_vl_formfactory_style": "#6A3D9A",
    }
    markers = ["o", "s", "^", "D", "P", "X"]
    form_counts_exact = {int(row["question_count"]): int(row["form_count"]) for row in distribution_rows}
    bucket_counts = {
        "6": form_counts_exact.get(6, 0),
        "7-8": form_counts_exact.get(7, 0) + form_counts_exact.get(8, 0),
        "9-10": form_counts_exact.get(9, 0) + form_counts_exact.get(10, 0),
        "11-12": form_counts_exact.get(11, 0) + form_counts_exact.get(12, 0),
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 160,
        }
    )
    fig, (ax_position, ax_length) = plt.subplots(1, 2, figsize=(14.2, 6.4))

    for index, condition in enumerate(config["conditions"]):
        condition_id = str(condition["id"])
        selected = [row for row in position_rows if row["condition_id"] == condition_id]
        selected.sort(key=lambda row: int(row["question_position"]))
        adequately_sampled = [row for row in selected if not parse_bool(row["low_sample_flag"])]
        line_style = "--" if condition["interface"] == "formfactory_style_visual" else "-"
        ax_position.plot(
            [row["question_position"] for row in adequately_sampled],
            [row["accuracy_pct"] for row in adequately_sampled],
            color=colors[condition_id],
            marker=markers[index],
            markersize=4.5,
            linewidth=1.8,
            linestyle=line_style,
            label=condition["label"],
        )
        low_sample = [row for row in selected if parse_bool(row["low_sample_flag"])]
        if low_sample:
            ax_position.scatter(
                [row["question_position"] for row in low_sample],
                [row["accuracy_pct"] for row in low_sample],
                facecolors="white",
                edgecolors=colors[condition_id],
                marker=markers[index],
                s=43,
                linewidths=1.2,
                zorder=5,
            )

    ax_position.axvspan(9.5, 12.35, color="#eeeeee", alpha=0.8, zorder=-5)
    ax_position.text(10.8, 4, "positions 10–12:\n≤7 eligible forms", ha="center", va="bottom", color="#555555", fontsize=8)
    ax_position.set_title("A. Accuracy by absolute question position")
    ax_position.set_xlabel("Question position in form")
    ax_position.set_ylabel("Non-dropdown field accuracy (%)")
    ax_position.set_xlim(0.7, 12.3)
    ax_position.set_ylim(0, 102)
    ax_position.set_xticks(range(1, 13))
    ax_position.set_yticks(range(0, 101, 20))
    ax_position.grid(axis="y", color="#d9d9d9", linewidth=0.7)

    for index, condition in enumerate(config["conditions"]):
        condition_id = str(condition["id"])
        selected = [row for row in length_rows if row["condition_id"] == condition_id]
        selected.sort(key=lambda row: LENGTH_BUCKETS.index(str(row["length_bucket"])))
        adequately_sampled = [
            (position, row)
            for position, row in enumerate(selected)
            if not parse_bool(row["low_sample_flag"])
        ]
        line_style = "--" if condition["interface"] == "formfactory_style_visual" else "-"
        ax_length.plot(
            [position for position, _ in adequately_sampled],
            [row["macro_form_accuracy_pct"] for _, row in adequately_sampled],
            color=colors[condition_id],
            marker=markers[index],
            markersize=5,
            linewidth=1.8,
            linestyle=line_style,
        )
        low_sample = [
            (position, row)
            for position, row in enumerate(selected)
            if parse_bool(row["low_sample_flag"])
        ]
        if low_sample:
            ax_length.scatter(
                [position for position, _ in low_sample],
                [row["macro_form_accuracy_pct"] for _, row in low_sample],
                facecolors="white",
                edgecolors=colors[condition_id],
                marker=markers[index],
                s=49,
                linewidths=1.2,
                zorder=5,
            )

    ax_length.set_title("B. Macro accuracy by total form length")
    ax_length.set_xlabel("Questions per form (number of forms)")
    ax_length.set_ylabel("Mean non-dropdown form accuracy (%)")
    ax_length.set_xticks(
        range(len(LENGTH_BUCKETS)),
        [f"{bucket}\n(n={bucket_counts[bucket]})" for bucket in LENGTH_BUCKETS],
    )
    ax_length.set_ylim(0, 102)
    ax_length.set_yticks(range(0, 101, 20))
    ax_length.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    ax_length.text(2.98, 4, "Open markers denote <5 forms", ha="right", va="bottom", color="#555555", fontsize=8)

    handles, labels = ax_position.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=3, frameon=False)
    fig.suptitle(
        "Field accuracy by position and form length across six 50-form conditions",
        y=1.075,
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "Primary metric excludes dropdown fields; dashed lines are FormFactory-style visual conditions. "
        "One deterministic run per form.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92), w_pad=2.5)
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_report(
    output_path: Path,
    config: dict[str, Any],
    audit_rows: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
    position_rows: list[dict[str, Any]],
    length_rows: list[dict[str, Any]],
    first_last_rows: list[dict[str, Any]],
) -> None:
    labels = {str(condition["id"]): str(condition["label"]) for condition in config["conditions"]}
    audit = {row["condition_id"]: row for row in audit_rows}
    first_last = {row["condition_id"]: row for row in first_last_rows}
    distribution = {int(row["question_count"]): int(row["form_count"]) for row in distribution_rows}

    def pos(condition_id: str, position: int) -> dict[str, Any]:
        return lookup(position_rows, condition_id, "question_position", position)

    def length(condition_id: str, bucket: str) -> dict[str, Any]:
        return lookup(length_rows, condition_id, "length_bucket", bucket)

    summary_rows = []
    for condition in config["conditions"]:
        condition_id = str(condition["id"])
        fl = first_last[condition_id]
        summary_rows.append(
            f"| {labels[condition_id]} | {audit[condition_id]['non_dropdown_accuracy_pct']:.2f}% "
            f"| {fl['early_macro_form_accuracy_pct']:.2f}% | {fl['late_macro_form_accuracy_pct']:.2f}% "
            f"| {fl['late_minus_early_macro_pp']:+.2f} pp "
            f"| {length(condition_id, '7-8')['macro_form_accuracy_pct']:.2f}% "
            f"| {length(condition_id, '9-10')['macro_form_accuracy_pct']:.2f}% |"
        )

    model_sections = [
        (
            "Gemini 3.5 Flash",
            f"Non-dropdown accuracy is {audit['gemini_35_flash']['non_dropdown_accuracy_pct']:.2f}%. "
            f"It remains {pos('gemini_35_flash', 1)['accuracy_pct']:.0f}% at question 1 and "
            f"{pos('gemini_35_flash', 4)['accuracy_pct']:.0f}% at question 4, then falls to "
            f"{pos('gemini_35_flash', 8)['accuracy_pct']:.0f}% at question 8 and "
            f"{pos('gemini_35_flash', 10)['accuracy_pct']:.0f}% at question 10. "
            f"Macro accuracy is {length('gemini_35_flash', '7-8')['macro_form_accuracy_pct']:.1f}% "
            f"on 7–8-question forms versus {length('gemini_35_flash', '9-10')['macro_form_accuracy_pct']:.1f}% "
            "on 9–10-question forms.",
        ),
        (
            "OpenCUA direct-MCP",
            f"Non-dropdown accuracy is {audit['opencua_direct_mcp']['non_dropdown_accuracy_pct']:.2f}%. "
            f"Its positional pattern is not monotonic: accuracy is {pos('opencua_direct_mcp', 1)['accuracy_pct']:.0f}% "
            f"at question 1, rises to {pos('opencua_direct_mcp', 7)['accuracy_pct']:.1f}% at question 7, "
            f"and remains {pos('opencua_direct_mcp', 9)['accuracy_pct']:.1f}% at question 9. "
            f"Nevertheless, macro form accuracy drops from {length('opencua_direct_mcp', '7-8')['macro_form_accuracy_pct']:.1f}% "
            f"for 7–8 questions to {length('opencua_direct_mcp', '9-10')['macro_form_accuracy_pct']:.1f}% "
            "for 9–10 questions. This contrast shows that total form length, field composition, and absolute position cannot be treated as the same effect.",
        ),
        (
            "Qwen3 Text",
            f"Non-dropdown accuracy is {audit['qwen3_text']['non_dropdown_accuracy_pct']:.2f}%. "
            f"Accuracy declines from {pos('qwen3_text', 1)['accuracy_pct']:.0f}% at question 1 to "
            f"{pos('qwen3_text', 7)['accuracy_pct']:.1f}% at question 7 and "
            f"{pos('qwen3_text', 10)['accuracy_pct']:.1f}% at question 10. "
            f"Macro accuracy is {length('qwen3_text', '7-8')['macro_form_accuracy_pct']:.1f}% "
            f"on 7–8-question forms and {length('qwen3_text', '9-10')['macro_form_accuracy_pct']:.1f}% "
            "on 9–10-question forms.",
        ),
        (
            "Qwen3-VL direct-MCP",
            f"Non-dropdown accuracy is {audit['qwen3_vl_direct_mcp']['non_dropdown_accuracy_pct']:.2f}%. "
            f"It is relatively stable through question 6, then falls from "
            f"{pos('qwen3_vl_direct_mcp', 7)['accuracy_pct']:.1f}% at question 7 to "
            f"{pos('qwen3_vl_direct_mcp', 10)['accuracy_pct']:.1f}% at question 10. "
            f"Macro accuracy falls from {length('qwen3_vl_direct_mcp', '7-8')['macro_form_accuracy_pct']:.1f}% "
            f"for 7–8 questions to {length('qwen3_vl_direct_mcp', '9-10')['macro_form_accuracy_pct']:.1f}% "
            "for 9–10 questions.",
        ),
        (
            "OpenCUA FormFactory-style",
            f"Non-dropdown accuracy is {audit['opencua_formfactory_style']['non_dropdown_accuracy_pct']:.2f}%. "
            f"The positional collapse is sharp: {pos('opencua_formfactory_style', 1)['accuracy_pct']:.0f}% "
            f"at question 1, {pos('opencua_formfactory_style', 4)['accuracy_pct']:.1f}% at question 4, "
            f"{pos('opencua_formfactory_style', 5)['accuracy_pct']:.1f}% at question 5, and "
            f"{pos('opencua_formfactory_style', 8)['accuracy_pct']:.0f}% from question 8 onward. "
            "The trace audit found no scrolling actions, so later-position failure is consistent with viewport traversal failure rather than answer extraction alone.",
        ),
        (
            "Qwen3-VL FormFactory-style",
            f"Non-dropdown accuracy is {audit['qwen3_vl_formfactory_style']['non_dropdown_accuracy_pct']:.2f}%. "
            f"It falls from {pos('qwen3_vl_formfactory_style', 1)['accuracy_pct']:.0f}% at question 1 "
            f"and {pos('qwen3_vl_formfactory_style', 3)['accuracy_pct']:.0f}% at question 3 to "
            f"{pos('qwen3_vl_formfactory_style', 4)['accuracy_pct']:.1f}% at question 4 and "
            f"{pos('qwen3_vl_formfactory_style', 5)['accuracy_pct']:.0f}% at question 5. "
            "The same qualitative failure across two visual models strengthens the interpretation that the screenshot-coordinate workflow is the shared stressor.",
        ),
    ]

    content = [
        "# Form Length and Question Position Analysis",
        "",
        "## Technical summary",
        "",
        "The six completed conditions use the same 50 forms and 409 form/question pairs. All forms contain at least six questions; the mean is 8.18, the median is 8, and the range is 6–12. Forty-seven of the 50 forms contain 7–10 questions.",
        "",
        "The two FormFactory-style visual conditions show the clearest absolute-position collapse. OpenCUA falls from 92% non-dropdown accuracy at question 1 to 9.3% at question 5, while Qwen3-VL falls from 100% at question 1 to 2.4% at question 4. Gemini, Qwen3 Text, and Qwen3-VL direct-MCP also deteriorate at later positions, but OpenCUA direct-MCP is a counterexample: its position curve is non-monotonic even though its performance is lower on longer forms. The evidence is therefore descriptive and does not isolate form length from widget composition or position.",
        "",
        "The primary analysis excludes dropdowns because the historical four-model cohort and later visual runs used different dropdown-verifier versions. Each condition contributes 384 comparable non-dropdown fields. Confidence intervals use 10,000 form-level bootstrap samples and describe variation across these fixed forms, not decoding randomness.",
        "",
        "![Accuracy by question position and form length](form_length_position_accuracy.png)",
        "",
        "The left panel uses absolute question positions. Open markers at positions 10–12 denote fewer than ten eligible forms. The right panel reports macro per-form accuracy so longer forms do not receive extra weight; the 6-question group contains one form and the 11–12-question group contains only two.",
        "",
        "## Comparable summaries",
        "",
        "| Condition | Overall non-dropdown | First three, macro | Last three, macro | Late − early | 7–8 questions | 9–10 questions |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *summary_rows,
        "",
        "The first/last comparison uses the structural first three and last three questions of every form. Because widget types are not evenly distributed by position, these differences should not be interpreted as a causal effect of scrolling or sequence length.",
        "",
        "## Model-specific findings",
        "",
    ]
    for title, paragraph in model_sections:
        content.extend([f"### {title}", "", paragraph, ""])

    content.extend(
        [
            "## Scope and method",
            "",
            f"The form-count distribution is: {', '.join(f'{count} form(s) with {length} questions' for length, count in sorted(distribution.items()))}. Accuracy is final verified browser-state correctness. Position curves are field-weighted; form-length curves and first/last summaries use macro per-form averages. Bootstrap sampling treats the form as the independent resampling unit.",
            "",
            "The six cohorts were loaded only from the explicit analysis manifest. Validation requires exactly 50 forms, 409 unique form/question keys, 384 non-dropdown fields, no duplicates, and an identical question index, total, and widget type for every key.",
            "",
            "## Limitations and robustness",
            "",
            "- Form length, widget type, viewport location, and question position are correlated. These cuts identify stress patterns but do not estimate independent causal effects.",
            "- Positions 10, 11, and 12 contain at most 7, 2, and 1 eligible forms respectively. The 11–12-question bucket contains only two forms.",
            "- There is one temperature-zero run per form. Bootstrap intervals measure across-form heterogeneity, not stochastic model variability.",
            "- Direct-MCP and visual actions have different granularity and observation contracts. The comparison supports claims about these implemented protocols, not an interface-independent model ranking.",
            "",
            "## Minimal FormFactory fidelity improvement",
            "",
            "A pixel-ruler overlay is the smallest paper-aligned change because it preserves the screenshot-coordinate task and action grammar. It should be evaluated as an isolated ruler ablation with all other settings unchanged. It may improve coordinate grounding, but it is unlikely to solve the dominant lack of scrolling by itself.",
            "",
            "Automatic scrolling, detailed verifier feedback, multi-action page plans, 2880×1800 rendering, the official FormFactory input documents, submission, and the paper's Click/Value metrics would materially change the workflow. Those changes belong in a separately named replication rather than being folded into this baseline.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "MPLCONFIGDIR=/tmp/form-length-position-mpl python3 scripts/analyze_form_length_position.py \\",
            "  --project-root . \\",
            "  --config configs/baselines/form_length_position_analysis.json \\",
            "  --output-dir docs/eval_results/form_length_position_analysis",
            "```",
            "",
            "The machine-readable tables and analysis manifest are stored beside this report.",
        ]
    )
    output_path.write_text("\n".join(content) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Keep the logical workspace path. On the cluster, resolving the workspace
    # symlink can point at a read-only storage mount even though the /home path
    # is the writable project view.
    logical_cwd = Path(os.environ.get("PWD") or str(Path.cwd()))
    project_root = args.project_root if args.project_root.is_absolute() else logical_cwd / args.project_root
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(config_path)
    cohorts = load_conditions(project_root, config)
    audit_rows = validate_cohorts(cohorts, config)
    reference_id = str(config["conditions"][0]["id"])
    distribution_rows = form_length_distribution(cohorts[reference_id])
    position_rows = position_table(cohorts, config)
    length_rows = form_length_table(cohorts, config)
    first_last_rows = first_last_three_table(cohorts, config)

    write_csv(output_dir / "cohort_audit.csv", audit_rows)
    write_csv(output_dir / "form_length_distribution.csv", distribution_rows)
    write_csv(output_dir / "accuracy_by_question_position.csv", position_rows)
    write_csv(output_dir / "accuracy_by_form_length.csv", length_rows)
    write_csv(output_dir / "first_last_three_summary.csv", first_last_rows)

    build_figure(
        output_dir / "form_length_position_accuracy",
        config,
        distribution_rows,
        position_rows,
        length_rows,
    )
    build_report(
        output_dir / "REPORT.md",
        config,
        audit_rows,
        distribution_rows,
        position_rows,
        length_rows,
        first_last_rows,
    )

    source_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest = {
        "analysis_id": config["analysis_id"],
        "config_path": str(config_path.relative_to(project_root)),
        "config_sha256": source_hash,
        "answer_run_id": config["answer_run_id"],
        "condition_count": len(config["conditions"]),
        "expected_form_count_per_condition": config["expected_form_count"],
        "expected_field_count_per_condition": config["expected_field_count"],
        "primary_population": "non_dropdown_fields",
        "expected_non_dropdown_field_count_per_condition": config["expected_non_dropdown_field_count"],
        "bootstrap_samples": config["bootstrap_samples"],
        "bootstrap_seed": config["bootstrap_seed"],
        "validation_status": "passed",
        "outputs": [
            "REPORT.md",
            "cohort_audit.csv",
            "form_length_distribution.csv",
            "accuracy_by_question_position.csv",
            "accuracy_by_form_length.csv",
            "first_last_three_summary.csv",
            "form_length_position_accuracy.png",
            "form_length_position_accuracy.pdf",
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
