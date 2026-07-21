#!/usr/bin/env python3
"""Validate failure-analysis grains, denominators, matrices, and report links."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ["Gemini 3.5 Flash", "OpenCUA direct-MCP", "Qwen3 Text", "Qwen3-VL"]


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    fields = rows("field_outcomes.csv")
    trials = rows("trial_action_profile.csv")
    matrix = rows("failure_matrix_by_model.csv")
    stops = rows("trial_stop_reasons_by_model.csv")
    ranks = rows("hardest_forms_by_model.csv")
    baseline = rows("baseline_primary_trial_action_profile.csv")
    items = rows("question_difficulty_cross_model.csv")
    action_efficiency = rows("action_efficiency_by_model_outcome.csv")
    action_mix = rows("action_mix_by_model.csv")
    gaps = rows("methodology_gap_register.csv")
    ideal_runs = rows("ideal_reference_runs.csv")
    model_ideal = rows("model_vs_ideal_summary.csv")
    model_ideal_forms = rows("model_vs_ideal_by_form.csv")
    action_correction = rows("action_count_correction_by_model.csv")
    dropdown_audit = rows("dropdown_verifier_audit_by_model.csv")
    dropdown_selected = rows("dropdown_selected_state_audit.csv")
    dropdown_selected_summary = rows("dropdown_selected_state_summary.csv")
    without_dropdown = rows("performance_forms_without_dropdown.csv")
    with_dropdown = rows("performance_forms_with_dropdown.csv")
    canonical_submission = rows("canonical_submission_audit.csv")
    submission_audit = rows("baseline_submission_scoring_audit.csv")
    recovered_perfect_fields = rows("baseline_recovered_perfect_submitted_fields.csv")

    require(len(fields) == 1636, f"expected 1,636 field outcomes, got {len(fields)}")
    require(len(trials) == 200, f"expected 200 trials, got {len(trials)}")
    require(len(baseline) == 978, f"expected 978 baseline trials, got {len(baseline)}")
    require(len(items) == 409, f"expected 409 cross-model question items, got {len(items)}")
    require(all(int(r["model_observations"]) == 4 for r in items),
            "cross-model question denominator is not four for every item")
    nondrop_items = [r for r in items if r["ambiguous_dropdown_measurement"] == "False"]
    require(len(nondrop_items) == 384, f"expected 384 non-dropdown items, got {len(nondrop_items)}")
    require(sum(int(r["models_failed"]) == 4 for r in nondrop_items) == 14,
            "expected 14 non-dropdown questions failed by all four conditions")
    require(len(action_efficiency) == 8, "expected model-by-outcome action-efficiency rows")
    require(sum(r["priority"] == "P0" for r in gaps) == 2, "expected two P0 methodology gaps")
    require(len(ideal_runs) == 300, f"expected 300 ideal reference runs, got {len(ideal_runs)}")
    require(all(r["usable"] == "True" and r["submit_success"] == "True" for r in ideal_runs),
            "ideal reference contains a non-usable or unsuccessful run")
    require(len(model_ideal) == 4, "expected four model-to-ideal summary rows")
    require(len(model_ideal_forms) == 200, "expected 200 model-to-ideal matched form rows")
    require(all(int(r["matched_forms"]) == 50 for r in model_ideal),
            "not every model is matched to 50 ideal run-2 forms")
    require(all(float(r["median_model_to_ideal_time_ratio"]) > 1 for r in model_ideal),
            "expected every model condition to be slower than the ideal reference")
    require(all(float(r["interaction_actions"]) > 0 for r in ideal_runs),
            "ideal reference has a zero interaction count")
    require(len({(r['model'], r['form_id'], r['question_id']) for r in fields}) == len(fields),
            "field grain is not unique at model/form/question")

    for model in MODELS:
        model_fields = [r for r in fields if r["model"] == model]
        model_trials = [r for r in trials if r["model"] == model]
        require(len(model_fields) == 409, f"{model}: expected 409 fields")
        require(len(model_trials) == 50, f"{model}: expected 50 trials")
        require(len({r["form_id"] for r in model_trials}) == 50, f"{model}: forms are not unique")

        misses = [r for r in model_fields if r["verified_correct"] != "True"]
        matrix_row = next(r for r in matrix if r["model"] == model)
        require(int(matrix_row["total_missed_fields"]) == len(misses), f"{model}: miss total mismatch")
        subtype_counts = Counter(r["failure_subtype"] for r in misses)
        for subtype, count in subtype_counts.items():
            require(int(matrix_row[f"{subtype}_count"]) == count,
                    f"{model}: {subtype} count mismatch")

        incomplete = sum(r["full_fill_success"] != "true" for r in model_trials)
        stop_incomplete = sum(int(r["incomplete_trials"]) for r in stops if r["model"] == model)
        require(incomplete == stop_incomplete, f"{model}: incomplete stop-reason reconciliation failed")

        model_ranks = sorted(int(r["difficulty_rank_within_model"]) for r in ranks if r["model"] == model)
        require(model_ranks == list(range(1, 51)), f"{model}: form ranks are not 1..50")

        mix_total = sum(int(r["action_count"]) for r in action_mix if r["model"] == model)
        trial_action_total = sum(int(r["interaction_action_count"]) for r in model_trials)
        require(mix_total == trial_action_total, f"{model}: action mix does not reconcile to interaction actions")

        for outcome in ("full fill", "incomplete"):
            eff = next(r for r in action_efficiency if r["model"] == model and r["outcome"] == outcome)
            selected = [r for r in model_trials if ("full fill" if r["full_fill_success"] == "true" else "incomplete") == outcome]
            require(int(eff["trials"]) == len(selected), f"{model}/{outcome}: trial count mismatch")
            require(int(eff["total_actions"]) == sum(int(r["interaction_action_count"]) for r in selected),
                    f"{model}/{outcome}: action total mismatch")

    dropdowns = [r for r in fields if r["widget_type"] == "dropdown"]
    require(len(dropdowns) == 100, "expected 100 dropdown observations")
    require(not any(r["verified_correct"] == "True" for r in dropdowns),
            "dropdown caveat changed: at least one dropdown is now marked correct")
    require(len(dropdown_audit) == 4 and sum(int(r["dropdown_targets"]) for r in dropdown_audit) == 100,
            "dropdown audit does not cover 100 observations")
    require(sum(int(r["expected_embedded_in_container"]) for r in dropdown_audit) == 89,
            "expected 89 indeterminate option-container readings")
    require(sum(int(r["blank_or_not_attempted"]) for r in dropdown_audit) == 11,
            "expected 11 blank or not-attempted dropdowns")
    require(len(dropdown_selected) == 100, "selected-state audit must cover 100 dropdowns")
    require(len({(r["model"], r["form_id"], r["question_id"]) for r in dropdown_selected}) == 100,
            "selected-state dropdown audit grain is not unique")
    selected_status = Counter(r["audit_status"] for r in dropdown_selected)
    require(selected_status == {"confirmed_correct": 79, "unresolved_excerpt_gap": 21},
            f"unexpected selected-state audit counts: {dict(selected_status)}")
    require(all((ROOT.parents[2] / r["evidence_path"]).exists() for r in dropdown_selected),
            "selected-state audit contains a missing evidence path")
    expected_selected = {
        "Gemini 3.5 Flash": (25, 0, 6, 0),
        "OpenCUA direct-MCP": (20, 5, 13, 2),
        "Qwen3 Text": (13, 12, 1, 2),
        "Qwen3-VL": (21, 4, 11, 1),
    }
    require(len(dropdown_selected_summary) == 4, "expected four selected-state model rows")
    for row in dropdown_selected_summary:
        observed = (
            int(row["artifact_confirmed_correct"]),
            int(row["artifact_unresolved"]),
            int(row["artifact_confirmed_full_fills"]),
            int(row["additional_full_fills_if_all_unresolved_correct"]),
        )
        require(observed == expected_selected[row["model"]],
                f"{row['model']}: selected-state audit summary changed")
    require(len(without_dropdown) == 4 and len(with_dropdown) == 4,
            "expected four model rows in each dropdown form table")
    require(all(int(r["forms"]) == 25 for r in without_dropdown + with_dropdown),
            "dropdown form split is not 25/25")
    require(all(int(r["dropdown_targets_unresolved"]) == 0 for r in without_dropdown),
            "no-dropdown table contains unresolved dropdowns")
    require(all(int(r["dropdown_targets_unresolved"]) == 25 for r in with_dropdown),
            "with-dropdown table does not contain 25 unresolved targets per model")

    require(len(canonical_submission) == 4, "expected four canonical submission-audit rows")
    require(sum(int(r["submit_attempts"]) for r in canonical_submission) == 0,
            "canonical fill-only cohort unexpectedly attempted submission")
    require(sum(int(r["successful_submissions"]) for r in canonical_submission) == 0,
            "canonical fill-only cohort unexpectedly submitted")
    require(sum(int(r["submitted_trials"]) for r in submission_audit) == 519,
            "historical submitted-trial count mismatch")
    require(sum(int(r["final_page_zero_trials"]) for r in submission_audit) == 519,
            "historical final-page zero count mismatch")
    require(sum(int(r["recovered_nonzero_trials"]) for r in submission_audit) == 512,
            "historical recovered nonzero count mismatch")
    require(sum(int(r["recovered_perfect_trials"]) for r in submission_audit) == 153,
            "historical recovered perfect count mismatch")
    require(sum(int(r["unrecovered_zero_trials"]) for r in submission_audit) == 7,
            "historical unrecovered zero count mismatch")
    require(recovered_perfect_fields and all(r["recovered_verified_correct"] == "True" for r in recovered_perfect_fields),
            "recovered perfect field file contains a non-correct row")

    expected_correction = {
        "Gemini 3.5 Flash": (1382, 1372, 0, 10),
        "OpenCUA direct-MCP": (684, 683, 0, 1),
        "Qwen3 Text": (577, 573, 4, 0),
        "Qwen3-VL": (714, 591, 122, 1),
    }
    for row in action_correction:
        observed = (int(row["model_issued_actions_raw"]), int(row["interaction_actions"]),
                    int(row["excluded_observation_actions"]), int(row["excluded_other_noninteraction_actions"]))
        require(observed == expected_correction[row["model"]],
                f"{row['model']}: raw-to-interaction reconciliation changed")

    artifact = json.loads((ROOT / "artifact.json").read_text(encoding="utf-8"))
    datasets = artifact["snapshot"]["datasets"]
    table_ids = {table["id"] for table in artifact["manifest"]["tables"]}
    chart_ids = {chart["id"] for chart in artifact["manifest"]["charts"]}
    source_ids = {source["id"] for source in artifact["sources"]}
    for table in artifact["manifest"]["tables"]:
        require(table["dataset"] in datasets, f"table {table['id']} has no dataset")
        require(table["sourceId"] in source_ids, f"table {table['id']} has no source")
    for chart in artifact["manifest"]["charts"]:
        require(chart["dataset"] in datasets, f"chart {chart['id']} has no dataset")
        require(chart["sourceId"] in source_ids, f"chart {chart['id']} has no source")
    for block in artifact["manifest"]["blocks"]:
        if block["type"] == "table":
            require(block["tableId"] in table_ids, f"block {block['id']} has no table")
        if block["type"] == "chart":
            require(block["chartId"] in chart_ids, f"block {block['id']} has no chart")
        if block.get("sourceId"):
            require(block["sourceId"] in source_ids, f"block {block['id']} has no source")

    print(json.dumps({
        "status": "passed",
        "canonical_trials": len(trials),
        "field_outcomes": len(fields),
        "baseline_trials": len(baseline),
        "cross_model_question_items": len(items),
        "consensus_hard_questions": 14,
        "ideal_reference_runs": len(ideal_runs),
        "model_ideal_matches": len(model_ideal_forms),
        "historical_submissions_recovered_nonzero": 512,
        "historical_submissions_recovered_perfect": 153,
        "dropdown_artifact_confirmed_correct": 79,
        "dropdown_artifact_unresolved": 21,
        "models": len(MODELS),
        "report_tables": len(table_ids),
        "report_charts": len(chart_ids),
    }, indent=2))


if __name__ == "__main__":
    main()
