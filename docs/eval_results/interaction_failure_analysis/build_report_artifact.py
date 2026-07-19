#!/usr/bin/env python3
"""Create the canonical portable-report artifact for interaction failures."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_ORDER = ["Gemini 3.5 Flash", "OpenCUA direct-MCP", "Qwen3 Text", "Qwen3-VL"]


def rows(name):
    with (DATA / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(value):
    return float(value) if value not in (None, "") else None


def inum(value):
    return int(value) if value not in (None, "") else None


field_rows = rows("field_outcomes.csv")
widget_rows = rows("widget_difficulty_by_model.csv")
hard_rows = rows("hardest_forms_cross_model.csv")
gemini_action = rows("gemini_action_productivity.csv")
position_rows = rows("position_difficulty_by_model.csv")
baseline_behavior = rows("baseline_behavior_by_submission.csv")
baseline_mix = rows("baseline_action_mix_by_submission.csv")
failure_matrix_rows = rows("failure_matrix_by_model.csv")
stop_reason_rows = rows("trial_stop_reasons_by_model.csv")
hardest_model_rows = rows("hardest_forms_by_model.csv")
question_label_rows = rows("question_label_difficulty_by_model.csv")
question_item_rows = rows("question_difficulty_cross_model.csv")
action_efficiency_rows = rows("action_efficiency_by_model_outcome.csv")
action_mix_rows = rows("action_mix_by_model.csv")
methodology_gap_rows = rows("methodology_gap_register.csv")
ideal_reference_rows = rows("ideal_reference_summary.csv")
model_ideal_rows = rows("model_vs_ideal_summary.csv")
action_correction_rows = rows("action_count_correction_by_model.csv")
dropdown_audit_rows = rows("dropdown_verifier_audit_by_model.csv")
forms_without_dropdown_rows = rows("performance_forms_without_dropdown.csv")
forms_with_dropdown_rows = rows("performance_forms_with_dropdown.csv")
submission_audit_rows = rows("baseline_submission_scoring_audit.csv")
canonical_submission_rows = rows("canonical_submission_audit.csv")

adjusted = []
for model in MODEL_ORDER:
    model_rows = [r for r in field_rows if r["model"] == model]
    nondrop = [r for r in model_rows if r["widget_type"] != "dropdown"]
    correct = sum(r["verified_correct"] == "True" for r in nondrop)
    full = len({r["form_id"] for r in model_rows if r["full_fill_success"] == "true"})
    adjusted.append({
        "model": model,
        "observed_correct": sum(r["verified_correct"] == "True" for r in model_rows),
        "observed_targets": len(model_rows),
        "observed_correctness_pct": round(100 * sum(r["verified_correct"] == "True" for r in model_rows) / len(model_rows), 2),
        "nondrop_correct": correct,
        "nondrop_targets": len(nondrop),
        "nondrop_correctness_pct": round(100 * correct / len(nondrop), 2),
        "full_fills": full,
        "scoreable_nondrop_forms": 25,
        "full_fill_rate_nondrop_pct": round(100 * full / 25, 2),
    })

widget_overall_groups = defaultdict(list)
for r in field_rows:
    if r["widget_type"] != "dropdown":
        widget_overall_groups[r["widget_type"]].append(r)
widget_overall = []
for widget, items in widget_overall_groups.items():
    misses = sum(r["verified_correct"] != "True" for r in items)
    widget_overall.append({"widget_type": widget, "target_fields": len(items), "missed_fields": misses,
                           "failure_rate_pct": round(100 * misses / len(items), 2)})
widget_overall.sort(key=lambda r: -r["failure_rate_pct"])

widget_model_table = []
for r in widget_rows:
    if r["widget_type"] == "dropdown":
        continue
    widget_model_table.append({
        "model": r["model"], "widget_type": r["widget_type"],
        "target_fields": inum(r["target_fields"]), "missed_fields": inum(r["missed_fields"]),
        "failure_rate_pct": fnum(r["failure_rate_pct"]),
    })

composition = defaultdict(Counter)
for r in field_rows:
    if r["model"] == "Gemini 3.5 Flash":
        composition[r["form_id"]][r["widget_type"]] += 1
hardest = []
for r in hard_rows[:12]:
    c = composition[r["form_id"]]
    complex_widgets = sum(c[w] for w in ("dropdown", "time", "multi_choice", "date"))
    hardest.append({
        "form_id": r["form_id"],
        "fields": inum(r["fields_per_model"]),
        "complex_widgets": complex_widgets,
        "widget_mix": ", ".join(f"{k}:{v}" for k, v in sorted(c.items())),
        "models_full": inum(r["models_full"]),
        "cross_model_correctness_pct": fnum(r["correctness_pct_all_models"]),
    })

gemini_productivity = [{
    "action_type": r["action_type"], "actions": inum(r["actions"]),
    "actions_with_new_correct_field": inum(r["actions_with_new_correct_field"]),
    "no_field_progress_actions": inum(r["no_field_progress_actions"]),
    "field_progress_rate_pct": fnum(r["field_progress_rate_pct"]),
} for r in gemini_action]

gemini_position = [{
    "position": r["position_bucket"], "target_fields": inum(r["target_fields"]),
    "missed_fields": inum(r["missed_fields"]), "failure_rate_pct": fnum(r["failure_rate_pct"]),
} for r in position_rows if r["model"] == "Gemini 3.5 Flash"]
position_order = {"first third": 1, "middle third": 2, "final third": 3}
gemini_position.sort(key=lambda r: position_order[r["position"]])

baseline_table = []
for r in baseline_behavior:
    baseline_table.append({
        "model": r["model"], "outcome": r["outcome"], "trials": inum(r["trials"]),
        "median_actions": fnum(r["median_actions"]), "median_click_share_pct": fnum(r["median_click_share_pct"]),
        "median_type_share_pct": fnum(r["median_type_share_pct"]),
        "perfect_pre_submit_trials": inum(r.get("perfect_pre_submit_trials")),
        "postsubmit_zero_artifacts": inum(r.get("postsubmit_zero_artifacts")),
        "median_final_page_accuracy_pct": fnum(r.get("median_final_page_accuracy_pct")),
        "median_scored_accuracy_pct": fnum(r.get("median_scored_accuracy_pct")),
    })

failure_matrix = [{
    "model": r["model"],
    "total_missed_fields": inum(r["total_missed_fields"]),
    "dominant_behavioral_failure": r["dominant_behavioral_failure"],
    "dominant_behavioral_failure_count": inum(r["dominant_behavioral_failure_count"]),
    "dominant_behavioral_share_of_misses_pct": fnum(r["dominant_behavioral_share_of_misses_pct"]),
    "not_attempted_count": inum(r.get("not_attempted_count")),
    "attempted_but_blank_count": inum(r.get("attempted_but_blank_count")),
    "wrong_value_count": inum(r.get("wrong_value_count")),
    "multi_choice_error_count": sum(inum(r.get(k)) or 0 for k in (
        "partial_multi_choice_count", "wrong_multi_choice_count", "extra_multi_choice_count"
    )),
    "ambiguous_option_text_count": inum(r.get("option_container_text_count")),
} for r in failure_matrix_rows]

stop_reason_table = []
for r in stop_reason_rows:
    if r["stop_reason"] == "filled_without_submit":
        continue
    stop_reason_table.append({
        "model": r["model"],
        "stop_reason": r["stop_reason"],
        "trials": inum(r["trials"]),
        "share_of_model_trials_pct": fnum(r["share_of_model_trials_pct"]),
        "incomplete_trials": inum(r["incomplete_trials"]),
        "share_of_model_incomplete_trials_pct": fnum(r["share_of_model_incomplete_trials_pct"]),
    })

hardest_by_model = [{
    "model": r["model"],
    "rank": inum(r["difficulty_rank_within_model"]),
    "form_id": r["form_id"],
    "target_fields": inum(r["target_fields"]),
    "missed_fields": inum(r["missed_fields"]),
    "correctness_pct": fnum(r["correctness_pct"]),
} for r in hardest_model_rows if inum(r["difficulty_rank_within_model"]) <= 5]

question_labels = []
label_counts = Counter()
for r in question_label_rows:
    if r["widget_type"] == "dropdown" or inum(r["missed_fields"]) == 0 or label_counts[r["model"]] >= 5:
        continue
    question_labels.append({
        "model": r["model"],
        "label": r["label"],
        "widget_type": r["widget_type"],
        "observations": inum(r["observations"]),
        "missed_fields": inum(r["missed_fields"]),
        "failure_rate_pct": fnum(r["failure_rate_pct"]),
    })
    label_counts[r["model"]] += 1

consensus_questions = [{
    "form_id": r["form_id"],
    "question_index": inum(r["question_index"]),
    "label": r["label"],
    "widget_type": r["widget_type"],
    "position_bucket": r["position_bucket"],
    "models_failed": inum(r["models_failed"]),
    "models_not_attempted": inum(r["models_not_attempted"]),
    "models_attempted_but_failed": inum(r["models_attempted_but_failed"]),
} for r in question_item_rows
  if r["ambiguous_dropdown_measurement"] == "False" and inum(r["models_failed"]) == 4]

action_efficiency = [{
    "model": r["model"],
    "outcome": r["outcome"],
    "trials": inum(r["trials"]),
    "total_actions": inum(r["total_actions"]),
    "median_actions": fnum(r["median_actions"]),
    "actions_per_verified_field": fnum(r["actions_per_verified_field"]),
    "median_correctness_pct": fnum(r["median_correctness_pct"]),
    "click_share_pct": fnum(r["click_share_pct"]),
    "type_share_pct": fnum(r["type_share_pct"]),
    "scroll_share_pct": fnum(r["scroll_share_pct"]),
    "other_share_pct": fnum(r["other_share_pct"]),
    "adjacent_identical_rate_pct": fnum(r["adjacent_identical_rate_pct"]),
} for r in action_efficiency_rows]

action_mix = [{
    "model": r["model"],
    "action_type": r["action_type"],
    "action_count": inum(r["action_count"]),
    "share_of_actions_pct": fnum(r["share_of_actions_pct"]),
} for r in action_mix_rows]

methodology_gaps = [{
    "priority": r["priority"],
    "gap": r["gap"],
    "why_it_matters": r["why_it_matters"],
    "concrete_fix": r["concrete_fix"],
    "paper_treatment": r["paper_treatment"],
} for r in methodology_gap_rows if r["priority"] in {"P0", "P1"}]

ideal_reference = [{
    "reference_runs": inum(r["reference_runs"]),
    "usable_runs": inum(r["usable_runs"]),
    "successful_submissions": inum(r["successful_submissions"]),
    "forms": inum(r["forms"]),
    "answer_runs_per_form": inum(r["answer_runs_per_form"]),
    "median_total_trace_events": fnum(r["median_total_trace_events"]),
    "median_interaction_actions": fnum(r["median_interaction_actions"]),
    "median_script_operations": fnum(r["median_script_operations"]),
    "median_screenshot_events": fnum(r["median_screenshot_events"]),
    "median_wait_events": fnum(r["median_wait_events"]),
    "median_duration_s": fnum(r["median_duration_s"]),
} for r in ideal_reference_rows]

model_vs_ideal = [{
    "model": r["model"],
    "matched_forms": inum(r["matched_forms"]),
    "full_fills": inum(r["full_fills"]),
    "median_model_actions": fnum(r["median_model_actions"]),
    "median_ideal_interaction_actions": fnum(r["median_ideal_interaction_actions"]),
    "median_ideal_total_trace_events": fnum(r["median_ideal_total_trace_events"]),
    "median_ideal_script_operations": fnum(r["median_ideal_script_operations"]),
    "median_model_duration_s": fnum(r["median_model_duration_s"]),
    "median_ideal_duration_s": fnum(r["median_ideal_duration_s"]),
    "median_model_to_ideal_interaction_action_ratio": fnum(r["median_model_to_ideal_interaction_action_ratio"]),
    "median_model_to_ideal_total_event_ratio": fnum(r["median_model_to_ideal_total_event_ratio"]),
    "median_model_to_ideal_script_operation_ratio": fnum(r["median_model_to_ideal_script_operation_ratio"]),
    "median_model_to_ideal_time_ratio": fnum(r["median_model_to_ideal_time_ratio"]),
} for r in model_ideal_rows]

action_correction = [{
    "model": r["model"],
    "raw_actions": inum(r["model_issued_actions_raw"]),
    "interaction_actions": inum(r["interaction_actions"]),
    "excluded_observations": inum(r["excluded_observation_actions"]),
    "excluded_other": inum(r["excluded_other_noninteraction_actions"]),
    "excluded_share_pct": fnum(r["excluded_share_of_raw_pct"]),
    "median_raw": fnum(r["median_raw_actions_per_trial"]),
    "median_interactions": fnum(r["median_interaction_actions_per_trial"]),
} for r in action_correction_rows]

def form_group_table(source_rows):
    return [{
        "model": r["model"],
        "forms": inum(r["forms"]),
        "raw_field_correctness_pct": fnum(r["raw_field_correctness_pct"]),
        "non_dropdown_field_correctness_pct": fnum(r["non_dropdown_field_correctness_pct"]),
        "dropdown_targets_unresolved": inum(r["dropdown_targets_unresolved"]),
        "raw_full_fills": inum(r["raw_full_fills"]),
        "raw_full_fill_rate_pct": fnum(r["raw_full_fill_rate_pct"]),
        "non_dropdown_complete_forms": inum(r["non_dropdown_complete_forms"]),
        "non_dropdown_complete_rate_pct": fnum(r["non_dropdown_complete_rate_pct"]),
    } for r in source_rows]

forms_without_dropdown = form_group_table(forms_without_dropdown_rows)
forms_with_dropdown = form_group_table(forms_with_dropdown_rows)
dropdown_audit = [{
    "model": r["model"],
    "dropdown_targets": inum(r["dropdown_targets"]),
    "raw_verified_correct": inum(r["raw_verified_correct"]),
    "expected_embedded_in_container": inum(r["expected_embedded_in_container"]),
    "blank_or_not_attempted": inum(r["blank_or_not_attempted"]),
    "measurement_indeterminate_pct": fnum(r["measurement_indeterminate_pct"]),
} for r in dropdown_audit_rows]
submission_audit = [{
    "model": r["model"],
    "submitted_trials": inum(r["submitted_trials"]),
    "final_page_zero_trials": inum(r["final_page_zero_trials"]),
    "recovered_nonzero_trials": inum(r["recovered_nonzero_trials"]),
    "recovered_perfect_trials": inum(r["recovered_perfect_trials"]),
    "recovered_partial_trials": inum(r["recovered_partial_trials"]),
    "unrecovered_zero_trials": inum(r["unrecovered_zero_trials"]),
    "median_final_page_accuracy_pct": fnum(r["median_final_page_accuracy_pct"]),
    "median_recovered_pre_submit_accuracy_pct": fnum(r["median_recovered_pre_submit_accuracy_pct"]),
} for r in submission_audit_rows]
canonical_submission = [{
    "model": r["model"], "trials": inum(r["trials"]),
    "task_mode": r["task_modes"], "submit_attempts": inum(r["submit_attempts"]),
    "successful_submissions": inum(r["successful_submissions"]),
} for r in canonical_submission_rows]

baseline_action_lookup = {(r["model"], r["outcome"], r["action_type"]): inum(r["action_count"]) for r in baseline_mix}
baseline_highlights = [
    {"finding": "OpenCUA Native scroll actions", "failed_or_nonsubmitted": baseline_action_lookup.get(("OpenCUA Native", "not submitted", "scroll"), 0), "comparison": "200 non-submitted trials"},
    {"finding": "Qwen Text click share", "failed_or_nonsubmitted": 76.67, "comparison": "57.11% in submitted trials"},
    {"finding": "Qwen VLM click share", "failed_or_nonsubmitted": 80.80, "comparison": "67.10% in submitted trials"},
]

source = {
    "id": "interaction_analysis",
    "query": {
        "engine": "local_files",
        "language": "python",
        "description": "Canonical 50-form question/action decomposition plus primary baseline failure-only decomposition.",
        "tables_used": [
            "data/model_baseline_exports/fill_only_done_50_20260714/trials.csv",
            "docs/eval_results/analysis/model_action_trial_counts.csv",
            "data/model_baselines/**/annotations.json",
            "data/model_baselines/**/model_io.jsonl",
            "interaction_failure_analysis/data/baseline_submission_scoring_audit.csv",
            "interaction_failure_analysis/data/dropdown_verifier_audit_by_model.csv",
            "interaction_failure_analysis/data/action_count_correction_by_model.csv",
            "interaction_failure_analysis/data/performance_forms_without_dropdown.csv",
            "interaction_failure_analysis/data/performance_forms_with_dropdown.csv",
        ],
        "filters": [
            "Fill-only comparison: four models, 50 usable run_0002 trials each",
            "Primary baseline cohort: 978 trials from the existing action-trial index",
            "Baseline failure cuts: 459 non-submitted trials; aggregate-perfect submitted fields recovered separately",
            "Primary action count excludes screenshots, snapshots, waits, DONE, setup, and close",
        ],
        "metric_definitions": {
            "field_correctness": "final verified_correct target fields / target fields",
            "field_progress_action": "Gemini action after which at least one additional target field is verified correct",
            "failure_subtype": "classification from attempted flag and final actual versus expected value",
            "nondrop_correctness": "field correctness after excluding dropdown targets affected by verifier ambiguity",
            "interaction_action": "navigation, click, type/fill, scroll, select/check, keypress, hover, drag, upload, or scripted fill/submit",
            "submitted_score": "pre-successful-submit correctness, then pre-first-submit correctness, before final-page fallbacks",
        },
        "executed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    },
}


def sql_source(source_id, description, sql, tables_used):
    return {
        "id": source_id,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": sql,
            "description": description,
            "tables_used": tables_used,
            "filters": ["Canonical usable artifacts only", "Model-issued actions only"],
            "executed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    }


widget_source = sql_source(
    "widget_failures",
    "Aggregates correct and missed fields by widget type, excluding ambiguous dropdown results.",
    "SELECT widget_type, COUNT(*) AS target_fields, SUM(CASE WHEN verified_correct = 'False' THEN 1 ELSE 0 END) AS missed_fields, 100.0 * missed_fields / target_fields AS failure_rate_pct FROM read_csv_auto('interaction_failure_analysis/data/field_outcomes.csv') WHERE widget_type <> 'dropdown' GROUP BY widget_type",
    ["interaction_failure_analysis/data/field_outcomes.csv"],
)
gemini_source = sql_source(
    "gemini_actions",
    "Aggregates Gemini actions by action type and immediate verified-field progress.",
    "SELECT action_type, COUNT(*) AS actions, SUM(CASE WHEN field_progress = 'True' THEN 1 ELSE 0 END) AS actions_with_new_correct_field, 100.0 * actions_with_new_correct_field / actions AS field_progress_rate_pct FROM read_csv_auto('interaction_failure_analysis/data/gemini_action_progress.csv') GROUP BY action_type",
    ["interaction_failure_analysis/data/gemini_action_progress.csv"],
)
adjusted_source = sql_source(
    "adjusted_outcomes",
    "Calculates observed and non-dropdown field correctness by model.",
    "SELECT model, SUM(CASE WHEN verified_correct = 'True' THEN 1 ELSE 0 END) AS observed_correct, COUNT(*) AS observed_targets, SUM(CASE WHEN widget_type <> 'dropdown' AND verified_correct = 'True' THEN 1 ELSE 0 END) AS nondrop_correct, SUM(CASE WHEN widget_type <> 'dropdown' THEN 1 ELSE 0 END) AS nondrop_targets FROM read_csv_auto('interaction_failure_analysis/data/field_outcomes.csv') GROUP BY model",
    ["interaction_failure_analysis/data/field_outcomes.csv"],
)
hard_source = sql_source(
    "hard_forms",
    "Ranks forms by field correctness pooled across the four fill-only conditions.",
    "SELECT form_id, COUNT(*) AS target_fields_all_models, SUM(CASE WHEN verified_correct = 'True' THEN 1 ELSE 0 END) AS correct_fields_all_models, 100.0 * correct_fields_all_models / target_fields_all_models AS correctness_pct_all_models FROM read_csv_auto('interaction_failure_analysis/data/field_outcomes.csv') GROUP BY form_id ORDER BY correctness_pct_all_models ASC",
    ["interaction_failure_analysis/data/field_outcomes.csv"],
)
position_source = sql_source(
    "gemini_position",
    "Aggregates Gemini misses by within-form field-position third.",
    "SELECT position_bucket, COUNT(*) AS target_fields, SUM(CASE WHEN verified_correct = 'False' THEN 1 ELSE 0 END) AS missed_fields, 100.0 * missed_fields / target_fields AS failure_rate_pct FROM read_csv_auto('interaction_failure_analysis/data/field_outcomes.csv') WHERE model = 'Gemini 3.5 Flash' GROUP BY position_bucket",
    ["interaction_failure_analysis/data/field_outcomes.csv"],
)
baseline_source = sql_source(
    "baseline_behavior",
    "Summarizes action behavior by baseline model/interface and submission outcome.",
    "SELECT model, CASE WHEN submit_success THEN 'submitted' ELSE 'not submitted' END AS outcome, COUNT(*) AS trials, MEDIAN(executable_actions) AS median_actions, MEDIAN(100.0 * click_actions / executable_actions) AS median_click_share_pct, MEDIAN(100.0 * type_actions / executable_actions) AS median_type_share_pct FROM read_csv_auto('interaction_failure_analysis/data/baseline_primary_trial_action_profile.csv') GROUP BY model, outcome",
    ["interaction_failure_analysis/data/baseline_primary_trial_action_profile.csv"],
)
failure_source = sql_source(
    "failure_matrix",
    "Classifies final field misses and trial stop reasons by model in the canonical 50-form comparison.",
    "SELECT model, failure_subtype, COUNT(*) AS missed_fields FROM read_csv_auto('interaction_failure_analysis/data/field_outcomes.csv') WHERE verified_correct = 'False' GROUP BY model, failure_subtype",
    ["interaction_failure_analysis/data/field_outcomes.csv", "interaction_failure_analysis/data/trial_action_profile.csv"],
)
stop_source = sql_source(
    "trial_stop_reasons",
    "Counts incomplete trial stop reasons by model in the canonical 50-form comparison.",
    "SELECT model, stop_reason, COUNT(*) AS incomplete_trials FROM read_csv_auto('interaction_failure_analysis/data/trial_action_profile.csv') WHERE full_fill_success <> 'true' GROUP BY model, stop_reason",
    ["interaction_failure_analysis/data/trial_action_profile.csv"],
)
model_forms_source = sql_source(
    "model_hard_forms",
    "Ranks forms and repeated question labels separately within each model.",
    "SELECT model, form_id, COUNT(*) AS target_fields, SUM(CASE WHEN verified_correct = 'True' THEN 1 ELSE 0 END) AS correct_fields FROM read_csv_auto('interaction_failure_analysis/data/field_outcomes.csv') GROUP BY model, form_id",
    ["interaction_failure_analysis/data/field_outcomes.csv"],
)
question_labels_source = sql_source(
    "question_labels",
    "Ranks recurring non-dropdown question labels by missed fields within each model.",
    "SELECT model, label, widget_type, COUNT(*) AS observations, SUM(CASE WHEN verified_correct = 'False' THEN 1 ELSE 0 END) AS missed_fields FROM read_csv_auto('interaction_failure_analysis/data/field_outcomes.csv') WHERE widget_type <> 'dropdown' GROUP BY model, label, widget_type HAVING COUNT(*) >= 2",
    ["interaction_failure_analysis/data/field_outcomes.csv"],
)
question_items_source = sql_source(
    "question_item_consensus",
    "Ranks individual non-dropdown questions by the number of model-interface conditions that failed them.",
    "SELECT form_id, question_id, label, widget_type, COUNT(*) AS model_observations, SUM(CASE WHEN verified_correct = 'False' THEN 1 ELSE 0 END) AS models_failed FROM read_csv_auto('interaction_failure_analysis/data/field_outcomes.csv') WHERE widget_type <> 'dropdown' GROUP BY form_id, question_id, label, widget_type ORDER BY models_failed DESC",
    ["interaction_failure_analysis/data/field_outcomes.csv"],
)
action_trace_source = sql_source(
    "canonical_action_behavior",
    "Compares model-issued action mix, efficiency, and adjacent exact repetition by model and trial outcome.",
    "SELECT model, CASE WHEN full_fill_success = 'true' THEN 'full fill' ELSE 'incomplete' END AS outcome, COUNT(*) AS trials, SUM(interaction_action_count) AS total_actions, MEDIAN(interaction_action_count) AS median_actions, 100.0 * SUM(adjacent_identical_actions) / SUM(GREATEST(interaction_action_count - 1, 0)) AS adjacent_identical_rate_pct FROM read_csv_auto('interaction_failure_analysis/data/trial_action_profile.csv') GROUP BY model, outcome",
    ["interaction_failure_analysis/data/trial_action_profile.csv", "interaction_failure_analysis/data/action_counts_long.csv"],
)
methodology_source = sql_source(
    "methodology_gaps",
    "Prioritized measurement and design gaps with concrete repair actions and paper-safe treatment.",
    "SELECT priority, gap, why_it_matters, concrete_fix, paper_treatment FROM read_csv_auto('interaction_failure_analysis/data/methodology_gap_register.csv') WHERE priority IN ('P0', 'P1')",
    ["interaction_failure_analysis/data/methodology_gap_register.csv"],
)
ideal_reference_source = sql_source(
    "ideal_reference",
    "Summarizes the 300 successful scripted Playwright reference runs across 50 forms and six answer sets.",
    "SELECT COUNT(*) AS reference_runs, SUM(CASE WHEN usable = 'True' THEN 1 ELSE 0 END) AS usable_runs, MEDIAN(interaction_actions) AS median_interaction_actions, MEDIAN(total_trace_events) AS median_total_trace_events, MEDIAN(duration_s) AS median_duration_s FROM read_csv_auto('interaction_failure_analysis/data/ideal_reference_runs.csv')",
    ["interaction_failure_analysis/data/ideal_reference_runs.csv"],
)
model_ideal_source = sql_source(
    "model_vs_ideal",
    "Pairs each canonical run-2 model trial with the scripted reference for the same form and answer run.",
    "SELECT model, COUNT(*) AS matched_forms, MEDIAN(model_actions) AS median_model_actions, MEDIAN(ideal_interaction_actions) AS median_ideal_interaction_actions, MEDIAN(model_to_ideal_interaction_action_ratio) AS median_action_ratio, MEDIAN(model_duration_s) AS median_model_duration_s, MEDIAN(ideal_duration_s) AS median_ideal_duration_s, MEDIAN(model_to_ideal_time_ratio) AS median_time_ratio FROM read_csv_auto('interaction_failure_analysis/data/model_vs_ideal_by_form.csv') GROUP BY model",
    ["interaction_failure_analysis/data/model_vs_ideal_by_form.csv"],
)
form_group_source = sql_source(
    "dropdown_form_groups",
    "Separates canonical performance for forms with and without dropdown controls.",
    "SELECT * FROM read_csv_auto(['interaction_failure_analysis/data/performance_forms_without_dropdown.csv', 'interaction_failure_analysis/data/performance_forms_with_dropdown.csv'])",
    ["interaction_failure_analysis/data/performance_forms_without_dropdown.csv", "interaction_failure_analysis/data/performance_forms_with_dropdown.csv"],
)
dropdown_audit_source = sql_source(
    "dropdown_verifier_audit",
    "Audits raw dropdown outcomes and option-container readings by model.",
    "SELECT * FROM read_csv_auto('interaction_failure_analysis/data/dropdown_verifier_audit_by_model.csv')",
    ["interaction_failure_analysis/data/dropdown_verifier_audit_by_model.csv", "interaction_failure_analysis/data/dropdown_verifier_examples.csv"],
)
action_correction_source = sql_source(
    "action_count_correction",
    "Reconciles raw model-issued calls to normalized state-changing interactions.",
    "SELECT * FROM read_csv_auto('interaction_failure_analysis/data/action_count_correction_by_model.csv')",
    ["interaction_failure_analysis/data/action_count_correction_by_model.csv"],
)
submission_audit_source = sql_source(
    "submission_scoring_audit",
    "Recovers historical submitted-trial correctness from pre-submit verification and audits the canonical no-submit contract.",
    "SELECT * FROM read_csv_auto('interaction_failure_analysis/data/baseline_submission_scoring_audit.csv')",
    ["interaction_failure_analysis/data/baseline_submission_scoring_audit.csv", "interaction_failure_analysis/data/canonical_submission_audit.csv", "interaction_failure_analysis/data/baseline_recovered_perfect_submitted_fields.csv"],
)

blocks = [
    {"id": "title", "type": "markdown", "body": "# Where Web-Form Models Fail"},
    {"id": "executive_summary", "type": "markdown", "sourceId": "interaction_analysis", "body": """## Executive Summary

- **Dropdown scoring materially suppresses exact completion.** All 100 dropdown targets were scored wrong, but 89 saved values contain the expected option inside the full option-list text. Those 89 outcomes are measurement-indeterminate; the remaining 11 are blank or not attempted.
- **The canonical comparison has no submission artifact.** All 200 trials used `fill_only_done`, made zero submit attempts, and were graded while the form was still visible.
- **Historical submitted scores are recoverable.** All 519 submitted trials ended with a final-page zero, but pre-submit logs recover 512 nonzero task scores and identify 153 verified-perfect submissions.
- **Primary action counts now exclude observation overhead.** Screenshots, snapshots, waits, `DONE`, setup, and close are omitted. This removes 123 of Qwen3-VL's 714 raw calls and lowers its incomplete-trial exact-repeat rate from the previous screenshot-contaminated estimate to 28.20%."""},
    {"id": "scope", "type": "markdown", "sourceId": "interaction_analysis", "body": """## What this analysis counts

The fill-only comparison contains **200 canonical trials** and **1,636 target-field outcomes**: 50 run-2 forms for each of Gemini, OpenCUA direct-MCP, Qwen Text, and Qwen VLM. A field is correct only when the saved final verifier marks the expected value correct. A full fill requires every field to be correct; submission is intentionally disabled.

The historical submission-enabled baseline contains **978 primary trials**. Failure cuts use the **459 non-submitted trials**; pre-submit scores additionally recover task correctness for submitted trials and every field for the 153 aggregate-perfect submissions. Primary action metrics count state-changing interactions only."""},
    {"id": "dropdown_risk", "type": "markdown", "sourceId": "interaction_analysis", "body": """## Dropdown scoring is a high-severity measurement risk

Every model is recorded at **0/25 dropdown fields**. In **89 of 100 observations**, the expected option is embedded in the recorded option-container text: 22 Gemini, 22 OpenCUA, 24 Qwen Text, and 21 Qwen VLM. Because every option appears in that container whether selected or not, these cases cannot be safely changed to correct; they are measurement-indeterminate. The other 11 are blank or not attempted.

The results are therefore split below. The no-dropdown table supports exact full-fill claims. The with-dropdown table reports non-dropdown correctness and non-dropdown-complete forms while leaving 25 dropdown targets per model unresolved."""},
    {"id": "dropdown_audit_table_block", "type": "table", "tableId": "dropdown_audit_table"},
    {"id": "forms_without_dropdown_table_block", "type": "table", "tableId": "forms_without_dropdown_table"},
    {"id": "forms_with_dropdown_table_block", "type": "table", "tableId": "forms_with_dropdown_table"},
    {"id": "widget_story", "type": "markdown", "sourceId": "interaction_analysis", "body": """## Time controls are the hardest defensible shared task

After setting dropdowns aside, time fields have the highest aggregate failure rate (**53.45%**). The weakness is concentrated in Gemini and the two Qwen conditions: both Gemini and Qwen Text fail 20 of 29 time fields, while Qwen VLM fails 17. OpenCUA direct-MCP fails only five.

Gemini's second distinctive weakness is paragraph entry: **22 of 45** paragraph fields fail, mostly after an attempted interaction that leaves the field blank. By contrast, its short text and choice controls are comparatively strong. This suggests a retention/focus problem on long text, not a general inability to type."""},
    {"id": "widget_chart_block", "type": "chart", "chartId": "widget_failure_chart"},
    {"id": "widget_table_block", "type": "table", "tableId": "widget_model_table"},
    {"id": "failure_story", "type": "markdown", "sourceId": "failure_matrix", "body": """## The dominant field failure differs sharply by model

At field level, **Gemini is split between attempted-but-blank fields (44; 38.94% of its misses) and fields it never attempted (43; 38.05%)**. The other three conditions are dominated by fields never attempted: **94 for OpenCUA direct-MCP (78.99% of misses), 114 for Qwen3 Text (70.81%), and 97 for Qwen3-VL (76.98%)**. The separate option-container column is retained as a measurement warning, not treated as a behavioral failure.
"""},
    {"id": "failure_matrix_table_block", "type": "table", "tableId": "failure_matrix_table"},
    {"id": "trial_failure_story", "type": "markdown", "sourceId": "trial_stop_reasons", "body": """## Step exhaustion is Gemini's main trial failure; premature DONE dominates direct-MCP

Gemini usually exhausts its step budget (**28 of 38 incomplete trials**). The direct-MCP conditions more often stop voluntarily while still incomplete: **29 of 35 incomplete OpenCUA trials, 33 of 42 Qwen3 Text trials, and 29 of 39 Qwen3-VL trials**. That distinction matters for fixes: Gemini needs more efficient navigation and progress control, while the MCP conditions need stronger completion checks before DONE."""},
    {"id": "stop_reason_table_block", "type": "table", "tableId": "stop_reason_table"},
    {"id": "form_story", "type": "markdown", "sourceId": "interaction_analysis", "body": """## Difficulty concentrates in long, mixed-control forms

The three hardest shared forms are `club_event_planning`, `purchase_request`, and `club_application`, each at **27.5%-32.5% cross-model correctness** with no full fills. All have ten fields and combine free text with several structured controls. `internship_app` is longer still at 12 fields and reaches only 35.42%.

This is not just a model-ranking effect: the same forms are difficult across all four conditions. They are the best candidates for qualitative trace review and targeted regression tests because improvement there would exercise scrolling, mixed widgets, and state tracking together."""},
    {"id": "hard_forms_table_block", "type": "table", "tableId": "hard_forms_table"},
    {"id": "model_form_story", "type": "markdown", "sourceId": "model_hard_forms", "body": """## Each model has a distinct hardest-form profile

The pooled ranking identifies shared stress tests, but the within-model table is the better regression target. It lists the five lowest-correctness forms for each condition, preserving ties by a deterministic form-name order. The repeated-question table then shows labels that recur across forms and accumulate the most misses, excluding dropdowns because their selected value is not reliably observable.

Use these two tables together: the form ranking finds end-to-end scenarios worth replaying, while the question-label ranking identifies reusable field prompts and widgets for focused unit tests."""},
    {"id": "model_hard_forms_table_block", "type": "table", "tableId": "model_hard_forms_table"},
    {"id": "question_label_table_block", "type": "table", "tableId": "question_label_table"},
    {"id": "question_consensus_story", "type": "markdown", "sourceId": "question_item_consensus", "body": """## Fourteen non-dropdown questions fail across all four conditions

The strongest item-level evidence is cross-model consensus: **14 of 384 non-dropdown questions were missed by all four model-interface conditions**. These questions concentrate in five long, mixed-control forms and include time entry, long text, multiselect, short text, and single choice. The table reports the exact item and whether failures were unattempted or attempted but still wrong.

This is an observed shared-difficulty index with a denominator of four evaluated conditions and is used as a practical regression-test list."""},
    {"id": "consensus_question_table_block", "type": "table", "tableId": "consensus_question_table"},
    {"id": "action_comparison_story", "type": "markdown", "sourceId": "canonical_action_behavior", "body": """## Repetition separates incomplete direct-MCP trials from full fills

Exact adjacent interaction repetition is **0% in every full-fill direct-MCP cohort**, but rises to **20.19% for incomplete OpenCUA, 29.15% for incomplete Qwen3 Text, and 28.20% for incomplete Qwen3-VL**. Snapshot and wait calls are excluded before forming adjacent pairs.

Gemini is different: exact repetition is not higher in incomplete trials. Its inefficiency appears as navigation and action volume—**5.33 interactions per verified field when incomplete versus 2.93 in full fills**, with scroll share rising from 6.75% to 17.95%. These patterns support separate loop-control and navigation hypotheses rather than one universal failure mechanism."""},
    {"id": "action_repeat_chart_block", "type": "chart", "chartId": "action_repeat_chart"},
    {"id": "action_efficiency_table_block", "type": "table", "tableId": "action_efficiency_table"},
    {"id": "action_mix_table_block", "type": "table", "tableId": "action_mix_table"},
    {"id": "action_correction_story", "type": "markdown", "sourceId": "action_count_correction", "body": """## Observation calls are excluded from the corrected action count

The correction is small for Gemini, OpenCUA, and Qwen Text, but material for Qwen3-VL. Normalized totals are **1,372 Gemini interactions, 683 OpenCUA interactions, 573 Qwen Text interactions, and 591 Qwen3-VL interactions**. Qwen3-VL excludes 122 snapshots and one wait from 714 raw calls. The raw-to-normalized table preserves the full reconciliation."""},
    {"id": "action_correction_table_block", "type": "table", "tableId": "action_correction_table"},
    {"id": "gemini_action_story", "type": "markdown", "sourceId": "interaction_analysis", "body": """## Gemini spends most actions without immediate field completion

Gemini uses **926 clicks, 228 typing actions, and 218 scrolls**. Typing is productive: 193 of 228 type actions immediately produce a newly correct field. Clicks are much less direct: 103 of 926 do so. Scrolls never directly complete a field, but the concentration is telling—201 scrolls occur in the 38 incomplete forms, compared with 17 in the 12 full fills.

The failure also moves down the page. Gemini misses only **3.73%** of first-third fields, then 24.82% in the middle third and 54.48% in the final third. That pattern points to navigation and state persistence on long forms, not weak initial field discovery."""},
    {"id": "gemini_action_chart_block", "type": "chart", "chartId": "gemini_productivity_chart"},
    {"id": "gemini_position_table_block", "type": "table", "tableId": "gemini_position_table"},
    {"id": "baseline_story", "type": "markdown", "sourceId": "interaction_analysis", "body": """## Baseline failures reveal three different control problems

**OpenCUA Native loses the page after the visible top section.** It scores 89.93% of first-third fields in non-submitted trials, 23.76% in the middle third, and 1.12% in the final third. Only three scroll actions across 200 trials make the mechanism unusually clear.

**Failed Qwen runs over-click.** Clicks are 76.67% of Qwen Text interactions and 80.80% of Qwen VLM interactions in non-submitted trials, versus 57.11% and 67.10% in submitted trials. The likely failure mode is repeated control manipulation or navigation without enough value entry.

**OpenCUA MCP fails differently.** Its non-submitted action mix is 64.23% typing and 34.65% clicking, suggesting repeated or misdirected entry rather than the click-dominant Qwen pattern."""},
    {"id": "baseline_table_block", "type": "table", "tableId": "baseline_behavior_table"},
    {"id": "submission_correction_story", "type": "markdown", "sourceId": "submission_scoring_audit", "body": """## Pre-submit logs repair the historical confirmation-page zeros

The historical baseline has **519 successful submissions**, and every one has zero final-page correctness because the confirmation page contains no controls. Pre-submit fields recover nonzero task scores for **512 trials** and prove **153 trials perfect**: 77 OpenCUA MCP, 44 Qwen Text, and 32 Qwen VLM. The seven remaining zero trials lack a usable nonzero pre-submit score.

For partial submitted trials, the recovered count is valid but the logs do not consistently retain which individual fields were correct. The report therefore repairs task-level accuracy for all recoverable submissions and restores field-level correctness only for the 153 aggregate-perfect trials. The canonical fill-only cohort is separately confirmed at zero submit attempts."""},
    {"id": "submission_audit_table_block", "type": "table", "tableId": "submission_audit_table"},
    {"id": "canonical_submission_table_block", "type": "table", "tableId": "canonical_submission_table"},
    {"id": "ideal_reference_story", "type": "markdown", "sourceId": "ideal_reference", "body": """## The ideal reference is complete, fast, and fully successful

The scripted Playwright reference contains **300 usable and successful submissions**: 50 forms across six answer runs. Its median completion time is **56.44 seconds**. A median run records **41 raw trace events** but only **10 normalized interactions**: navigation plus scripted field entry and submission. Screenshots, waits, setup, and close are excluded.

This dataset is the correct ideal-state anchor for form coverage, successful final state, and elapsed time. Its total trace-event count is not a clean interaction denominator because most events are observation or synchronization overhead."""},
    {"id": "ideal_reference_table_block", "type": "table", "tableId": "ideal_reference_table"},
    {"id": "model_ideal_story", "type": "markdown", "sourceId": "model_vs_ideal", "body": """## Every model condition is much slower than the scripted ideal

On the same 50 run-2 forms, median completion time is **767.90 seconds for Gemini, 294.83 for OpenCUA direct-MCP, 202.17 for Qwen3 Text, and 216.38 for Qwen3-VL**, versus a matched ideal median of **55.76 seconds**. The median per-form time ratios are **11.78×, 5.21×, 3.62×, and 3.86×**, respectively.

Using the normalized definition, median per-form model/ideal action ratios are **2.72× for Gemini, 1.09× for OpenCUA, 1.00× for Qwen Text, and 0.90× for Qwen VLM**. A ratio below one does not imply better performance because incomplete trials can stop early and scripted or fill-form calls can cover different amounts of work. Completion, action count, and time must be read together."""},
    {"id": "ideal_time_chart_block", "type": "chart", "chartId": "ideal_time_ratio_chart"},
    {"id": "model_ideal_table_block", "type": "table", "tableId": "model_ideal_table"},
    {"id": "methodology_story", "type": "markdown", "sourceId": "methodology_gaps", "body": """## The remaining priority is selected-option verification

Post-submission task scores and observation-contaminated action counts are now repaired from existing logs. Dropdown correctness remains unresolved because container text does not prove selection. The next tier concerns per-action field attribution, action-call granularity, usable-trial replacement, and mixing submission-enabled historical evidence with the fill-only cohort.

The current repair plan is concrete: fix selected-option verification, instrument per-action field transitions, retain raw-to-normalized action reconciliation, and publish capability and operational reliability as separate result sets."""},
    {"id": "methodology_gap_table_block", "type": "table", "tableId": "methodology_gap_table"},
    {"id": "recommendations", "type": "markdown", "body": """## Recommended next steps

1. **Repair and validate dropdown verification before citing widget-level or full-fill results.** Read the selected option through `aria-selected`, the selected option node, or the collapsed trigger label; add a regression test where the expected value is both selected and unselected.
2. **Use the repaired pre-submit score hierarchy for every historical submitted trial.** Restore field-level rows only when the pre-submit aggregate proves all fields correct.
3. **Keep normalized interactions as the primary action count.** Preserve raw calls in an audit table and interpret actions with completion and time.
4. **Instrument every action with its target and before/after field state.** This makes actions without state change, corrections, regressions, and actions per newly correct field measurable for every model.
5. **Add progress-aware control policies.** Require a completion verification before DONE and cap exact repeated commands or scrolling without state change.
6. **Separate capability from operational reliability.** Preserve every provider, node, startup, context, and harness failure in an inclusion ledger even when a usable replacement is selected for capability scoring.
7. **Make the hardest mixed-control forms the regression suite.** Start with `club_event_planning`, `purchase_request`, `club_application`, `internship_app`, and `publication_submission`."""},
    {"id": "further_questions", "type": "markdown", "body": """## Further questions

- Can the 89 dropdown-indeterminate outcomes be reconstructed from saved screenshots or accessibility state without assuming that container text proves selection?
- Do Gemini's blank paragraph fields lose focus before typing, or does text disappear after scrolling?
- Which exact Qwen click targets repeat in failed trials, and are they dropdown triggers, options, or already-selected controls?
- Does a progress-aware retry policy improve hard forms without increasing actions on easy forms?"""},
    {"id": "caveats", "type": "markdown", "sourceId": "interaction_analysis", "body": """## Caveats and Assumptions

This is descriptive attribution, not a causal experiment. Widget type, form length, and field position are correlated, so unadjusted cuts should not be interpreted as isolated causal effects. Normalized interaction calls still differ in granularity; interpret them with completion and time. The canonical cohort is conditional on usable artifacts because infrastructure failures were replaced or excluded. Gemini “field-progress actions” count only immediate additions to verified correctness. Historical partial submitted trials are repaired at task level, while field-level restoration is limited to aggregate-perfect pre-submit scores."""},
]

artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "Where Web-Form Models Fail",
        "description": "Task- and action-level failure analysis for primary baselines and the 50-form Gemini comparison.",
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cards": [],
        "charts": [
            {"id": "widget_failure_chart", "title": "Failure rate by widget type", "subtitle": "All four fill-only conditions combined; dropdown omitted because selection verification is ambiguous.", "type": "bar", "dataset": "widget_overall", "sourceId": "widget_failures", "valueFormat": "number", "encodings": {"x": {"field": "widget_type", "type": "nominal", "label": "Widget type"}, "y": {"field": "failure_rate_pct", "type": "quantitative", "label": "Failure rate (%)"}, "tooltip": [{"field": "target_fields", "type": "quantitative", "label": "Target fields"}, {"field": "missed_fields", "type": "quantitative", "label": "Missed fields"}]}},
            {"id": "gemini_productivity_chart", "title": "Gemini field progress by action type", "subtitle": "Share of actions immediately followed by at least one newly verified field.", "type": "bar", "dataset": "gemini_productivity", "sourceId": "gemini_actions", "valueFormat": "number", "encodings": {"x": {"field": "action_type", "type": "nominal", "label": "Action type"}, "y": {"field": "field_progress_rate_pct", "type": "quantitative", "label": "Actions with field progress (%)"}, "tooltip": [{"field": "actions", "type": "quantitative", "label": "Actions"}, {"field": "no_field_progress_actions", "type": "quantitative", "label": "No immediate field progress"}]}},
            {"id": "action_repeat_chart", "title": "Exact adjacent interaction repetition by outcome", "subtitle": "State-changing interactions only in the canonical 50-form comparison; full fills versus incomplete trials.", "type": "bar", "dataset": "action_efficiency", "sourceId": "canonical_action_behavior", "valueFormat": "number", "encodings": {"x": {"field": "model", "type": "nominal", "label": "Model condition"}, "y": {"field": "adjacent_identical_rate_pct", "type": "quantitative", "label": "Exact adjacent repetition (%)"}, "color": {"field": "outcome", "type": "nominal", "label": "Trial outcome"}, "tooltip": [{"field": "trials", "type": "quantitative", "label": "Trials"}, {"field": "median_actions", "type": "quantitative", "label": "Median interactions"}, {"field": "actions_per_verified_field", "type": "quantitative", "label": "Interactions per verified field"}]}},
            {"id": "ideal_time_ratio_chart", "title": "Model completion time relative to scripted ideal", "subtitle": "Median of matched per-form time ratios across the 50 canonical run-2 forms.", "type": "bar", "dataset": "model_vs_ideal", "sourceId": "model_vs_ideal", "valueFormat": "number", "encodings": {"x": {"field": "model", "type": "nominal", "label": "Model condition"}, "y": {"field": "median_model_to_ideal_time_ratio", "type": "quantitative", "label": "Median time ratio (× ideal)"}, "tooltip": [{"field": "matched_forms", "type": "quantitative", "label": "Matched forms"}, {"field": "median_model_duration_s", "type": "quantitative", "label": "Model median duration (s)"}, {"field": "median_ideal_duration_s", "type": "quantitative", "label": "Ideal median duration (s)"}]}},
        ],
        "tables": [
            {"id": "adjusted_model_table", "title": "Observed and non-dropdown outcomes", "subtitle": "Field correctness excludes the 25 ambiguous dropdown targets per model; full-fill rate uses the 25 forms without dropdowns.", "dataset": "adjusted_models", "sourceId": "adjusted_outcomes", "defaultSort": {"field": "nondrop_correctness_pct", "direction": "desc"}, "columns": [{"field": "model", "label": "Model", "type": "text"}, {"field": "observed_correctness_pct", "label": "Observed correct (%)", "type": "number"}, {"field": "nondrop_correctness_pct", "label": "Non-dropdown correct (%)", "type": "number"}, {"field": "full_fills", "label": "Full fills", "type": "number"}, {"field": "full_fill_rate_nondrop_pct", "label": "Full fills / 25 no-dropdown forms (%)", "type": "number"}]},
            {"id": "dropdown_audit_table", "title": "Dropdown verifier audit", "subtitle": "Twenty-five dropdown targets per model; expected-in-container readings are unresolved, not corrected successes.", "dataset": "dropdown_audit", "sourceId": "dropdown_verifier_audit", "defaultSort": {"field": "measurement_indeterminate_pct", "direction": "desc"}, "columns": [{"field": "model", "label": "Model", "type": "text"}, {"field": "dropdown_targets", "label": "Targets", "type": "number"}, {"field": "raw_verified_correct", "label": "Raw correct", "type": "number"}, {"field": "expected_embedded_in_container", "label": "Expected in container", "type": "number"}, {"field": "blank_or_not_attempted", "label": "Blank / not attempted", "type": "number"}, {"field": "measurement_indeterminate_pct", "label": "Indeterminate (%)", "type": "number"}]},
            {"id": "forms_without_dropdown_table", "title": "Performance on forms without dropdowns", "subtitle": "Twenty-five forms per model; field correctness and exact full-fill rates are directly scoreable.", "dataset": "forms_without_dropdown", "sourceId": "dropdown_form_groups", "defaultSort": {"field": "raw_full_fill_rate_pct", "direction": "desc"}, "columns": [{"field": "model", "label": "Model", "type": "text"}, {"field": "forms", "label": "Forms", "type": "number"}, {"field": "raw_field_correctness_pct", "label": "Field correct (%)", "type": "number"}, {"field": "raw_full_fills", "label": "Full fills", "type": "number"}, {"field": "raw_full_fill_rate_pct", "label": "Full-fill rate (%)", "type": "number"}]},
            {"id": "forms_with_dropdown_table", "title": "Performance on forms containing dropdowns", "subtitle": "Twenty-five forms per model; dropdown targets remain unresolved, so non-dropdown completion is the defensible comparison.", "dataset": "forms_with_dropdown", "sourceId": "dropdown_form_groups", "defaultSort": {"field": "non_dropdown_complete_rate_pct", "direction": "desc"}, "columns": [{"field": "model", "label": "Model", "type": "text"}, {"field": "forms", "label": "Forms", "type": "number"}, {"field": "dropdown_targets_unresolved", "label": "Unresolved dropdowns", "type": "number"}, {"field": "non_dropdown_field_correctness_pct", "label": "Non-dropdown correct (%)", "type": "number"}, {"field": "non_dropdown_complete_forms", "label": "Non-dropdown-complete forms", "type": "number"}, {"field": "non_dropdown_complete_rate_pct", "label": "Non-dropdown-complete (%)", "type": "number"}]},
            {"id": "widget_model_table", "title": "Widget failure rates by model", "subtitle": "Fill-only run 2; dropdown omitted; sorted by model and failure rate.", "dataset": "widget_by_model", "sourceId": "widget_failures", "defaultSort": {"field": "failure_rate_pct", "direction": "desc"}, "columns": [{"field": "model", "label": "Model", "type": "text"}, {"field": "widget_type", "label": "Widget", "type": "text"}, {"field": "target_fields", "label": "Targets", "type": "number"}, {"field": "missed_fields", "label": "Misses", "type": "number"}, {"field": "failure_rate_pct", "label": "Failure rate (%)", "type": "number"}]},
            {"id": "hard_forms_table", "title": "Hardest forms across all four models", "subtitle": "Lowest cross-model field correctness; widget mix shown per form.", "dataset": "hardest_forms", "sourceId": "hard_forms", "defaultSort": {"field": "cross_model_correctness_pct", "direction": "asc"}, "columns": [{"field": "form_id", "label": "Form", "type": "text"}, {"field": "fields", "label": "Fields", "type": "number"}, {"field": "complex_widgets", "label": "Structured widgets", "type": "number"}, {"field": "widget_mix", "label": "Widget mix", "type": "text"}, {"field": "models_full", "label": "Models full", "type": "number"}, {"field": "cross_model_correctness_pct", "label": "Correctness (%)", "type": "number"}]},
            {"id": "failure_matrix_table", "title": "Field-failure matrix by model", "subtitle": "Counts among final incorrect fields; ambiguous option-container text is shown separately.", "dataset": "failure_matrix", "sourceId": "failure_matrix", "defaultSort": {"field": "total_missed_fields", "direction": "desc"}, "columns": [{"field": "model", "label": "Model", "type": "text"}, {"field": "total_missed_fields", "label": "All misses", "type": "number"}, {"field": "dominant_behavioral_failure", "label": "Most common behavioral failure", "type": "text"}, {"field": "dominant_behavioral_failure_count", "label": "Dominant count", "type": "number"}, {"field": "dominant_behavioral_share_of_misses_pct", "label": "Share of misses (%)", "type": "number"}, {"field": "not_attempted_count", "label": "Not attempted", "type": "number"}, {"field": "attempted_but_blank_count", "label": "Attempted, blank", "type": "number"}, {"field": "wrong_value_count", "label": "Wrong value", "type": "number"}, {"field": "multi_choice_error_count", "label": "Multi-choice error", "type": "number"}, {"field": "ambiguous_option_text_count", "label": "Ambiguous option text", "type": "number"}]},
            {"id": "stop_reason_table", "title": "Incomplete-trial stop reasons", "subtitle": "Canonical fill-only run 2; successful filled-without-submit trials omitted.", "dataset": "stop_reasons", "sourceId": "trial_stop_reasons", "defaultSort": {"field": "incomplete_trials", "direction": "desc"}, "columns": [{"field": "model", "label": "Model", "type": "text"}, {"field": "stop_reason", "label": "Stop reason", "type": "text"}, {"field": "incomplete_trials", "label": "Incomplete trials", "type": "number"}, {"field": "share_of_model_incomplete_trials_pct", "label": "Share of incomplete trials (%)", "type": "number"}]},
            {"id": "model_hard_forms_table", "title": "Five hardest forms within each model", "subtitle": "Ranked by lowest field correctness within each 50-form condition.", "dataset": "hardest_by_model", "sourceId": "model_hard_forms", "defaultSort": {"field": "correctness_pct", "direction": "asc"}, "columns": [{"field": "model", "label": "Model", "type": "text"}, {"field": "rank", "label": "Within-model rank", "type": "number"}, {"field": "form_id", "label": "Form", "type": "text"}, {"field": "target_fields", "label": "Fields", "type": "number"}, {"field": "missed_fields", "label": "Misses", "type": "number"}, {"field": "correctness_pct", "label": "Correctness (%)", "type": "number"}]},
            {"id": "question_label_table", "title": "Repeated question labels with the most misses", "subtitle": "Up to five per model among labels observed on at least two forms; dropdowns omitted.", "dataset": "question_labels", "sourceId": "question_labels", "defaultSort": {"field": "missed_fields", "direction": "desc"}, "columns": [{"field": "model", "label": "Model", "type": "text"}, {"field": "label", "label": "Question label", "type": "text"}, {"field": "widget_type", "label": "Widget", "type": "text"}, {"field": "observations", "label": "Occurrences", "type": "number"}, {"field": "missed_fields", "label": "Misses", "type": "number"}, {"field": "failure_rate_pct", "label": "Failure rate (%)", "type": "number"}]},
            {"id": "consensus_question_table", "title": "Questions missed by all four model-interface conditions", "subtitle": "Non-dropdown items only; one canonical observation per condition and question.", "dataset": "consensus_questions", "sourceId": "question_item_consensus", "defaultSort": {"field": "form_id", "direction": "asc"}, "columns": [{"field": "form_id", "label": "Form", "type": "text"}, {"field": "question_index", "label": "Question", "type": "number"}, {"field": "label", "label": "Question label", "type": "text"}, {"field": "widget_type", "label": "Widget", "type": "text"}, {"field": "position_bucket", "label": "Position", "type": "text"}, {"field": "models_not_attempted", "label": "Models not attempting", "type": "number"}, {"field": "models_attempted_but_failed", "label": "Models attempting but failing", "type": "number"}]},
            {"id": "action_efficiency_table", "title": "Interaction behavior by model and trial outcome", "subtitle": "Canonical fill-only trials; screenshots, snapshots, waits, DONE, setup, and close excluded before computing totals and repeats.", "dataset": "action_efficiency", "sourceId": "canonical_action_behavior", "defaultSort": {"field": "adjacent_identical_rate_pct", "direction": "desc"}, "columns": [{"field": "model", "label": "Model", "type": "text"}, {"field": "outcome", "label": "Outcome", "type": "text"}, {"field": "trials", "label": "Trials", "type": "number"}, {"field": "median_actions", "label": "Median interactions", "type": "number"}, {"field": "actions_per_verified_field", "label": "Interactions / verified field", "type": "number"}, {"field": "click_share_pct", "label": "Click share (%)", "type": "number"}, {"field": "type_share_pct", "label": "Type share (%)", "type": "number"}, {"field": "scroll_share_pct", "label": "Scroll share (%)", "type": "number"}, {"field": "other_share_pct", "label": "Other share (%)", "type": "number"}, {"field": "adjacent_identical_rate_pct", "label": "Exact adjacent repeats (%)", "type": "number"}]},
            {"id": "action_mix_table", "title": "Most-used interaction types by model", "subtitle": "All 50 canonical trials per condition; state-changing interactions only.", "dataset": "action_mix", "sourceId": "canonical_action_behavior", "defaultSort": {"field": "action_count", "direction": "desc"}, "columns": [{"field": "model", "label": "Model", "type": "text"}, {"field": "action_type", "label": "Interaction type", "type": "text"}, {"field": "action_count", "label": "Interactions", "type": "number"}, {"field": "share_of_actions_pct", "label": "Share of interactions (%)", "type": "number"}]},
            {"id": "action_correction_table", "title": "Raw calls reconciled to task interactions", "subtitle": "Fifty canonical trials per model; observation and terminal overhead shown explicitly.", "dataset": "action_correction", "sourceId": "action_count_correction", "defaultSort": {"field": "excluded_share_pct", "direction": "desc"}, "columns": [{"field": "model", "label": "Model", "type": "text"}, {"field": "raw_actions", "label": "Raw calls", "type": "number"}, {"field": "interaction_actions", "label": "Task interactions", "type": "number"}, {"field": "excluded_observations", "label": "Observations excluded", "type": "number"}, {"field": "excluded_other", "label": "Other excluded", "type": "number"}, {"field": "excluded_share_pct", "label": "Excluded (%)", "type": "number"}, {"field": "median_interactions", "label": "Median interactions", "type": "number"}]},
            {"id": "gemini_position_table", "title": "Gemini failures by field position", "subtitle": "Target-field thirds within each form; 409 total field outcomes.", "dataset": "gemini_position", "sourceId": "gemini_position", "defaultSort": {"field": "failure_rate_pct", "direction": "asc"}, "columns": [{"field": "position", "label": "Position", "type": "text"}, {"field": "target_fields", "label": "Targets", "type": "number"}, {"field": "missed_fields", "label": "Misses", "type": "number"}, {"field": "failure_rate_pct", "label": "Failure rate (%)", "type": "number"}]},
            {"id": "baseline_behavior_table", "title": "Historical baseline interaction behavior by submission outcome", "subtitle": "978 primary trials; task interactions and recovered correctness medians within model and outcome.", "dataset": "baseline_behavior", "sourceId": "baseline_behavior", "defaultSort": {"field": "model", "direction": "asc"}, "columns": [{"field": "model", "label": "Model", "type": "text"}, {"field": "outcome", "label": "Outcome", "type": "text"}, {"field": "trials", "label": "Trials", "type": "number"}, {"field": "median_actions", "label": "Median interactions", "type": "number"}, {"field": "median_click_share_pct", "label": "Median click share (%)", "type": "number"}, {"field": "median_type_share_pct", "label": "Median type share (%)", "type": "number"}, {"field": "median_final_page_accuracy_pct", "label": "Final-page correct (%)", "type": "number"}, {"field": "median_scored_accuracy_pct", "label": "Recovered correct (%)", "type": "number"}]},
            {"id": "submission_audit_table", "title": "Historical submission-score recovery", "subtitle": "Successful submissions only; final-page zero contrasted with pre-submit recovery.", "dataset": "submission_audit", "sourceId": "submission_scoring_audit", "defaultSort": {"field": "submitted_trials", "direction": "desc"}, "columns": [{"field": "model", "label": "Model", "type": "text"}, {"field": "submitted_trials", "label": "Submitted", "type": "number"}, {"field": "final_page_zero_trials", "label": "Final-page zero", "type": "number"}, {"field": "recovered_nonzero_trials", "label": "Recovered nonzero", "type": "number"}, {"field": "recovered_perfect_trials", "label": "Recovered perfect", "type": "number"}, {"field": "recovered_partial_trials", "label": "Recovered partial", "type": "number"}, {"field": "unrecovered_zero_trials", "label": "Still zero", "type": "number"}, {"field": "median_recovered_pre_submit_accuracy_pct", "label": "Median recovered (%)", "type": "number"}]},
            {"id": "canonical_submission_table", "title": "Canonical submission audit", "subtitle": "All 200 comparison trials used the fill-only contract and made no submit attempts.", "dataset": "canonical_submission", "sourceId": "submission_scoring_audit", "defaultSort": {"field": "model", "direction": "asc"}, "columns": [{"field": "model", "label": "Model", "type": "text"}, {"field": "trials", "label": "Trials", "type": "number"}, {"field": "task_mode", "label": "Task mode", "type": "text"}, {"field": "submit_attempts", "label": "Submit attempts", "type": "number"}, {"field": "successful_submissions", "label": "Successful submissions", "type": "number"}]},
            {"id": "ideal_reference_table", "title": "Scripted ideal reference summary", "subtitle": "All 300 references: normalized interactions exclude observation, synchronization, setup, and teardown.", "dataset": "ideal_reference", "sourceId": "ideal_reference", "defaultSort": {"field": "reference_runs", "direction": "desc"}, "columns": [{"field": "reference_runs", "label": "Runs", "type": "number"}, {"field": "successful_submissions", "label": "Successful submissions", "type": "number"}, {"field": "median_interaction_actions", "label": "Median interactions", "type": "number"}, {"field": "median_total_trace_events", "label": "Raw trace events", "type": "number"}, {"field": "median_screenshot_events", "label": "Screenshots", "type": "number"}, {"field": "median_wait_events", "label": "Waits", "type": "number"}, {"field": "median_duration_s", "label": "Median duration (s)", "type": "number"}]},
            {"id": "model_ideal_table", "title": "Canonical models versus matched scripted ideal", "subtitle": "Fifty run-2 forms per condition; primary actions are normalized state-changing interactions.", "dataset": "model_vs_ideal", "sourceId": "model_vs_ideal", "defaultSort": {"field": "median_model_to_ideal_time_ratio", "direction": "desc"}, "columns": [{"field": "model", "label": "Model", "type": "text"}, {"field": "matched_forms", "label": "Forms", "type": "number"}, {"field": "full_fills", "label": "Full fills", "type": "number"}, {"field": "median_model_actions", "label": "Model interactions", "type": "number"}, {"field": "median_ideal_interaction_actions", "label": "Ideal interactions", "type": "number"}, {"field": "median_model_to_ideal_interaction_action_ratio", "label": "Interaction ratio", "type": "number"}, {"field": "median_model_duration_s", "label": "Model duration (s)", "type": "number"}, {"field": "median_ideal_duration_s", "label": "Ideal duration (s)", "type": "number"}, {"field": "median_model_to_ideal_time_ratio", "label": "Time ratio (× ideal)", "type": "number"}]},
            {"id": "methodology_gap_table", "title": "Priority methodology gap register", "subtitle": "P0 and P1 issues that affect interpretation or comparability.", "dataset": "methodology_gaps", "sourceId": "methodology_gaps", "defaultSort": {"field": "priority", "direction": "asc"}, "columns": [{"field": "priority", "label": "Priority", "type": "text"}, {"field": "gap", "label": "Gap", "type": "text"}, {"field": "why_it_matters", "label": "Why it matters", "type": "text"}, {"field": "concrete_fix", "label": "Concrete fix", "type": "text"}, {"field": "paper_treatment", "label": "Current paper treatment", "type": "text"}]},
        ],
        "sources": [{"id": s["id"], "label": s["query"]["description"]} for s in [source, widget_source, gemini_source, adjusted_source, hard_source, position_source, baseline_source, failure_source, stop_source, model_forms_source, question_labels_source, question_items_source, action_trace_source, methodology_source, ideal_reference_source, model_ideal_source, form_group_source, dropdown_audit_source, action_correction_source, submission_audit_source]],
        "blocks": blocks,
    },
    "snapshot": {
        "version": 1,
        "status": "ready",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "datasets": {
            "adjusted_models": adjusted,
            "widget_overall": widget_overall,
            "widget_by_model": widget_model_table,
            "hardest_forms": hardest,
            "hardest_by_model": hardest_by_model,
            "question_labels": question_labels,
            "consensus_questions": consensus_questions,
            "failure_matrix": failure_matrix,
            "stop_reasons": stop_reason_table,
            "action_efficiency": action_efficiency,
            "action_mix": action_mix,
            "action_correction": action_correction,
            "forms_without_dropdown": forms_without_dropdown,
            "forms_with_dropdown": forms_with_dropdown,
            "dropdown_audit": dropdown_audit,
            "submission_audit": submission_audit,
            "canonical_submission": canonical_submission,
            "methodology_gaps": methodology_gaps,
            "ideal_reference": ideal_reference,
            "model_vs_ideal": model_vs_ideal,
            "gemini_productivity": gemini_productivity,
            "gemini_position": gemini_position,
            "baseline_behavior": baseline_table,
            "baseline_highlights": baseline_highlights,
        },
        "access_issues": [],
    },
    "sources": [source, widget_source, gemini_source, adjusted_source, hard_source, position_source, baseline_source, failure_source, stop_source, model_forms_source, question_labels_source, question_items_source, action_trace_source, methodology_source, ideal_reference_source, model_ideal_source, form_group_source, dropdown_audit_source, action_correction_source, submission_audit_source],
}

# Keep the reader-facing report focused. The analysis script still produces the
# full comparison CSV interface, while the portable report carries only the
# evidence needed for the paper's main claims.
REPORT_BLOCKS = {
    "title", "executive_summary", "scope", "dropdown_risk",
    "dropdown_audit_table_block", "forms_without_dropdown_table_block",
    "forms_with_dropdown_table_block", "widget_story", "widget_chart_block",
    "widget_table_block", "failure_story", "failure_matrix_table_block",
    "trial_failure_story", "stop_reason_table_block", "form_story",
    "hard_forms_table_block", "question_consensus_story",
    "consensus_question_table_block", "action_comparison_story",
    "action_repeat_chart_block", "action_efficiency_table_block",
    "action_correction_story", "action_correction_table_block",
    "submission_correction_story", "submission_audit_table_block",
    "ideal_reference_story", "model_ideal_story", "ideal_time_chart_block",
    "model_ideal_table_block", "methodology_story", "recommendations", "caveats",
}
REPORT_CHARTS = {"widget_failure_chart", "action_repeat_chart", "ideal_time_ratio_chart"}
REPORT_TABLES = {
    "dropdown_audit_table", "forms_without_dropdown_table",
    "forms_with_dropdown_table", "widget_model_table", "hard_forms_table",
    "failure_matrix_table", "stop_reason_table", "consensus_question_table",
    "action_efficiency_table", "action_correction_table",
    "submission_audit_table", "model_ideal_table",
}
REPORT_DATASETS = {
    "widget_overall", "widget_by_model", "hardest_forms", "consensus_questions",
    "failure_matrix", "stop_reasons", "action_efficiency", "action_correction",
    "forms_without_dropdown", "forms_with_dropdown", "dropdown_audit",
    "submission_audit", "model_vs_ideal",
}
REPORT_SOURCES = {
    "interaction_analysis", "widget_failures", "hard_forms", "failure_matrix",
    "trial_stop_reasons", "question_item_consensus", "canonical_action_behavior",
    "action_count_correction", "submission_scoring_audit", "ideal_reference",
    "model_vs_ideal", "dropdown_form_groups", "dropdown_verifier_audit",
    "methodology_gaps",
}

artifact["manifest"]["blocks"] = [b for b in artifact["manifest"]["blocks"] if b["id"] in REPORT_BLOCKS]
artifact["manifest"]["charts"] = [c for c in artifact["manifest"]["charts"] if c["id"] in REPORT_CHARTS]
artifact["manifest"]["tables"] = [t for t in artifact["manifest"]["tables"] if t["id"] in REPORT_TABLES]
artifact["snapshot"]["datasets"] = {k: v for k, v in artifact["snapshot"]["datasets"].items() if k in REPORT_DATASETS}
artifact["sources"] = [s for s in artifact["sources"] if s["id"] in REPORT_SOURCES]
artifact["manifest"]["sources"] = [s for s in artifact["manifest"]["sources"] if s["id"] in REPORT_SOURCES]

(ROOT / "artifact.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(ROOT / "artifact.json")
