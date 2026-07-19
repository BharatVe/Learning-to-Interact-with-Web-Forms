#!/usr/bin/env python3
"""Build task/action-level failure tables for the canonical 50-form comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


MODEL_ORDER = ["Gemini 3.5 Flash", "OpenCUA direct-MCP", "Qwen3 Text", "Qwen3-VL"]
BASELINE_LABELS = {
    "computer_use_opencua_32b": "OpenCUA Native",
    "computer_use_opencua_32b_direct_mcp": "OpenCUA MCP",
    "text_qwen3_30b_a3b_instruct_2507": "Qwen Text",
    "vlm_qwen3_vl_30b_a3b_instruct": "Qwen VLM",
}

# Primary action-efficiency metrics count operations that change browser or form
# state. Observation, synchronization, terminal, and teardown calls are retained
# in audit columns but excluded from the normalized interaction count.
INTERACTION_ACTIONS = {
    "navigate", "click", "type", "scroll", "fill_form", "select_option",
    "check", "press_key", "hover", "drag", "upload_file", "run_interaction",
}
OBSERVATION_ACTIONS = {"snapshot", "screenshot"}
NON_INTERACTION_ACTIONS = OBSERVATION_ACTIONS | {"wait", "done", "close"}


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pct(num, den):
    return round(100 * num / den, 2) if den else 0.0


def is_empty(value):
    return value is None or value == "" or value == []


def failure_subtype(question):
    if question.get("verified_correct"):
        return "correct"
    expected = question.get("value")
    actual = question.get("actual_value")
    attempted = bool(question.get("attempted"))
    widget = question.get("widget_type") or "unknown"
    if is_empty(actual):
        return "attempted_but_blank" if attempted else "not_attempted"
    if isinstance(expected, list) and isinstance(actual, list):
        exp, act = set(map(str, expected)), set(map(str, actual))
        if act and act < exp:
            return "partial_multi_choice"
        if exp and exp < act:
            return "extra_multi_choice"
        return "wrong_multi_choice"
    if isinstance(expected, str) and isinstance(actual, str):
        if expected.strip() and expected.strip() in actual.strip() and expected.strip() != actual.strip():
            return "option_container_text" if widget in {"dropdown", "single_choice"} else "expected_embedded_in_wrong_value"
    return "wrong_value"


def position_bucket(index, total):
    relative = (index + 0.5) / total
    if relative <= 1 / 3:
        return "first third"
    if relative <= 2 / 3:
        return "middle third"
    return "final third"


def length_bucket(total):
    if total <= 6:
        return "4-6 fields"
    if total <= 8:
        return "7-8 fields"
    if total <= 10:
        return "9-10 fields"
    return "11-12 fields"


def normalize_action(name):
    aliases = {
        "click_mouse": "click",
        "browser_click": "click",
        "type_text": "type",
        "browser_type": "type",
        "browser_fill_form": "fill_form",
        "browser_select_option": "select_option",
        "browser_check": "check",
        "browser_navigate": "navigate",
        "browser_close": "close",
        "browser_snapshot": "snapshot",
        "browser_take_screenshot": "screenshot",
        "browser_hover": "hover",
        "browser_drag": "drag",
        "browser_file_upload": "upload_file",
        "scroll": "scroll",
        "browser_mouse_wheel": "scroll",
        "browser_press_key": "press_key",
        "press_key": "press_key",
        "wait": "wait",
        "browser_wait_for": "wait",
        "done": "done",
    }
    return aliases.get(name or "unknown", name or "unknown")


def is_interaction_action(name):
    """Return whether a normalized model-issued action changes task state."""
    return normalize_action(name) in INTERACTION_ACTIONS


def recovered_submission_score(summary):
    """Recover the last meaningful score before a confirmation page replaced the form."""
    submitted = bool(summary.get("submit_success"))
    candidates = []
    if submitted:
        candidates.extend([
            ("pre_successful_submit_verified_correctness", summary.get("pre_successful_submit_verified_correctness")),
            ("pre_first_submit_verified_correctness", summary.get("pre_first_submit_verified_correctness")),
        ])
    candidates.extend([
        ("scored_correctness", summary.get("scored_correctness")),
        ("question_correctness", summary.get("question_correctness")),
        ("verified_correctness", summary.get("verified_correctness")),
    ])
    for source, value in candidates:
        if value is not None:
            return int(value or 0), source
    return 0, "unavailable"


def ideal_interaction_count(trace_rows):
    """Count reference operations that navigate, fill, or submit the form."""
    count = 0
    for event in trace_rows:
        name = normalize_action(event.get("name"))
        if name in INTERACTION_ACTIONS:
            count += 1
            continue
        if event.get("name") == "browser_run_code":
            purpose = str((event.get("args") or {}).get("purpose") or "").lower()
            if purpose in {"fill_step", "submit", "interact", "interaction"}:
                count += 1
    return count


def trial_actions(trial_dir: Path, annotations: dict):
    # Gemini stores one parsed action per annotation step. Direct-MCP conditions
    # store one or more model tool calls per model_io row.
    steps = annotations.get("steps") or []
    if steps:
        return [normalize_action((s.get("action") or {}).get("action")) for s in steps]
    actions = []
    for row in read_jsonl(trial_dir / "model_io.jsonl"):
        for call in row.get("tool_calls") or []:
            actions.append(normalize_action(call.get("name")))
    return actions


def canonical_action_signatures(trial_dir: Path, annotations: dict):
    """Return exact model-issued action signatures for the 200-trial cohort."""
    steps = annotations.get("steps") or []
    if steps:
        return [json.dumps(step.get("action") or {}, sort_keys=True, ensure_ascii=False) for step in steps]
    signatures = []
    for row in read_jsonl(trial_dir / "model_io.jsonl"):
        for call in row.get("tool_calls") or []:
            signatures.append(json.dumps({
                "name": normalize_action(call.get("name")),
                "arguments": call.get("arguments") or {},
            }, sort_keys=True, ensure_ascii=False))
    return signatures


def action_signatures(trial_dir: Path, annotations: dict):
    steps = annotations.get("steps") or []
    if steps:
        signatures = []
        for step in steps:
            action = step.get("action") or {}
            signatures.append(json.dumps(action, sort_keys=True, ensure_ascii=False))
        return signatures
    # The baseline index already contains direct-MCP action counts. Avoid loading
    # hundreds of very large model_io histories just to estimate adjacency.
    return []


def adjacent_repeat_count(signatures):
    return sum(signatures[i] == signatures[i - 1] for i in range(1, len(signatures)))


def pearson(xs, ys):
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return round(numerator / denominator, 3) if denominator else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--trials-csv", type=Path, required=True)
    parser.add_argument("--baseline-actions-csv", type=Path)
    parser.add_argument("--reference-runs-csv", type=Path)
    parser.add_argument("--reference-actions-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    with args.trials_csv.open(encoding="utf-8") as handle:
        canonical = list(csv.DictReader(handle))

    field_rows = []
    trial_rows = []
    action_rows = []
    gemini_progress_rows = []
    missing_annotations = []

    for row in canonical:
        summary_path = args.project_root / row["source_summary"]
        trial_dir = summary_path.parent
        annotation_path = trial_dir / "annotations.json"
        if not annotation_path.exists():
            missing_annotations.append(str(annotation_path))
            continue
        ann = read_json(annotation_path)
        summary = read_json(summary_path)
        questions = ann.get("questions") or []
        actions = trial_actions(trial_dir, ann)
        signatures = canonical_action_signatures(trial_dir, ann)
        interaction_indexes = [i for i, action in enumerate(actions) if is_interaction_action(action)]
        interaction_actions = [actions[i] for i in interaction_indexes]
        interaction_signatures = [signatures[i] for i in interaction_indexes if i < len(signatures)]
        raw_counts = Counter(actions)
        counts = Counter(interaction_actions)
        correct = sum(bool(q.get("verified_correct")) for q in questions)
        adjacent_exact = adjacent_repeat_count(interaction_signatures)
        adjacent_same_type = sum(interaction_actions[i] == interaction_actions[i - 1] for i in range(1, len(interaction_actions)))
        has_dropdown = any((q.get("widget_type") or "unknown") == "dropdown" for q in questions)
        trial_rows.append({
            "model": row["model"],
            "model_id": row["model_id"],
            "form_id": row["form_id"],
            "answer_run_id": row.get("answer_run_id") or "run_0002",
            "question_total": len(questions),
            "verified_correct": correct,
            "correctness_pct": pct(correct, len(questions)),
            "full_fill_success": row["full_fill_success"],
            "stop_reason": row["stop_reason"],
            "task_mode": summary.get("task_mode") or "fill_only_done",
            "submit_success": bool(summary.get("submit_success")),
            "submit_attempt_count": int(summary.get("submit_attempt_count") or 0),
            "form_group": "forms_with_dropdown" if has_dropdown else "forms_without_dropdown",
            "action_count_export": int(row["action_count"] or 0),
            "action_count_parsed": len(actions),
            "interaction_action_count": len(interaction_actions),
            "observation_action_count": sum(raw_counts[a] for a in OBSERVATION_ACTIONS),
            "excluded_noninteraction_action_count": len(actions) - len(interaction_actions),
            "click_actions": counts["click"],
            "type_actions": counts["type"],
            "scroll_actions": counts["scroll"],
            "select_actions": counts["select_option"] + counts["check"],
            "other_actions": len(interaction_actions) - counts["click"] - counts["type"] - counts["scroll"] - counts["select_option"] - counts["check"],
            "distinct_action_types": len(counts),
            "adjacent_same_type_actions": adjacent_same_type,
            "adjacent_same_type_rate_pct": pct(adjacent_same_type, max(len(interaction_actions) - 1, 1)),
            "adjacent_identical_actions": adjacent_exact,
            "adjacent_identical_rate_pct": pct(adjacent_exact, max(len(interaction_signatures) - 1, 1)),
            "duration_s": float(row["duration_s"] or 0),
            "source_summary": row["source_summary"],
        })
        for action, count in counts.items():
            action_rows.append({
                "model": row["model"], "form_id": row["form_id"], "full_fill_success": row["full_fill_success"],
                "stop_reason": row["stop_reason"], "action_type": action, "action_count": count,
            })
        for idx, question in enumerate(questions):
            field_rows.append({
                "model": row["model"],
                "model_id": row["model_id"],
                "form_id": row["form_id"],
                "question_id": question.get("question_id"),
                "question_index": idx + 1,
                "question_total": len(questions),
                "position_bucket": position_bucket(idx, len(questions)),
                "length_bucket": length_bucket(len(questions)),
                "label": question.get("label"),
                "widget_type": question.get("widget_type") or "unknown",
                "attempted": bool(question.get("attempted")),
                "verified": bool(question.get("verified")),
                "verified_correct": bool(question.get("verified_correct")),
                "failure_subtype": failure_subtype(question),
                "expected_value": json.dumps(question.get("value"), ensure_ascii=False),
                "actual_value": json.dumps(question.get("actual_value"), ensure_ascii=False),
                "full_fill_success": row["full_fill_success"],
                "stop_reason": row["stop_reason"],
                "source_summary": row["source_summary"],
            })

        if row["model"] == "Gemini 3.5 Flash":
            prior_correct = set()
            for step in ann.get("steps") or []:
                current_correct = {v.get("question_id") for v in step.get("verification") or [] if v.get("verified_correct")}
                newly_correct = current_correct - prior_correct
                action = normalize_action((step.get("action") or {}).get("action"))
                if not is_interaction_action(action):
                    continue
                gemini_progress_rows.append({
                    "form_id": row["form_id"],
                    "step_index": step.get("step_index"),
                    "action_type": action,
                    "new_correct_fields": len(newly_correct),
                    "field_progress": bool(newly_correct),
                    "harness_progress_flag": bool(step.get("progress_made")),
                    "remaining_answers_before": step.get("remaining_answers_before"),
                    "status": step.get("status"),
                    "error": step.get("error"),
                })
                prior_correct = current_correct

    write_csv(out / "field_outcomes.csv", field_rows)
    write_csv(out / "trial_action_profile.csv", trial_rows)
    write_csv(out / "action_counts_long.csv", action_rows)
    write_csv(out / "gemini_action_progress.csv", gemini_progress_rows)

    canonical_submission_audit = []
    action_count_correction = []
    for model in MODEL_ORDER:
        selected = [r for r in trial_rows if r["model"] == model]
        canonical_submission_audit.append({
            "model": model,
            "trials": len(selected),
            "task_modes": ";".join(sorted({r["task_mode"] for r in selected})),
            "submit_attempts": sum(r["submit_attempt_count"] for r in selected),
            "successful_submissions": sum(r["submit_success"] for r in selected),
            "expected_no_submit_trials": sum(r["task_mode"] == "fill_only_done" for r in selected),
        })
        raw_total = sum(r["action_count_parsed"] for r in selected)
        interaction_total = sum(r["interaction_action_count"] for r in selected)
        action_count_correction.append({
            "model": model,
            "trials": len(selected),
            "model_issued_actions_raw": raw_total,
            "interaction_actions": interaction_total,
            "excluded_observation_actions": sum(r["observation_action_count"] for r in selected),
            "excluded_other_noninteraction_actions": raw_total - interaction_total - sum(r["observation_action_count"] for r in selected),
            "excluded_share_of_raw_pct": pct(raw_total - interaction_total, raw_total),
            "median_raw_actions_per_trial": round(statistics.median(r["action_count_parsed"] for r in selected), 2),
            "median_interaction_actions_per_trial": round(statistics.median(r["interaction_action_count"] for r in selected), 2),
        })
    write_csv(out / "canonical_submission_audit.csv", canonical_submission_audit)
    write_csv(out / "action_count_correction_by_model.csv", action_count_correction)

    def summarize_fields(keys):
        groups = defaultdict(list)
        for item in field_rows:
            groups[tuple(item[k] for k in keys)].append(item)
        result = []
        for key, items in groups.items():
            correct = sum(i["verified_correct"] for i in items)
            misses = len(items) - correct
            row = dict(zip(keys, key))
            row.update({"target_fields": len(items), "correct_fields": correct, "missed_fields": misses,
                        "correctness_pct": pct(correct, len(items)), "failure_rate_pct": pct(misses, len(items))})
            result.append(row)
        return result

    widget_by_model = summarize_fields(["model", "widget_type"])
    widget_by_model.sort(key=lambda r: (MODEL_ORDER.index(r["model"]), -r["failure_rate_pct"], r["widget_type"]))
    write_csv(out / "widget_difficulty_by_model.csv", widget_by_model)

    widget_overall = summarize_fields(["widget_type"])
    widget_overall.sort(key=lambda r: (-r["failure_rate_pct"], -r["target_fields"]))
    write_csv(out / "widget_difficulty_overall.csv", widget_overall)

    # Separate forms by whether the form contains any dropdown. On dropdown
    # forms, raw full-fill remains verifier-contaminated; non-dropdown-complete
    # shows whether every independently measurable field was correct.
    dropdown_forms = {
        r["form_id"] for r in field_rows if r["widget_type"] == "dropdown"
    }
    form_group_rows = []
    for model in MODEL_ORDER:
        for group_name, group_forms in (
            ("forms_without_dropdown", {r["form_id"] for r in field_rows} - dropdown_forms),
            ("forms_with_dropdown", dropdown_forms),
        ):
            selected = [r for r in field_rows if r["model"] == model and r["form_id"] in group_forms]
            nondrop = [r for r in selected if r["widget_type"] != "dropdown"]
            per_form = defaultdict(list)
            for item in selected:
                per_form[item["form_id"]].append(item)
            raw_full = sum(all(i["verified_correct"] for i in items) for items in per_form.values())
            nondrop_full = sum(
                all(i["verified_correct"] for i in items if i["widget_type"] != "dropdown")
                for items in per_form.values()
            )
            form_group_rows.append({
                "model": model,
                "form_group": group_name,
                "forms": len(per_form),
                "target_fields": len(selected),
                "correct_fields_raw": sum(i["verified_correct"] for i in selected),
                "raw_field_correctness_pct": pct(sum(i["verified_correct"] for i in selected), len(selected)),
                "non_dropdown_targets": len(nondrop),
                "non_dropdown_correct": sum(i["verified_correct"] for i in nondrop),
                "non_dropdown_field_correctness_pct": pct(sum(i["verified_correct"] for i in nondrop), len(nondrop)),
                "dropdown_targets_unresolved": len(selected) - len(nondrop),
                "raw_full_fills": raw_full,
                "raw_full_fill_rate_pct": pct(raw_full, len(per_form)),
                "non_dropdown_complete_forms": nondrop_full,
                "non_dropdown_complete_rate_pct": pct(nondrop_full, len(per_form)),
            })
    write_csv(out / "performance_forms_without_dropdown.csv", [r for r in form_group_rows if r["form_group"] == "forms_without_dropdown"])
    write_csv(out / "performance_forms_with_dropdown.csv", [r for r in form_group_rows if r["form_group"] == "forms_with_dropdown"])

    dropdown_audit = []
    dropdown_examples = []
    for model in MODEL_ORDER:
        selected = [r for r in field_rows if r["model"] == model and r["widget_type"] == "dropdown"]
        container = [r for r in selected if r["failure_subtype"] == "option_container_text"]
        empty = [r for r in selected if r["failure_subtype"] in {"not_attempted", "attempted_but_blank"}]
        dropdown_audit.append({
            "model": model,
            "dropdown_targets": len(selected),
            "raw_verified_correct": sum(r["verified_correct"] for r in selected),
            "raw_verified_incorrect": sum(not r["verified_correct"] for r in selected),
            "attempted": sum(r["attempted"] for r in selected),
            "actual_value_nonempty": sum(not is_empty(json.loads(r["actual_value"])) for r in selected),
            "expected_embedded_in_container": len(container),
            "blank_or_not_attempted": len(empty),
            "other_mismatch": len(selected) - len(container) - len(empty),
            "measurement_indeterminate_pct": pct(len(container), len(selected)),
        })
        dropdown_examples.extend({
            "model": r["model"], "form_id": r["form_id"], "question_id": r["question_id"],
            "label": r["label"], "attempted": r["attempted"],
            "expected_value": r["expected_value"], "recorded_actual_value": r["actual_value"],
            "failure_subtype": r["failure_subtype"],
        } for r in selected)
    write_csv(out / "dropdown_verifier_audit_by_model.csv", dropdown_audit)
    write_csv(out / "dropdown_verifier_examples.csv", dropdown_examples)

    pos = summarize_fields(["model", "position_bucket"])
    write_csv(out / "position_difficulty_by_model.csv", pos)
    lens = summarize_fields(["model", "length_bucket"])
    write_csv(out / "form_length_difficulty_by_model.csv", lens)

    failure_groups = defaultdict(list)
    for item in field_rows:
        if not item["verified_correct"]:
            failure_groups[(item["model"], item["failure_subtype"])].append(item)
    failure_summary = []
    model_misses = Counter(i["model"] for i in field_rows if not i["verified_correct"])
    for (model, subtype), items in failure_groups.items():
        failure_summary.append({"model": model, "failure_subtype": subtype, "missed_fields": len(items),
                                "share_of_model_misses_pct": pct(len(items), model_misses[model])})
    failure_summary.sort(key=lambda r: (MODEL_ORDER.index(r["model"]), -r["missed_fields"]))
    write_csv(out / "failure_subtypes_by_model.csv", failure_summary)

    # Wide, comparison-ready failure matrix plus the dominant behavioral miss.
    # Keep option_container_text visible but exclude it from the behavioral
    # winner because custom-dropdown verification is known to be ambiguous.
    failure_types = sorted({r["failure_subtype"] for r in failure_summary})
    failure_lookup = {(r["model"], r["failure_subtype"]): r for r in failure_summary}
    failure_matrix = []
    for model in MODEL_ORDER:
        behavioral = [
            r for r in failure_summary
            if r["model"] == model and r["failure_subtype"] != "option_container_text"
        ]
        behavioral.sort(key=lambda r: (-r["missed_fields"], r["failure_subtype"]))
        top = behavioral[0] if behavioral else None
        row = {
            "model": model,
            "total_missed_fields": model_misses[model],
            "dominant_behavioral_failure": top["failure_subtype"] if top else "",
            "dominant_behavioral_failure_count": top["missed_fields"] if top else 0,
            "dominant_behavioral_share_of_misses_pct": top["share_of_model_misses_pct"] if top else 0.0,
        }
        for subtype in failure_types:
            item = failure_lookup.get((model, subtype))
            row[f"{subtype}_count"] = item["missed_fields"] if item else 0
            row[f"{subtype}_share_pct"] = item["share_of_model_misses_pct"] if item else 0.0
        failure_matrix.append(row)
    write_csv(out / "failure_matrix_by_model.csv", failure_matrix)

    form_groups = defaultdict(list)
    for item in field_rows:
        form_groups[item["form_id"]].append(item)
    hard_forms = []
    for form, items in form_groups.items():
        by_model = defaultdict(list)
        for item in items:
            by_model[item["model"]].append(item)
        total = len(items)
        correct = sum(i["verified_correct"] for i in items)
        hard_forms.append({
            "form_id": form,
            "fields_per_model": len(items) // len(by_model),
            "models_full": sum(all(x["verified_correct"] for x in xs) for xs in by_model.values()),
            "correct_fields_all_models": correct,
            "target_fields_all_models": total,
            "correctness_pct_all_models": pct(correct, total),
            **{f"{model}_correctness_pct": pct(sum(x["verified_correct"] for x in by_model.get(model, [])), len(by_model.get(model, []))) for model in MODEL_ORDER},
        })
    hard_forms.sort(key=lambda r: (r["correctness_pct_all_models"], r["models_full"], r["form_id"]))
    write_csv(out / "hardest_forms_cross_model.csv", hard_forms)

    hardest_by_model = []
    for model in MODEL_ORDER:
        model_rows = []
        for form, items in form_groups.items():
            selected = [i for i in items if i["model"] == model]
            correct = sum(i["verified_correct"] for i in selected)
            model_rows.append({
                "model": model,
                "form_id": form,
                "target_fields": len(selected),
                "correct_fields": correct,
                "missed_fields": len(selected) - correct,
                "correctness_pct": pct(correct, len(selected)),
                "full_fill": bool(selected) and correct == len(selected),
            })
        model_rows.sort(key=lambda r: (r["correctness_pct"], -r["target_fields"], r["form_id"]))
        for rank, item in enumerate(model_rows, start=1):
            item["difficulty_rank_within_model"] = rank
            hardest_by_model.append(item)
    write_csv(out / "hardest_forms_by_model.csv", hardest_by_model)

    # Labels repeat across forms for common identity/contact fields; rank only
    # when a label has at least four model-field observations.
    label_groups = defaultdict(list)
    for item in field_rows:
        label_groups[(item["label"], item["widget_type"])].append(item)
    labels = []
    for (label, widget), items in label_groups.items():
        if len(items) < 4:
            continue
        correct = sum(i["verified_correct"] for i in items)
        labels.append({"label": label, "widget_type": widget, "observations": len(items),
                       "missed_fields": len(items)-correct, "failure_rate_pct": pct(len(items)-correct, len(items)),
                       "forms": len({i["form_id"] for i in items})})
    labels.sort(key=lambda r: (-r["missed_fields"], -r["failure_rate_pct"], r["label"]))
    write_csv(out / "repeated_field_label_difficulty.csv", labels)

    label_model_groups = defaultdict(list)
    for item in field_rows:
        label_model_groups[(item["model"], item["label"], item["widget_type"])].append(item)
    labels_by_model = []
    for (model, label, widget), items in label_model_groups.items():
        if len(items) < 2:
            continue
        correct = sum(i["verified_correct"] for i in items)
        labels_by_model.append({
            "model": model,
            "label": label,
            "widget_type": widget,
            "observations": len(items),
            "forms": len({i["form_id"] for i in items}),
            "missed_fields": len(items) - correct,
            "failure_rate_pct": pct(len(items) - correct, len(items)),
        })
    labels_by_model.sort(key=lambda r: (MODEL_ORDER.index(r["model"]), -r["missed_fields"], -r["failure_rate_pct"], r["label"]))
    write_csv(out / "question_label_difficulty_by_model.csv", labels_by_model)

    item_groups = defaultdict(list)
    for item in field_rows:
        item_groups[(
            item["form_id"], item["question_id"], item["label"], item["widget_type"],
            item["question_index"], item["question_total"], item["position_bucket"],
        )].append(item)
    item_difficulty = []
    for key, items in item_groups.items():
        failed = [i for i in items if not i["verified_correct"]]
        successful = [i for i in items if i["verified_correct"]]
        item_difficulty.append({
            "form_id": key[0],
            "question_id": key[1],
            "label": key[2],
            "widget_type": key[3],
            "question_index": key[4],
            "question_total": key[5],
            "position_bucket": key[6],
            "model_observations": len(items),
            "models_correct": len(successful),
            "models_failed": len(failed),
            "cross_model_failure_rate_pct": pct(len(failed), len(items)),
            "models_not_attempted": sum(not i["attempted"] for i in failed),
            "models_attempted_but_failed": sum(i["attempted"] for i in failed),
            "failed_model_names": "; ".join(i["model"] for i in failed),
            "ambiguous_dropdown_measurement": key[3] == "dropdown",
        })
    item_difficulty.sort(key=lambda r: (
        r["ambiguous_dropdown_measurement"], -r["models_failed"], -r["question_total"],
        r["form_id"], r["question_index"],
    ))
    write_csv(out / "question_difficulty_cross_model.csv", item_difficulty)

    stop_groups = defaultdict(list)
    for item in trial_rows:
        stop_groups[(item["model"], item["stop_reason"])].append(item)
    stop_rows = []
    model_trial_counts = Counter(i["model"] for i in trial_rows)
    model_incomplete_counts = Counter(i["model"] for i in trial_rows if i["full_fill_success"] != "true")
    for (model, reason), items in stop_groups.items():
        incomplete = sum(i["full_fill_success"] != "true" for i in items)
        stop_rows.append({
            "model": model,
            "stop_reason": reason,
            "trials": len(items),
            "share_of_model_trials_pct": pct(len(items), model_trial_counts[model]),
            "incomplete_trials": incomplete,
            "share_of_model_incomplete_trials_pct": pct(incomplete, model_incomplete_counts[model]),
        })
    stop_rows.sort(key=lambda r: (MODEL_ORDER.index(r["model"]), -r["incomplete_trials"], -r["trials"], r["stop_reason"]))
    write_csv(out / "trial_stop_reasons_by_model.csv", stop_rows)

    outcome_action_groups = defaultdict(int)
    for item in action_rows:
        outcome = "full fill" if item["full_fill_success"] == "true" else "incomplete"
        outcome_action_groups[(item["model"], outcome, item["action_type"])] += item["action_count"]
    outcome_action_totals = Counter()
    for (model, outcome, _), count in outcome_action_groups.items():
        outcome_action_totals[(model, outcome)] += count
    outcome_action_mix = []
    for (model, outcome, action), count in outcome_action_groups.items():
        outcome_action_mix.append({
            "model": model,
            "outcome": outcome,
            "action_type": action,
            "action_count": count,
            "share_of_outcome_actions_pct": pct(count, outcome_action_totals[(model, outcome)]),
        })
    outcome_action_mix.sort(key=lambda r: (
        MODEL_ORDER.index(r["model"]), r["outcome"], -r["action_count"], r["action_type"]
    ))
    write_csv(out / "action_mix_by_model_outcome.csv", outcome_action_mix)

    action_efficiency = []
    for model in MODEL_ORDER:
        for outcome in ("full fill", "incomplete"):
            selected = [r for r in trial_rows if r["model"] == model and ("full fill" if r["full_fill_success"] == "true" else "incomplete") == outcome]
            if not selected:
                continue
            total_actions = sum(r["interaction_action_count"] for r in selected)
            total_correct = sum(r["verified_correct"] for r in selected)
            possible_adjacencies = sum(max(r["interaction_action_count"] - 1, 0) for r in selected)
            action_efficiency.append({
                "model": model,
                "outcome": outcome,
                "trials": len(selected),
                "total_actions": total_actions,
                "median_actions": round(statistics.median(r["interaction_action_count"] for r in selected), 2),
                "total_verified_fields": total_correct,
                "actions_per_verified_field": round(total_actions / total_correct, 2) if total_correct else "",
                "median_correctness_pct": round(statistics.median(r["correctness_pct"] for r in selected), 2),
                "click_share_pct": pct(sum(r["click_actions"] for r in selected), total_actions),
                "type_share_pct": pct(sum(r["type_actions"] for r in selected), total_actions),
                "scroll_share_pct": pct(sum(r["scroll_actions"] for r in selected), total_actions),
                "other_share_pct": pct(sum(r["other_actions"] for r in selected), total_actions),
                "adjacent_same_type_rate_pct": pct(sum(r["adjacent_same_type_actions"] for r in selected), possible_adjacencies),
                "adjacent_identical_rate_pct": pct(sum(r["adjacent_identical_actions"] for r in selected), possible_adjacencies),
            })
    write_csv(out / "action_efficiency_by_model_outcome.csv", action_efficiency)

    action_group = defaultdict(int)
    for item in action_rows:
        action_group[(item["model"], item["action_type"])] += item["action_count"]
    action_summary = []
    model_actions = Counter()
    for (model, action), count in action_group.items():
        model_actions[model] += count
    for (model, action), count in action_group.items():
        action_summary.append({"model": model, "action_type": action, "action_count": count,
                               "share_of_actions_pct": pct(count, model_actions[model])})
    action_summary.sort(key=lambda r: (MODEL_ORDER.index(r["model"]), -r["action_count"]))
    write_csv(out / "action_mix_by_model.csv", action_summary)

    gp = defaultdict(lambda: {"actions": 0, "progress": 0, "new": 0})
    for item in gemini_progress_rows:
        bucket = gp[item["action_type"]]
        bucket["actions"] += 1
        bucket["progress"] += int(item["field_progress"])
        bucket["new"] += int(item["new_correct_fields"])
    gemini_action_summary = []
    for action, vals in gp.items():
        gemini_action_summary.append({"action_type": action, "actions": vals["actions"],
                                      "actions_with_new_correct_field": vals["progress"],
                                      "new_correct_fields": vals["new"],
                                      "field_progress_rate_pct": pct(vals["progress"], vals["actions"]),
                                      "no_field_progress_actions": vals["actions"] - vals["progress"]})
    gemini_action_summary.sort(key=lambda r: -r["actions"])
    write_csv(out / "gemini_action_productivity.csv", gemini_action_summary)

    correlation_rows = []
    for model in MODEL_ORDER:
        rows = [r for r in trial_rows if r["model"] == model]
        correlation_rows.append({
            "model": model,
            "trials": len(rows),
            "action_correctness_pearson_r": pearson([r["interaction_action_count"] for r in rows], [r["correctness_pct"] for r in rows]),
            "median_actions_full_fill": round(statistics.median([r["interaction_action_count"] for r in rows if r["full_fill_success"] == "true"]), 2) if any(r["full_fill_success"] == "true" for r in rows) else "",
            "median_actions_incomplete": round(statistics.median([r["interaction_action_count"] for r in rows if r["full_fill_success"] != "true"]), 2) if any(r["full_fill_success"] != "true" for r in rows) else "",
            "median_correctness_full_fill": 100.0,
            "median_correctness_incomplete": round(statistics.median([r["correctness_pct"] for r in rows if r["full_fill_success"] != "true"]), 2),
        })
    write_csv(out / "action_outcome_relationship.csv", correlation_rows)

    row_counts = {
        "canonical_trials": len(canonical),
        "field_outcomes": len(field_rows),
        "missing_annotations": len(missing_annotations),
    }
    if missing_annotations:
        (out / "missing_annotations.txt").write_text("\n".join(missing_annotations) + "\n", encoding="utf-8")
    if args.baseline_actions_csv:
        with args.baseline_actions_csv.open(encoding="utf-8") as handle:
            baseline_index = list(csv.DictReader(handle))
        baseline_trials = []
        baseline_fields = []
        baseline_recovered_perfect_fields = []
        baseline_action_mix = defaultdict(int)
        for row in baseline_index:
            summary_path = args.project_root / row["summary_path"]
            trial_dir = summary_path.parent
            summary = read_json(summary_path)
            annotation_path = trial_dir / "annotations.json"
            if not annotation_path.exists():
                continue
            ann = read_json(annotation_path)
            actions = json.loads(row.get("action_counts_json") or "{}")
            normalized_actions = Counter()
            for action, count in actions.items():
                normalized_actions[normalize_action(action)] += int(count)
            interaction_counts = Counter({
                action: count for action, count in normalized_actions.items()
                if is_interaction_action(action)
            })
            model_issued = sum(normalized_actions.values())
            executable = sum(interaction_counts.values())
            signatures = action_signatures(trial_dir, ann)
            model = BASELINE_LABELS.get(row["model_id"], row["display_label"])
            submitted = bool(summary.get("submit_success"))
            scored_correctness, score_source = recovered_submission_score(summary)
            question_total = int(summary.get("question_total") or 0)
            final_verified = int(summary.get("verified_correctness") or 0)
            perfect_pre_submit = submitted and question_total > 0 and scored_correctness == question_total
            baseline_trials.append({
                "model": model,
                "model_id": row["model_id"],
                "form_id": row["form_id"],
                "answer_run_id": row["answer_run_id"],
                "submit_success": submitted,
                "exact_success": bool(summary.get("success")),
                "question_total": question_total,
                "final_verified_correctness": final_verified,
                "scored_correctness": scored_correctness,
                "score_source": score_source,
                "perfect_pre_submit": perfect_pre_submit,
                "postsubmit_zero_artifact": submitted and final_verified == 0 and scored_correctness > 0,
                "model_issued_actions": model_issued,
                "executable_actions": executable,
                "excluded_noninteraction_actions": model_issued - executable,
                "click_actions": interaction_counts["click"],
                "type_actions": interaction_counts["type"] + interaction_counts["fill_form"],
                "scroll_actions": interaction_counts["scroll"],
                "adjacent_identical_actions": adjacent_repeat_count(signatures) if signatures else "",
                "adjacent_repeat_rate_pct": pct(adjacent_repeat_count(signatures), max(len(signatures) - 1, 1)) if signatures else "",
                "nonprogress_ratio": ann.get("nonprogress_ratio"),
                "loop_ratio": ann.get("loop_ratio"),
                "source_summary": row["summary_path"],
            })
            outcome = "submitted" if submitted else "not submitted"
            for action, count in interaction_counts.items():
                baseline_action_mix[(model, outcome, action)] += int(count)
            if perfect_pre_submit:
                for idx, question in enumerate(ann.get("questions") or []):
                    baseline_recovered_perfect_fields.append({
                        "model": model,
                        "model_id": row["model_id"],
                        "form_id": row["form_id"],
                        "answer_run_id": row["answer_run_id"],
                        "question_index": idx + 1,
                        "question_total": question_total,
                        "label": question.get("label"),
                        "widget_type": question.get("widget_type") or "unknown",
                        "expected_value": json.dumps(question.get("value"), ensure_ascii=False),
                        "recovered_verified_correct": True,
                        "recovery_source": score_source,
                        "source_summary": row["summary_path"],
                    })
            # Final-page verification is invalid after successful submission, so
            # task-level baseline failure cuts intentionally use non-submitted trials.
            if submitted:
                continue
            questions = ann.get("questions") or []
            for idx, question in enumerate(questions):
                baseline_fields.append({
                    "model": model,
                    "model_id": row["model_id"],
                    "form_id": row["form_id"],
                    "answer_run_id": row["answer_run_id"],
                    "question_index": idx + 1,
                    "question_total": len(questions),
                    "position_bucket": position_bucket(idx, len(questions)),
                    "label": question.get("label"),
                    "widget_type": question.get("widget_type") or "unknown",
                    "attempted": bool(question.get("attempted")),
                    "verified_correct": bool(question.get("verified_correct")),
                    "failure_subtype": failure_subtype(question),
                    "expected_value": json.dumps(question.get("value"), ensure_ascii=False),
                    "actual_value": json.dumps(question.get("actual_value"), ensure_ascii=False),
                    "source_summary": row["summary_path"],
                })
        write_csv(out / "baseline_primary_trial_action_profile.csv", baseline_trials)
        write_csv(out / "baseline_recovered_perfect_submitted_fields.csv", baseline_recovered_perfect_fields)

        baseline_widget_groups = defaultdict(list)
        baseline_pos_groups = defaultdict(list)
        baseline_failure_groups = defaultdict(list)
        for item in baseline_fields:
            baseline_widget_groups[(item["model"], item["widget_type"])].append(item)
            baseline_pos_groups[(item["model"], item["position_bucket"])].append(item)
            if not item["verified_correct"]:
                baseline_failure_groups[(item["model"], item["failure_subtype"])].append(item)
        baseline_widget = []
        for (model, widget), items in baseline_widget_groups.items():
            correct = sum(i["verified_correct"] for i in items)
            baseline_widget.append({"model": model, "widget_type": widget, "target_fields_in_nonsubmitted_trials": len(items),
                                    "correct_fields": correct, "missed_fields": len(items)-correct,
                                    "correctness_pct": pct(correct, len(items)), "failure_rate_pct": pct(len(items)-correct, len(items))})
        baseline_widget.sort(key=lambda r: (r["model"], -r["failure_rate_pct"], r["widget_type"]))
        write_csv(out / "baseline_widget_difficulty_nonsubmitted.csv", baseline_widget)

        baseline_position = []
        for (model, position), items in baseline_pos_groups.items():
            correct = sum(i["verified_correct"] for i in items)
            baseline_position.append({"model": model, "position_bucket": position, "target_fields_in_nonsubmitted_trials": len(items),
                                      "correct_fields": correct, "missed_fields": len(items)-correct,
                                      "correctness_pct": pct(correct, len(items)), "failure_rate_pct": pct(len(items)-correct, len(items))})
        write_csv(out / "baseline_position_difficulty_nonsubmitted.csv", baseline_position)

        baseline_misses = Counter(i["model"] for i in baseline_fields if not i["verified_correct"])
        baseline_failure = []
        for (model, subtype), items in baseline_failure_groups.items():
            baseline_failure.append({"model": model, "failure_subtype": subtype, "missed_fields": len(items),
                                     "share_of_model_misses_pct": pct(len(items), baseline_misses[model])})
        baseline_failure.sort(key=lambda r: (r["model"], -r["missed_fields"]))
        write_csv(out / "baseline_failure_subtypes_nonsubmitted.csv", baseline_failure)

        baseline_mix_rows = []
        mix_totals = Counter()
        for (model, outcome, action), count in baseline_action_mix.items():
            mix_totals[(model, outcome)] += count
        for (model, outcome, action), count in baseline_action_mix.items():
            baseline_mix_rows.append({"model": model, "outcome": outcome, "action_type": action, "action_count": count,
                                      "share_of_actions_pct": pct(count, mix_totals[(model, outcome)])})
        baseline_mix_rows.sort(key=lambda r: (r["model"], r["outcome"], -r["action_count"]))
        write_csv(out / "baseline_action_mix_by_submission.csv", baseline_mix_rows)

        baseline_behavior = []
        for model in sorted({r["model"] for r in baseline_trials}):
            for submitted in (True, False):
                rows = [r for r in baseline_trials if r["model"] == model and r["submit_success"] == submitted]
                if not rows:
                    continue
                baseline_behavior.append({
                    "model": model,
                    "outcome": "submitted" if submitted else "not submitted",
                    "trials": len(rows),
                    "median_actions": round(statistics.median(r["executable_actions"] for r in rows), 2),
                    "median_click_share_pct": round(statistics.median(pct(r["click_actions"], r["executable_actions"]) for r in rows), 2),
                    "median_type_share_pct": round(statistics.median(pct(r["type_actions"], r["executable_actions"]) for r in rows), 2),
                    "median_adjacent_repeat_rate_pct": round(statistics.median(r["adjacent_repeat_rate_pct"] for r in rows if r["adjacent_repeat_rate_pct"] != ""), 2) if any(r["adjacent_repeat_rate_pct"] != "" for r in rows) else "",
                    "perfect_pre_submit_trials": sum(r["perfect_pre_submit"] for r in rows),
                    "postsubmit_zero_artifacts": sum(r["postsubmit_zero_artifact"] for r in rows),
                    "median_final_page_accuracy_pct": round(statistics.median(pct(r["final_verified_correctness"], r["question_total"]) for r in rows), 2),
                    "median_scored_accuracy_pct": round(statistics.median(pct(r["scored_correctness"], r["question_total"]) for r in rows), 2),
                })
        write_csv(out / "baseline_behavior_by_submission.csv", baseline_behavior)

        submission_audit = []
        for model in sorted({r["model"] for r in baseline_trials}):
            rows = [r for r in baseline_trials if r["model"] == model and r["submit_success"]]
            if not rows:
                continue
            pre_available = [r for r in rows if r["score_source"].startswith("pre_")]
            submission_audit.append({
                "model": model,
                "submitted_trials": len(rows),
                "final_page_zero_trials": sum(r["final_verified_correctness"] == 0 for r in rows),
                "pre_submit_score_available": len(pre_available),
                "recovered_nonzero_trials": sum(r["scored_correctness"] > 0 for r in rows),
                "recovered_perfect_trials": sum(r["perfect_pre_submit"] for r in rows),
                "recovered_partial_trials": sum(0 < r["scored_correctness"] < r["question_total"] for r in rows),
                "unrecovered_zero_trials": sum(r["scored_correctness"] == 0 for r in rows),
                "median_final_page_accuracy_pct": round(statistics.median(pct(r["final_verified_correctness"], r["question_total"]) for r in rows), 2),
                "median_recovered_pre_submit_accuracy_pct": round(statistics.median(pct(r["scored_correctness"], r["question_total"]) for r in rows), 2),
            })
        write_csv(out / "baseline_submission_scoring_audit.csv", submission_audit)

    if args.reference_runs_csv and args.reference_actions_csv:
        with args.reference_runs_csv.open(encoding="utf-8") as handle:
            reference_runs = list(csv.DictReader(handle))
        with args.reference_actions_csv.open(encoding="utf-8") as handle:
            reference_actions = list(csv.DictReader(handle))

        action_lookup = defaultdict(Counter)
        for row in reference_actions:
            action_lookup[(row["form_id"], row["answer_run_id"])][row["tool_name"]] += int(row["count"])

        reference_compact = []
        reference_lookup = {}
        for row in reference_runs:
            counts = action_lookup[(row["form_id"], row["answer_run_id"])]
            trace_path = args.project_root / row["trace_path"]
            trace_rows = read_jsonl(trace_path)
            interaction_count = ideal_interaction_count(trace_rows)
            compact = {
                "form_id": row["form_id"],
                "answer_run_id": row["answer_run_id"],
                "usable": row["usable"],
                "submit_success": row["submit_success"],
                "total_trace_events": int(row["action_count"] or 0),
                "interaction_actions": interaction_count,
                "script_operations": counts["browser_run_code"],
                "screenshot_events": counts["browser_take_screenshot"],
                "wait_events": counts["browser_wait_for"],
                "navigation_events": counts["browser_navigate"] + counts["browser_close"],
                "duration_s": round(float(row["duration_s"] or 0), 3),
            }
            reference_compact.append(compact)
            reference_lookup[(row["form_id"], row["answer_run_id"])] = compact
        write_csv(out / "ideal_reference_runs.csv", reference_compact)

        comparison_rows = []
        for row in trial_rows:
            ref = reference_lookup.get((row["form_id"], row["answer_run_id"]))
            if not ref:
                continue
            comparison_rows.append({
                "model": row["model"],
                "form_id": row["form_id"],
                "answer_run_id": row["answer_run_id"],
                "model_full_fill": row["full_fill_success"],
                "model_correctness_pct": row["correctness_pct"],
                "model_actions": row["interaction_action_count"],
                "model_issued_actions": row["action_count_parsed"],
                "model_duration_s": round(row["duration_s"], 3),
                "ideal_total_trace_events": ref["total_trace_events"],
                "ideal_interaction_actions": ref["interaction_actions"],
                "ideal_script_operations": ref["script_operations"],
                "ideal_duration_s": ref["duration_s"],
                "model_to_ideal_interaction_action_ratio": round(row["interaction_action_count"] / ref["interaction_actions"], 3) if ref["interaction_actions"] else "",
                "model_to_ideal_total_event_ratio": round(row["interaction_action_count"] / ref["total_trace_events"], 3) if ref["total_trace_events"] else "",
                "model_to_ideal_script_operation_ratio": round(row["interaction_action_count"] / ref["script_operations"], 3) if ref["script_operations"] else "",
                "model_to_ideal_time_ratio": round(row["duration_s"] / ref["duration_s"], 3) if ref["duration_s"] else "",
            })
        write_csv(out / "model_vs_ideal_by_form.csv", comparison_rows)

        model_ideal_summary = []
        for model in MODEL_ORDER:
            selected = [r for r in comparison_rows if r["model"] == model]
            model_ideal_summary.append({
                "model": model,
                "matched_forms": len(selected),
                "full_fills": sum(r["model_full_fill"] == "true" for r in selected),
                "median_model_actions": round(statistics.median(r["model_actions"] for r in selected), 2),
                "median_ideal_interaction_actions": round(statistics.median(r["ideal_interaction_actions"] for r in selected), 2),
                "median_ideal_total_trace_events": round(statistics.median(r["ideal_total_trace_events"] for r in selected), 2),
                "median_ideal_script_operations": round(statistics.median(r["ideal_script_operations"] for r in selected), 2),
                "median_model_duration_s": round(statistics.median(r["model_duration_s"] for r in selected), 2),
                "median_ideal_duration_s": round(statistics.median(r["ideal_duration_s"] for r in selected), 2),
                "median_model_to_ideal_interaction_action_ratio": round(statistics.median(r["model_to_ideal_interaction_action_ratio"] for r in selected), 3),
                "median_model_to_ideal_total_event_ratio": round(statistics.median(r["model_to_ideal_total_event_ratio"] for r in selected), 3),
                "median_model_to_ideal_script_operation_ratio": round(statistics.median(r["model_to_ideal_script_operation_ratio"] for r in selected), 3),
                "median_model_to_ideal_time_ratio": round(statistics.median(r["model_to_ideal_time_ratio"] for r in selected), 3),
            })
        write_csv(out / "model_vs_ideal_summary.csv", model_ideal_summary)

        reference_summary = [{
            "reference_runs": len(reference_compact),
            "usable_runs": sum(r["usable"] == "True" for r in reference_compact),
            "successful_submissions": sum(r["submit_success"] == "True" for r in reference_compact),
            "forms": len({r["form_id"] for r in reference_compact}),
            "answer_runs_per_form": len({r["answer_run_id"] for r in reference_compact}),
            "median_total_trace_events": round(statistics.median(r["total_trace_events"] for r in reference_compact), 2),
            "median_interaction_actions": round(statistics.median(r["interaction_actions"] for r in reference_compact), 2),
            "median_script_operations": round(statistics.median(r["script_operations"] for r in reference_compact), 2),
            "median_screenshot_events": round(statistics.median(r["screenshot_events"] for r in reference_compact), 2),
            "median_wait_events": round(statistics.median(r["wait_events"] for r in reference_compact), 2),
            "median_duration_s": round(statistics.median(r["duration_s"] for r in reference_compact), 2),
        }]
        write_csv(out / "ideal_reference_summary.csv", reference_summary)
    print(json.dumps(row_counts, indent=2))


if __name__ == "__main__":
    main()
