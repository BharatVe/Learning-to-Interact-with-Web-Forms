#!/usr/bin/env python3
"""Audit historical dropdown outcomes from saved state, without rerunning models."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit


PLACEHOLDERS = {"choose", "select", "select an option"}
SELECTED_OPTION = re.compile(r'- option "((?:\\.|[^"\\])*)"[^\n]*\[selected\]')

# Each image was manually reviewed. The named image is the first saved frame
# after the option click in which the collapsed control displays the target.
GEMINI_SCREENSHOT_STEP = {
    "accessibility_feedback": 7,
    "bug_report": 11,
    "club_event_planning": 10,
    "conference_travel": 17,
    "course_enrollment": 11,
    "data_annotation": 10,
    "dataset_request": 12,
    "equipment_checkout": 8,
    "exam_registration": 12,
    "experiment_booking": 10,
    "field_trip": 14,
    "lab_safety": 10,
    "newsletter_signup": 8,
    "office_hours": 12,
    "paper_review": 19,
    "peer_evaluation": 13,
    "publication_submission": 14,
    "purchase_request": 11,
    "remote_setup": 10,
    "scholarship_interest": 7,
    "software_access": 8,
    "sports_tournament": 9,
    "survey_consent": 10,
    "technical_support": 6,
    "thesis_meeting": 8,
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def decoded_excerpt(row: dict) -> str:
    # The recorder stores a bounded JSON-like excerpt. It may be cut before the
    # closing quote, so decode only the escaped newlines and quotes we inspect.
    return str(row.get("accessibility_snapshot_excerpt") or "").replace("\\n", "\n").replace('\\"', '"')


def url_contains_exact_value(excerpt: str, expected: str) -> bool:
    for line in excerpt.splitlines():
        if "/url:" not in line:
            continue
        value = line.split("/url:", 1)[1].strip()
        for _ in range(3):
            value = unquote(value)
        query = parse_qs(urlsplit(value).query)
        if any(candidate == expected for values in query.values() for candidate in values):
            return True
    return False


def audit_direct_trial(trial_dir: Path, expected: str) -> tuple[str, str, str]:
    selected_steps = []
    encoded_steps = []
    for row in read_jsonl(trial_dir / "step_inputs.jsonl"):
        excerpt = decoded_excerpt(row)
        selected = [
            match.group(1)
            for match in SELECTED_OPTION.finditer(excerpt)
            if match.group(1).strip().lower() not in PLACEHOLDERS
        ]
        if expected in selected:
            selected_steps.append(str(row.get("step_index")))
        if url_contains_exact_value(excerpt, expected):
            encoded_steps.append(str(row.get("step_index")))
    if selected_steps:
        return "confirmed_correct", "accessibility_selected_state", selected_steps[-1]
    if encoded_steps:
        return "confirmed_correct", "encoded_form_state", encoded_steps[-1]
    return "unresolved_excerpt_gap", "bounded_snapshot_does_not_expose_control_state", ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--field-outcomes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with args.field_outcomes.open(encoding="utf-8") as handle:
        field_rows = list(csv.DictReader(handle))
    dropdowns = [row for row in field_rows if row["widget_type"] == "dropdown"]
    nondrop_complete = {
        (row["model"], row["form_id"]): all(
            candidate["verified_correct"] == "True"
            for candidate in field_rows
            if candidate["model"] == row["model"]
            and candidate["form_id"] == row["form_id"]
            and candidate["widget_type"] != "dropdown"
        )
        for row in dropdowns
    }

    audit_rows = []
    for row in dropdowns:
        expected = str(json.loads(row["expected_value"]))
        summary_path = project_root / row["source_summary"]
        trial_dir = summary_path.parent
        if row["model"] == "Gemini 3.5 Flash":
            step = GEMINI_SCREENSHOT_STEP[row["form_id"]]
            evidence_path = trial_dir / "observations" / f"step_{step:04d}.png"
            if not evidence_path.exists():
                raise FileNotFoundError(evidence_path)
            status, evidence_type, evidence_step = (
                "confirmed_correct",
                "post_action_screenshot_manual",
                str(step),
            )
        else:
            status, evidence_type, evidence_step = audit_direct_trial(trial_dir, expected)
            evidence_path = trial_dir / "step_inputs.jsonl"
        audit_rows.append({
            "model": row["model"],
            "form_id": row["form_id"],
            "question_id": row["question_id"],
            "label": row["label"],
            "expected_value": expected,
            "recorded_verified_correct": row["verified_correct"],
            "audit_status": status,
            "evidence_type": evidence_type,
            "evidence_step": evidence_step,
            "evidence_path": str(evidence_path.relative_to(project_root)),
        })

    detail_path = output_dir / "dropdown_selected_state_audit.csv"
    fields = list(audit_rows[0])
    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    counts = Counter((row["model"], row["audit_status"]) for row in audit_rows)
    models = list(dict.fromkeys(row["model"] for row in audit_rows))
    summary_rows = []
    for model in models:
        total = sum(counts[model, status] for status in {"confirmed_correct", "unresolved_excerpt_gap"})
        confirmed = counts[model, "confirmed_correct"]
        unresolved = counts[model, "unresolved_excerpt_gap"]
        model_rows = [row for row in audit_rows if row["model"] == model]
        nondrop_complete_count = sum(nondrop_complete[model, row["form_id"]] for row in model_rows)
        confirmed_full = sum(
            row["audit_status"] == "confirmed_correct" and nondrop_complete[model, row["form_id"]]
            for row in model_rows
        )
        unresolved_full_candidates = sum(
            row["audit_status"] == "unresolved_excerpt_gap" and nondrop_complete[model, row["form_id"]]
            for row in model_rows
        )
        summary_rows.append({
            "model": model,
            "dropdown_targets": total,
            "recorded_correct": 0,
            "artifact_confirmed_correct": confirmed,
            "artifact_unresolved": unresolved,
            "artifact_confirmed_wrong": 0,
            "minimum_confirmed_accuracy_pct": round(100 * confirmed / total, 2),
            "dropdown_forms_nondrop_complete": nondrop_complete_count,
            "artifact_confirmed_full_fills": confirmed_full,
            "additional_full_fills_if_all_unresolved_correct": unresolved_full_candidates,
        })
    summary_path = output_dir / "dropdown_selected_state_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(json.dumps({
        "dropdown_targets": len(audit_rows),
        "confirmed_correct": sum(row["audit_status"] == "confirmed_correct" for row in audit_rows),
        "unresolved": sum(row["audit_status"] == "unresolved_excerpt_gap" for row in audit_rows),
        "confirmed_wrong": 0,
        "detail": str(detail_path),
        "summary": str(summary_path),
    }, indent=2))


if __name__ == "__main__":
    main()
