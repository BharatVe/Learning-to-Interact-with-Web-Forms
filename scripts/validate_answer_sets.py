#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set


WIDGET_TO_QTYPE = {
    "short_text": "SHORT_TEXT",
    "paragraph_text": "PARAGRAPH",
    "date": "DATE",
    "time": "TIME",
    "single_choice": "SINGLE_CHOICE",
    "multi_choice": "MULTI_CHOICE",
    "dropdown": "DROPDOWN",
}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^\d{2}:\d{2}$")


def _split_options(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [item.strip() for item in text.split(";") if item.strip()]


def _load_question_meta(forms_master: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not forms_master.exists():
        raise FileNotFoundError(f"forms_master not found: {forms_master}")
    by_form: Dict[str, Dict[str, Dict[str, Any]]] = {}
    with forms_master.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_cols = {"form_id", "q_title", "q_type", "required", "options"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"forms_master missing columns: {sorted(missing)}")
        for row in reader:
            form_id = str(row.get("form_id") or "").strip()
            q_title = str(row.get("q_title") or "").strip()
            if not form_id or not q_title:
                continue
            by_form.setdefault(form_id, {})[q_title] = {
                "q_type": str(row.get("q_type") or "").strip().upper(),
                "required": str(row.get("required") or "").strip().lower() in {"1", "true", "t", "yes", "y"},
                "options": _split_options(str(row.get("options") or "")),
            }
    return by_form


def _resolve_form_ids(forms_root: Path, form_ids_arg: str) -> List[str]:
    discovered = sorted(entry.name for entry in forms_root.iterdir() if entry.is_dir() and (entry / "spec.json").exists())
    raw = str(form_ids_arg or "").strip()
    if not raw or raw.lower() == "all":
        return discovered
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    missing = sorted(set(requested) - set(discovered))
    if missing:
        raise ValueError(f"unknown form ids requested: {missing}")
    return requested


def _validate_answer_value(
    form_id: str,
    run_idx: int,
    label: str,
    answer: Dict[str, Any],
    q_meta: Dict[str, Any],
    errors: List[str],
) -> None:
    expected_q_type = str(q_meta.get("q_type") or "")
    widget_type = str(answer.get("widget_type") or "").strip()
    value = answer.get("value")
    mapped_q_type = WIDGET_TO_QTYPE.get(widget_type)
    if mapped_q_type != expected_q_type:
        errors.append(
            f"{form_id} run_{run_idx:04d} question '{label}': widget_type={widget_type!r} does not match q_type={expected_q_type!r}"
        )
        return
    if expected_q_type in {"SHORT_TEXT", "PARAGRAPH"}:
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{form_id} run_{run_idx:04d} question '{label}': text value must be non-empty string")
    elif expected_q_type == "DATE":
        if not isinstance(value, str) or not DATE_RE.match(value):
            errors.append(f"{form_id} run_{run_idx:04d} question '{label}': date must be YYYY-MM-DD")
    elif expected_q_type == "TIME":
        if not isinstance(value, str) or not TIME_RE.match(value):
            errors.append(f"{form_id} run_{run_idx:04d} question '{label}': time must be HH:MM")
    elif expected_q_type in {"SINGLE_CHOICE", "DROPDOWN"}:
        options = q_meta.get("options") or []
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{form_id} run_{run_idx:04d} question '{label}': {expected_q_type.lower()} must be non-empty string")
        elif options and value not in options:
            errors.append(f"{form_id} run_{run_idx:04d} question '{label}': value '{value}' not in options")
    elif expected_q_type == "MULTI_CHOICE":
        options = q_meta.get("options") or []
        if not isinstance(value, list) or not value:
            errors.append(f"{form_id} run_{run_idx:04d} question '{label}': multi choice must be non-empty list")
        else:
            for item in value:
                if not isinstance(item, str) or not item.strip():
                    errors.append(f"{form_id} run_{run_idx:04d} question '{label}': multi choice items must be non-empty strings")
                    break
                if options and item not in options:
                    errors.append(f"{form_id} run_{run_idx:04d} question '{label}': value '{item}' not in options")
                    break


def _validate_form_runs(
    form_id: str,
    answers_root: Path,
    q_meta_by_label: Dict[str, Dict[str, Any]],
    required_runs: int,
    required_run_indexes: List[int],
    strict: bool,
    strict_exact_run_count: bool,
    errors: List[str],
) -> None:
    runs_path = answers_root / form_id / "runs.json"
    if not runs_path.exists():
        errors.append(f"{form_id}: missing runs.json at {runs_path}")
        return
    try:
        payload = json.loads(runs_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{form_id}: invalid JSON in {runs_path}: {exc}")
        return
    runs = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(runs, list):
        errors.append(f"{form_id}: 'runs' is missing or invalid")
        return

    if strict and strict_exact_run_count and len(runs) != required_runs:
        errors.append(f"{form_id}: expected exactly {required_runs} runs, found {len(runs)}")
    elif len(runs) < required_runs:
        errors.append(f"{form_id}: expected at least {required_runs} runs, found {len(runs)}")

    for req_idx in required_run_indexes:
        if req_idx < 1 or req_idx > len(runs):
            errors.append(f"{form_id}: required run index {req_idx} missing (total runs={len(runs)})")

    required_labels: Set[str] = {label for label, meta in q_meta_by_label.items() if bool(meta.get("required"))}
    all_labels: Set[str] = set(q_meta_by_label.keys())
    for run_idx, run in enumerate(runs, start=1):
        answers = run.get("answers") if isinstance(run, dict) else None
        if not isinstance(answers, list):
            errors.append(f"{form_id} run_{run_idx:04d}: missing/non-list answers")
            continue
        seen_labels: Set[str] = set()
        for answer_idx, answer in enumerate(answers):
            if not isinstance(answer, dict):
                errors.append(f"{form_id} run_{run_idx:04d} answer[{answer_idx}]: answer must be object")
                continue
            for key in ("label", "widget_type", "value"):
                if key not in answer:
                    errors.append(f"{form_id} run_{run_idx:04d} answer[{answer_idx}]: missing key '{key}'")
            label = str(answer.get("label") or "").strip()
            if not label:
                errors.append(f"{form_id} run_{run_idx:04d} answer[{answer_idx}]: empty label")
                continue
            if label in seen_labels:
                errors.append(f"{form_id} run_{run_idx:04d}: duplicate answer label '{label}'")
                continue
            seen_labels.add(label)
            if label not in q_meta_by_label:
                errors.append(f"{form_id} run_{run_idx:04d}: answer label '{label}' not found in forms_master")
                continue
            _validate_answer_value(form_id, run_idx, label, answer, q_meta_by_label[label], errors)

        missing_required = sorted(required_labels - seen_labels)
        if missing_required:
            errors.append(f"{form_id} run_{run_idx:04d}: missing required labels {missing_required}")
        if strict:
            missing_any = sorted(all_labels - seen_labels)
            extra_any = sorted(seen_labels - all_labels)
            if missing_any:
                errors.append(f"{form_id} run_{run_idx:04d}: strict mode missing labels {missing_any}")
            if extra_any:
                errors.append(f"{form_id} run_{run_idx:04d}: strict mode unexpected labels {extra_any}")


def _parse_required_indexes(raw: str) -> List[int]:
    if not raw:
        return []
    out: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return sorted(set(out))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated answer sets for baseline eval.")
    parser.add_argument("--forms-root", default="src/forms")
    parser.add_argument("--answers-root", default="data/answers")
    parser.add_argument("--forms-master", default="data/specs/forms_master.csv")
    parser.add_argument("--required-runs", type=int, default=10)
    parser.add_argument("--required-run-indexes", default="")
    parser.add_argument("--form-ids", default="")
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument(
        "--strict-exact-run-count",
        action="store_true",
        default=False,
        help="When set with --strict, require len(runs) == --required-runs. Default strict behavior allows supersets.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    forms_root = (repo_root / args.forms_root).resolve()
    answers_root = (repo_root / args.answers_root).resolve()
    forms_master = (repo_root / args.forms_master).resolve()
    required_indexes = _parse_required_indexes(args.required_run_indexes)

    if args.required_runs <= 0:
        raise ValueError("--required-runs must be positive")
    q_meta = _load_question_meta(forms_master)
    form_ids = _resolve_form_ids(forms_root, args.form_ids)
    errors: List[str] = []
    for form_id in form_ids:
        if form_id not in q_meta:
            errors.append(f"{form_id}: no question metadata found in {forms_master}")
            continue
        _validate_form_runs(
            form_id=form_id,
            answers_root=answers_root,
            q_meta_by_label=q_meta[form_id],
            required_runs=int(args.required_runs),
            required_run_indexes=required_indexes,
            strict=bool(args.strict),
            strict_exact_run_count=bool(args.strict_exact_run_count),
            errors=errors,
        )

    print(
        f"[INFO] forms_checked={len(form_ids)} required_runs={args.required_runs} "
        f"strict={args.strict} strict_exact_run_count={bool(args.strict_exact_run_count)}"
    )
    if errors:
        print(f"[FAIL] validation errors={len(errors)}")
        for item in errors:
            print(f" - {item}")
        return 1
    print("[PASS] answer-set validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
