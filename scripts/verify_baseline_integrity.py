#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_supported_actions(contract_path: Path) -> List[str]:
    namespace: Dict[str, object] = {}
    code = contract_path.read_text(encoding="utf-8")
    exec(compile(code, str(contract_path), "exec"), namespace)
    fn = namespace.get("supported_actions")
    if not callable(fn):
        raise RuntimeError("supported_actions() not found in trace_contract.py")
    actions = fn()
    if not isinstance(actions, list):
        raise RuntimeError("supported_actions() must return a list")
    return [str(x) for x in actions]


def validate_answers_runs(repo_root: Path, form_ids: List[str], errors: List[str]) -> None:
    answers_root = repo_root / "data" / "answers"
    for form_id in form_ids:
        runs_path = answers_root / form_id / "runs.json"
        if not runs_path.exists():
            errors.append(f"Missing answers file: {runs_path}")
            continue
        try:
            payload = json.loads(runs_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"Invalid JSON in {runs_path}: {exc}")
            continue
        runs = payload.get("runs") if isinstance(payload, dict) else None
        if not isinstance(runs, list) or not runs:
            errors.append(f"'runs' missing/invalid/empty in {runs_path}")
            continue
        for i, run in enumerate(runs):
            answers = run.get("answers") if isinstance(run, dict) else None
            if not isinstance(answers, list):
                errors.append(f"{runs_path}: run index {i} has missing/non-list 'answers'")
                break
            for j, answer in enumerate(answers):
                if not isinstance(answer, dict):
                    errors.append(f"{runs_path}: run {i} answer {j} is not an object")
                    continue
                for key in ("label", "widget_type", "value"):
                    if key not in answer:
                        errors.append(f"{runs_path}: run {i} answer {j} missing key '{key}'")


def validate_forms_master(repo_root: Path, form_ids: List[str], errors: List[str]) -> None:
    csv_path = repo_root / "data" / "specs" / "forms_master.csv"
    if not csv_path.exists():
        errors.append(f"Missing forms_master: {csv_path}")
        return
    csv_form_ids: List[str] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "form_id" not in (reader.fieldnames or []):
            errors.append(f"forms_master missing 'form_id' column: {csv_path}")
            return
        for row in reader:
            fid = (row.get("form_id") or "").strip()
            if fid:
                csv_form_ids.append(fid)
    missing_specs = sorted(set(csv_form_ids) - set(form_ids))
    missing_csv = sorted(set(form_ids) - set(csv_form_ids))
    if missing_specs:
        errors.append(f"forms_master references missing specs: {missing_specs}")
    if missing_csv:
        errors.append(f"specs missing in forms_master: {missing_csv}")


def validate_existing_artifacts(repo_root: Path, supported_actions: List[str], errors: List[str]) -> Tuple[int, int]:
    runs_root = repo_root / "data" / "forms"
    run_dirs = sorted(runs_root.glob("*/runs/run_*"))
    for run_dir in run_dirs:
        for name in ("annotations.json", "answers_instance.json", "tool_trace.jsonl"):
            if not (run_dir / name).exists():
                errors.append(f"Missing artifact in {run_dir}: {name}")
        trace_path = run_dir / "tool_trace.jsonl"
        if not trace_path.exists():
            continue
        for line_num, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                errors.append(f"Invalid trace JSON {trace_path}:{line_num}: {exc}")
                continue
            action_name = obj.get("name")
            if not isinstance(action_name, str):
                errors.append(f"Missing action name {trace_path}:{line_num}")
                continue
            if action_name not in supported_actions:
                errors.append(f"Unsupported action '{action_name}' at {trace_path}:{line_num}")
    return len(run_dirs), len(supported_actions)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    errors: List[str] = []

    forms_root = repo_root / "src" / "forms"
    form_ids = sorted([p.name for p in forms_root.iterdir() if (p / "spec.json").exists()])
    if not form_ids:
        errors.append(f"No form specs found under {forms_root}")

    validate_answers_runs(repo_root, form_ids, errors)
    validate_forms_master(repo_root, form_ids, errors)
    supported_actions = load_supported_actions(repo_root / "src" / "engine" / "trace_contract.py")
    run_count, action_count = validate_existing_artifacts(repo_root, supported_actions, errors)

    print(f"[INFO] forms: {len(form_ids)}")
    print(f"[INFO] supported actions: {action_count}")
    print(f"[INFO] existing run directories: {run_count}")
    if errors:
        print(f"[FAIL] integrity errors: {len(errors)}")
        for item in errors:
            print(f" - {item}")
        return 1
    print("[PASS] integrity checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
