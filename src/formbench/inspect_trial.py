"""`formbench inspect`: one trial at a glance (replaces inspect_step_inputs.py / list_baseline_media.py)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from formbench.common import FormbenchError
from formbench.settings import Settings

SUMMARY_FIELDS = [
    "experiment_id", "model_id", "form_id", "answer_run_id", "trial_id", "track", "task_mode", "success",
    "submit_success", "stop_reason", "failure_category", "failure_detail", "question_total", "verified_correctness",
    "scored_correctness", "action_count", "reference_action_count", "action_overhead_ratio", "duration_s",
    "time_overhead_ratio",
]


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def resolve_trial(settings: Settings, value: str) -> Path:
    candidates = [Path(value), settings.root / value, settings.dataset_root / value]
    for path in candidates:
        if path.is_dir():
            path = path.resolve()
            if (path / "summary.json").is_file() or (path / "annotations.json").is_file():
                return path
            trials = sorted((p for p in path.glob("trial_*") if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
            if trials:
                return trials[0]
    raise FormbenchError(f"trial not found: {value}", hint="pass a trial dir, or <experiment>/<model>/<form>/run_XXXX[/trial_id] under data/model_baselines")


def _steps_wanted(spec: Optional[str], available: List[int]) -> List[int]:
    if not spec or spec == "first":
        return available[:3]
    if spec == "all":
        return available
    return [int(x) for x in spec.split(",") if x.strip()]


def inspect(settings: Settings, trial: str, steps: Optional[str] = None, media: bool = False) -> int:
    trial_dir = resolve_trial(settings, trial)
    print(f"trial_dir: {trial_dir}")
    summary_path = trial_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    width = max(len(k) for k in SUMMARY_FIELDS)
    for key in SUMMARY_FIELDS:
        if key in summary:
            print(f"  {key.ljust(width)}  {summary[key]}")

    if steps is not None:
        step_inputs = {int(r["step_index"]): r for r in _jsonl(trial_dir / "step_inputs.jsonl") if isinstance(r.get("step_index"), int)}
        model_io = {int(r["step_index"]): r for r in _jsonl(trial_dir / "model_io.jsonl") if r.get("phase") == "step" and isinstance(r.get("step_index"), int)}
        for step in _steps_wanted(steps, sorted(set(step_inputs) | set(model_io))):
            print("\n" + "=" * 80 + f"\nSTEP {step}")
            print("input:", json.dumps(step_inputs.get(step, "<missing>"), indent=2, ensure_ascii=True))
            io = model_io.get(step)
            compact = {k: io.get(k) for k in ("raw_model_output", "parsed_action", "warnings", "error", "model_inference")} if io else "<missing>"
            print("output:", json.dumps(compact, indent=2, ensure_ascii=True))

    if media:
        print("\nmedia:")
        for path in sorted(trial_dir.glob("*.webm")) + sorted(trial_dir.glob("*.png")):
            print(f"  {path}")
        shots = sorted((trial_dir / "observations").glob("*.png"))
        if shots:
            print(f"  observations/: {len(shots)} screenshots ({shots[0].name} .. {shots[-1].name})")
        form_id = str(summary.get("form_id") or trial_dir.parents[1].name)
        run_id = str(summary.get("answer_run_id") or trial_dir.parent.name)
        ref = settings.reference_root / form_id / "runs" / run_id
        print(f"\nideal reference run: {ref if ref.is_dir() else 'none'}")
        for path in sorted(ref.glob("*.webm")) + [ref / "tool_trace.jsonl", ref / "annotations.json"]:
            if path.exists():
                print(f"  {path}")
    return 0
