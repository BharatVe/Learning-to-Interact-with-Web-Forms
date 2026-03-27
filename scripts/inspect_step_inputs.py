#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _resolve_trial_dir(repo_root: Path, args: argparse.Namespace) -> Path:
    if args.trial_dir:
        trial_dir = (repo_root / args.trial_dir).resolve()
        if not trial_dir.exists():
            raise FileNotFoundError(f"trial_dir not found: {trial_dir}")
        return trial_dir

    if not args.experiment_id or not args.model_id or not args.form_id or not args.run_id:
        raise ValueError("Either --trial-dir or all of --experiment-id --model-id --form-id --run-id are required")

    run_dir = (repo_root / args.dataset_root / args.experiment_id / args.model_id / args.form_id / args.run_id).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    if args.trial_id:
        trial_dir = run_dir / args.trial_id
        if not trial_dir.exists():
            raise FileNotFoundError(f"trial not found: {trial_dir}")
        return trial_dir

    trials = [p for p in run_dir.iterdir() if p.is_dir()]
    if not trials:
        raise FileNotFoundError(f"no trials under: {run_dir}")
    trials.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return trials[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect step_inputs.jsonl and matching model_io step rows.")
    parser.add_argument("--dataset-root", default="data/model_baselines")
    parser.add_argument("--trial-dir")
    parser.add_argument("--experiment-id")
    parser.add_argument("--model-id")
    parser.add_argument("--form-id")
    parser.add_argument("--run-id")
    parser.add_argument("--trial-id")
    parser.add_argument("--steps", default="", help="Comma-separated step indices (e.g., 0,1,4). Empty = first N steps")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--show-full-io", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    trial_dir = _resolve_trial_dir(repo_root, args)
    step_inputs_path = trial_dir / "step_inputs.jsonl"
    model_io_path = trial_dir / "model_io.jsonl"

    step_inputs = _load_jsonl(step_inputs_path)
    model_io = _load_jsonl(model_io_path)
    model_steps = {int(row.get("step_index")): row for row in model_io if row.get("phase") == "step" and isinstance(row.get("step_index"), int)}

    if args.steps.strip():
        wanted = []
        for part in args.steps.split(","):
            part = part.strip()
            if not part:
                continue
            wanted.append(int(part))
    else:
        wanted = [int(row.get("step_index")) for row in step_inputs if isinstance(row.get("step_index"), int)][: max(1, int(args.limit))]

    print(f"trial_dir={trial_dir}")
    print(f"step_inputs_path={step_inputs_path}")
    print(f"model_io_path={model_io_path}")
    print(f"available_step_inputs={len(step_inputs)}")

    step_input_map = {int(row.get("step_index")): row for row in step_inputs if isinstance(row.get("step_index"), int)}

    for step in wanted:
        print("\n" + "=" * 80)
        print(f"STEP {step}")
        step_row = step_input_map.get(step)
        if step_row is None:
            print("step_input: <missing>")
        else:
            print("step_input:")
            print(json.dumps(step_row, indent=2, ensure_ascii=True))

        io_row = model_steps.get(step)
        if io_row is None:
            print("model_output: <missing>")
            continue

        print("model_output:")
        if args.show_full_io:
            print(json.dumps(io_row, indent=2, ensure_ascii=True))
            continue

        compact = {
            "step_index": io_row.get("step_index"),
            "raw_model_output": io_row.get("raw_model_output"),
            "parsed_action": io_row.get("parsed_action"),
            "warnings": io_row.get("warnings"),
            "error": io_row.get("error"),
            "model_inference": io_row.get("model_inference"),
        }
        print(json.dumps(compact, indent=2, ensure_ascii=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
