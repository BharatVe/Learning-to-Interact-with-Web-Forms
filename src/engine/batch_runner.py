import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from engine import runner  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run form datasets sequentially across multiple form specs"
    )
    parser.add_argument(
        "--form-ids",
        help="Comma-separated list of form IDs. Defaults to all specs under specs-root.",
    )
    parser.add_argument("--answers-root", default="data/answers")
    parser.add_argument("--answers-file", default="runs.json")
    parser.add_argument("--dataset-root", default=runner.DEFAULT_DATASET_ROOT)
    parser.add_argument("--specs-root")
    parser.add_argument("--num-runs-per-form", type=int)
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-existing-video", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument(
        "--events-layout",
        choices=["flat", "per-action"],
        default="flat",
        help="Store events as a flat list or attach them to each action entry.",
    )
    parser.add_argument("--omit-hover-events", action="store_true")
    parser.add_argument("--slow-mo", type=int, default=runner.DEFAULT_SLOW_MO)
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    parser.add_argument("--type-delay", type=int, default=runner.DEFAULT_TYPE_DELAY)
    parser.add_argument("--action-delay", type=int, default=runner.DEFAULT_ACTION_DELAY)
    parser.add_argument("--video-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args(argv)
    if args.num_runs_per_form is not None and args.num_runs_per_form < 1:
        parser.error("--num-runs-per-form must be >= 1")
    return args


def discover_form_ids(specs_root: Path) -> List[str]:
    if not specs_root.exists():
        raise FileNotFoundError(f"Specs root not found: {specs_root}")
    form_ids: List[str] = []
    for entry in specs_root.iterdir():
        if entry.is_dir() and (entry / "spec.json").exists():
            form_ids.append(entry.name)
    return sorted(form_ids)


def parse_form_ids(raw: Optional[str], specs_root: Path) -> List[str]:
    if raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    return discover_form_ids(specs_root)


def count_runs(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return 1
    if isinstance(data, dict):
        runs = data.get("runs")
        if runs is None:
            runs = data.get("multi_runs")
        if runs is None and "answers" in data:
            return 1
        if isinstance(runs, list):
            return len(runs)
    raise ValueError(f"Unsupported answers file format: {path}")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    specs_root = Path(args.specs_root) if args.specs_root else (runner.SRC_DIR / "forms")
    answers_root = Path(args.answers_root)
    form_ids = parse_form_ids(args.form_ids, specs_root)

    if not form_ids:
        raise ValueError("No form IDs found to run")

    for form_id in form_ids:
        answers_path = answers_root / form_id / args.answers_file
        if not answers_path.exists():
            message = f"[WARN] Answers file not found for {form_id}: {answers_path}"
            if args.skip_missing:
                print(message, file=sys.stderr)
                continue
            raise FileNotFoundError(message)

        max_runs = count_runs(answers_path)
        run_limit = max_runs
        if args.num_runs_per_form is not None:
            run_limit = min(args.num_runs_per_form, max_runs)
        print(f"[INFO] {form_id}: {run_limit} run(s) of {max_runs} available.")

        if args.dry_run:
            continue

        runner_args: List[str] = [
            "--form-id",
            form_id,
            "--answers",
            str(answers_path),
            "--dataset-root",
            str(args.dataset_root),
            "--specs-root",
            str(specs_root),
            "--slow-mo",
            str(args.slow_mo),
            "--pause-seconds",
            str(args.pause_seconds),
            "--type-delay",
            str(args.type_delay),
            "--action-delay",
            str(args.action_delay),
            "--events-layout",
            args.events_layout,
        ]
        if args.video_dir:
            runner_args.extend(["--video-dir", str(args.video_dir)])
        if args.num_runs_per_form is not None:
            runner_args.extend(["--num-runs", str(run_limit)])
        if args.start_index is not None:
            runner_args.extend(["--start-index", str(args.start_index)])
        if args.resume:
            runner_args.append("--resume")
        if args.skip_existing:
            runner_args.append("--skip-existing")
        if args.skip_existing_video:
            runner_args.append("--skip-existing-video")
        if args.overwrite_existing:
            runner_args.append("--overwrite-existing")
        if args.omit_hover_events:
            runner_args.append("--omit-hover-events")

        completed = runner.main(runner_args)
        if completed is False:
            print("[WARN] Batch run interrupted by user.", file=sys.stderr)
            return


if __name__ == "__main__":
    main()
