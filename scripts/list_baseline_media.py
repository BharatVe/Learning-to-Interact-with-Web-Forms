#!/usr/bin/env python3
import argparse
from pathlib import Path


def _sorted_existing(paths):
    existing = [path for path in paths if path.exists()]
    return sorted(existing, key=lambda item: item.stat().st_mtime, reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="List latest reference and baseline media artifacts.")
    parser.add_argument("--experiment-id", default="baseline_mcp_v1")
    parser.add_argument("--model-id")
    parser.add_argument("--form-id", required=True)
    parser.add_argument("--run-id", default="run_0001")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    limit = max(1, int(args.limit))

    reference_root = repo_root / "data" / "forms" / args.form_id / "runs" / args.run_id
    print(f"reference_root={reference_root}")
    reference_videos = _sorted_existing(reference_root.glob("*.webm"))[:limit] if reference_root.exists() else []
    if reference_videos:
        for video in reference_videos:
            print(f"reference_video={video}")
            print(f"reference_annotations={reference_root / 'annotations.json'}")
            print(f"reference_trace={reference_root / 'tool_trace.jsonl'}")
    else:
        print("reference_video=NONE")

    baseline_root = repo_root / "data" / "model_baselines" / args.experiment_id
    if args.model_id:
        baseline_glob = baseline_root / args.model_id / args.form_id / args.run_id / "trial_*" / "*.webm"
    else:
        baseline_glob = baseline_root / "*" / args.form_id / args.run_id / "trial_*" / "*.webm"
    baseline_videos = _sorted_existing(repo_root.glob(str(baseline_glob.relative_to(repo_root))))[:limit]
    if baseline_videos:
        for video in baseline_videos:
            artifact_dir = video.parent
            print(f"baseline_video={video}")
            print(f"baseline_summary={artifact_dir / 'summary.json'}")
            print(f"baseline_annotations={artifact_dir / 'annotations.json'}")
            print(f"baseline_trace={artifact_dir / 'tool_trace.jsonl'}")
            print(f"baseline_model_io={artifact_dir / 'model_io.jsonl'}")
    else:
        print("baseline_video=NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
