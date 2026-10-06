"""Export the four matched fill-only additions into compact, trackable CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

EXPERIMENT_ID = "qwen3vl_direct_mcp_fill_only_50_r2_20260727"
MODEL_ID = "vlm_qwen3_vl_30b_a3b_instruct"
ANSWER_RUN_ID = "run_0002"
FORMS = ("usability_test", "volunteer_shift", "wellbeing_check", "workshop_signup")

TRIAL_COLUMNS = (
    "experiment_id", "model_id", "form_id", "answer_run_id", "trial_id",
    "run_started_utc", "run_completed_utc", "task_mode", "stop_reason",
    "success", "submit_success", "question_total", "scored_correctness",
    "scored_correctness_source", "verified_count", "verified_correctness",
    "action_count", "trace_action_count", "duration_s", "source_summary_path",
    "source_summary_sha256", "source_annotations_path", "source_annotations_sha256",
)
FIELD_COLUMNS = (
    "experiment_id", "model_id", "form_id", "answer_run_id", "trial_id",
    "question_index", "question_id", "label", "widget_type", "expected_value",
    "actual_value", "attempted", "attempted_correct", "verified",
    "verified_correct", "final_status", "source_annotations_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_value(value: object) -> object:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def render_csv(columns: tuple[str, ...], rows: list[dict[str, object]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def single_artifact(run_root: Path, filename: str) -> Path:
    matches = sorted(run_root.glob(f"trial_*/{filename}"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {filename} under {run_root}, found {len(matches)}")
    return matches[0]


def build_exports(project_root: Path) -> dict[str, str]:
    experiment_root = project_root / "data" / "model_baselines" / EXPERIMENT_ID / MODEL_ID
    trial_rows: list[dict[str, object]] = []
    field_rows: list[dict[str, object]] = []
    for form_id in FORMS:
        run_root = experiment_root / form_id / ANSWER_RUN_ID
        summary_path = single_artifact(run_root, "summary.json")
        annotations_path = single_artifact(run_root, "annotations.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
        expected = (EXPERIMENT_ID, MODEL_ID, form_id, ANSWER_RUN_ID)
        actual = tuple(summary.get(key) for key in ("experiment_id", "model_id", "form_id", "answer_run_id"))
        if actual != expected or summary.get("task_mode") != "fill_only_done":
            raise RuntimeError(f"unexpected trial identity or mode in {summary_path}")

        summary_hash = sha256(summary_path)
        annotations_hash = sha256(annotations_path)
        trial_row = {key: summary.get(key) for key in TRIAL_COLUMNS if not key.startswith("source_")}
        trial_row.update({
            "source_summary_path": summary_path.relative_to(project_root).as_posix(),
            "source_summary_sha256": summary_hash,
            "source_annotations_path": annotations_path.relative_to(project_root).as_posix(),
            "source_annotations_sha256": annotations_hash,
        })
        trial_rows.append(trial_row)

        questions = annotations.get("questions")
        if not isinstance(questions, list) or len(questions) != summary.get("question_total"):
            raise RuntimeError(f"incomplete question records in {annotations_path}")
        for index, question in enumerate(questions, start=1):
            field_rows.append({
                "experiment_id": EXPERIMENT_ID, "model_id": MODEL_ID,
                "form_id": form_id, "answer_run_id": ANSWER_RUN_ID,
                "trial_id": summary["trial_id"], "question_index": index,
                "question_id": question.get("question_id"), "label": question.get("label"),
                "widget_type": question.get("widget_type"),
                "expected_value": csv_value(question.get("value")),
                "actual_value": csv_value(question.get("actual_value")),
                "attempted": question.get("attempted"),
                "attempted_correct": question.get("attempted_correct"),
                "verified": question.get("verified"),
                "verified_correct": question.get("verified_correct"),
                "final_status": question.get("final_status"),
                "source_annotations_sha256": annotations_hash,
            })
    return {
        "trial_summary.csv": render_csv(TRIAL_COLUMNS, trial_rows),
        "field_results.csv": render_csv(FIELD_COLUMNS, field_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = project_root / "evaluation_additions" / "missing_fill_only_runs"
    exports = build_exports(project_root)
    if args.check:
        stale = [name for name, content in exports.items() if not (output_dir / name).is_file() or (output_dir / name).read_text(encoding="utf-8") != content]
        if stale:
            raise SystemExit(f"missing or stale exports: {', '.join(stale)}")
        print("validated missing fill-only exports: 4 trials")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in exports.items():
        (output_dir / name).write_text(content, encoding="utf-8")
    print("wrote missing fill-only exports: 4 trials")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
