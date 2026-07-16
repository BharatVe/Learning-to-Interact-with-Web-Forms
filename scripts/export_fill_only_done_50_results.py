#!/usr/bin/env python3
"""Export the compact 50-form fill-only/DONE comparison for Git/GitHub."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "model_baselines"
OUTPUT = ROOT / "data" / "model_baseline_exports" / "fill_only_done_50_20260714"
MANIFEST = ROOT / "configs" / "baselines" / "fill_only_done_50_completion_20260713.json"

MODEL_LABELS = {
    "computer_use_gemini_35_flash_lowcost": "Gemini 3.5 Flash",
    "computer_use_opencua_32b_direct_mcp": "OpenCUA direct-MCP",
    "text_qwen3_30b_a3b_instruct_2507": "Qwen3 Text",
    "vlm_qwen3_vl_30b_a3b_instruct": "Qwen3-VL",
}

# Later experiments override an earlier artifact for the same model/form/run.
EXPERIMENTS = [
    "gemini_35_flash_fill_only_done_30_seed20260709_r2_step32",
    "gemini_35_flash_fill_only_done_50_completion_20260713_r2_step32",
    "opencua_direct_mcp_fill_only_done_30_seed20260709_r2_step32",
    "opencua_direct_mcp_fill_only_done_50_topup20_20260713_r2_step32",
    "qwen_direct_mcp_fill_only_done_50_20260713_r2_step32",
    "qwen_vlm_fill_only_done_50_topup11_20260714_r2_step32",
    "qwen_vlm_fill_only_done_50_history2_topup4_20260715_r2_step32",
]

INFRA_STOP_REASONS = {"provider_capacity_error", "model_inference_failed"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summaries(experiment: str) -> Iterable[tuple[Path, dict[str, Any]]]:
    root = RAW_ROOT / experiment
    if not root.exists():
        return
    for path in sorted(root.glob("**/summary.json")):
        yield path, load_json(path)


def is_usable(summary: dict[str, Any]) -> bool:
    stop_reason = str(summary.get("stop_reason") or "")
    failure_category = str(summary.get("failure_category") or "")
    if stop_reason in INFRA_STOP_REASONS or stop_reason == "environment_error":
        return False
    return failure_category not in {"environment_error", "provider_capacity_error"}


def canonical_rows() -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for experiment in EXPERIMENTS:
        for path, summary in summaries(experiment):
            model_id = str(summary.get("model_id") or "")
            if model_id not in MODEL_LABELS:
                continue
            key = (
                model_id,
                str(summary.get("form_id") or ""),
                str(summary.get("answer_run_id") or ""),
            )
            row = dict(summary)
            row["source_summary"] = str(path.relative_to(ROOT))
            row["usable"] = is_usable(summary)
            # Do not replace usable evidence with an unusable retry artifact.
            if key in selected and selected[key]["usable"] and not row["usable"]:
                continue
            selected[key] = row
    return sorted(selected.values(), key=lambda row: (MODEL_LABELS[row["model_id"]], row["form_id"]))


def csv_row(row: dict[str, Any]) -> dict[str, Any]:
    correct = int(row.get("verified_correctness") or 0)
    total = int(row.get("question_total") or 0)
    token_usage = row.get("token_usage") or {}
    cost = row.get("cost_estimate") or {}
    return {
        "model": MODEL_LABELS[row["model_id"]],
        "model_id": row["model_id"],
        "form_id": row.get("form_id", ""),
        "answer_run_id": row.get("answer_run_id", ""),
        "experiment_id": row.get("experiment_id", ""),
        "trial_id": row.get("trial_id", ""),
        "usable": str(bool(row.get("usable"))).lower(),
        "full_fill_success": str(bool(row.get("success"))).lower(),
        "stop_reason": row.get("stop_reason", ""),
        "verified_correctness": correct,
        "question_total": total,
        "correctness_pct": round(100 * correct / total, 2) if total else "",
        "action_count": int(row.get("action_count") or row.get("tool_call_count") or 0),
        "duration_s": round(float(row.get("duration_s") or 0), 3),
        "total_tokens": int(token_usage.get("total_tokens") or 0),
        "estimated_usd_lower_bound": cost.get("estimated_usd", ""),
        "failure_category": row.get("failure_category", ""),
        "failure_detail": row.get("failure_detail", ""),
        "source_summary": row.get("source_summary", ""),
    }


def aggregate(rows: list[dict[str, Any]], all_forms: set[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model_id, label in MODEL_LABELS.items():
        model_rows = [row for row in rows if row["model_id"] == model_id]
        usable = [row for row in model_rows if row["usable"]]
        correct = sum(int(row.get("verified_correctness") or 0) for row in usable)
        questions = sum(int(row.get("question_total") or 0) for row in usable)
        actions = sum(int(row.get("action_count") or row.get("tool_call_count") or 0) for row in usable)
        completed_forms = {str(row.get("form_id") or "") for row in usable}
        output.append(
            {
                "model": label,
                "model_id": model_id,
                "usable_forms": len(completed_forms),
                "target_forms": len(all_forms),
                "missing_forms": sorted(all_forms - completed_forms),
                "full_fill_successes": sum(bool(row.get("success")) for row in usable),
                "verified_correctness": correct,
                "question_total": questions,
                "correctness_pct": round(100 * correct / questions, 2) if questions else 0,
                "model_actions": actions,
                "average_actions": round(actions / len(usable), 2) if usable else 0,
                "total_tokens": sum(int((row.get("token_usage") or {}).get("total_tokens") or 0) for row in usable),
                "estimated_usd_lower_bound": round(
                    sum(float((row.get("cost_estimate") or {}).get("estimated_usd") or 0) for row in usable), 6
                ),
                "stop_reasons": dict(sorted(Counter(str(row.get("stop_reason") or "") for row in usable).items())),
                "unusable_artifacts": sum(not row["usable"] for row in model_rows),
            }
        )
    return output


def write_readme(aggregates: list[dict[str, Any]]) -> None:
    lines = [
        "# Fill-only/DONE 50-form comparison",
        "",
        "Generated from compact per-trial `summary.json` artifacts. Raw screenshots, videos, and model traces are intentionally excluded from Git.",
        "",
        "No form was submitted. `full_fill_successes` means the visible fields were verified as correct before returning DONE.",
        "",
        "| Model | Usable forms | Full fills | Field correctness | Model actions |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        lines.append(
            f"| {item['model']} | {item['usable_forms']}/{item['target_forms']} | "
            f"{item['full_fill_successes']}/{item['usable_forms']} | "
            f"{item['verified_correctness']}/{item['question_total']} ({item['correctness_pct']:.2f}%) | "
            f"{item['model_actions']} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `trials.csv`: one row per canonical model/form trial, including usability and source artifact path.",
            "- `aggregate.json`: aggregate metrics, stop-reason counts, and missing-form lists.",
            "",
            "Recreate this export with:",
            "",
            "```bash",
            ".venv/bin/python scripts/export_fill_only_done_50_results.py",
            "```",
            "",
        ]
    )
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    all_forms = set(load_json(MANIFEST)["forms"])
    rows = canonical_rows()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_rows = [csv_row(row) for row in rows]
    with (OUTPUT / "trials.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    aggregates = aggregate(rows, all_forms)
    (OUTPUT / "aggregate.json").write_text(json.dumps(aggregates, indent=2) + "\n", encoding="utf-8")
    write_readme(aggregates)
    print(f"Wrote {len(csv_rows)} trial rows to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
