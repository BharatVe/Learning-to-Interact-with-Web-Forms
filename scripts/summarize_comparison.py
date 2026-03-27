#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Tuple


def _load_manifest_rows(manifest_path: Path) -> List[Dict[str, Any]]:
    if not manifest_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
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


def _load_summary(summary_path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summaries: List[Dict[str, Any]] = []
    for row in rows:
        path_value = row.get("summary_path")
        if not isinstance(path_value, str) or not path_value.strip():
            continue
        summary = _load_summary(Path(path_value))
        if summary:
            summaries.append(summary)

    runs = len(summaries)
    verified_correctness = sum(int(item.get("verified_correctness") or 0) for item in summaries)
    question_total = sum(int(item.get("question_total") or 0) for item in summaries)
    submit_successes = sum(1 for item in summaries if bool(item.get("submit_success")))
    failures = sum(1 for item in summaries if not bool(item.get("success")))
    durations = [float(item.get("duration_s")) for item in summaries if item.get("duration_s") is not None]

    verified_rate = (verified_correctness / question_total) if question_total else 0.0
    submit_rate = (submit_successes / runs) if runs else 0.0
    failure_rate = (failures / runs) if runs else 0.0
    median_duration = float(median(durations)) if durations else None

    return {
        "runs": runs,
        "question_total": question_total,
        "verified_correctness": verified_correctness,
        "verified_correctness_rate": round(verified_rate, 6),
        "submit_successes": submit_successes,
        "submit_success_rate": round(submit_rate, 6),
        "failures": failures,
        "failure_rate": round(failure_rate, 6),
        "median_duration_s": median_duration,
    }


def choose_winner(mediated: Dict[str, Any], direct: Dict[str, Any]) -> Tuple[str, str]:
    m_v = float(mediated.get("verified_correctness_rate") or 0.0)
    d_v = float(direct.get("verified_correctness_rate") or 0.0)
    if d_v > m_v:
        return "direct_api_tool_use", "higher verified_correctness_rate"
    if m_v > d_v:
        return "mediated", "higher verified_correctness_rate"

    m_s = float(mediated.get("submit_success_rate") or 0.0)
    d_s = float(direct.get("submit_success_rate") or 0.0)
    if d_s > m_s:
        return "direct_api_tool_use", "tie-break: higher submit_success_rate"
    if m_s > d_s:
        return "mediated", "tie-break: higher submit_success_rate"

    m_f = float(mediated.get("failure_rate") or 0.0)
    d_f = float(direct.get("failure_rate") or 0.0)
    if d_f < m_f:
        return "direct_api_tool_use", "tie-break: lower failure_rate"
    if m_f < d_f:
        return "mediated", "tie-break: lower failure_rate"

    m_d = mediated.get("median_duration_s")
    d_d = direct.get("median_duration_s")
    if isinstance(d_d, (int, float)) and isinstance(m_d, (int, float)):
        if d_d < m_d:
            return "direct_api_tool_use", "tie-break: lower median_duration_s"
        if m_d < d_d:
            return "mediated", "tie-break: lower median_duration_s"

    return "tie", "all configured metrics tied"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize mediated vs direct baseline comparison.")
    parser.add_argument("--dataset-root", default="data/model_baselines")
    parser.add_argument("--mediated-experiment-id", default="baseline_mcp_v1")
    parser.add_argument("--direct-experiment-id", default="baseline_direct_api_v1")
    parser.add_argument("--output", default="logs/comparison_summary.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = (repo_root / args.dataset_root).resolve()
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mediated_manifest = dataset_root / args.mediated_experiment_id / "manifest.jsonl"
    direct_manifest = dataset_root / args.direct_experiment_id / "manifest.jsonl"

    mediated_rows = _load_manifest_rows(mediated_manifest)
    direct_rows = _load_manifest_rows(direct_manifest)

    mediated_stats = _aggregate(mediated_rows)
    direct_stats = _aggregate(direct_rows)
    winner, reason = choose_winner(mediated_stats, direct_stats)

    report = {
        "mediated_experiment_id": args.mediated_experiment_id,
        "direct_experiment_id": args.direct_experiment_id,
        "paths": {
            "dataset_root": str(dataset_root),
            "mediated_manifest": str(mediated_manifest),
            "direct_manifest": str(direct_manifest),
        },
        "mediated": mediated_stats,
        "direct_api_tool_use": direct_stats,
        "winner": winner,
        "winner_reason": reason,
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[INFO] wrote comparison summary: {output_path}")
    print(f"[INFO] winner: {winner}")
    print(f"[INFO] reason: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
