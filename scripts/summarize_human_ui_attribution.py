#!/usr/bin/env python3
import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _load_summary(path_value: str) -> Dict[str, Any]:
    path = Path(path_value)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = max(0.0, min(1.0, p)) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    weight = pos - lo
    return float(xs[lo] * (1.0 - weight) + xs[hi] * weight)


def _mean_ci(values: List[float], z: float = 1.96) -> Optional[Dict[str, float]]:
    n = len(values)
    if n == 0:
        return None
    m = _mean(values)
    if n == 1:
        return {"mean": round(m, 6), "lower": round(m, 6), "upper": round(m, 6), "n": 1}
    variance = sum((x - m) ** 2 for x in values) / max(1, n - 1)
    se = math.sqrt(variance / n)
    return {
        "mean": round(m, 6),
        "lower": round(m - z * se, 6),
        "upper": round(m + z * se, 6),
        "n": n,
    }


def _binomial_ci(successes: int, n: int, z: float = 1.96) -> Optional[Dict[str, float]]:
    if n <= 0:
        return None
    p = successes / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / n) + ((z * z) / (4 * n * n)))
    return {
        "rate": round(p, 6),
        "lower": round(max(0.0, center - margin), 6),
        "upper": round(min(1.0, center + margin), 6),
        "n": n,
    }


def _extract_model_size_b(model_id: str) -> Optional[int]:
    text = str(model_id or "").lower()
    m = re.search(r"(?<!\d)(\d{1,3})\s*[_-]?b(?![a-z0-9])", text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _trial_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    question_total = max(1, int(summary.get("question_total") or 0))
    return {
        "verified_rate": _safe_rate(float(summary.get("verified_correctness") or 0), float(question_total)),
        "attempted_rate": _safe_rate(float(summary.get("attempted_correctness") or 0), float(question_total)),
        "composite_score": float(summary.get("composite_score") or 0.0),
        "autonomy_step_rate": float(summary.get("autonomy_step_rate") or 0.0),
        "action_diversity": float(summary.get("action_diversity") or 0.0),
        "loop_ratio": float(summary.get("loop_ratio") or 0.0),
        "correction_count": float(summary.get("correction_count") or 0.0),
    }


def _aggregate_model(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    runs = len(rows)
    successes = sum(1 for row in rows if bool(row.get("success")))
    submit_successes = sum(1 for row in rows if bool(row.get("submit_success")))
    verified_rates = [float(row.get("verified_rate") or 0.0) for row in rows]
    attempted_rates = [float(row.get("attempted_rate") or 0.0) for row in rows]
    composite_scores = [float(row.get("composite_score") or 0.0) for row in rows]
    autonomy_rates = [float(row.get("autonomy_step_rate") or 0.0) for row in rows]
    diversity_rates = [float(row.get("action_diversity") or 0.0) for row in rows]
    loop_ratios = [float(row.get("loop_ratio") or 0.0) for row in rows]
    correction_counts = [float(row.get("correction_count") or 0.0) for row in rows]

    return {
        "runs": runs,
        "success_rate_ci95": _binomial_ci(successes, runs),
        "submit_success_rate_ci95": _binomial_ci(submit_successes, runs),
        "verified_correctness_rate_ci95": _mean_ci(verified_rates),
        "attempted_correctness_rate_ci95": _mean_ci(attempted_rates),
        "quality_metrics": {
            "composite_score_ci95": _mean_ci(composite_scores),
            "autonomy_step_rate_ci95": _mean_ci(autonomy_rates),
            "action_diversity_ci95": _mean_ci(diversity_rates),
            "loop_ratio_ci95": _mean_ci(loop_ratios),
            "correction_count_ci95": _mean_ci(correction_counts),
            "composite_score_distribution": {
                "p25": _percentile(composite_scores, 0.25),
                "p50": _percentile(composite_scores, 0.50),
                "p75": _percentile(composite_scores, 0.75),
            },
        },
    }


def _effect_delta_ci(a: List[float], b: List[float]) -> Optional[Dict[str, float]]:
    if not a or not b:
        return None
    ma = _mean(a)
    mb = _mean(b)
    if len(a) <= 1:
        va = 0.0
    else:
        va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    if len(b) <= 1:
        vb = 0.0
    else:
        vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt((va / max(1, len(a))) + (vb / max(1, len(b))))
    delta = mb - ma
    margin = 1.96 * se
    return {
        "low_size_mean": round(ma, 6),
        "high_size_mean": round(mb, 6),
        "delta_high_minus_low": round(delta, 6),
        "ci95_lower": round(delta - margin, 6),
        "ci95_upper": round(delta + margin, 6),
        "n_low": len(a),
        "n_high": len(b),
    }


def _size_pair_decomposition(kind_rows: List[Dict[str, Any]], low_b: int, high_b: int) -> Dict[str, Any]:
    low_rows = [row for row in kind_rows if int(row.get("model_size_b") or -1) == low_b]
    high_rows = [row for row in kind_rows if int(row.get("model_size_b") or -1) == high_b]
    low_verified = [float(row.get("verified_rate") or 0.0) for row in low_rows]
    high_verified = [float(row.get("verified_rate") or 0.0) for row in high_rows]
    low_composite = [float(row.get("composite_score") or 0.0) for row in low_rows]
    high_composite = [float(row.get("composite_score") or 0.0) for row in high_rows]

    return {
        "low_size_b": low_b,
        "high_size_b": high_b,
        "verified_correctness_effect_ci95": _effect_delta_ci(low_verified, high_verified),
        "composite_score_effect_ci95": _effect_delta_ci(low_composite, high_composite),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize human_ui_v1 attribution and model-size effects.")
    parser.add_argument("--dataset-root", default="data/model_baselines")
    parser.add_argument("--experiment-id", default="baseline_mcp_v1")
    parser.add_argument("--interaction-protocol", default="human_ui_v1")
    parser.add_argument("--expected-forms", type=int, default=0)
    parser.add_argument("--expected-runs-per-form", type=int, default=0)
    parser.add_argument("--expected-models", type=int, default=0)
    parser.add_argument("--output", default="logs/human_ui_attribution_report.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = (repo_root / args.dataset_root).resolve()
    experiment_root = dataset_root / args.experiment_id
    manifest_path = experiment_root / "manifest.jsonl"
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows = _load_jsonl(manifest_path)
    trial_rows: List[Dict[str, Any]] = []
    excluded_non_human = 0
    manifest_rows_matching_protocol = 0
    for row in manifest_rows:
        if str(row.get("interaction_protocol") or "") == str(args.interaction_protocol):
            manifest_rows_matching_protocol += 1
        summary_path = str(row.get("summary_path") or "").strip()
        if not summary_path:
            continue
        summary = _load_summary(summary_path)
        if not summary:
            continue
        if str(summary.get("interaction_protocol") or "") != str(args.interaction_protocol):
            excluded_non_human += 1
            continue
        model_id = str(summary.get("model_id") or row.get("model_id") or "")
        model_kind = str(summary.get("model_kind") or row.get("model_kind") or "")
        metric_row = {
            "trial_id": summary.get("trial_id") or row.get("trial_id"),
            "model_id": model_id,
            "model_kind": model_kind,
            "model_size_b": _extract_model_size_b(model_id),
            "form_id": str(summary.get("form_id") or row.get("form_id") or ""),
            "answer_run_id": str(summary.get("answer_run_id") or row.get("answer_run_id") or ""),
            "success": bool(summary.get("success")),
            "submit_success": bool(summary.get("submit_success")),
        }
        metric_row.update(_trial_metrics(summary))
        trial_rows.append(metric_row)

    per_model: Dict[str, Dict[str, Any]] = {}
    model_ids = sorted({str(row.get("model_id") or "") for row in trial_rows if str(row.get("model_id") or "")})
    for model_id in model_ids:
        model_trials = [row for row in trial_rows if str(row.get("model_id") or "") == model_id]
        model_kind = str(model_trials[0].get("model_kind") or "") if model_trials else ""
        per_model[model_id] = {
            "model_kind": model_kind,
            "model_size_b": _extract_model_size_b(model_id),
            **_aggregate_model(model_trials),
        }

    text_trials = [row for row in trial_rows if str(row.get("model_kind") or "") == "text_llm"]
    vlm_trials = [row for row in trial_rows if str(row.get("model_kind") or "") == "vlm"]

    unique_model_count = len({str(row.get("model_id") or "") for row in trial_rows if str(row.get("model_id") or "")})
    unique_form_count = len({str(row.get("form_id") or "") for row in trial_rows if str(row.get("form_id") or "")})
    unique_run_count = len({str(row.get("answer_run_id") or "") for row in trial_rows if str(row.get("answer_run_id") or "")})
    inferred_expected = unique_model_count * unique_form_count * unique_run_count
    explicit_expected = None
    if args.expected_forms > 0 and args.expected_runs_per_form > 0 and args.expected_models > 0:
        explicit_expected = int(args.expected_forms) * int(args.expected_runs_per_form) * int(args.expected_models)
    expected_trials = explicit_expected if explicit_expected is not None else inferred_expected
    included_trials = len(trial_rows)
    missing_trials = max(0, int(expected_trials) - included_trials) if expected_trials > 0 else 0
    failed_included_trials = sum(1 for row in trial_rows if not bool(row.get("success")))
    success_included_trials = included_trials - failed_included_trials

    report = {
        "generated_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment_id": args.experiment_id,
        "filters": {
            "interaction_protocol": args.interaction_protocol,
        },
        "paths": {
            "dataset_root": str(dataset_root),
            "manifest_path": str(manifest_path),
            "output": str(output_path),
        },
        "counts": {
            "manifest_rows": len(manifest_rows),
            "manifest_rows_matching_protocol": manifest_rows_matching_protocol,
            "included_trials": included_trials,
            "excluded_non_matching_protocol": excluded_non_human,
            "success_included_trials": success_included_trials,
            "failed_included_trials": failed_included_trials,
            "expected_trials": expected_trials,
            "expected_trials_source": "explicit" if explicit_expected is not None else "inferred",
            "missing_trials_vs_expected": missing_trials,
            "unique_models_included": unique_model_count,
            "unique_forms_included": unique_form_count,
            "unique_answer_runs_included": unique_run_count,
        },
        "per_model": per_model,
        "size_decomposition": {
            "text_7b_vs_30b": _size_pair_decomposition(text_trials, low_b=7, high_b=30),
            "vlm_8b_vs_30b": _size_pair_decomposition(vlm_trials, low_b=8, high_b=30),
        },
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[INFO] wrote human-ui attribution report: {output_path}")
    print(f"[INFO] included_trials={len(trial_rows)} excluded_non_matching_protocol={excluded_non_human}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
