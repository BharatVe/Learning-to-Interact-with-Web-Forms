#!/usr/bin/env python3
"""Paired form-level statistics for the LocalForms vs. Google Forms comparison.

Mirrors this repo's established comparison-analysis methodology (see
scripts/analyze_opencua_ruler_comparison.py, scripts/analyze_formfactory_qwen3vl_comparison.py):
form-level paired bootstrap over accuracy differences, resampling whole forms
(not individual fields) with replacement so fields within a form are not
treated as independent observations. Standard library + numpy only.

Outputs (data/localforms_comparison_analysis/):
  - comparison_summary.csv : condition-level totals
  - paired_forms.csv       : one row per form, both conditions' V/T
  - robustness.json        : bootstrap intervals, win/tie/loss, sign test
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "localforms_comparison_analysis"
N_BOOTSTRAP = 10000
SEED = 20260819


def load_localforms():
    lf = {}
    for exp in [
        "opencua_localforms_direct_mcp_fill_only_10_20260819",
        "opencua_localforms_direct_mcp_fill_only_40_20260819",
    ]:
        pattern = str(
            ROOT / "data" / "model_baselines" / exp
            / "computer_use_opencua_32b_direct_mcp" / "*" / "run_0002" / "*" / "summary.json"
        )
        for fp in glob.glob(pattern):
            d = json.load(open(fp))
            fid = (d.get("form_id") or "").removeprefix("lf_")
            prev = lf.get(fid)
            if prev is None or (d.get("verified_correctness") or 0) > (prev.get("verified_correctness") or 0):
                lf[fid] = d
    return lf


def load_gf32():
    gf = {}
    for exp_dir in glob.glob(str(ROOT / "data" / "model_baselines" / "opencua_direct_mcp_fill_only_done_*_step32")):
        pattern = str(Path(exp_dir) / "computer_use_opencua_32b_direct_mcp" / "*" / "run_0002" / "*" / "summary.json")
        for fp in glob.glob(pattern):
            d = json.load(open(fp))
            fid = d.get("form_id")
            if fid not in gf:
                gf[fid] = d
    return gf


def load_gf128():
    gf = {}
    pattern = str(
        ROOT / "data" / "model_baselines" / "opencua_direct_mcp_tools_target300_run2_20260609" / "**" / "summary.json"
    )
    for fp in glob.glob(pattern, recursive=True):
        d = json.load(open(fp))
        gf[d.get("form_id")] = d
    return gf


def gf128_verified(d):
    pre = d.get("pre_successful_submit_verified_correctness")
    return pre if pre is not None else d.get("verified_correctness", 0)


def paired_bootstrap(pairs, rng):
    """pairs: list of (lf_v, lf_t, gf_v, gf_t). Returns dict of stats."""
    lf_v = np.array([p[0] for p in pairs], dtype=float)
    lf_t = np.array([p[1] for p in pairs], dtype=float)
    gf_v = np.array([p[2] for p in pairs], dtype=float)
    gf_t = np.array([p[3] for p in pairs], dtype=float)
    n = len(pairs)

    point_diff = (lf_v.sum() / lf_t.sum()) - (gf_v.sum() / gf_t.sum())

    diffs = np.empty(N_BOOTSTRAP)
    idx_all = np.arange(n)
    for i in range(N_BOOTSTRAP):
        idx = rng.choice(idx_all, size=n, replace=True)
        lf_acc = lf_v[idx].sum() / lf_t[idx].sum()
        gf_acc = gf_v[idx].sum() / gf_t[idx].sum()
        diffs[i] = lf_acc - gf_acc

    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])

    per_form_lf_acc = lf_v / lf_t
    per_form_gf_acc = gf_v / gf_t
    wins = int((per_form_lf_acc > per_form_gf_acc).sum())
    losses = int((per_form_lf_acc < per_form_gf_acc).sum())
    ties = n - wins - losses

    # Exact two-sided sign test (binomial, p=0.5) excluding ties
    from math import comb

    n_eff = wins + losses
    k = min(wins, losses)
    if n_eff > 0:
        p_sign = sum(comb(n_eff, i) for i in range(0, k + 1)) * 2 / (2 ** n_eff)
        p_sign = min(p_sign, 1.0)
    else:
        p_sign = 1.0

    return {
        "n_forms": n,
        "localforms_total": f"{int(lf_v.sum())}/{int(lf_t.sum())}",
        "localforms_accuracy": round(lf_v.sum() / lf_t.sum() * 100, 2),
        "comparison_total": f"{int(gf_v.sum())}/{int(gf_t.sum())}",
        "comparison_accuracy": round(gf_v.sum() / gf_t.sum() * 100, 2),
        "point_diff_pp": round(point_diff * 100, 2),
        "bootstrap_mean_diff_pp": round(float(diffs.mean()) * 100, 2),
        "bootstrap_95ci_pp": [round(float(ci_lo) * 100, 2), round(float(ci_hi) * 100, 2)],
        "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        "wins_localforms": wins,
        "wins_comparison": losses,
        "ties": ties,
        "sign_test_two_sided_p": round(p_sign, 4),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    lf = load_localforms()
    gf32 = load_gf32()
    gf128 = load_gf128()

    all_forms = sorted(lf.keys())

    # Comparison A: LocalForms vs GF32, fill-only matched, full coverage expected
    pairs_a = []
    rows_a = []
    for fid in all_forms:
        if fid not in gf32:
            continue
        d_lf, d_gf = lf[fid], gf32[fid]
        lf_v, lf_t = d_lf.get("verified_correctness", 0), d_lf.get("question_total", 0)
        gf_v, gf_t = d_gf.get("verified_correctness", 0), d_gf.get("question_total", 0)
        pairs_a.append((lf_v, lf_t, gf_v, gf_t))
        rows_a.append({
            "form_id": fid,
            "localforms_v": lf_v, "localforms_t": lf_t,
            "localforms_stop_reason": d_lf.get("stop_reason"),
            "gf32_v": gf_v, "gf32_t": gf_t,
            "gf32_stop_reason": d_gf.get("stop_reason"),
        })
    stats_a = paired_bootstrap(pairs_a, rng)

    # Comparison B: LocalForms vs GF128, step-cap matched, partial coverage
    pairs_b = []
    rows_b = []
    for fid in all_forms:
        if fid not in gf128:
            continue
        d_lf, d_gf = lf[fid], gf128[fid]
        lf_v, lf_t = d_lf.get("verified_correctness", 0), d_lf.get("question_total", 0)
        gf_v = gf128_verified(d_gf)
        gf_t = d_gf.get("question_total", 0)
        pairs_b.append((lf_v, lf_t, gf_v, gf_t))
        rows_b.append({
            "form_id": fid,
            "localforms_v": lf_v, "localforms_t": lf_t,
            "localforms_stop_reason": d_lf.get("stop_reason"),
            "gf128_v": gf_v, "gf128_t": gf_t,
            "gf128_stop_reason": d_gf.get("stop_reason"),
            "gf128_submitted": d_gf.get("submit_success"),
        })
    stats_b = paired_bootstrap(pairs_b, rng)

    robustness = {
        "methodology": (
            f"{N_BOOTSTRAP} form-level bootstrap resamples (resample forms with "
            "replacement, recompute aggregate field accuracy per condition, take "
            "the difference). Exact two-sided sign test on per-form win/loss "
            "(ties excluded), matching this repo's existing paired-comparison "
            "methodology (see scripts/analyze_opencua_ruler_comparison.py)."
        ),
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "comparison_a_localforms_vs_gf32_fillonly": stats_a,
        "comparison_b_localforms_vs_gf128_submitcap128": stats_b,
    }

    (OUT_DIR / "robustness.json").write_text(json.dumps(robustness, indent=2), encoding="utf-8")

    with open(OUT_DIR / "paired_forms_gf32.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_a[0].keys()))
        w.writeheader()
        w.writerows(rows_a)

    with open(OUT_DIR / "paired_forms_gf128.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_b[0].keys()))
        w.writeheader()
        w.writerows(rows_b)

    with open(OUT_DIR / "comparison_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "n_forms", "localforms_accuracy_pct", "comparison_accuracy_pct",
                    "point_diff_pp", "bootstrap_95ci_low_pp", "bootstrap_95ci_high_pp",
                    "ci_excludes_zero", "wins_localforms", "wins_comparison", "ties", "sign_test_p"])
        w.writerow(["A: LocalForms vs GF32 (fill-only matched)", stats_a["n_forms"],
                    stats_a["localforms_accuracy"], stats_a["comparison_accuracy"],
                    stats_a["point_diff_pp"], *stats_a["bootstrap_95ci_pp"],
                    stats_a["ci_excludes_zero"], stats_a["wins_localforms"],
                    stats_a["wins_comparison"], stats_a["ties"], stats_a["sign_test_two_sided_p"]])
        w.writerow(["B: LocalForms vs GF128 (step-cap matched)", stats_b["n_forms"],
                    stats_b["localforms_accuracy"], stats_b["comparison_accuracy"],
                    stats_b["point_diff_pp"], *stats_b["bootstrap_95ci_pp"],
                    stats_b["ci_excludes_zero"], stats_b["wins_localforms"],
                    stats_b["wins_comparison"], stats_b["ties"], stats_b["sign_test_two_sided_p"]])

    print(json.dumps(robustness, indent=2))


if __name__ == "__main__":
    main()
