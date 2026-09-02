#!/usr/bin/env python3
"""Paired form-level statistics for the LocalForms vs. Google Forms comparison.

Mirrors this repo's established comparison-analysis methodology (see
scripts/analyze_opencua_ruler_comparison.py, scripts/analyze_formfactory_qwen3vl_comparison.py):
form-level paired bootstrap over accuracy differences, resampling whole forms
(not individual fields) with replacement so fields within a form are not
treated as independent observations. Standard library + numpy only.

Four comparisons are computed:
  A: LocalForms fill-only        vs GF32  (fill-only, 32-step)   -- task matched, step cap NOT matched
  B: LocalForms fill-only        vs GF128 (submit-enabled, 128-step) -- step cap matched, task NOT matched
  C: LocalForms fill-only        vs GF-fill128 (fill-only, 128-step) -- FULLY MATCHED
  D: LocalForms submit-enabled   vs GF128 (submit-enabled, 128-step) -- FULLY MATCHED

C and D are the clean comparisons (same model, same task mode, same step cap,
same run_0002, same 50 forms, platform is the only thing that varies). A and B
are kept for continuity with earlier partial-evidence reports but should not be
read as platform-isolating on their own.

For any trial that submitted, use `scored_correctness` (the harness's own
pre-submit-snapshot score), not raw `verified_correctness` -- post-submit page
state is frequently unreadable after navigation, so `verified_correctness`
reads near-zero for a successful submission. This bit both the LocalForms
submit-enabled cohort and the GF128 cohort; both are read via `scored_correctness`
here for consistency (GF128 was previously read via
`pre_successful_submit_verified_correctness` directly, which is one of the two
inputs `scored_correctness` already resolves to, so this is not a behavior change
for GF128, just a single consistent code path for both).

Outputs (data/localforms_comparison_analysis/):
  - comparison_summary.csv     : condition-level totals, all four comparisons
  - paired_forms_{a,b,c,d}.csv : one row per form for each comparison
  - robustness.json            : bootstrap intervals, win/tie/loss, sign test, all four
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


def _scored(d):
    """Prefer the harness's own pre-submit-snapshot score; fall back to
    verified_correctness for trials that never reached submission."""
    sc = d.get("scored_correctness")
    if sc is not None:
        return sc
    pre = d.get("pre_successful_submit_verified_correctness")
    if pre is not None:
        return pre
    return d.get("verified_correctness", 0)


def _load(exp_names, prefer_max=False):
    out = {}
    for exp in exp_names:
        pattern = str(
            ROOT / "data" / "model_baselines" / exp
            / "computer_use_opencua_32b_direct_mcp" / "*" / "run_0002" / "*" / "summary.json"
        )
        for fp in glob.glob(pattern):
            d = json.load(open(fp))
            fid = (d.get("form_id") or "").removeprefix("lf_")
            if prefer_max and fid in out:
                if _scored(d) > _scored(out[fid]):
                    out[fid] = d
            else:
                out[fid] = d
    return out


def load_localforms_fillonly():
    return _load(
        ["opencua_localforms_direct_mcp_fill_only_10_20260819", "opencua_localforms_direct_mcp_fill_only_40_20260819"],
        prefer_max=True,
    )


def load_localforms_submit():
    return _load(
        ["opencua_localforms_direct_mcp_submit_10_20260827", "opencua_localforms_direct_mcp_submit_40_20260827"]
    )


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


def load_gf_fill128():
    gf = {}
    pattern = str(
        ROOT / "data" / "model_baselines" / "opencua_direct_mcp_fill_only_128_run0002_20260901"
        / "computer_use_opencua_32b_direct_mcp" / "*" / "run_0002" / "*" / "summary.json"
    )
    for fp in glob.glob(pattern):
        d = json.load(open(fp))
        gf[d.get("form_id")] = d
    return gf


def paired_bootstrap(pairs, rng):
    """pairs: list of (a_v, a_t, b_v, b_t). Returns dict of stats."""
    a_v = np.array([p[0] for p in pairs], dtype=float)
    a_t = np.array([p[1] for p in pairs], dtype=float)
    b_v = np.array([p[2] for p in pairs], dtype=float)
    b_t = np.array([p[3] for p in pairs], dtype=float)
    n = len(pairs)

    point_diff = (a_v.sum() / a_t.sum()) - (b_v.sum() / b_t.sum())

    diffs = np.empty(N_BOOTSTRAP)
    idx_all = np.arange(n)
    for i in range(N_BOOTSTRAP):
        idx = rng.choice(idx_all, size=n, replace=True)
        a_acc = a_v[idx].sum() / a_t[idx].sum()
        b_acc = b_v[idx].sum() / b_t[idx].sum()
        diffs[i] = a_acc - b_acc

    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])

    per_form_a_acc = a_v / a_t
    per_form_b_acc = b_v / b_t
    wins = int((per_form_a_acc > per_form_b_acc).sum())
    losses = int((per_form_a_acc < per_form_b_acc).sum())
    ties = n - wins - losses

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
        "a_total": f"{int(a_v.sum())}/{int(a_t.sum())}",
        "a_accuracy": round(a_v.sum() / a_t.sum() * 100, 2),
        "b_total": f"{int(b_v.sum())}/{int(b_t.sum())}",
        "b_accuracy": round(b_v.sum() / b_t.sum() * 100, 2),
        "point_diff_pp": round(point_diff * 100, 2),
        "bootstrap_mean_diff_pp": round(float(diffs.mean()) * 100, 2),
        "bootstrap_95ci_pp": [round(float(ci_lo) * 100, 2), round(float(ci_hi) * 100, 2)],
        "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        "wins_a": wins,
        "wins_b": losses,
        "ties": ties,
        "sign_test_two_sided_p": round(p_sign, 4),
    }


def multi_section_forms():
    """Form ids whose original Google Forms spec has more than one section_order
    value -- these render as multi-page (paginated) forms on Google Forms. The
    LocalForms recreation renders every form as a single page regardless, so
    these forms are NOT an apples-to-apples page-count comparison."""
    out = set()
    for fp in glob.glob(str(ROOT / "src" / "forms" / "*" / "spec.json")):
        s = json.load(open(fp))
        sections = set(q.get("section_order") for q in s.get("questions", []))
        if len(sections) > 1:
            out.add(s.get("form_id"))
    return out


def build_pairs(a_dict, b_dict, only=None, exclude=None):
    pairs, rows = [], []
    for fid in sorted(a_dict.keys()):
        if fid not in b_dict:
            continue
        if only is not None and fid not in only:
            continue
        if exclude is not None and fid in exclude:
            continue
        da, db = a_dict[fid], b_dict[fid]
        av, at = _scored(da), da.get("question_total", 0)
        bv, bt = _scored(db), db.get("question_total", 0)
        pairs.append((av, at, bv, bt))
        rows.append({
            "form_id": fid,
            "a_v": av, "a_t": at, "a_stop_reason": da.get("stop_reason"),
            "b_v": bv, "b_t": bt, "b_stop_reason": db.get("stop_reason"),
        })
    return pairs, rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    lf_fill = load_localforms_fillonly()
    lf_submit = load_localforms_submit()
    gf32 = load_gf32()
    gf128 = load_gf128()
    gf_fill128 = load_gf_fill128()
    multi = multi_section_forms()

    comparisons = {
        "A_localforms_fillonly_vs_gf32_fillonly_32step": (lf_fill, gf32, None, None, "task matched, step cap NOT matched (128 vs 32)"),
        "B_localforms_fillonly_vs_gf128_submit_128step": (lf_fill, gf128, None, None, "step cap matched (128), task NOT matched (fill-only vs submit-enabled)"),
        "C_localforms_fillonly_vs_gf_fillonly_128step": (lf_fill, gf_fill128, None, None, "FULLY MATCHED: fill-only, 128-step, run_0002, both platforms"),
        "D_localforms_submit_vs_gf128_submit_128step": (lf_submit, gf128, None, None, "FULLY MATCHED but includes a page-count confound: submit-enabled, 128-step, run_0002, both platforms -- 16 of these forms are multi-page on Google Forms but single-page on the LocalForms recreation (see E/F)"),
        "E_D_restricted_single_page_only": (lf_submit, gf128, None, multi, "D restricted to forms that are single-page on BOTH platforms (excludes forms multi-page on Google Forms) -- the real apples-to-apples submit-enabled comparison"),
        "F_D_restricted_multipage_only": (lf_submit, gf128, multi, None, "D restricted to forms that are multi-page on Google Forms but single-page on LocalForms -- isolates the page-count confound itself, NOT a platform DOM-implementation effect"),
    }

    robustness = {
        "methodology": (
            f"{N_BOOTSTRAP} form-level bootstrap resamples (resample forms with "
            "replacement, recompute aggregate field accuracy per condition, take "
            "the difference). Exact two-sided sign test on per-form win/loss "
            "(ties excluded). Scores use the harness's scored_correctness field "
            "(pre-submit snapshot for submitted trials, final verification "
            "otherwise) so post-submit-navigation state loss does not bias "
            "submit-enabled comparisons toward zero. Comparisons E and F split "
            "D by whether the source Google Forms form has more than one "
            "section_order (i.e. renders as multiple pages on Google Forms); "
            "the LocalForms recreation renders every form as a single page "
            "regardless, so D alone conflates a platform-DOM effect with a "
            "page-count effect -- E is the corrected apples-to-apples reading."
        ),
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "multi_section_form_count": len(multi),
    }

    summary_rows = []
    for key, (a_dict, b_dict, only, exclude, note) in comparisons.items():
        pairs, rows = build_pairs(a_dict, b_dict, only=only, exclude=exclude)
        stats = paired_bootstrap(pairs, rng)
        stats["note"] = note
        robustness[key] = stats
        letter = key[0].lower()
        with open(OUT_DIR / f"paired_forms_{letter}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        summary_rows.append([
            key, stats["n_forms"], stats["a_accuracy"], stats["b_accuracy"],
            stats["point_diff_pp"], *stats["bootstrap_95ci_pp"], stats["ci_excludes_zero"],
            stats["wins_a"], stats["wins_b"], stats["ties"], stats["sign_test_two_sided_p"],
        ])

    (OUT_DIR / "robustness.json").write_text(json.dumps(robustness, indent=2), encoding="utf-8")

    with open(OUT_DIR / "comparison_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "n_forms", "a_accuracy_pct", "b_accuracy_pct",
                    "point_diff_pp", "bootstrap_95ci_low_pp", "bootstrap_95ci_high_pp",
                    "ci_excludes_zero", "wins_a", "wins_b", "ties", "sign_test_p"])
        w.writerows(summary_rows)

    print(json.dumps(robustness, indent=2))


if __name__ == "__main__":
    main()
