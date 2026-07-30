# Qwen3-VL FormFactory-style vs direct-MCP comparison

This analysis uses only the explicit cohorts in `configs/baselines/formfactory_qwen3vl_comparison_analysis.json`; it never scans unrelated legacy experiments.

The primary endpoint is non-dropdown field accuracy with a form-level bootstrap confidence interval. All four cohorts use the current corrected verifier. Submission-enabled correctness is captured immediately before successful submission. Action counts are secondary because coordinate UI primitives and semantic MCP calls have different granularity.

Current completion: **3/4 cohorts**.

> **Interim analysis:** the incomplete cohort is excluded from aggregate and paired tables; no partial paired result is published.

Incomplete checks:

- direct_mcp_fill_only: missing 4 forms

## Complete-cohort results

| Condition | Trials | Primary accuracy (95% CI) | All-field accuracy | Full fills | Submissions |
|---|---:|---:|---:|---:|---:|
| FormFactory-style visual (fill-only) | 50 | 34.1% (31.9%–36.4%) | 33.7% | 0.0% | 0.0% |
| Direct MCP (submit-enabled) | 50 | 63.8% (55.8%–72.1%) | 64.1% | 26.0% | 64.0% |
| FormFactory-style visual (submit-enabled) | 50 | 28.6% (25.7%–31.7%) | 28.4% | 0.0% | 0.0% |

## Complete paired comparisons

Differences are visual minus direct MCP; negative values favour direct MCP.

| Comparison | Forms | Primary difference (95% CI) | Full-fill difference | Submission difference |
|---|---:|---:|---:|---:|
| submit_visual_minus_direct | 50 | -37.9% (-46.2%–-29.8%) | -26.0% | -64.0% |

## Interpretation limits

- This is a FormFactory-style adaptation, not a reproduction of the FormFactory study.
- Each condition has one deterministic run per form; confidence intervals quantify variation across forms, not decoding randomness.
- Correctness is the primary comparison. Semantic MCP actions and primitive visual actions do not have equivalent granularity.

Machine-readable outputs are the `formfactory_qwen3vl_*` files under `data/model_baseline_exports`.
