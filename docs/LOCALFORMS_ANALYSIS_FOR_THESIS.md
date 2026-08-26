# LocalForms platform comparison — analysis package for thesis write-up

Purpose of this document: everything needed to write or rewrite the analysis of
the platform-comparison section — data, statistics, ready-to-adapt prose, and
what to still qualify or leave for a follow-up. It supersedes reading the raw
percentage differences in `docs/LOCALFORMS_50FORM_FILLONLY_RESULTS.md` at face
value; that document has the full per-form table, this one has the statistical
treatment that should govern how those numbers are actually claimed.

## The one-sentence correction this document exists to make

**The raw percentage-point gaps reported earlier (LocalForms ahead by 2.7–7.3
points, depending on which Google Forms cohort is used) are not statistically
distinguishable from zero.** A form-level paired bootstrap (10,000 resamples,
the same methodology this project already uses for its other platform/ablation
comparisons) gives 95% confidence intervals that comfortably cross zero in both
comparisons, and an exact sign test on per-form wins/losses gives p ≈ 0.6–0.7 —
nowhere near conventional significance. **Do not write "LocalForms outperformed
Google Forms by N points."** The defensible claim is the opposite: under two
different matching conditions, no significant difference between platforms was
detected, and the difference that is directionally observed is fully
consistent with ordinary form-to-form variation.

## Statistical results

Computed by `scripts/analyze_localforms_comparison.py` (numpy + stdlib only,
reproducible, seed `20260819`), reading directly from the same `summary.json`
files documented in `docs/LOCALFORMS_50FORM_FILLONLY_RESULTS.md`. Raw outputs:
`data/localforms_comparison_analysis/{robustness.json,comparison_summary.csv,paired_forms_gf32.csv,paired_forms_gf128.csv}`.

| Comparison | n forms | LocalForms acc. | Comparison acc. | Point diff | 95% bootstrap CI | Wins / Losses / Ties | Sign test p |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A** — vs. GF32 (fill-only task, matched; 32-step cap, not matched) | 50 | 73.35% | 70.66% | **+2.69pp** | **[-9.52, +14.14]** | 14 / 11 / 25 | 0.69 |
| **B** — vs. GF128 (128-step cap, matched; submit-enabled task, not matched) | 45 | 73.64% | 66.30% | **+7.34pp** | **[-7.40, +21.08]** | 15 / 11 / 19 | 0.56 |

Reading each column:

- **Point diff** is the naive headline number — what a first pass at the data
  would report, and what should *not* be reported alone.
- **95% bootstrap CI** resamples which 50 (or 45) forms were drawn, rather than
  resampling individual fields, because fields within one form are not
  independent observations (this matches the "form-level bootstrap" standard
  already used in `docs/eval_results/interaction_failure_analysis/` and the
  OpenCUA ruler comparison in `evaluation_additions/opencua_ruler_comparison/`).
  Both intervals span roughly 20+ percentage points and straddle zero — the
  data cannot rule out a true difference of 0, or for that matter a true
  difference running the other direction.
- **Wins/Losses/Ties** counts forms where one condition's field accuracy beat
  the other's outright. Roughly even in both comparisons (14–11 and 15–11),
  with about half the forms tied or effectively identical — not the pattern
  you'd expect if the platform mattered much.
- **Sign test p** is the exact two-sided binomial test on wins vs. losses
  (ties excluded). 0.69 and 0.56 are far from any conventional significance
  threshold.

## Why the interval is this wide (and why that's expected, not a bug)

Every cell in this comparison is a **single deterministic run** (temperature 0,
one seed) per form per condition. With n=50 (or 45) forms and no repeated
sampling, the bootstrap interval is inherently wide — it is measuring
form-to-form variability, not run-to-run model variance, and this project's own
prior documentation (`docs/eval_results/FORMFACTORY_QWEN3VL_COMPARISON.md`,
"each form has one deterministic run... confidence intervals quantify variation
across forms rather than repeated-sampling variance") already flags this
limitation for its other comparisons. It is not specific to this pilot. The
correct response is not to distrust the bootstrap — it's doing its job — but to
size any claim to what a single-run-per-cell design can actually support:
directional, descriptive, hypothesis-generating, not a confirmed effect.

## What *is* supportable from this data (qualitative, not the percentage)

The percentage difference is not supportable as a finding. Something else in
the data is, and it is arguably the more interesting result for a methods
thesis:

- **Both platforms fail on the same forms, for the same reason.**
  `conference_travel` and `publication_submission` — both dropdown-containing
  forms — are the worst or near-worst performer under every condition tested
  (LocalForms, GF32, and GF128 alike). Root-cause tracing of the raw
  `tool_trace.jsonl` for both platforms shows the identical failure mechanism:
  the model calls `browser_type` on a `<select>` element instead of
  `browser_select_option`, loops, and on `publication_submission` specifically
  this occurred on **both** platforms independently and both times terminated
  in the same way — an aborted trial from exceeding vLLM's context-length
  limit after enough repeated failed attempts (documented with full trace
  excerpts in `docs/LOCALFORMS_10FORM_PILOT_RESULTS.md`).
- This is platform-*independent* evidence that OpenCUA-32B's dropdown-handling
  weakness (already documented for this project's Google Forms condition in
  `docs/eval_results/interaction_failure_analysis/DROPDOWN_FAILURE_ANALYSIS.md`)
  is a property of the model's policy, not an artifact of Google Forms'
  particular custom-widget DOM implementation. That is a genuine, positive
  finding: it rules out one hypothesis (that the earlier dropdown results were
  somehow a Google-Forms-DOM-specific measurement artifact) rather than merely
  failing to confirm a different one.
- Every one of the six other `stop_reason` failure categories observed in the
  LocalForms run (`max_steps_exceeded`, `model_no_tool_calls`,
  `done_incomplete_fill_only`, `repeat_invalid_tool_call`) also has a
  same-named, same-behavior counterpart in the Google Forms cohorts — the
  *taxonomy* of how the model fails transfers across platforms even where raw
  scores don't line up form-by-form.

## Suggested prose for the thesis (adapt, don't paste verbatim)

**Results section:**

> To test whether the platform's DOM/widget implementation affects computer-use
> agent performance independent of form content, all 50 forms were recreated as
> a locally-hosted, FormFactory-style [cite] platform using native HTML controls,
> holding the model (OpenCUA-32B), target content, and answer set constant. Under
> a fill-only task with a matched step budget, LocalForms field-level accuracy
> was 73.4% (300/409) versus 70.7% (289/409) for the existing Google Forms
> condition at the same task setting but a smaller step budget (32 vs. 128); a
> form-level paired bootstrap (10,000 resamples) gives a 95% CI of [-9.5, +14.1]
> percentage points, which includes zero. Against a Google Forms cohort matched
> on step budget but not task mode (submit-enabled), the gap was larger in point
> terms (+7.3pp) but the bootstrap CI was wider still ([-7.4, +21.1]) and again
> included zero. Neither comparison supports a claim that the platform
> materially changed agent performance in this pilot.

**Discussion / limitations:**

> The absence of a statistically distinguishable platform effect should not be
> read as a null result about platform choice in general — no archived cohort
> in this study matched LocalForms on both task mode and step budget
> simultaneously, each available comparison holds a different variable fixed,
> and every cell is a single deterministic run. What the data does support is
> that the two platforms fail in the same way on the same forms: both showed
> their worst performance on the two dropdown-heavy forms in the set, and
> root-cause analysis of the raw interaction traces showed an identical
> mechanism (the model attempting `browser_type` on a `<select>` element) on
> both platforms independently. This is evidence against the alternative
> hypothesis that Google Forms' custom-ARIA dropdown implementation was itself
> responsible for previously reported dropdown-handling weaknesses; the same
> weakness reproduces on a platform using native `<select>` elements instead.

## What would upgrade this from descriptive to confirmatory

In priority order:

1. **A single cohort matched on both task mode and step cap** (fill-only, 128
   steps, `run_0002`, all 50 forms, Google Forms) — currently in progress
   (submit-enabled LocalForms runs were also just queued to complete the
   task-mode side of the matrix; see job status in the session log). This
   removes both confounds simultaneously rather than reporting two
   partially-confounded comparisons.
2. **Repeated sampling** (temperature > 0, multiple seeds, or multiple answer
   sets per form) to separate run-to-run model variance from form-to-form
   content variance — the current n=50/45 single-run design cannot distinguish
   these, which is exactly why the bootstrap interval is as wide as it is.
3. **Reruns of the 7 `environment_error`-terminated LocalForms trials**
   (`data_annotation`, `dataset_request`, `field_trip`, `office_hours`,
   `peer_evaluation`, `purchase_request`, `remote_setup`) — each is currently
   scored 0 due to the harness skipping final verification after an aborted
   trial, not because the model necessarily filled nothing; the one form
   already rerun under this exact condition (`publication_submission`) went
   from 0/10 to 9/10. This alone would likely narrow, not widen, the observed
   gap, since it raises LocalForms' floor without changing Google Forms' score.

## Reproducing every number in this document

```bash
.venv/bin/python3 scripts/analyze_localforms_comparison.py
```

Reads only `data/model_baselines/opencua_localforms_direct_mcp_fill_only_{10,40}_20260819/`,
`data/model_baselines/opencua_direct_mcp_fill_only_done_*_step32/`, and
`data/model_baselines/opencua_direct_mcp_tools_target300_run2_20260609/`. No
number in this document was computed by hand.
