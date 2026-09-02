# LocalForms platform comparison — analysis package for thesis write-up

**Status: complete.** Both matched-condition data collections finished
(Google Forms fill-only 128-step, 50/50 forms; LocalForms submit-enabled,
50/50 forms), closing every confound flagged in the previous version of this
document. This is the full, corrected analysis — read this version, not
`docs/LOCALFORMS_15FORM_CHECKPOINT.md` or the earlier draft of this file for
the platform-comparison conclusion; those were provisional snapshots taken
mid-collection.

## The one-sentence finding

**No statistically detectable platform effect survives once task mode, step
budget, and page count are all controlled for.** Two independent, fully
matched comparisons (fill-only and submit-enabled, each same model, same
128-step cap, same `run_0002` answers, same forms, platform as the only
variable) both give bootstrap 95% confidence intervals that comfortably
include zero. A third comparison that initially looked statistically
significant (submit-enabled, p = 0.02) turned out to be driven almost entirely
by a design confound in the LocalForms recreation, not by the platform's DOM
implementation — see "The near-miss" below. That confound, once corrected for,
is itself a real and reportable finding, just not the one it first appeared to
be.

## Statistical results — six comparisons, in order of how well-matched they are

Computed by `scripts/analyze_localforms_comparison.py` (numpy + stdlib only,
reproducible, seed `20260819`). All scores use the harness's own
`scored_correctness` field — the pre-submit verification snapshot for trials
that submitted, final verification otherwise — because raw
`verified_correctness` reads near-zero for successful submissions (page state
is unreadable after the POST navigation; this bit both the LocalForms and the
Google Forms submit-enabled cohorts equally, and is corrected uniformly here).
Raw outputs: `data/localforms_comparison_analysis/{robustness.json, comparison_summary.csv, paired_forms_{a..f}.csv}`.

| # | Comparison | n | LocalForms | Google Forms | Point diff | 95% CI | W/L/T | p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C | **Fill-only, fully matched** (128-step, both platforms) | 50 | 73.3% | 68.0% | +5.4pp | **[-9.1, +19.6]** | 18/19/13 | 1.00 |
| E | **Submit-enabled, fully matched, single-page-only** (the corrected apples-to-apples read) | 29 | 86.4% | 87.3% | -0.9pp | **[-10.2, +8.7]** | 4/4/21 | 1.00 |
| A | vs. GF32 — fill-only matched, step cap NOT matched (32 vs 128) | 50 | 73.3% | 70.7% | +2.7pp | [-9.5, +14.1] | 14/11/25 | 0.69 |
| B | vs. GF128 — step cap matched, task NOT matched (submit vs fill-only) | 45 | 73.6% | 66.3% | +7.3pp | [-7.4, +21.1] | 15/11/19 | 0.56 |
| D | Submit-enabled, fully matched but **confounded by page count** | 45 | 84.8% | 66.3% | +18.5pp | [+7.2, +29.8]† | 18/6/21 | 0.02 |
| F | The 16 forms driving D's confound (multi-page on Google Forms only) | 16 | 82.4% | 35.1% | +47.3pp | [+30.1, +62.4]† | 14/2/0 | 0.004 |

† CI excludes zero.

**C and E are the two comparisons to cite as the platform-effect result.** Both
control every dimension this study set out to test (model, task, step budget,
answer set, forms) and leave only the platform's DOM/widget implementation
varying. Both are null. A and B are the earlier, partially-confounded
comparisons kept for continuity with prior drafts; D and F exist specifically
to show *why* the one seemingly-significant result doesn't mean what it first
appeared to.

## The near-miss: how a real effect turned out to be a confound, and how that was caught

Comparison D — LocalForms submit-enabled vs. Google Forms submit-enabled, both
at the 128-step cap, both `run_0002`, all 45 shared forms — came back with a
95% CI of [+7.2, +29.8], excluding zero, and a sign test p = 0.023.

Before accepting a "LocalForms outperforms Google Forms" claim on the back of
that, the underlying trials were checked field by field:

1. Google Forms submitted successfully on 38/45 forms; LocalForms only 20/45
   — Google Forms submitted *more* often, not less. That alone is inconsistent
   with a simple "LocalForms is the better platform" story, so it needed
   explaining before being trusted.
2. Checking `submitted_while_incomplete_count` on every submitted trial
   resolved it: **all 38 of Google Forms' successful submissions were flagged
   as submitted while the form was still incomplete. All 23 of LocalForms'
   successful submissions were fully complete (0 flagged incomplete).** This
   is a stark, near-categorical behavioral difference, not noise.
3. Splitting those Google Forms submissions by whether the source form has one
   `section_order` or more than one revealed the mechanism: Google Forms
   renders forms with multiple sections as **multiple pages**, requiring
   "Next" navigation before the submit button becomes reachable. 18 of the 50
   forms are multi-section. On those, Google Forms' submitted-trial
   completeness was 38.3% (49/128 fields) — the model evidently submits on an
   earlier page, unaware later sections remain. On the 24 shared forms that
   are single-page on Google Forms too, completeness was 91.7% (165/180) —
   close to LocalForms' pattern.
4. **The LocalForms recreation renders every form as a single page,
   regardless of how many sections the source spec has** (a generation choice
   made for simplicity when the recreation was built, not something re-derived
   for this comparison). So comparison D was silently comparing "Google Forms
   navigating multiple real pages" against "LocalForms never needing to
   navigate at all" for 16 of its 45 forms — a page-count confound, not a
   DOM-widget comparison.
5. Comparison E redoes D restricted to the 29 forms that are single-page on
   *both* platforms — the actual apples-to-apples test the study intended.
   Result: 86.4% vs. 87.3%, CI [-10.2, +8.7], p = 1.00. The effect vanishes
   entirely under the aligned comparison.
6. Comparison F isolates the excluded 16 multi-page-on-Google-Forms forms on
   their own: 82.4% vs. 35.1%, CI [+30.1, +62.4], p = 0.004 — very real and
   very large, but it is evidence about **page count / pagination**, not about
   native-HTML-vs-custom-ARIA-widget DOM implementation, and it is itself
   confounded by the LocalForms recreation's single-page design rather than a
   controlled test of pagination handling.

This is worth stating explicitly as methodology, not just result: a
significant bootstrap CI was found, investigated rather than reported at face
value, and shown to depend on a specific mechanism (`submitted_while_incomplete_count`
+ section count) that a design choice in the recreation — not the underlying
research question — was responsible for.

## What *is* supportable from this data

- **No platform DOM-implementation effect detected**, under either task mode,
  once properly matched (C, E). This is the primary, citable result.
- **A genuine, large, page-count-driven premature-submission effect exists on
  the Google Forms side for multi-section forms** (F), but it is not evidence
  about the LocalForms recreation's fidelity — it's a property of Google
  Forms' pagination interacting with the model's completion judgment, observed
  through a comparison that itself is not fully controlled for page count (see
  "What would close this gap" below).
- **Both platforms fail on the same forms for the same reason, independent of
  task mode.** `conference_travel` and `publication_submission` — both
  dropdown-containing — are the worst or near-worst performer under every
  condition tested. Root-cause tracing of raw `tool_trace.jsonl` shows the
  identical failure mechanism on both platforms independently: the model
  calling `browser_type` on a `<select>` element instead of
  `browser_select_option`, looping until vLLM's context-length limit aborts
  the trial (`environment_error`). This is direct evidence that OpenCUA-32B's
  documented dropdown-handling weakness
  (`docs/eval_results/interaction_failure_analysis/DROPDOWN_FAILURE_ANALYSIS.md`)
  is a model-policy property, not an artifact of Google Forms' particular
  custom-widget DOM — the same weakness reproduces against a native `<select>`.
- The full taxonomy of failure modes (`max_steps_exceeded`, `model_no_tool_calls`,
  `done_incomplete_fill_only`, `repeat_invalid_tool_call`, `environment_error`)
  appears on both platforms with comparable frequency — the *ways* the model
  fails transfer across platforms even where raw per-form scores don't align.

## Suggested prose for the thesis (adapt, don't paste verbatim)

**Results section:**

> To test whether a form platform's DOM/widget implementation affects
> computer-use agent performance independent of form content, all 50 forms
> were recreated as a locally-hosted, FormFactory-style [cite] platform using
> native HTML controls, holding the model (OpenCUA-32B), task, step budget,
> target content, and answer set constant against the existing Google Forms
> condition. Under a fill-only task (128-step cap, `run_0002`, all 50 forms),
> LocalForms field-level accuracy was 73.3% versus 68.0% for Google Forms; a
> form-level paired bootstrap (10,000 resamples) gives a 95% CI of [-9.1,
> +19.6] percentage points, which includes zero. Under a submit-enabled task,
> restricted to the 29 forms that render as a single page on both platforms
> (excluding 16 forms Google Forms presents as multi-page), LocalForms and
> Google Forms scored within one point of each other (86.4% vs. 87.3%, CI
> [-10.2, +8.7]). Neither comparison supports a claim that the platform's
> DOM/widget implementation materially changes agent performance.

**Discussion — the pagination finding:**

> An unrestricted submit-enabled comparison across all 45 shared forms did
> show a large, statistically significant gap (+18.5pp, CI [+7.2, +29.8]).
> Investigating this before accepting it revealed it was driven almost
> entirely by 16 forms that Google Forms presents as multi-page (via its
> native section/pagination feature) but which the LocalForms recreation, by
> construction, always renders as a single page. On these forms specifically,
> the agent submitted the Google Forms version while it was still incomplete
> in every recorded case (`submitted_while_incomplete_count` = 1 for all 38
> Google Forms submissions among the shared forms, versus 0 for all 23
> LocalForms submissions), consistent with the agent judging the form complete
> upon reaching an early page's end rather than recognizing further pages
> remained below. This is a genuine finding about how pagination interacts
> with this agent's completion judgment, but the current study design cannot
> attribute it to the platform's DOM implementation versus its own choice to
> flatten multi-section forms to a single page; a paginated LocalForms
> recreation of these 16 forms would be needed to separate a true
> platform-pagination effect from the recreation's simplification.

## What would close the remaining gap

1. **A paginated LocalForms recreation of the 16 multi-section forms** —
   `<form>`-per-section with a "Next" control gating the final submit, mirroring
   Google Forms' structure. This is the one remaining design choice that
   confounds a result (F) that is otherwise large and robust. Everything else
   in the platform-comparison design is now controlled for.
2. **Reruns of the LocalForms `environment_error` trials** (7 in fill-only, 2
   in submit-enabled) — each scores 0 because the harness skips final
   verification after an aborted trial, not because the model necessarily
   filled nothing; the one form already rerun under this exact condition
   (`publication_submission`) went from 0/10 to 9/10 on retry. This would
   likely narrow the already-null gap in C further, not open one.
3. **Repeated sampling** (temperature > 0, multiple seeds) to separate
   run-to-run model variance from form-to-form content variance — every cell
   in this study is a single deterministic run, which is why even the null
   comparisons' intervals are as wide as ±10-20pp.

## Reproducing every number in this document

```bash
.venv/bin/python3 scripts/analyze_localforms_comparison.py
```

Reads `data/model_baselines/opencua_localforms_direct_mcp_{fill_only,submit}_{10,40}_*/`,
`data/model_baselines/opencua_direct_mcp_fill_only_done_*_step32/`,
`data/model_baselines/opencua_direct_mcp_tools_target300_run2_20260609/`, and
`data/model_baselines/opencua_direct_mcp_fill_only_128_run0002_20260901/`. The
multi-section form list is derived live from `src/forms/*/spec.json`. No number
in this document was computed by hand.
