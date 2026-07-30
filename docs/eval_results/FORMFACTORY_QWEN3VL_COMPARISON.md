# Qwen3-VL FormFactory-style vs direct-MCP comparison

## Research question

How much does the interaction interface affect form-filling performance when the underlying model, target answers, forms, answer seed, and scoring policy are held constant?

The model in every new condition is `Qwen/Qwen3-VL-30B-A3B-Instruct`. The comparison is inspired by [FormFactory](https://dl.acm.org/doi/10.1145/3746027.3758285), but is deliberately described as **FormFactory-style**: it adapts screenshot-coordinate interaction to this repository's 50 Google Forms. It is not a reproduction of FormFactory's original forms, prompts, or external API models.

## Paired design

| Task | Interface A | Interface B | Forms and seed |
|---|---|---|---|
| Fill-only | Screenshot + primitive coordinate/keyboard actions | Direct Playwright MCP form tools | Same 50 forms, `run_0002` |
| Submit-enabled | Screenshot + primitive coordinate/keyboard actions | Direct Playwright MCP form tools | Same 50 forms, `run_0002` |

The screenshot condition receives the same remaining-answer task state used by the direct-MCP condition and the current screenshot. It does not receive page text, accessibility snapshots, DOM-derived interaction maps, focus state, visible element IDs, actual field values from the verifier, or explicit verifier-error feedback. The evaluator may use DOM state internally for scoring and to update the shared remaining-answer state, but the DOM evidence itself is not placed in the model prompt.

## Metrics and safeguards

- Primary endpoint: non-dropdown field accuracy, with 10,000 form-level bootstrap samples.
- Full-form completion and submission success are reported separately.
- Submission-enabled correctness is captured immediately before the successful submission, avoiding the loss of readable field state after navigation.
- All four conditions are rerun with the current evaluator. This avoids mixing the new 128-step visual protocol with the earlier direct-MCP fill-only export, which combined three batches, used a 32-step cap, and predated the current scoring metadata.
- Action counts exclude setup navigation, screenshots, internal DOM verification, harness synchronization, and browser close. `browser_fill_form` is additionally expanded by its number of fields. Action efficiency remains secondary because semantic MCP calls and UI primitives have inherently different granularity.
- `scripts/analyze_formfactory_qwen3vl_comparison.py` reads only the explicit sources in `configs/baselines/formfactory_qwen3vl_comparison_analysis.json`. It fails in strict mode on missing forms, duplicates, or mixed task modes.

## Current execution status (2026-07-26)

Five-form pilots were submitted before the full campaign:

- Slurm `2304519`: direct-MCP submit-enabled; completed 5/5 trials.
- Slurm `2304517` and `2304518`: initial visual jobs failed before model startup because the loaded OpenSSL library was incompatible with the evaluation Python virtualenv. They produced no trials.
- Slurm `2305016` and `2305017`: the Qwen3-VL server became ready, but both visual reruns failed before their first trial because the warm-up process did not inherit the Python 3.12 runtime library path. They produced no trials.
- Slurm `2305253` and `2305254`: the Qwen3-VL server became ready and the corrected warm-up process ran, but its embedded PNG was malformed and vLLM rejected it before the first trial. They produced no trials.
- Slurm `2306745`: FormFactory-style fill-only pilot completed 5/5 trials, but all five exhausted the non-progress budget with 0/41 verified target fields.
- Slurm `2306746`: submit-enabled rerun cancelled before trials after Slurm colocated both pilots on one node with the same localhost server port.
- Slurm `2306772`: submit-enabled pilot completed 5/5 trials, but all five failed (three non-progress exhaustion, two maximum-step exhaustion), with 0/41 verified target fields and no submissions.

The completed visual pilots are diagnostic rather than paper-ready results. Across both conditions, 299/336 model steps were rejected because Qwen3-VL commonly emitted a coordinate pair in `args.x` (for example, `[377, 440]`) while the action parser required scalar `x` and `y` values. The accepted clicks also received only a generic model-visible `clicked` result, which encouraged repeated clicks on already-focused text fields. The full 50-form visual campaign remains blocked until coordinate parsing and model-visible action history are corrected and another pilot passes.

The parser now deterministically accepts Qwen-style coordinate pairs only when their redundant `y` representation agrees, and logs every coercion. Offline replay accepts 336/336 recorded pilot outputs. Screenshot-only prompts now include a sanitized two-step history containing only the agent's own tool, arguments, execution status, and syntax errors; DOM maps, page text, focus state, and verifier feedback remain hidden. The 38-test baseline contract suite passes.

- Slurm `2307716` and `2307717`: infrastructure-only failures before model startup; allocated node `i8012` returned CUDA system error 802. They produced no trials.
- Slurm `2307720`: corrected fill-only pilot completed 5/5 held-out trials. It verified 14/43 fields (32.6%); no form was fully completed. Every trial stopped on a genuine repeated-action loop.
- Slurm `2307721`: corrected submit-enabled pilot completed 5/5 held-out trials. It verified 13/43 fields (30.2%); no form was completed or submitted. Every trial stopped on a genuine repeated-action loop.

The held-out forms are `conference_travel`, `course_enrollment`, `exam_registration`, `job_fair`, and `lab_safety`, all using `run_0002`. Across the two corrected pilots, all 98 model steps parsed successfully, all 88 non-initial prompts contained the sanitized action history, and no prompt contained an interaction map, validation feedback, or page-text block. The terminal scroll loops produced identical screenshots, and the remaining loops repeatedly clicked or appended text without improving correctness. These are therefore model/interface outcomes rather than parser or hidden-context failures. The corrected visual method is ready for a full campaign, with its weak pilot performance retained as an expected possible outcome rather than tuned away.

The direct-MCP submit pilot submitted 4/5 forms and scored 25/41 target fields from pre-submit state; two forms were completely correct before submission. A completed historical direct-MCP fill-only cohort exists, but it is not used in the paired primary analysis because it combines three earlier batches and had a 32-step cap. The primary campaign therefore reruns all four cells with current code: direct-MCP and FormFactory-style visual interaction, each in fill-only and submit-enabled mode.

## Scale-up decision

The corrected held-out visual pilots are sufficient to validate the execution method and justify a 50-form run. They are not sufficient to estimate final performance: five forms cover only 43 fields, no visual trial reached the lower widgets, and the form-level sampling uncertainty would be very wide.

The observed low score is retained rather than tuned away. The evaluator executed the model's coordinates and scroll deltas as emitted; positive `deltaY` is documented and implemented as downward scrolling. Eight terminal scroll sequences showed identical screenshots because the model repeatedly requested upward scrolling at the top boundary. The remaining two terminal sequences repeated a click or appended text to an already focused wrong field. Full terminal verification scored every question, so target-only step verification did not suppress the reported final score. Every text-entry action that made progress was focused on its claimed field; one submit-mode loop claimed a later question while the earlier email field remained focused, and it correctly made no scoring progress.

Known non-model incidents are excluded from the evidence: OpenSSL/runtime startup failures, a malformed warm-up PNG, a localhost port collision, and CUDA error 802 on node `i8012` all occurred before trials and produced no scored results. The earlier 0/41 visual pilots are also excluded because 299/336 actions were rejected by the old coordinate parser. The corrected pilots had 0/98 invalid actions and are the implementation-validation evidence.

The remaining limitations are study-design caveats rather than launch blockers: this is a FormFactory-style adaptation rather than a reproduction; each form has one deterministic (`temperature=0`) run, so confidence intervals quantify variation across forms rather than repeated-sampling variance; and semantic MCP calls and visual UI primitives have different action granularity, so action efficiency is secondary to correctness.

## Full campaign submission (2026-07-27)

| Slurm job | Condition | Experiment ID |
|---|---|---|
| `2309099` | FormFactory-style visual, fill-only | `formfactory_style_qwen3vl_fill_only_50_r2_20260724` |
| `2309097` | FormFactory-style visual, submit-enabled | `formfactory_style_qwen3vl_submit_50_r2_20260724` |
| `2309098` | Direct MCP, fill-only | `qwen3vl_direct_mcp_fill_only_50_r2_20260727` |
| `2309096` | Direct MCP, submit-enabled | `qwen3vl_direct_mcp_submit_50_r2_20260724` |

All jobs use the same Qwen3-VL checkpoint, `run_0002`, 50-form set, temperature 0, 128-step cap, 160-token per-decision cap, and two-turn action history. They exclude node `i8012`, use distinct localhost inference ports, retain failed trials as outcomes (`FAIL_ON_TRIAL_FAILURE=0`), and write only to their isolated experiment directories. The direct-MCP jobs do not update the global results tracker while running.

## Campaign update (2026-07-28)

- Visual fill-only (`2309099`) completed 50/50: 138/409 all fields correct (33.7%), 0/50 full fills, and no invalid actions.
- Visual submit-enabled (`2309097`) completed 50/50: 116/409 all fields correct (28.4%), 0/50 submissions, and no invalid actions.
- Direct-MCP submit-enabled (`2309096`) completed 50/50: 262/409 all fields correct from the registered pre-submit/final-state policy (64.1%), 13/50 full fills, 32/50 successful submissions, and no invalid tool calls.
- Direct-MCP fill-only (`2309098`) reached 46/50 before Slurm's 24-hour job limit. It is not treated as a complete cohort. The missing forms are `usability_test`, `volunteer_shift`, `wellbeing_check`, and `workshop_signup`; four-form top-up `2310575` was submitted with the same experiment ID and settings.

The older proprietary-model comparison contains all four missing Qwen3-VL direct-MCP records, but they do not replace the primary cohort. They used a 32-step cap and 768-token output allowance; three used the earlier unbounded-history protocol, while only `wellbeing_check` used two-turn history. `wellbeing_check` reached the old 32-step ceiling, so its 4/7 outcome is specifically censored by a different budget. As a labelled sensitivity analysis only, combining the 46 current trials with those four historical records gives direct-MCP fill-only primary accuracy 72.9% (95% bootstrap CI 65.2%–80.6%) versus 34.1% (31.9%–36.4%) visual. The paired visual-minus-direct difference is -41.2 percentage points (95% CI -48.1 to -33.9). This supports the direction of the submit comparison but is not the registered primary fill-only result.

The strict primary analysis remains gated on 50/50 current-protocol direct-MCP fill-only summaries. The analysis generator now suppresses paired statistics unless both registered cohorts are complete, preventing the 46-form partial cohort from being published accidentally.
