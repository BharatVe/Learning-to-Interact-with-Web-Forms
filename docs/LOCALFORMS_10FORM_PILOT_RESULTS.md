# LocalForms 10-form pilot: results and comparison to Google Forms

Companion to `docs/LOCALFORMS_METHODOLOGY.md` (how the platform was built) and
`docs/ALTERNATIVE_PLATFORM_PLAN.md` (why). This note reports the first pilot
result and compares it against the two closest available Google Forms
baselines in this repo's history — one matched on task mode but not step cap,
one matched on step cap but not task mode. No cohort matching both exists yet
(see "Next step"), so both are reported, each with its own confound stated
plainly rather than picking whichever looks cleaner.

## Conditions compared

| | LocalForms pilot | Google Forms — fill-only match | Google Forms — step-cap match |
|---|---|---|---|
| Model | `computer_use_opencua_32b_direct_mcp` (OpenCUA-32B) | same | same |
| Interface | Direct Playwright MCP tools | same | same |
| Task mode | `fill_only_done` | `fill_only_done` (matched) | `fill_and_submit` (**not** matched) |
| Answer set | `run_0002` | `run_0002` | `run_0002` (matched) |
| Temperature | 0 | 0 | 0 |
| **Step cap** | **128** | **32** (**not** matched) | **128** (matched) |
| Experiment ID | `opencua_localforms_direct_mcp_fill_only_10_20260819` | `opencua_direct_mcp_fill_only_done_10_seed20260702_r2_step32` (8/10 forms) + `opencua_direct_mcp_fill_only_done_30_seed20260709_r2_step32` (`event_rsvp`, `publication_submission`) | `opencua_direct_mcp_tools_target300_run2_20260609` (7/10 forms) |

**Neither comparison is a clean platform-only test.** The fill-only match uses a
32-step cap instead of 128, so a larger retry budget can inflate LocalForms'
numbers on its own. The step-cap match uses submit-enabled task mode instead of
fill-only, so extra step spend on locating/clicking submit can deflate Google
Forms' numbers on its own. Both tables are given below; read them as two
different partial views, not as two confirmations of the same effect.

## Comparison 1: fill-only match (32-step Google Forms baseline)

`V/T` = verified-correct fields / total fields for that form.

| Form | LocalForms V/T | LocalForms outcome | Google Forms V/T (32-step) | Google Forms outcome |
|---|---:|---|---:|---|
| `conference_travel` | 9/10 | `max_steps_exceeded` (dropdown retry loop, 52 tool errors) | 2/10 | `max_steps_exceeded` |
| `course_enrollment` | 7/8 | `done_incomplete_fill_only` | 7/8 | `done_incomplete_fill_only` |
| `exam_registration` | 7/8 | `repeat_invalid_tool_call` (4 invalid calls) | 7/8 | `max_steps_exceeded` (33 tool errors) |
| `lab_safety` | 7/8 | `done_incomplete_fill_only` | 3/8 | `timeout` |
| `job_fair` | 9/9 | **success** (`filled_without_submit`) | 4/9 | `done_incomplete_fill_only` |
| `publication_submission` | 9/10 | `model_no_tool_calls` after filling 9/10 correctly (rerun; see below) | 4/10 | `max_steps_exceeded` |
| `event_rsvp` | 6/6 | **success** | 6/6 | **success** |
| `internship_app` | 0/12 | `model_no_tool_calls` (malformed first tool call, see below) | 4/12 | `done_incomplete_fill_only` |
| `course_feedback` | 9/9 | **success** | 6/9 | `done_incomplete_fill_only` |
| `travel_reimbursement` | 11/11 | **success** | 5/11 | `max_steps_exceeded` |
| **Total** | **74/91 (81.3%)** | 4/10 fully complete | **48/91 (52.7%)** | 1/10 fully complete |

`publication_submission`'s LocalForms figure reflects the completed rerun (see
"Live redo status," now resolved): the original attempt's `environment_error`
did not recur; the model filled 9/10 fields correctly in one batched call, then
stopped responding in a recognized tool-call format on the next step (the same
`model_no_tool_calls` parser issue as `internship_app`, not a repeat of the
original dropdown loop). Its `success` flag is still `False` because the harness
requires all fields verified, not 9/10 — the number here is the honest field
count, not a rounded-up "pass."

## Two LocalForms outcomes needed root-causing before trusting them

Both looked like platform/infrastructure failures at first glance
(`environment_error`, and a zero-tool-call trial); neither turned out to be a
LocalForms DOM/widget defect on inspection of the raw tool traces:

- **`publication_submission` — `environment_error`.** The trace shows the model
  repeatedly calling `browser_type` on the form's `<select>` dropdown instead of
  `browser_select_option` (same failure class as `conference_travel`'s
  `max_steps_exceeded`, both dropdown-related). It looped long enough — 125 tool
  calls — that the accumulated conversation context exceeded vLLM's configured
  32,768-token limit, which raises a hard HTTP 400 that this harness's generic
  exception handler labels `environment_error`. Every one of the 116 rejected
  calls was Playwright correctly refusing an invalid action; the interface did
  not misbehave at any point. This is a model tool-choice loop that happened to
  end in a context-limit exception rather than the step cap, not a platform bug.
  **A live rerun of this one trial is in progress** (see status note below) to
  check whether it reproduces or was a one-off amplified by an unusually long
  retry streak.
- **`internship_app` — `model_no_tool_calls`.** Failed at step 0, before any
  page interaction. The model's raw output was
  `browser_type{"ref": "e11", "text": "Jordan Kim"}` — a tool-call format the
  text-based fallback parser (used because `native_tool_calls_enabled=False` for
  `computer_use_agent`) didn't recognize. This is a pre-existing parser
  brittleness in the shared harness code (`run_qwen_direct_mcp_eval.py`'s
  text-tool-call fallback), independent of which platform is being filled — it
  happened before the model had seen any LocalForms-specific markup.

Both are documented here as genuine pilot outcomes, not swept into "environment
error, discard" — per the working rule for this pilot: rerun trials whose
*harness/infrastructure* broke before or independent of a real model action;
keep trials where the model interacted correctly and simply performed poorly or
looped, since that is signal, not noise.

## Live redo status (resolved)

`publication_submission` was rerun by attaching to the in-flight 40-form batch
job's already-warm vLLM server (`srun --jobid=2322142 --overlap`, reusing its
existing GPU allocation and OpenAI-compatible endpoint directly rather than
queuing a new Slurm job and paying another ~13-minute vLLM cold start). Outcome:
9/10 fields verified correct, `model_no_tool_calls` (see per-form table above).
The original dropdown-loop-into-context-overflow failure did not reproduce.

## Comparison 2: step-cap match (128-step, submit-enabled Google Forms baseline)

The 32-step comparison above was the closest same-task-mode match found
initially, but a genuine 128-step-cap OpenCUA direct-MCP Google Forms cohort at
`run_0002` also exists in this repo's history:
`opencua_direct_mcp_tools_target300_run2_20260609`
(`DIRECT_MCP_MAX_STEPS=128`, confirmed directly from the launch config in
`scripts/submit_eval_target_chain.py:196-206`). This is the right cohort to use
for a step-matched comparison. It has two limitations of its own, both stated
plainly rather than absorbed into the numbers:

1. **It only covers 7 of the 10 pilot forms** at `run_0002`
   (`event_rsvp`, `internship_app`, `course_feedback` are absent from this
   cohort — not run, not just filtered out).
2. **It is submit-enabled (`fill_and_submit`), not `fill_only_done`.** No
   `--fill-only-done` flag was passed for this cohort. This repo's own prior
   documentation (`docs/eval_results/FORMFACTORY_QWEN3VL_COMPARISON.md`) already
   flags why this matters: post-submit page state is frequently unreadable after
   navigation, so the raw `verified_correctness` field for a *successfully
   submitted* trial reads misleadingly low (often near 0) and the correct field
   to read instead is `pre_successful_submit_verified_correctness` — the
   verification snapshot taken immediately before the submit click. The table
   below uses `pre_successful_submit_verified_correctness` for trials that
   submitted, and `verified_correctness` for trials that never reached submit
   (`max_steps_exceeded`, `environment_error`), which is the only reading
   available for those.

| Form | LocalForms V/T (fill-only, 128-step) | Google Forms V/T (submit-enabled, 128-step) | Google Forms outcome |
|---|---:|---:|---|
| `conference_travel` | 9/10 | 3/10 | `done` (submitted; pre-submit reading) |
| `course_enrollment` | 7/8 | 5/8 | `max_steps_exceeded` (not submitted) |
| `exam_registration` | 7/8 | 7/8 | `done` (submitted; pre-submit reading) |
| `lab_safety` | 7/8 | 7/8 | `done` (submitted; pre-submit reading) |
| `job_fair` | 9/9 | 4/9 | `done` (submitted; pre-submit reading) |
| `publication_submission` | 9/10 | 0/10 | `environment_error` — same dropdown-retry-into-context-overflow pattern, 51 tool errors, on Google Forms too (not submitted) |
| `travel_reimbursement` | 11/11 | 2/11 | `done` (submitted; pre-submit reading) |
| **Total (7 forms only)** | **59/64 (92.2%)** | **28/64 (43.8%)** | |

Two things make this table *not* a clean platform effect either, and both
should travel with any citation of it:

- **Task mode is now the confound instead of step cap.** Submit-enabled trials
  must locate and click a working submit control in addition to filling every
  field, and can burn steps on that before hitting the cap — LocalForms' 92.2%
  is under the easier fill-only task, not a same-task comparison.
- `publication_submission` hitting `environment_error` on **both** platforms at
  128 steps (LocalForms: 116 tool errors before the rerun; Google Forms: 51 tool
  errors here) is good corroborating evidence that the dropdown-driven
  `browser_type`-on-`<select>` loop is a general OpenCUA behavior, not something
  the LocalForms recreation introduced — that part of the finding is credible
  independent of the task-mode confound.

## Aggregate read — both comparisons, both caveated

- **vs. 32-step, fill-only Google Forms** (10/10 forms): LocalForms 81.3%
  (74/91) vs. Google Forms 52.7% (48/91). Confounded by step cap (128 vs. 32).
- **vs. 128-step, submit-enabled Google Forms** (7/10 forms): LocalForms 92.2%
  (59/64, fill-only) vs. Google Forms 43.8% (28/64, submit-enabled). Confounded
  by task mode (fill-only vs. submit-enabled).
- Both platforms hit their worst outcomes on the same two forms
  (`conference_travel`, `publication_submission`) — both contain dropdowns, and
  both platforms show the model struggling with dropdown interaction
  specifically, consistent with this project's existing
  `docs/eval_results/interaction_failure_analysis/DROPDOWN_FAILURE_ANALYSIS.md`
  finding that dropdowns are hard for these models generally, on either
  platform. This directional pattern is the most defensible part of the pilot;
  the headline accuracy percentages above are not, given the confounds.

## Next step for a clean claim

Neither table above isolates platform. The needed run is: Google Forms,
`fill_only_done`, 128-step cap, `run_0002`, temperature 0, the same 10 forms —
identical to the LocalForms pilot's settings in every dimension except
platform. Until that specific cohort exists, both tables in this report should
be cited as descriptive/pilot evidence with their stated confound, not as an
isolated platform effect.
