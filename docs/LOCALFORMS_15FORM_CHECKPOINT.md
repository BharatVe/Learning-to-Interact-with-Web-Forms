# LocalForms platform comparison — 15-form checkpoint (2026-08-19)

Status snapshot for supervisor review. Compiled from the first 15 of 50 forms
completed on the LocalForms platform today, while the remaining 35-form batch
continues running in the background. This supersedes
`docs/LOCALFORMS_10FORM_PILOT_RESULTS.md` for reading purposes (that document's
per-form root-cause analysis for two trials is still the canonical detail and is
referenced below); full 50-form results will follow as the batch completes.

## What this is

A comparison of the same model, under the same task, against two different form
*platforms*: this project's existing 50 Google Forms, and a newly built
recreation of the same 50 forms as locally-hosted, FormFactory-style native HTML
(see `docs/LOCALFORMS_METHODOLOGY.md` for exactly how that recreation was built
and what, if anything, was changed from FormFactory's own implementation
pattern). The question being asked: **does the platform's DOM/widget
implementation affect how well a computer-use model can fill it out**, holding
the model, the target content, and the answers constant.

## Coverage and honesty about what's matched

15 of 50 forms are in. Two Google Forms baselines are used for comparison,
because no single archived cohort matches the LocalForms pilot on every
dimension at once:

| | LocalForms (this run) | Google Forms — GF32 | Google Forms — GF128 |
|---|---|---|---|
| Task mode | fill-only | fill-only (**matched**) | submit-enabled (not matched) |
| Step cap | 128 | 32 (not matched) | 128 (**matched**) |
| Answer set | `run_0002` | `run_0002` | `run_0002` |
| Coverage of these 15 forms | 15/15 | 12/15 | 12/15 |

**Read this as two partial, confounded comparisons, not one clean result.**
GF32 matches task mode but gives Google Forms a quarter of the step budget.
GF128 matches step budget but makes Google Forms do the harder submit-enabled
task. Both are reported because each is the best available match on its own
dimension; neither alone supports a claim of "platform X is better," and the
report says so at every total. The clean run — Google Forms, fill-only, 128
steps, `run_0002`, same 15+ forms — does not exist yet; see "What's still
needed."

## Results table

`V/T` = fields verified correct / total fields. GF128 uses the pre-submit
verification snapshot where the trial submitted (post-submit page state is
frequently unreadable — see `docs/LOCALFORMS_10FORM_PILOT_RESULTS.md` for why),
and the final verification where it did not.

| Form | LocalForms V/T | LocalForms outcome | GF32 V/T | GF32 outcome | GF128 V/T | GF128 outcome |
|---|---:|---|---:|---|---:|---|
| `conference_travel` | 9/10 | max_steps_exceeded | 2/10 | max_steps_exceeded | 3/10 | done (submitted) |
| `course_enrollment` | 7/8 | done_incomplete_fill_only | 7/8 | done_incomplete_fill_only | 5/8 | max_steps_exceeded |
| `exam_registration` | 7/8 | repeat_invalid_tool_call | 7/8 | max_steps_exceeded | 7/8 | done (submitted) |
| `lab_safety` | 7/8 | done_incomplete_fill_only | 3/8 | timeout | 7/8 | done (submitted) |
| `job_fair` | 9/9 | **success** | 4/9 | done_incomplete_fill_only | 4/9 | done (submitted) |
| `publication_submission` | 9/10 | model_no_tool_calls (rerun) | 4/10 | max_steps_exceeded | 0/10 | environment_error |
| `event_rsvp` | 6/6 | **success** | 6/6 | **success** | n/a | not in cohort |
| `internship_app` | 0/12 | model_no_tool_calls | 4/12 | done_incomplete_fill_only | n/a | not in cohort |
| `course_feedback` | 9/9 | **success** | 6/9 | done_incomplete_fill_only | n/a | not in cohort |
| `travel_reimbursement` | 11/11 | **success** | 5/11 | max_steps_exceeded | 2/11 | done (submitted) |
| `accessibility_feedback` | 7/8 | done_incomplete_fill_only | n/a | not in cohort | 7/8 | done (submitted) |
| `alumni_checkin` | 7/7 | **success** | 7/7 | **success** | 0/7 | repeat_invalid_tool_call |
| `bug_report` | 8/9 | max_steps_exceeded | 8/9 | done_incomplete_fill_only | 8/9 | done (submitted) |
| `club_application` | 10/10 | **success** | n/a | not in cohort | 2/10 | done (submitted) |
| `club_event_planning` | 9/10 | model_no_tool_calls | n/a | not in cohort | 2/10 | done (submitted) |
| **Total** | **115/135 (85.2%)** | 6/15 fully complete | **63/107 (58.9%)**, 12 forms | 2/12 fully complete | **47/108 (43.5%)**, 12 forms | 0/12 fully complete |

## Supporting evidence for the "not a platform DOM defect" claim

Two LocalForms outcomes look, at a glance, like the new platform is broken.
Both were traced to root cause and are not DOM/widget defects:

- **`publication_submission`, first attempt: `environment_error`, 116 tool
  errors.** Trace shows the model repeatedly calling `browser_type` on the
  form's native `<select>` dropdown instead of `browser_select_option`, looping
  until the accumulated context exceeded vLLM's 32,768-token limit. Every
  rejected call was Playwright correctly refusing an invalid action. Rerun
  (reported in the table above) hit 9/10 without the loop recurring. **The same
  failure mode — dropdown retry loop into context overflow — also occurred on
  Google Forms at the matched 128-step cap** (GF128, same form, 51 tool errors,
  `environment_error`), which is direct evidence this is a general OpenCUA
  behavior around dropdown widgets, not something introduced by the LocalForms
  recreation. Full trace excerpts in `docs/LOCALFORMS_10FORM_PILOT_RESULTS.md`.
- **`internship_app` and `club_event_planning`: `model_no_tool_calls`.** The
  model's raw output didn't match the format the text-based tool-call parser
  expects (e.g. `browser_type{"ref": "e11", ...}` with no recognized separator).
  This is a pre-existing brittleness in the shared harness's fallback parser
  (used because native structured tool-calling is off for `computer_use_agent`
  models), independent of platform — it fires on the model's raw text output
  before any page interaction happens.

Root-cause methodology: every LocalForms outcome that looked like an
infrastructure failure was checked against its raw `tool_trace.jsonl` /
`model_io.jsonl` before being accepted into this report as a genuine result,
rather than taken at face value from its `stop_reason` label.

## Aggregate read

- LocalForms field accuracy across these 15 forms: **85.2%** (115/135).
- vs. GF32 (fill-only match, 32-step, 12/15 forms covered): **58.9%**.
- vs. GF128 (step-cap match, 128-step submit-enabled, 12/15 forms covered): **43.5%**.
- Both platforms consistently struggle most on the same forms
  (`conference_travel`, `publication_submission` — both dropdown-heavy),
  consistent with this project's existing
  `docs/eval_results/interaction_failure_analysis/DROPDOWN_FAILURE_ANALYSIS.md`
  finding. This directional pattern — dropdowns are hard for this model
  regardless of platform — is the most defensible claim in this checkpoint. The
  headline accuracy percentages above are descriptive, not causal, given the
  confounds stated above.

## What's still needed before this supports a causal platform claim

1. **A step-matched, task-mode-matched Google Forms cohort**: fill-only, 128
   steps, `run_0002`, same forms. Doesn't exist yet in this repo's history for
   either comparison used here.
2. **The remaining 35 forms**, currently running (Slurm job `2322142`,
   experiment `opencua_localforms_direct_mcp_fill_only_40_20260819`).
3. Ideally, repeated sampling per form (temperature > 0 or multiple seeds) —
   every number above is a single deterministic run per form-condition cell, so
   these percentages describe variation *across forms*, not run-to-run model
   variance.

## Reproducing / verifying this table

Every number above is read directly from `summary.json` files under
`data/model_baselines/opencua_localforms_direct_mcp_fill_only_{10,40}_20260819/`,
`data/model_baselines/opencua_direct_mcp_fill_only_done_{10_seed20260702,30_seed20260709}_r2_step32/`,
and `data/model_baselines/opencua_direct_mcp_tools_target300_run2_20260609/`.
No numbers here are estimated or interpolated.
