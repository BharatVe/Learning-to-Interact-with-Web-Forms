# LocalForms platform comparison — full 50-form fill-only results (2026-08-19)

Update to `docs/LOCALFORMS_15FORM_CHECKPOINT.md`: the 40-form batch
(`opencua_localforms_direct_mcp_fill_only_40_20260819`) finished while that
checkpoint was being written, completing the full 50-form fill-only LocalForms
cohort. This document is the complete version of that comparison. It does not
replace the 15-form checkpoint's root-cause analysis of individual trials
(`docs/LOCALFORMS_10FORM_PILOT_RESULTS.md` and the checkpoint both still hold
the detailed trace-level diagnosis) — it extends the same comparison to all 50
forms.

## Conditions

Unchanged from the checkpoint: `computer_use_opencua_32b_direct_mcp`
(OpenCUA-32B), direct Playwright MCP tools, `run_0002`, temperature 0. Two
Google Forms comparisons, neither matching LocalForms on every dimension (see
`docs/LOCALFORMS_15FORM_CHECKPOINT.md` for the full explanation of why both are
reported rather than one being picked as "the" baseline):

- **GF32** — fill-only match, 32-step cap. Assembled from three incremental
  Google Forms cohorts that together give full 50-form coverage:
  `opencua_direct_mcp_fill_only_done_10_seed20260702_r2_step32`,
  `opencua_direct_mcp_fill_only_done_30_seed20260709_r2_step32`,
  `opencua_direct_mcp_fill_only_done_50_topup20_20260713_r2_step32`.
- **GF128** — step-cap match, 128-step, submit-enabled:
  `opencua_direct_mcp_tools_target300_run2_20260609`. Covers 45/50 forms
  (`conf_interest`, `course_feedback`, `event_rsvp`, `internship_app`,
  `workshop_signup` are not in this cohort).

## Aggregate results

| Condition | Verified / Total | Accuracy | Coverage |
|---|---:|---:|---:|
| LocalForms (fill-only, 128-step) | 300/409 | **73.3%** | 50/50 |
| GF32 (fill-only, 32-step) | 289/409 | **70.7%** | 50/50 |
| GF128 (submit-enabled, 128-step) | 244/368 | **66.3%** | 45/50 |

**This is the important correction to make before anything else is read from
this table: at full 50-form scale, the gap between LocalForms and GF32 that
looked large in the 15-form checkpoint (85.2% vs. 58.9%) has almost entirely
closed (73.3% vs. 70.7%).** The 15-form subset was not representative — it
happened to include a disproportionate share of forms where LocalForms did
well. This is exactly why the checkpoint document said its numbers were
provisional and the full batch needed to finish before drawing conclusions.
With all 50 forms in, the fill-only, task-matched comparison shows LocalForms
and Google Forms within about 2.6 percentage points of each other.

LocalForms fully-complete forms: 22/50. Stop-reason distribution across all 50
LocalForms trials:

| `stop_reason` | Count |
|---|---:|
| `filled_without_submit` (success) | 22 |
| `max_steps_exceeded` | 9 |
| `environment_error` | 7 |
| `model_no_tool_calls` | 6 |
| `done_incomplete_fill_only` | 5 |
| `repeat_invalid_tool_call` | 1 |

## A known undercount: 7 forms need rerunning

Seven forms — `data_annotation`, `dataset_request`, `field_trip`,
`office_hours`, `peer_evaluation`, `purchase_request`, `remote_setup` — hit
`environment_error` and are recorded as `0/N` verified. Per the diagnosis
already established for `publication_submission` in the 10-form pilot (see
`docs/LOCALFORMS_10FORM_PILOT_RESULTS.md`), `environment_error` fires when an
unhandled exception (typically: vLLM's context-length limit, hit after a
dropdown-retry loop) aborts the trial *before* the harness's final verification
pass runs — so `0/N` likely understates what was actually filled correctly, not
just what was scored. `publication_submission`'s own rerun went from 0/10 to
9/10 once the loop didn't recur. **These 7 are queued to be rerun** the same
way — attaching to a live job's already-warm vLLM server via
`srun --overlap` rather than a fresh cold start — and this document will be
updated with corrected numbers once that finishes. Until then, the 73.3%
LocalForms figure above should be read as a likely **floor**, not the true
number.

## Per-form results

`V/T` = verified-correct fields / total fields. GF128 column uses the
pre-submit verification snapshot for submitted trials (see the 15-form
checkpoint for why), final verification otherwise.

| Form | LocalForms V/T | LocalForms outcome | GF32 V/T | GF32 outcome | GF128 V/T | GF128 outcome |
|---|---:|---|---:|---|---:|---|
| `accessibility_feedback` | 7/8 | done_incomplete_fill_only | 6/8 | max_steps_exceeded | 7/8 | done |
| `alumni_checkin` | 7/7 | **success** | 7/7 | **success** | 0/7 | repeat_invalid_tool_call |
| `bug_report` | 8/9 | max_steps_exceeded | 8/9 | done_incomplete_fill_only | 8/9 | done |
| `club_application` | 10/10 | **success** | 3/10 | done_incomplete_fill_only | 2/10 | done |
| `club_event_planning` | 9/10 | model_no_tool_calls | 2/10 | done_incomplete_fill_only | 2/10 | done |
| `conf_interest` | 7/7 | **success** | 7/7 | **success** | n/a | not in cohort |
| `conference_travel` | 9/10 | max_steps_exceeded | 2/10 | max_steps_exceeded | 3/10 | done |
| `course_enrollment` | 7/8 | done_incomplete_fill_only | 7/8 | done_incomplete_fill_only | 5/8 | max_steps_exceeded |
| `course_feedback` | 9/9 | **success** | 6/9 | done_incomplete_fill_only | n/a | not in cohort |
| `data_annotation` | 0/9 → *rerun queued* | environment_error | 4/9 | done_incomplete_fill_only | 4/9 | done |
| `dataset_request` | 0/9 → *rerun queued* | environment_error | 4/9 | done_incomplete_fill_only | 4/9 | done |
| `equipment_checkout` | 8/9 | max_steps_exceeded | 8/9 | done_incomplete_fill_only | 8/9 | timeout |
| `event_rsvp` | 6/6 | **success** | 6/6 | **success** | n/a | not in cohort |
| `exam_registration` | 7/8 | repeat_invalid_tool_call | 7/8 | max_steps_exceeded | 7/8 | done |
| `experiment_booking` | 8/9 | max_steps_exceeded | 4/9 | done_incomplete_fill_only | 3/9 | done |
| `field_trip` | 0/8 → *rerun queued* | environment_error | 7/8 | done_incomplete_fill_only | 7/8 | done |
| `hackathon_signup` | 0/9 | model_no_tool_calls | 5/9 | done_incomplete_fill_only | 5/9 | done |
| `housing_preference` | 7/7 | **success** | 7/7 | **success** | 7/7 | done |
| `internship_app` | 0/12 | model_no_tool_calls | 3/12 | done_incomplete_fill_only | n/a | not in cohort |
| `job_fair` | 9/9 | **success** | 4/9 | done_incomplete_fill_only | 4/9 | done |
| `lab_safety` | 7/8 | done_incomplete_fill_only | 7/8 | done_incomplete_fill_only | 7/8 | done |
| `lab_visit` | 8/8 | **success** | 8/8 | **success** | 7/8 | max_steps_exceeded |
| `language_exchange` | 0/9 | model_no_tool_calls | 5/9 | done_incomplete_fill_only | 5/9 | done |
| `library_membership` | 7/7 | **success** | 7/7 | **success** | 7/7 | done |
| `meal_plan` | 7/7 | **success** | 7/7 | **success** | 7/7 | done |
| `mentor_match` | 7/7 | **success** | 7/7 | **success** | 7/7 | done |
| `newsletter_signup` | 4/7 | max_steps_exceeded | 6/7 | done_incomplete_fill_only | 6/7 | done |
| `office_hours` | 0/8 → *rerun queued* | environment_error | 7/8 | done_incomplete_fill_only | 7/8 | done |
| `orientation_signup` | 8/8 | **success** | 8/8 | **success** | 8/8 | done |
| `paper_review` | 6/7 | model_no_tool_calls | 6/7 | done_incomplete_fill_only | 6/7 | done |
| `peer_evaluation` | 0/7 → *rerun queued* | environment_error | 6/7 | timeout | 6/7 | done |
| `project_update` | 8/8 | **success** | 8/8 | **success** | 7/8 | model_no_tool_calls |
| `publication_submission` | 9/10 | model_no_tool_calls (rerun) | 4/10 | max_steps_exceeded | 0/10 | environment_error |
| `purchase_request` | 0/10 → *rerun queued* | environment_error | 3/10 | done_incomplete_fill_only | 3/10 | timeout |
| `remote_setup` | 0/7 → *rerun queued* | environment_error | 6/7 | done_incomplete_fill_only | 6/7 | done |
| `research_interest` | 7/7 | **success** | 7/7 | **success** | 7/7 | done |
| `room_booking` | 9/9 | **success** | 3/9 | done_incomplete_fill_only | 3/9 | done |
| `scholarship_interest` | 6/7 | max_steps_exceeded | 6/7 | done_incomplete_fill_only | 6/7 | done |
| `seminar_proposal` | 8/8 | **success** | 4/8 | done_incomplete_fill_only | 4/8 | done |
| `software_access` | 7/8 | done_incomplete_fill_only | 7/8 | done_incomplete_fill_only | 7/8 | done |
| `sports_tournament` | 8/9 | done_incomplete_fill_only | 5/9 | done_incomplete_fill_only | 5/9 | done |
| `study_group_match` | 7/7 | **success** | 7/7 | **success** | 7/7 | done |
| `survey_consent` | 7/8 | max_steps_exceeded | 7/8 | done_incomplete_fill_only | 7/8 | done |
| `technical_support` | 7/8 | max_steps_exceeded | 6/8 | max_steps_exceeded | 7/8 | done |
| `thesis_meeting` | 6/8 | max_steps_exceeded | 6/8 | max_steps_exceeded | 7/8 | done |
| `travel_reimbursement` | 11/11 | **success** | 5/11 | max_steps_exceeded | 2/11 | done |
| `usability_test` | 7/7 | **success** | 7/7 | **success** | 7/7 | done |
| `volunteer_shift` | 7/7 | **success** | 7/7 | **success** | 7/7 | done |
| `wellbeing_check` | 7/7 | **success** | 3/7 | done_incomplete_fill_only | 3/7 | done |
| `workshop_signup` | 7/7 | **success** | 7/7 | **success** | n/a | not in cohort |

## Reading this table honestly

- The revised, full-scale picture: **LocalForms and Google Forms perform
  similarly under matched task conditions** (73.3% vs. 70.7%), not
  dramatically differently as the smaller sample suggested. That is itself a
  meaningful methodological finding for the thesis — it suggests the
  platform's DOM implementation (native HTML vs. Google's custom ARIA widgets)
  is not the dominant driver of these scores; model behavior (dropdown
  handling, tool-call formatting, context-budget exhaustion under retry loops)
  is, and it shows up on both platforms.
- The GF128 comparison remains confounded by submit-enabled task mode, as
  before, and should not be read as a platform effect.
- The 7 pending reruns will very likely raise the LocalForms number further
  (each has a GF32 counterpart scoring 4–7 correct fields, suggesting the
  underlying forms are not unusually hard — the 0s are a scoring artifact of
  an aborted trial, not a demonstrated model or platform failure).
- Submit-enabled LocalForms runs (mirroring GF128's task mode properly, i.e. a
  true same-task-mode-and-step-cap comparison) are in progress as of this
  writing; see the session log / job queue for current status.

## Reproducing

Same sources as the 15-form checkpoint, now with
`opencua_direct_mcp_fill_only_done_50_topup20_20260713_r2_step32` added for
full GF32 coverage. All figures read directly from `summary.json`, no manual
edits.
