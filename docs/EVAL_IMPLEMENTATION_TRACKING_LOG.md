# Evaluation Implementation Tracking Log

Last updated: 2026-05-28

Use this file as the current source of truth before rerunning evaluation jobs. It records the active evaluation design, implementation status, latest outputs, known issues, and next actions.

## Current Decision

- Qwen text/VLM thesis track: `direct_mcp_tool_use`.
- Qwen models interact through documented Playwright MCP tools, using `browser_snapshot` refs as the primary UI substrate.
- OpenCUA thesis track: `computer_use_native`.
- OpenCUA remains screenshot-native with pyautogui/action-output style and Qwen2.5-VL coordinate conversion.
- Do not aggregate Qwen direct-MCP and OpenCUA native as one interface condition; report them as different web-form completion systems.
- Scripted Playwright generation runs remain the efficiency reference baseline for matching `form_id + run_XXXX`.

## Current State

The serving and basic orchestration problems are mostly solved. The remaining blockers are agent behavior and metric interpretation.

- Qwen direct-MCP now produces usable submitted trials and pre-submit scoring snapshots.
- Qwen still frequently stops with `DONE` before submission, especially the text model.
- Qwen VLM can time out on some longer forms despite making partial progress.
- OpenCUA serving is working, but the native screenshot agent still fails to submit and usually loops on repeated clicks around Google Forms controls.
- Final post-submit verification still reads as `0/N` on submitted Qwen trials because the browser is on the Google Forms confirmation page. For submitted Qwen trials, use `scored_correctness` from `pre_successful_submit_verified_correctness`.
- Latest reruns add CUDA preflight checks before vLLM startup, better OpenCUA vLLM launch diagnostics, and `FORM_OFFSET`/`FORM_LIMIT` support for `FORM_IDS=all`.
- Latest analysis artifacts live under `docs/eval_results/analysis/`.

## Implementation Map

### English Browser/Form Default

Files:

- `src/engine/browser_language.py`
- `src/engine/runner.py`
- `src/engine/mcp_browser_engine.py`
- `src/engine/form_engine.py`
- `src/baselines/run_baseline_eval.py`
- `src/baselines/run_direct_api_eval.py`
- `src/baselines/run_gemini_native_computer_use_eval.py`
- `src/baselines/run_qwen_direct_mcp_eval.py`
- `src/baselines/run_opencua_direct_eval.py`
- `scripts/sync_generator_dataset.py`

Current behavior:

- Google Forms URLs default to `hl=en`.
- Local Playwright contexts use locale `en-US`.
- Local Playwright contexts send `Accept-Language: en-US,en;q=0.9`.
- Playwright MCP receives a generated config with locale `en-US`, the same `Accept-Language`, and Chromium launch arg `--lang=en-US`.
- Dataset sync writes Google Forms URLs with `hl=en`.
- Submit-success detection still accepts German confirmation text for historical/already-running pages.

Reason:

- Job `2212084` loaded Google Forms confirmation pages in German. That caused a real submitted page to be treated as a failed submit path because the English-only confirmation detector did not match the page text.

### Qwen Direct-MCP Runner

File: `src/baselines/run_qwen_direct_mcp_eval.py`

Current behavior:

- Uses Playwright MCP directly instead of the legacy mediated `human_ui_v1` path.
- Prompts the model to inspect `browser_snapshot` and call documented browser tools with snapshot refs.
- Exposes `browser_check` for radio/checkbox controls.
- Hides `browser_select_option` unless a visible control is a real HTML `select`.
- Tracks submit attempts and stores:
  - `submit_attempt_count`
  - `successful_submit_attempt_count`
  - `failed_submit_attempt_count`
  - `submitted_while_incomplete_count`
  - `first_submit_step`
  - `pre_first_submit_verified_correctness`
  - `pre_successful_submit_verified_correctness`
- Terminal contract: `DONE` means an observed submission confirmation page, not "ready to submit".
- A `browser_click` that reaches the confirmation page is treated as a submit attempt even if the clicked ref was not recognized as a submit control before the click.

Important metric caveat:

- `verified_correctness` can be `0/N` after a successful submission because final verification runs on the confirmation page.
- `scored_correctness` in `docs/eval_results` correctly prefers `pre_successful_submit_verified_correctness` when available.
- `submitted_while_incomplete_count` is currently suspect for Qwen because it is derived from a post-click verification snapshot that can also be on the confirmation page. Interpret it only after checking the corresponding pre-submit correctness fields.

Current behavioral issues:

- Text Qwen often fills many fields correctly, then replies `DONE` without submitting.
- VLM Qwen submits more often, but sometimes spends a very long time before submission or times out after partial progress.
- The direct-MCP implementation is usable for evaluation, but the terminal behavior is still an experimental variable to discuss in the thesis.

### OpenCUA Native Runner

File: `src/baselines/run_opencua_direct_eval.py`

Current behavior:

- Defaults to screenshot-native prompt input.
- Excludes symbolic interaction maps by default.
- Keeps pyautogui-style outputs and Qwen2.5 coordinate conversion.
- Supports `--include-symbolic-support` only for ablation/debug.
- Records premature/incomplete submits as metrics instead of blocking them in code.
- Prompts the model to work top-to-bottom, use one GUI action at a time, scroll when the target is not visible, and change strategy if state does not change.

Current behavioral issue:

- The model usually completes a few early text entries, then falls into repeated click loops.
- In the latest 20-form run, all trials hit the 128-step cap and none submitted.
- Example traces show 120+ clicks with only 2-4 text-entry actions on several forms.
- This is not a serving failure. It is now an agent/control-loop failure.
- The first forms in job `2228808` showed a more specific pattern: text fields are often filled, but radio, checkbox, and dropdown controls trigger repeated clicks or typed option values in the wrong place.

Recent implementation fix:

- The runner now detects repeated identical click/wait action signatures and terminates with `repeated_action_loop` instead of burning the full step cap.
- This keeps the main baseline screenshot-native while making failures easier to classify.
- The prompt now includes a small Google Forms control guide:
  - Do not type option values into radio, checkbox, or dropdown controls.
  - For radio/checkbox answers, click the visible circle/box or exact option label once, then use a different point if no mark appears.
  - For dropdown answers, click the dropdown field first, then click the exact option text in the opened list.
  - If the target option is not visible, scroll instead of repeating the same coordinate.

### SLURM Matrix Scripts

Files:

- `scripts/run_opencua_direct_matrix.sh`
- `scripts/slurm_opencua_direct.sbatch`
- `scripts/run_opencua_vllm_server.sh`
- `scripts/run_qwen_direct_mcp_matrix.sh`

Current behavior:

- OpenCUA and Qwen matrix scripts support `FORM_IDS=all` with `FORM_OFFSET` and `FORM_LIMIT`.
- OpenCUA SLURM jobs run a CUDA preflight before starting vLLM and fail early if CUDA is unavailable.
- Qwen direct-MCP jobs run the same CUDA preflight before each text/VLM vLLM server launch.
- OpenCUA vLLM launch logs now include the vLLM binary/version, model, served model name, port, tensor-parallel settings, CUDA devices, and exact command.

Reason:

- Job `2220805` failed during text-model vLLM startup with `cudaGetDeviceCount Error 802` before trials began.
- The previous comma-separated `FORM_IDS` export was vulnerable to Slurm `--export` parsing; offset/limit selection avoids that for larger sorted batches.

### Consolidated Results Tracker

Files:

- `scripts/update_eval_results_tracker.py`
- `docs/eval_results/README.md`
- `docs/eval_results/metrics.csv`
- `docs/eval_results/metrics.jsonl`

Current behavior:

- Scans `data/model_baselines/**/summary.json`.
- Writes one consolidated CSV, JSONL, and Markdown summary.
- Rebuilds from all summaries, so rerunning after a new job naturally adds rows.
- Supports `--experiment-id` filters.
- Current tracker is filtered to `qwen_direct_mcp_english_stepcap128_5form_20260515`.
- Current tracker has `188` Qwen trial rows.
- Adds `scored_correctness`, `scored_correctness_source`, and `metric_warning`.
- `scored_correctness` prefers pre-submit snapshots for submitted trials.

Command:

```bash
.venv/bin/python scripts/update_eval_results_tracker.py --experiment-id qwen_direct_mcp_english_stepcap128_5form_20260515
```

### Evaluation Analysis Artifacts

Files:

- `scripts/analyze_eval_results.py`
- `docs/eval_results/analysis/latest_analysis.md`
- `docs/eval_results/analysis/canonical_trials.csv`
- `docs/eval_results/analysis/answer_validation.csv`
- `docs/eval_results/analysis/experiment_coverage.csv`
- `docs/eval_results/analysis/model_summary.csv`
- `docs/eval_results/analysis/form_summary.csv`
- `docs/eval_results/analysis/failure_summary.csv`
- `docs/eval_results/analysis/cohort_summary.csv`
- `docs/eval_results/analysis/per_form_summary.csv`
- `docs/eval_results/analysis/question_type_summary.csv`
- `docs/eval_results/analysis/plots/model_overview.svg`
- `docs/eval_results/analysis/plots/stop_reasons.svg`
- `docs/eval_results/analysis/plots/opencua_control_forms.svg`
- `docs/eval_results/analysis/plots/qwen_latest_forms.svg`

Current headline:

- Canonical thesis-facing index: `274` rows covering all discovered Qwen direct-MCP and OpenCUA native summaries for the three evaluated model IDs.
- The canonical index includes `8` partial rows from running jobs `2240221` and `2240222`; the primary model summary excludes known incomplete fixed-size batches until their expected trial count is reached.
- Answer-set validation is currently `274/274 ok`: every indexed trial points to the expected `data/answers/<form_id>/runs.json` run index and has a matching saved `answers_instance.json`.
- Metadata/path validation is currently `274/274 ok`: summary metadata matches the artifact path fields for experiment, model, form, answer run, and trial.
- Primary analysis tables are now model/form/run-first. `canonical_trials.csv` starts with `model_id`, `model_kind`, `track`, `form_id`, `answer_run_id`, and `trial_id`; `model_summary.csv`, `form_summary.csv`, `question_type_summary.csv`, and `failure_summary.csv` also keep the model fields first and move result-set labels to `analysis_scope`.
- Qwen direct-MCP grouping now includes completed experiments whose IDs start with `qwen_direct_mcp_`; OpenCUA control-guidance grouping includes completed experiments whose IDs start with `opencua_control_guidance`.
- Qwen latest completed batch `2228977`: `8` new trial rows, `7/8` submit success, `36/64` scored correctness. `workshop_signup` was skipped because run-1 summaries already existed.
- Qwen all current clean experiment: `188` trials, `111/188` submit success, `898/1542` scored correctness.
- OpenCUA control-guidance all-50: `50` trials, `0/50` submit success, `157/409` scored correctness, `43` repeated-action loops and `7` max-step failures.
- The analysis report now includes canonical trial, model, form, failure, interaction-count, incomplete-submit, and question-type tables.

Command:

```bash
.venv/bin/python scripts/analyze_eval_results.py
```

## Latest Output Analysis

### Qwen Job `2218695`

Experiment: `qwen_direct_mcp_english_stepcap128_5form_20260515`

Forms:

- `meal_plan`
- `mentor_match`
- `newsletter_signup`
- `office_hours`
- `orientation_signup`
- `paper_review`
- `peer_evaluation`
- `project_update`
- `publication_submission`
- `purchase_request`

SLURM/log status:

- The VLM vLLM server log `logs/slurm/vlm_qwen3_vl_30b_a3b_instruct-direct-mcp-vllm-2218695.log` shows successful model serving and clean shutdown.
- The meaningful failures are trial-level `timeout` and `premature_done_without_submit`, not vLLM/server crashes.
- The Qwen matrix refreshed `docs/eval_results`.

Subset result for these 10 forms:

| Model | Trials | Submit Success | Scored Correctness | Stop Reasons |
|---|---:|---:|---:|---|
| `text_qwen3_30b_a3b_instruct_2507` | 10 | 5 | `58/79` (`73.4%`) | `done`: 5, `premature_done_without_submit`: 4, `timeout`: 1 |
| `vlm_qwen3_vl_30b_a3b_instruct` | 10 | 6 | `56/79` (`70.9%`) | `done`: 6, `timeout`: 2, `premature_done_without_submit`: 2 |

Interpretation:

- This is a stronger batch than earlier Qwen batches by scored correctness.
- Submission remains the main weakness; correctness is often high before the model stops or times out.
- VLM is not uniformly better here: it submits slightly more often, but it also timed out twice.

### Full Qwen Consolidated Tracker

Tracker: `docs/eval_results/metrics.csv`

Current totals:

| Model | Trials | Submit Success | Success | Scored Correctness | Stop Reasons |
|---|---:|---:|---:|---:|---|
| `text_qwen3_30b_a3b_instruct_2507` | 80 | 36 | 36 | `386/660` (`58.5%`) | `premature_done_without_submit`: 42, `done`: 36, `max_steps_exceeded`: 1, `timeout`: 1 |
| `vlm_qwen3_vl_30b_a3b_instruct` | 80 | 58 | 58 | `382/660` (`57.9%`) | `done`: 58, `premature_done_without_submit`: 19, `timeout`: 3 |

Interpretation:

- VLM has a materially higher submit rate: `58/80` vs `36/80`.
- Overall scored correctness is very close across text and VLM.
- Text Qwen's main failure mode is premature `DONE`.
- VLM Qwen's main failure mode is still premature `DONE`, with a smaller but real timeout tail.

### OpenCUA Job `2218699`

Experiment: `opencua_topdown_prompt_20form_20260519`

SLURM/log status:

- Produced `20` summaries.
- Serving and model calls worked.
- All trials completed as failed benchmark trials rather than infrastructure crashes.

Results:

- Trials: `20`
- Submit successes: `0`
- Successes: `0`
- Stop reasons: `20 x max_steps_exceeded`
- Scored/verified correctness: `65/173` (`37.6%`)
- Each trial used all `128` actions.

Per-form correctness:

| Form | Correct | Total |
|---|---:|---:|
| `accessibility_feedback` | 2 | 8 |
| `alumni_checkin` | 4 | 7 |
| `bug_report` | 4 | 9 |
| `club_application` | 4 | 10 |
| `club_event_planning` | 3 | 10 |
| `conf_interest` | 3 | 7 |
| `conference_travel` | 3 | 10 |
| `course_enrollment` | 3 | 8 |
| `course_feedback` | 2 | 9 |
| `data_annotation` | 3 | 9 |
| `dataset_request` | 4 | 9 |
| `equipment_checkout` | 3 | 9 |
| `event_rsvp` | 3 | 6 |
| `exam_registration` | 4 | 8 |
| `experiment_booking` | 3 | 9 |
| `field_trip` | 4 | 8 |
| `hackathon_signup` | 4 | 9 |
| `housing_preference` | 2 | 7 |
| `internship_app` | 4 | 12 |
| `job_fair` | 3 | 9 |

Trace-level pattern:

- `event_rsvp`: last steps repeatedly click the same bottom-of-screen coordinate while verification remains stuck at `3/6`.
- `accessibility_feedback`: last steps repeatedly click the same coordinate while several option/date controls remain unresolved.
- Several forms show action distributions like `124-126` clicks and only `2-4` text-entry actions.

Interpretation:

- OpenCUA is no longer blocked by model readiness, served-model-name, or endpoint mismatch.
- The current top-to-bottom prompt did not fix native control behavior.
- Larger OpenCUA batches should pause until repeated-click loop handling or an alternative interaction strategy is implemented.

## Known Issues

### Metrics

- Qwen final `verified_correctness` is misleading after successful submission because the form controls are gone.
- `scored_correctness` is the right field for Qwen submitted trials.
- Qwen `submitted_while_incomplete_count` likely overcounts because it currently looks at a post-submit snapshot.
- OpenCUA scoring is final-page scoring because it usually does not submit; its `verified_correctness` is currently meaningful for partial completion.

### Qwen Behavior

- Text Qwen prematurely emits `DONE` even when no submission confirmation has been observed.
- VLM Qwen has better submit rate but still has long/timeout trials.
- Time controls and custom Google Forms controls remain fragile.

### OpenCUA Behavior

- OpenCUA loops on repeated clicks after partial field completion.
- The prompt has generic anti-loop guidance, and the runner now has a minimal hard loop breaker for repeated identical clicks/waits.
- Without symbolic support, the screenshot-only contract may be too weak for reliable Google Forms option/date/time controls.

### Code Quality

- The OpenCUA model-inference exception block formatting has been cleaned up.
- Several timestamps still use `datetime.utcnow()`, which emits deprecation warnings under Python 3.12. This is low risk but should be migrated to timezone-aware UTC timestamps.

## Recommended Next Steps

1. Monitor jobs `2240221` and `2240222`.
   - `2240221` is the Qwen direct-MCP 50-form run-2 batch for both Qwen models.
   - `2240222` is the OpenCUA native control-guidance 50-form run-2 batch.
   - After either job completes, rerun `scripts/analyze_eval_results.py`; the prefix-based grouping will pull the new summaries into the model/form/run tables.

2. Fix Qwen `submitted_while_incomplete_count`.
   - Base the incomplete-submit metric on `pre_submit_verified_correctness`, not post-submit verification.
   - Keep post-submit verification only as confirmation-page diagnostic data.

3. Inspect Qwen timeout trials.
   - Current examples: text `office_hours`, VLM `peer_evaluation`, VLM `project_update`, earlier VLM `field_trip`.
   - Determine whether the timeout is model latency, repeated tool use, or a specific form control pattern.

4. Improve Qwen terminal behavior only after inspecting `model_io.jsonl`.
   - The main text-model problem is early `DONE`, not field-filling inability.
   - Avoid overfitting a prompt change before checking whether the model is confusing "ready to submit" with "submitted".

5. Monitor the new OpenCUA control-guidance batch.
   - Job `2228827` uses the same screenshot-native condition and adds only general Google Forms control instructions.
   - Compare it against `2228808`: if loop rate drops or action count increases before loop termination, the issue was partly control affordance understanding rather than model serving.

6. Refresh the tracker after any new Qwen summaries.

```bash
.venv/bin/python scripts/update_eval_results_tracker.py --experiment-id qwen_direct_mcp_english_stepcap128_5form_20260515
```

## Recent Submitted Jobs

### `2240221`: Qwen Direct-MCP All-50 Run-2 Batch

- Status at submission check: running on `i8022`.
- Experiment ID: `qwen_direct_mcp_all50_run2_20260528`
- Models: `text_qwen3_30b_a3b_instruct_2507`, `vlm_qwen3_vl_30b_a3b_instruct`
- Run indexes: `2`
- Forms resolved by `FORM_IDS=all`, `FORM_OFFSET=0`, `FORM_LIMIT=50`: all 50 sorted forms.
- Settings: `DIRECT_MCP_MAX_STEPS=128`, `DIRECT_MCP_TIMEOUT_S=9000`, `DIRECT_MCP_TEXT_MAX_NEW_TOKENS=1024`, `DIRECT_MCP_VLM_MAX_NEW_TOKENS=1024`, `SKIP_COMPLETED=1`, `FAIL_ON_TRIAL_FAILURE=0`
- Node constraint: submitted with `--exclude=i8033`.
- Purpose: add one full reference-run-2 pass for both Qwen models without colliding with earlier experiment directories.

### `2240222`: OpenCUA Control-Guidance All-50 Run-2 Batch

- Status at latest check: running on `i8024`.
- Experiment ID: `opencua_control_guidance_all50_run2_20260528`
- Model: `computer_use_opencua_32b`
- Run indexes: `2`
- Forms resolved by `FORM_IDS=all`, `FORM_OFFSET=0`, `FORM_LIMIT=50`: all 50 sorted forms.
- Settings: `DIRECT_MAX_STEPS=128`, `DIRECT_TIMEOUT_S=3600`, `DIRECT_MAX_NEW_TOKENS=96`, `DIRECT_API_TIMEOUT_S=180`, `OPENCUA_GPU_MEMORY_UTILIZATION=0.85`, `OPENCUA_MAX_MODEL_LEN=16384`, `FAIL_ON_TRIAL_FAILURE=0`
- Node constraint: submitted with `--exclude=i8033`.
- Purpose: add a comparable OpenCUA native control-guidance run-2 pass across all 50 forms.

### `2228977`: Qwen Direct-MCP Final-5 Batch

- Status: completed.
- Experiment ID: `qwen_direct_mcp_english_stepcap128_5form_20260515`
- Models: `text_qwen3_30b_a3b_instruct_2507`, `vlm_qwen3_vl_30b_a3b_instruct`
- Run indexes: `1`
- Forms resolved by `FORM_IDS=all`, `FORM_OFFSET=45`, `FORM_LIMIT=5`: `travel_reimbursement`, `usability_test`, `volunteer_shift`, `wellbeing_check`, `workshop_signup`
- Settings: `DIRECT_MCP_MAX_STEPS=128`, `DIRECT_MCP_TIMEOUT_S=9000`, `DIRECT_MCP_TEXT_MAX_NEW_TOKENS=1024`, `DIRECT_MCP_VLM_MAX_NEW_TOKENS=1024`, `SKIP_COMPLETED=1`, `FAIL_ON_TRIAL_FAILURE=0`
- Node constraint: submitted with `--exclude=i8033`.
- Result: `8` new summaries. `workshop_signup` run `1` was skipped for both models because summaries already existed. Submit success `7/8`; scored correctness `36/64`.
- Purpose: finish Qwen run-1 coverage for sorted forms 46-50.

### `2229250`: OpenCUA Control-Guidance Remaining-20 Batch

- Status: completed.
- Experiment ID: `opencua_control_guidance_remaining20_20260527`
- Model: `computer_use_opencua_32b`
- Run indexes: `1`
- Forms resolved by `FORM_IDS=all`, `FORM_OFFSET=30`, `FORM_LIMIT=20`: `peer_evaluation`, `project_update`, `publication_submission`, `purchase_request`, `remote_setup`, `research_interest`, `room_booking`, `scholarship_interest`, `seminar_proposal`, `software_access`, `sports_tournament`, `study_group_match`, `survey_consent`, `technical_support`, `thesis_meeting`, `travel_reimbursement`, `usability_test`, `volunteer_shift`, `wellbeing_check`, `workshop_signup`
- Settings: `DIRECT_MAX_STEPS=128`, `DIRECT_TIMEOUT_S=3600`, `DIRECT_MAX_NEW_TOKENS=96`, `DIRECT_API_TIMEOUT_S=180`, `OPENCUA_GPU_MEMORY_UTILIZATION=0.85`, `OPENCUA_MAX_MODEL_LEN=16384`, `FAIL_ON_TRIAL_FAILURE=0`
- Node constraint: submitted with `--exclude=i8033`.
- Result: `20` summaries, `0/20` submit success, `58/160` scored correctness, `17` repeated-action loops, `3` max-step failures.
- Purpose: extend OpenCUA control-guidance coverage from the first 30 sorted forms to the remaining 20 sorted forms.

### `2228807`: OpenCUA Loop-Detector Smoke Retry

- Status: completed.
- Experiment ID: `opencua_loopdetector_smoke_retry_20260526`
- Model: `computer_use_opencua_32b`
- Form/run: `event_rsvp`, run `1`
- Settings: `DIRECT_MAX_STEPS=128`, `DIRECT_TIMEOUT_S=3600`, `DIRECT_MAX_NEW_TOKENS=96`, `DIRECT_API_TIMEOUT_S=180`, `OPENCUA_GPU_MEMORY_UTILIZATION=0.85`, `OPENCUA_MAX_MODEL_LEN=16384`, `FAIL_ON_TRIAL_FAILURE=0`
- Node constraint: submitted with `--exclude=i8033`.
- Result: `repeated_action_loop`, `10` actions, `3/6` verified correctness, no submit.
- Purpose: validated that the loop detector produces an early `repeated_action_loop` classification instead of wasting all 128 steps on the old repeated click behavior.

### `2228827`: OpenCUA Control-Guidance 30-Form Batch

- Status: completed.
- Experiment ID: `opencua_control_guidance_30form_20260526`
- Model: `computer_use_opencua_32b`
- Run indexes: `1`
- Forms: `accessibility_feedback`, `alumni_checkin`, `bug_report`, `club_application`, `club_event_planning`, `conf_interest`, `conference_travel`, `course_enrollment`, `course_feedback`, `data_annotation`, `dataset_request`, `equipment_checkout`, `event_rsvp`, `exam_registration`, `experiment_booking`, `field_trip`, `hackathon_signup`, `housing_preference`, `internship_app`, `job_fair`, `lab_safety`, `lab_visit`, `language_exchange`, `library_membership`, `meal_plan`, `mentor_match`, `newsletter_signup`, `office_hours`, `orientation_signup`, `paper_review`
- Selection: `FORM_IDS=all`, `FORM_OFFSET=0`, `FORM_LIMIT=30`.
- Settings: `DIRECT_MAX_STEPS=128`, `DIRECT_TIMEOUT_S=3600`, `DIRECT_MAX_NEW_TOKENS=96`, `DIRECT_API_TIMEOUT_S=180`, `OPENCUA_GPU_MEMORY_UTILIZATION=0.85`, `OPENCUA_MAX_MODEL_LEN=16384`, `FAIL_ON_TRIAL_FAILURE=0`
- Node constraint: submitted with `--exclude=i8033`.
- Result: `30` summaries, `0` submit successes, `99/249` scored correctness, `26` repeated-action loops, `4` max-step failures.
- Interpretation: general control guidance did not make OpenCUA reliable on native Google Forms controls. It remains useful as a failure taxonomy baseline.

### `2228809`: Qwen Direct-MCP Next-10 Batch Retry

- Status: completed.
- Experiment ID: `qwen_direct_mcp_english_stepcap128_5form_20260515`
- Models: `text_qwen3_30b_a3b_instruct_2507`, `vlm_qwen3_vl_30b_a3b_instruct`
- Run indexes: `1`
- Forms resolved by `FORM_IDS=all`, `FORM_OFFSET=35`, `FORM_LIMIT=10`: `research_interest`, `room_booking`, `scholarship_interest`, `seminar_proposal`, `software_access`, `sports_tournament`, `study_group_match`, `survey_consent`, `technical_support`, `thesis_meeting`
- Settings: `DIRECT_MCP_MAX_STEPS=128`, `DIRECT_MCP_TIMEOUT_S=9000`, `DIRECT_MCP_TEXT_MAX_NEW_TOKENS=1024`, `DIRECT_MCP_VLM_MAX_NEW_TOKENS=1024`, `SKIP_COMPLETED=1`, `FAIL_ON_TRIAL_FAILURE=0`
- Node constraint: submitted with `--exclude=i8033`.
- Result: `20` summaries, `10` submit successes, `94/158` scored correctness.
- Text Qwen latest batch: `3/10` submit success, `43/79` scored correctness.
- VLM Qwen latest batch: `7/10` submit success, `51/79` scored correctness.
- Note: one text-model row is an environment/browser launch timeout, not a model reasoning failure.

### Superseded Failed Jobs

- `2220804`: OpenCUA smoke failed before trial output; only the initial vLLM launch line was written, and no summaries were produced.
- `2220805`: Qwen batch failed before trials during text vLLM startup with `cudaGetDeviceCount Error 802`; it also exposed the fragile comma-separated `FORM_IDS` export path.
- `2220806`: OpenCUA 30-form batch became stale with `DependencyNeverSatisfied` and was cancelled after the retry jobs were submitted.
- `2228808`: OpenCUA loop-detector 30-form retry was cancelled after the first forms all reproduced `repeated_action_loop` under the older prompt. Early results: `accessibility_feedback` `2/8` in `8` actions, `alumni_checkin` `4/7` in `7` actions, `bug_report` `4/9` in `8` actions, `club_application` `4/10` in `8` actions.

## Recent Job Ledger

### `2218695`: Qwen Direct-MCP Batch

- Experiment ID: `qwen_direct_mcp_english_stepcap128_5form_20260515`
- Models: `text_qwen3_30b_a3b_instruct_2507`, `vlm_qwen3_vl_30b_a3b_instruct`
- Forms: `meal_plan`, `mentor_match`, `newsletter_signup`, `office_hours`, `orientation_signup`, `paper_review`, `peer_evaluation`, `project_update`, `publication_submission`, `purchase_request`
- Output root: `data/model_baselines/qwen_direct_mcp_english_stepcap128_5form_20260515/`
- Result: completed and added 20 summaries.

### `2218699`: OpenCUA 20-Form Native Batch

- Experiment ID: `opencua_topdown_prompt_20form_20260519`
- Model: `computer_use_opencua_32b`
- Forms: `accessibility_feedback`, `alumni_checkin`, `bug_report`, `club_application`, `club_event_planning`, `conf_interest`, `conference_travel`, `course_enrollment`, `course_feedback`, `data_annotation`, `dataset_request`, `equipment_checkout`, `event_rsvp`, `exam_registration`, `experiment_booking`, `field_trip`, `hackathon_signup`, `housing_preference`, `internship_app`, `job_fair`
- Output root: `data/model_baselines/opencua_topdown_prompt_20form_20260519/`
- Result: 20 summaries, all `max_steps_exceeded`, no submissions.

### `2216288`: Qwen Direct-MCP Batch

- Experiment ID: `qwen_direct_mcp_english_stepcap128_5form_20260515`
- Forms: `exam_registration`, `experiment_booking`, `field_trip`, `hackathon_signup`, `housing_preference`, `job_fair`, `lab_safety`, `lab_visit`, `language_exchange`, `library_membership`
- Result: completed with 20 summaries.
- Text Qwen: 4/10 submit success, `53/82` scored.
- VLM Qwen: 9/10 submit success, `50/82` scored.
- Main failures: text premature `DONE`; VLM timeout on `field_trip`.

### `2216283` and `2216287`: OpenCUA One-Form Smokes

- Form/run: `event_rsvp`, run `1`.
- Result: both served correctly but failed behaviorally.
- Both stopped at `max_steps_exceeded`, correctness `3/6`, action count `128`.

### `2212101`: Fresh Five-Form English Step-Cap Pilot

- Qwen produced 10 summaries.
- Text Qwen submit-success rate: `3/5`.
- VLM Qwen submit-success rate: `4/5`.
- The run exposed the post-submit verification problem that motivated pre-submit scoring snapshots.
- OpenCUA did not produce five-form summaries because readiness failed against the local endpoint before trials started.

### `2212085`: OpenCUA Two-Form Native Follow-Up

- Result: server reachable but model inference failed with `opencua_http_error:404`.
- Root issue: requested model name did not match the served model ID.
- Later readiness fixes addressed this class of failure.

### `2212084`: Qwen Direct-MCP Two-Form Test

- Result: useful debugging run, superseded by later English/default and pre-submit scoring fixes.
- Exposed premature `DONE`, custom control fragility, and German confirmation detection.

## Useful Commands

Check relevant job accounting:

```bash
sacct -j 2212084,2212085,2212101,2216283,2216287,2216288,2218695,2218699,2220804,2220805,2220806,2228807,2228808,2228809,2228827,2228977,2229250 --format=JobID,JobName%30,Partition,State,ExitCode,Elapsed,MaxRSS,NodeList -P
```

List recent SLURM logs:

```bash
ls -lt logs/slurm | head -40
```

Summarize current Qwen tracker:

```bash
.venv/bin/python scripts/update_eval_results_tracker.py --experiment-id qwen_direct_mcp_english_stepcap128_5form_20260515
```

Run focused tests:

```bash
.venv/bin/python -m unittest tests.test_browser_language_defaults tests.test_qwen_direct_mcp_eval tests.test_opencua_direct_eval
```

Syntax checks:

```bash
bash -n scripts/run_opencua_direct_matrix.sh scripts/run_qwen_direct_mcp_matrix.sh scripts/slurm_opencua_direct.sbatch scripts/slurm_qwen_direct_mcp.sbatch
.venv/bin/python -m py_compile src/baselines/run_qwen_direct_mcp_eval.py src/baselines/run_opencua_direct_eval.py scripts/update_eval_results_tracker.py
```
