# Evaluation Implementation Tracking Log

Last updated: 2026-05-18T18:20:23Z

Use this file as the current source of truth before rerunning evaluation jobs. Older debugging detail has been collapsed into short historical notes so the active state is easier to read.

## Current Decision

- Qwen text/VLM thesis track: `direct_mcp_tool_use`.
- Qwen models interact through documented Playwright MCP tools, using `browser_snapshot` refs as the primary UI substrate.
- OpenCUA thesis track: `computer_use_native`.
- OpenCUA remains screenshot-native with pyautogui/action-output style and Qwen2.5-VL coordinate conversion.
- Do not aggregate Qwen direct-MCP and OpenCUA native as one interface condition; report them as different web-form completion systems.
- Scripted Playwright generation runs remain the efficiency reference baseline for matching `form_id + run_XXXX`.

## Current Implementation

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

- Job `2212084` loaded Google Forms confirmation pages in German. This caused a real submitted page to be treated as a failed submit path because the English-only confirmation detector did not match the page text.

### Qwen Direct-MCP Runner

File: `src/baselines/run_qwen_direct_mcp_eval.py`

Current behavior:

- Uses Playwright MCP directly instead of the legacy mediated `human_ui_v1` path.
- Prompts the model to use snapshot refs and documented browser tools.
- Exposes `browser_check` for radio/checkbox controls.
- Hides `browser_select_option` unless a real HTML `select` is visible.
- Tracks submit attempts:
  - `submit_attempt_count`
  - `successful_submit_attempt_count`
  - `failed_submit_attempt_count`
  - `submitted_while_incomplete_count`
  - `first_submit_step`
- Terminal contract: `DONE` means an observed submission confirmation page, not “ready to submit”.
- Future runs now record pre-submit correctness for any `browser_click` that actually lands on a submission confirmation page, even if the clicked ref was not recognized as a submit control before the click.

Reason for current `container_not_visible` results:

- In job `2212101`, successful Qwen submissions reached the Google Forms confirmation page.
- Final verification then searched for original question containers on the confirmation page, where they no longer exist.
- This produced `container_not_visible` and `verified_correctness=0` even when model actions likely filled the form before submit.
- Current fix improves future summaries by preserving `pre_successful_submit_verified_correctness`.

### Consolidated Results Tracker

Files:

- `scripts/update_eval_results_tracker.py`
- `docs/eval_results/README.md`
- `docs/eval_results/metrics.csv`
- `docs/eval_results/metrics.jsonl`

Current behavior:

- Scans `data/model_baselines/**/summary.json`.
- Writes one consolidated CSV, JSONL, and Markdown summary.
- Upserts naturally by rebuilding from all available summaries, so rerunning after a new job adds missing rows.
- Supports `--experiment-id` filters so the tracker can stay focused on the current clean evaluation iteration.
- Current tracker is filtered to `qwen_direct_mcp_english_stepcap128_5form_20260515`.
- Current tracker has `10` trial rows.
- Adds `scored_correctness`, `scored_correctness_source`, and `metric_warning` columns.
- Current submitted rows have `metric_warning=submitted_without_pre_submit_snapshot...` because these artifacts were produced before the pre-submit snapshot fix.
- `scripts/run_qwen_direct_mcp_matrix.sh` now refreshes this tracker at the end of every Qwen batch for the active experiment.

Command:

```bash
.venv/bin/python scripts/update_eval_results_tracker.py --experiment-id qwen_direct_mcp_english_stepcap128_5form_20260515
```

Cleanup performed:

- Removed obsolete historical experiment outputs from `data/model_baselines`.
- Kept only `data/model_baselines/qwen_direct_mcp_english_stepcap128_5form_20260515`.
- Removed stale SLURM/log files.
- Kept only current Qwen `2212101` logs and `logs/qwen_direct_mcp_english_stepcap128_5form_20260515_reference_efficiency_summary.json`.

### OpenCUA Native Runner

File: `src/baselines/run_opencua_direct_eval.py`

Current behavior:

- Defaults to screenshot-native prompt input.
- Excludes symbolic interaction maps by default.
- Keeps pyautogui-style outputs and Qwen2.5 coordinate conversion.
- Supports `--include-symbolic-support` only for ablation/debug.
- Records premature/incomplete submits as metrics instead of blocking them in code.
- Prompts the model to double-check visible form state before submitting.

### OpenCUA Compatibility Verifier

File: `scripts/verify_opencua_compatibility.py`

Current behavior:

- Checks config compatibility, forms, answer runs, parser behavior, smart-resize coordinate conversion, and prompt contract.
- Warns when local runtime dependencies/server are missing unless `--require-endpoint` is used.
- Latest focused verifier run passed the code/config checks; local shell warnings were expected because vLLM and Playwright MCP are provided inside SLURM/runtime environments.

### OpenCUA Serving Readiness Fix

Files:

- `scripts/check_openai_compat_server.py`
- `scripts/verify_opencua_compatibility.py`
- `scripts/run_track_baseline_matrix.sh`
- `scripts/run_opencua_direct_matrix.sh`
- `scripts/slurm_opencua_direct.sbatch`

Current behavior:

- OpenCUA SLURM jobs use a job-specific default vLLM port instead of fixed port `8000`.
- vLLM is launched with `OPENCUA_SERVED_MODEL_NAME=$OPENAI_MODEL`.
- Readiness no longer accepts any process answering `/v1/models`; it requires the expected model ID to be advertised.
- Readiness logs the advertised `/v1/models` IDs.
- A small OpenAI-compatible chat smoke is run against the exact expected model before browser evaluation starts.

Reason:

- Job `2212101` appeared ready at `elapsed_s=0`, which is impossible for a fresh OpenCUA load and likely came from a stale/wrong process on port `8000`.
- Job `2212085` reached vLLM but the evaluator got `404` for model `opencua-32b`; model-specific readiness should catch that before wasting a form run.

## Verification

Focused tests:

```bash
.venv/bin/python -m unittest tests.test_browser_language_defaults tests.test_qwen_direct_mcp_eval tests.test_opencua_direct_eval
```

Latest result: `31 tests OK`.

Syntax check:

```bash
.venv/bin/python -m py_compile src/engine/browser_language.py src/engine/runner.py src/engine/mcp_browser_engine.py src/engine/form_engine.py src/baselines/run_baseline_eval.py src/baselines/run_direct_api_eval.py src/baselines/run_gemini_native_computer_use_eval.py src/baselines/run_qwen_direct_mcp_eval.py src/baselines/run_opencua_direct_eval.py scripts/sync_generator_dataset.py
```

Latest result: passed.

OpenCUA serving patch checks:

```bash
bash -n scripts/run_track_baseline_matrix.sh scripts/run_opencua_direct_matrix.sh scripts/slurm_opencua_direct.sbatch scripts/run_opencua_vllm_server.sh
.venv/bin/python -m py_compile scripts/check_openai_compat_server.py scripts/verify_opencua_compatibility.py scripts/update_eval_results_tracker.py src/baselines/run_qwen_direct_mcp_eval.py
```

Latest result: passed.

## Active Jobs

### Qwen Direct-MCP Next-10-Forms Batch

Job: `2215472`

SLURM status at latest check: `PENDING`, reason `Priority`.

Experiment: `qwen_direct_mcp_english_stepcap128_5form_20260515`

Forms/runs:

- `accessibility_feedback`, run `1`
- `alumni_checkin`, run `1`
- `bug_report`, run `1`
- `club_application`, run `1`
- `club_event_planning`, run `1`
- `conference_travel`, run `1`
- `course_enrollment`, run `1`
- `data_annotation`, run `1`
- `dataset_request`, run `1`
- `equipment_checkout`, run `1`

Models:

- `text_qwen3_30b_a3b_instruct_2507`
- `vlm_qwen3_vl_30b_a3b_instruct`

Submitted settings:

- `DIRECT_MCP_MAX_STEPS=128`
- `DIRECT_MCP_TIMEOUT_S=9000`
- `DIRECT_MCP_TEXT_MAX_NEW_TOKENS=1024`
- `DIRECT_MCP_VLM_MAX_NEW_TOKENS=1024`
- `SKIP_COMPLETED=1`
- `FAIL_ON_TRIAL_FAILURE=0`

Purpose:

- Continue the clean consolidated Qwen direct-MCP dataset in the same experiment/output folder.
- Adds 20 expected trial rows when complete: 10 forms x 2 Qwen models.
- The Qwen matrix will refresh `docs/eval_results` automatically at job end.

### Qwen Direct-MCP First-Five Remaining Answer Sets

Job: `2215473`

SLURM status at latest check: `PENDING`, reason `Priority`.

Experiment: `qwen_direct_mcp_english_stepcap128_5form_20260515`

Forms:

- `conf_interest`
- `event_rsvp`
- `course_feedback`
- `internship_app`
- `workshop_signup`

Run indexes:

- `2,3,4,5,6,7,8,9,10`

Models:

- `text_qwen3_30b_a3b_instruct_2507`
- `vlm_qwen3_vl_30b_a3b_instruct`

Submitted settings:

- `DIRECT_MCP_MAX_STEPS=128`
- `DIRECT_MCP_TIMEOUT_S=9000`
- `DIRECT_MCP_TEXT_MAX_NEW_TOKENS=1024`
- `DIRECT_MCP_VLM_MAX_NEW_TOKENS=1024`
- `SKIP_COMPLETED=1`
- `FAIL_ON_TRIAL_FAILURE=0`

Purpose:

- Complete the 10 answer-set coverage for the original five pilot forms.
- Adds 90 expected trial rows when complete: 5 forms x 9 remaining runs x 2 Qwen models.
- Outputs stay in the same consolidated Qwen experiment folder.
- The Qwen matrix will refresh `docs/eval_results` automatically at job end.

### OpenCUA Serving-Fix One-Form Smoke

Job: `2215471`

SLURM status at latest check: `PENDING`, reason `Resources`.

Experiment: `opencua_native_serving_fix_smoke_20260517`

Forms/runs:

- `event_rsvp`, run `1`

Submitted settings:

- `DIRECT_MAX_STEPS=128`
- `DIRECT_TIMEOUT_S=3600`
- `DIRECT_API_TIMEOUT_S=240`
- `DIRECT_MAX_NEW_TOKENS=384`
- `FAIL_ON_TRIAL_FAILURE=0`

Purpose:

- Validate the job-specific port, exact served-model readiness check, and exact-model smoke chat before rerunning larger OpenCUA jobs.

## Recent Completed Jobs

### Fresh Five-Form English Step-Cap Pilot

Job: `2212101`

SLURM result: `FAILED`, exit `1:0`, elapsed `00:26:34`.

Experiments:

- Qwen direct-MCP: `qwen_direct_mcp_english_stepcap128_5form_20260515`
- OpenCUA native: `opencua_native_english_stepcap128_5form_20260515`

Forms/runs:

- `conf_interest`, run `1`
- `event_rsvp`, run `1`
- `course_feedback`, run `1`
- `internship_app`, run `1`
- `workshop_signup`, run `1`

Submitted settings:

- `DIRECT_MAX_STEPS=128`
- `DIRECT_TIMEOUT_S=9000`
- `DIRECT_MCP_TEXT_MAX_NEW_TOKENS=1024`
- `DIRECT_MCP_VLM_MAX_NEW_TOKENS=1024`
- `DIRECT_MAX_NEW_TOKENS=384`
- `FAIL_ON_TRIAL_FAILURE=0`
- `TRACK_REPORT_OUTPUT=logs/english_stepcap128_5form_20260515_track_summary.json`

Purpose:

- First real end-to-end validation of the English browser/form default.
- Give Qwen text, Qwen VLM, and OpenCUA a larger interaction budget.
- Confirm model servers are loaded once per job and reused across the five forms.

Observed Qwen direct-MCP results:

- 10 Qwen summaries were produced: 5 text-model trials and 5 VLM trials.
- Text Qwen submit-success rate: `3/5`.
- VLM Qwen submit-success rate: `4/5`.
- Remaining Qwen failures were `premature_done_without_submit`.
- All submitted trials currently show `verified_correctness=0`, which needs artifact/trace review before treating as final correctness.
- Reference-efficiency summary: `logs/qwen_direct_mcp_english_stepcap128_5form_20260515_reference_efficiency_summary.json`.

Observed OpenCUA native result:

- No OpenCUA five-form summaries were produced for `opencua_native_english_stepcap128_5form_20260515`.
- The combined job failed before OpenCUA trials because compatibility verification failed against the local OpenCUA endpoint:
  - `OpenCUA endpoint reachable at 127.0.0.1:8000 but /models failed: timed out`
- The OpenCUA vLLM track log only recorded launch, so this looks like a server readiness/endpoint collision or startup-health-check issue rather than a completed model evaluation.

### OpenCUA Two-Form Native Follow-Up

Job: `2212085`

SLURM result: `TIMEOUT`, elapsed `1-00:00:07`.

Experiment: `opencua_native_doc_aligned_2form_20260515`

Forms/runs:

- `event_rsvp`, run `1`
- `conf_interest`, run `1`

Submitted setting of interest:

- `DIRECT_MAX_STEPS=80`

Purpose:

- Pre-English-fresh-pilot OpenCUA follow-up after prompt fixes.
- Still useful for checking whether OpenCUA improves with a moderate step increase, but `2212101` is now the preferred fresh English pilot.

Observed result:

- OpenCUA vLLM started and `/v1/models` passed preflight.
- Two OpenCUA summaries were produced.
- Both trials failed before any actions with `model_inference_failed`.
- Failure detail:
  - `opencua_http_error:404`
  - `The model opencua-32b does not exist.`
- Likely issue: the requested served model name did not match the actual model ID advertised by the vLLM server, despite the launch script passing `--served-model-name opencua-32b`. Inspect `/v1/models` output or pin/pass the exact advertised model string before rerunning.

### Qwen Direct-MCP Two-Form Test

Job: `2212084`

SLURM result: `COMPLETED`, exit `0:0`, elapsed `00:18:13`.

Experiment: `qwen_direct_mcp_ref_first_2form_20260515`

Result summary:

- Text Qwen, `event_rsvp`: `0/6`, `submit_success=false`, `premature_done_without_submit`.
- Text Qwen, `conf_interest`: `0/7`, `submit_success=false`, `premature_done_without_submit`.
- VLM Qwen, `event_rsvp`: `3/6`, `submit_success=false`, `max_steps_exceeded`.
- VLM Qwen, `conf_interest`: `0/7`, `submit_success=false`, `premature_done_without_submit`.

Interpretation:

- This run was useful for exposing remaining terminal/submit behavior issues.
- It was not the final English-fix validation; `2212101` supersedes it for the current evaluation direction.

### Earlier Smoke Runs

Jobs:

- `2212064`: Qwen direct-MCP smoke, completed.
- `2212065`: OpenCUA native smoke, completed.

Lessons retained:

- Qwen can fill some fields through direct MCP, but may stop early with `DONE`.
- `browser_select_option` must not be shown globally because Google Forms time inputs can look like comboboxes while not being real HTML `select` elements.
- OpenCUA can perform some correct screenshot-native interactions, but the original cap was too low and it often reached `max_steps_exceeded`.

## Current Known Risks

- Qwen may still say `DONE` before actually submitting.
- Submitted Qwen trials can still verify as `0/N`; inspect traces to determine whether this is scoring-after-submit visibility, wrong-filled fields, or a submit/verification timing issue.
- Qwen may still ignore tool visibility or emit unsupported tool calls.
- Google Forms time controls remain fragile because they are often input-backed controls with combobox-like roles.
- OpenCUA currently has a served-model-name/readiness problem before meaningful step-cap conclusions can be drawn.
- Separate SLURM jobs still reload models separately; reuse is only within each submitted job.

## Next Steps

1. Inspect Qwen `2212101` artifacts where `submit_success=true` but `verified_correctness=0`; determine whether verification is running after the form is already on the confirmation page.
2. Inspect `model_io.jsonl` for the Qwen `premature_done_without_submit` failures before changing prompts again.
3. Monitor Qwen batch jobs `2215472`, `2215473`, and `2216288`.
4. Monitor OpenCUA serving/prompt smoke jobs `2215471`, `2216283`, and `2216287`.
5. If `2215471` passes readiness but fails during model inference, inspect the exact smoke-chat/eval error before changing prompts or step caps.
6. If `2215471` produces real OpenCUA actions, rerun the five-form English pilot.
7. After Qwen job `2216288` completes, inspect `docs/eval_results/metrics.csv` and submit the next remaining Qwen form batch into the same experiment.

## 2026-05-18 Qwen Batch Submission

Submitted a new Qwen direct-MCP batch for both Qwen models using the consolidated experiment/results setup.

- Job: `2216288`
- Submitted: `2026-05-18T18:20:23Z`
- Experiment ID: `qwen_direct_mcp_english_stepcap128_5form_20260515`
- Models: `text_qwen3_30b_a3b_instruct_2507`, `vlm_qwen3_vl_30b_a3b_instruct`
- Track: `direct_mcp_tool_use`
- Step cap: `128`
- Run indexes: `1`
- Forms: `exam_registration`, `experiment_booking`, `field_trip`, `hackathon_signup`, `housing_preference`, `job_fair`, `lab_safety`, `lab_visit`, `language_exchange`, `library_membership`
- Output root: `data/model_baselines/qwen_direct_mcp_english_stepcap128_5form_20260515/`
- Consolidated tracker: `docs/eval_results/metrics.csv` and `docs/eval_results/metrics.jsonl`

Before this submission, Qwen summaries covered 15 of 50 forms. This batch targets the next 10 uncovered forms, bringing expected coverage to 25 forms after completion.

## Useful Commands

Check active job state:

```bash
squeue -j 2215471,2215472,2215473,2216283,2216287,2216288 -o "%.18i %.9P %.30j %.8T %.10M %.10l %.6D %R"
```

Check accounting:

```bash
sacct -j 2212101,2215471,2215472,2215473,2216283,2216287,2216288 --format=JobID,JobName%30,Partition,State,ExitCode,Elapsed,MaxRSS,NodeList -P
```

List recent SLURM logs:

```bash
ls -lt logs/slurm | head -40
```

Summarize Qwen two-form reference efficiency:

```bash
.venv/bin/python scripts/summarize_reference_efficiency.py --experiment-id qwen_direct_mcp_ref_first_2form_20260515 --output logs/qwen_direct_mcp_ref_first_2form_20260515_reference_efficiency_summary.json
```

Run focused tests:

```bash
.venv/bin/python -m unittest tests.test_browser_language_defaults tests.test_qwen_direct_mcp_eval tests.test_opencua_direct_eval
```
