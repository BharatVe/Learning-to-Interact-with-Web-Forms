# Evaluation Implementation Tracking Log

Last updated: 2026-05-15T13:34:23Z

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

## Active Jobs

### Fresh Five-Form English Step-Cap Pilot

Job: `2212101`

SLURM status at latest check: `PENDING`, reason `Priority`.

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

### OpenCUA Two-Form Native Follow-Up

Job: `2212085`

SLURM status at latest check: `PENDING`, reason `Priority`.

Experiment: `opencua_native_doc_aligned_2form_20260515`

Forms/runs:

- `event_rsvp`, run `1`
- `conf_interest`, run `1`

Submitted setting of interest:

- `DIRECT_MAX_STEPS=80`

Purpose:

- Pre-English-fresh-pilot OpenCUA follow-up after prompt fixes.
- Still useful for checking whether OpenCUA improves with a moderate step increase, but `2212101` is now the preferred fresh English pilot.

## Recent Completed Jobs

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
- Qwen may still ignore tool visibility or emit unsupported tool calls.
- Google Forms time controls remain fragile because they are often input-backed controls with combobox-like roles.
- OpenCUA may still need more than `128` steps or a better visual cadence.
- Separate SLURM jobs still reload models separately; reuse is only within each submitted job.

## Next Steps

1. Monitor `2212101` first; it is the current fresh pilot with English browser/form defaults and the larger step cap.
2. When `2212101` starts producing artifacts, check traces for English pages and English submission confirmations.
3. If Qwen still stops with `premature_done_without_submit`, inspect `model_io.jsonl` before changing prompts again.
4. If OpenCUA still reaches `max_steps_exceeded` at `128`, inspect the trace/video for loop type before raising the cap again.
5. Use `2212085` as a secondary OpenCUA comparison only if it starts before or near `2212101`.
6. After the 5-form pilot is stable, submit the 20-form baseline in resumable chunks.

## Useful Commands

Check active job state:

```bash
squeue -j 2212085,2212101 -o "%.18i %.9P %.30j %.8T %.10M %.10l %.6D %R"
```

Check accounting:

```bash
sacct -j 2212084,2212085,2212101 --format=JobID,JobName%30,Partition,State,ExitCode,Elapsed,MaxRSS,NodeList -P
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
