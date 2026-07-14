# Evaluation Implementation Tracking Log

Last updated: 2026-07-09

Use this file as the current source of truth before rerunning evaluation jobs. It records the active evaluation design, implementation status, latest outputs, known issues, and next actions.

## Current Decision

- Active thesis comparison is now four interface/model conditions. The original full target remains `300` unique form-run trials per condition, but the current interim thesis target is capped at `200` unique form-run trials per condition because the remaining OpenCUA/Qwen batches are slow.
  - `text_qwen3_30b_a3b_instruct_2507` with direct Playwright MCP tools.
  - `vlm_qwen3_vl_30b_a3b_instruct` with direct Playwright MCP tools.
  - `computer_use_opencua_32b` with native screenshot/coordinate computer use.
  - `computer_use_opencua_32b_direct_mcp` with the same direct Playwright MCP tool contract used for Qwen.
- Target progress is counted by unique `(model_id, form_id, answer_run_id)` pairs under `data/model_baselines/**/summary.json`, not by raw duplicate summaries.
- With 50 forms and the interim `target_trials=200`, the active thesis-ready target is runs `1-4` for every condition. Runs `5-6` remain useful for the full target-300 extension but are no longer blocking the next thesis analysis pass.
- Do not aggregate Qwen direct-MCP, OpenCUA native, and OpenCUA direct-MCP as one interface condition; report them as separate conditions.
- Scripted Playwright generation runs remain the efficiency reference baseline for matching `form_id + run_XXXX`.

## Current State

The serving and basic orchestration problems are mostly solved. The remaining blockers are agent behavior and metric interpretation.

- Four-way target-chain scheduling is implemented in `scripts/submit_eval_target_chain.py`.
- Global unique-pair skip behavior is implemented for Qwen direct-MCP, OpenCUA native, and OpenCUA direct-MCP matrix runners.
- OpenCUA direct-MCP is now a first-class target using `scripts/slurm_opencua_direct_mcp.sbatch` and `MODEL_ID=computer_use_opencua_32b_direct_mcp`.
- OpenCUA direct-MCP smoke validation completed 5/5 trials with no hidden-tool errors and no invalid tool calls.
- A finite dependency chain for the remaining OpenCUA direct-MCP target-300 work was submitted on 2026-06-09 as jobs `2248104` through `2248109`.
- Qwen direct-MCP now produces usable submitted trials and pre-submit scoring snapshots.
- Qwen still frequently stops with `DONE` before submission, especially the text model.
- Qwen VLM can time out on some longer forms despite making partial progress.
- OpenCUA serving is working, but the native screenshot agent still fails to submit and usually loops on repeated clicks around Google Forms controls.
- Final post-submit verification can read as `0/N` on submitted direct-MCP trials because the browser is on the Google Forms confirmation page. For submitted direct-MCP trials, use `scored_correctness` from `pre_successful_submit_verified_correctness`.
- Latest reruns add CUDA preflight checks before vLLM startup, better OpenCUA vLLM launch diagnostics, and `FORM_OFFSET`/`FORM_LIMIT` support for `FORM_IDS=all`.
- Latest analysis artifacts live under `docs/eval_results/analysis/`.
- Gemini 3.5 Flash native Computer Use integration exists as a paid proprietary pilot path, but live Google Forms submission is blocked by Google safety/platform policy. Fill-only live Google Forms interaction is possible but expensive and not yet reliable enough for a larger run.

## 2026-06-11 Interim Target-200 Top-Up

Decision:

- Because full target-300 generation is taking too long, use an interim thesis analysis target of `200` unique form-runs per primary model condition.
- This corresponds to 50 forms x runs `1-4`.
- Do not submit more OpenCUA Native jobs for this interim target: the saved analysis already has `200/200` unique form-runs for `computer_use_opencua_32b`.

Submitted top-up jobs:

- `2248660`: OpenCUA direct-MCP run-4 top-up, experiment `opencua_direct_mcp_tools_target200_run4_20260611`, `FORM_IDS=all`, `FORM_OFFSET=0`, `FORM_LIMIT=50`, `RUN_INDEXES=4`, no dependency.
- `2248661`: Qwen direct-MCP run-1 top-up for missing `remote_setup`, experiment `qwen_direct_mcp_target300_run1_20260611_target200`, no dependency.
- `2248662`: Qwen direct-MCP run-4 top-up for remaining 15 run-4 forms, experiment `qwen_direct_mcp_target300_run4_20260611_target200`, dependency `afterok:2248661`.

Queue state immediately after submission:

- `2248661`: `PENDING (None)`.
- `2248662`: `PENDING (Dependency)`, `afterok:2248661`.
- `2248660`: `PENDING (Priority)`.
- Existing full target-300 chain remains queued separately:
  - `2248106`: OpenCUA direct-MCP run-3 running.
  - `2248107`: OpenCUA direct-MCP run-4 dependency-held after `2248106`.
  - `2248108`-`2248109`: OpenCUA direct-MCP runs `5-6`, no longer blocking interim target-200.
  - `2248111`-`2248114`: older Qwen backfill jobs, no longer the fastest path to interim target-200.
  - `2248115`-`2248116`: OpenCUA Native runs `5-6`, no longer needed for interim target-200.

Follow-up on 2026-06-12:

- Qwen top-up jobs `2248661` and `2248662` completed successfully.
- Interim target-200 coverage is reached for Qwen Text, Qwen VLM, and OpenCUA Native: each has `200/200` unique form-runs across runs `1-4`.
- OpenCUA direct-MCP is still short: `170/200`, with run `0004` at `20/50`.
- Duplicate OpenCUA direct-MCP run-4 work was found between:
  - `2248107`: original target-300 run-4 chain job.
  - `2248660`: extra target-200 run-4 top-up job.
- `2248660` was cancelled to reduce overlap and free resources. `2248107` remains running because it is the original chain job and keeps downstream dependencies coherent.
- Reference/ideal dataset generation job `2248565` completed successfully.
- `scripts/analyze_reference_dataset.py` reports `300/300` usable ideal references for 50 forms x runs `1-6`: no missing annotations, answers instances, traces, videos, invalid traces, failed runs, or submit failures.

Follow-up on 2026-06-13:

- OpenCUA direct-MCP reached the interim target-200 set and continued into the full target-300 chain.
- Current OpenCUA direct-MCP unique coverage is `263/300`: runs `1-5` are complete at `50/50`, and run `0006` is `13/50`.
- OpenCUA direct-MCP job `2248109` is still running on run `0006`; the full target-300 evaluation is not finished yet.
- OpenCUA Native remains `200/300`: runs `1-4` complete, runs `5-6` not run.
- `scripts/analyze_eval_results.py` was rerun and refreshed `docs/eval_results/analysis/` with `1006` total discovered trials.

## 2026-06-30 Gemini Proprietary Computer-Use Pilot

Status:

- Separate Gemini low-cost runner implemented in `src/baselines/run_gemini_low_cost_eval.py`, with wrapper `scripts/run_gemini_low_cost_matrix.sh`.
- API key is read from `GEMINI_API_KEY`, `GEMINI_API_KEY_FILE`, or `.secrets/gemini_api_key`; do not print or commit the key.
- Config entry `computer_use_gemini_35_flash_lowcost` uses provider `gemini_low_cost`, model `gemini-3.5-flash`, and the Gemini Interactions `computer_use` browser tool.
- Prompt mode is intentionally concise: screenshot plus exact remaining answers plus short browser-action guidance. Full interaction maps are omitted unless `INCLUDE_CONTROLS=1`.

Live Google Forms findings:

- Submit-enabled pilot on `bug_report/run_0002` verified API/browser/tool capability but was blocked by Google policy after partial progress. The model filled `2/9` fields over `5` actions before the provider returned an external Google Form automation block.
- Fill-only mode was added with `--fill-only` / `FILL_ONLY=1`. The prompt says to fill fields but never submit, and the harness refuses to execute a `submit` action in this mode.
- Fill-only pilot `gemini_35_flash_lowcost_fill_only_probe_v1`, trial `trial_20260630T113905803432Z`, ran one form only: `bug_report/run_0002`.
  - Result: no provider policy block, `24` actions, `7/9` fields verified correct, stopped at `max_steps_exceeded`.
  - Token metadata: `39,462` total tokens, lower-bound cost estimate `$0.059193` / `EUR 0.055049`.
  - Projected 30-run lower bound from this run: `$1.77579` / `EUR 1.65147`.
- Controls fallback `gemini_35_flash_lowcost_fill_only_controls_probe_v1`, trial `trial_20260630T114804827360Z`, was stopped manually after cost concern.
  - It was still the same single form, not a multi-form job.
  - Partial logs show `46` steps and `83,420` total tokens before interrupt, explaining the unexpectedly high spend.

Decision:

- Do not run larger paid Gemini jobs on live Google Forms yet.
- Treat live Google Forms submission as blocked by provider/platform safety policy; do not try to bypass it.
- Fill-only live Google Forms can be used as evidence that entry interaction may work, but it changes the task definition and must be labeled as fill-only/no-submit.
- Best thesis-defensible route for a larger proprietary pilot is a controlled local Google-Forms-style page: Gemini still uses screenshots and browser actions, but submissions are local and do not hit Google Forms platform automation controls.
- If any more paid live-Google-Forms probes are needed, cap at `8-12` max steps, keep `INCLUDE_CONTROLS=0` by default, and stop as soon as local verification reaches the target threshold.

Follow-up on 2026-07-09:

- Additional Gemini live-Google-Forms probes confirm that waiting until off-peak reduces provider-capacity failures but does not solve form-completion quality.
- Daytime one-form probes on `conference_travel/run_0002` repeatedly hit provider capacity or read-timeout failures:
  - `gemini_35_flash_fill_only_done_single_step32_20260708`: `provider_capacity_error`, `0` actions, `0/10`, `0` tokens.
  - `gemini_35_flash_fill_only_done_single_step32_retry_20260708`: `provider_capacity_error`, `0` actions, `0/10`, `0` tokens.
  - `gemini_35_flash_fill_only_done_single_step32_retry2_20260708`: `provider_capacity_error`, `0` actions, `0/10`, `0` tokens.
  - `gemini_35_flash_fill_only_done_step24_probe_20260702`: reached `7/10` after `21` actions and `35,845` tokens, then failed on a provider read timeout.
  - `gemini_35_flash_fill_only_done_controls_probe_20260702`: reached `4/10` after `9` actions and `18,309` tokens, then failed on provider high demand.
- Overnight probe `gemini_35_flash_fill_only_done_overnight_step32_20260708` avoided provider-capacity failure but still did not complete the first form:
  - Form/run: `conference_travel/run_0002`.
  - Result: `7/10` verified correct, `32` actions, `max_steps_exceeded`, `submit_success=false`.
  - Token metadata: `53,634` total tokens, lower-bound cost estimate `$0.080451` / `EUR 0.074819`.
  - Missed/incorrect fields: preferred travel mode remained the full option text rather than `Car`; preferred departure time was not filled; additional travel notes remained empty.
- Interpretation: Gemini can fill many simple fields on live Google Forms but remains unreliable on Google Forms-specific controls and long scrolling forms. The main blocker is now UI/task performance and cost, not only provider high demand.
- Recommendation: stop expanding live-Google-Forms Gemini runs for the thesis. Keep the Gemini evidence as a feasibility/negative pilot and, if a proprietary comparison is still needed, move it to a controlled local Google-Forms-style clone or omit the proprietary condition rather than spending more time and money on live Google Forms.

Open-source fill-only comparison status from the same follow-up:

- `opencua_direct_mcp_fill_only_done_10_seed20260702_r2_step32` produced `10` summaries with `1/10` full success; average verified correctness was about `5.1` fields per form. This condition is usable as a local open-source fill-only/DONE diagnostic, but it is not directly comparable to Gemini native Computer Use without labeling the interface difference.
- `qwen_direct_mcp_fill_only_done_10_seed20260702_r2_step32` produced no summaries because job `2269411` failed before trials with an environment/library issue: `.venv-opencua/bin/python` could not load `libpython3.12.so.1.0`. This is an execution-environment failure, not a model-result failure.

Comparator framing:

- Label the current condition as `Gemini 3.5 Flash native Computer Use, low-token prompt`.
- Compare native proprietary computer use against native OpenCUA separately from direct-MCP Qwen conditions; fairness means each model receives a competent interface for its intended operating mode, not identical prompt text.
- Use a shared 30-form sample only after the one-form local-page token/cost measurement is available.

30-form fill-only/DONE comparison launch on 2026-07-09:

- Added fixed 30-form sample manifest `configs/baselines/fill_only_done_30_seed20260709.json` and runner `scripts/run_fill_only_done_30form_eval.sh`.
- Task mode: `fill_only_done`; all models fill visible fields but must not submit. Success is local field-value verification, with `submit_success=false`.
- Seed/run: seed `20260709`, `run_0002`, step cap `32`.
- Forms: `alumni_checkin`, `bug_report`, `conference_travel`, `course_enrollment`, `course_feedback`, `data_annotation`, `dataset_request`, `event_rsvp`, `exam_registration`, `hackathon_signup`, `housing_preference`, `job_fair`, `language_exchange`, `library_membership`, `meal_plan`, `newsletter_signup`, `office_hours`, `orientation_signup`, `paper_review`, `project_update`, `publication_submission`, `room_booking`, `scholarship_interest`, `sports_tournament`, `study_group_match`, `survey_consent`, `technical_support`, `usability_test`, `volunteer_shift`, `workshop_signup`.
- Gemini condition: `gemini_35_flash_fill_only_done_30_seed20260709_r2_step32`, model `computer_use_gemini_35_flash_lowcost`, `FILL_ONLY=1`, `INCLUDE_CONTROLS=0`, `DIRECT_MAX_STEPS=32`. Submitted as Slurm job `2292319` with `--begin=23:00` to avoid daytime provider high-demand errors.
- Qwen direct-MCP condition: `qwen_direct_mcp_fill_only_done_30_seed20260709_r2_step32`, models `text_qwen3_30b_a3b_instruct_2507` and `vlm_qwen3_vl_30b_a3b_instruct`, `FILL_ONLY_DONE=1`, `DIRECT_MCP_MAX_STEPS=32`. Submitted as Slurm job `2292320`.
- OpenCUA direct-MCP condition: `opencua_direct_mcp_fill_only_done_30_seed20260709_r2_step32`, model `computer_use_opencua_32b_direct_mcp`, `FILL_ONLY_DONE=1`, `DIRECT_MCP_MAX_STEPS=32`. Submitted as Slurm job `2292321`.
- Native OpenCUA `computer_use_opencua_32b` remains excluded because the native path is not usable for this comparison.
- Implementation note: `scripts/run_qwen_direct_mcp_matrix.sh` now runs CUDA preflight and vLLM startup with the module `LD_LIBRARY_PATH` needed by `.venv-opencua`, while keeping the harness Python path clean. This addresses the previous `libpython3.12.so.1.0` pre-trial failure.
- Pre-submit checks passed: shell syntax for the edited wrappers, Gemini/direct-MCP unit tests (`35` tests), and validation that all `30` sampled forms have `run_0002` answer files.
- Initial scheduler state after submission: job `2292319` pending on `BeginTime`, job `2292320` pending on `Resources`, job `2292321` pending on `Priority`.
- User requested immediate Gemini execution after the delayed submission. Cancelled delayed Gemini job `2292319` and resubmitted the same Gemini condition without `--begin` as job `2292504` at `2026-07-09T20:52:31Z`. Initial state after resubmission: job `2292504` pending on `Resources`; Qwen job `2292320` running on `i8015`.

## 2026-07-13 50-Form Fill-Only/DONE Completion

- Added `configs/baselines/fill_only_done_50_completion_20260713.json` and `scripts/run_fill_only_done_50form_completion.sh` to extend the fixed run-2 comparison from 30 forms to the full 50-form corpus.
- Existing usable work is preserved: Gemini and OpenCUA keep their earlier 30-form artifacts. OpenCUA runs only the 20 newly added forms. Gemini runs those 20 plus retries the six earlier non-usable attempts (`alumni_checkin`, `bug_report`, `conference_travel`, `hackathon_signup`, `orientation_signup`, `technical_support`).
- Qwen must run all 50 forms for both the text and VLM conditions because job `2292320` produced zero trial summaries.
- Qwen root-cause update: the earlier `libpython3.12.so.1.0` problem was fixed, but job `2292320` hit a readiness race. The text model became healthy after about `30m36s`, just after the launcher's `30m` readiness limit. Its vLLM children then outlived the failed launcher until Slurm's `24h` wall-time limit.
- Qwen fix: raise the default vLLM readiness allowance from `30m` to `70m`, start vLLM in a separate process group, and terminate the whole group during cleanup.
- Comparison contract remains `fill_only_done`, `run_0002`, maximum `32` steps, and no form submission. Gemini remains in compact mode with `INCLUDE_CONTROLS=0`.
- Validation passed: JSON parsing, shell syntax, all 50 forms have `run_0002` answers, and `35` focused Gemini/Qwen unit tests.
- Planned scheduler start: `2026-07-13 23:30` Europe/Berlin for Gemini completion, Qwen 50-form text+VLM, and OpenCUA direct-MCP 20-form top-up.
- Initial submissions `2296095`, `2296096`, and `2296097` were cancelled while still pending after detecting that an intermediate shell variable could expand the inherited `FORM_IDS` value incorrectly; no trials ran.
- Corrected delayed submissions: Gemini job `2296098`, Qwen job `2296099`, and OpenCUA direct-MCP job `2296100`. Scheduler verification showed all three `PENDING (BeginTime)` with `StartTime=2026-07-13T23:30:00`.

## 2026-07-14 50-Form Results and Qwen3-VL Top-Up

- Gemini job `2296098` completed all `26` completion/retry trials. Combined with the usable prior batch, Gemini has `50/50` forms, `12/50` full fills, and `296/409` verified fields.
- OpenCUA direct-MCP job `2296100` completed all `20` top-up trials. Combined coverage is `50/50` forms, `15/50` full fills, and `290/409` verified fields.
- Qwen job `2296099` ended as Slurm `NODE_FAIL` after Qwen Text completed `50/50` forms and Qwen3-VL produced `42` summaries. Qwen3-VL has `39` usable results, three context-limit errors, and eight forms interrupted before summary creation.
- Added `QWEN_MODEL_IDS` filtering to `scripts/run_qwen_direct_mcp_matrix.sh` so a VLM-only top-up does not rerun Qwen Text.
- Submitted VLM-only top-up job `2298494` for the `11` missing/unusable forms, experiment `qwen_vlm_fill_only_done_50_topup11_20260714_r2_step32`, with `DIRECT_MCP_VLM_MAX_NEW_TOKENS=768` to avoid the prior 32,768-token context overflow. Initial scheduler state: `PENDING`.
- Added a compact, GitHub-trackable export at `data/model_baseline_exports/fill_only_done_50_20260714/`, containing `README.md`, `aggregate.json`, and `trials.csv`. Raw screenshots, videos, and model traces remain ignored because the three raw experiment directories total roughly `7 GB`.
- Added `scripts/export_fill_only_done_50_results.py` to regenerate the compact export after the VLM top-up completes.

## 2026-06-09 Target-300 Chain Update

Implementation changes:

- `scripts/submit_eval_target_chain.py` now derives the active run indexes from `--target-trials`; `300 / 50` forms means runs `1-6`.
- The scheduler now targets all four conditions independently:
  - Qwen text direct-MCP.
  - Qwen VLM direct-MCP.
  - OpenCUA native screenshot.
  - OpenCUA direct-MCP tools.
- The scheduler scans `data/model_baselines/**/summary.json` globally and treats each model independently, so completed Qwen text pairs do not hide missing Qwen VLM or OpenCUA pairs.
- `scripts/run_qwen_direct_mcp_matrix.sh`, `scripts/run_opencua_direct_matrix.sh`, and `scripts/run_opencua_direct_mcp_matrix.sh` now use global unique-pair skip behavior when `SKIP_COMPLETED=1`.
- `src/baselines/run_qwen_direct_mcp_eval.py` now filters visible control metadata and prompt text against the actual Playwright MCP tools visible to the model. The prompt no longer advertises `browser_check` unless the active MCP server exposes it.
- Radio/checkbox controls use `browser_click` as the compatible default direct-MCP action.
- `scripts/analyze_eval_results.py` now reports target coverage against `300` unique form-runs, not the earlier `500` target.

Validation:

- Unit tests passed for the direct-MCP prompt/tool visibility and target-chain counting:

```bash
python -m unittest tests.test_qwen_direct_mcp_eval tests.test_submit_eval_target_chain
```

- Syntax checks passed for the updated Python and shell entrypoints.
- First OpenCUA direct-MCP smoke submission `2247425` failed before trials because `.venv-opencua/bin/vllm` had a stale shebang. The vLLM launchers were updated to call `python -m vllm.entrypoints.openai.api_server` through the active virtualenv interpreter.
- Follow-up OpenCUA direct-MCP smoke job `2247451` completed all five forms:
  - Experiment ID: `opencua_direct_mcp_tools_toolcontract_smoke_v2_20260608`
  - Forms: `conf_interest`, `event_rsvp`, `course_feedback`, `internship_app`, `workshop_signup`
  - Run index: `2`
  - Result: `5/5` direct-MCP trials completed and submitted.
  - Tool transport: `text_tool_call_fallback`.
  - Invalid tool calls: `0`.
  - Hidden `browser_check` errors: `0`.
  - Scored correctness: `conf_interest 7/7`, `event_rsvp 6/6`, `course_feedback 6/9`, `internship_app 3/12`, `workshop_signup 7/7`.

New submitted chain:

- Attempted helper submission:

```bash
CHAIN_TARGET_TRIALS=300 CHAIN_TRACKS=opencua-direct-mcp CHAIN_DATE_STAMP=20260609 sbatch scripts/slurm_submit_eval_target_chain.sbatch
```

- Result: rejected by cluster policy because the helper job requested no GPU resources (`QOSMinGRES`). No evaluation work was queued by this failed helper submission.
- Direct local submission was then used:

```bash
.venv/bin/python scripts/submit_eval_target_chain.py --target-trials 300 --tracks opencua-direct-mcp --date-stamp 20260609 --submit
```

- Submitted dependency chain:
  - `2248104`: `opencua_direct_mcp_tools_target300_run1_20260609`, pending on priority at submission check.
  - `2248105`: `opencua_direct_mcp_tools_target300_run2_20260609`, dependency `afterok:2248104`.
  - `2248106`: `opencua_direct_mcp_tools_target300_run3_20260609`, dependency `afterok:2248105`.
  - `2248107`: `opencua_direct_mcp_tools_target300_run4_20260609`, dependency `afterok:2248106`.
  - `2248108`: `opencua_direct_mcp_tools_target300_run5_20260609`, dependency `afterok:2248107`.
  - `2248109`: `opencua_direct_mcp_tools_target300_run6_20260609`, dependency `afterok:2248108`.
- Queue state at submission check:
  - `2248104`: `PD (Priority)`.
  - `2248105`-`2248109`: `PD (Dependency)`.
- This is finite, dependency-chained work toward the 300-trial cap, not an infinite resubmission loop.

Additional remaining-condition backfill submitted after resource check:

- Resource requests checked before submitting:
  - Qwen direct-MCP jobs request `2` GPUs, `12` CPUs, `120G` memory, and `24:00:00`.
  - OpenCUA native jobs request `4` GPUs, `16` CPUs, `180G` memory, and `24:00:00`.
  - OpenCUA direct-MCP jobs request `4` GPUs, `16` CPUs, `180G` memory, and `24:00:00`.
- Full user queue check before adding the remaining jobs showed only the existing OpenCUA direct-MCP chain (`2248104`-`2248109`), with `2248104` pending on priority and the rest dependency-held.
- To avoid simultaneous heavy GPU demand, the remaining jobs were submitted as one serial dependency chain:
  - OpenCUA direct-MCP target chain first: `2248104` -> `2248105` -> `2248106` -> `2248107` -> `2248108` -> `2248109`.
  - Qwen backfill after OpenCUA direct-MCP: `2248111` -> `2248112` -> `2248113` -> `2248114`.
  - OpenCUA native backfill after Qwen: `2248115` -> `2248116`.
- Qwen submitted jobs:
  - `2248111`: `qwen_direct_mcp_target300_run1_20260609`, dependency `afterok:2248109`, missing `remote_setup` run `1`.
  - `2248112`: `qwen_direct_mcp_target300_run4_20260609`, dependency `afterok:2248111`, remaining run-4 forms.
  - `2248113`: `qwen_direct_mcp_target300_run5_20260609`, dependency `afterok:2248112`, remaining run-5 forms.
  - `2248114`: `qwen_direct_mcp_target300_run6_20260609`, dependency `afterok:2248113`, remaining run-6 forms.
- OpenCUA native submitted jobs:
  - `2248115`: `opencua_control_guidance_target300_run5_20260609`, dependency `afterok:2248114`, all 50 run-5 forms.
  - `2248116`: `opencua_control_guidance_target300_run6_20260609`, dependency `afterok:2248115`, all 50 run-6 forms.
- Queue state after adding the remaining jobs:
  - `2248104`: `PD (Priority)`, `gres/gpu:4`, `16` CPUs, `180G`, `24:00:00`.
  - `2248105`-`2248109`: `PD (Dependency)`, each `gres/gpu:4`, `16` CPUs, `180G`, `24:00:00`.
  - `2248111`-`2248114`: `PD (Dependency)`, each `gres/gpu:2`, `12` CPUs, `120G`, `24:00:00`.
  - `2248115`-`2248116`: `PD (Dependency)`, each `gres/gpu:4`, `16` CPUs, `180G`, `24:00:00`.
- This should keep the active load to one evaluation batch at a time and avoid a burst of parallel 4-GPU jobs.

## 2026-06-10 Job Status Check

Checked SLURM accounting and refreshed `docs/eval_results/analysis/` on 2026-06-10.

Evaluation chain status:

- `2248104` (`opencua-mcp`, experiment `opencua_direct_mcp_tools_target300_run1_20260609`) completed successfully:
  - SLURM state: `COMPLETED`, exit code `0:0`.
  - Runtime: `13:01:40`, started `2026-06-09T17:54:22`, ended `2026-06-10T06:56:02`.
  - Matrix output: `direct_eval_total=50`, `direct_eval_passed=46`, `direct_eval_failed=4`.
  - Failed form-level statuses in stderr: `course_enrollment`, `field_trip`, `publication_submission`, `sports_tournament`.
  - The run still wrote `50` `summary.json` files under `data/model_baselines/opencua_direct_mcp_tools_target300_run1_20260609/...`.
- The next chain job `2248105` is pending on cluster priority:
  - `2248105`: `PENDING (Priority)`.
  - `2248106`-`2248109`: `PENDING (Dependency)`.
  - `2248111`-`2248114`: `PENDING (Dependency)`.
  - `2248115`-`2248116`: `PENDING (Dependency)`.
- No submitted target-chain job is currently running at this check.

Refreshed thesis analysis status after `2248104`:

- `scripts/analyze_eval_results.py` completed and wrote updated outputs to `docs/eval_results/analysis/`.
- Canonical discovered trials increased from `716` to `766`.
- OpenCUA MCP thesis row now shows:
  - `70` trials across `50` forms.
  - `55/300` unique target form-runs (`18.3%` target coverage).
  - Submit rate `78.6%`.
  - Exact success rate `27.1%`.
  - Scored accuracy `57.8%`.
  - Reference coverage `29/70` (`41.4%`).
- `paired_efficiency_comparison.csv` now has `249` rows.

Reference generation status:

- Reference jobs submitted on 2026-06-09 all failed:
  - `2248123`: `FAILED`, exit code `1:0`, failed after `00:00:14`.
  - `2248128`: `FAILED`, exit code `1:0`, failed after `00:00:02`.
  - `2248140`: `FAILED`, exit code `1:0`, failed after `00:24:46`.
- `2248123` and `2248128` failed before useful generation because the Playwright Chromium executable was missing:
  - `2248123` looked under `/home/bhve224e/.cache/ms-playwright/...`.
  - `2248128` looked under the workspace `.playwright-browsers/...`.
- `2248140` generated partial reference artifacts, then failed on a Google Forms time dropdown:
  - Failure: `TimeoutError: locator.click` while clicking `PM`.
  - Cause in Playwright log: Google Forms overlay/options intercepted pointer events.
- Current reference coverage after refresh:
  - `reference_coverage.csv`: `300` target form-runs, `38` usable references, `38` artifact-present rows, `38` video-available rows.
  - Local `data/forms` currently has `89` `run_000*` directories and `28` `.webm` files under those run directories.
- Efficiency plots and summaries remain provisional until the reference-generation job is fixed and rerun for the missing form-runs.

## 2026-06-10 Reference Generation Fix

Implementation changes:

- `src/engine/runner.py` now has `--continue-on-run-error` for all-form/batch reference generation.
- Failed reference runs write `failure_manifest.json` with form/run identity, error type/message, and artifact presence.
- The runner now defaults Node Playwright browser binaries to the workspace cache `.playwright-browsers-node`, avoiding ambiguity between home-cache and workspace-cache Chromium paths.
- Browser preflight now checks for an actual executable Chromium binary, not just a cache directory name.
- `src/engine/mcp_browser_engine.py` now uses a robust meridiem setter for Google Forms time controls:
  - normal click,
  - forced click,
  - DOM click fallback.
- `scripts/analyze_reference_dataset.py` was added to compute reference coverage and efficiency inputs from `tool_trace.jsonl`.
- Reference action count is now the number of valid tool-trace events with a non-empty `name`.
- Reference duration is now `last_valid_trace_event.t_s - first_valid_trace_event.t_s`, not annotation max time.
- `scripts/analyze_eval_results.py` and the baseline efficiency helper now use the same tool-trace action/duration semantics.

Validation:

```bash
.venv/bin/python -m py_compile src/engine/runner.py src/engine/mcp_browser_engine.py scripts/analyze_eval_results.py scripts/analyze_reference_dataset.py src/baselines/run_baseline_eval.py
.venv/bin/python -m unittest tests.test_reference_efficiency
.venv/bin/python -m unittest tests.test_mcp_browser_engine
.venv/bin/python scripts/analyze_reference_dataset.py
.venv/bin/python scripts/analyze_eval_results.py
```

Current refreshed reference state:

- `docs/eval_results/reference_analysis/reference_coverage_summary.csv`: `28/300` usable references.
- `docs/eval_results/analysis/reference_coverage.csv`: `28/300` usable references and `28` video-available references.
- The lower count compared with the previous `38/300` is intentional: references with stale `annotations.video_path` values but no actual `.webm` file are no longer counted as usable.

Recommended rerun command:

```bash
sbatch \
  --nodes=1 \
  --gres=gpu:1 \
  --job-name=reference-forms-runs1-6 \
  --output=logs/slurm/reference-forms-runs1-6-%j.out \
  --error=logs/slurm/reference-forms-runs1-6-%j.err \
  --time=12:00:00 \
  --cpus-per-task=4 \
  --mem=8G \
  --wrap='.venv/bin/python src/engine/runner.py --all-forms --dataset-root data/forms --num-runs 6 --start-index 1 --skip-existing-video --overwrite-missing-video --continue-on-run-error --trace-mode mcp --interaction-mode mcp_server --headless --screenshots'
```

Rerun submission:

- First submission attempt without node/GRES flags was rejected by cluster policy (`QOSMinGRES` and missing node count).
- Resubmitted with `--nodes=1 --gres=gpu:1`.
- New reference-generation job: `2248236`.
- Result: `2248236` failed after `00:00:24` because the bare `sbatch --wrap` environment did not load Node.js, so `npx` was not available.
- Fix: resubmit through `scripts/run_baselines_mcp.sh`, which loads `nodejs/20.13.1`, adds `.node-tools/node_modules/.bin` to `PATH`, and sets `PLAYWRIGHT_BROWSERS_PATH`.
- Additional code fix: Chromium cache detection now recognizes both `chrome-linux/chrome` and `chrome-linux64/chrome`; the current workspace cache uses `.playwright-browsers-node/chromium-1212/chrome-linux64/chrome`.
- Replacement reference-generation job: `2248565`.
- Initial queue state: `PENDING (Priority)` on 2026-06-11.
- Replacement command:

```bash
sbatch \
  --nodes=1 \
  --gres=gpu:1 \
  --job-name=reference-forms-runs1-6 \
  --output=logs/slurm/reference-forms-runs1-6-%j.out \
  --error=logs/slurm/reference-forms-runs1-6-%j.err \
  --time=12:00:00 \
  --cpus-per-task=4 \
  --mem=8G \
  --wrap='bash scripts/run_baselines_mcp.sh --all-forms --num-runs 6 --start-index 1 --skip-existing-video --overwrite-missing-video --continue-on-run-error --screenshots'
```

Integrity verification after resubmission:

- `.venv/bin/python -m py_compile src/engine/runner.py src/engine/mcp_browser_engine.py scripts/analyze_eval_results.py scripts/analyze_reference_dataset.py src/baselines/run_baseline_eval.py` passed.
- `.venv/bin/python -m unittest tests.test_reference_efficiency tests.test_mcp_browser_engine` passed (`10` tests).
- `.venv/bin/python scripts/analyze_reference_dataset.py` passed and still reports `28/300` usable references, as expected before `2248565` runs.
- `.venv/bin/python scripts/analyze_eval_results.py` passed and refreshed thesis outputs.
- Current thesis analysis now sees `826` total model trials; OpenCUA MCP is at `130` trials and `115/300` unique target form-runs.

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
- Filters advertised control tools against the actual model-visible Playwright MCP tools.
- Uses `browser_click` for radio/checkbox controls unless a future MCP server exposes a more specific compatible tool.
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
