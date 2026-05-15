# HPC Workflow README

This file covers HPC-specific workflow details that are intentionally not expanded in `README.md`.

## Canonical Working Directory

Use this as the canonical git working copy:

- `/home/h1/bhve224e/workspaces/Learning-to-Interact-with-Web-Forms`

The backing workspace resolves to the Horse storage path used in batch jobs:

- `/data/horse/ws/bhve224e-thesis-draft-20260224/Learning-to-Interact-with-Web-Forms`

Check the resolved path if needed:

```bash
readlink -f /home/h1/bhve224e/workspaces/Learning-to-Interact-with-Web-Forms
```

## One-Time Setup

```bash
cd /home/h1/bhve224e/workspaces/Learning-to-Interact-with-Web-Forms
bash scripts/hpc_setup.sh
source .venv/bin/activate
```

## Model Installation

Configured baseline models live in:

- `configs/baselines/track_baseline_models.json`

Install or verify them with:

```bash
source .venv/bin/activate
python3 scripts/install_minimal_models.py --config configs/baselines/track_baseline_models.json
python3 scripts/verify_runtime_setup.py --config configs/baselines/track_baseline_models.json --skip-playwright-smoke
```

Expected local model directories:

- `models/text_qwen3_30b_a3b_instruct_2507`
- `models/vlm_qwen3_vl_30b_a3b_instruct`

Computer-use track uses local `vLLM` serving and does not install weights into `models/` through the baseline installer.

## MCP Runtime Requirements

The MCP-backed baseline path requires Node tooling and the Playwright MCP package.

This project uses:

- `module load release/25.06 GCCcore/13.3.0 nodejs/20.13.1`
- local package cache under `.node-tools/`
- Node Playwright browser cache under `.playwright-browsers-node/`

The baseline scripts export:

- `PATH=$ROOT_DIR/.node-tools/node_modules/.bin:$PATH`
- `PLAYWRIGHT_BROWSERS_PATH=$ROOT_DIR/.playwright-browsers-node`

## Reference Runs vs Model Baselines

Reference/scripted generation runs remain immutable under:

- `data/forms/<form_id>/runs/run_XXXX/`

Model baseline evaluation artifacts are written under:

- `data/model_baselines/<experiment_id>/<model_id>/<form_id>/<answer_run_id>/<trial_id>/`

Each canonical baseline trial now contains:

- `summary.json`
- `annotations.json`
- `model_io.jsonl`
- `tool_trace.jsonl`
- `answers_instance.json`
- `<form_id>_<trial_id>.webm`
- `observations/step_0000.png`, `step_0001.png`, ...
- `final.png` or `error.png` when available

Each experiment also gets:

- `data/model_baselines/<experiment_id>/manifest.jsonl`

The runner never uses `--overwrite-existing`. Repeated trials create new `trial_id` directories.

## Canonical Track Baseline Matrix

Canonical thesis-primary setup now separates two benchmark families:

- Family A: direct Playwright MCP tool use
  - `text_qwen3_30b_a3b_instruct_2507`
  - `vlm_qwen3_vl_30b_a3b_instruct`
- Family B: native computer-use
  - `computer_use_opencua_32b` (`OpenCUA-32B` over local `vLLM` + MCP-backed browser execution)

Protocol defaults:

- execution backend: `mcp_server`
- inputs before interaction: form URL + answer entries (`label`, `widget_type`, `value`)
- no full form spec
- no helper-assisted fill path in the thesis-primary comparison
- Family A uses raw Playwright MCP tool calls rather than the legacy benchmark action schema
- Family B uses native screenshot-to-action prediction
- OpenCUA direct pilot defaults to `max_steps=64`, `timeout_s=5400`
- invalid action budget: `0` (`0` means unlimited until `max_steps`)
- OpenCUA direct pilot defaults to `max_new_tokens=384`
- partial success metrics:
  - `attempted_correctness`
  - `verified_correctness` as headline metric
- efficiency baseline:
  - the matching scripted Playwright reference run in `data/forms/<form_id>/runs/run_XXXX/`
  - same `form_id` and same `answer_run_id`
  - action count is defined as the count of valid browser/tool events in `tool_trace.jsonl`

## Baseline Commands

Run the canonical 3-track baseline orchestrator directly:

```bash
CONFIG_PATH=configs/baselines/track_baseline_models.json \
DIRECT_PROVIDER=opencua_local \
bash scripts/run_track_baseline_matrix.sh
```

Pilot example (`5 forms x runs 1..3 x 3 models`):

```bash
CONFIG_PATH=configs/baselines/track_baseline_models.json \
FORM_IDS=conf_interest,event_rsvp,course_feedback,internship_app,workshop_signup \
RUN_INDEXES=1,2,3 \
DIRECT_MCP_EXPERIMENT_ID=track_baseline_qwen_direct_mcp_pilot \
NATIVE_EXPERIMENT_ID=track_baseline_opencua_native_pilot \
DIRECT_PROVIDER=opencua_local \
bash scripts/run_track_baseline_matrix.sh
```

Full run example (`20 forms x runs 1..10 x 3 models`):

```bash
CONFIG_PATH=configs/baselines/track_baseline_models.json \
FORM_IDS=all \
RUN_INDEXES=1,2,3,4,5,6,7,8,9,10 \
DIRECT_MCP_EXPERIMENT_ID=track_baseline_qwen_direct_mcp_all20 \
NATIVE_EXPERIMENT_ID=track_baseline_opencua_native_all20 \
DIRECT_PROVIDER=opencua_local \
bash scripts/run_track_baseline_matrix.sh
```

Submit through Slurm:

```bash
FORM_IDS=all \
RUN_INDEXES=1,2,3,4,5,6,7,8,9,10 \
DIRECT_MCP_EXPERIMENT_ID=track_baseline_qwen_direct_mcp_all20 \
NATIVE_EXPERIMENT_ID=track_baseline_opencua_native_all20 \
DIRECT_PROVIDER=opencua_local \
sbatch scripts/slurm_track_baseline.sbatch
```

Check status and logs:

```bash
squeue -u "$USER"
sacct -j <job_id> --format=JobID,JobName,Partition,State,Elapsed,ExitCode -P
tail -f logs/slurm/track-baseline-<job_id>.out
```

## Validation Before Submission

For OpenCUA validation, make sure `vllm` is available on `PATH` and that the local endpoint is started before running the computer-use smoke check.

```bash
python3 scripts/verify_baseline_integrity.py
python3 scripts/verify_runtime_setup.py --config configs/baselines/track_baseline_models.json --skip-playwright-smoke
python3 scripts/eval_model_baseline_smoke.py --config configs/baselines/track_baseline_models.json --include-kinds text_llm,vlm --strict
python3 scripts/eval_model_baseline_smoke.py --config configs/baselines/track_baseline_models.json --include-kinds computer_use_agent --strict
python3 -m py_compile src/baselines/run_baseline_eval.py src/engine/form_engine.py src/engine/mcp_browser_engine.py
bash -n scripts/run_track_baseline_matrix.sh scripts/slurm_track_baseline.sbatch
```

Legacy size-based and benchmark-action mediated scripts remain available for historical analysis, but they are no longer the thesis-primary runbook.

## Cache Policy

To avoid `/home` quota lockouts, batch scripts now relocate reproducible caches to `/data/horse/ws/bhve224e-thesis-draft-20260224/cache` via:

- `XDG_CACHE_HOME`
- `HF_HOME`
- `TRANSFORMERS_CACHE`
- `PIP_CACHE_DIR`
- `UV_CACHE_DIR`
- `PLAYWRIGHT_BROWSERS_PATH`

## Reference Efficiency Reporting

Use the scripted reference runs as the efficiency baseline for model trials:

```bash
python3 scripts/summarize_reference_efficiency.py \
  --experiment-id track_baseline_qwen_direct_mcp_pilot \
  --experiment-id track_baseline_opencua_native_pilot \
  --output logs/reference_efficiency_summary.json
```

This reports raw and normalized efficiency:

- model action count and duration
- matching reference action count and duration
- `action_overhead_ratio`
- `time_overhead_ratio`

## Workspace Cleanup

Legacy baseline outputs are no longer part of the active workflow:

- `data/baseline_eval/`
- mirrored `logs/baseline_eval/`

Preview cleanup:

```bash
bash scripts/cleanup_legacy_baseline_outputs.sh
```

Apply cleanup:

```bash
bash scripts/cleanup_legacy_baseline_outputs.sh --apply
```

## Storage and Git Hygiene

Ignored runtime-heavy paths include:

- `.venv/`
- `models/`
- `.playwright-browsers/`
- `.playwright-browsers-node/`
- `.node-tools/`
- `logs/`
- `data/model_baselines/`
- `data/baseline_eval/`

Reference run artifacts under `data/forms/**/runs/` are also ignored to avoid committing generated files.

## Cluster Notes

- Run actual baseline evaluations through Slurm, not on the login node.
- TU Dresden enforces short runtime limits on login nodes.
- Request GPUs explicitly for VLM evaluation.
- The matrix wrapper continues across failed form/model trials and reports aggregate pass/fail counts at the end.
