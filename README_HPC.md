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

- `configs/baselines/minimal_models.json`

Install or verify them with:

```bash
source .venv/bin/activate
python3 scripts/install_minimal_models.py
python3 scripts/verify_runtime_setup.py --skip-playwright-smoke
```

Expected local model directories:

- `models/text_qwen25_3b_instruct`
- `models/vlm_qwen25_vl_3b_instruct`

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

## Pilot Baseline Matrix

Current v1 pilot matrix:

- `text_qwen25_3b_instruct` × `conf_interest` × `run_0001`
- `text_qwen25_3b_instruct` × `event_rsvp` × `run_0001`
- `vlm_qwen25_vl_3b_instruct` × `conf_interest` × `run_0001`
- `vlm_qwen25_vl_3b_instruct` × `event_rsvp` × `run_0001`

Protocol defaults:

- execution backend: `mcp_server`
- inputs before interaction: form URL + answer entries (`label`, `widget_type`, `value`)
- no full form spec
- no upfront DOM dump
- per-step observations: screenshot + compact page text + last action result
- one strict JSON action per turn
- `max_steps=20`
- `timeout_s=900`
- invalid action budget: `0` (`0` means unlimited until `max_steps`)
- default `max_new_tokens=160` for baseline action generation
- partial success metrics:
  - `attempted_correctness`
  - `verified_correctness` as headline metric

## Baseline Commands

Run the MCP-backed model baseline matrix directly:

```bash
bash scripts/run_model_baseline_matrix.sh
```

Useful overrides:

```bash
EXPERIMENT_ID=baseline_mcp_trial2 bash scripts/run_model_baseline_matrix.sh
MAX_STEPS=25 TIMEOUT_S=1200 bash scripts/run_model_baseline_matrix.sh
RUN_INDEX=1 MAX_NEW_TOKENS=160 INVALID_ACTION_BUDGET=0 bash scripts/run_model_baseline_matrix.sh
```

Submit the same matrix through Slurm:

```bash
sbatch scripts/slurm_baseline_mcp.sbatch
```

Check status and logs:

```bash
squeue -u "$USER"
sacct -j <job_id> --format=JobID,JobName,Partition,State,Elapsed,ExitCode -P
tail -f logs/slurm/model-baseline-mcp-<job_id>.out
```

## Validation Before Submission

```bash
python3 scripts/verify_baseline_integrity.py
python3 scripts/verify_runtime_setup.py --skip-playwright-smoke
python3 scripts/eval_model_baseline_smoke.py --include-kinds text_llm,vlm --exclude-providers api_over_mcp --strict
python3 -m py_compile src/baselines/run_baseline_eval.py src/engine/form_engine.py src/engine/mcp_browser_engine.py
bash -n scripts/run_model_baseline_matrix.sh scripts/slurm_baseline_mcp.sbatch
```

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
