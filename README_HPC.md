# HPC Workflow README

This file covers HPC-specific workflow details that are intentionally not expanded in `README.md`.

## Canonical Working Directory

Use this as the canonical git working copy:

- `/home/h1/bhve224e/workspaces/Learning-to-Interact-with-Web-Forms`

The path below is a symlinked storage location and should be treated as an upstream/source mirror, not the main place for edits:

- `/home/h1/bhve224e/workspaces/horse/Learning-to-Interact-with-Web-Forms`

If you need to check it:

```bash
readlink -f /home/h1/bhve224e/workspaces/Learning-to-Interact-with-Web-Forms
```

## Why Two Paths Exist

- `horse/Learning-to-Interact-with-Web-Forms` is the backing location in this environment.
- `workspaces/Learning-to-Interact-with-Web-Forms` is your local writable project copy used for iterative development.

To avoid overlap/confusion:

1. Run git commands only in the canonical working directory.
2. Keep scripts, configs, and docs changes in the canonical directory.
3. Treat generated artifacts and model weights as local runtime assets (ignored by git).

## One-Time Setup (Canonical Directory)

```bash
cd /home/h1/bhve224e/workspaces/Learning-to-Interact-with-Web-Forms
bash scripts/hpc_setup.sh
source .venv/bin/activate
```

## Install Minimal Baseline Models

Model set is defined in:

- `configs/baselines/minimal_models.json`

Install:

```bash
source .venv/bin/activate
python3 scripts/install_minimal_models.py
```

Expected local storage:

- `models/text_qwen25_3b_instruct`
- `models/vlm_qwen25_vl_3b_instruct`

Model installer safeguards:

- validates config schema before download
- supports swapping models via config without crashing unrelated installs
- skips valid existing models (`--skip-if-valid`)
- continues on per-model failure by default (use `--strict` to fail fast)
- writes run logs to `logs/model_install_*.log`

Useful commands:

```bash
# show what would happen, no download
python3 scripts/install_minimal_models.py --dry-run

# install only selected ids
python3 scripts/install_minimal_models.py --include-ids text_qwen25_3b_instruct

# exclude one model id
python3 scripts/install_minimal_models.py --exclude-ids vlm_qwen25_vl_3b_instruct
```

## MCP/Computer-Use Runtime Requirements

For `--interaction-mode mcp_server`, install Node tooling on the execution environment:

- `node`
- `npm`
- `npx`
- `@playwright/mcp` (global or via `npx`)

Without Node, local Playwright mode still works (`--interaction-mode local`).

## Baseline Execution (Headless)

```bash
bash scripts/run_baselines_headless.sh --smoke-test-all-forms --overwrite-existing
```

Or single form:

```bash
bash scripts/run_baselines_headless.sh --form-id conf_interest --num-runs 3
```

## Slurm Example

```bash
sbatch scripts/slurm_baseline.sbatch
```

GPU-backed local baseline pilot:

```bash
sbatch scripts/slurm_local_baseline_eval.sbatch
```

Check queue and logs:

```bash
squeue -u "$USER"
scontrol show job <job_id>
tail -f logs/slurm/baseline-local-<job_id>.out
```

Important cluster notes:

- Productive baseline runs should go through Slurm, not the login node. ZIH enforces a 600-second runtime limit on login nodes.
- On `Capella`, request GPUs explicitly and stay within the documented CPU/memory-per-GPU limits.
- `Capella` recommends workspaces on `/data/cat` for active ML work and Python environments.
- If `sbatch` is rejected with a quota lock for `/home/$USER`, reduce home usage or move runtime-heavy assets (`.venv`, model caches, generated artifacts) into a workspace before retrying.

## Integrity Check Before Runs

```bash
python3 scripts/verify_baseline_integrity.py
python3 scripts/verify_runtime_setup.py
python3 scripts/preflight_baseline_eval.py
python3 scripts/eval_model_baseline_smoke.py --include-kinds text_llm,vlm --exclude-providers api_over_mcp
```

Start a safe pilot run (preflight gate + smoke evaluation):

```bash
bash scripts/start_baseline_pilot.sh
```

If you only want fast checks without model load generation:

```bash
python3 scripts/preflight_baseline_eval.py --skip-smoke-load
```

## Storage and Git Hygiene

Ignored runtime-heavy directories/files:

- `.venv/`
- `.playwright-browsers/`
- `models/`
- generated run artifacts under `data/forms/**/runs/`
- slurm logs and scratch outputs

This prevents accidental upload of large or irrelevant files.
