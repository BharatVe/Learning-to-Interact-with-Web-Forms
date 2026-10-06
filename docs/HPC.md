# Running on the HPC cluster (Slurm)

The same `make` targets work on a laptop and on the cluster. Everything
cluster-specific lives in `.env` (copy `.env.example`); nothing is hard-coded.

## One-time setup (login node)

```bash
cp .env.example .env
#   MODULES="release/25.06 GCCcore/13.3.0 Python/3.12.3 nodejs/20.13.1"
#   CACHE_ROOT=/data/horse/ws/<your-workspace>/cache        # large filesystem, not /home
#   SLURM_ACCOUNT= / SLURM_PARTITION= / SLURM_EXCLUDE=        # if your project needs them
make setup              # core env, Playwright MCP, Chromium
make setup WITH=vllm    # also the vLLM env (.venv-opencua) for locally served models
make doctor
```

Two Python environments are used on purpose:

| Env | Interpreter | Used for |
|---|---|---|
| `.venv` | system Python 3.9 | CLI, dataset, ideal runs, API models, Qwen direct-MCP runner, analytics |
| `.venv-opencua` | module Python 3.12 | vLLM servers and the OpenCUA runners (`python_env: vllm`) |

`scripts/env.sh` (used by every `make` target) loads `MODULES`, keeps the module
`LD_LIBRARY_PATH` in `MODULE_LD_LIBRARY_PATH`, and restores the original one: the system
Python in `.venv` breaks with the module OpenSSL, while Node (Playwright MCP) and the
Python 3.12 vLLM env need the module libraries. formbench passes them to exactly those
subprocesses.

## Submitting work

Never run model evaluations on login nodes: they have short runtime limits and no GPUs.
`make eval`/`matrix` stop with a hint if the model needs GPUs that aren't visible.

```bash
make submit EXPERIMENT=track_baseline_pilot DRY_RUN=1   # print job scripts + resources, submit nothing
make submit EXPERIMENT=track_baseline_pilot SPLIT=model CHAIN=afterok
make eval MODEL=vlm_qwen3_vl_30b_a3b_instruct FORM=conf_interest RUN=2 SUBMIT=1
make ideal-runs FORMS=all RUNS=7-10 SUBMIT=1            # CPU-only job
```

- Resources per job are the maximum `resources` of the models in it (from
  `configs/models.json`). `SPLIT=cohort|model|run` creates one job per unit, and
  `CHAIN=afterok` makes each job wait for the previous one (`AFTER=<jobid>` for the first).
- Job scripts are written to `logs/slurm/jobs/`; output goes to
  `logs/slurm/<job-name>-<jobid>.out|.err`, vLLM logs to `logs/serve/`, and a JSON summary of
  each matrix run to `logs/matrix/`.
- Inside a job, `LOCAL=1` is set so the matrix runs in the allocation. vLLM
  (`serve.port: "auto"`) and the LocalForms site (`LOCALFORMS_PORT=auto`) use ports derived
  from the job id, so several jobs on one node never collide.
- Resubmitting is safe: completed (model, form, answer set) pairs are skipped according to
  the manifest's `skip_completed` setting.

```bash
squeue -u "$USER"
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,ExitCode -P
tail -f logs/slurm/fb-<experiment>-<cohort>-<jobid>.out
```

## Storage

| Path | Contents | In git |
|---|---|---|
| `data/forms/**/runs/run_*/` | Ideal runs: trace + annotations (videos/screens ignored) | partly |
| `data/model_baselines/` | Raw trials, tens of GB | no |
| `models/` | Local weights | no |
| `$CACHE_ROOT` | HF, pip, uv, Playwright browsers | no |
| `logs/`, `reports/` | Job logs, matrix summaries, current analytics | no |

Retention: each runner keeps the newest 5 trials per (experiment, model, form, answer set)
and moves older repeats to `<experiment>/_archive/` (`--set retention_window=N`).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `doctor`: `mcp chromium … not installed`, or MCP fails with a missing Chrome path after the workspace moved | The recorded Chromium path is absolute: run `make setup` (it re-resolves the record) |
| `node: error while loading shared libraries: libicu…` | Node started without module libraries: set `MODULES` in `.env` and run through `make` / `scripts/env.sh` |
| `.venv-opencua/bin/python: libpython3.12.so… not found` | Same: the vLLM env needs `MODULES` (Python 3.12) loaded |
| `vLLM exited … before becoming ready` | Read the log tail printed with the error (`logs/serve/<model>-vllm-<job>.log`); usually GPU memory or too few GPUs for `serve.tensor_parallel` |
| `port in use: … serves [...]` | Another server holds the port: stop it, or set a different `serve.port` |
| `LocalForms site did not start` | See `logs/serve/localforms-<port>.log`; check `make doctor` (`py:flask`) |
| Trials missing from summaries after a workspace move | Stored absolute paths are rebased onto the current checkout automatically; `scripts/env.sh .venv/bin/python -m formbench data relpaths` (dry run) lists the absolute paths left in committed reference runs |
