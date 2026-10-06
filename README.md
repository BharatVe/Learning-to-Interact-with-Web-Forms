# Learning to Interact with Web Forms

A benchmark for LLM, VLM and computer-use agents that fill in web forms, compared
against scripted "ideal" Playwright runs of the same forms and answers.

```
data/generator/*.csv ──make data──▶ src/forms/<id>/spec.json + data/answers/<id>/runs.json  (50 forms × 10 answer sets)
                                    └─▶ LocalForms copy: self-hosted Flask site, src/forms_localforms/, data/answers_localforms/

make ideal-runs ──▶ data/forms/<id>/runs/run_XXXX/      scripted reference: video, tool_trace.jsonl, annotations.json
make eval/matrix ─▶ data/model_baselines/<experiment>/<model>/<form>/run_XXXX/<trial>/   one directory per model trial
make report ──────▶ reports/                             accuracy, completion, efficiency vs the ideal run, failure mix
```

Everything runs through `make <target>`, a thin layer over one Python CLI
(`python -m formbench`). Every command validates its inputs, prints the resolved
settings, and supports `DRY_RUN=1` where it would start something expensive.

## Quick start

```bash
cp .env.example .env        # HPC: set MODULES and CACHE_ROOT (see docs/HPC.md); laptop: clear MODULES
make setup                  # .venv, pinned Playwright MCP, Chromium   (WITH=vllm also builds the vLLM env)
make doctor                 # checks everything and prints the fix for anything missing
make test                   # ~220 tests, ~25 s, no GPU or network needed
```

## Commands

| Area | Command | What it does |
|---|---|---|
| **Setup** | `make setup` | Create/refresh environments and browsers (idempotent). `WITH=localhf\|vllm\|all` |
| | `make doctor` | Environment, dataset and registry health check, with fixes |
| | `make test` | Unit + integration tests (`V=1` for verbose) |
| **Dataset** | `make data` | Rebuild answer sets and the LocalForms copy, then validate (`FROM_CSV=1` also re-syncs specs from the generator CSVs) |
| | `make data-check` | Validate specs, answers and committed reference traces |
| | `make serve-forms` | Serve the LocalForms site locally |
| **Ideal runs** | `make ideal-status` | Which form/answer-set reference runs exist (`PLATFORM=localforms`) |
| | `make ideal-runs` | Regenerate scripted Playwright reference runs. `FORMS=a,b RUNS=1-10 PLATFORM=… OVERWRITE=1 HEADED=1 SUBMIT=1 DRY_RUN=1` |
| **Models** | `make models` | Registered models, what each needs, whether weights are local (`ALL=1` adds legacy) |
| | `make model-check MODEL=…` | Preflight: registry, API key, weights, endpoint, GPUs (`SMOKE=1` sends a test request) |
| | `make serve-model MODEL=…` | Start a model's vLLM server in the foreground (eval/matrix reuse it) |
| | `make install-models MODEL=…` | Download Hugging Face weights into `models/` (`DRY_RUN=1`) |
| **Evaluation** | `make eval MODEL=… FORM=… RUN=…` | One trial. `SET="max_steps=8 fill_only_done=true" PLATFORM=localforms DRY_RUN=1 SUBMIT=1` |
| | `make experiments` | List experiment manifests (`configs/experiments/`) |
| | `make matrix EXPERIMENT=…` | Run an experiment here: forms × runs × models, servers started/stopped per model, completed trials skipped. `COHORT= MODELS= FORMS= RUNS= SUFFIX= DRY_RUN=1` |
| | `make submit EXPERIMENT=…` | Same, as Slurm jobs sized from the registry. `SPLIT=cohort\|model\|run CHAIN=afterok DRY_RUN=1` |
| **Analytics** | `make report` | Current analytics into `reports/`: core thesis tables + plots, per-experiment overview, trial tracker |
| | `make reference-report` | Ideal-run summary into `reports/reference/` |
| | `make studies` / `make study NAME=…` | List / regenerate a committed thesis study (`CHECK=1` only verifies) |
| | `make inspect TRIAL=…` | One trial's summary, step inputs/outputs (`STEPS=all`) and media (`MEDIA=1`) |

`make help` prints the same list. Each target maps to `python -m formbench …`
(see `--help` on any subcommand) if you prefer calling the CLI directly.

## Common workflows

**Try a model on one form before spending GPU hours**

```bash
make model-check MODEL=computer_use_gemini_35_flash_lowcost
make eval MODEL=computer_use_gemini_35_flash_lowcost FORM=conf_interest RUN=1 SET="max_steps=4" DRY_RUN=1   # see the exact command
make eval MODEL=computer_use_gemini_35_flash_lowcost FORM=conf_interest RUN=1 SET="max_steps=4"
make inspect TRIAL=adhoc_computer_use_gemini_35_flash_lowcost_<date>/computer_use_gemini_35_flash_lowcost/conf_interest/run_0001 STEPS=all
```

**Run or resume a full experiment on the cluster**

```bash
make experiments                                         # what exists
make submit EXPERIMENT=fill_only_done_30 DRY_RUN=1       # inspect the job scripts and resources
make submit EXPERIMENT=fill_only_done_30 SPLIT=cohort CHAIN=afterok
squeue -u $USER; tail -f logs/slurm/fb-fill_only_done_30-*.out
```

Re-submitting the same manifest only runs trials that are still missing. To repeat
an experiment from scratch into new folders use `SUFFIX=_rerun1`.

**Regenerate ideal runs** (official Playwright MCP server, headless, existing videos skipped)

```bash
make ideal-status
make ideal-runs FORMS=conf_interest RUNS=7-10 SUBMIT=1        # HPC: as a CPU job
make ideal-runs PLATFORM=localforms FORMS=conf_interest RUNS=1   # LocalForms site is started for you
```

> Google Forms runs **submit real responses** to the live forms.

**Swap or add a model:** edit `configs/models.json` and run `make model-check MODEL=<id>`.
[docs/MODELS.md](docs/MODELS.md) covers the fields, every provider, and what each
check failure means.

**Look at results:** `make report`, then open `reports/experiment_overview.csv`,
`reports/thesis_model_summary.csv` and `reports/plots/*.svg`. Thesis-cited outputs
live in `docs/eval_results/` and change only through `make study`; see
[docs/eval_results/README.md](docs/eval_results/README.md).

## Failsafes

- `make model-check` (run automatically before `eval`/`matrix`) validates the registry
  entry against a per-provider schema, checks API keys (and key-file permissions), local
  weights, transformers compatibility (`SMOKE=1`), the endpoint (reachable and serving the
  expected model name), GPU count and memory, and **warns about every environment variable
  that overrides the registry** (`OPENAI_BASE_URL`, `OPENAI_MODEL`, …).
- On a node without the required GPUs a run stops with a hint to use `make submit`.
- vLLM servers are reused if already serving the right model, rejected if another model
  holds the port, and always stopped (whole process group) when the run ends. Failed
  startups print the end of the server log.
- Slurm resources come from the registry. Inside jobs, server and LocalForms ports derive
  from the job id, so concurrent jobs on one node do not collide.
- Trial failures are recorded as benchmark outcomes, and a run summary is written to
  `logs/matrix/`. `--fail-fast` stops at the first runner error. VLM timeouts and OOM
  failures trigger the registered fallback model.
- `make report` never touches committed thesis files. `evaluation_additions/manifest.json`
  hashes every thesis input, and `make test` verifies those hashes.

## Repository layout

```
Makefile, .env.example         entry points and site settings
configs/models.json            model registry (one entry per model; docs/MODELS.md)
configs/experiments/*.json     reproducible experiment manifests
configs/analysis/*.json        report cohorts and study configs
src/formbench/                 CLI: settings, checks, serving, matrix, Slurm, ideal runs, reports
src/engine/                    scripted Playwright engine for ideal runs (local + MCP backends)
src/baselines/                 model runners (one per protocol) + model adapters
src/dataset/                   generator CSV → specs → answers → LocalForms
src/analysis/                  core report, overview, tracker, studies/
scripts/                       env.sh (environment wrapper), setup.sh, Playwright MCP runtime helper
data/                          forms, answers, reference runs (traces committed, videos ignored), exports
docs/                          methodology, results, HPC and model guides
evaluation_additions/          LocalForms site, additional comparisons, provenance manifest
tests/                         unittest suite (make test)
```

Raw trial artefacts (`data/model_baselines/`), model weights (`models/`), logs, reports
and caches are git-ignored.

## Further reading

| Document | Contents |
|---|---|
| [docs/MODELS.md](docs/MODELS.md) | Model registry, providers and protocols, adding/swapping models, experiment manifests |
| [docs/HPC.md](docs/HPC.md) | Cluster setup, modules, caches, Slurm, troubleshooting |
| [docs/eval_results/README.md](docs/eval_results/README.md) | Index of results and the command that regenerates each |
| [docs/LOCALFORMS_METHODOLOGY.md](docs/LOCALFORMS_METHODOLOGY.md) | How the self-hosted LocalForms platform was built |
| [docs/MIGRATION.md](docs/MIGRATION.md) | Old scripts → new commands |
| [docs/EVAL_IMPLEMENTATION_TRACKING_LOG.md](docs/EVAL_IMPLEMENTATION_TRACKING_LOG.md) | Historical decision and job log |
