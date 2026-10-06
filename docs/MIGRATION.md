# Migration: old scripts → new commands

The repository was reorganised around `make` + `python -m formbench`. Everything removed
here is still in git tag `pre-cleanup-thesis-results` (`git show pre-cleanup-thesis-results:<path>`).
Output layouts (`data/forms/…`, `data/model_baselines/…`, `docs/eval_results/…`) are
unchanged, and existing results stay where they are.

## Setup and checks

| Before | Now |
|---|---|
| `scripts/hpc_setup.sh`, `setup_form_agent_env.sh` | `make setup` (`WITH=vllm` for the vLLM env) |
| `scripts/ensure_playwright_mcp_runtime.sh` (called by every matrix) | run once by `make setup`; `make doctor` checks it |
| `scripts/verify_runtime_setup.py` | `make doctor` |
| `scripts/preflight_baseline_eval.py`, `scripts/eval_model_baseline_smoke.py`, `scripts/check_openai_compat_server.py` | `make model-check MODEL=… [SMOKE=1]` (runs automatically before `eval`/`matrix`) |
| `scripts/verify_opencua_compatibility.py` | parser/prompt checks → `tests/test_opencua_direct_eval.py`; endpoint/forms checks → `model-check` / `data-check` |
| `scripts/verify_baseline_integrity.py`, `scripts/validate_answer_sets.py` | `make data-check` (`src/dataset/integrity.py`, `validate.py`) |
| `scripts/install_minimal_models.py` | `make install-models [MODEL=…] [DRY_RUN=1]` |
| module loads / cache exports repeated in ~12 scripts | `.env` (`MODULES`, `CACHE_ROOT`) + `scripts/env.sh` |

## Dataset and ideal runs

| Before | Now |
|---|---|
| `scripts/sync_generator_dataset.py`, `"From Generator - *.csv"` | `make data FROM_CSV=1` (`src/dataset/sync_specs.py`, `data/generator/*.csv`) |
| `scripts/generate_answer_sets.py --seed …` | `make data` (`src/dataset/answers.py`). `--seed` now takes effect; 0 reproduces the committed sets (the old `--seed` was silently ignored) |
| `scripts/generate_localforms.py` | `make data` (`src/dataset/localforms.py`) |
| `scripts/run_localforms_server.sh` | `make serve-forms` (started automatically for LocalForms runs) |
| `python3 src/engine/runner.py --all-forms --interaction-mode mcp_server --skip-existing-video …`, `scripts/run_baselines_{headless,mcp}.sh`, `slurm_baseline.sbatch` | `make ideal-runs [FORMS=] [RUNS=] [PLATFORM=] [SUBMIT=1]`; `make ideal-status`. `runner.py` now defaults to `--interaction-mode mcp_server` |

## Models, matrices and Slurm

| Before | Now |
|---|---|
| `configs/baselines/{track_baseline_models,minimal_models,qwen_localhf_matrix,text_size_localhf_models,formfactory_style_qwen3_vl,scads_gemma_api_matrix}.json` | one registry `configs/models.json` (legacy variants: `*_mediated`, `*_localhf`, see `docs/MODELS.md`). The SCADS Gemma API entry is not registered; it is the worked example for adding a hosted API model in `docs/MODELS.md` |
| `scripts/run_qwen_vllm_server.sh`, `scripts/run_opencua_vllm_server.sh` | `make serve-model MODEL=…`, or automatic per model inside `eval`/`matrix` (`serve` block) |
| `scripts/run_qwen_direct_mcp_matrix.sh` | `make matrix MODELS=text_qwen3_30b_a3b_instruct_2507,vlm_qwen3_vl_30b_a3b_instruct FORMS=… RUNS=…` |
| `scripts/run_opencua_direct_matrix.sh` | `make matrix MODELS=computer_use_opencua_32b …` (e.g. `EXPERIMENT=formfactory_opencua`) |
| `scripts/run_opencua_direct_mcp_matrix.sh` | `make matrix MODELS=computer_use_opencua_32b_direct_mcp …` |
| `scripts/run_opencua_direct_mcp_localforms_matrix.sh` | `make matrix EXPERIMENT=localforms_opencua_direct_mcp` or `PLATFORM=localforms` |
| `scripts/run_gemini_low_cost_matrix.sh` | `make matrix MODELS=computer_use_gemini_35_flash_lowcost …` |
| `scripts/run_model_baseline_matrix.sh` (mediated, budget profiles, fallbacks) | `make matrix MODELS=<mediated model> BUDGET=large_qwen3 …` (fallback models run automatically) |
| `scripts/run_track_baseline_matrix.sh` | `make matrix EXPERIMENT=track_baseline_pilot` / `track_baseline_full` |
| `scripts/run_fill_only_done_{10form,30form}_eval.sh`, `run_fill_only_done_50form_completion.sh` | `make matrix EXPERIMENT=fill_only_done_10` / `fill_only_done_30` / `fill_only_done_50_completion` (`COHORT=gemini\|qwen\|opencua_mcp` = old `TARGET`) |
| `scripts/run_comparison_matrix.sh`, `scripts/start_baseline_pilot.sh` | removed (legacy pilots); use a manifest |
| `scripts/slurm_*.sbatch` (10 files) | `make submit EXPERIMENT=… [SPLIT=…] [CHAIN=afterok]`: job scripts are generated with resources from the registry |
| `scripts/submit_eval_target_chain.py` | `make submit EXPERIMENT=target300 SPLIT=run CHAIN=afterok` (`skip_completed: any`) |
| `src/baselines/run_opencua_direct_mcp_eval.py` (thin wrapper) | still present; `formbench` calls the direct-MCP runner with `--model-kind computer_use_agent` |
| `src/baselines/run_gemini_native_computer_use_eval.py` | removed (no registry model used it; superseded by the low-cost Gemini runner) |
| env vars like `DIRECT_MCP_MAX_STEPS=32`, `FILL_ONLY_DONE=1` | `SET="max_steps=32 fill_only_done=true"` or a manifest cohort's `args` |
| `SKIP_COMPLETED=1` (matched any experiment) | manifest `skip_completed`: `experiment` (new default), `any` (old behaviour), `none` |

## Analytics

| Before | Now |
|---|---|
| `scripts/analyze_eval_results.py` | `make report` (→ `reports/`) / `make study NAME=core_report` (→ `docs/eval_results/analysis/`); cohorts in `configs/analysis/core_report.json` |
| `scripts/analyze_reference_dataset.py` | `make reference-report` / `make study NAME=reference` |
| `scripts/update_eval_results_tracker.py` | part of `make report` (`reports/metrics.csv`, `reports/README.md`); after each matrix run too |
| `scripts/summarize_reference_efficiency.py`, `summarize_human_ui_attribution.py`, `summarize_track_baseline.py` | `python -m analysis.reference_efficiency` / `analysis.human_ui_attribution` / `analysis.track_baseline` (run automatically after a matrix) |
| `scripts/summarize_comparison.py` | removed (compared two legacy experiments not on disk) |
| `scripts/analyze_form_length_position.py` | `make study NAME=form_length_position` |
| `scripts/analyze_formfactory_qwen3vl_comparison.py` | `make study NAME=formfactory_qwen3vl` |
| `scripts/analyze_opencua_ruler_comparison.py [--check]` | `make study NAME=opencua_ruler [CHECK=1]` |
| `scripts/analyze_localforms_comparison.py` | `make study NAME=localforms_comparison` |
| `scripts/export_fill_only_done_50_results.py` | `make study NAME=fill_only_done_50_export` |
| `scripts/export_missing_fill_only_additions.py [--check]` | `make study NAME=missing_fill_only_additions [CHECK=1]` |
| `docs/eval_results/interaction_failure_analysis/*.py` (4 commands) | `make study NAME=interaction_failure [CHECK=1]` |
| `scripts/inspect_step_inputs.py`, `scripts/list_baseline_media.py` | `make inspect TRIAL=… [STEPS=all] [MEDIA=1]` |
| — | new: `reports/experiment_overview.csv` + plot covering every experiment |

## Documentation

| Before | Now |
|---|---|
| `README_HPC.md`, `docs/HPC_RUNBOOK.md` | `docs/HPC.md` |
| `docs/BASELINE1_STANDARD_PLAN.md` | removed (pre-thesis plan; tag) |
| `docs/LOCALFORMS_{10FORM_PILOT_RESULTS,15FORM_CHECKPOINT,50FORM_FILLONLY_RESULTS}.md` | superseded by `docs/LOCALFORMS_ANALYSIS_FOR_THESIS.md` (tag) |
| `docs/eval_results/README.md` (auto-generated tracker), `metrics.csv/jsonl` | `docs/eval_results/README.md` is now the results index; the tracker is in `reports/` |

## Files outside the repository

The duplicated media copies next to the checkout (`interaction_media.zip`,
`interaction-media-on-main.bundle`, `interaction_media_bundle/`, `interaction_media_release/`,
~400 MB on disk) were deleted after verifying that every file is byte-identical to
`docs/eval_results/interaction_media/` and that the bundle's commit (`a85ab4c`) is on
`origin/main`. The optional GitHub-release upload is now `scripts/upload_media_release.sh`.
