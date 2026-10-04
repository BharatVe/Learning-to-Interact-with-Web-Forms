# Evaluation results: index

Committed results are thesis records. They change only when you regenerate a study
on purpose (`make study NAME=<name>`, then review `git diff`). `make report` writes
*current* analytics for everything on disk to the git-ignored `reports/` and never
touches this folder. `evaluation_additions/manifest.json` stores a SHA-256 for every
thesis input, and `make test` verifies them.

| Result | Location | Regenerate | Notes |
|---|---|---|---|
| Core thesis report: Qwen text/VL direct-MCP and OpenCUA native + direct-MCP, 50 forms × runs 1–6 | `analysis/` (`thesis_model_summary.csv`, `efficiency_summary.csv`, `failure_summary.csv`, `plots/`, `latest_analysis.md`) | `make study NAME=core_report` | Cohorts are defined in `configs/analysis/core_report.json`. The committed copy predates later experiments, and regenerating it also refreshes the inputs of `interaction_failure` |
| Ideal (scripted) reference runs: coverage, action breakdown, durations | `reference_analysis/` | `make study NAME=reference` | Source: `data/forms/**/tool_trace.jsonl` |
| Qwen3-VL interface study: direct MCP vs FormFactory-style, submit and fill-only | `FORMFACTORY_QWEN3VL_COMPARISON.md` (design), `FORMFACTORY_QWEN3VL_RESULTS.md`, `../../data/model_baseline_exports/formfactory_qwen3vl_*` | `make study NAME=formfactory_qwen3vl` | Config: `configs/analysis/formfactory_qwen3vl.json`; experiments: `make experiments` → `qwen3vl_interface_comparison` |
| Accuracy by form length and question position | `form_length_position_analysis/` | `make study NAME=form_length_position` | Dropdowns excluded |
| Interaction failures, incl. the dropdown audit | `interaction_failure_analysis/` (`REPORT_SUMMARY.md`, `DROPDOWN_FAILURE_ANALYSIS.md`, `report.html`) | `make study NAME=interaction_failure` (`CHECK=1` validates) | Reads `fill_only_done_50` export + `analysis/` + `reference_analysis/` |
| 50-form fill-only/DONE export: Gemini, Qwen text/VL, OpenCUA MCP | `../../data/model_baseline_exports/fill_only_done_50_20260714/` | `make study NAME=fill_only_done_50_export` | Experiments: `fill_only_done_10/30/50_completion` |
| Late fill-only additions | `../../evaluation_additions/missing_fill_only_runs/` | `make study NAME=missing_fill_only_additions` (`CHECK=1`) | |
| OpenCUA pixel-ruler comparison | `../../evaluation_additions/opencua_ruler_comparison/` (`report.html`) | `make study NAME=opencua_ruler` (`CHECK=1`) | Experiments: `formfactory_opencua` |
| LocalForms vs Google Forms platform comparison | `../LOCALFORMS_ANALYSIS_FOR_THESIS.md`, `../../data/localforms_comparison_analysis/` | `make study NAME=localforms_comparison` | Method: `../LOCALFORMS_METHODOLOGY.md`; experiments: `localforms_opencua_direct_mcp`, `opencua_direct_mcp_fill_only_128` |
| Curated interaction videos and screenshots | `interaction_media/`, `INTERACTION_MEDIA.md` | n/a | |
| Gemini rerun decision (cost/benefit of re-running the 50-form Gemini evaluation, 2026-07-18) | `decisions/gemini_rerun_decision_20260718/report.html` | n/a (decision record) | Moved from the outer workspace repo |
| Run settings per thesis experiment | `../../evaluation_additions/run_settings.json` | n/a (hand-curated) | See caveats |

For any experiment on disk, `make report` gives the per-experiment overview
(`reports/experiment_overview.csv`, `reports/plots/experiment_overview_accuracy.svg`)
and the per-trial tracker (`reports/metrics.csv`, `reports/README.md`).

## Caveats

- **Dropdown scores.** On Google Forms the verifier read every option label of the custom
  dropdown instead of the selected one. In the audited 50-form fill-only cohorts every
  dropdown target got 0 credit, and other experiments used the same verifier. The stored-artifact audit (`interaction_failure_analysis/DROPDOWN_FAILURE_ANALYSIS.md`)
  confirms 79 of 100 audited selections as correct, none as wrong, and 21 as
  unresolved. Tables computed directly from trial summaries, including `analysis/`, the
  `reports/` overview and the tracker, still contain the raw dropdown scores. The
  form-length and FormFactory studies exclude dropdowns.
- **`run_settings.json` trial timeouts.** For `opencua_direct_mcp_fill_only_done_30_seed20260709_r2_step32`
  and `opencua_direct_mcp_fill_only_done_50_topup20_20260713_r2_step32` the file records
  `trial_s: 1800`, but the Slurm logs of those jobs show `direct_mcp_timeout_s=9000`
  (jobs 2292321 and 2296100). The manifests in `configs/experiments/` use 9000. With the
  32-step cap the timeout is unlikely to have been binding.
- Superseded LocalForms snapshots (10-form pilot, 15-form checkpoint, 50-form fill-only
  update) and the old baseline plan were removed from the working tree; they are in git tag
  `pre-cleanup-thesis-results`.
