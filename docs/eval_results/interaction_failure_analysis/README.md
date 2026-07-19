# Interaction failure analysis

This package analyzes 200 canonical fill-only trials (50 forms × four
model/interface conditions), 978 historical baseline trials, and 300 scripted
ideal-reference runs.

## Start here

- `report.html` — self-contained, paper-oriented report.
- `REPORT_SUMMARY.md` — concise findings and comparison tables.
- `METHODS_AND_LIMITATIONS.md` — metric definitions, corrections, and caveats.

## Reproduce and validate

```bash
python docs/eval_results/interaction_failure_analysis/analyze_action_failures.py \
  --project-root . \
  --trials-csv data/model_baseline_exports/fill_only_done_50_20260714/trials.csv \
  --baseline-actions-csv docs/eval_results/analysis/model_action_trial_counts.csv \
  --reference-runs-csv docs/eval_results/reference_analysis/reference_runs.csv \
  --reference-actions-csv docs/eval_results/reference_analysis/reference_action_breakdown.csv \
  --output-dir docs/eval_results/interaction_failure_analysis/data
python docs/eval_results/interaction_failure_analysis/build_report_artifact.py
python docs/eval_results/interaction_failure_analysis/validate_outputs.py
```

`analyze_action_failures.py` reads the canonical trial export, historical
action-trial index, and scripted reference exports. `build_report_artifact.py`
creates `artifact.json`; `validate_outputs.py` reconciles grains, totals,
dropdown classifications, submissions, and report links.

## Stable comparison files

- Failure: `failure_matrix_by_model.csv`, `trial_stop_reasons_by_model.csv`,
  `hardest_forms_by_model.csv`, `question_difficulty_cross_model.csv`, and
  `widget_difficulty_by_model.csv`.
- Actions: `action_mix_by_model_outcome.csv`,
  `action_efficiency_by_model_outcome.csv`, and
  `action_count_correction_by_model.csv`.
- Dropdown split: `performance_forms_without_dropdown.csv`,
  `performance_forms_with_dropdown.csv`, and
  `dropdown_verifier_audit_by_model.csv`.
- Submission audit: `canonical_submission_audit.csv`,
  `baseline_submission_scoring_audit.csv`, and
  `baseline_recovered_perfect_submitted_fields.csv`.
- Ideal comparison: `ideal_reference_runs.csv`, `model_vs_ideal_by_form.csv`,
  and `model_vs_ideal_summary.csv`.

All paths above are under `data/`.

## Definitions that matter

- An interaction action changes or navigates task state. Screenshots,
  snapshots, waits, `DONE`, setup, and close are excluded.
- Exact full-fill claims use only the 25 forms without dropdowns. Dropdown
  outcomes remain unresolved because the verifier often saved the full option
  list instead of the selected value.
- Historical submitted trials use the last meaningful pre-submit score because
  confirmation pages contain no form controls.
- Action counts are interpreted with completion and elapsed time because one
  call can represent different amounts of work across interfaces.

The canonical comparison is conditional on usable trials. Operational failures
should be reported separately from capability.
