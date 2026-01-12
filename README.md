# Learning-to-Interact-with-Web-Forms

## Overview

This project automates Google Form submissions with Playwright to generate a dataset of
videos and annotations. The engine runs multiple submissions per form, records each run,
and writes a per-run `annotations.json` with timing metadata for each filled question.

Key points:
- Reusable engine in `src/engine/` handles run orchestration and output structure.
- Form-specific configuration lives in `src/forms/<form_id>/spec.json`.
- Form-specific entrypoints (e.g. `src/conf_intrest/fill_google_form.py`) are thin wrappers.
- Outputs are stored under `data/forms/<form_id>/runs/run_XXXX/`.

## Dataset generation

Example command for the Conference Interest form:

```bash
python3 src/engine/runner.py --form-id conf_interest --answers src/conf_intrest/answers_conference.json --dataset-root data/forms --num-runs 3
```

You can also run the form-specific entrypoint:

```bash
python3 src/conf_intrest/fill_google_form.py --num-runs 3
```

## Inputs

The engine accepts two formats:

- JSON: either a single run (a list of answer entries) or a multi-run object with `runs`.
- JSONL: one run per line, each line is a JSON object describing a run.

Each answer entry must contain:

- `label` (question label text to match)
- `widget_type` (short_text, paragraph_text, single_choice, multi_choice, date, time)
- `value` (string or list, depending on widget type)

Optional run metadata (e.g. `run_name`, `seed`, `notes`) can be included and will be
carried into the annotations.

## Outputs

Each run generates:

- `data/forms/<form_id>/runs/run_XXXX/<form_id>_run_XXXX.webm`
- `data/forms/<form_id>/runs/run_XXXX/annotations.json`
- `data/forms/<form_id>/runs/run_XXXX/answers_instance.json`

`annotations.json` includes form/run identifiers, the final video path, run parameters,
per-question action timing, and submit timing.

## Run controls

Useful flags:

- `--num-runs` limit how many runs to generate in one execution.
- `--start-index` force the starting run index.
- `--resume` continue from the next missing run index.
- `--skip-existing` skip runs whose output directory already exists.
- `--form-url` override the URL from the spec file.
