# Learning-to-Interact-with-Web-Forms

## Overview

This project generates a dataset of Google Form interactions using **direct Playwright Python** (no MCP). Each run fills a form, records a video, saves step screenshots, and writes annotations + a tool trace.

## Install

```bash
pip install -r requirements.txt
python -m playwright install chromium
# Linux only (if needed):
python -m playwright install --with-deps chromium
```

If `PLAYWRIGHT_SKIP_FFMPEG_INSTALL` is set, video recording may fail.

## Dataset Generation (Single Form)

```bash
python3 src/engine/runner.py \
  --form-id conf_interest \
  --answers data/answers/conf_interest/runs.json \
  --dataset-root data/forms \
  --num-runs 1
```

By default the browser runs **headed** (visible). Use `--headless` to disable UI.

## Inputs

The engine accepts two formats:

- JSON: either a single run (a list of answer entries) or a multi-run object with `runs`.
- JSONL: one run per line, each line is a JSON object describing a run.

Each answer entry must contain:

- `label` (question label text to match)
- `widget_type` (short_text, paragraph_text, single_choice, multi_choice, date, time)
- `value` (string or list, depending on widget type)

Optional run metadata can be included and is carried into `annotations.json`.

## Outputs

Each run generates:

- `data/forms/<form_id>/runs/run_XXXX/<form_id>_run_XXXX.webm`
- `data/forms/<form_id>/runs/run_XXXX/annotations.json`
- `data/forms/<form_id>/runs/run_XXXX/answers_instance.json`
- `data/forms/<form_id>/runs/run_XXXX/tool_trace.jsonl`
- `data/forms/<form_id>/runs/run_XXXX/observations/step_XXXX_pre.png`
- `data/forms/<form_id>/runs/run_XXXX/observations/step_XXXX_post.png`
- `data/forms/<form_id>/runs/run_XXXX/observations/submit_pre.png`
- `data/forms/<form_id>/runs/run_XXXX/observations/submit_post.png`

`annotations.json` includes form/run identifiers, video path, run parameters, macro actions, submit timing, and trace pointers.
`tool_trace.jsonl` is JSONL with Gemini-style micro-actions (click_at, hover_at, type_text_at, key_combination, scroll_document).

## Run Controls

Useful flags:

- `--num-runs` limit how many runs to generate in one execution.
- `--start-index` force the starting run index.
- `--resume` continue from the next missing run index.
- `--skip-existing` skip runs whose output directory already exists.
- `--skip-existing-video` skip runs whose output directory already contains a `.webm`.
- `--overwrite-existing` delete an existing run directory and regenerate it.
- `--form-url` override the URL from the spec file.
- `--headless` run without visible browser UI.
- `--slow-mo` add a delay (ms) to Playwright actions.
- `--viewport-width` / `--viewport-height` set the browser viewport.
- `--timeout-ms` set Playwright timeout for waits.

Overwrite existing run:

```bash
python3 src/engine/runner.py \
  --form-id conf_interest \
  --answers data/answers/conf_interest/runs.json \
  --dataset-root data/forms \
  --num-runs 1 \
  --start-index 1 \
  --overwrite-existing
```
