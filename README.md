# Learning-to-Interact-with-Web-Forms

## Overview

This project generates a dataset of Google Form interactions using direct Playwright Python for browser control.
Tool traces are validated and normalized through an MCP server by default, then written to `tool_trace.jsonl`.

## Install

```bash
pip install -r requirements.txt
python -m playwright install chromium
# Linux only (if needed):
python -m playwright install --with-deps chromium
```

HPC-friendly setup wrapper (no `sudo`, project-local venv/browser cache):

```bash
bash scripts/hpc_setup.sh
```

Runtime readiness check:

```bash
python3 scripts/verify_runtime_setup.py
python3 scripts/preflight_baseline_eval.py
```

Headless baseline wrapper:

```bash
bash scripts/run_baselines_headless.sh --smoke-test-all-forms --overwrite-existing
```

For cluster workflow (canonical directory policy, model install, Slurm/headless usage), see:
`README_HPC.md`

If `PLAYWRIGHT_SKIP_FFMPEG_INSTALL` is set, video recording may fail.
`mcp_server` interaction mode also requires `node` + `npx` to run the official Playwright MCP server.
The default MCP command now forces `--browser chromium` to avoid system Chrome dependency.
Default MCP launch no longer enables `--caps=vision`, which improves compatibility in WSL/restricted environments.
When Python Playwright is installed, runner automatically passes its Chromium executable path to MCP (`--executable-path`) to avoid Node browser mismatch.
In WSL, runner also adds `--no-sandbox` for MCP browser launch.
If `@playwright/mcp` is not already cached, the first `mcp_server` run may need network access for `npx` package resolution.
Runner now auto-installs Node Playwright `chromium` for `mcp_server` mode unless `--no-mcp-browser-install` is provided.
Runner also performs an MCP package preflight (`npx @playwright/mcp --version`) before run start in `mcp_server` mode.
For more stable startup in WSL/offline scenarios, install MCP globally once: `npm i -g @playwright/mcp`.

## Dataset Generation (Single Form)

```bash
python3 src/engine/runner.py \
  --form-id conf_interest \
  --dataset-root data/forms \
  --num-runs 1
```

Answers are matched automatically from:
`data/answers/<form_id>/runs.json`

Full dataset run (all forms under `src/forms`, each auto-matched to `data/answers/<form_id>/runs.json`):

```bash
python3 src/engine/runner.py \
  --all-forms \
  --dataset-root data/forms \
  --skip-existing-video
```

Smoke test across all forms (runs exactly one answer instance per form, prints pass/fail summary):

```bash
python3 src/engine/runner.py \
  --smoke-test-all-forms \
  --dataset-root data/forms \
  --overwrite-existing
```

Smoke test using full official Playwright MCP browser execution:

```bash
python3 src/engine/runner.py \
  --smoke-test-all-forms \
  --dataset-root data/forms \
  --overwrite-existing \
  --interaction-mode mcp_server
```

By default the browser runs **headed** (visible). Use `--headless` to disable UI.
Mouse overlay is enabled by default for video clarity. Use `--no-mouse-overlay` to disable it.
Screenshots are optional and disabled by default. Use `--screenshots` to save `observations/*.png`.
Trace mode defaults to `mcp` and auto-starts the bundled server `src/engine/mcp_trace_server.py`.
Interaction mode defaults to `local`.
Use `--interaction-mode mcp_server` to execute browser interaction via the official Playwright MCP server.

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
`tool_trace.jsonl` is JSONL with Playwright MCP-style action names.
In `local` mode it records low-level interaction actions (`browser_mouse_click_xy`, `browser_mouse_move_xy`, `browser_type`, `browser_press_key`, `browser_mouse_wheel`).
In `mcp_server` mode it records official MCP tool calls (`browser_navigate`, `browser_run_code`, `browser_wait_for`, `browser_take_screenshot`, `browser_close`).
Each action now also includes required-field metadata when detectable: `required`, `required_attr`, `required_marker`.

## Run Controls

Useful flags:

- `--num-runs` limit how many runs to generate in one execution.
- `--start-index` force the starting run index.
- `--resume` continue from the next missing run index.
- `--skip-existing` skip runs whose output directory already exists.
- `--skip-existing-video` skip runs whose output directory already contains a `.webm`.
- `--overwrite-existing` delete an existing run directory and regenerate it.
- `--all-forms` run all form specs in `src/forms`.
- `--smoke-test-all-forms` run one test run per form and continue through failures, then print summary.
- `--form-url` override the URL from the spec file.
- `--answers-root` base directory for automatic answer matching (default: `data/answers`).
- `--answers-file` primary filename to look for in each form answer directory (default: `runs.json`).
- Fallback if missing: `runs.jsonl`, `runs.ndjson` (fails with explicit error if none exist).
- `--headless` run without visible browser UI.
- `--slow-mo` add a delay (ms) to Playwright actions.
- `--type-delay-ms` delay (ms) between typed characters.
- `--action-delay-ms` delay (ms) after each action for visibility.
- `--viewport-width` / `--viewport-height` set the browser viewport.
- `--timeout-ms` set Playwright timeout for waits.
- `--screenshots` enable per-step and submit screenshots.
- `--no-mouse-overlay` disable the visible mouse overlay.
- `--interaction-mode` choose browser action backend: `local` (default) or `mcp_server`.
- `--trace-mode` choose trace backend: `mcp` (default) or `local`.
- `--mcp-server-cmd` override MCP server command (defaults to bundled trace server).
- `--mcp-tool-name` MCP tool used for event normalization (default: `record_action`).
- `--mcp-timeout-ms` MCP request timeout (default: `5000`).
- `--browser-mcp-cmd` override official Playwright MCP browser command used in `mcp_server` mode.
- `--browser-mcp-timeout-ms` timeout for browser MCP tool calls (default: `120000`).
- `--no-mcp-browser-install` disable automatic Node Playwright browser install preflight.
- `--mcp-browser-install-timeout-s` timeout for browser preflight install (default: `600`).
- `--no-mcp-verify-trace` disable MCP action-schema validation for trace events.
- `--no-mcp-strict` keep validation on but do not fail the run on validation errors.

Overwrite existing run:

```bash
python3 src/engine/runner.py \
  --form-id conf_interest \
  --dataset-root data/forms \
  --num-runs 1 \
  --start-index 1 \
  --overwrite-existing
```
