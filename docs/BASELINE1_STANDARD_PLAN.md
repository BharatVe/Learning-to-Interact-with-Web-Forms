# Baseline-1 Standard Plan (UI-based, Answers + Link Only)

> Legacy planning document. Thesis-primary workflow now separates:
> direct Playwright MCP tool use for Qwen `text_llm`/`vlm`, and native computer-use for `OpenCUA-32B`.
> See `README_HPC.md` and `configs/baselines/track_baseline_models.json`.

## Objective

Establish the first standardized baseline where each model starts with only:

- `form_url`
- `answers` (label/value pairs, optional widget hints)

No full form specification is provided at prompt start in this phase.

## Core Principle

MCP is used as infrastructure for tool access, tracing, and normalization.
MCP is not used as a model performance booster.

## Model Set (Minimal)

- Text model: `Qwen/Qwen2.5-3B-Instruct`
- Vision model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Computer-use reference: API-backed agent through MCP tool path

Model list file: `configs/baselines/minimal_models.json`

Swap-safe rule:

- Model selection is config-driven; changing model ids/repos in config must not break other entries.
- Installer should skip invalid/unavailable models and continue unless strict mode is explicitly enabled.

## Input/Output Contract

Input per step to model:

- Current observation (screenshot path or encoded image, and optional compact page text)
- Remaining unanswered fields
- Last action result
- Allowed action schema (strict JSON)

Output per step from model:

- Exactly one action in strict JSON:
  - `click`
  - `type`
  - `select_option`
  - `press_key`
  - `scroll`
  - `wait`
  - `submit`
  - `done`

## Execution Loop

1. Runner opens form URL in Playwright headless browser.
2. Capture observation.
3. Query model with current state.
4. Validate returned action against schema.
5. Execute action through Playwright/MCP.
6. Log trace event (`tool_trace.jsonl`) and state transition.
7. Stop on success, failure, step budget, or timeout.

## Standard Budgets

- Max steps per run: 60
- Max wall-clock per run: 180 seconds
- Action execution timeout: 15 seconds
- Retries for transient UI failure: 1

## Metrics

- Form submission success rate
- Question-level correctness
- Invalid action rate
- Timeout rate
- Action overhead vs scripted reference
- Runtime per run
- Cost per run (for paid APIs)

## Failure Taxonomy

- Field discovery failure
- Wrong widget interaction
- Option/value mismatch
- Date/time formatting mismatch
- Navigation/scroll state failure
- Submit not completed

## Fairness Rules

- Same action schema for all models
- Same budgets/timeouts
- Same observation frequency and limits
- No model-specific hidden hints
- Fixed seeds for repeated runs

## Integrity and Setup Checks

Run integrity checker before experiments:

```bash
python3 scripts/verify_baseline_integrity.py
python3 scripts/verify_runtime_setup.py
python3 scripts/preflight_baseline_eval.py
```

Pilot execution (gated):

```bash
bash scripts/start_baseline_pilot.sh
```

## Install Minimal Models

```bash
python3 scripts/install_minimal_models.py
```

If cluster node has no DNS/network, pre-download models on internet-enabled node and copy into `models/`.
