#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate
export PLAYWRIGHT_BROWSERS_PATH="$ROOT_DIR/.playwright-browsers"
export PYTHONUNBUFFERED=1

python3 src/engine/runner.py \
  --dataset-root data/forms \
  --trace-mode mcp \
  --interaction-mode local \
  --headless \
  "$@"
