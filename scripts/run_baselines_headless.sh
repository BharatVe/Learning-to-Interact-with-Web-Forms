#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[FAIL] missing virtualenv interpreter: $PYTHON_BIN" >&2
  exit 1
fi
export PLAYWRIGHT_BROWSERS_PATH="$ROOT_DIR/.playwright-browsers"
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" src/engine/runner.py \
  --dataset-root data/forms \
  --trace-mode mcp \
  --interaction-mode local \
  --headless \
  "$@"
