#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[FAIL] missing virtualenv interpreter: $PYTHON_BIN" >&2
  exit 1
fi

module load release/25.06 GCCcore/13.3.0 nodejs/20.13.1

export PATH="$ROOT_DIR/.node-tools/node_modules/.bin:$PATH"
export PLAYWRIGHT_BROWSERS_PATH="$ROOT_DIR/.playwright-browsers-node"
export PYTHONUNBUFFERED=1

if ! command -v playwright-mcp >/dev/null 2>&1; then
  echo "[FAIL] playwright-mcp not found on PATH. Install it with:" >&2
  echo "       module load release/25.06 GCCcore/13.3.0 nodejs/20.13.1" >&2
  echo "       npm install --prefix .node-tools @playwright/mcp@latest" >&2
  exit 1
fi

"$PYTHON_BIN" src/engine/runner.py \
  --dataset-root data/forms \
  --trace-mode mcp \
  --interaction-mode mcp_server \
  --headless \
  --no-mcp-browser-install \
  "$@"
