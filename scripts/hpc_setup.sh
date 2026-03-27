#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m venv .venv

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
PIP_BIN="$ROOT_DIR/.venv/bin/pip"

"$PYTHON_BIN" -m pip install --upgrade pip
"$PIP_BIN" install -r requirements.txt

export PLAYWRIGHT_BROWSERS_PATH="$ROOT_DIR/.playwright-browsers"
"$PYTHON_BIN" -m playwright install chromium

"$PYTHON_BIN" scripts/verify_runtime_setup.py || true

echo "Setup complete."
echo "Activate with: source $ROOT_DIR/.venv/bin/activate"
