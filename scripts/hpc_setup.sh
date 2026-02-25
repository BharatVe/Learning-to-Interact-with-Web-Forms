#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

export PLAYWRIGHT_BROWSERS_PATH="$ROOT_DIR/.playwright-browsers"
python -m playwright install chromium

python scripts/verify_runtime_setup.py || true

echo "Setup complete."
echo "Activate with: source $ROOT_DIR/.venv/bin/activate"
