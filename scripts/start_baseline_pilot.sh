#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate

echo "[INFO] Running baseline preflight gate..."
python3 scripts/preflight_baseline_eval.py "$@"

echo "[INFO] Preflight passed. Starting baseline pilot smoke run..."
PILOT_TIMEOUT_S="${PILOT_TIMEOUT_S:-900}"
set +e
timeout "${PILOT_TIMEOUT_S}"s bash scripts/run_baselines_headless.sh \
  --smoke-test-all-forms \
  --overwrite-existing
RC=$?
set -e

if [ "$RC" -eq 124 ]; then
  echo "[FAIL] Pilot run timed out after ${PILOT_TIMEOUT_S}s."
  exit 124
fi
if [ "$RC" -ne 0 ]; then
  echo "[FAIL] Pilot run failed with exit code ${RC}."
  exit "$RC"
fi

echo "[DONE] Pilot baseline run completed."
