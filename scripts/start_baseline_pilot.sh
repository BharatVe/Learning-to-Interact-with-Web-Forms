#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[FAIL] missing virtualenv interpreter: $PYTHON_BIN" >&2
  exit 1
fi

echo "[INFO] Running baseline preflight gate..."
"$PYTHON_BIN" scripts/preflight_baseline_eval.py "$@"

PILOT_MODE="${PILOT_MODE:-recording}"
echo "[INFO] Preflight passed. Starting pilot mode: ${PILOT_MODE}"
PILOT_TIMEOUT_S="${PILOT_TIMEOUT_S:-900}"
set +e
if [ "$PILOT_MODE" = "recording" ]; then
  timeout "${PILOT_TIMEOUT_S}"s bash scripts/run_baselines_headless.sh \
    --smoke-test-all-forms \
    --overwrite-existing \
    "$@"
elif [ "$PILOT_MODE" = "baseline_matrix" ]; then
  timeout "${PILOT_TIMEOUT_S}"s bash scripts/run_model_baseline_matrix.sh "$@"
else
  echo "[FAIL] unsupported PILOT_MODE=$PILOT_MODE (expected recording|baseline_matrix)" >&2
  exit 1
fi
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

echo "[DONE] Pilot run completed."
