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
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$ROOT_DIR/.playwright-browsers-node}"
export PYTHONUNBUFFERED=1

CONFIG_PATH="${CONFIG_PATH:-configs/baselines/track_baseline_models.json}"
FORM_IDS="${FORM_IDS:-conf_interest,event_rsvp}"
RUN_INDEX="${RUN_INDEX:-1}"
RUN_INDEXES="${RUN_INDEXES:-$RUN_INDEX}"
DIRECT_EXPERIMENT_ID="${DIRECT_EXPERIMENT_ID:-track_baseline_gemini_v1}"
DIRECT_PROVIDER="${DIRECT_PROVIDER:-gemini_native}"
DIRECT_TRACK="${DIRECT_TRACK:-direct_api_tool_use}"
DIRECT_MODEL_ID="${DIRECT_MODEL_ID:-}"
DIRECT_MAX_STEPS="${DIRECT_MAX_STEPS:-48}"
DIRECT_TIMEOUT_S="${DIRECT_TIMEOUT_S:-3600}"
DIRECT_MAX_NEW_TOKENS="${DIRECT_MAX_NEW_TOKENS:-256}"
DIRECT_API_TIMEOUT_S="${DIRECT_API_TIMEOUT_S:-180}"
DIRECT_BROWSER_MCP_TIMEOUT_MS="${DIRECT_BROWSER_MCP_TIMEOUT_MS:-600000}"
GEMINI_MIN_REQUEST_INTERVAL_S="${GEMINI_MIN_REQUEST_INTERVAL_S:-8.0}"
GEMINI_MAX_INFER_RETRIES="${GEMINI_MAX_INFER_RETRIES:-2}"
GEMINI_MAX_RETRY_DELAY_S="${GEMINI_MAX_RETRY_DELAY_S:-75}"
INTERACTION_PROTOCOL="${INTERACTION_PROTOCOL:-human_ui_v1}"
OBSERVATION_MODE="${OBSERVATION_MODE:-vision_coords}"
SCORING_MODE="${SCORING_MODE:-soft_quality_v1}"
RETENTION_WINDOW="${RETENTION_WINDOW:-5}"
FAIL_ON_TRIAL_FAILURE="${FAIL_ON_TRIAL_FAILURE:-0}"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "[FAIL] config_path does not exist: $CONFIG_PATH" >&2
  exit 1
fi

if [ "$DIRECT_PROVIDER" != "gemini_native" ]; then
  echo "[FAIL] scripts/run_gemini_direct_matrix.sh only supports DIRECT_PROVIDER=gemini_native" >&2
  exit 1
fi

if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "[FAIL] GEMINI_API_KEY is required for gemini_native direct baseline" >&2
  exit 1
fi

if ! command -v playwright-mcp >/dev/null 2>&1; then
  echo "[FAIL] playwright-mcp not found on PATH" >&2
  exit 1
fi

if [ "$FORM_IDS" = "all" ]; then
  RESOLVED_FORM_IDS="$($PYTHON_BIN - <<'PY_FORMS'
from pathlib import Path
root = Path('src/forms')
form_ids = sorted(entry.name for entry in root.iterdir() if entry.is_dir() and (entry / 'spec.json').exists())
print(','.join(form_ids))
PY_FORMS
)"
else
  RESOLVED_FORM_IDS="$FORM_IDS"
fi

if [ -z "$RESOLVED_FORM_IDS" ]; then
  echo "[FAIL] no forms resolved (FORM_IDS=$FORM_IDS)" >&2
  exit 1
fi

IFS=',' read -r -a RUN_INDEX_LIST <<<"$RUN_INDEXES"
if [ "${#RUN_INDEX_LIST[@]}" -eq 0 ]; then
  echo "[FAIL] RUN_INDEXES resolved to empty list" >&2
  exit 1
fi

if [ -z "$DIRECT_MODEL_ID" ]; then
  DIRECT_MODEL_ID="$($PYTHON_BIN - "$CONFIG_PATH" "$DIRECT_TRACK" <<'PY_MODEL'
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
track = str(sys.argv[2] or '').strip()
for model in cfg.get('models', []):
    if not isinstance(model, dict):
        continue
    if str(model.get('track') or '') != track:
        continue
    if str(model.get('provider') or '') != 'gemini_native':
        continue
    model_id = str(model.get('id') or '').strip()
    if model_id:
        print(model_id)
        break
PY_MODEL
)"
fi

if [ -z "$DIRECT_MODEL_ID" ]; then
  echo "[FAIL] no gemini_native direct model found in $CONFIG_PATH" >&2
  exit 1
fi

if ! "$PYTHON_BIN" scripts/eval_model_baseline_smoke.py \
  --config "$CONFIG_PATH" \
  --include-kinds computer_use_agent \
  --exclude-providers openai_compat,local_hf,api_over_mcp \
  --strict; then
  echo "[FAIL] Gemini direct preflight failed" >&2
  exit 1
fi

make_trial_id() {
  "$PYTHON_BIN" - <<'PY_TRIAL'
from datetime import datetime
print('trial_' + datetime.utcnow().strftime('%Y%m%dT%H%M%S%fZ'))
PY_TRIAL
}

make_run_label() {
  "$PYTHON_BIN" - <<'PY_LABEL'
import os
import re
from datetime import datetime
job_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(os.environ.get('SLURM_JOB_ID') or 'na').strip()) or 'na'
print(datetime.utcnow().strftime('%Y%m%dT%H%M%SZ') + f'_job{job_id}')
PY_LABEL
}

direct_total=0
direct_passed=0
direct_failed=0

echo "[INFO] config_path=$CONFIG_PATH"
echo "[INFO] forms=$RESOLVED_FORM_IDS"
echo "[INFO] run_indexes=$RUN_INDEXES"
echo "[INFO] direct_experiment_id=$DIRECT_EXPERIMENT_ID"
echo "[INFO] direct_model_id=$DIRECT_MODEL_ID"
echo "[INFO] direct_provider=$DIRECT_PROVIDER"
echo "[INFO] interaction_protocol=$INTERACTION_PROTOCOL observation_mode=$OBSERVATION_MODE scoring_mode=$SCORING_MODE"
echo "[INFO] gemini_limits min_request_interval_s=$GEMINI_MIN_REQUEST_INTERVAL_S max_infer_retries=$GEMINI_MAX_INFER_RETRIES max_retry_delay_s=$GEMINI_MAX_RETRY_DELAY_S max_steps=$DIRECT_MAX_STEPS max_new_tokens=$DIRECT_MAX_NEW_TOKENS"

IFS=',' read -r -a FORMS <<<"$RESOLVED_FORM_IDS"
for form_id in "${FORMS[@]}"; do
  for run_idx in "${RUN_INDEX_LIST[@]}"; do
    trial_id="$(make_trial_id)"
    run_label="$(make_run_label)"
    direct_total=$((direct_total + 1))
    echo "[INFO] direct_eval model_id=${DIRECT_MODEL_ID} form_id=${form_id} run_index=${run_idx} trial_id=${trial_id} provider=${DIRECT_PROVIDER}"

    if GEMINI_MIN_REQUEST_INTERVAL_S="$GEMINI_MIN_REQUEST_INTERVAL_S" \
      GEMINI_MAX_INFER_RETRIES="$GEMINI_MAX_INFER_RETRIES" \
      GEMINI_MAX_RETRY_DELAY_S="$GEMINI_MAX_RETRY_DELAY_S" \
      "$PYTHON_BIN" src/baselines/run_gemini_native_computer_use_eval.py \
      --config "$CONFIG_PATH" \
      --model-id "$DIRECT_MODEL_ID" \
      --form-id "$form_id" \
      --run-index "$run_idx" \
      --trial-id "$trial_id" \
      --experiment-id "$DIRECT_EXPERIMENT_ID" \
      --api-timeout-s "$DIRECT_API_TIMEOUT_S" \
      --execution-backend mcp_server \
      --headless \
      --max-new-tokens "$DIRECT_MAX_NEW_TOKENS" \
      --max-steps "$DIRECT_MAX_STEPS" \
      --timeout-s "$DIRECT_TIMEOUT_S" \
      --browser-mcp-timeout-ms "$DIRECT_BROWSER_MCP_TIMEOUT_MS" \
      --interaction-protocol "$INTERACTION_PROTOCOL" \
      --observation-mode "$OBSERVATION_MODE" \
      --scoring-mode "$SCORING_MODE" \
      --retention-window "$RETENTION_WINDOW" \
      --run-label "$run_label" \
      --disable-action-coercion; then
      direct_passed=$((direct_passed + 1))
    else
      status=$?
      echo "[WARN] direct eval failed for form_id=${form_id} run_index=${run_idx} status=${status}" >&2
      direct_failed=$((direct_failed + 1))
      if [ "$status" -eq 2 ]; then
        echo "[FAIL] Gemini quota exhausted; aborting remaining direct trials." >&2
        exit 2
      fi
    fi
  done
done

echo "[INFO] direct_eval_total=$direct_total"
echo "[INFO] direct_eval_passed=$direct_passed"
echo "[INFO] direct_eval_failed=$direct_failed"

if [ "$direct_failed" -gt 0 ] && [ "$FAIL_ON_TRIAL_FAILURE" = "1" ]; then
  exit 1
fi
