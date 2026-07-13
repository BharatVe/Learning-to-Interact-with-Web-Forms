#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[FAIL] missing virtualenv interpreter: $PYTHON_BIN" >&2
  exit 1
fi

ORIGINAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
module load release/25.06 GCCcore/13.3.0 nodejs/20.13.1
export NODE_LD_LIBRARY_PATH_FOR_MCP="${LD_LIBRARY_PATH-}"
if [ -n "$ORIGINAL_LD_LIBRARY_PATH" ]; then
  export LD_LIBRARY_PATH="$ORIGINAL_LD_LIBRARY_PATH"
else
  unset LD_LIBRARY_PATH
fi

export PATH="$ROOT_DIR/.node-tools/node_modules/.bin:$PATH"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$ROOT_DIR/.playwright-browsers-node}"
export PYTHONUNBUFFERED=1

CONFIG_PATH="${CONFIG_PATH:-configs/baselines/track_baseline_models.json}"
DIRECT_MODEL_ID="${DIRECT_MODEL_ID:-computer_use_gemini_35_flash_lowcost}"
DIRECT_EXPERIMENT_ID="${DIRECT_EXPERIMENT_ID:-gemini_35_flash_lowcost_token_pilot_v1}"
FORM_IDS="${FORM_IDS:-bug_report}"
RUN_INDEX="${RUN_INDEX:-2}"
RUN_INDEXES="${RUN_INDEXES:-$RUN_INDEX}"
DIRECT_MAX_STEPS="${DIRECT_MAX_STEPS:-48}"
DIRECT_TIMEOUT_S="${DIRECT_TIMEOUT_S:-3600}"
DIRECT_MAX_NEW_TOKENS="${DIRECT_MAX_NEW_TOKENS:-128}"
DIRECT_API_TIMEOUT_S="${DIRECT_API_TIMEOUT_S:-300}"
DIRECT_BROWSER_MCP_TIMEOUT_MS="${DIRECT_BROWSER_MCP_TIMEOUT_MS:-600000}"
GEMINI_MAX_INFER_RETRIES="${GEMINI_MAX_INFER_RETRIES:-4}"
GEMINI_RETRY_DELAY_S="${GEMINI_RETRY_DELAY_S:-30}"
GEMINI_RETRY_BACKOFF="${GEMINI_RETRY_BACKOFF:-2}"
GEMINI_RETRY_MAX_DELAY_S="${GEMINI_RETRY_MAX_DELAY_S:-240}"
INTERACTION_PROTOCOL="${INTERACTION_PROTOCOL:-human_ui_v1}"
OBSERVATION_MODE="${OBSERVATION_MODE:-vision_coords}"
SCORING_MODE="${SCORING_MODE:-soft_quality_v1}"
RETENTION_WINDOW="${RETENTION_WINDOW:-5}"
FAIL_ON_TRIAL_FAILURE="${FAIL_ON_TRIAL_FAILURE:-0}"
INCLUDE_CONTROLS="${INCLUDE_CONTROLS:-0}"
FILL_ONLY="${FILL_ONLY:-0}"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "[FAIL] config_path does not exist: $CONFIG_PATH" >&2
  exit 1
fi

if [ -z "${GEMINI_API_KEY:-}" ]; then
  key_file="${GEMINI_API_KEY_FILE:-$ROOT_DIR/.secrets/gemini_api_key}"
  if [ ! -s "$key_file" ]; then
    echo "[FAIL] GEMINI_API_KEY is unset and Gemini key file is missing/empty: $key_file" >&2
    echo "[INFO] create it with: mkdir -p .secrets && chmod 700 .secrets && printf '%s' '<key>' > .secrets/gemini_api_key && chmod 600 .secrets/gemini_api_key" >&2
    exit 1
  fi
fi

if ! command -v playwright-mcp >/dev/null 2>&1; then
  echo "[FAIL] playwright-mcp not found on PATH" >&2
  exit 1
fi

if ! "$PYTHON_BIN" scripts/eval_model_baseline_smoke.py \
  --config "$CONFIG_PATH" \
  --include-kinds computer_use_agent \
  --exclude-providers openai_compat,local_hf,api_over_mcp,gemini_native \
  --strict; then
  echo "[FAIL] Gemini low-cost preflight failed" >&2
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

echo "[INFO] config_path=$CONFIG_PATH"
echo "[INFO] forms=$FORM_IDS"
echo "[INFO] run_indexes=$RUN_INDEXES"
echo "[INFO] direct_experiment_id=$DIRECT_EXPERIMENT_ID"
echo "[INFO] direct_model_id=$DIRECT_MODEL_ID"
echo "[INFO] max_steps=$DIRECT_MAX_STEPS max_new_tokens=$DIRECT_MAX_NEW_TOKENS include_controls=$INCLUDE_CONTROLS fill_only=$FILL_ONLY"
echo "[INFO] gemini_retries max=$GEMINI_MAX_INFER_RETRIES delay_s=$GEMINI_RETRY_DELAY_S backoff=$GEMINI_RETRY_BACKOFF max_delay_s=$GEMINI_RETRY_MAX_DELAY_S api_timeout_s=$DIRECT_API_TIMEOUT_S"

direct_total=0
direct_passed=0
direct_failed=0
IFS=',' read -r -a FORMS <<<"$FORM_IDS"
IFS=',' read -r -a RUN_INDEX_LIST <<<"$RUN_INDEXES"

for form_id in "${FORMS[@]}"; do
  for run_idx in "${RUN_INDEX_LIST[@]}"; do
    trial_id="$(make_trial_id)"
    run_label="$(make_run_label)"
    direct_total=$((direct_total + 1))
    echo "[INFO] gemini_low_cost_eval model_id=${DIRECT_MODEL_ID} form_id=${form_id} run_index=${run_idx} trial_id=${trial_id}"

    args=(
      --config "$CONFIG_PATH"
      --model-id "$DIRECT_MODEL_ID"
      --form-id "$form_id"
      --run-index "$run_idx"
      --trial-id "$trial_id"
      --experiment-id "$DIRECT_EXPERIMENT_ID"
      --api-timeout-s "$DIRECT_API_TIMEOUT_S"
      --execution-backend mcp_server
      --headless
      --max-new-tokens "$DIRECT_MAX_NEW_TOKENS"
      --max-steps "$DIRECT_MAX_STEPS"
      --timeout-s "$DIRECT_TIMEOUT_S"
      --browser-mcp-timeout-ms "$DIRECT_BROWSER_MCP_TIMEOUT_MS"
      --interaction-protocol "$INTERACTION_PROTOCOL"
      --observation-mode "$OBSERVATION_MODE"
      --scoring-mode "$SCORING_MODE"
      --retention-window "$RETENTION_WINDOW"
      --run-label "$run_label"
      --disable-action-coercion
    )
    if [ "$INCLUDE_CONTROLS" = "1" ]; then
      args+=(--include-controls)
    fi
    if [ "$FILL_ONLY" = "1" ]; then
      args+=(--fill-only)
    fi

    if "$PYTHON_BIN" src/baselines/run_gemini_low_cost_eval.py "${args[@]}"; then
      direct_passed=$((direct_passed + 1))
    else
      status=$?
      echo "[WARN] gemini low-cost eval failed for form_id=${form_id} run_index=${run_idx} status=${status}" >&2
      direct_failed=$((direct_failed + 1))
    fi
  done
done

echo "[INFO] direct_eval_total=$direct_total"
echo "[INFO] direct_eval_passed=$direct_passed"
echo "[INFO] direct_eval_failed=$direct_failed"

if [ "$direct_failed" -gt 0 ] && [ "$FAIL_ON_TRIAL_FAILURE" = "1" ]; then
  exit 1
fi
