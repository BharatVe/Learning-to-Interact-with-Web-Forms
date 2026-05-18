#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[FAIL] missing virtualenv interpreter: $PYTHON_BIN" >&2
  exit 1
fi

module load release/25.06 GCCcore/13.3.0 Python/3.12.3 nodejs/20.13.1

CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/.runtime-cache}"
mkdir -p "$CACHE_ROOT"/{xdg,hf,pip,uv,playwright}
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export PATH="$ROOT_DIR/.node-tools/node_modules/.bin:$PATH"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CACHE_ROOT/playwright}"
export PYTHONUNBUFFERED=1
OPENCUA_PYTHON_BIN="${OPENCUA_PYTHON_BIN:-$ROOT_DIR/.venv-opencua/bin/python}"

CONFIG_PATH="${CONFIG_PATH:-configs/baselines/track_baseline_models.json}"
FORM_IDS="${FORM_IDS:-conf_interest,event_rsvp}"
RUN_INDEX="${RUN_INDEX:-1}"
RUN_INDEXES="${RUN_INDEXES:-$RUN_INDEX}"

DIRECT_MCP_EXPERIMENT_ID="${DIRECT_MCP_EXPERIMENT_ID:-track_baseline_qwen_direct_mcp_v1}"
NATIVE_EXPERIMENT_ID="${NATIVE_EXPERIMENT_ID:-track_baseline_opencua_native_v1}"
TRACK_REPORT_OUTPUT="${TRACK_REPORT_OUTPUT:-logs/track_baseline_summary.json}"

API_TIMEOUT_S="${API_TIMEOUT_S:-240}"
BROWSER_INIT_RETRIES="${BROWSER_INIT_RETRIES:-2}"
BROWSER_INIT_RETRY_DELAY_S="${BROWSER_INIT_RETRY_DELAY_S:-1.5}"
FAIL_ON_TRIAL_FAILURE="${FAIL_ON_TRIAL_FAILURE:-0}"

DIRECT_PROVIDER="${DIRECT_PROVIDER:-opencua_local}"
DIRECT_TRACK="${DIRECT_TRACK:-computer_use_native}"
DIRECT_MODEL_ID="${DIRECT_MODEL_ID:-}"
DIRECT_MODEL_PROVIDER="${DIRECT_MODEL_PROVIDER:-}"
DIRECT_MAX_STEPS="${DIRECT_MAX_STEPS:-128}"
DIRECT_TIMEOUT_S="${DIRECT_TIMEOUT_S:-5400}"
DIRECT_MAX_NEW_TOKENS="${DIRECT_MAX_NEW_TOKENS:-96}"
DIRECT_API_TIMEOUT_S="${DIRECT_API_TIMEOUT_S:-180}"
DIRECT_BROWSER_MCP_TIMEOUT_MS="${DIRECT_BROWSER_MCP_TIMEOUT_MS:-600000}"
GEMINI_MIN_REQUEST_INTERVAL_S="${GEMINI_MIN_REQUEST_INTERVAL_S:-8.0}"
GEMINI_MAX_INFER_RETRIES="${GEMINI_MAX_INFER_RETRIES:-2}"
GEMINI_MAX_RETRY_DELAY_S="${GEMINI_MAX_RETRY_DELAY_S:-75}"
OPENCUA_VLLM_HOST="${OPENCUA_VLLM_HOST:-127.0.0.1}"
if [ -z "${OPENCUA_VLLM_PORT:-}" ]; then
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    OPENCUA_VLLM_PORT="$((18000 + (SLURM_JOB_ID % 20000)))"
  else
    OPENCUA_VLLM_PORT="8000"
  fi
fi
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://${OPENCUA_VLLM_HOST}:${OPENCUA_VLLM_PORT}/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
OPEN_CUA_MIN_REQUEST_INTERVAL_S="${OPEN_CUA_MIN_REQUEST_INTERVAL_S:-2.0}"
OPEN_CUA_HISTORY_IMAGES="${OPEN_CUA_HISTORY_IMAGES:-3}"

INTERACTION_PROTOCOL="${INTERACTION_PROTOCOL:-human_ui_v1}"
OBSERVATION_MODE="${OBSERVATION_MODE:-vision_coords}"
SCORING_MODE="${SCORING_MODE:-soft_quality_v1}"
RETENTION_WINDOW="${RETENTION_WINDOW:-5}"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "[FAIL] config_path does not exist: $CONFIG_PATH" >&2
  exit 1
fi

case "$DIRECT_PROVIDER" in
  opencua_local)
    if [ ! -x "$OPENCUA_PYTHON_BIN" ]; then
      echo "[FAIL] OPENCUA_PYTHON_BIN is not executable: $OPENCUA_PYTHON_BIN" >&2
      exit 1
    fi
    if ! command -v vllm >/dev/null 2>&1; then
      echo "[FAIL] vllm is required for opencua_local computer-use baseline" >&2
      exit 1
    fi
    ;;
  gemini_native)
    if [ -z "${GEMINI_API_KEY:-}" ]; then
      echo "[FAIL] GEMINI_API_KEY is required for gemini_native computer-use baseline" >&2
      exit 1
    fi
    ;;
  anthropic)
    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
      echo "[FAIL] ANTHROPIC_API_KEY is required for anthropic direct baseline" >&2
      exit 1
    fi
    ;;
  openai)
    if [ -z "${OPENAI_API_KEY:-}" ]; then
      echo "[FAIL] OPENAI_API_KEY is required for openai direct baseline" >&2
      exit 1
    fi
    ;;
  auto)
    if [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
      echo "[FAIL] DIRECT_PROVIDER=auto requires OPENAI_API_KEY or ANTHROPIC_API_KEY" >&2
      exit 1
    fi
    ;;
  *)
    echo "[FAIL] unsupported DIRECT_PROVIDER: $DIRECT_PROVIDER (expected opencua_local|gemini_native|anthropic|openai|auto)" >&2
    exit 1
    ;;
esac

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
EXPECTED_RUNS_PER_FORM="${#RUN_INDEX_LIST[@]}"
EXPECTED_FORM_COUNT="$($PYTHON_BIN - "$RESOLVED_FORM_IDS" <<'PY_COUNT'
import sys
raw = str(sys.argv[1] if len(sys.argv) > 1 else '').strip()
parts = [tok.strip() for tok in raw.split(',') if tok.strip()]
print(len(parts))
PY_COUNT
)"

if [ -z "$DIRECT_MODEL_ID" ]; then
  DIRECT_MODEL_ID="$($PYTHON_BIN - "$CONFIG_PATH" "$DIRECT_TRACK" "$DIRECT_PROVIDER" <<'PY_MODEL'
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
track = str(sys.argv[2] or '').strip()
direct_provider = str(sys.argv[3] or '').strip()
if direct_provider == 'gemini_native':
    target_provider = 'gemini_native'
elif direct_provider == 'opencua_local':
    target_provider = 'openai_compat'
else:
    target_provider = 'api_over_mcp'
for model in cfg.get('models', []):
    if not isinstance(model, dict):
        continue
    if str(model.get('track') or '') != track:
        continue
    if str(model.get('provider') or '') != target_provider:
        continue
    model_id = str(model.get('id') or '').strip()
    if model_id:
        print(model_id)
        break
PY_MODEL
)"
fi

if [ -z "$DIRECT_MODEL_ID" ]; then
  echo "[FAIL] no direct API model found in $CONFIG_PATH (track=$DIRECT_TRACK provider=${DIRECT_PROVIDER})" >&2
  exit 1
fi

if [ -z "$DIRECT_MODEL_PROVIDER" ]; then
  DIRECT_MODEL_PROVIDER="$($PYTHON_BIN - "$CONFIG_PATH" "$DIRECT_MODEL_ID" <<'PY_MODEL_PROVIDER'
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
model_id = str(sys.argv[2] or '').strip()
for model in cfg.get('models', []):
    if not isinstance(model, dict):
        continue
    if str(model.get('id') or '').strip() == model_id:
        print(str(model.get('provider') or '').strip())
        break
PY_MODEL_PROVIDER
)"
fi

if [ "$DIRECT_PROVIDER" = "gemini_native" ] && [ "$DIRECT_MODEL_PROVIDER" != "gemini_native" ]; then
  echo "[FAIL] selected direct model '$DIRECT_MODEL_ID' is provider=$DIRECT_MODEL_PROVIDER, expected gemini_native" >&2
  exit 1
fi
if [ "$DIRECT_PROVIDER" = "opencua_local" ] && [ "$DIRECT_MODEL_PROVIDER" != "openai_compat" ]; then
  echo "[FAIL] selected direct model '$DIRECT_MODEL_ID' is provider=$DIRECT_MODEL_PROVIDER, expected openai_compat for DIRECT_PROVIDER=$DIRECT_PROVIDER" >&2
  exit 1
fi
if [ "$DIRECT_PROVIDER" != "gemini_native" ] && [ "$DIRECT_PROVIDER" != "opencua_local" ] && [ "$DIRECT_MODEL_PROVIDER" != "api_over_mcp" ]; then
  echo "[FAIL] selected direct model '$DIRECT_MODEL_ID' is provider=$DIRECT_MODEL_PROVIDER, expected api_over_mcp for DIRECT_PROVIDER=$DIRECT_PROVIDER" >&2
  exit 1
fi

echo "[INFO] config_path=$CONFIG_PATH"
echo "[INFO] forms=$RESOLVED_FORM_IDS"
echo "[INFO] run_indexes=$RUN_INDEXES"
echo "[INFO] direct_mcp_experiment_id=$DIRECT_MCP_EXPERIMENT_ID"
echo "[INFO] native_experiment_id=$NATIVE_EXPERIMENT_ID"
echo "[INFO] direct_model_id=$DIRECT_MODEL_ID"
echo "[INFO] direct_model_provider=$DIRECT_MODEL_PROVIDER"
echo "[INFO] direct_provider=$DIRECT_PROVIDER"
echo "[INFO] interaction_protocol=$INTERACTION_PROTOCOL observation_mode=$OBSERVATION_MODE scoring_mode=$SCORING_MODE"
if [ "$DIRECT_PROVIDER" = "opencua_local" ]; then
  echo "[INFO] openai_base_url=$OPENAI_BASE_URL"
fi

if ! EXPERIMENT_ID="$DIRECT_MCP_EXPERIMENT_ID" \
  CONFIG_PATH="$CONFIG_PATH" \
  FORM_IDS="$RESOLVED_FORM_IDS" \
  RUN_INDEXES="$RUN_INDEXES" \
  API_TIMEOUT_S="$API_TIMEOUT_S" \
  FAIL_ON_TRIAL_FAILURE="$FAIL_ON_TRIAL_FAILURE" \
  DIRECT_MCP_TIMEOUT_S="$DIRECT_TIMEOUT_S" \
  DIRECT_MCP_MAX_STEPS="$DIRECT_MAX_STEPS" \
  BROWSER_MCP_TIMEOUT_MS="$DIRECT_BROWSER_MCP_TIMEOUT_MS" \
  bash scripts/run_qwen_direct_mcp_matrix.sh; then
  echo "[FAIL] direct MCP track orchestration failed; aborting before native computer-use track" >&2
  exit 1
fi

# Direct track preflight checks.
SERVER_PID=""
cleanup_direct_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup_direct_server EXIT

if [ "$DIRECT_PROVIDER" = "opencua_local" ]; then
  export OPENAI_BASE_URL OPENAI_API_KEY
  OPENAI_MODEL="$($PYTHON_BIN - "$CONFIG_PATH" "$DIRECT_MODEL_ID" <<'PY_OPENAI_MODEL'
import json
import sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
model_id = str(sys.argv[2] or '').strip()
for model in cfg.get('models', []):
    if not isinstance(model, dict):
        continue
    if str(model.get('id') or '').strip() == model_id:
        print(str(model.get('openai_model') or model.get('served_model_name') or 'opencua-32b').strip())
        break
PY_OPENAI_MODEL
)"
  export OPENAI_MODEL
  export OPENCUA_SERVED_MODEL_NAME="$OPENAI_MODEL"
  export OPENCUA_VLLM_HOST OPENCUA_VLLM_PORT OPENAI_BASE_URL
  OPENCUA_VLLM_LOG="${OPENCUA_VLLM_LOG:-logs/slurm/opencua-vllm-track-${SLURM_JOB_ID:-na}.log}"
  bash scripts/run_opencua_vllm_server.sh >"$OPENCUA_VLLM_LOG" 2>&1 &
  SERVER_PID=$!
  echo "[INFO] opencua_vllm_pid=$SERVER_PID"
  echo "[INFO] opencua_vllm_log=$OPENCUA_VLLM_LOG"
  READY_ATTEMPTS="${OPENCUA_READY_ATTEMPTS:-180}"
  READY_SLEEP_S="${OPENCUA_READY_SLEEP_S:-10}"
  READY_STARTED="$(date +%s)"
  for attempt in $(seq 1 "$READY_ATTEMPTS"); do
    if "$OPENCUA_PYTHON_BIN" scripts/check_openai_compat_server.py \
      --base-url "$OPENAI_BASE_URL" \
      --model "$OPENAI_MODEL" \
      --api-key "$OPENAI_API_KEY" \
      --timeout-s "${OPENCUA_READY_TIMEOUT_S:-10}" \
      --print-models >/tmp/opencua-ready-${SLURM_JOB_ID:-na}.log 2>&1; then
      cat /tmp/opencua-ready-${SLURM_JOB_ID:-na}.log
      echo "[INFO] opencua_vllm_ready attempt=$attempt elapsed_s=$(( $(date +%s) - READY_STARTED ))"
      break
    fi
    if [ $((attempt % 6)) -eq 0 ]; then
      echo "[INFO] waiting_for_opencua_vllm attempt=${attempt}/${READY_ATTEMPTS} elapsed_s=$(( $(date +%s) - READY_STARTED ))"
    fi
    sleep "$READY_SLEEP_S"
    if [ "$attempt" -eq "$READY_ATTEMPTS" ]; then
      echo "[FAIL] OpenCUA vLLM server did not become ready" >&2
      cat /tmp/opencua-ready-${SLURM_JOB_ID:-na}.log >&2 || true
      tail -n 120 "$OPENCUA_VLLM_LOG" >&2 || true
      exit 1
    fi
  done
  "$OPENCUA_PYTHON_BIN" scripts/check_openai_compat_server.py \
    --base-url "$OPENAI_BASE_URL" \
    --model "$OPENAI_MODEL" \
    --api-key "$OPENAI_API_KEY" \
    --timeout-s "${OPENCUA_SMOKE_TIMEOUT_S:-60}" \
    --print-models \
    --smoke-chat
fi

if [ "$DIRECT_PROVIDER" = "gemini_native" ]; then
  DIRECT_SMOKE_EXCLUDE_PROVIDERS="openai_compat,local_hf,api_over_mcp"
elif [ "$DIRECT_PROVIDER" = "opencua_local" ]; then
  DIRECT_SMOKE_EXCLUDE_PROVIDERS="local_hf,api_over_mcp,gemini_native"
else
  DIRECT_SMOKE_EXCLUDE_PROVIDERS="openai_compat,local_hf,gemini_native"
fi
if [ "$DIRECT_PROVIDER" = "opencua_local" ]; then
  "$OPENCUA_PYTHON_BIN" scripts/verify_opencua_compatibility.py \
    --config "$CONFIG_PATH" \
    --model-id "$DIRECT_MODEL_ID" \
    --form-ids "$RESOLVED_FORM_IDS" \
    --run-indexes "$RUN_INDEXES" \
    --base-url "$OPENAI_BASE_URL" \
    --require-endpoint
fi
if ! "$PYTHON_BIN" scripts/eval_model_baseline_smoke.py \
  --config "$CONFIG_PATH" \
  --include-kinds computer_use_agent \
  --exclude-providers "$DIRECT_SMOKE_EXCLUDE_PROVIDERS" \
  --strict; then
  echo "[FAIL] computer-use preflight failed" >&2
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

summary_path_for() {
  local form_id="$1"
  local run_index="$2"
  local trial_id="$3"
  printf '%s/data/model_baselines/%s/%s/%s/run_%04d/%s/summary.json\n' \
    "$ROOT_DIR" "$NATIVE_EXPERIMENT_ID" "$DIRECT_MODEL_ID" "$form_id" "$run_index" "$trial_id"
}

direct_total=0
direct_passed=0
direct_failed=0
declare -a direct_failed_summaries=()

IFS=',' read -r -a FORMS <<<"$RESOLVED_FORM_IDS"
for form_id in "${FORMS[@]}"; do
  for run_idx in "${RUN_INDEX_LIST[@]}"; do
    trial_id="$(make_trial_id)"
    run_label="$(make_run_label)"
    summary_path="$(summary_path_for "$form_id" "$run_idx" "$trial_id")"
    direct_total=$((direct_total + 1))
    echo "[INFO] direct_eval model_id=${DIRECT_MODEL_ID} form_id=${form_id} run_index=${run_idx} trial_id=${trial_id} provider=${DIRECT_PROVIDER}"

    if [ "$DIRECT_PROVIDER" = "gemini_native" ]; then
      if GEMINI_MIN_REQUEST_INTERVAL_S="$GEMINI_MIN_REQUEST_INTERVAL_S" \
        GEMINI_MAX_INFER_RETRIES="$GEMINI_MAX_INFER_RETRIES" \
        GEMINI_MAX_RETRY_DELAY_S="$GEMINI_MAX_RETRY_DELAY_S" \
        "$PYTHON_BIN" src/baselines/run_gemini_native_computer_use_eval.py \
        --config "$CONFIG_PATH" \
        --model-id "$DIRECT_MODEL_ID" \
        --form-id "$form_id" \
        --run-index "$run_idx" \
        --trial-id "$trial_id" \
        --experiment-id "$NATIVE_EXPERIMENT_ID" \
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
        direct_failed_summaries+=("$summary_path")
        if [ "$status" -eq 2 ]; then
          echo "[FAIL] Gemini quota exhausted; aborting remaining direct trials." >&2
          exit 2
        fi
      fi
    elif [ "$DIRECT_PROVIDER" = "opencua_local" ]; then
      if OPEN_CUA_MIN_REQUEST_INTERVAL_S="$OPEN_CUA_MIN_REQUEST_INTERVAL_S" \
        OPENAI_BASE_URL="$OPENAI_BASE_URL" \
        OPENAI_API_KEY="$OPENAI_API_KEY" \
        OPENAI_MODEL="${OPENAI_MODEL:-opencua-32b}" \
        "$OPENCUA_PYTHON_BIN" src/baselines/run_opencua_direct_eval.py \
        --config "$CONFIG_PATH" \
        --model-id "$DIRECT_MODEL_ID" \
        --form-id "$form_id" \
        --run-index "$run_idx" \
        --trial-id "$trial_id" \
        --experiment-id "$NATIVE_EXPERIMENT_ID" \
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
        --base-url "$OPENAI_BASE_URL" \
        --served-model-name "${OPENAI_MODEL:-opencua-32b}" \
        --coordinate-type qwen25 \
        --history-images "$OPEN_CUA_HISTORY_IMAGES" \
        --disable-action-coercion; then
        direct_passed=$((direct_passed + 1))
      else
        echo "[WARN] direct eval failed for form_id=${form_id} run_index=${run_idx}" >&2
        direct_failed=$((direct_failed + 1))
        direct_failed_summaries+=("$summary_path")
      fi
    elif "$PYTHON_BIN" src/baselines/run_direct_api_eval.py \
      --config "$CONFIG_PATH" \
      --model-id "$DIRECT_MODEL_ID" \
      --form-id "$form_id" \
      --run-index "$run_idx" \
      --trial-id "$trial_id" \
      --experiment-id "$NATIVE_EXPERIMENT_ID" \
      --provider "$DIRECT_PROVIDER" \
      --api-timeout-s "$DIRECT_API_TIMEOUT_S" \
      --execution-backend mcp_server \
      --headless \
      --max-new-tokens "$DIRECT_MAX_NEW_TOKENS" \
      --max-steps "$DIRECT_MAX_STEPS" \
      --timeout-s "$DIRECT_TIMEOUT_S" \
      --browser-mcp-timeout-ms "$DIRECT_BROWSER_MCP_TIMEOUT_MS" \
      --retention-window "$RETENTION_WINDOW" \
      --run-label "$run_label" \
      --disable-action-coercion; then
      direct_passed=$((direct_passed + 1))
    else
      echo "[WARN] direct eval failed for form_id=${form_id} run_index=${run_idx}" >&2
      direct_failed=$((direct_failed + 1))
      direct_failed_summaries+=("$summary_path")
    fi
  done
done

echo "[INFO] direct_eval_total=$direct_total"
echo "[INFO] direct_eval_passed=$direct_passed"
echo "[INFO] direct_eval_failed=$direct_failed"
if [ "$direct_failed" -gt 0 ]; then
  printf '[INFO] direct_failed_summary=%s\n' "${direct_failed_summaries[@]}"
fi

"$PYTHON_BIN" scripts/summarize_track_baseline.py \
  --family-a-experiment-id "$DIRECT_MCP_EXPERIMENT_ID" \
  --family-b-experiment-id "$NATIVE_EXPERIMENT_ID" \
  --config-path "$CONFIG_PATH" \
  --expected-forms "$EXPECTED_FORM_COUNT" \
  --expected-runs-per-form "$EXPECTED_RUNS_PER_FORM" \
  --output "$TRACK_REPORT_OUTPUT"

overall_status=0
if [ "$direct_failed" -gt 0 ]; then
  if [ "$FAIL_ON_TRIAL_FAILURE" = "1" ]; then
    overall_status=1
  else
    echo "[WARN] trial-level failures detected but FAIL_ON_TRIAL_FAILURE=0; keeping successful exit" >&2
  fi
fi

exit "$overall_status"
