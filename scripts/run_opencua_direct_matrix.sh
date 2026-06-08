#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${OPENCUA_PYTHON_BIN:-$ROOT_DIR/.venv-opencua/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[FAIL] missing virtualenv interpreter: $PYTHON_BIN" >&2
  exit 1
fi

module load release/25.06 GCCcore/13.3.0 Python/3.12.3 nodejs/20.13.1

export PATH="$ROOT_DIR/.node-tools/node_modules/.bin:$PATH"
CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/.runtime-cache}"
mkdir -p "$CACHE_ROOT"/{xdg,hf,pip,uv,playwright}
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CACHE_ROOT/playwright}"
export PYTHONUNBUFFERED=1

CONFIG_PATH="${CONFIG_PATH:-configs/baselines/track_baseline_models.json}"
FORM_IDS="${FORM_IDS:-conf_interest,event_rsvp,course_feedback,internship_app,workshop_signup}"
RUN_INDEX="${RUN_INDEX:-1}"
RUN_INDEXES="${RUN_INDEXES:-$RUN_INDEX}"
FORM_OFFSET="${FORM_OFFSET:-0}"
FORM_LIMIT="${FORM_LIMIT:-0}"
DIRECT_EXPERIMENT_ID="${DIRECT_EXPERIMENT_ID:-${EXPERIMENT_ID:-track_baseline_opencua_native_pilot}}"
DIRECT_TRACK="${DIRECT_TRACK:-computer_use_native}"
DIRECT_MODEL_ID="${DIRECT_MODEL_ID:-}"
DIRECT_MAX_STEPS="${DIRECT_MAX_STEPS:-128}"
DIRECT_TIMEOUT_S="${DIRECT_TIMEOUT_S:-7200}"
DIRECT_MAX_NEW_TOKENS="${DIRECT_MAX_NEW_TOKENS:-96}"
DIRECT_API_TIMEOUT_S="${DIRECT_API_TIMEOUT_S:-180}"
DIRECT_BROWSER_MCP_TIMEOUT_MS="${DIRECT_BROWSER_MCP_TIMEOUT_MS:-600000}"
INTERACTION_PROTOCOL="${INTERACTION_PROTOCOL:-human_ui_v1}"
OBSERVATION_MODE="${OBSERVATION_MODE:-vision_coords}"
SCORING_MODE="${SCORING_MODE:-soft_quality_v1}"
RETENTION_WINDOW="${RETENTION_WINDOW:-5}"
FAIL_ON_TRIAL_FAILURE="${FAIL_ON_TRIAL_FAILURE:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
OPEN_CUA_MIN_REQUEST_INTERVAL_S="${OPEN_CUA_MIN_REQUEST_INTERVAL_S:-2.0}"
OPEN_CUA_HISTORY_IMAGES="${OPEN_CUA_HISTORY_IMAGES:-3}"
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

export OPENAI_BASE_URL OPENAI_API_KEY

if [ ! -f "$CONFIG_PATH" ]; then
  echo "[FAIL] config_path does not exist: $CONFIG_PATH" >&2
  exit 1
fi

if ! command -v playwright-mcp >/dev/null 2>&1; then
  echo "[FAIL] playwright-mcp not found on PATH" >&2
  exit 1
fi
if ! command -v vllm >/dev/null 2>&1; then
  echo "[FAIL] vllm not found on PATH" >&2
  exit 1
fi

PLAYWRIGHT_MCP_INSTALL_LOG="${PLAYWRIGHT_MCP_INSTALL_LOG:-/tmp/playwright-mcp-install-opencua.log}" \
  "$ROOT_DIR/scripts/ensure_playwright_mcp_runtime.sh"
export PLAYWRIGHT_MCP_CHROMIUM_EXECUTABLE="$(cat "$PLAYWRIGHT_BROWSERS_PATH/.mcp-chromium-executable")"
echo "[INFO] playwright_mcp_chromium_executable=$PLAYWRIGHT_MCP_CHROMIUM_EXECUTABLE"

if [ "$FORM_IDS" = "all" ]; then
  RESOLVED_FORM_IDS="$($PYTHON_BIN - "$FORM_OFFSET" "$FORM_LIMIT" <<'PY_FORMS'
import sys
from pathlib import Path
offset = max(0, int(sys.argv[1]))
limit = max(0, int(sys.argv[2]))
root = Path('src/forms')
form_ids = sorted(entry.name for entry in root.iterdir() if entry.is_dir() and (entry / 'spec.json').exists())
if offset:
    form_ids = form_ids[offset:]
if limit:
    form_ids = form_ids[:limit]
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
    if str(model.get('provider') or '') != 'openai_compat':
        continue
    if str(model.get('kind') or '') != 'computer_use_agent':
        continue
    model_id = str(model.get('id') or '').strip()
    if model_id:
        print(model_id)
        break
PY_MODEL
)"
fi

if [ -z "$DIRECT_MODEL_ID" ]; then
  echo "[FAIL] no OpenCUA direct model found in $CONFIG_PATH" >&2
  exit 1
fi

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

if ! "$PYTHON_BIN" scripts/verify_opencua_compatibility.py \
  --config "$CONFIG_PATH" \
  --model-id "$DIRECT_MODEL_ID" \
  --form-ids "$RESOLVED_FORM_IDS" \
  --run-indexes "$RUN_INDEXES" \
  --base-url "$OPENAI_BASE_URL" \
  --require-endpoint; then
  echo "[FAIL] OpenCUA compatibility verification failed" >&2
  exit 1
fi

if ! "$PYTHON_BIN" scripts/eval_model_baseline_smoke.py \
  --config "$CONFIG_PATH" \
  --include-kinds computer_use_agent \
  --strict; then
  echo "[FAIL] OpenCUA direct preflight failed" >&2
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

trial_completed() {
  local form_id="$1"
  local run_idx="$2"
  local answer_run_id
  answer_run_id="$(printf 'run_%04d' "$run_idx")"
  if compgen -G "$ROOT_DIR/data/model_baselines/*/$DIRECT_MODEL_ID/$form_id/$answer_run_id/*/summary.json" >/dev/null; then
    return 0
  fi
  compgen -G "$ROOT_DIR/data/model_baselines/$DIRECT_EXPERIMENT_ID/$DIRECT_MODEL_ID/$form_id/$answer_run_id/*/summary.json" >/dev/null
}

direct_total=0
direct_passed=0
direct_failed=0

echo "[INFO] config_path=$CONFIG_PATH"
echo "[INFO] forms=$RESOLVED_FORM_IDS"
echo "[INFO] run_indexes=$RUN_INDEXES"
echo "[INFO] form_offset=$FORM_OFFSET"
echo "[INFO] form_limit=$FORM_LIMIT"
echo "[INFO] direct_experiment_id=$DIRECT_EXPERIMENT_ID"
echo "[INFO] direct_model_id=$DIRECT_MODEL_ID"
echo "[INFO] direct_provider=opencua_local"
echo "[INFO] served_model_name=$OPENAI_MODEL"
echo "[INFO] base_url=$OPENAI_BASE_URL"
echo "[INFO] skip_completed=$SKIP_COMPLETED"

IFS=',' read -r -a FORMS <<<"$RESOLVED_FORM_IDS"
for form_id in "${FORMS[@]}"; do
  for run_idx in "${RUN_INDEX_LIST[@]}"; do
    if [ "$SKIP_COMPLETED" = "1" ] && trial_completed "$form_id" "$run_idx"; then
      echo "[INFO] opencua_direct_skip_completed model_id=${DIRECT_MODEL_ID} form_id=${form_id} run_index=${run_idx}"
      continue
    fi
    trial_id="$(make_trial_id)"
    run_label="$(make_run_label)"
    direct_total=$((direct_total + 1))
    echo "[INFO] direct_eval model_id=${DIRECT_MODEL_ID} form_id=${form_id} run_index=${run_idx} trial_id=${trial_id} provider=opencua_local"

    if OPEN_CUA_MIN_REQUEST_INTERVAL_S="$OPEN_CUA_MIN_REQUEST_INTERVAL_S" \
      "$PYTHON_BIN" src/baselines/run_opencua_direct_eval.py \
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
      --base-url "$OPENAI_BASE_URL" \
      --served-model-name "$OPENAI_MODEL" \
      --coordinate-type qwen25 \
      --history-images "$OPEN_CUA_HISTORY_IMAGES" \
      --disable-action-coercion; then
      direct_passed=$((direct_passed + 1))
    else
      echo "[WARN] direct eval failed for form_id=${form_id} run_index=${run_idx}" >&2
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
