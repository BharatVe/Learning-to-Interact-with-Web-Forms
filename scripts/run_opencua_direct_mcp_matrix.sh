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

export PATH="$ROOT_DIR/.venv-opencua/bin:$ROOT_DIR/.node-tools/node_modules/.bin:$PATH"
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
EXPERIMENT_ID="${EXPERIMENT_ID:-opencua_direct_mcp_tools_smoke_$(date -u +%Y%m%d)}"
MODEL_ID="${MODEL_ID:-computer_use_opencua_32b_direct_mcp}"
FORM_IDS="${FORM_IDS:-conf_interest,event_rsvp,course_feedback,internship_app,workshop_signup}"
RUN_INDEX="${RUN_INDEX:-2}"
RUN_INDEXES="${RUN_INDEXES:-$RUN_INDEX}"
FORM_OFFSET="${FORM_OFFSET:-0}"
FORM_LIMIT="${FORM_LIMIT:-0}"
if [ -z "${API_TIMEOUT_S:-}" ]; then
  if [[ "$EXPERIMENT_ID" == *smoke* ]]; then
    API_TIMEOUT_S=180
  else
    API_TIMEOUT_S=300
  fi
fi
if [ -z "${DIRECT_MCP_TIMEOUT_S:-}" ]; then
  if [[ "$EXPERIMENT_ID" == *smoke* ]]; then
    DIRECT_MCP_TIMEOUT_S=1800
  else
    DIRECT_MCP_TIMEOUT_S=9000
  fi
fi
if [ -z "${DIRECT_MCP_MAX_STEPS:-}" ]; then
  if [[ "$EXPERIMENT_ID" == *smoke* ]]; then
    DIRECT_MCP_MAX_STEPS=32
  else
    DIRECT_MCP_MAX_STEPS=128
  fi
fi
DIRECT_MCP_MAX_NEW_TOKENS="${DIRECT_MCP_MAX_NEW_TOKENS:-1024}"
BROWSER_MCP_TIMEOUT_MS="${BROWSER_MCP_TIMEOUT_MS:-600000}"
FAIL_ON_TRIAL_FAILURE="${FAIL_ON_TRIAL_FAILURE:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
OPENAI_MODEL="${OPENAI_MODEL:-opencua-32b}"

export OPENAI_BASE_URL OPENAI_API_KEY OPENAI_MODEL

if ! command -v playwright-mcp >/dev/null 2>&1; then
  echo "[FAIL] playwright-mcp not found on PATH" >&2
  exit 1
fi

PLAYWRIGHT_MCP_INSTALL_LOG="${PLAYWRIGHT_MCP_INSTALL_LOG:-/tmp/playwright-mcp-install-opencua-direct-mcp.log}" \
  "$ROOT_DIR/scripts/ensure_playwright_mcp_runtime.sh"
export PLAYWRIGHT_MCP_CHROMIUM_EXECUTABLE="$(cat "$PLAYWRIGHT_BROWSERS_PATH/.mcp-chromium-executable")"
echo "[INFO] playwright_mcp_chromium_executable=$PLAYWRIGHT_MCP_CHROMIUM_EXECUTABLE"

if [ "$FORM_IDS" = "all" ]; then
  RESOLVED_FORM_IDS="$("$PYTHON_BIN" - "$FORM_OFFSET" "$FORM_LIMIT" <<'PY_FORMS'
import sys
from pathlib import Path
offset = max(0, int(sys.argv[1]))
limit = max(0, int(sys.argv[2]))
forms = sorted(entry.name for entry in Path("src/forms").iterdir() if entry.is_dir() and (entry / "spec.json").exists())
if offset:
    forms = forms[offset:]
if limit:
    forms = forms[:limit]
print(",".join(forms))
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
IFS=',' read -r -a FORMS <<<"$RESOLVED_FORM_IDS"

trial_completed() {
  local form_id="$1"
  local run_idx="$2"
  local answer_run_id
  answer_run_id="$(printf 'run_%04d' "$run_idx")"
  if compgen -G "$ROOT_DIR/data/model_baselines/*/$MODEL_ID/$form_id/$answer_run_id/*/summary.json" >/dev/null; then
    return 0
  fi
  compgen -G "$ROOT_DIR/data/model_baselines/$EXPERIMENT_ID/$MODEL_ID/$form_id/$answer_run_id/*/summary.json" >/dev/null
}

make_trial_id() {
  "$PYTHON_BIN" - <<'PY_TRIAL'
from datetime import datetime
print("trial_" + datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ"))
PY_TRIAL
}

make_run_label() {
  "$PYTHON_BIN" - <<'PY_LABEL'
import os
import re
from datetime import datetime
job_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(os.environ.get("SLURM_JOB_ID") or "na").strip()) or "na"
print(datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + f"_job{job_id}")
PY_LABEL
}

echo "[INFO] opencua_direct_mcp_experiment_id=$EXPERIMENT_ID"
echo "[INFO] model_id=$MODEL_ID"
echo "[INFO] forms=$RESOLVED_FORM_IDS"
echo "[INFO] run_indexes=$RUN_INDEXES"
echo "[INFO] skip_completed=$SKIP_COMPLETED"
echo "[INFO] openai_base_url=$OPENAI_BASE_URL"
echo "[INFO] openai_model=$OPENAI_MODEL"
echo "[INFO] api_timeout_s=$API_TIMEOUT_S"
echo "[INFO] direct_mcp_timeout_s=$DIRECT_MCP_TIMEOUT_S"
echo "[INFO] direct_mcp_max_steps=$DIRECT_MCP_MAX_STEPS"
echo "[INFO] direct_mcp_max_new_tokens=$DIRECT_MCP_MAX_NEW_TOKENS"

direct_total=0
direct_passed=0
direct_failed=0

for form_id in "${FORMS[@]}"; do
  for run_idx in "${RUN_INDEX_LIST[@]}"; do
    if [ "$SKIP_COMPLETED" = "1" ] && trial_completed "$form_id" "$run_idx"; then
      echo "[INFO] opencua_direct_mcp_skip_completed form_id=${form_id} run_index=${run_idx}"
      continue
    fi
    trial_id="$(make_trial_id)"
    run_label="$(make_run_label)"
    direct_total=$((direct_total + 1))
    echo "[INFO] opencua_direct_mcp_eval model_id=${MODEL_ID} form_id=${form_id} run_index=${run_idx} trial_id=${trial_id}"
    echo "[INFO] opencua_direct_mcp_trial_start index=${direct_total} form_id=${form_id} run_index=${run_idx} started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    set +e
    "$PYTHON_BIN" src/baselines/run_opencua_direct_mcp_eval.py \
      --config "$CONFIG_PATH" \
      --model-id "$MODEL_ID" \
      --form-id "$form_id" \
      --run-index "$run_idx" \
      --trial-id "$trial_id" \
      --experiment-id "$EXPERIMENT_ID" \
      --api-timeout-s "$API_TIMEOUT_S" \
      --timeout-s "$DIRECT_MCP_TIMEOUT_S" \
      --max-steps "$DIRECT_MCP_MAX_STEPS" \
      --max-new-tokens "$DIRECT_MCP_MAX_NEW_TOKENS" \
      --browser-mcp-timeout-ms "$BROWSER_MCP_TIMEOUT_MS" \
      --headless \
      --run-label "$run_label"
    trial_status=$?
    set -e
    if [ "$trial_status" -eq 0 ]; then
      direct_passed=$((direct_passed + 1))
      echo "[INFO] opencua_direct_mcp_trial_complete status=0 form_id=${form_id} run_index=${run_idx} completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    else
      echo "[WARN] opencua_direct_mcp_trial_failed status=$trial_status form_id=${form_id} run_index=${run_idx}" >&2
      echo "[INFO] opencua_direct_mcp_trial_complete status=${trial_status} form_id=${form_id} run_index=${run_idx} completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      direct_failed=$((direct_failed + 1))
      if [ "$FAIL_ON_TRIAL_FAILURE" = "1" ]; then
        exit "$trial_status"
      fi
    fi
  done
done

echo "[INFO] direct_eval_total=$direct_total"
echo "[INFO] direct_eval_passed=$direct_passed"
echo "[INFO] direct_eval_failed=$direct_failed"
