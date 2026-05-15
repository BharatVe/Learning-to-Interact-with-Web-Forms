#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[WARN] scripts/run_comparison_matrix.sh is legacy; use scripts/run_track_baseline_matrix.sh for thesis-primary baseline runs."

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[FAIL] missing virtualenv interpreter: $PYTHON_BIN" >&2
  exit 1
fi

module load release/25.06 GCCcore/13.3.0 nodejs/20.13.1

export PATH="$ROOT_DIR/.node-tools/node_modules/.bin:$PATH"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$ROOT_DIR/.playwright-browsers-node}"
export PYTHONUNBUFFERED=1

CONFIG_PATH="${CONFIG_PATH:-configs/baselines/minimal_models.json}"
FORM_IDS="${FORM_IDS:-conf_interest,event_rsvp}"
RUN_INDEX="${RUN_INDEX:-1}"
RUN_INDEXES="${RUN_INDEXES:-$RUN_INDEX}"

MEDIATED_EXPERIMENT_ID="${MEDIATED_EXPERIMENT_ID:-comparison_mediated_v1}"
DIRECT_EXPERIMENT_ID="${DIRECT_EXPERIMENT_ID:-comparison_direct_api_v1}"
COMPARISON_OUTPUT="${COMPARISON_OUTPUT:-logs/comparison_summary.json}"

MEDIATED_BUDGET_PROFILE="${MEDIATED_BUDGET_PROFILE:-large_qwen3}"
RESOURCE_PROFILE="${RESOURCE_PROFILE:-large_qwen3}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-auto}"
API_TIMEOUT_S="${API_TIMEOUT_S:-120}"
BROWSER_INIT_RETRIES="${BROWSER_INIT_RETRIES:-2}"
BROWSER_INIT_RETRY_DELAY_S="${BROWSER_INIT_RETRY_DELAY_S:-1.5}"

case "$MEDIATED_BUDGET_PROFILE" in
  balanced)
    PROFILE_MEDIATED_MAX_STEPS=36
    PROFILE_MEDIATED_TIMEOUT_S=1800
    PROFILE_MEDIATED_MAX_NEW_TOKENS=224
    PROFILE_MEDIATED_STEP_SOFT_TIMEOUT_S=60
    PROFILE_MEDIATED_STEP_RETRY_MAX_NEW_TOKENS=160
    PROFILE_MEDIATED_COMPACT_PAGE_TEXT_MAX_CHARS=5000
    PROFILE_MEDIATED_BROWSER_MCP_TIMEOUT_MS=120000
    PROFILE_PROMPT_PROFILE="detailed_v1"
    ;;
  large)
    PROFILE_MEDIATED_MAX_STEPS=56
    PROFILE_MEDIATED_TIMEOUT_S=3600
    PROFILE_MEDIATED_MAX_NEW_TOKENS=384
    PROFILE_MEDIATED_STEP_SOFT_TIMEOUT_S=180
    PROFILE_MEDIATED_STEP_RETRY_MAX_NEW_TOKENS=256
    PROFILE_MEDIATED_COMPACT_PAGE_TEXT_MAX_CHARS=9000
    PROFILE_MEDIATED_BROWSER_MCP_TIMEOUT_MS=240000
    PROFILE_PROMPT_PROFILE="detailed_v1"
    ;;
  xlarge)
    PROFILE_MEDIATED_MAX_STEPS=72
    PROFILE_MEDIATED_TIMEOUT_S=5400
    PROFILE_MEDIATED_MAX_NEW_TOKENS=512
    PROFILE_MEDIATED_STEP_SOFT_TIMEOUT_S=240
    PROFILE_MEDIATED_STEP_RETRY_MAX_NEW_TOKENS=320
    PROFILE_MEDIATED_COMPACT_PAGE_TEXT_MAX_CHARS=12000
    PROFILE_MEDIATED_BROWSER_MCP_TIMEOUT_MS=300000
    PROFILE_PROMPT_PROFILE="detailed_v1"
    ;;
  large_qwen3)
    PROFILE_MEDIATED_MAX_STEPS=64
    PROFILE_MEDIATED_TIMEOUT_S=5400
    PROFILE_MEDIATED_MAX_NEW_TOKENS=320
    PROFILE_MEDIATED_STEP_SOFT_TIMEOUT_S=300
    PROFILE_MEDIATED_STEP_RETRY_MAX_NEW_TOKENS=192
    PROFILE_MEDIATED_COMPACT_PAGE_TEXT_MAX_CHARS=6000
    PROFILE_MEDIATED_BROWSER_MCP_TIMEOUT_MS=300000
    PROFILE_PROMPT_PROFILE="runtime_safe_v1"
    ;;
  *)
    echo "[FAIL] unsupported MEDIATED_BUDGET_PROFILE=$MEDIATED_BUDGET_PROFILE (expected balanced|large|xlarge|large_qwen3)" >&2
    exit 1
    ;;
esac

MEDIATED_MAX_STEPS="${MEDIATED_MAX_STEPS:-$PROFILE_MEDIATED_MAX_STEPS}"
MEDIATED_TIMEOUT_S="${MEDIATED_TIMEOUT_S:-$PROFILE_MEDIATED_TIMEOUT_S}"
MEDIATED_MAX_NEW_TOKENS="${MEDIATED_MAX_NEW_TOKENS:-$PROFILE_MEDIATED_MAX_NEW_TOKENS}"
MEDIATED_STEP_SOFT_TIMEOUT_S="${MEDIATED_STEP_SOFT_TIMEOUT_S:-$PROFILE_MEDIATED_STEP_SOFT_TIMEOUT_S}"
MEDIATED_STEP_RETRY_MAX_NEW_TOKENS="${MEDIATED_STEP_RETRY_MAX_NEW_TOKENS:-$PROFILE_MEDIATED_STEP_RETRY_MAX_NEW_TOKENS}"
MEDIATED_COMPACT_PAGE_TEXT_MAX_CHARS="${MEDIATED_COMPACT_PAGE_TEXT_MAX_CHARS:-$PROFILE_MEDIATED_COMPACT_PAGE_TEXT_MAX_CHARS}"
MEDIATED_BROWSER_MCP_TIMEOUT_MS="${MEDIATED_BROWSER_MCP_TIMEOUT_MS:-$PROFILE_MEDIATED_BROWSER_MCP_TIMEOUT_MS}"
PROMPT_PROFILE="${PROMPT_PROFILE:-$PROFILE_PROMPT_PROFILE}"
MEDIATED_CONTROL_LEVEL="${MEDIATED_CONTROL_LEVEL:-high_level}"
INTERACTION_PROTOCOL="${INTERACTION_PROTOCOL:-human_ui_v1}"
OBSERVATION_MODE="${OBSERVATION_MODE:-vision_coords}"
SCORING_MODE="${SCORING_MODE:-soft_quality_v1}"
RETENTION_WINDOW="${RETENTION_WINDOW:-5}"

case "$MEDIATED_CONTROL_LEVEL" in
  high_level|low_level)
    ;;
  *)
    echo "[FAIL] unsupported MEDIATED_CONTROL_LEVEL=$MEDIATED_CONTROL_LEVEL (expected high_level|low_level)" >&2
    exit 1
    ;;
esac

case "$INTERACTION_PROTOCOL" in
  legacy_semantic_v1|human_ui_v1)
    ;;
  *)
    echo "[FAIL] unsupported INTERACTION_PROTOCOL=$INTERACTION_PROTOCOL (expected legacy_semantic_v1|human_ui_v1)" >&2
    exit 1
    ;;
esac

case "$OBSERVATION_MODE" in
  vision_coords|vision_coords_text)
    ;;
  *)
    echo "[FAIL] unsupported OBSERVATION_MODE=$OBSERVATION_MODE (expected vision_coords|vision_coords_text)" >&2
    exit 1
    ;;
esac

case "$SCORING_MODE" in
  soft_quality_v1|legacy_binary_v1)
    ;;
  *)
    echo "[FAIL] unsupported SCORING_MODE=$SCORING_MODE (expected soft_quality_v1|legacy_binary_v1)" >&2
    exit 1
    ;;
esac

DIRECT_TRACK="${DIRECT_TRACK:-direct_api_tool_use}"
DIRECT_MODEL_ID="${DIRECT_MODEL_ID:-}"
DIRECT_MAX_STEPS="${DIRECT_MAX_STEPS:-15}"
DIRECT_TIMEOUT_S="${DIRECT_TIMEOUT_S:-300}"
DIRECT_MAX_NEW_TOKENS="${DIRECT_MAX_NEW_TOKENS:-192}"
DIRECT_API_TIMEOUT_S="${DIRECT_API_TIMEOUT_S:-60}"
DIRECT_PROVIDER="${DIRECT_PROVIDER:-auto}"

if ! command -v playwright-mcp >/dev/null 2>&1; then
  echo "[FAIL] playwright-mcp not found on PATH" >&2
  exit 1
fi

if [ -z "$DIRECT_MODEL_ID" ]; then
  DIRECT_MODEL_ID="$($PYTHON_BIN - "$CONFIG_PATH" "$DIRECT_TRACK" <<'PY2'
import json
import sys
from pathlib import Path
cfg_path = Path(sys.argv[1])
track = str(sys.argv[2] or '').strip()
payload = json.loads(cfg_path.read_text(encoding='utf-8'))
for model in payload.get('models', []):
    if not isinstance(model, dict):
        continue
    if str(model.get('track') or '') != track:
        continue
    if model.get('provider') != 'api_over_mcp':
        continue
    model_id = str(model.get('id') or '').strip()
    if model_id:
        print(model_id)
        break
PY2
)"
fi

if [ -z "$DIRECT_MODEL_ID" ]; then
  echo "[FAIL] no direct API model found in $CONFIG_PATH (track=$DIRECT_TRACK provider=api_over_mcp)" >&2
  exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "[FAIL] direct API track requires OPENAI_API_KEY or ANTHROPIC_API_KEY" >&2
  exit 1
fi

echo "[INFO] comparison forms=$FORM_IDS run_indexes=$RUN_INDEXES"
echo "[INFO] mediated_experiment_id=$MEDIATED_EXPERIMENT_ID"
echo "[INFO] direct_experiment_id=$DIRECT_EXPERIMENT_ID"
echo "[INFO] direct_model_id=$DIRECT_MODEL_ID"
echo "[INFO] mediated_budget_profile=$MEDIATED_BUDGET_PROFILE"
echo "[INFO] resource_profile=$RESOURCE_PROFILE inference_backend=$INFERENCE_BACKEND api_timeout_s=$API_TIMEOUT_S"
echo "[INFO] mediated_budgets max_steps=$MEDIATED_MAX_STEPS timeout_s=$MEDIATED_TIMEOUT_S max_new_tokens=$MEDIATED_MAX_NEW_TOKENS step_soft_timeout_s=$MEDIATED_STEP_SOFT_TIMEOUT_S step_retry_max_new_tokens=$MEDIATED_STEP_RETRY_MAX_NEW_TOKENS compact_page_text_max_chars=$MEDIATED_COMPACT_PAGE_TEXT_MAX_CHARS browser_mcp_timeout_ms=$MEDIATED_BROWSER_MCP_TIMEOUT_MS"
echo "[INFO] mediated_control_level=$MEDIATED_CONTROL_LEVEL interaction_protocol=$INTERACTION_PROTOCOL observation_mode=$OBSERVATION_MODE scoring_mode=$SCORING_MODE"

mediated_status=0
if ! EXPERIMENT_ID="$MEDIATED_EXPERIMENT_ID" \
  CONFIG_PATH="$CONFIG_PATH" \
  TRACK_FILTER="mediated" \
  FORM_IDS="$FORM_IDS" \
  RUN_INDEXES="$RUN_INDEXES" \
  RESOURCE_PROFILE="$RESOURCE_PROFILE" \
  INFERENCE_BACKEND="$INFERENCE_BACKEND" \
  API_TIMEOUT_S="$API_TIMEOUT_S" \
  BROWSER_INIT_RETRIES="$BROWSER_INIT_RETRIES" \
  BROWSER_INIT_RETRY_DELAY_S="$BROWSER_INIT_RETRY_DELAY_S" \
  MAX_STEPS="$MEDIATED_MAX_STEPS" \
  TIMEOUT_S="$MEDIATED_TIMEOUT_S" \
  MAX_NEW_TOKENS="$MEDIATED_MAX_NEW_TOKENS" \
  STEP_SOFT_TIMEOUT_S="$MEDIATED_STEP_SOFT_TIMEOUT_S" \
  STEP_RETRY_MAX_NEW_TOKENS="$MEDIATED_STEP_RETRY_MAX_NEW_TOKENS" \
  COMPACT_PAGE_TEXT_MAX_CHARS="$MEDIATED_COMPACT_PAGE_TEXT_MAX_CHARS" \
  BROWSER_MCP_TIMEOUT_MS="$MEDIATED_BROWSER_MCP_TIMEOUT_MS" \
  PROMPT_PROFILE="$PROMPT_PROFILE" \
  CONTROL_LEVEL="$MEDIATED_CONTROL_LEVEL" \
  INTERACTION_PROTOCOL="$INTERACTION_PROTOCOL" \
  OBSERVATION_MODE="$OBSERVATION_MODE" \
  SCORING_MODE="$SCORING_MODE" \
  RETENTION_WINDOW="$RETENTION_WINDOW" \
  DISABLE_ACTION_COERCION=1 \
  bash scripts/run_model_baseline_matrix.sh; then
  mediated_status=1
  echo "[WARN] mediated track finished with failures" >&2
fi

make_trial_id() {
  "$PYTHON_BIN" - <<'PY3'
from datetime import datetime
print("trial_" + datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ"))
PY3
}

make_run_label() {
  "$PYTHON_BIN" - <<'PY4'
import os
import re
from datetime import datetime
job_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(os.environ.get("SLURM_JOB_ID") or "na").strip()) or "na"
print(datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + f"_job{job_id}")
PY4
}

summary_path_for() {
  local form_id="$1"
  local run_index="$2"
  local trial_id="$3"
  printf '%s/data/model_baselines/%s/%s/%s/run_%04d/%s/summary.json\n' \
    "$ROOT_DIR" "$DIRECT_EXPERIMENT_ID" "$DIRECT_MODEL_ID" "$form_id" "$run_index" "$trial_id"
}

direct_total=0
direct_passed=0
direct_failed=0
declare -a direct_failed_summaries=()

IFS=',' read -r -a FORMS <<<"$FORM_IDS"
IFS=',' read -r -a RUN_INDEX_LIST <<<"$RUN_INDEXES"
for form_id in "${FORMS[@]}"; do
  for run_idx in "${RUN_INDEX_LIST[@]}"; do
    trial_id="$(make_trial_id)"
    run_label="$(make_run_label)"
    summary_path="$(summary_path_for "$form_id" "$run_idx" "$trial_id")"
    direct_total=$((direct_total + 1))
    echo "[INFO] direct_eval model_id=${DIRECT_MODEL_ID} form_id=${form_id} run_index=${run_idx} trial_id=${trial_id}"

    if "$PYTHON_BIN" src/baselines/run_direct_api_eval.py \
      --config "$CONFIG_PATH" \
      --model-id "$DIRECT_MODEL_ID" \
      --form-id "$form_id" \
      --run-index "$run_idx" \
      --trial-id "$trial_id" \
      --experiment-id "$DIRECT_EXPERIMENT_ID" \
      --provider "$DIRECT_PROVIDER" \
      --api-timeout-s "$DIRECT_API_TIMEOUT_S" \
      --execution-backend mcp_server \
      --headless \
      --max-new-tokens "$DIRECT_MAX_NEW_TOKENS" \
      --max-steps "$DIRECT_MAX_STEPS" \
      --timeout-s "$DIRECT_TIMEOUT_S" \
      --compact-page-text-max-chars "$MEDIATED_COMPACT_PAGE_TEXT_MAX_CHARS" \
      --browser-mcp-timeout-ms "$MEDIATED_BROWSER_MCP_TIMEOUT_MS" \
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

"$PYTHON_BIN" scripts/summarize_comparison.py \
  --mediated-experiment-id "$MEDIATED_EXPERIMENT_ID" \
  --direct-experiment-id "$DIRECT_EXPERIMENT_ID" \
  --output "$COMPARISON_OUTPUT"

"$PYTHON_BIN" scripts/summarize_human_ui_attribution.py \
  --experiment-id "$MEDIATED_EXPERIMENT_ID" \
  --interaction-protocol "$INTERACTION_PROTOCOL" \
  --output "logs/${MEDIATED_EXPERIMENT_ID}_human_ui_attribution.json" || true

overall_status=0
if [ "$mediated_status" -ne 0 ] || [ "$direct_failed" -gt 0 ]; then
  overall_status=1
fi
exit "$overall_status"
