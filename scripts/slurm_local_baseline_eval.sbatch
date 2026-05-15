#!/usr/bin/env bash
#SBATCH --job-name=baseline-local
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=04:00:00

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

mkdir -p logs/slurm logs/baseline_eval

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[FAIL] missing virtualenv interpreter: $PYTHON_BIN" >&2
  exit 1
fi

export PLAYWRIGHT_BROWSERS_PATH="$ROOT_DIR/.playwright-browsers"
export PYTHONUNBUFFERED=1
INFERENCE_BACKEND="${INFERENCE_BACKEND:-auto}"
API_TIMEOUT_S="${API_TIMEOUT_S:-240}"
BROWSER_INIT_RETRIES="${BROWSER_INIT_RETRIES:-2}"
BROWSER_INIT_RETRY_DELAY_S="${BROWSER_INIT_RETRY_DELAY_S:-1.5}"
RUN_INDEX="${RUN_INDEX:-1}"
RUN_INDEXES="${RUN_INDEXES:-$RUN_INDEX}"
CONFIG_PATH="${CONFIG_PATH:-configs/baselines/minimal_models.json}"

MEDIATED_BUDGET_PROFILE="${MEDIATED_BUDGET_PROFILE:-large_qwen3}"
case "$MEDIATED_BUDGET_PROFILE" in
  balanced)
    PROFILE_MAX_NEW_TOKENS=224
    PROFILE_MAX_STEPS=36
    PROFILE_TIMEOUT_S=1800
    PROFILE_STEP_SOFT_TIMEOUT_S=60
    PROFILE_STEP_RETRY_MAX_NEW_TOKENS=160
    PROFILE_COMPACT_PAGE_TEXT_MAX_CHARS=5000
    PROFILE_BROWSER_MCP_TIMEOUT_MS=120000
    ;;
  large)
    PROFILE_MAX_NEW_TOKENS=384
    PROFILE_MAX_STEPS=56
    PROFILE_TIMEOUT_S=3600
    PROFILE_STEP_SOFT_TIMEOUT_S=180
    PROFILE_STEP_RETRY_MAX_NEW_TOKENS=256
    PROFILE_COMPACT_PAGE_TEXT_MAX_CHARS=9000
    PROFILE_BROWSER_MCP_TIMEOUT_MS=240000
    ;;
  xlarge)
    PROFILE_MAX_NEW_TOKENS=512
    PROFILE_MAX_STEPS=72
    PROFILE_TIMEOUT_S=5400
    PROFILE_STEP_SOFT_TIMEOUT_S=240
    PROFILE_STEP_RETRY_MAX_NEW_TOKENS=320
    PROFILE_COMPACT_PAGE_TEXT_MAX_CHARS=12000
    PROFILE_BROWSER_MCP_TIMEOUT_MS=300000
    PROFILE_PROMPT_PROFILE=detailed_v1
    ;;
  large_qwen3)
    PROFILE_MAX_NEW_TOKENS=640
    PROFILE_MAX_STEPS=128
    PROFILE_TIMEOUT_S=10800
    PROFILE_STEP_SOFT_TIMEOUT_S=600
    PROFILE_STEP_RETRY_MAX_NEW_TOKENS=384
    PROFILE_COMPACT_PAGE_TEXT_MAX_CHARS=12000
    PROFILE_BROWSER_MCP_TIMEOUT_MS=600000
    PROFILE_PROMPT_PROFILE=runtime_safe_v1
    ;;
  *)
    echo "[FAIL] unsupported MEDIATED_BUDGET_PROFILE=$MEDIATED_BUDGET_PROFILE (expected balanced|large|xlarge|large_qwen3)" >&2
    exit 1
    ;;
esac

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-$PROFILE_MAX_NEW_TOKENS}"
MAX_STEPS="${MAX_STEPS:-$PROFILE_MAX_STEPS}"
TIMEOUT_S="${TIMEOUT_S:-$PROFILE_TIMEOUT_S}"
STEP_SOFT_TIMEOUT_S="${STEP_SOFT_TIMEOUT_S:-$PROFILE_STEP_SOFT_TIMEOUT_S}"
STEP_RETRY_MAX_NEW_TOKENS="${STEP_RETRY_MAX_NEW_TOKENS:-$PROFILE_STEP_RETRY_MAX_NEW_TOKENS}"
COMPACT_PAGE_TEXT_MAX_CHARS="${COMPACT_PAGE_TEXT_MAX_CHARS:-$PROFILE_COMPACT_PAGE_TEXT_MAX_CHARS}"
BROWSER_MCP_TIMEOUT_MS="${BROWSER_MCP_TIMEOUT_MS:-$PROFILE_BROWSER_MCP_TIMEOUT_MS}"
PROMPT_PROFILE="${PROMPT_PROFILE:-${PROFILE_PROMPT_PROFILE:-detailed_v1}}"
CONTROL_LEVEL="${CONTROL_LEVEL:-high_level}"
INTERACTION_PROTOCOL="${INTERACTION_PROTOCOL:-human_ui_v1}"
OBSERVATION_MODE="${OBSERVATION_MODE:-vision_coords}"
SCORING_MODE="${SCORING_MODE:-soft_quality_v1}"
RETENTION_WINDOW="${RETENTION_WINDOW:-5}"

case "$CONTROL_LEVEL" in
  high_level|low_level)
    ;;
  *)
    echo "[FAIL] unsupported CONTROL_LEVEL=$CONTROL_LEVEL (expected high_level|low_level)" >&2
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

echo "[INFO] mediated_budget_profile=$MEDIATED_BUDGET_PROFILE"
echo "[INFO] budgets max_steps=$MAX_STEPS timeout_s=$TIMEOUT_S max_new_tokens=$MAX_NEW_TOKENS step_soft_timeout_s=$STEP_SOFT_TIMEOUT_S step_retry_max_new_tokens=$STEP_RETRY_MAX_NEW_TOKENS compact_page_text_max_chars=$COMPACT_PAGE_TEXT_MAX_CHARS browser_mcp_timeout_ms=$BROWSER_MCP_TIMEOUT_MS"
echo "[INFO] prompt_profile=$PROMPT_PROFILE control_level=$CONTROL_LEVEL interaction_protocol=$INTERACTION_PROTOCOL observation_mode=$OBSERVATION_MODE scoring_mode=$SCORING_MODE"
echo "[INFO] run_indexes=$RUN_INDEXES"
echo "[INFO] config_path=$CONFIG_PATH"
echo "[INFO] inference_backend=$INFERENCE_BACKEND api_timeout_s=$API_TIMEOUT_S browser_init_retries=$BROWSER_INIT_RETRIES browser_init_retry_delay_s=$BROWSER_INIT_RETRY_DELAY_S"

echo "[INFO] repo_root=$ROOT_DIR"
echo "[INFO] job_id=${SLURM_JOB_ID:-unknown}"
echo "[INFO] host=$(hostname)"
echo "[INFO] start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

nvidia-smi || true

run_step() {
  srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" "$@"
}

run_step "$PYTHON_BIN" scripts/verify_baseline_integrity.py
run_step "$PYTHON_BIN" scripts/verify_runtime_setup.py --config "$CONFIG_PATH" --playwright-smoke-timeout-s 120
run_step "$PYTHON_BIN" scripts/eval_model_baseline_smoke.py --config "$CONFIG_PATH" --include-kinds text_llm,vlm --exclude-providers api_over_mcp

FAILURES=0

run_eval() {
  local model_id="$1"
  local model_kind="$2"
  local form_id="$3"
  local run_index="$4"
  shift 4

  echo "[INFO] running model_id=${model_id} model_kind=${model_kind} form_id=${form_id} run_index=${run_index}"
  if ! run_step "$PYTHON_BIN" src/baselines/run_baseline_eval.py \
    --config "$CONFIG_PATH" \
    --model-id "${model_id}" \
    --model-kind "${model_kind}" \
    --form-id "${form_id}" \
    --run-index "${run_index}" \
    --headless \
    --inference-backend "$INFERENCE_BACKEND" \
    --api-timeout-s "$API_TIMEOUT_S" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-steps "$MAX_STEPS" \
    --timeout-s "$TIMEOUT_S" \
    --step-soft-timeout-s "$STEP_SOFT_TIMEOUT_S" \
    --step-retry-max-new-tokens "$STEP_RETRY_MAX_NEW_TOKENS" \
    --compact-page-text-max-chars "$COMPACT_PAGE_TEXT_MAX_CHARS" \
    --prompt-profile "$PROMPT_PROFILE" \
    --control-level "$CONTROL_LEVEL" \
    --interaction-protocol "$INTERACTION_PROTOCOL" \
    --observation-mode "$OBSERVATION_MODE" \
    --scoring-mode "$SCORING_MODE" \
    --browser-mcp-timeout-ms "$BROWSER_MCP_TIMEOUT_MS" \
    --browser-init-retries "$BROWSER_INIT_RETRIES" \
    --browser-init-retry-delay-s "$BROWSER_INIT_RETRY_DELAY_S" \
    --retention-window "$RETENTION_WINDOW" \
    "$@"; then
    echo "[WARN] baseline eval failed for model_id=${model_id} form_id=${form_id} run_index=${run_index}"
    FAILURES=$((FAILURES + 1))
  fi
}

IFS=',' read -r -a RUN_INDEX_LIST <<<"$RUN_INDEXES"
for run_idx in "${RUN_INDEX_LIST[@]}"; do
  run_eval text_qwen3_30b_a3b_instruct_2507 text_llm conf_interest "$run_idx"
  run_eval text_qwen3_30b_a3b_instruct_2507_fp8 text_llm conf_interest "$run_idx"
  run_eval text_qwen3_30b_a3b_instruct_2507 text_llm event_rsvp "$run_idx"
  run_eval text_qwen3_30b_a3b_instruct_2507_fp8 text_llm event_rsvp "$run_idx"
  run_eval vlm_qwen3_vl_30b_a3b_instruct vlm conf_interest "$run_idx"
  run_eval vlm_qwen3_vl_30b_a3b_instruct vlm event_rsvp "$run_idx"
done

echo "[INFO] end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[INFO] failures=${FAILURES}"

if [ "$FAILURES" -ne 0 ]; then
  exit 1
fi
