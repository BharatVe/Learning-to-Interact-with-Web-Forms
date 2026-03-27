#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[FAIL] missing virtualenv interpreter: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p logs/slurm data/model_baselines

module load release/25.06 GCCcore/13.3.0 nodejs/20.13.1

export PATH="$ROOT_DIR/.node-tools/node_modules/.bin:$PATH"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$ROOT_DIR/.playwright-browsers-node}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

EXPERIMENT_ID="${EXPERIMENT_ID:-baseline_mcp_v1}"
CONFIG_PATH="${CONFIG_PATH:-configs/baselines/minimal_models.json}"
TRACK_FILTER="${TRACK_FILTER:-mediated}"
FORM_IDS="${FORM_IDS:-conf_interest,event_rsvp}"
RUN_INDEX="${RUN_INDEX:-1}"

RESOURCE_PROFILE="${RESOURCE_PROFILE:-default}"
case "$RESOURCE_PROFILE" in
  default)
    PROFILE_GPU_COUNT=1
    PROFILE_CPUS_PER_TASK=6
    ;;
  large_qwen3)
    PROFILE_GPU_COUNT=2
    PROFILE_CPUS_PER_TASK=12
    ;;
  *)
    echo "[FAIL] unsupported RESOURCE_PROFILE=$RESOURCE_PROFILE (expected default|large_qwen3)" >&2
    exit 1
    ;;
esac
GPU_COUNT="${GPU_COUNT:-$PROFILE_GPU_COUNT}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$PROFILE_CPUS_PER_TASK}"

MEDIATED_BUDGET_PROFILE="${MEDIATED_BUDGET_PROFILE:-large_qwen3}"
case "$MEDIATED_BUDGET_PROFILE" in
  balanced)
    PROFILE_MAX_STEPS=36
    PROFILE_TIMEOUT_S=1800
    PROFILE_MAX_NEW_TOKENS=224
    PROFILE_STEP_SOFT_TIMEOUT_S=60
    PROFILE_STEP_RETRY_MAX_NEW_TOKENS=160
    PROFILE_COMPACT_PAGE_TEXT_MAX_CHARS=5000
    PROFILE_BROWSER_MCP_TIMEOUT_MS=120000
    PROFILE_PROMPT_PROFILE="detailed_v1"
    ;;
  large)
    PROFILE_MAX_STEPS=56
    PROFILE_TIMEOUT_S=3600
    PROFILE_MAX_NEW_TOKENS=384
    PROFILE_STEP_SOFT_TIMEOUT_S=180
    PROFILE_STEP_RETRY_MAX_NEW_TOKENS=256
    PROFILE_COMPACT_PAGE_TEXT_MAX_CHARS=9000
    PROFILE_BROWSER_MCP_TIMEOUT_MS=240000
    PROFILE_PROMPT_PROFILE="detailed_v1"
    ;;
  xlarge)
    PROFILE_MAX_STEPS=72
    PROFILE_TIMEOUT_S=5400
    PROFILE_MAX_NEW_TOKENS=512
    PROFILE_STEP_SOFT_TIMEOUT_S=240
    PROFILE_STEP_RETRY_MAX_NEW_TOKENS=320
    PROFILE_COMPACT_PAGE_TEXT_MAX_CHARS=12000
    PROFILE_BROWSER_MCP_TIMEOUT_MS=300000
    PROFILE_PROMPT_PROFILE="detailed_v1"
    ;;
  large_qwen3)
    PROFILE_MAX_STEPS=64
    PROFILE_TIMEOUT_S=5400
    PROFILE_MAX_NEW_TOKENS=320
    PROFILE_STEP_SOFT_TIMEOUT_S=300
    PROFILE_STEP_RETRY_MAX_NEW_TOKENS=192
    PROFILE_COMPACT_PAGE_TEXT_MAX_CHARS=6000
    PROFILE_BROWSER_MCP_TIMEOUT_MS=300000
    PROFILE_PROMPT_PROFILE="runtime_safe_v1"
    ;;
  *)
    echo "[FAIL] unsupported MEDIATED_BUDGET_PROFILE=$MEDIATED_BUDGET_PROFILE (expected balanced|large|xlarge|large_qwen3)" >&2
    exit 1
    ;;
esac

MAX_STEPS="${MAX_STEPS:-$PROFILE_MAX_STEPS}"
TIMEOUT_S="${TIMEOUT_S:-$PROFILE_TIMEOUT_S}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-$PROFILE_MAX_NEW_TOKENS}"
INVALID_ACTION_BUDGET="${INVALID_ACTION_BUDGET:-0}"
STEP_SOFT_TIMEOUT_S="${STEP_SOFT_TIMEOUT_S:-$PROFILE_STEP_SOFT_TIMEOUT_S}"
STEP_RETRY_MAX_NEW_TOKENS="${STEP_RETRY_MAX_NEW_TOKENS:-$PROFILE_STEP_RETRY_MAX_NEW_TOKENS}"
IDLE_STEP_THRESHOLD="${IDLE_STEP_THRESHOLD:-4}"
IDLE_NUDGE_MAX="${IDLE_NUDGE_MAX:-3}"
COMPACT_PAGE_TEXT_MAX_CHARS="${COMPACT_PAGE_TEXT_MAX_CHARS:-$PROFILE_COMPACT_PAGE_TEXT_MAX_CHARS}"
BROWSER_MCP_TIMEOUT_MS="${BROWSER_MCP_TIMEOUT_MS:-$PROFILE_BROWSER_MCP_TIMEOUT_MS}"
PROMPT_PROFILE="${PROMPT_PROFILE:-$PROFILE_PROMPT_PROFILE}"
HISTORY_WINDOW="${HISTORY_WINDOW:-4}"
FEWSHOT_ENABLED="${FEWSHOT_ENABLED:-1}"
FEWSHOT_COUNT="${FEWSHOT_COUNT:-3}"
RETENTION_WINDOW="${RETENTION_WINDOW:-5}"
DISABLE_ACTION_COERCION="${DISABLE_ACTION_COERCION:-1}"
SKIP_RUNTIME_SETUP_CHECK="${SKIP_RUNTIME_SETUP_CHECK:-0}"
SKIP_MODEL_SMOKE="${SKIP_MODEL_SMOKE:-0}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-auto}"
API_TIMEOUT_S="${API_TIMEOUT_S:-120}"
BROWSER_INIT_RETRIES="${BROWSER_INIT_RETRIES:-2}"
BROWSER_INIT_RETRY_DELAY_S="${BROWSER_INIT_RETRY_DELAY_S:-1.5}"

MANIFEST_PATH="$ROOT_DIR/data/model_baselines/$EXPERIMENT_ID/manifest.jsonl"

if ! command -v playwright-mcp >/dev/null 2>&1; then
  echo "[FAIL] playwright-mcp not found on PATH" >&2
  echo "[INFO] expected local install under $ROOT_DIR/.node-tools" >&2
  exit 1
fi

echo "[INFO] experiment_id=$EXPERIMENT_ID"
echo "[INFO] repo_root=$ROOT_DIR"
echo "[INFO] python_bin=$PYTHON_BIN"
echo "[INFO] playwright_browsers_path=$PLAYWRIGHT_BROWSERS_PATH"
echo "[INFO] track_filter=$TRACK_FILTER"
echo "[INFO] forms=$FORM_IDS"
echo "[INFO] resource_profile=$RESOURCE_PROFILE gpu_count=$GPU_COUNT cpus_per_task=$CPUS_PER_TASK"
echo "[INFO] mediated_budget_profile=$MEDIATED_BUDGET_PROFILE"
echo "[INFO] budgets max_steps=$MAX_STEPS timeout_s=$TIMEOUT_S max_new_tokens=$MAX_NEW_TOKENS step_soft_timeout_s=$STEP_SOFT_TIMEOUT_S step_retry_max_new_tokens=$STEP_RETRY_MAX_NEW_TOKENS compact_page_text_max_chars=$COMPACT_PAGE_TEXT_MAX_CHARS browser_mcp_timeout_ms=$BROWSER_MCP_TIMEOUT_MS"
echo "[INFO] prompt_profile=$PROMPT_PROFILE history_window=$HISTORY_WINDOW fewshot_enabled=$FEWSHOT_ENABLED fewshot_count=$FEWSHOT_COUNT"
echo "[INFO] inference_backend=$INFERENCE_BACKEND api_timeout_s=$API_TIMEOUT_S browser_init_retries=$BROWSER_INIT_RETRIES browser_init_retry_delay_s=$BROWSER_INIT_RETRY_DELAY_S"
playwright-mcp --version || true

"$PYTHON_BIN" scripts/verify_baseline_integrity.py
if [ "$SKIP_RUNTIME_SETUP_CHECK" = "1" ]; then
  echo "[INFO] skipping verify_runtime_setup.py (SKIP_RUNTIME_SETUP_CHECK=1)"
else
  "$PYTHON_BIN" scripts/verify_runtime_setup.py --skip-playwright-smoke
fi
if [ "$SKIP_MODEL_SMOKE" = "1" ]; then
  echo "[INFO] skipping eval_model_baseline_smoke.py (SKIP_MODEL_SMOKE=1)"
else
  "$PYTHON_BIN" scripts/eval_model_baseline_smoke.py --include-kinds text_llm,vlm --exclude-providers api_over_mcp --strict
fi

mapfile -t MODEL_ROWS < <("$PYTHON_BIN" - "$CONFIG_PATH" "$TRACK_FILTER" <<'PY2'
import json
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
track_filter = str(sys.argv[2] or "").strip()
payload = json.loads(cfg_path.read_text(encoding="utf-8"))
for model in payload.get("models", []):
    if not isinstance(model, dict):
        continue
    provider = str(model.get("provider") or "")
    if provider not in {"local_hf", "openai_compat"}:
        continue
    kind = str(model.get("kind") or "")
    if kind not in {"text_llm", "vlm"}:
        continue
    track = str(model.get("track") or "mediated")
    if track_filter and track != track_filter:
        continue
    model_id = str(model.get("id") or "").strip()
    if not model_id:
        continue
    requires_gpu = 1 if bool(model.get("requires_gpu")) else 0
    is_fallback = 1 if bool(model.get("is_fallback")) else 0
    fallback_for = str(model.get("fallback_for") or "").strip()
    print(f"{model_id}|{kind}|{requires_gpu}|{provider}|{is_fallback}|{fallback_for}")
PY2
)

if [ "${#MODEL_ROWS[@]}" -eq 0 ]; then
  echo "[FAIL] no mediated models selected from $CONFIG_PATH (track=$TRACK_FILTER)" >&2
  exit 1
fi

NEEDS_GPU=0
for row in "${MODEL_ROWS[@]}"; do
  IFS='|' read -r _mid _mkind req_gpu _provider _is_fallback _fallback_for <<<"$row"
  if [ "$req_gpu" = "1" ]; then
    NEEDS_GPU=1
    break
  fi
done
if [ "$NEEDS_GPU" = "1" ]; then
  "$PYTHON_BIN" - <<'PY3'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[FAIL] GPU required by selected model set, but CUDA is unavailable")
print("[INFO] CUDA is available for GPU-required model set")
PY3
fi

declare -A FALLBACK_ROWS=()
PRIMARY_ROWS=()
for row in "${MODEL_ROWS[@]}"; do
  IFS='|' read -r model_id _kind _req_gpu _provider is_fallback fallback_for <<<"$row"
  if [ "$is_fallback" = "1" ] && [ -n "$fallback_for" ]; then
    FALLBACK_ROWS["$fallback_for"]="$row"
  else
    PRIMARY_ROWS+=("$row")
  fi
done

if [ "${#PRIMARY_ROWS[@]}" -eq 0 ]; then
  echo "[FAIL] no primary models to run after fallback filtering" >&2
  exit 1
fi

TOTAL=0
PASSED=0
FAILED=0
FAILED_SUMMARIES=()
RUN_EVAL_FAILURE_CATEGORY=""
RUN_EVAL_FAILURE_DETAIL=""

make_trial_id() {
  "$PYTHON_BIN" - <<'PY4'
from datetime import datetime
print("trial_" + datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ"))
PY4
}

make_run_label() {
  "$PYTHON_BIN" - <<'PY5'
import os
import re
from datetime import datetime
job_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(os.environ.get("SLURM_JOB_ID") or "na").strip()) or "na"
print(datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + f"_job{job_id}")
PY5
}

summary_path_for() {
  local model_id="$1"
  local form_id="$2"
  local run_index="$3"
  local trial_id="$4"
  printf '%s/data/model_baselines/%s/%s/%s/run_%04d/%s/summary.json\n' "$ROOT_DIR" "$EXPERIMENT_ID" "$model_id" "$form_id" "$run_index" "$trial_id"
}

extract_failure_from_summary() {
  local summary_path="$1"
  "$PYTHON_BIN" - "$summary_path" <<'PY6'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("|")
    raise SystemExit(0)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("|")
    raise SystemExit(0)
cat = str(payload.get("failure_category") or "")
detail = str(payload.get("failure_detail") or "")
print(f"{cat}|{detail}")
PY6
}

should_trigger_fallback() {
  local category="$1"
  local detail="$2"
  case "$category" in
    model_inference_failed|environment_error|timeout)
      return 0
      ;;
  esac
  local combined
  combined="$(printf '%s %s' "$category" "$detail" | tr '[:upper:]' '[:lower:]')"
  if [[ "$combined" == *"out of memory"* ]] || [[ "$combined" == *"snapshotforai"* ]] || [[ "$combined" == *"timed out"* ]]; then
    return 0
  fi
  return 1
}

run_eval() {
  local model_id="$1"
  local model_kind="$2"
  local form_id="$3"
  local run_index="$4"
  local requires_gpu="$5"
  local provider="$6"
  local is_fallback_model="$7"
  local fallback_for="$8"

  local trial_id
  local run_label
  local summary_path

  trial_id="$(make_trial_id)"
  run_label="$(make_run_label)"
  summary_path="$(summary_path_for "$model_id" "$form_id" "$run_index" "$trial_id")"
  RUN_EVAL_FAILURE_CATEGORY=""
  RUN_EVAL_FAILURE_DETAIL=""
  TOTAL=$((TOTAL + 1))

  echo "[INFO] baseline_eval model_id=${model_id} model_kind=${model_kind} provider=${provider} form_id=${form_id} run_index=${run_index} trial_id=${trial_id} backend=mcp_server is_fallback_model=${is_fallback_model} fallback_for=${fallback_for}"
  EXTRA_ARGS=()
  if [ "$requires_gpu" = "1" ]; then
    EXTRA_ARGS+=(--require-gpu)
  fi
  if [ "$DISABLE_ACTION_COERCION" = "1" ]; then
    EXTRA_ARGS+=(--disable-action-coercion)
  else
    EXTRA_ARGS+=(--enable-action-coercion)
  fi
  if [ "$FEWSHOT_ENABLED" = "1" ]; then
    EXTRA_ARGS+=(--fewshot-enabled)
  else
    EXTRA_ARGS+=(--no-fewshot-enabled)
  fi

  if "$PYTHON_BIN" src/baselines/run_baseline_eval.py \
    --config "$CONFIG_PATH" \
    --model-id "$model_id" \
    --model-kind "$model_kind" \
    --form-id "$form_id" \
    --run-index "$run_index" \
    --trial-id "$trial_id" \
    --experiment-id "$EXPERIMENT_ID" \
    --execution-backend mcp_server \
    --inference-backend "$INFERENCE_BACKEND" \
    --api-timeout-s "$API_TIMEOUT_S" \
    --headless \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-steps "$MAX_STEPS" \
    --timeout-s "$TIMEOUT_S" \
    --invalid-action-budget "$INVALID_ACTION_BUDGET" \
    --step-soft-timeout-s "$STEP_SOFT_TIMEOUT_S" \
    --step-retry-max-new-tokens "$STEP_RETRY_MAX_NEW_TOKENS" \
    --idle-step-threshold "$IDLE_STEP_THRESHOLD" \
    --idle-nudge-max "$IDLE_NUDGE_MAX" \
    --compact-page-text-max-chars "$COMPACT_PAGE_TEXT_MAX_CHARS" \
    --prompt-profile "$PROMPT_PROFILE" \
    --history-window "$HISTORY_WINDOW" \
    --fewshot-count "$FEWSHOT_COUNT" \
    --browser-mcp-timeout-ms "$BROWSER_MCP_TIMEOUT_MS" \
    --browser-init-retries "$BROWSER_INIT_RETRIES" \
    --browser-init-retry-delay-s "$BROWSER_INIT_RETRY_DELAY_S" \
    --retention-window "$RETENTION_WINDOW" \
    --run-label "$run_label" \
    "${EXTRA_ARGS[@]}"; then
    PASSED=$((PASSED + 1))
    return 0
  fi

  echo "[WARN] baseline eval failed for model_id=${model_id} form_id=${form_id} run_index=${run_index}" >&2
  FAILED=$((FAILED + 1))
  FAILED_SUMMARIES+=("$summary_path")
  IFS='|' read -r RUN_EVAL_FAILURE_CATEGORY RUN_EVAL_FAILURE_DETAIL <<<"$(extract_failure_from_summary "$summary_path")"
  echo "[WARN] failure_category=${RUN_EVAL_FAILURE_CATEGORY}"
  return 1
}

IFS=',' read -r -a FORMS <<<"$FORM_IDS"
for row in "${PRIMARY_ROWS[@]}"; do
  IFS='|' read -r model_id model_kind requires_gpu provider _is_fallback _fallback_for <<<"$row"
  for form_id in "${FORMS[@]}"; do
    if run_eval "$model_id" "$model_kind" "$form_id" "$RUN_INDEX" "$requires_gpu" "$provider" "0" ""; then
      continue
    fi
    fallback_row="${FALLBACK_ROWS[$model_id]:-}"
    if [ -n "$fallback_row" ] && [ "$model_kind" = "vlm" ] && should_trigger_fallback "$RUN_EVAL_FAILURE_CATEGORY" "$RUN_EVAL_FAILURE_DETAIL"; then
      IFS='|' read -r fb_model_id fb_kind fb_requires_gpu fb_provider fb_is_fallback fb_fallback_for <<<"$fallback_row"
      echo "[INFO] triggering fallback model=${fb_model_id} for primary_model=${model_id} form_id=${form_id}"
      run_eval "$fb_model_id" "$fb_kind" "$form_id" "$RUN_INDEX" "$fb_requires_gpu" "$fb_provider" "$fb_is_fallback" "$fb_fallback_for" || true
    fi
  done
done

echo "[INFO] baseline_eval_total=$TOTAL"
echo "[INFO] baseline_eval_passed=$PASSED"
echo "[INFO] baseline_eval_failed=$FAILED"
echo "[INFO] baseline_manifest=$MANIFEST_PATH"
if [ "$FAILED" -gt 0 ]; then
  printf '[INFO] failed_summary=%s\n' "${FAILED_SUMMARIES[@]}"
  exit 1
fi
