#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[FAIL] missing virtualenv interpreter: $PYTHON_BIN" >&2
  exit 1
fi
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-$ROOT_DIR/.venv-opencua/bin/python}"

mkdir -p logs/slurm data/model_baselines

module load release/25.06 GCCcore/13.3.0 nodejs/20.13.1

export PATH="$ROOT_DIR/.venv-opencua/bin:$ROOT_DIR/.node-tools/node_modules/.bin:$PATH"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$ROOT_DIR/.playwright-browsers-node}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

EXPERIMENT_ID="${EXPERIMENT_ID:-baseline_mcp_v1}"
CONFIG_PATH="${CONFIG_PATH:-configs/baselines/minimal_models.json}"
TRACK_FILTER="${TRACK_FILTER:-mediated}"
FORM_IDS="${FORM_IDS:-conf_interest,event_rsvp}"
RUN_INDEX="${RUN_INDEX:-1}"
RUN_INDEXES="${RUN_INDEXES:-$RUN_INDEX}"

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

MEDIATED_BUDGET_PROFILE="${MEDIATED_BUDGET_PROFILE:-balanced}"
case "$MEDIATED_BUDGET_PROFILE" in
  balanced)
    PROFILE_MAX_STEPS=24
    PROFILE_TIMEOUT_S=900
    PROFILE_MAX_NEW_TOKENS=128
    PROFILE_STEP_SOFT_TIMEOUT_S=30
    PROFILE_STEP_RETRY_MAX_NEW_TOKENS=96
    PROFILE_COMPACT_PAGE_TEXT_MAX_CHARS=5000
    PROFILE_BROWSER_MCP_TIMEOUT_MS=180000
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
    PROFILE_MAX_STEPS=128
    PROFILE_TIMEOUT_S=10800
    PROFILE_MAX_NEW_TOKENS=640
    PROFILE_STEP_SOFT_TIMEOUT_S=600
    PROFILE_STEP_RETRY_MAX_NEW_TOKENS=384
    PROFILE_COMPACT_PAGE_TEXT_MAX_CHARS=12000
    PROFILE_BROWSER_MCP_TIMEOUT_MS=600000
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
VERIFICATION_SCOPE="${VERIFICATION_SCOPE:-target_only}"
CONTROL_LEVEL="${CONTROL_LEVEL:-high_level}"
INTERACTION_PROTOCOL="${INTERACTION_PROTOCOL:-human_ui_v1}"
OBSERVATION_MODE="${OBSERVATION_MODE:-vision_coords}"
SCORING_MODE="${SCORING_MODE:-soft_quality_v1}"
HISTORY_WINDOW="${HISTORY_WINDOW:-2}"
FEWSHOT_ENABLED="${FEWSHOT_ENABLED:-0}"
FEWSHOT_COUNT="${FEWSHOT_COUNT:-1}"
RETENTION_WINDOW="${RETENTION_WINDOW:-5}"
DISABLE_ACTION_COERCION="${DISABLE_ACTION_COERCION:-1}"
SKIP_RUNTIME_SETUP_CHECK="${SKIP_RUNTIME_SETUP_CHECK:-0}"
SKIP_MODEL_SMOKE="${SKIP_MODEL_SMOKE:-0}"
# When 0, trial-level failures are treated as benchmark outcomes and the job stays successful.
FAIL_ON_TRIAL_FAILURE="${FAIL_ON_TRIAL_FAILURE:-0}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-auto}"
API_TIMEOUT_S="${API_TIMEOUT_S:-240}"
BROWSER_INIT_RETRIES="${BROWSER_INIT_RETRIES:-2}"
BROWSER_INIT_RETRY_DELAY_S="${BROWSER_INIT_RETRY_DELAY_S:-1.5}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
MEDIATED_TEXT_VLLM_PORT="${MEDIATED_TEXT_VLLM_PORT:-8010}"
MEDIATED_VLM_VLLM_PORT="${MEDIATED_VLM_VLLM_PORT:-8011}"
MEDIATED_VLLM_HOST="${MEDIATED_VLLM_HOST:-127.0.0.1}"
MEDIATED_VLLM_STARTUP_ATTEMPTS="${MEDIATED_VLLM_STARTUP_ATTEMPTS:-180}"
MEDIATED_VLLM_STARTUP_SLEEP_S="${MEDIATED_VLLM_STARTUP_SLEEP_S:-10}"
MEDIATED_VLLM_GPU_MEMORY_UTILIZATION="${MEDIATED_VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
MEDIATED_TEXT_MAX_NEW_TOKENS="${MEDIATED_TEXT_MAX_NEW_TOKENS:-128}"
MEDIATED_VLM_MAX_NEW_TOKENS="${MEDIATED_VLM_MAX_NEW_TOKENS:-160}"

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

if [ "$FORM_IDS" = "all" ]; then
  RESOLVED_FORM_IDS="$("$PYTHON_BIN" - <<'PY_FORMS'
from pathlib import Path
root = Path("src/forms")
form_ids = sorted(entry.name for entry in root.iterdir() if entry.is_dir() and (entry / "spec.json").exists())
print(",".join(form_ids))
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
REQUIRED_RUN_MAX="$("$PYTHON_BIN" - "$RUN_INDEXES" <<'PY_RUNS'
import sys
raw = str(sys.argv[1] if len(sys.argv) > 1 else "").strip()
vals = [int(tok.strip()) for tok in raw.split(",") if tok.strip()]
if not vals:
    raise SystemExit("0")
print(max(vals))
PY_RUNS
)"
if [ -z "$REQUIRED_RUN_MAX" ] || [ "$REQUIRED_RUN_MAX" -le 0 ]; then
  echo "[FAIL] invalid REQUIRED_RUN_MAX derived from RUN_INDEXES=$RUN_INDEXES" >&2
  exit 1
fi

MANIFEST_PATH="$ROOT_DIR/data/model_baselines/$EXPERIMENT_ID/manifest.jsonl"

if ! command -v playwright-mcp >/dev/null 2>&1; then
  echo "[FAIL] playwright-mcp not found on PATH" >&2
  echo "[INFO] expected local install under $ROOT_DIR/.node-tools" >&2
  exit 1
fi

echo "[INFO] experiment_id=$EXPERIMENT_ID"
echo "[INFO] repo_root=$ROOT_DIR"
echo "[INFO] config_path=$CONFIG_PATH"
echo "[INFO] python_bin=$PYTHON_BIN"
echo "[INFO] playwright_browsers_path=$PLAYWRIGHT_BROWSERS_PATH"
echo "[INFO] track_filter=$TRACK_FILTER"
echo "[INFO] forms=$RESOLVED_FORM_IDS"
echo "[INFO] run_indexes=$RUN_INDEXES"
echo "[INFO] resource_profile=$RESOURCE_PROFILE gpu_count=$GPU_COUNT cpus_per_task=$CPUS_PER_TASK"
echo "[INFO] mediated_budget_profile=$MEDIATED_BUDGET_PROFILE"
echo "[INFO] budgets max_steps=$MAX_STEPS timeout_s=$TIMEOUT_S max_new_tokens=$MAX_NEW_TOKENS step_soft_timeout_s=$STEP_SOFT_TIMEOUT_S step_retry_max_new_tokens=$STEP_RETRY_MAX_NEW_TOKENS compact_page_text_max_chars=$COMPACT_PAGE_TEXT_MAX_CHARS browser_mcp_timeout_ms=$BROWSER_MCP_TIMEOUT_MS"
echo "[INFO] prompt_profile=$PROMPT_PROFILE verification_scope=$VERIFICATION_SCOPE control_level=$CONTROL_LEVEL interaction_protocol=$INTERACTION_PROTOCOL observation_mode=$OBSERVATION_MODE scoring_mode=$SCORING_MODE history_window=$HISTORY_WINDOW fewshot_enabled=$FEWSHOT_ENABLED fewshot_count=$FEWSHOT_COUNT"
echo "[INFO] fail_on_trial_failure=$FAIL_ON_TRIAL_FAILURE"
echo "[INFO] inference_backend=$INFERENCE_BACKEND api_timeout_s=$API_TIMEOUT_S browser_init_retries=$BROWSER_INIT_RETRIES browser_init_retry_delay_s=$BROWSER_INIT_RETRY_DELAY_S"
playwright-mcp --version || true

"$PYTHON_BIN" scripts/verify_baseline_integrity.py --min-runs-per-form "$REQUIRED_RUN_MAX"
"$PYTHON_BIN" scripts/validate_answer_sets.py \
  --forms-root src/forms \
  --answers-root data/answers \
  --forms-master data/specs/forms_master.csv \
  --form-ids "$RESOLVED_FORM_IDS" \
  --required-runs "$REQUIRED_RUN_MAX" \
  --required-run-indexes "$RUN_INDEXES" \
  --strict
if [ "$SKIP_RUNTIME_SETUP_CHECK" = "1" ]; then
  echo "[INFO] skipping verify_runtime_setup.py (SKIP_RUNTIME_SETUP_CHECK=1)"
else
  "$PYTHON_BIN" scripts/verify_runtime_setup.py --config "$CONFIG_PATH" --skip-playwright-smoke
fi
if [ "$SKIP_MODEL_SMOKE" = "1" ]; then
  echo "[INFO] skipping eval_model_baseline_smoke.py (SKIP_MODEL_SMOKE=1)"
else
  "$PYTHON_BIN" scripts/eval_model_baseline_smoke.py --config "$CONFIG_PATH" --include-kinds text_llm,vlm --exclude-providers api_over_mcp,openai_compat --strict
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
    hf_repo = str(model.get("hf_repo") or "").strip()
    openai_model = str(model.get("openai_model") or model.get("served_model_name") or "").strip()
    server_backend = str(model.get("server_backend") or "").strip()
    print(f"{model_id}|{kind}|{requires_gpu}|{provider}|{is_fallback}|{fallback_for}|{hf_repo}|{openai_model}|{server_backend}")
PY2
)

if [ "${#MODEL_ROWS[@]}" -eq 0 ]; then
  echo "[FAIL] no mediated models selected from $CONFIG_PATH (track=$TRACK_FILTER)" >&2
  exit 1
fi

OPENAI_SERVER_PID=""
OPENAI_SERVER_LOG=""
OPENAI_SERVER_BASE_URL=""
OPENAI_SERVER_MODEL=""
OPENAI_SERVER_STARTUP_S="0"
OPENAI_SERVER_WARMUP_S="0"
OPENAI_SERVER_BACKEND=""
OPENAI_SERVING_MODE="local_hf_trial_local"

cleanup_openai_server() {
  if [ -n "$OPENAI_SERVER_PID" ] && kill -0 "$OPENAI_SERVER_PID" >/dev/null 2>&1; then
    kill "$OPENAI_SERVER_PID" >/dev/null 2>&1 || true
    wait "$OPENAI_SERVER_PID" 2>/dev/null || true
  fi
  OPENAI_SERVER_PID=""
}

on_exit() {
  cleanup_openai_server
}
trap on_exit EXIT

start_openai_compat_server() {
  local model_id="$1"
  local model_kind="$2"
  local hf_repo="$3"
  local served_model_name="$4"
  local server_backend="$5"

  if ! command -v vllm >/dev/null 2>&1; then
    echo "[FAIL] vllm not found on PATH; persistent openai_compat serving cannot start" >&2
    exit 1
  fi

  cleanup_openai_server

  local port model_impl limit_mm mm_encoder model_spec startup_started attempt sleep_s startup_attempts elapsed
  if [ "$model_kind" = "vlm" ]; then
    port="$MEDIATED_VLM_VLLM_PORT"
    model_impl="${MEDIATED_VLM_MODEL_IMPL:-}"
    limit_mm='{"image":1,"video":0}'
    mm_encoder="${MEDIATED_VLM_MM_ENCODER_TP_MODE:-}"
  else
    port="$MEDIATED_TEXT_VLLM_PORT"
    model_impl=""
    limit_mm=""
    mm_encoder=""
  fi
  model_spec="$ROOT_DIR/models/$model_id"
  if [ ! -d "$model_spec" ]; then
    model_spec="$hf_repo"
  fi
  OPENAI_SERVER_BASE_URL="http://${MEDIATED_VLLM_HOST}:${port}/v1"
  OPENAI_SERVER_MODEL="$served_model_name"
  OPENAI_SERVER_BACKEND="${server_backend:-vllm}"
  OPENAI_SERVING_MODE="openai_compat_persistent"
  OPENAI_SERVER_LOG="logs/slurm/${model_id}-vllm-${SLURM_JOB_ID:-na}.log"
  startup_started="$(date +%s)"
  echo "[INFO] starting_persistent_server model_id=${model_id} model_kind=${model_kind} base_url=${OPENAI_SERVER_BASE_URL} model=${OPENAI_SERVER_MODEL} log=${OPENAI_SERVER_LOG}"
  QWEN_VLLM_PORT="$port" \
    QWEN_VLLM_HOST="$MEDIATED_VLLM_HOST" \
    QWEN_MODEL_SPEC="$model_spec" \
    QWEN_SERVED_MODEL_NAME="$served_model_name" \
    QWEN_TENSOR_PARALLEL_SIZE="$GPU_COUNT" \
    QWEN_GPU_MEMORY_UTILIZATION="$MEDIATED_VLLM_GPU_MEMORY_UTILIZATION" \
    QWEN_MODEL_IMPL="$model_impl" \
    QWEN_LIMIT_MM_PER_PROMPT="$limit_mm" \
    QWEN_MM_ENCODER_TP_MODE="$mm_encoder" \
    bash scripts/run_qwen_vllm_server.sh >"$OPENAI_SERVER_LOG" 2>&1 &
  OPENAI_SERVER_PID=$!
  sleep_s="$MEDIATED_VLLM_STARTUP_SLEEP_S"
  startup_attempts="$MEDIATED_VLLM_STARTUP_ATTEMPTS"
  for attempt in $(seq 1 "$startup_attempts"); do
    if curl -fsS "${OPENAI_SERVER_BASE_URL}/models" >/dev/null 2>&1; then
      elapsed=$(( $(date +%s) - startup_started ))
      OPENAI_SERVER_STARTUP_S="$elapsed"
      echo "[INFO] persistent_server_ready model_id=${model_id} attempt=${attempt} startup_s=${OPENAI_SERVER_STARTUP_S}"
      return 0
    fi
    if ! kill -0 "$OPENAI_SERVER_PID" >/dev/null 2>&1; then
      echo "[FAIL] persistent server exited before readiness model_id=${model_id}" >&2
      tail -n 80 "$OPENAI_SERVER_LOG" >&2 || true
      exit 1
    fi
    if [ $((attempt % 6)) -eq 0 ]; then
      elapsed=$(( $(date +%s) - startup_started ))
      echo "[INFO] waiting_for_persistent_server model_id=${model_id} attempt=${attempt}/${startup_attempts} elapsed_s=${elapsed}"
    fi
    sleep "$sleep_s"
  done
  echo "[FAIL] persistent server did not become ready for model_id=${model_id}" >&2
  tail -n 80 "$OPENAI_SERVER_LOG" >&2 || true
  exit 1
}

warmup_openai_compat_server() {
  local model_kind="$1"
  local warmup_started
  warmup_started="$(date +%s)"
  if [ "$model_kind" = "vlm" ]; then
    "$VLLM_PYTHON_BIN" - "$OPENAI_SERVER_BASE_URL" "$OPENAI_SERVER_MODEL" "$OPENAI_API_KEY" <<'PY_WARMUP'
import json
import sys
import urllib.request
base_url, model, api_key = sys.argv[1:4]
payload = {
    "model": model,
    "max_tokens": 8,
    "temperature": 0,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Return exactly one JSON object."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9s2qW/sAAAAASUVORK5CYII="}},
            ],
        }
    ],
}
req = urllib.request.Request(
    url=base_url.rstrip("/") + "/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    method="POST",
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
)
with urllib.request.urlopen(req, timeout=180) as response:
    response.read()
PY_WARMUP
  else
    "$VLLM_PYTHON_BIN" - "$OPENAI_SERVER_BASE_URL" "$OPENAI_SERVER_MODEL" "$OPENAI_API_KEY" <<'PY_WARMUP'
import json
import sys
import urllib.request
base_url, model, api_key = sys.argv[1:4]
payload = {
    "model": model,
    "max_tokens": 8,
    "temperature": 0,
    "messages": [{"role": "user", "content": "Return exactly one JSON object."}],
}
req = urllib.request.Request(
    url=base_url.rstrip("/") + "/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    method="POST",
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
)
with urllib.request.urlopen(req, timeout=180) as response:
    response.read()
PY_WARMUP
  fi
  OPENAI_SERVER_WARMUP_S="$(( $(date +%s) - warmup_started ))"
  echo "[INFO] persistent_server_warmup_done model=${OPENAI_SERVER_MODEL} warmup_s=${OPENAI_SERVER_WARMUP_S}"
}

NEEDS_GPU=0
for row in "${MODEL_ROWS[@]}"; do
  IFS='|' read -r _mid _mkind req_gpu _provider _is_fallback _fallback_for _hf_repo _openai_model _server_backend <<<"$row"
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
  IFS='|' read -r model_id _kind _req_gpu _provider is_fallback fallback_for _hf_repo _openai_model _server_backend <<<"$row"
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
EXPECTED_MODEL_COUNT="${#PRIMARY_ROWS[@]}"
IFS=',' read -r -a FORMS <<<"$RESOLVED_FORM_IDS"
EXPECTED_FORM_COUNT="${#FORMS[@]}"

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

  local model_max_new_tokens="$MAX_NEW_TOKENS"
  if [ "$provider" = "openai_compat" ]; then
    if [ "$model_kind" = "vlm" ]; then
      model_max_new_tokens="$MEDIATED_VLM_MAX_NEW_TOKENS"
    else
      model_max_new_tokens="$MEDIATED_TEXT_MAX_NEW_TOKENS"
    fi
  fi

  if OPENAI_BASE_URL="$OPENAI_SERVER_BASE_URL" \
    OPENAI_MODEL="$OPENAI_SERVER_MODEL" \
    OPENAI_API_KEY="$OPENAI_API_KEY" \
    BASELINE_SERVING_MODE="$OPENAI_SERVING_MODE" \
    BASELINE_SERVER_BACKEND="$OPENAI_SERVER_BACKEND" \
    BASELINE_SERVER_STARTUP_S="$OPENAI_SERVER_STARTUP_S" \
    BASELINE_SERVER_WARMUP_S="$OPENAI_SERVER_WARMUP_S" \
    BASELINE_SERVER_WARM_STATE="warm" \
    "$PYTHON_BIN" src/baselines/run_baseline_eval.py \
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
    --max-new-tokens "$model_max_new_tokens" \
    --max-steps "$MAX_STEPS" \
    --timeout-s "$TIMEOUT_S" \
    --invalid-action-budget "$INVALID_ACTION_BUDGET" \
    --step-soft-timeout-s "$STEP_SOFT_TIMEOUT_S" \
    --step-retry-max-new-tokens "$STEP_RETRY_MAX_NEW_TOKENS" \
    --idle-step-threshold "$IDLE_STEP_THRESHOLD" \
    --idle-nudge-max "$IDLE_NUDGE_MAX" \
    --compact-page-text-max-chars "$COMPACT_PAGE_TEXT_MAX_CHARS" \
    --prompt-profile "$PROMPT_PROFILE" \
    --control-level "$CONTROL_LEVEL" \
    --interaction-protocol "$INTERACTION_PROTOCOL" \
    --observation-mode "$OBSERVATION_MODE" \
    --scoring-mode "$SCORING_MODE" \
    --verification-scope "$VERIFICATION_SCOPE" \
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

for row in "${PRIMARY_ROWS[@]}"; do
  IFS='|' read -r model_id model_kind requires_gpu provider _is_fallback _fallback_for hf_repo openai_model server_backend <<<"$row"
  if [ "$provider" = "openai_compat" ]; then
    if [ ! -x "$VLLM_PYTHON_BIN" ]; then
      echo "[FAIL] missing vLLM Python interpreter for persistent serving: $VLLM_PYTHON_BIN" >&2
      exit 1
    fi
    start_openai_compat_server "$model_id" "$model_kind" "$hf_repo" "$openai_model" "$server_backend"
    warmup_openai_compat_server "$model_kind"
  else
    cleanup_openai_server
    OPENAI_SERVER_BASE_URL=""
    OPENAI_SERVER_MODEL=""
    OPENAI_SERVER_STARTUP_S="0"
    OPENAI_SERVER_WARMUP_S="0"
    OPENAI_SERVER_BACKEND=""
    OPENAI_SERVING_MODE="local_hf_trial_local"
  fi
  for form_id in "${FORMS[@]}"; do
    for run_idx in "${RUN_INDEX_LIST[@]}"; do
      if run_eval "$model_id" "$model_kind" "$form_id" "$run_idx" "$requires_gpu" "$provider" "0" ""; then
        continue
      fi
      fallback_row="${FALLBACK_ROWS[$model_id]:-}"
      if [ -n "$fallback_row" ] && [ "$model_kind" = "vlm" ] && should_trigger_fallback "$RUN_EVAL_FAILURE_CATEGORY" "$RUN_EVAL_FAILURE_DETAIL"; then
        IFS='|' read -r fb_model_id fb_kind fb_requires_gpu fb_provider fb_is_fallback fb_fallback_for _fb_hf_repo _fb_openai_model _fb_server_backend <<<"$fallback_row"
        echo "[INFO] triggering fallback model=${fb_model_id} for primary_model=${model_id} form_id=${form_id} run_index=${run_idx}"
        run_eval "$fb_model_id" "$fb_kind" "$form_id" "$run_idx" "$fb_requires_gpu" "$fb_provider" "$fb_is_fallback" "$fb_fallback_for" || true
      fi
    done
  done
  cleanup_openai_server
done

echo "[INFO] baseline_eval_total=$TOTAL"
echo "[INFO] baseline_eval_passed=$PASSED"
echo "[INFO] baseline_eval_failed=$FAILED"
echo "[INFO] baseline_manifest=$MANIFEST_PATH"
"$PYTHON_BIN" scripts/summarize_human_ui_attribution.py \
  --experiment-id "$EXPERIMENT_ID" \
  --interaction-protocol "$INTERACTION_PROTOCOL" \
  --expected-forms "$EXPECTED_FORM_COUNT" \
  --expected-runs-per-form "$EXPECTED_RUNS_PER_FORM" \
  --expected-models "$EXPECTED_MODEL_COUNT" \
  --output "logs/${EXPERIMENT_ID}_human_ui_attribution.json" || true
if [ "$FAILED" -gt 0 ]; then
  printf '[INFO] failed_summary=%s\n' "${FAILED_SUMMARIES[@]}"
  if [ "$FAIL_ON_TRIAL_FAILURE" = "1" ]; then
    echo "[FAIL] trial failures detected and FAIL_ON_TRIAL_FAILURE=1"
    exit 1
  fi
  echo "[WARN] trial failures detected; keeping successful job exit for benchmark reporting (set FAIL_ON_TRIAL_FAILURE=1 to fail job)"
fi
