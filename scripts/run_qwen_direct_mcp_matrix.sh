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
module load release/25.06 GCCcore/13.3.0 Python/3.12.3 nodejs/20.13.1
export NODE_LD_LIBRARY_PATH_FOR_MCP="${LD_LIBRARY_PATH-}"

CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/.runtime-cache}"
mkdir -p "$CACHE_ROOT"/{xdg,hf,pip,uv,playwright}
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CACHE_ROOT/playwright}"
export PATH="$ROOT_DIR/.venv-opencua/bin:$ROOT_DIR/.node-tools/node_modules/.bin:$PATH"
export PYTHONUNBUFFERED=1
QWEN_VLLM_LD_LIBRARY_PATH="${QWEN_VLLM_LD_LIBRARY_PATH:-$NODE_LD_LIBRARY_PATH_FOR_MCP}"
QWEN_CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES:-}"
if [ -n "$QWEN_CUDA_VISIBLE_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES="$QWEN_CUDA_VISIBLE_DEVICES"
  echo "[INFO] qwen_cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
fi

CONFIG_PATH="${CONFIG_PATH:-configs/baselines/track_baseline_models.json}"
EXPERIMENT_ID="${EXPERIMENT_ID:-track_baseline_qwen_direct_mcp_v1}"
FORM_IDS="${FORM_IDS:-conf_interest,event_rsvp}"
RUN_INDEX="${RUN_INDEX:-1}"
RUN_INDEXES="${RUN_INDEXES:-$RUN_INDEX}"
FORM_OFFSET="${FORM_OFFSET:-0}"
FORM_LIMIT="${FORM_LIMIT:-0}"
API_TIMEOUT_S="${API_TIMEOUT_S:-300}"
DIRECT_MCP_TIMEOUT_S="${DIRECT_MCP_TIMEOUT_S:-1800}"
DIRECT_MCP_MAX_STEPS="${DIRECT_MCP_MAX_STEPS:-128}"
DIRECT_MCP_TEXT_MAX_NEW_TOKENS="${DIRECT_MCP_TEXT_MAX_NEW_TOKENS:-1024}"
DIRECT_MCP_VLM_MAX_NEW_TOKENS="${DIRECT_MCP_VLM_MAX_NEW_TOKENS:-1024}"
QWEN_MODEL_IDS="${QWEN_MODEL_IDS:-}"
DIRECT_MCP_HISTORY_TURNS="${DIRECT_MCP_HISTORY_TURNS:-0}"
BROWSER_MCP_TIMEOUT_MS="${BROWSER_MCP_TIMEOUT_MS:-600000}"
FAIL_ON_TRIAL_FAILURE="${FAIL_ON_TRIAL_FAILURE:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
FILL_ONLY_DONE="${FILL_ONLY_DONE:-0}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-logs/${EXPERIMENT_ID}_reference_efficiency_summary.json}"
GPU_COUNT="${GPU_COUNT:-2}"
CUDA_PREFLIGHT_PYTHON_BIN="${CUDA_PREFLIGHT_PYTHON_BIN:-$ROOT_DIR/.venv-opencua/bin/python}"
MEDIATED_VLLM_HOST="${MEDIATED_VLLM_HOST:-127.0.0.1}"
MEDIATED_TEXT_VLLM_PORT="${MEDIATED_TEXT_VLLM_PORT:-8010}"
MEDIATED_VLM_VLLM_PORT="${MEDIATED_VLM_VLLM_PORT:-8011}"
MEDIATED_VLLM_STARTUP_ATTEMPTS="${MEDIATED_VLLM_STARTUP_ATTEMPTS:-420}"
MEDIATED_VLLM_STARTUP_SLEEP_S="${MEDIATED_VLLM_STARTUP_SLEEP_S:-10}"
MEDIATED_VLLM_GPU_MEMORY_UTILIZATION="${MEDIATED_VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

if ! command -v playwright-mcp >/dev/null 2>&1; then
  echo "[FAIL] playwright-mcp not found on PATH" >&2
  exit 1
fi
if ! command -v vllm >/dev/null 2>&1; then
  echo "[FAIL] vllm not found on PATH" >&2
  exit 1
fi

PLAYWRIGHT_MCP_INSTALL_LOG="${PLAYWRIGHT_MCP_INSTALL_LOG:-/tmp/playwright-mcp-install-qwen.log}" \
  "$ROOT_DIR/scripts/ensure_playwright_mcp_runtime.sh"
export PLAYWRIGHT_MCP_CHROMIUM_EXECUTABLE="$(cat "$PLAYWRIGHT_BROWSERS_PATH/.mcp-chromium-executable")"
echo "[INFO] playwright_mcp_chromium_executable=$PLAYWRIGHT_MCP_CHROMIUM_EXECUTABLE"

if [ -n "$ORIGINAL_LD_LIBRARY_PATH" ]; then
  export LD_LIBRARY_PATH="$ORIGINAL_LD_LIBRARY_PATH"
else
  unset LD_LIBRARY_PATH
fi

if [ "$FORM_IDS" = "all" ]; then
  RESOLVED_FORM_IDS="$("$PYTHON_BIN" - "$FORM_OFFSET" "$FORM_LIMIT" <<'PY_FORMS'
import sys
from pathlib import Path
offset = max(0, int(sys.argv[1]))
limit = max(0, int(sys.argv[2]))
root = Path("src/forms")
forms = sorted(entry.name for entry in root.iterdir() if entry.is_dir() and (entry / "spec.json").exists())
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
IFS=',' read -r -a RUN_INDEX_LIST <<<"$RUN_INDEXES"

mapfile -t MODEL_ROWS < <("$PYTHON_BIN" - "$CONFIG_PATH" "$QWEN_MODEL_IDS" <<'PY_MODELS'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
selected = {item.strip() for item in sys.argv[2].split(",") if item.strip()}
for model in cfg.get("models", []):
    if not isinstance(model, dict):
        continue
    if str(model.get("track") or "") != "direct_mcp_tool_use":
        continue
    if str(model.get("provider") or "") != "openai_compat":
        continue
    if str(model.get("kind") or "") not in {"text_llm", "vlm"}:
        continue
    if selected and str(model.get("id") or "").strip() not in selected:
        continue
    print("|".join([
        str(model.get("id") or "").strip(),
        str(model.get("kind") or "").strip(),
        str(model.get("hf_repo") or "").strip(),
        str(model.get("openai_model") or model.get("served_model_name") or "").strip(),
        str(model.get("server_backend") or "").strip(),
    ]))
PY_MODELS
)

if [ "${#MODEL_ROWS[@]}" -eq 0 ]; then
  echo "[FAIL] no direct_mcp_tool_use models found in $CONFIG_PATH" >&2
  exit 1
fi

SERVER_PID=""
SERVER_LOG=""
OPENAI_BASE_URL=""
OPENAI_MODEL=""

cuda_preflight() {
  if [ ! -x "$CUDA_PREFLIGHT_PYTHON_BIN" ]; then
    echo "[FAIL] cuda preflight interpreter missing: $CUDA_PREFLIGHT_PYTHON_BIN" >&2
    exit 1
  fi
  LD_LIBRARY_PATH="$QWEN_VLLM_LD_LIBRARY_PATH" "$CUDA_PREFLIGHT_PYTHON_BIN" - "$GPU_COUNT" <<'PY_CUDA'
import sys
expected = max(1, int(sys.argv[1]))
try:
    import torch
except Exception as exc:
    print(f"[FAIL] cuda_preflight_import_torch_failed: {exc}", file=sys.stderr)
    raise SystemExit(1)
count = torch.cuda.device_count()
print(f"[INFO] cuda_preflight torch_cuda_available={torch.cuda.is_available()} device_count={count} expected_min={expected}")
if not torch.cuda.is_available() or count < expected:
    raise SystemExit("[FAIL] cuda_preflight_insufficient_cuda_devices")
for idx in range(count):
    print(f"[INFO] cuda_preflight device_{idx}={torch.cuda.get_device_name(idx)}")
PY_CUDA
}

cleanup_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    # The vLLM API server starts worker children. Terminate the complete
    # process group so a failed readiness check cannot leave an orphaned
    # server holding the Slurm allocation until its wall-time limit.
    kill -TERM -- "-$SERVER_PID" >/dev/null 2>&1 || kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup_server EXIT

start_server() {
  local model_id="$1"
  local model_kind="$2"
  local hf_repo="$3"
  local served_model_name="$4"
  local port model_impl limit_mm mm_encoder model_spec
  cleanup_server
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
  OPENAI_BASE_URL="http://${MEDIATED_VLLM_HOST}:${port}/v1"
  OPENAI_MODEL="$served_model_name"
  SERVER_LOG="logs/slurm/${model_id}-direct-mcp-vllm-${SLURM_JOB_ID:-na}.log"
  echo "[INFO] starting_qwen_direct_mcp_server model_id=${model_id} model_kind=${model_kind} base_url=${OPENAI_BASE_URL}"
  cuda_preflight
  setsid env \
    QWEN_VLLM_PORT="$port" \
    QWEN_VLLM_HOST="$MEDIATED_VLLM_HOST" \
    QWEN_MODEL_SPEC="$model_spec" \
    QWEN_SERVED_MODEL_NAME="$served_model_name" \
    QWEN_TENSOR_PARALLEL_SIZE="$GPU_COUNT" \
    QWEN_GPU_MEMORY_UTILIZATION="$MEDIATED_VLLM_GPU_MEMORY_UTILIZATION" \
    QWEN_MODEL_IMPL="$model_impl" \
    QWEN_LIMIT_MM_PER_PROMPT="$limit_mm" \
    QWEN_MM_ENCODER_TP_MODE="$mm_encoder" \
    LD_LIBRARY_PATH="$QWEN_VLLM_LD_LIBRARY_PATH" bash scripts/run_qwen_vllm_server.sh >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  for attempt in $(seq 1 "$MEDIATED_VLLM_STARTUP_ATTEMPTS"); do
    if curl -fsS "${OPENAI_BASE_URL}/models" >/dev/null 2>&1; then
      echo "[INFO] qwen_direct_mcp_server_ready model_id=${model_id} attempt=${attempt}"
      return 0
    fi
    sleep "$MEDIATED_VLLM_STARTUP_SLEEP_S"
  done
  echo "[FAIL] qwen direct MCP server did not become ready for ${model_id}" >&2
  tail -n 120 "$SERVER_LOG" >&2 || true
  exit 1
}

warmup_server() {
  "$PYTHON_BIN" - "$OPENAI_BASE_URL" "$OPENAI_MODEL" "$OPENAI_API_KEY" <<'PY_WARM'
import json, sys, urllib.request
base_url, model, api_key = sys.argv[1:4]
payload = {
    "model": model,
    "temperature": 0,
    "max_tokens": 8,
    "messages": [{"role": "user", "content": "OK"}],
}
req = urllib.request.Request(
    url=base_url.rstrip("/") + "/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    method="POST",
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
)
with urllib.request.urlopen(req, timeout=180) as response:
    response.read()
PY_WARM
}

trial_summary_glob() {
  local model_id="$1"
  local form_id="$2"
  local run_idx="$3"
  local answer_run_id
  answer_run_id="$(printf 'run_%04d' "$run_idx")"
  printf '%s/data/model_baselines/%s/%s/%s/%s/*/summary.json' "$ROOT_DIR" "$EXPERIMENT_ID" "$model_id" "$form_id" "$answer_run_id"
}

trial_completed() {
  local model_id="$1"
  local form_id="$2"
  local run_idx="$3"
  local glob
  local answer_run_id
  answer_run_id="$(printf 'run_%04d' "$run_idx")"
  glob="$ROOT_DIR/data/model_baselines/*/$model_id/$form_id/$answer_run_id/*/summary.json"
  if compgen -G "$glob" >/dev/null; then
    return 0
  fi
  glob="$(trial_summary_glob "$model_id" "$form_id" "$run_idx")"
  compgen -G "$glob" >/dev/null
}

write_progress_summary() {
  echo "[INFO] writing_qwen_direct_mcp_progress_summary output=$SUMMARY_OUTPUT"
  "$PYTHON_BIN" scripts/summarize_reference_efficiency.py \
    --experiment-id "$EXPERIMENT_ID" \
    --output "$SUMMARY_OUTPUT" || true
  "$PYTHON_BIN" scripts/update_eval_results_tracker.py \
    --experiment-id "$EXPERIMENT_ID" || true
}

echo "[INFO] qwen_direct_mcp_experiment_id=$EXPERIMENT_ID"
echo "[INFO] forms=$RESOLVED_FORM_IDS"
echo "[INFO] run_indexes=$RUN_INDEXES"
echo "[INFO] form_offset=$FORM_OFFSET"
echo "[INFO] form_limit=$FORM_LIMIT"
echo "[INFO] skip_completed=$SKIP_COMPLETED"
echo "[INFO] fill_only_done=$FILL_ONLY_DONE"
echo "[INFO] qwen_model_ids=${QWEN_MODEL_IDS:-all}"
echo "[INFO] direct_mcp_history_turns=$DIRECT_MCP_HISTORY_TURNS"
echo "[INFO] summary_output=$SUMMARY_OUTPUT"

IFS=',' read -r -a FORMS <<<"$RESOLVED_FORM_IDS"
for row in "${MODEL_ROWS[@]}"; do
  IFS='|' read -r model_id model_kind hf_repo served_model_name _server_backend <<<"$row"
  model_has_work=0
  for form_id in "${FORMS[@]}"; do
    for run_idx in "${RUN_INDEX_LIST[@]}"; do
      if [ "$SKIP_COMPLETED" = "1" ] && trial_completed "$model_id" "$form_id" "$run_idx"; then
        continue
      fi
      model_has_work=1
    done
  done
  if [ "$model_has_work" = "0" ]; then
    echo "[INFO] qwen_direct_mcp_model_all_trials_completed model_id=${model_id}; skipping server startup"
    continue
  fi
  start_server "$model_id" "$model_kind" "$hf_repo" "$served_model_name"
  warmup_server
  for form_id in "${FORMS[@]}"; do
    for run_idx in "${RUN_INDEX_LIST[@]}"; do
      if [ "$SKIP_COMPLETED" = "1" ] && trial_completed "$model_id" "$form_id" "$run_idx"; then
        echo "[INFO] qwen_direct_mcp_skip_completed model_id=${model_id} form_id=${form_id} run_index=${run_idx}"
        continue
      fi
      trial_id="$("$PYTHON_BIN" - <<'PY_TRIAL'
from datetime import datetime
print("trial_" + datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ"))
PY_TRIAL
)"
      model_tokens="$DIRECT_MCP_TEXT_MAX_NEW_TOKENS"
      if [ "$model_kind" = "vlm" ]; then
        model_tokens="$DIRECT_MCP_VLM_MAX_NEW_TOKENS"
      fi
      echo "[INFO] qwen_direct_mcp_eval model_id=${model_id} form_id=${form_id} run_index=${run_idx} trial_id=${trial_id}"
      args=(
        --config "$CONFIG_PATH"
        --model-id "$model_id"
        --model-kind "$model_kind"
        --form-id "$form_id"
        --run-index "$run_idx"
        --trial-id "$trial_id"
        --experiment-id "$EXPERIMENT_ID"
        --api-timeout-s "$API_TIMEOUT_S"
        --timeout-s "$DIRECT_MCP_TIMEOUT_S"
        --max-steps "$DIRECT_MCP_MAX_STEPS"
        --max-new-tokens "$model_tokens"
        --history-turns "$DIRECT_MCP_HISTORY_TURNS"
        --browser-mcp-timeout-ms "$BROWSER_MCP_TIMEOUT_MS"
        --headless
      )
      if [ "$FILL_ONLY_DONE" = "1" ]; then
        args+=(--fill-only-done)
      fi
      set +e
      OPENAI_BASE_URL="$OPENAI_BASE_URL" \
        OPENAI_MODEL="$OPENAI_MODEL" \
        OPENAI_API_KEY="$OPENAI_API_KEY" \
        "$PYTHON_BIN" src/baselines/run_qwen_direct_mcp_eval.py "${args[@]}"
      trial_status=$?
      set -e
      if [ "$trial_status" -ne 0 ]; then
        echo "[WARN] qwen_direct_mcp_trial_failed status=$trial_status model_id=${model_id} form_id=${form_id} run_index=${run_idx}"
        if [ "$FAIL_ON_TRIAL_FAILURE" = "1" ]; then
          exit "$trial_status"
        fi
      fi
    done
  done
  cleanup_server
done

write_progress_summary
