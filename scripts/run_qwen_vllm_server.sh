#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT="${QWEN_VLLM_PORT:-8000}"
HOST="${QWEN_VLLM_HOST:-127.0.0.1}"
MODEL_SPEC="${QWEN_MODEL_SPEC:-}"
SERVED_MODEL_NAME="${QWEN_SERVED_MODEL_NAME:-}"
TP_SIZE="${QWEN_TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTIL="${QWEN_GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-32768}"
MODEL_IMPL="${QWEN_MODEL_IMPL:-}"
LIMIT_MM_PER_PROMPT="${QWEN_LIMIT_MM_PER_PROMPT:-}"
MM_ENCODER_TP_MODE="${QWEN_MM_ENCODER_TP_MODE:-}"
TRUST_REMOTE_CODE="${QWEN_TRUST_REMOTE_CODE:-1}"
DISABLE_CUSTOM_ALL_REDUCE="${QWEN_DISABLE_CUSTOM_ALL_REDUCE:-1}"
ENABLE_AUTO_TOOL_CHOICE="${QWEN_ENABLE_AUTO_TOOL_CHOICE:-1}"
TOOL_CALL_PARSER="${QWEN_TOOL_CALL_PARSER:-hermes}"
CHAT_TEMPLATE="${QWEN_CHAT_TEMPLATE:-}"
GENERATION_CONFIG="${QWEN_GENERATION_CONFIG:-vllm}"
VLLM_PYTHON_BIN="${QWEN_PYTHON_BIN:-${OPENCUA_PYTHON_BIN:-$ROOT_DIR/.venv-opencua/bin/python}}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

if [ -z "$MODEL_SPEC" ]; then
  echo "[FAIL] QWEN_MODEL_SPEC is required" >&2
  exit 1
fi
if [ -z "$SERVED_MODEL_NAME" ]; then
  echo "[FAIL] QWEN_SERVED_MODEL_NAME is required" >&2
  exit 1
fi
if [ ! -x "$VLLM_PYTHON_BIN" ]; then
  echo "[FAIL] vllm python interpreter missing: $VLLM_PYTHON_BIN" >&2
  exit 1
fi

CMD=(
  "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server
  --model "$MODEL_SPEC"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTIL"
  --max-model-len "$MAX_MODEL_LEN"
)

if [ -n "$MODEL_IMPL" ]; then
  CMD+=(--model-impl "$MODEL_IMPL")
fi
if [ -n "$LIMIT_MM_PER_PROMPT" ]; then
  CMD+=(--limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT")
fi
if [ -n "$MM_ENCODER_TP_MODE" ]; then
  CMD+=(--mm-encoder-tp-mode "$MM_ENCODER_TP_MODE")
fi
if [ "$TRUST_REMOTE_CODE" = "1" ]; then
  CMD+=(--trust-remote-code)
fi
if [ "$DISABLE_CUSTOM_ALL_REDUCE" = "1" ]; then
  CMD+=(--disable-custom-all-reduce)
fi
if [ "$ENABLE_AUTO_TOOL_CHOICE" = "1" ]; then
  CMD+=(--enable-auto-tool-choice)
  if [ -n "$TOOL_CALL_PARSER" ]; then
    CMD+=(--tool-call-parser "$TOOL_CALL_PARSER")
  fi
fi
if [ -n "$CHAT_TEMPLATE" ]; then
  CMD+=(--chat-template "$CHAT_TEMPLATE")
fi
if [ -n "$GENERATION_CONFIG" ]; then
  CMD+=(--generation-config "$GENERATION_CONFIG")
fi

echo "[INFO] qwen_vllm_launch nccl_p2p_disable=$NCCL_P2P_DISABLE nccl_ib_disable=$NCCL_IB_DISABLE disable_custom_all_reduce=$DISABLE_CUSTOM_ALL_REDUCE enable_auto_tool_choice=$ENABLE_AUTO_TOOL_CHOICE tool_call_parser=${TOOL_CALL_PARSER:-none} model_impl=${MODEL_IMPL:-auto} generation_config=${GENERATION_CONFIG:-model}"

exec "${CMD[@]}"
