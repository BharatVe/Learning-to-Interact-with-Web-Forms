#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT="${OPENCUA_VLLM_PORT:-8000}"
HOST="${OPENCUA_VLLM_HOST:-127.0.0.1}"
MODEL_HF_REPO="${OPENCUA_MODEL_HF_REPO:-xlangai/OpenCUA-32B}"
SERVED_MODEL_NAME="${OPENCUA_SERVED_MODEL_NAME:-opencua-32b}"
TP_SIZE="${OPENCUA_TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTIL="${OPENCUA_GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${OPENCUA_MAX_MODEL_LEN:-32768}"
DISABLE_CUSTOM_ALL_REDUCE="${OPENCUA_DISABLE_CUSTOM_ALL_REDUCE:-1}"
VLLM_PYTHON_BIN="${OPENCUA_PYTHON_BIN:-$ROOT_DIR/.venv-opencua/bin/python}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

if [ ! -x "$VLLM_PYTHON_BIN" ]; then
  echo "[FAIL] vllm python interpreter missing: $VLLM_PYTHON_BIN" >&2
  exit 1
fi

echo "[INFO] opencua_vllm_python=$VLLM_PYTHON_BIN"
"$VLLM_PYTHON_BIN" - <<'PY_VLLM_VERSION' || true
import vllm
print(f"[INFO] opencua_vllm_version={getattr(vllm, '__version__', 'unknown')}")
PY_VLLM_VERSION
echo "[INFO] opencua_vllm_env host=$HOST port=$PORT model=$MODEL_HF_REPO served_model=$SERVED_MODEL_NAME tp_size=$TP_SIZE gpu_memory_util=$GPU_MEMORY_UTIL max_model_len=$MAX_MODEL_LEN"
echo "[INFO] opencua_vllm_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

CMD=(
  "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server
  --model "$MODEL_HF_REPO"
  --trust-remote-code
  --tensor-parallel-size "$TP_SIZE"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --gpu-memory-utilization "$GPU_MEMORY_UTIL"
  --max-model-len "$MAX_MODEL_LEN"
)

if [ "$DISABLE_CUSTOM_ALL_REDUCE" = "1" ]; then
  CMD+=(--disable-custom-all-reduce)
fi

echo "[INFO] opencua_vllm_launch nccl_p2p_disable=$NCCL_P2P_DISABLE nccl_ib_disable=$NCCL_IB_DISABLE disable_custom_all_reduce=$DISABLE_CUSTOM_ALL_REDUCE"
printf '[INFO] opencua_vllm_command'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
