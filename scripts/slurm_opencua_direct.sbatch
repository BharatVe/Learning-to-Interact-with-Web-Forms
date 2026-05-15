#!/usr/bin/env bash
#SBATCH --job-name=opencua-direct
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=24:00:00

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"
mkdir -p logs/slurm

module load release/25.06 GCCcore/13.3.0 Python/3.12.3 nodejs/20.13.1

export PYTHONUNBUFFERED=1
export CONFIG_PATH="${CONFIG_PATH:-configs/baselines/track_baseline_models.json}"
export OPENCUA_VLLM_PORT="${OPENCUA_VLLM_PORT:-8000}"
export OPENCUA_VLLM_HOST="${OPENCUA_VLLM_HOST:-127.0.0.1}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://${OPENCUA_VLLM_HOST}:${OPENCUA_VLLM_PORT}/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENCUA_PYTHON_BIN="${OPENCUA_PYTHON_BIN:-$ROOT_DIR/.venv-opencua/bin/python}"
export PATH="$ROOT_DIR/.venv-opencua/bin:$ROOT_DIR/.node-tools/node_modules/.bin:$PATH"
export CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/.runtime-cache}"
mkdir -p "$CACHE_ROOT"/{xdg,hf,pip,uv,playwright}
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CACHE_ROOT/playwright}"

SERVER_LOG="logs/slurm/opencua-vllm-${SLURM_JOB_ID:-na}.log"
SERVER_PID=""

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[INFO] repo_root=$ROOT_DIR"
echo "[INFO] job_id=${SLURM_JOB_ID:-unknown}"
echo "[INFO] host=$(hostname)"
echo "[INFO] start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[INFO] config_path=${CONFIG_PATH}"
echo "[INFO] forms=${FORM_IDS:-unset} run_indexes=${RUN_INDEXES:-unset}"
echo "[INFO] openai_base_url=${OPENAI_BASE_URL}"

nvidia-smi || true

bash scripts/run_opencua_vllm_server.sh >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[INFO] opencua_vllm_pid=$SERVER_PID"
echo "[INFO] opencua_vllm_log=$SERVER_LOG"

READY_ATTEMPTS="${OPENCUA_READY_ATTEMPTS:-180}"
READY_SLEEP_S="${OPENCUA_READY_SLEEP_S:-10}"
READY_STARTED="$(date +%s)"
for attempt in $(seq 1 "$READY_ATTEMPTS"); do
  if curl -fsS "${OPENAI_BASE_URL}/models" >/dev/null 2>&1; then
    echo "[INFO] vllm_ready attempt=$attempt elapsed_s=$(( $(date +%s) - READY_STARTED ))"
    break
  fi
  if [ $((attempt % 6)) -eq 0 ]; then
    echo "[INFO] waiting_for_opencua_vllm attempt=${attempt}/${READY_ATTEMPTS} elapsed_s=$(( $(date +%s) - READY_STARTED ))"
  fi
  sleep "$READY_SLEEP_S"
  if [ "$attempt" -eq "$READY_ATTEMPTS" ]; then
    echo "[FAIL] OpenCUA vLLM server did not become ready" >&2
    tail -n 120 "$SERVER_LOG" >&2 || true
    exit 1
  fi
done

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" bash scripts/run_opencua_direct_matrix.sh

echo "[INFO] end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
