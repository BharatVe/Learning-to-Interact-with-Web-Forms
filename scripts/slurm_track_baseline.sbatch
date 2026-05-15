#!/usr/bin/env bash
#SBATCH --job-name=track-baseline
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
export DIRECT_PROVIDER="${DIRECT_PROVIDER:-opencua_local}"
export BROWSER_INIT_RETRIES="${BROWSER_INIT_RETRIES:-2}"
export BROWSER_INIT_RETRY_DELAY_S="${BROWSER_INIT_RETRY_DELAY_S:-1.5}"
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

if [ -z "${TRACK_REPORT_OUTPUT:-}" ]; then
  export TRACK_REPORT_OUTPUT="logs/track_baseline_summary-${SLURM_JOB_ID:-na}.json"
fi

echo "[INFO] repo_root=$ROOT_DIR"
echo "[INFO] job_id=${SLURM_JOB_ID:-unknown}"
echo "[INFO] host=$(hostname)"
echo "[INFO] start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[INFO] config_path=${CONFIG_PATH}"
echo "[INFO] direct_provider=${DIRECT_PROVIDER}"
echo "[INFO] forms=${FORM_IDS:-unset} run_indexes=${RUN_INDEXES:-unset}"

nvidia-smi || true

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" bash scripts/run_track_baseline_matrix.sh

echo "[INFO] end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
