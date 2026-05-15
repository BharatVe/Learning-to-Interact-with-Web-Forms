#!/usr/bin/env bash
#SBATCH --job-name=model-baseline-mcp
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=24:00:00

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

mkdir -p logs/slurm

echo "[WARN] scripts/slurm_baseline_mcp.sbatch is legacy-mediated only; use scripts/slurm_track_baseline.sbatch for thesis-primary track baseline."

export PYTHONUNBUFFERED=1
export RESOURCE_PROFILE="${RESOURCE_PROFILE:-large_qwen3}"
export GPU_COUNT="${GPU_COUNT:-2}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-12}"
export MEDIATED_BUDGET_PROFILE="${MEDIATED_BUDGET_PROFILE:-balanced}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-auto}"
export API_TIMEOUT_S="${API_TIMEOUT_S:-240}"
export BROWSER_INIT_RETRIES="${BROWSER_INIT_RETRIES:-2}"
export BROWSER_INIT_RETRY_DELAY_S="${BROWSER_INIT_RETRY_DELAY_S:-1.5}"
export CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/.runtime-cache}"
mkdir -p "$CACHE_ROOT"/{xdg,hf,pip,uv,playwright}
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CACHE_ROOT/playwright}"

echo "[INFO] repo_root=$ROOT_DIR"
echo "[INFO] job_id=${SLURM_JOB_ID:-unknown}"
echo "[INFO] host=$(hostname)"
echo "[INFO] start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[INFO] resource_profile=${RESOURCE_PROFILE} gpu_count=${GPU_COUNT} cpus_per_task=${CPUS_PER_TASK}"
echo "[INFO] mediated_budget_profile=${MEDIATED_BUDGET_PROFILE} inference_backend=${INFERENCE_BACKEND}"

nvidia-smi || true

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" bash scripts/run_model_baseline_matrix.sh

echo "[INFO] end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
