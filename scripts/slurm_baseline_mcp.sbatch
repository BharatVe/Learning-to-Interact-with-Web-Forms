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
#SBATCH --time=04:00:00

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

mkdir -p logs/slurm

export PYTHONUNBUFFERED=1
export RESOURCE_PROFILE="${RESOURCE_PROFILE:-large_qwen3}"
export GPU_COUNT="${GPU_COUNT:-2}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-12}"
export MEDIATED_BUDGET_PROFILE="${MEDIATED_BUDGET_PROFILE:-large_qwen3}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-auto}"
export API_TIMEOUT_S="${API_TIMEOUT_S:-120}"
export BROWSER_INIT_RETRIES="${BROWSER_INIT_RETRIES:-2}"
export BROWSER_INIT_RETRY_DELAY_S="${BROWSER_INIT_RETRY_DELAY_S:-1.5}"

echo "[INFO] repo_root=$ROOT_DIR"
echo "[INFO] job_id=${SLURM_JOB_ID:-unknown}"
echo "[INFO] host=$(hostname)"
echo "[INFO] start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[INFO] resource_profile=${RESOURCE_PROFILE} gpu_count=${GPU_COUNT} cpus_per_task=${CPUS_PER_TASK}"
echo "[INFO] mediated_budget_profile=${MEDIATED_BUDGET_PROFILE} inference_backend=${INFERENCE_BACKEND}"

nvidia-smi || true

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" bash scripts/run_model_baseline_matrix.sh

echo "[INFO] end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
