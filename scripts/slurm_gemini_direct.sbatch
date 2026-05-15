#!/usr/bin/env bash
#SBATCH --job-name=gemini-direct
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

mkdir -p logs/slurm

export PYTHONUNBUFFERED=1
export CONFIG_PATH="${CONFIG_PATH:-configs/baselines/track_baseline_models.json}"
export DIRECT_PROVIDER="${DIRECT_PROVIDER:-gemini_native}"

echo "[INFO] repo_root=$ROOT_DIR"
echo "[INFO] job_id=${SLURM_JOB_ID:-unknown}"
echo "[INFO] host=$(hostname)"
echo "[INFO] start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[INFO] config_path=${CONFIG_PATH}"
echo "[INFO] direct_provider=${DIRECT_PROVIDER}"
echo "[INFO] forms=${FORM_IDS:-unset} run_indexes=${RUN_INDEXES:-unset}"

nvidia-smi || true

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" bash scripts/run_gemini_direct_matrix.sh

echo "[INFO] end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
