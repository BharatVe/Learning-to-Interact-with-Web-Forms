#!/usr/bin/env bash
#SBATCH --job-name=baseline-local
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=04:00:00

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs/slurm logs/baseline_eval

source .venv/bin/activate

export PLAYWRIGHT_BROWSERS_PATH="$ROOT_DIR/.playwright-browsers"
export PYTHONUNBUFFERED=1

echo "[INFO] repo_root=$ROOT_DIR"
echo "[INFO] job_id=${SLURM_JOB_ID:-unknown}"
echo "[INFO] host=$(hostname)"
echo "[INFO] start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

nvidia-smi || true

run_step() {
  srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" "$@"
}

run_step python scripts/verify_baseline_integrity.py
run_step python scripts/verify_runtime_setup.py
run_step python scripts/eval_model_baseline_smoke.py --include-kinds text_llm,vlm --exclude-providers api_over_mcp

FAILURES=0

run_eval() {
  local model_id="$1"
  local model_kind="$2"
  local form_id="$3"
  local run_index="$4"
  shift 4

  echo "[INFO] running model_id=${model_id} model_kind=${model_kind} form_id=${form_id} run_index=${run_index}"
  if ! run_step python src/baselines/run_baseline_eval.py \
    --model-id "${model_id}" \
    --model-kind "${model_kind}" \
    --form-id "${form_id}" \
    --run-index "${run_index}" \
    --headless \
    --max-new-tokens 24 \
    --max-steps 12 \
    --timeout-s 600 \
    "$@"; then
    echo "[WARN] baseline eval failed for model_id=${model_id} form_id=${form_id} run_index=${run_index}"
    FAILURES=$((FAILURES + 1))
  fi
}

run_eval text_qwen25_3b_instruct text_llm conf_interest 1
run_eval text_qwen25_3b_instruct text_llm event_rsvp 1
run_eval vlm_qwen25_vl_3b_instruct vlm conf_interest 1
run_eval vlm_qwen25_vl_3b_instruct vlm event_rsvp 1

echo "[INFO] end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[INFO] failures=${FAILURES}"

if [ "$FAILURES" -ne 0 ]; then
  exit 1
fi
