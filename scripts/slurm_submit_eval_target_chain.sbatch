#!/usr/bin/env bash
#SBATCH --job-name=eval-chain-submit
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"
mkdir -p logs/slurm

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[FAIL] missing virtualenv interpreter: $PYTHON_BIN" >&2
  exit 1
fi

CHAIN_RUN_INDEXES="${CHAIN_RUN_INDEXES:-}"
CHAIN_TARGET_TRIALS="${CHAIN_TARGET_TRIALS:-300}"
CHAIN_TRACKS="${CHAIN_TRACKS:-all}"
CHAIN_DEPENDENCY_TYPE="${CHAIN_DEPENDENCY_TYPE:-afterok}"
CHAIN_EXCLUDE="${CHAIN_EXCLUDE:-i8033}"
CHAIN_DATE_STAMP="${CHAIN_DATE_STAMP:-}"

args=(
  scripts/submit_eval_target_chain.py
  --target-trials "$CHAIN_TARGET_TRIALS"
  --tracks "$CHAIN_TRACKS"
  --dependency-type "$CHAIN_DEPENDENCY_TYPE"
  --exclude "$CHAIN_EXCLUDE"
  --submit
)

if [ -n "$CHAIN_RUN_INDEXES" ]; then
  args+=(--run-indexes "$CHAIN_RUN_INDEXES")
fi

if [ -n "$CHAIN_DATE_STAMP" ]; then
  args+=(--date-stamp "$CHAIN_DATE_STAMP")
fi

echo "[INFO] eval_chain_submit_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[INFO] chain_run_indexes=$CHAIN_RUN_INDEXES"
echo "[INFO] chain_target_trials=$CHAIN_TARGET_TRIALS"
echo "[INFO] chain_tracks=$CHAIN_TRACKS"
"$PYTHON_BIN" "${args[@]}"
echo "[INFO] eval_chain_submit_end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
