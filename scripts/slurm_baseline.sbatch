#!/usr/bin/env bash
#SBATCH --job-name=form-baseline
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT_DIR"

source .venv/bin/activate
export PLAYWRIGHT_BROWSERS_PATH="$ROOT_DIR/.playwright-browsers"
export PYTHONUNBUFFERED=1

python3 src/engine/runner.py \
  --smoke-test-all-forms \
  --dataset-root data/forms \
  --trace-mode mcp \
  --interaction-mode local \
  --headless \
  --overwrite-existing
