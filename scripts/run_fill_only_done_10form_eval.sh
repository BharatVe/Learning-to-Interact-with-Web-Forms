#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FORMS="conference_travel,course_enrollment,course_feedback,exam_registration,internship_app,job_fair,lab_safety,technical_support,travel_reimbursement,usability_test"
RUN_INDEXES="${RUN_INDEXES:-2}"
GEMINI_MAX_STEPS="${GEMINI_MAX_STEPS:-24}"
DIRECT_MCP_MAX_STEPS="${DIRECT_MCP_MAX_STEPS:-32}"
TARGET="${1:-${TARGET:-all}}"

case "$TARGET" in
  all|gemini|qwen|opencua-direct-mcp) ;;
  *)
    echo "[FAIL] target must be one of: all, gemini, qwen, opencua-direct-mcp" >&2
    exit 1
    ;;
esac

run_gemini() {
  echo "[INFO] running Gemini fill-only/DONE 10-form condition"
  FORM_IDS="$FORMS" \
  RUN_INDEXES="$RUN_INDEXES" \
  DIRECT_MAX_STEPS="$GEMINI_MAX_STEPS" \
  DIRECT_EXPERIMENT_ID="gemini_35_flash_fill_only_done_10_seed20260702_r2_step24" \
  FILL_ONLY=1 \
  INCLUDE_CONTROLS=0 \
  FAIL_ON_TRIAL_FAILURE="${FAIL_ON_TRIAL_FAILURE:-0}" \
  bash scripts/run_gemini_low_cost_matrix.sh
}

run_qwen() {
  echo "[INFO] running Qwen direct-MCP fill-only/DONE 10-form condition"
  FORM_IDS="$FORMS" \
  RUN_INDEXES="$RUN_INDEXES" \
  DIRECT_MCP_MAX_STEPS="$DIRECT_MCP_MAX_STEPS" \
  EXPERIMENT_ID="qwen_direct_mcp_fill_only_done_10_seed20260702_r2_step32" \
  FILL_ONLY_DONE=1 \
  SKIP_COMPLETED="${SKIP_COMPLETED:-0}" \
  FAIL_ON_TRIAL_FAILURE="${FAIL_ON_TRIAL_FAILURE:-0}" \
  bash scripts/run_qwen_direct_mcp_matrix.sh
}

run_opencua_direct_mcp() {
  echo "[INFO] running OpenCUA direct-MCP fill-only/DONE 10-form condition"
  FORM_IDS="$FORMS" \
  RUN_INDEXES="$RUN_INDEXES" \
  DIRECT_MCP_MAX_STEPS="$DIRECT_MCP_MAX_STEPS" \
  EXPERIMENT_ID="opencua_direct_mcp_fill_only_done_10_seed20260702_r2_step32" \
  FILL_ONLY_DONE=1 \
  SKIP_COMPLETED="${SKIP_COMPLETED:-0}" \
  FAIL_ON_TRIAL_FAILURE="${FAIL_ON_TRIAL_FAILURE:-0}" \
  bash scripts/run_opencua_direct_mcp_matrix.sh
}

if [ "$TARGET" = "all" ] || [ "$TARGET" = "gemini" ]; then
  run_gemini
fi
if [ "$TARGET" = "all" ] || [ "$TARGET" = "qwen" ]; then
  run_qwen
fi
if [ "$TARGET" = "all" ] || [ "$TARGET" = "opencua-direct-mcp" ]; then
  run_opencua_direct_mcp
fi
