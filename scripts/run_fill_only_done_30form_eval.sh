#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FORMS="alumni_checkin,bug_report,conference_travel,course_enrollment,course_feedback,data_annotation,dataset_request,event_rsvp,exam_registration,hackathon_signup,housing_preference,job_fair,language_exchange,library_membership,meal_plan,newsletter_signup,office_hours,orientation_signup,paper_review,project_update,publication_submission,room_booking,scholarship_interest,sports_tournament,study_group_match,survey_consent,technical_support,usability_test,volunteer_shift,workshop_signup"
RUN_INDEXES="${RUN_INDEXES:-2}"
GEMINI_MAX_STEPS="${GEMINI_MAX_STEPS:-32}"
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
  echo "[INFO] running Gemini fill-only/DONE 30-form condition"
  DIRECT_PROVIDER=gemini_low_cost \
  FORM_IDS="$FORMS" \
  RUN_INDEXES="$RUN_INDEXES" \
  DIRECT_MAX_STEPS="$GEMINI_MAX_STEPS" \
  DIRECT_EXPERIMENT_ID="gemini_35_flash_fill_only_done_30_seed20260709_r2_step32" \
  FILL_ONLY=1 \
  INCLUDE_CONTROLS=0 \
  FAIL_ON_TRIAL_FAILURE="${FAIL_ON_TRIAL_FAILURE:-0}" \
  GEMINI_MAX_INFER_RETRIES="${GEMINI_MAX_INFER_RETRIES:-8}" \
  GEMINI_RETRY_DELAY_S="${GEMINI_RETRY_DELAY_S:-120}" \
  GEMINI_RETRY_BACKOFF="${GEMINI_RETRY_BACKOFF:-1.5}" \
  GEMINI_RETRY_MAX_DELAY_S="${GEMINI_RETRY_MAX_DELAY_S:-900}" \
  DIRECT_API_TIMEOUT_S="${DIRECT_API_TIMEOUT_S:-300}" \
  bash scripts/run_gemini_low_cost_matrix.sh
}

run_qwen() {
  echo "[INFO] running Qwen direct-MCP fill-only/DONE 30-form condition"
  FORM_IDS="$FORMS" \
  RUN_INDEXES="$RUN_INDEXES" \
  DIRECT_MCP_MAX_STEPS="$DIRECT_MCP_MAX_STEPS" \
  EXPERIMENT_ID="qwen_direct_mcp_fill_only_done_30_seed20260709_r2_step32" \
  FILL_ONLY_DONE=1 \
  SKIP_COMPLETED="${SKIP_COMPLETED:-0}" \
  FAIL_ON_TRIAL_FAILURE="${FAIL_ON_TRIAL_FAILURE:-0}" \
  bash scripts/run_qwen_direct_mcp_matrix.sh
}

run_opencua_direct_mcp() {
  echo "[INFO] running OpenCUA direct-MCP fill-only/DONE 30-form condition"
  FORM_IDS="$FORMS" \
  RUN_INDEXES="$RUN_INDEXES" \
  DIRECT_MCP_MAX_STEPS="$DIRECT_MCP_MAX_STEPS" \
  EXPERIMENT_ID="opencua_direct_mcp_fill_only_done_30_seed20260709_r2_step32" \
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
