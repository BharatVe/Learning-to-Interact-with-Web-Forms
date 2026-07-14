#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ALL_FORMS="accessibility_feedback,alumni_checkin,bug_report,club_application,club_event_planning,conf_interest,conference_travel,course_enrollment,course_feedback,data_annotation,dataset_request,equipment_checkout,event_rsvp,exam_registration,experiment_booking,field_trip,hackathon_signup,housing_preference,internship_app,job_fair,lab_safety,lab_visit,language_exchange,library_membership,meal_plan,mentor_match,newsletter_signup,office_hours,orientation_signup,paper_review,peer_evaluation,project_update,publication_submission,purchase_request,remote_setup,research_interest,room_booking,scholarship_interest,seminar_proposal,software_access,sports_tournament,study_group_match,survey_consent,technical_support,thesis_meeting,travel_reimbursement,usability_test,volunteer_shift,wellbeing_check,workshop_signup"
NEW_FORMS="accessibility_feedback,club_application,club_event_planning,conf_interest,equipment_checkout,experiment_booking,field_trip,internship_app,lab_safety,lab_visit,mentor_match,peer_evaluation,purchase_request,remote_setup,research_interest,seminar_proposal,software_access,thesis_meeting,travel_reimbursement,wellbeing_check"
GEMINI_FORMS="${NEW_FORMS},alumni_checkin,bug_report,conference_travel,hackathon_signup,orientation_signup,technical_support"
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
  echo "[INFO] running Gemini 50-form completion: 20 new forms plus 6 unusable-attempt retries"
  DIRECT_PROVIDER=gemini_low_cost \
  FORM_IDS="$GEMINI_FORMS" \
  RUN_INDEXES="$RUN_INDEXES" \
  DIRECT_MAX_STEPS="$GEMINI_MAX_STEPS" \
  DIRECT_EXPERIMENT_ID="gemini_35_flash_fill_only_done_50_completion_20260713_r2_step32" \
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
  echo "[INFO] running both Qwen direct-MCP models on all 50 forms"
  FORM_IDS="$ALL_FORMS" \
  RUN_INDEXES="$RUN_INDEXES" \
  DIRECT_MCP_MAX_STEPS="$DIRECT_MCP_MAX_STEPS" \
  EXPERIMENT_ID="qwen_direct_mcp_fill_only_done_50_20260713_r2_step32" \
  FILL_ONLY_DONE=1 \
  SKIP_COMPLETED=0 \
  FAIL_ON_TRIAL_FAILURE="${FAIL_ON_TRIAL_FAILURE:-0}" \
  MEDIATED_VLLM_STARTUP_ATTEMPTS="${MEDIATED_VLLM_STARTUP_ATTEMPTS:-420}" \
  bash scripts/run_qwen_direct_mcp_matrix.sh
}

run_opencua_direct_mcp() {
  echo "[INFO] running OpenCUA direct-MCP on the 20 forms not in the earlier 30-form batch"
  FORM_IDS="$NEW_FORMS" \
  RUN_INDEXES="$RUN_INDEXES" \
  DIRECT_MCP_MAX_STEPS="$DIRECT_MCP_MAX_STEPS" \
  EXPERIMENT_ID="opencua_direct_mcp_fill_only_done_50_topup20_20260713_r2_step32" \
  FILL_ONLY_DONE=1 \
  SKIP_COMPLETED=0 \
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
