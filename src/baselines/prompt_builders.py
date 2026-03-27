import json
from pathlib import Path
from typing import Any, Dict, List, Optional


ACTION_SCHEMA_TEXT = json.dumps(
    {
        "action": "click|type|select_option|press_key|scroll|wait|submit|done",
        "target": {
            "question_id": "preferred exact id from remaining answers",
            "label": "question label when helpful",
            "text": "visible question text when helpful",
            "selector_hint": "optional backup hint",
        },
        "value": "optional string",
        "delta": "optional integer",
        "reason": "short optional string",
    },
    indent=2,
)

PROMPT_PROFILES = {"legacy", "detailed_v1", "runtime_safe_v1"}
CONTEXT_PACKAGE_VERSION = "context_package.v1"

WIDGET_ACTION_POLICY_TEXT = json.dumps(
    {
        "short_text": ["type"],
        "paragraph_text": ["type"],
        "date": ["type"],
        "time": ["type"],
        "single_choice": ["select_option", "click"],
        "multi_choice": ["select_option", "click"],
        "global_actions": ["wait", "scroll", "press_key", "submit", "done"],
    },
    indent=2,
)

CANONICAL_FEWSHOT_EXAMPLES: List[Dict[str, Any]] = [
    {
        "id": "ex_text_fill",
        "scenario": {
            "remaining_answers": [{"question_id": "q_001", "label": "Full name", "widget_type": "short_text", "value": "Olivia Brooks"}],
            "last_result": {"status": "observed", "error": None, "remaining_answers": 6},
        },
        "output": {
            "action": "type",
            "target": {"question_id": "q_001", "label": "Full name"},
            "value": "Olivia Brooks",
            "reason": "fill required short text",
        },
    },
    {
        "id": "ex_choice_select",
        "scenario": {
            "remaining_answers": [{"question_id": "q_003", "label": "Meal preference", "widget_type": "single_choice", "value": "Vegetarian"}],
            "last_result": {"status": "filled", "error": None, "remaining_answers": 4},
        },
        "output": {
            "action": "select_option",
            "target": {"question_id": "q_003", "label": "Meal preference"},
            "value": "Vegetarian",
            "reason": "select matching radio option",
        },
    },
    {
        "id": "ex_recovery_after_target_not_found",
        "scenario": {
            "remaining_answers": [
                {"question_id": "q_002", "label": "Email", "widget_type": "short_text", "value": "olivia@example.com"},
                {"question_id": "q_004", "label": "Attendance", "widget_type": "single_choice", "value": "Yes"},
            ],
            "last_result": {"status": "failed", "error": "target_not_found", "remaining_answers": 5},
        },
        "output": {
            "action": "type",
            "target": {"question_id": "q_002", "label": "Email"},
            "value": "olivia@example.com",
            "reason": "recover by selecting a different allowed id",
        },
    },
]


def compact_page_text(raw_text: str, max_chars: int = 5000) -> str:
    compact = " ".join(str(raw_text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."


def _remaining_question_ids(remaining_answers: List[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for item in remaining_answers:
        qid = str(item.get("question_id") or "").strip()
        if qid:
            ids.append(qid)
    return ids


def selected_canonical_fewshot_examples(enabled: bool = True, count: int = 3) -> List[Dict[str, Any]]:
    if not enabled or count <= 0:
        return []
    return [dict(item) for item in CANONICAL_FEWSHOT_EXAMPLES[: max(0, int(count))]]


def selected_canonical_fewshot_ids(enabled: bool = True, count: int = 3) -> List[str]:
    return [str(item.get("id") or "") for item in selected_canonical_fewshot_examples(enabled=enabled, count=count)]


def _legacy_shared_instruction_block(
    remaining_answers: List[Dict[str, Any]],
    last_result: Optional[Dict[str, Any]],
    behavior_nudge: Optional[str] = None,
) -> str:
    allowed_ids = _remaining_question_ids(remaining_answers)
    extra = ""
    if str((last_result or {}).get("error") or "") == "target_not_found":
        extra = (
            "Previous step failed with target_not_found. "
            "Pick a DIFFERENT target.question_id from allowed IDs and do not repeat stale IDs.\n"
        )
    if behavior_nudge:
        extra += f"Recovery nudge: {str(behavior_nudge).strip()}\n"
    return (
        "Return exactly one compact JSON object and nothing else.\n"
        "Choose one remaining answer to act on next.\n"
        "Use target.question_id from Remaining answers whenever possible.\n"
        f"Allowed target.question_id values: {json.dumps(allowed_ids, ensure_ascii=True)}\n"
        "Never use a question_id that is not in Allowed target.question_id values.\n"
        "Never choose submit while Remaining answers is non-empty.\n"
        "Use action type for short_text, paragraph_text, date, and time widgets.\n"
        "Use action select_option for single_choice and multi_choice widgets.\n"
        "Use submit only after the remaining answers are already filled or no answerable fields remain on the current form flow.\n"
        "Keep reason short or omit it.\n"
        "Do not explain your reasoning.\n"
        f"{extra}"
    )


def _detailed_instruction_block(
    remaining_answers: List[Dict[str, Any]],
    behavior_nudge: Optional[str] = None,
) -> str:
    allowed_ids = _remaining_question_ids(remaining_answers)
    nudge_block = f"Recovery nudge: {str(behavior_nudge).strip()}\n" if behavior_nudge else ""
    return (
        "Return exactly one compact JSON object and nothing else.\n"
        "Use this exact output schema and key names.\n"
        "Pick one concrete next UI action that advances form completion.\n"
        "Action-target contract:\n"
        f"- Allowed target.question_id values: {json.dumps(allowed_ids, ensure_ascii=True)}\n"
        "- Do not use IDs outside the allowed set.\n"
        "- Use widget-compatible actions only; see Widget action policy.\n"
        "- Do not submit while Remaining answers is non-empty.\n"
        "- Use submit only after all answerable fields are filled or form flow is complete.\n"
        "- Keep reason short (or omit it).\n"
        "- Never include explanations outside JSON.\n"
        f"{nudge_block}"
    )


def _format_examples(examples: List[Dict[str, Any]]) -> str:
    if not examples:
        return "[]"
    return json.dumps(examples, indent=2, ensure_ascii=True)


def _resolve_prompt_profile(prompt_profile: str) -> str:
    profile = str(prompt_profile or "legacy").strip().lower()
    return profile if profile in PROMPT_PROFILES else "legacy"


def build_text_prompt(
    form_url: str,
    remaining_answers: List[Dict[str, Any]],
    page_text: str,
    last_result: Optional[Dict[str, Any]],
    behavior_nudge: Optional[str] = None,
    compact_page_text_max_chars: int = 5000,
    *,
    prompt_profile: str = "legacy",
    visible_field_map: Optional[List[Dict[str, Any]]] = None,
    recent_history: Optional[List[Dict[str, Any]]] = None,
    validation_feedback: Optional[Dict[str, Any]] = None,
    fewshot_enabled: bool = True,
    fewshot_count: int = 3,
) -> str:
    profile = _resolve_prompt_profile(prompt_profile)
    if profile == "legacy":
        return (
            "You are controlling a Google Form filling agent.\n"
            f"{_legacy_shared_instruction_block(remaining_answers, last_result, behavior_nudge)}\n"
            f"Current URL:\n{form_url}\n\n"
            f"Remaining answers:\n{json.dumps(remaining_answers, indent=2, ensure_ascii=True)}\n\n"
            f"Recent action result:\n{json.dumps(last_result or {}, indent=2, ensure_ascii=True)}\n\n"
            f"Current page text:\n{compact_page_text(page_text, max_chars=int(compact_page_text_max_chars))}\n\n"
            f"Action schema:\n{ACTION_SCHEMA_TEXT}\n"
        )

    runtime_safe = profile == "runtime_safe_v1"
    effective_fewshot_count = 1 if runtime_safe else int(fewshot_count)
    examples = selected_canonical_fewshot_examples(enabled=bool(fewshot_enabled), count=effective_fewshot_count)
    history_payload = (recent_history or [])[-2:] if runtime_safe else (recent_history or [])
    profile_name = "runtime_safe_v1" if runtime_safe else "detailed_v1"
    return (
        "You are controlling a Google Form filling agent.\n"
        f"Prompt profile: {profile_name}\n"
        f"{_detailed_instruction_block(remaining_answers, behavior_nudge)}\n"
        f"Context package version: {CONTEXT_PACKAGE_VERSION}\n\n"
        f"Current URL:\n{form_url}\n\n"
        f"Output schema:\n{ACTION_SCHEMA_TEXT}\n\n"
        f"Widget action policy:\n{WIDGET_ACTION_POLICY_TEXT}\n\n"
        f"Canonical few-shot examples:\n{_format_examples(examples)}\n\n"
        f"Remaining answers:\n{json.dumps(remaining_answers, indent=2, ensure_ascii=True)}\n\n"
        f"Visible field map:\n{json.dumps(visible_field_map or remaining_answers, indent=2, ensure_ascii=True)}\n\n"
        f"Recent step history:\n{json.dumps(history_payload, indent=2, ensure_ascii=True)}\n\n"
        f"Validation feedback:\n{json.dumps(validation_feedback or {}, indent=2, ensure_ascii=True)}\n\n"
        f"Recent action result:\n{json.dumps(last_result or {}, indent=2, ensure_ascii=True)}\n\n"
        f"Current page text:\n{compact_page_text(page_text, max_chars=int(compact_page_text_max_chars))}\n\n"
    )


def build_vlm_prompt(
    form_url: str,
    remaining_answers: List[Dict[str, Any]],
    page_text: str,
    last_result: Optional[Dict[str, Any]],
    screenshot_path: Path,
    behavior_nudge: Optional[str] = None,
    compact_page_text_max_chars: int = 5000,
    *,
    prompt_profile: str = "legacy",
    visible_field_map: Optional[List[Dict[str, Any]]] = None,
    recent_history: Optional[List[Dict[str, Any]]] = None,
    validation_feedback: Optional[Dict[str, Any]] = None,
    fewshot_enabled: bool = True,
    fewshot_count: int = 3,
) -> str:
    profile = _resolve_prompt_profile(prompt_profile)
    if profile == "legacy":
        return (
            "You are controlling a Google Form filling agent from a screenshot.\n"
            f"{_legacy_shared_instruction_block(remaining_answers, last_result, behavior_nudge)}\n"
            f"Current URL:\n{form_url}\n\n"
            f"Screenshot path:\n{str(screenshot_path)}\n\n"
            f"Remaining answers:\n{json.dumps(remaining_answers, indent=2, ensure_ascii=True)}\n\n"
            f"Recent action result:\n{json.dumps(last_result or {}, indent=2, ensure_ascii=True)}\n\n"
            f"Compact page text:\n{compact_page_text(page_text, max_chars=int(compact_page_text_max_chars))}\n\n"
            f"Action schema:\n{ACTION_SCHEMA_TEXT}\n"
        )

    runtime_safe = profile == "runtime_safe_v1"
    effective_fewshot_count = 1 if runtime_safe else int(fewshot_count)
    examples = selected_canonical_fewshot_examples(enabled=bool(fewshot_enabled), count=effective_fewshot_count)
    history_payload = (recent_history or [])[-2:] if runtime_safe else (recent_history or [])
    profile_name = "runtime_safe_v1" if runtime_safe else "detailed_v1"
    return (
        "You are controlling a Google Form filling agent from a screenshot.\n"
        f"Prompt profile: {profile_name}\n"
        f"{_detailed_instruction_block(remaining_answers, behavior_nudge)}\n"
        f"Context package version: {CONTEXT_PACKAGE_VERSION}\n\n"
        f"Current URL:\n{form_url}\n\n"
        f"Screenshot path:\n{str(screenshot_path)}\n\n"
        f"Output schema:\n{ACTION_SCHEMA_TEXT}\n\n"
        f"Widget action policy:\n{WIDGET_ACTION_POLICY_TEXT}\n\n"
        f"Canonical few-shot examples:\n{_format_examples(examples)}\n\n"
        f"Remaining answers:\n{json.dumps(remaining_answers, indent=2, ensure_ascii=True)}\n\n"
        f"Visible field map:\n{json.dumps(visible_field_map or remaining_answers, indent=2, ensure_ascii=True)}\n\n"
        f"Recent step history:\n{json.dumps(history_payload, indent=2, ensure_ascii=True)}\n\n"
        f"Validation feedback:\n{json.dumps(validation_feedback or {}, indent=2, ensure_ascii=True)}\n\n"
        f"Recent action result:\n{json.dumps(last_result or {}, indent=2, ensure_ascii=True)}\n\n"
        f"Compact page text:\n{compact_page_text(page_text, max_chars=int(compact_page_text_max_chars))}\n\n"
    )
