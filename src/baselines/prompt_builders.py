import json
from pathlib import Path
from typing import Any, Dict, List, Optional


ACTION_SCHEMA_TEXT = json.dumps(
    {
        "action": "click|type|select_option|press_key|scroll|wait|submit|done",
        "target": {
            "label": "optional string",
            "text": "optional string",
            "selector_hint": "optional string",
        },
        "value": "optional string",
        "delta": "optional integer",
        "reason": "short optional string",
    },
    indent=2,
)


def compact_page_text(raw_text: str, max_chars: int = 3000) -> str:
    compact = " ".join(str(raw_text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."


def build_text_prompt(
    form_url: str,
    remaining_answers: List[Dict[str, Any]],
    page_text: str,
    last_result: Optional[Dict[str, Any]],
) -> str:
    return (
        "You are controlling a Google Form filling agent.\n"
        "Return exactly one JSON object and nothing else.\n"
        "Choose the next best single action using the remaining answers and current page text.\n"
        "Prefer actions type, select_option, submit, or wait.\n"
        "Do not explain your reasoning.\n\n"
        f"Current URL:\n{form_url}\n\n"
        f"Remaining answers:\n{json.dumps(remaining_answers, indent=2, ensure_ascii=True)}\n\n"
        f"Recent action result:\n{json.dumps(last_result or {}, indent=2, ensure_ascii=True)}\n\n"
        f"Current page text:\n{compact_page_text(page_text)}\n\n"
        f"Action schema:\n{ACTION_SCHEMA_TEXT}\n"
    )


def build_vlm_prompt(
    form_url: str,
    remaining_answers: List[Dict[str, Any]],
    page_text: str,
    last_result: Optional[Dict[str, Any]],
    screenshot_path: Path,
) -> str:
    return (
        "You are controlling a Google Form filling agent from a screenshot.\n"
        "Return exactly one JSON object and nothing else.\n"
        "Use the screenshot together with the remaining answers to choose the next single action.\n"
        "Prefer actions type, select_option, submit, or wait.\n"
        "Do not explain your reasoning.\n\n"
        f"Current URL:\n{form_url}\n\n"
        f"Screenshot path:\n{str(screenshot_path)}\n\n"
        f"Remaining answers:\n{json.dumps(remaining_answers, indent=2, ensure_ascii=True)}\n\n"
        f"Recent action result:\n{json.dumps(last_result or {}, indent=2, ensure_ascii=True)}\n\n"
        f"Compact page text:\n{compact_page_text(page_text)}\n\n"
        f"Action schema:\n{ACTION_SCHEMA_TEXT}\n"
    )
