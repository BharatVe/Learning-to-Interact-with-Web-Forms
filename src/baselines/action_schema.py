import json
from typing import Any, Dict, List, Tuple


ALLOWED_ACTIONS = {
    "click",
    "type",
    "select_option",
    "press_key",
    "scroll",
    "wait",
    "submit",
    "done",
}


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("empty_model_output")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    starts = [idx for idx, char in enumerate(text) if char == "{"]
    for start in starts:
        depth = 0
        for idx in range(start, len(text)):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start : idx + 1]
                    try:
                        parsed = json.loads(snippet)
                    except Exception:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
    raise ValueError("no_json_object_found")


def parse_action(raw_text: str) -> Dict[str, Any]:
    return _extract_json_object(raw_text)


def validate_action(action: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    if not isinstance(action, dict):
        raise ValueError("action_must_be_object")

    normalized = dict(action)
    warnings: List[str] = []
    action_name = normalized.get("action")
    if not isinstance(action_name, str) or action_name not in ALLOWED_ACTIONS:
        raise ValueError(f"invalid_action: {action_name}")

    target = normalized.get("target")
    if target is None:
        normalized["target"] = {}
    elif not isinstance(target, dict):
        raise ValueError("target_must_be_object")

    for key in list(normalized.keys()):
        if key not in {"action", "target", "value", "delta", "reason"}:
            warnings.append(f"unknown_top_level_key:{key}")

    for key in list(normalized["target"].keys()):
        if key not in {"question_id", "label", "text", "selector_hint"}:
            warnings.append(f"unknown_target_key:{key}")

    if "reason" in normalized and normalized["reason"] is not None and not isinstance(normalized["reason"], str):
        warnings.append("reason_not_string")
        normalized["reason"] = str(normalized["reason"])

    if "delta" in normalized and normalized["delta"] is not None:
        try:
            normalized["delta"] = int(normalized["delta"])
        except Exception as exc:
            raise ValueError(f"invalid_delta: {exc}") from exc

    return normalized, warnings
