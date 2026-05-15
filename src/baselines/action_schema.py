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

LOW_LEVEL_ALLOWED_ACTIONS = {
    "move_mouse",
    "click_mouse",
    "type_text",
    "press_key",
    "scroll",
    "wait",
    "submit",
    "done",
}

DIRECT_BROWSER_TOOLS = {
    "browser_mouse_move_xy",
    "browser_mouse_click_xy",
    "browser_type",
    "browser_press_key",
    "browser_mouse_wheel",
    "browser_wait_for",
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


def _validate_norm_coord(target: Dict[str, Any], key: str) -> int:
    value = target.get(key)
    if not isinstance(value, int):
        raise ValueError(f"invalid_target_{key}: expected integer in [0,999]")
    if value < 0 or value > 999:
        raise ValueError(f"invalid_target_{key}: expected integer in [0,999], got {value}")
    return value


def validate_low_level_action(action: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    if not isinstance(action, dict):
        raise ValueError("action_must_be_object")

    normalized = dict(action)
    warnings: List[str] = []
    tool_name = normalized.get("tool")
    action_name = normalized.get("action")
    direct_tool = isinstance(tool_name, str) and tool_name in DIRECT_BROWSER_TOOLS
    legacy_action = isinstance(action_name, str) and action_name in LOW_LEVEL_ALLOWED_ACTIONS
    if direct_tool:
        normalized["action"] = tool_name
        args = normalized.get("args")
        if args is None:
            normalized["args"] = {}
        elif not isinstance(args, dict):
            raise ValueError("args_must_be_object")
        action_name = tool_name
    elif not legacy_action:
        raise ValueError(f"invalid_low_level_action: {action_name}")

    target = normalized.get("target")
    if target is None:
        normalized["target"] = {}
    elif not isinstance(target, dict):
        raise ValueError("target_must_be_object")

    for key in list(normalized.keys()):
        if key not in {"action", "tool", "args", "target", "value", "delta", "reason"}:
            warnings.append(f"unknown_top_level_key:{key}")

    for key in list(normalized["target"].keys()):
        if key not in {"x", "y", "question_id", "label", "text", "selector_hint"}:
            warnings.append(f"unknown_target_key:{key}")

    args = normalized.get("args") if isinstance(normalized.get("args"), dict) else {}
    if direct_tool:
        for key in list(args.keys()):
            if key not in {"x", "y", "text", "slowly", "submit", "key", "deltaX", "deltaY", "time", "question_id", "label"}:
                warnings.append(f"unknown_args_key:{key}")

    if action_name in {"move_mouse", "click_mouse"}:
        _validate_norm_coord(normalized["target"], "x")
        _validate_norm_coord(normalized["target"], "y")
    elif action_name in {"browser_mouse_move_xy", "browser_mouse_click_xy"}:
        _validate_norm_coord(args, "x")
        _validate_norm_coord(args, "y")

    if action_name == "type_text":
        value = normalized.get("value")
        if not isinstance(value, str) or not value:
            raise ValueError("type_text_requires_non_empty_value")
    elif action_name == "browser_type":
        text = args.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("browser_type_requires_non_empty_args_text")

    if action_name == "press_key":
        value = normalized.get("value")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("press_key_requires_non_empty_value")
        normalized["value"] = value.strip()
    elif action_name == "browser_press_key":
        key = args.get("key")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("browser_press_key_requires_non_empty_args_key")
        args["key"] = key.strip()

    if action_name in {"scroll", "wait"}:
        delta = normalized.get("delta")
        if not isinstance(delta, int):
            raise ValueError(f"{action_name}_requires_integer_delta")
        if action_name == "wait" and delta < 0:
            raise ValueError("wait_requires_non_negative_delta")
    elif action_name == "browser_mouse_wheel":
        delta_x = args.get("deltaX", 0)
        delta_y = args.get("deltaY")
        if not isinstance(delta_x, int):
            raise ValueError("browser_mouse_wheel_requires_integer_args_deltaX")
        if not isinstance(delta_y, int):
            raise ValueError("browser_mouse_wheel_requires_integer_args_deltaY")
        args["deltaX"] = delta_x
    elif action_name == "browser_wait_for":
        time_s = args.get("time")
        if not isinstance(time_s, (int, float)) or float(time_s) < 0:
            raise ValueError("browser_wait_for_requires_non_negative_numeric_args_time")

    if "reason" in normalized and normalized["reason"] is not None and not isinstance(normalized["reason"], str):
        warnings.append("reason_not_string")
        normalized["reason"] = str(normalized["reason"])

    return normalized, warnings
