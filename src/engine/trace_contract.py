from typing import Any, Dict, List


class TraceValidationError(RuntimeError):
    """Raised when a trace event does not match the supported action contract."""


SUPPORTED_ACTIONS = {
    "click_at",
    "hover_at",
    "type_text_at",
    "key_combination",
    "scroll_document",
}


def supported_actions() -> List[str]:
    return sorted(SUPPORTED_ACTIONS)


def _validate_norm_coord(args: Dict[str, Any], key: str, name: str) -> None:
    if key not in args:
        raise TraceValidationError(f"{name} requires args.{key}")
    value = args.get(key)
    if value is None:
        return
    if not isinstance(value, int):
        raise TraceValidationError(f"{name} requires integer args.{key}, got {type(value).__name__}")
    if value < 0 or value > 999:
        raise TraceValidationError(f"{name} requires args.{key} in [0, 999], got {value}")


def validate_action(name: str, args: Dict[str, Any], ok: bool = True) -> None:
    if name not in SUPPORTED_ACTIONS:
        raise TraceValidationError(
            f"Unsupported action name '{name}'. Supported: {', '.join(supported_actions())}"
        )

    if name in {"click_at", "hover_at"}:
        _validate_norm_coord(args, "x", name)
        _validate_norm_coord(args, "y", name)
        return

    if name == "type_text_at":
        _validate_norm_coord(args, "x", name)
        _validate_norm_coord(args, "y", name)
        text = args.get("text")
        if text is None and ok:
            raise TraceValidationError("type_text_at requires args.text when ok=true")
        if text is not None and not isinstance(text, str):
            raise TraceValidationError("type_text_at requires args.text to be a string")
        return

    if name == "key_combination":
        keys = args.get("keys")
        if not isinstance(keys, str) or not keys.strip():
            raise TraceValidationError("key_combination requires non-empty args.keys string")
        return

    if name == "scroll_document":
        direction = args.get("direction")
        if direction not in {"up", "down"}:
            raise TraceValidationError("scroll_document requires args.direction of 'up' or 'down'")
        return
