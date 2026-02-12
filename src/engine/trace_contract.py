from typing import Any, Dict, List


class TraceValidationError(RuntimeError):
    """Raised when a trace event does not match the supported action contract."""


SUPPORTED_ACTIONS = {
    "browser_mouse_click_xy",
    "browser_mouse_move_xy",
    "browser_type",
    "browser_press_key",
    "browser_mouse_wheel",
    "browser_navigate",
    "browser_wait_for",
    "browser_run_code",
    "browser_take_screenshot",
    "browser_close",
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


def _validate_number(args: Dict[str, Any], key: str, name: str, ok: bool) -> None:
    if key not in args:
        raise TraceValidationError(f"{name} requires args.{key}")
    value = args.get(key)
    if value is None:
        if ok:
            raise TraceValidationError(f"{name} requires non-null args.{key} when ok=true")
        return
    if not isinstance(value, (int, float)):
        raise TraceValidationError(f"{name} requires numeric args.{key}, got {type(value).__name__}")


def validate_action(name: str, args: Dict[str, Any], ok: bool = True) -> None:
    if name not in SUPPORTED_ACTIONS:
        raise TraceValidationError(
            f"Unsupported action name '{name}'. Supported: {', '.join(supported_actions())}"
        )

    if name in {"browser_mouse_click_xy", "browser_mouse_move_xy"}:
        _validate_norm_coord(args, "x", name)
        _validate_norm_coord(args, "y", name)
        return

    if name == "browser_type":
        text = args.get("text")
        if text is None and ok:
            raise TraceValidationError("browser_type requires args.text when ok=true")
        if text is not None and not isinstance(text, str):
            raise TraceValidationError("browser_type requires args.text to be a string")
        slowly = args.get("slowly")
        if slowly is not None and not isinstance(slowly, bool):
            raise TraceValidationError("browser_type args.slowly must be a boolean when provided")
        submit = args.get("submit")
        if submit is not None and not isinstance(submit, bool):
            raise TraceValidationError("browser_type args.submit must be a boolean when provided")
        return

    if name == "browser_press_key":
        key = args.get("key")
        if not isinstance(key, str) or not key.strip():
            raise TraceValidationError("browser_press_key requires non-empty args.key string")
        return

    if name == "browser_mouse_wheel":
        _validate_number(args, "deltaX", name, ok=ok)
        _validate_number(args, "deltaY", name, ok=ok)
        return

    if name == "browser_navigate":
        url = args.get("url")
        if not isinstance(url, str) or not url.strip():
            raise TraceValidationError("browser_navigate requires non-empty args.url string")
        return

    if name == "browser_wait_for":
        time_s = args.get("time")
        text = args.get("text")
        text_gone = args.get("textGone")
        has_valid_time = isinstance(time_s, (int, float)) and float(time_s) >= 0
        has_valid_text = isinstance(text, str) and text.strip()
        has_valid_text_gone = isinstance(text_gone, str) and text_gone.strip()
        if not (has_valid_time or has_valid_text or has_valid_text_gone):
            raise TraceValidationError(
                "browser_wait_for requires at least one of args.time, args.text, args.textGone"
            )
        return

    if name == "browser_run_code":
        code = args.get("code")
        purpose = args.get("purpose")
        if not isinstance(code, str) and not isinstance(purpose, str):
            raise TraceValidationError("browser_run_code requires args.code or args.purpose string")
        return

    if name == "browser_take_screenshot":
        filename = args.get("filename")
        if filename is not None and (not isinstance(filename, str) or not filename.strip()):
            raise TraceValidationError("browser_take_screenshot args.filename must be a non-empty string")
        image_type = args.get("type")
        if image_type is not None and image_type not in {"png", "jpeg"}:
            raise TraceValidationError("browser_take_screenshot args.type must be png or jpeg")
        return

    if name == "browser_close":
        return
