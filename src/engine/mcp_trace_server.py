import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from engine.trace_contract import TraceValidationError, supported_actions, validate_action  # noqa: E402

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "thesis-trace-mcp"
SERVER_VERSION = "1.0.0"


def _ok_response(request_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error_response(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _tool_input_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "args": {"type": "object"},
            "step_ref": {"type": ["integer", "null"]},
            "ok": {"type": "boolean"},
            "error": {"type": ["string", "null"]},
            "extra": {"type": ["object", "null"]},
            "validate_mcp_actions": {"type": "boolean"},
            "strict_mcp_validation": {"type": "boolean"},
        },
        "required": ["name", "args"],
    }


def _tool_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=True)}],
        "structuredContent": payload,
        "isError": False,
    }


def _record_action(arguments: Dict[str, Any]) -> Dict[str, Any]:
    name = str(arguments.get("name", ""))
    args = arguments.get("args")
    if not isinstance(args, dict):
        raise TraceValidationError(f"Trace args must be an object, got {type(args).__name__}")

    ok = bool(arguments.get("ok", True))
    validate_mcp_actions = bool(arguments.get("validate_mcp_actions", True))
    strict_mcp_validation = bool(arguments.get("strict_mcp_validation", True))

    validation_error: Optional[str] = None
    if validate_mcp_actions:
        try:
            validate_action(name, args, ok=ok)
        except TraceValidationError as exc:
            validation_error = str(exc)

    accepted = validation_error is None or not strict_mcp_validation
    return {
        "accepted": accepted,
        "validation_error": validation_error,
        "name": name,
        "args": args,
        "step_ref": arguments.get("step_ref"),
        "ok": ok,
        "error": arguments.get("error"),
        "extra": arguments.get("extra"),
        "validate_mcp_actions": validate_mcp_actions,
        "strict_mcp_validation": strict_mcp_validation,
        "supported_actions": supported_actions(),
    }


def _write_message(message: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _handle_request(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    request_id = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}

    if method == "initialize":
        return _ok_response(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return _ok_response(
            request_id,
            {
                "tools": [
                    {
                        "name": "record_action",
                        "description": "Validate and normalize one tool-trace action event.",
                        "inputSchema": _tool_input_schema(),
                    }
                ]
            },
        )

    if method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments") or {}
        if tool_name != "record_action":
            return _error_response(request_id, -32601, f"Unknown tool: {tool_name}")
        try:
            payload = _record_action(tool_args)
        except TraceValidationError as exc:
            payload = {
                "accepted": False,
                "validation_error": str(exc),
                "name": tool_args.get("name"),
                "args": tool_args.get("args"),
                "step_ref": tool_args.get("step_ref"),
                "ok": bool(tool_args.get("ok", False)),
                "error": str(exc),
                "extra": tool_args.get("extra"),
                "validate_mcp_actions": bool(tool_args.get("validate_mcp_actions", True)),
                "strict_mcp_validation": bool(tool_args.get("strict_mcp_validation", True)),
                "supported_actions": supported_actions(),
            }
        return _ok_response(request_id, _tool_result(payload))

    return _error_response(request_id, -32601, f"Method not found: {method}")


def main() -> int:
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except Exception:
            continue
        if not isinstance(message, dict):
            continue
        if "id" not in message and message.get("method") == "notifications/initialized":
            continue
        if "id" not in message:
            continue
        response = _handle_request(message)
        if response is not None:
            _write_message(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
