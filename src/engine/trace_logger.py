import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from engine.trace_contract import TraceValidationError, supported_actions, validate_action

class TraceLogger:
    def __init__(
        self,
        path: Path,
        start_time: float,
        validate_mcp_actions: bool = True,
        strict_mcp_validation: bool = True,
        mcp_client: Optional[Any] = None,
    ) -> None:
        self.path = path
        self.start_time = start_time
        self.validate_mcp_actions = validate_mcp_actions
        self.strict_mcp_validation = strict_mcp_validation
        self.mcp_client = mcp_client
        self.mode = "mcp_server" if mcp_client is not None else "local"
        self.event_count = 0
        self.validation_error_count = 0
        self.events_by_name: Dict[str, int] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:
            pass
        if self.mcp_client is not None:
            try:
                self.mcp_client.close()
            except Exception:
                pass

    def now(self) -> float:
        return time.perf_counter() - self.start_time

    def summary(self) -> Dict[str, Any]:
        mcp_info: Optional[Dict[str, Any]] = None
        if self.mcp_client is not None:
            try:
                mcp_info = self.mcp_client.summary()
            except Exception as exc:
                mcp_info = {"mode": "mcp_server", "error": str(exc)}
        return {
            "format": "mcp_computer_use_compatible",
            "mode": self.mode,
            "validate_mcp_actions": self.validate_mcp_actions,
            "strict_mcp_validation": self.strict_mcp_validation,
            "event_count": self.event_count,
            "validation_error_count": self.validation_error_count,
            "events_by_name": dict(sorted(self.events_by_name.items())),
            "supported_actions": supported_actions(),
            "mcp": mcp_info,
        }

    def _validate_local(self, name: str, args: Dict[str, Any], ok: bool) -> None:
        if not self.validate_mcp_actions:
            return
        try:
            validate_action(name, args, ok=ok)
        except TraceValidationError:
            self.validation_error_count += 1
            if self.strict_mcp_validation:
                raise

    def _validate_via_mcp(
        self,
        name: str,
        args: Dict[str, Any],
        step_ref: Optional[int],
        ok: bool,
        error: Optional[str],
        extra: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if self.mcp_client is None:
            return {
                "name": name,
                "args": args,
                "step_ref": step_ref,
                "ok": ok,
                "error": error,
                "extra": extra,
            }
        payload = self.mcp_client.record_action(
            {
                "name": name,
                "args": args,
                "step_ref": step_ref,
                "ok": ok,
                "error": error,
                "extra": extra,
                "validate_mcp_actions": self.validate_mcp_actions,
                "strict_mcp_validation": self.strict_mcp_validation,
            }
        )
        validation_error = payload.get("validation_error")
        if validation_error:
            self.validation_error_count += 1
        if not payload.get("accepted", True):
            raise TraceValidationError(str(validation_error or "mcp_rejected_event"))
        return {
            "name": payload.get("name", name),
            "args": payload.get("args", args),
            "step_ref": payload.get("step_ref", step_ref),
            "ok": bool(payload.get("ok", ok)),
            "error": payload.get("error", error),
            "extra": payload.get("extra", extra),
        }

    def log_event(
        self,
        name: str,
        args: Dict[str, Any],
        step_ref: Optional[int],
        ok: bool = True,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(args, dict):
            raise TraceValidationError(f"Trace args must be an object, got {type(args).__name__}")

        if self.mcp_client is not None:
            event = self._validate_via_mcp(name, args, step_ref, ok, error, extra)
            name = event["name"]
            args = event["args"]
            step_ref = event["step_ref"]
            ok = event["ok"]
            error = event["error"]
            extra = event["extra"]
        else:
            self._validate_local(name, args, ok)

        record: Dict[str, Any] = {
            "t_s": self.now(),
            "step_ref": step_ref,
            "name": name,
            "args": args,
            "ok": ok,
            "error": error,
        }
        if extra:
            record.update(extra)
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()
        self.event_count += 1
        self.events_by_name[name] = self.events_by_name.get(name, 0) + 1
