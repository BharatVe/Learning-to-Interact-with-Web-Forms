import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class TraceLogger:
    def __init__(self, path: Path, start_time: float) -> None:
        self.path = path
        self.start_time = start_time
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:
            pass

    def now(self) -> float:
        return time.perf_counter() - self.start_time

    def log_event(
        self,
        name: str,
        args: Dict[str, Any],
        step_ref: Optional[int],
        ok: bool = True,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
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
