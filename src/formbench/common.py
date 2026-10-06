"""Small shared helpers for the formbench CLI (logging, ids, JSON, subprocess)."""

import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr, flush=True)


class FormbenchError(RuntimeError):
    """User-facing error: printed as [FAIL] with an optional fix hint, exit code 1."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_trial_id() -> str:
    return "trial_" + utc_now().strftime("%Y%m%dT%H%M%S%fZ")


def new_run_label() -> str:
    job_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(os.environ.get("SLURM_JOB_ID") or "na").strip()) or "na"
    return utc_now().strftime("%Y%m%dT%H%M%SZ") + f"_job{job_id}"


def answer_run_id(run_index: int) -> str:
    return f"run_{int(run_index):04d}"


def quote_cmd(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def short(value: Any) -> str:
    """Repo-relative rendering of paths (and of commands containing them) for display."""
    from formbench.settings import REPO_ROOT

    return str(value).replace(str(REPO_ROOT) + "/", "")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_csv_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_run_indexes(value: Any) -> List[int]:
    """Accept 2, "2", "1,2,3", "1-10", [1, 2] -> sorted unique ints."""
    if value is None or value == "":
        return []
    items: Iterable[Any] = value if isinstance(value, (list, tuple)) else str(value).split(",")
    result: List[int] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        if "-" in text:
            start, _, end = text.partition("-")
            lo, hi = int(start), int(end)
            if hi < lo:
                raise ValueError(f"invalid run range {text!r}")
            result.extend(range(lo, hi + 1))
        else:
            result.append(int(text))
    if any(idx < 1 for idx in result):
        raise ValueError("run indexes start at 1")
    return sorted(set(result))


def discover_form_ids(specs_root: Path) -> List[str]:
    if not specs_root.is_dir():
        return []
    return sorted(entry.name for entry in specs_root.iterdir() if entry.is_dir() and (entry / "spec.json").is_file())


def run(argv: Sequence[str], env: Optional[Mapping[str, str]] = None, cwd: Optional[Path] = None, check: bool = False) -> int:
    info("$ " + short(quote_cmd(argv)))
    completed = subprocess.run(list(argv), env=dict(env) if env is not None else None, cwd=str(cwd) if cwd else None)
    if check and completed.returncode != 0:
        raise FormbenchError(f"command failed with exit code {completed.returncode}: {quote_cmd(argv)}")
    return completed.returncode


def args_to_flags(args: Mapping[str, Any], negations: Optional[Mapping[str, str]] = None) -> List[str]:
    """{"max_steps": 32, "headless": True, "fill_only_done": False} -> ["--max-steps", "32", "--headless"].

    `negations` maps a boolean flag to the flag emitted when it is False
    (e.g. {"fewshot_enabled": "no_fewshot_enabled"}); other False/None values are omitted.
    """
    negations = negations or {}
    flags: List[str] = []
    for key, value in args.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                flags.append(flag)
            elif key in negations:
                flags.append("--" + negations[key].replace("_", "-"))
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                flags.extend([flag, str(item)])
            continue
        flags.extend([flag, str(value)])
    return flags


def dump_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    widths = {col: max(len(col), *(len(str(row.get(col, ""))) for row in rows)) if rows else len(col) for col in columns}
    lines = ["  ".join(col.ljust(widths[col]) for col in columns), "  ".join("-" * widths[col] for col in columns)]
    for row in rows:
        lines.append("  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))
    return "\n".join(lines)
