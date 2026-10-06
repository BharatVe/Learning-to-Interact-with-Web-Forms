"""Helpers shared by the evaluation runners (previously copy-pasted into each runner).

Runners keep module-level names (`_load_run_answers`, `_http_post_json`, ...) that
delegate here, so tests and callers that patch those names keep working. Error
prefixes are parameters because they end up in recorded failure details.
"""

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def load_run_answers(answers_path: Path, run_index: int) -> List[Dict[str, Any]]:
    """Answer entries of the 1-based `run_index` in a runs.json / JSONL answer file."""
    from engine.runner import iter_run_specs

    for idx, run_spec in enumerate(iter_run_specs(answers_path), start=1):
        if idx == run_index:
            answers = run_spec.get("answers", [])
            if not isinstance(answers, list):
                raise ValueError(f"Run {run_index} answers must be a list")
            return answers
    raise IndexError(f"Run index out of range: {run_index} for {answers_path}")


def http_post_json(
    url: str,
    payload: Dict[str, Any],
    timeout_s: int,
    headers: Optional[Mapping[str, str]] = None,
    error_prefix: str = "api",
    compact: bool = False,
) -> Dict[str, Any]:
    """POST JSON and return the decoded object; errors are RuntimeError('<prefix>_<kind>:...')."""
    body = json.dumps(payload, separators=(",", ":") if compact else None).encode("utf-8")
    request = urllib.request.Request(url=url, data=body, method="POST")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=max(1, int(timeout_s))) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        raise RuntimeError(f"{error_prefix}_http_error:{exc.code}:{raw}") from exc
    except Exception as exc:
        raise RuntimeError(f"{error_prefix}_request_failed:{exc}") from exc
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"{error_prefix}_invalid_json:{exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{error_prefix}_response_not_object")
    return parsed


def extract_openai_text(payload: Dict[str, Any], error_prefix: str = "openai_response") -> str:
    """Text of the first chat-completion choice (string or list-of-parts content)."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"{error_prefix}_missing_choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError(f"{error_prefix}_missing_message")
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [item["text"] for item in content if isinstance(item, dict) and isinstance(item.get("text"), str)]
        if parts:
            return "\n".join(parts).strip()
    raise RuntimeError(f"{error_prefix}_missing_text")
