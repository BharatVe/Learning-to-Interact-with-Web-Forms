#!/usr/bin/env python3
import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List


def _http_json(url: str, *, method: str = "GET", payload: Dict[str, Any] | None = None, timeout_s: float = 10.0) -> Dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, method=method, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=max(1.0, float(timeout_s))) as response:
        raw = response.read().decode("utf-8")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise RuntimeError("response_not_object")
    return result


def _model_ids(models_payload: Dict[str, Any]) -> List[str]:
    rows = models_payload.get("data")
    if not isinstance(rows, list):
        return []
    ids: List[str] = []
    for row in rows:
        if isinstance(row, dict) and str(row.get("id") or "").strip():
            ids.append(str(row.get("id")).strip())
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Check an OpenAI-compatible server advertises and serves a specific model.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--print-models", action="store_true")
    parser.add_argument("--smoke-chat", action="store_true")
    args = parser.parse_args()

    base_url = str(args.base_url).rstrip("/")
    expected_model = str(args.model).strip()
    if not expected_model:
        print("[FAIL] expected model is empty", file=sys.stderr)
        return 1

    try:
        models_payload = _http_json(f"{base_url}/models", timeout_s=args.timeout_s)
    except Exception as exc:
        print(f"[FAIL] models_request_failed: {exc}", file=sys.stderr)
        return 1

    ids = _model_ids(models_payload)
    if args.print_models:
        print("[INFO] advertised_models=" + json.dumps(ids, ensure_ascii=True))
    if expected_model not in ids:
        print(
            f"[FAIL] expected_model_not_advertised expected={expected_model!r} advertised={ids!r}",
            file=sys.stderr,
        )
        return 1

    if args.smoke_chat:
        payload = {
            "model": expected_model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 8,
            "temperature": 0,
        }
        try:
            _http_json(f"{base_url}/chat/completions", method="POST", payload=payload, timeout_s=args.timeout_s)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"[FAIL] smoke_chat_http_error:{exc.code}:{body}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"[FAIL] smoke_chat_failed: {exc}", file=sys.stderr)
            return 1
        print("[INFO] smoke_chat_ok=true")

    print(f"[PASS] server_ready model={expected_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
