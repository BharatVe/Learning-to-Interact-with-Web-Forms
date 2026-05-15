import json
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


DEFAULT_BROWSER_LOCALE = "en-US"
DEFAULT_ACCEPT_LANGUAGE = "en-US,en;q=0.9"
GOOGLE_FORMS_HOST_SUFFIX = "docs.google.com"


def force_english_google_forms_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return raw
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        return raw
    if not parsed.netloc.endswith(GOOGLE_FORMS_HOST_SUFFIX) or "/forms/" not in parsed.path:
        return raw
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params["hl"] = "en"
    return urlunparse(parsed._replace(query=urlencode(params)))


def english_context_options() -> Dict[str, Any]:
    return {
        "locale": DEFAULT_BROWSER_LOCALE,
        "extra_http_headers": {"Accept-Language": DEFAULT_ACCEPT_LANGUAGE},
    }


def write_playwright_mcp_english_config(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "playwright_mcp_english_config.json"
    payload = {
        "browser": {
            "launchOptions": {"args": [f"--lang={DEFAULT_BROWSER_LOCALE}"]},
            "contextOptions": {
                "locale": DEFAULT_BROWSER_LOCALE,
                "extraHTTPHeaders": {"Accept-Language": DEFAULT_ACCEPT_LANGUAGE},
            },
        }
    }
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config_path
