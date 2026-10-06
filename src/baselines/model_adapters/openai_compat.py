import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from baselines.common import extract_openai_text, http_post_json


def _http_post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    return http_post_json(url, payload, timeout_s, headers=headers, error_prefix="openai_compat")


def _extract_openai_text(payload: Dict[str, Any]) -> str:
    return extract_openai_text(payload, error_prefix="openai_compat")


def _image_to_data_url(path: Path) -> str:
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class OpenAICompatAdapter:
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        model_kind: str,
        max_new_tokens: int = 160,
        api_timeout_s: int = 120,
    ) -> None:
        self.model_cfg = dict(model_cfg or {})
        self.model_kind = str(model_kind or "").strip()
        self.max_new_tokens = int(max_new_tokens)
        self.api_timeout_s = max(1, int(api_timeout_s))
        self.base_url = str(
            os.environ.get("OPENAI_BASE_URL")
            or self.model_cfg.get("openai_base_url")
            or "http://127.0.0.1:8000/v1"
        ).rstrip("/")
        self.model = str(os.environ.get("OPENAI_MODEL") or self.model_cfg.get("openai_model") or "").strip()
        if not self.model:
            raise RuntimeError("openai_compat_missing_model: set openai_model in config or OPENAI_MODEL env")
        self.api_key = str(os.environ.get("OPENAI_API_KEY") or self.model_cfg.get("openai_api_key") or "EMPTY")
        self.last_infer_meta: Dict[str, Any] = {}

    def infer(
        self,
        prompt: str,
        image_path: Optional[Path] = None,
        max_new_tokens_override: Optional[int] = None,
    ) -> str:
        max_new_tokens = self.max_new_tokens if max_new_tokens_override is None else int(max_new_tokens_override)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "Return exactly one JSON action object and nothing else."}
        ]
        if self.model_kind == "vlm":
            if image_path is None:
                raise RuntimeError("openai_compat_vlm_requires_image_path")
            data_url = _image_to_data_url(Path(image_path))
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": str(prompt or "")},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": str(prompt or "")})

        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max(1, int(max_new_tokens)),
            "messages": messages,
        }
        started = time.perf_counter()
        response = _http_post_json(
            url=f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload=payload,
            timeout_s=self.api_timeout_s,
        )
        self.last_infer_meta = {
            "roundtrip_s": round(time.perf_counter() - started, 3),
            "server_backend": str(self.model_cfg.get("server_backend") or "openai_compat").strip() or "openai_compat",
            "serving_mode": str(os.environ.get("BASELINE_SERVING_MODE") or "").strip() or None,
            "server_warm_state": str(os.environ.get("BASELINE_SERVER_WARM_STATE") or "").strip() or None,
        }
        return _extract_openai_text(response)
