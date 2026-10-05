"""OpenAI-compatible endpoints and the vLLM server lifecycle.

One launcher for every locally served model; per-model settings come from the
registry `serve` block (replaces run_qwen_vllm_server.sh / run_opencua_vllm_server.sh
and the start/readiness/warmup/cleanup code copied across the matrix scripts).
"""

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlparse

from baselines.model_registry import resolve_model_path
from formbench.common import FormbenchError, info, quote_cmd
from formbench.settings import Settings

LOCAL_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0"}
TINY_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAAJ0lEQVR4nO3MMQEAAAgDILV/"
    "51nCwwMC0Enq2pyPUqlUKpVKpS/TBdFiAzXB+yhyAAAAAElFTkSuQmCC"
)
SERVE_DEFAULTS: Dict[str, Any] = {
    "backend": "vllm",
    "host": "127.0.0.1",
    "port": 8000,
    "tensor_parallel": 1,
    "gpu_memory_utilization": 0.92,
    "max_model_len": 32768,
    "trust_remote_code": True,
    "disable_custom_all_reduce": True,
    "enable_auto_tool_choice": True,
    "tool_call_parser": None,
    "generation_config": None,
    "limit_mm_per_prompt": None,
    "model_impl": None,
    "mm_encoder_tp_mode": None,
    "chat_template": None,
    "extra_args": [],
    "env": {},
    "startup_attempts": 420,
    "startup_sleep_s": 10,
}
NCCL_ENV = {
    "NCCL_DEBUG": "WARN",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
    "NCCL_SHM_DISABLE": "0",
    "CUDA_MODULE_LOADING": "LAZY",
}
ENDPOINT_ENV_VARS = ("OPENAI_BASE_URL", "OPENAI_MODEL")


def serve_config(model: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = dict(SERVE_DEFAULTS)
    cfg.update(dict(model.get("serve") or {}))
    return cfg


def resolve_port(port: Any) -> int:
    if port == "auto":
        job = os.environ.get("SLURM_JOB_ID")
        return 18000 + (int(job) % 20000) if job and job.isdigit() else 8000
    return int(port)


@dataclass
class Endpoint:
    model_id: str
    base_url: str
    model_name: str
    api_key: str
    managed: bool
    overrides: List[str] = field(default_factory=list)
    startup_s: float = 0.0
    warmup_s: float = 0.0
    server_backend: str = "vllm"

    def baseline_env(self) -> Dict[str, str]:
        """Serving metadata recorded by run_baseline_eval.py into annotations."""
        return {
            "BASELINE_SERVING_MODE": "openai_compat_persistent",
            "BASELINE_SERVER_BACKEND": self.server_backend,
            "BASELINE_SERVER_STARTUP_S": str(int(self.startup_s)),
            "BASELINE_SERVER_WARMUP_S": str(int(self.warmup_s)),
            "BASELINE_SERVER_WARM_STATE": "warm",
        }


def endpoint_for(model: Mapping[str, Any], environ: Optional[Mapping[str, str]] = None) -> Endpoint:
    """Effective endpoint for an openai_compat model, flagging env-var overrides."""
    env = os.environ if environ is None else environ
    model_name = str(model.get("openai_model") or model.get("served_model_name") or "")
    base_url = str(model.get("openai_base_url") or "")
    serve = model.get("serve")
    if not base_url and isinstance(serve, dict):
        cfg = serve_config(model)
        base_url = f"http://{cfg['host']}:{resolve_port(cfg['port'])}/v1"
    overrides: List[str] = []
    if env.get("OPENAI_BASE_URL") and env["OPENAI_BASE_URL"].rstrip("/") != base_url.rstrip("/"):
        overrides.append(f"OPENAI_BASE_URL={env['OPENAI_BASE_URL']} (registry: {base_url or 'unset'})")
        base_url = env["OPENAI_BASE_URL"]
    if env.get("OPENAI_MODEL") and env["OPENAI_MODEL"] != model_name:
        overrides.append(f"OPENAI_MODEL={env['OPENAI_MODEL']} (registry: {model_name or 'unset'})")
        model_name = env["OPENAI_MODEL"]
    key_env = str(model.get("openai_api_key_env") or "OPENAI_API_KEY")
    api_key = env.get(key_env) or "EMPTY"
    host = urlparse(base_url).hostname or ""
    managed = isinstance(serve, dict) and host in LOCAL_HOSTS
    return Endpoint(
        model_id=str(model.get("id")),
        base_url=base_url.rstrip("/"),
        model_name=model_name,
        api_key=api_key,
        managed=managed,
        overrides=overrides,
        server_backend=str(model.get("server_backend") or "vllm"),
    )


def _http_json(url: str, api_key: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8") or "null")


def list_served_models(endpoint: Endpoint, timeout: float = 10.0) -> Optional[List[str]]:
    """Model names advertised at /models, or None if the endpoint is unreachable."""
    try:
        payload = _http_json(endpoint.base_url + "/models", endpoint.api_key, timeout=timeout)
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    return [str(item.get("id")) for item in data if isinstance(item, dict) and item.get("id")]


def smoke_chat(endpoint: Endpoint, kind: str, timeout: float = 180.0) -> float:
    """Send one tiny request (with an image for VLMs); returns seconds taken."""
    content: Any = "Return exactly one JSON object."
    if kind == "vlm":
        content = [
            {"type": "text", "text": "Return exactly one JSON object."},
            {"type": "image_url", "image_url": {"url": TINY_PNG}},
        ]
    payload = {"model": endpoint.model_name, "max_tokens": 8, "temperature": 0, "messages": [{"role": "user", "content": content}]}
    started = time.time()
    _http_json(endpoint.base_url + "/chat/completions", endpoint.api_key, payload=payload, timeout=timeout)
    return time.time() - started


def build_vllm_command(settings: Settings, model: Mapping[str, Any], endpoint: Endpoint) -> Tuple[List[str], Dict[str, str]]:
    cfg = serve_config(model)
    if cfg["backend"] != "vllm":
        raise FormbenchError(f"unsupported serve backend {cfg['backend']!r} for {model.get('id')}")
    parsed = urlparse(endpoint.base_url)
    model_spec = resolve_model_path(dict(model), repo_root=settings.root, models_dir=settings.models_dir)
    if not model_spec:
        raise FormbenchError(f"model {model.get('id')} has no weights_dir, local models/<id> directory or hf_repo")
    argv = [
        str(settings.vllm_python_bin), "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_spec,
        "--served-model-name", endpoint.model_name,
        "--host", parsed.hostname or cfg["host"],
        "--port", str(parsed.port or resolve_port(cfg["port"])),
        "--tensor-parallel-size", str(cfg["tensor_parallel"]),
        "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]),
        "--max-model-len", str(cfg["max_model_len"]),
    ]
    optional = [("model_impl", "--model-impl"), ("limit_mm_per_prompt", "--limit-mm-per-prompt"), ("mm_encoder_tp_mode", "--mm-encoder-tp-mode")]
    for key, flag in optional:
        if cfg.get(key):
            argv += [flag, str(cfg[key])]
    if cfg["trust_remote_code"]:
        argv.append("--trust-remote-code")
    if cfg["disable_custom_all_reduce"]:
        argv.append("--disable-custom-all-reduce")
    if cfg["enable_auto_tool_choice"]:
        argv.append("--enable-auto-tool-choice")
        if cfg.get("tool_call_parser"):
            argv += ["--tool-call-parser", str(cfg["tool_call_parser"])]
    if cfg.get("chat_template"):
        argv += ["--chat-template", str(cfg["chat_template"])]
    if cfg.get("generation_config"):
        argv += ["--generation-config", str(cfg["generation_config"])]
    argv += [str(item) for item in cfg.get("extra_args") or []]
    env = settings.subprocess_env()
    for key, value in NCCL_ENV.items():
        env.setdefault(key, value)
    module_ld = os.environ.get("MODULE_LD_LIBRARY_PATH")
    if module_ld:
        env["LD_LIBRARY_PATH"] = module_ld
    env.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
    return argv, env


class VLLMServer:
    """Start/reuse a vLLM server for one registry model; always stopped on exit."""

    def __init__(self, settings: Settings, model: Mapping[str, Any], endpoint: Endpoint, log_dir: Optional[Path] = None):
        self.settings = settings
        self.model = dict(model)
        self.endpoint = endpoint
        self.cfg = serve_config(model)
        job = os.environ.get("SLURM_JOB_ID") or "local"
        self.log_path = (log_dir or settings.logs_dir / "serve") / f"{model['id']}-vllm-{job}.log"
        self.process: Optional[subprocess.Popen] = None
        self.reused = False

    def is_ready(self) -> bool:
        served = list_served_models(self.endpoint)
        return served is not None and self.endpoint.model_name in served

    def start(self) -> None:
        served = list_served_models(self.endpoint)
        if served is not None:
            if self.endpoint.model_name in served:
                info(f"reusing running server for {self.model['id']} at {self.endpoint.base_url}")
                self.reused = True
                return
            raise FormbenchError(
                f"port in use: {self.endpoint.base_url} serves {served}, not {self.endpoint.model_name!r}",
                hint="stop the other server or give this model a different serve.port",
            )
        if not self.settings.vllm_python_bin.exists():
            raise FormbenchError(f"vLLM interpreter missing: {self.settings.vllm_python_bin}", hint="run `make setup` or set VLLM_PYTHON_BIN in .env")
        argv, env = build_vllm_command(self.settings, self.model, self.endpoint)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        info(f"starting vLLM for {self.model['id']} -> {self.endpoint.base_url} (log: {self.log_path})")
        info("$ " + quote_cmd(argv))
        started = time.time()
        with open(self.log_path, "ab") as log:
            self.process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=str(self.settings.root))
        attempts = int(self.cfg["startup_attempts"])
        sleep_s = float(self.cfg["startup_sleep_s"])
        for attempt in range(1, attempts + 1):
            if self.is_ready():
                self.endpoint.startup_s = time.time() - started
                info(f"server ready for {self.model['id']} after {self.endpoint.startup_s:.0f}s (attempt {attempt})")
                return
            if self.process.poll() is not None:
                self._fail(f"vLLM exited with code {self.process.returncode} before becoming ready")
            if attempt % 6 == 0:
                info(f"waiting for vLLM {self.model['id']} attempt={attempt}/{attempts} elapsed={time.time() - started:.0f}s")
            time.sleep(sleep_s)
        self._fail(f"vLLM did not become ready after {attempts * sleep_s:.0f}s")

    def warmup(self) -> None:
        try:
            self.endpoint.warmup_s = smoke_chat(self.endpoint, str(self.model.get("kind")))
            info(f"warmup ok for {self.model['id']} ({self.endpoint.warmup_s:.1f}s)")
        except Exception as exc:  # noqa: BLE001 - surfaced as a user-facing failure
            self._fail(f"smoke chat request failed: {exc}")

    def _tail_log(self, lines: int = 80) -> str:
        try:
            return "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
        except OSError:
            return ""

    def _fail(self, message: str) -> None:
        tail = self._tail_log()
        self.stop()
        raise FormbenchError(
            f"{self.model['id']}: {message}" + (f"\n--- last lines of {self.log_path} ---\n{tail}" if tail else ""),
            hint="check GPU count/memory (resources.gpus vs serve.tensor_parallel), weights path, and the vLLM log",
        )

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            self.process = None
            return
        info(f"stopping vLLM for {self.model['id']}")
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except OSError:
            self.process.terminate()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except OSError:
                self.process.kill()
            self.process.wait(timeout=30)
        self.process = None

    def __enter__(self) -> "VLLMServer":
        self.start()
        if not self.reused:
            self.warmup()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()
