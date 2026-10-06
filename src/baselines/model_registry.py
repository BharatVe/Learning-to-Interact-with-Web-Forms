"""Model registry (`configs/models.json`): loading, lookup and schema validation.

`list_models` / `get_model_by_id` keep their historical light-weight behaviour so
runners accept ad-hoc configs; `validate_models` is the strict check used by
`formbench models check` and before every orchestrated run.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = "configs/models.json"

KINDS = {"text_llm", "vlm", "computer_use_agent"}
PROVIDERS = {"local_hf", "openai_compat", "api_over_mcp", "gemini_low_cost"}
TRACKS = {
    "mediated",
    "formfactory_style_visual",
    "direct_mcp_tool_use",
    "computer_use_native",
    "direct_api_tool_use",
    "proprietary_computer_use_low_cost",
}
STATUSES = {"current", "legacy"}
PYTHON_ENVS = {"default", "vllm"}

COMMON_KEYS = {
    "id", "kind", "provider", "track", "status", "requires_gpu", "notes", "hf_repo",
    "weights_dir", "is_fallback", "fallback_for", "resources", "runner_defaults", "python_env",
}
PROVIDER_KEYS = {
    "local_hf": set(),
    "openai_compat": {
        "openai_model", "served_model_name", "openai_base_url", "openai_api_key_env",
        "server_backend", "serve", "coordinate_type",
    },
    "api_over_mcp": {
        "openai_model", "openai_base_url", "openai_extra_body", "openai_api_key_env",
        "anthropic_model", "api_key_env",
    },
    "gemini_low_cost": {
        "gemini_model", "pricing", "interactions_endpoint", "api_key_env",
        "gemini_max_infer_retries", "gemini_retry_delay_s", "gemini_retry_backoff", "gemini_retry_max_delay_s",
    },
}
SERVE_KEYS = {
    "backend", "host", "port", "tensor_parallel", "gpu_memory_utilization", "max_model_len",
    "trust_remote_code", "disable_custom_all_reduce", "enable_auto_tool_choice", "tool_call_parser",
    "generation_config", "limit_mm_per_prompt", "model_impl", "mm_encoder_tp_mode", "chat_template",
    "extra_args", "env", "startup_attempts", "startup_sleep_s",
}
RESOURCE_KEYS = {"gpus", "cpus", "mem", "time", "min_vram_gb"}


def load_model_config(config_path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid model config object: {config_path}")
    return payload


def list_models(config_path: Path) -> List[Dict[str, Any]]:
    payload = load_model_config(config_path)
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError(f"Config 'models' must be a list: {config_path}")
    result: List[Dict[str, Any]] = []
    seen = set()
    for idx, model in enumerate(models):
        if not isinstance(model, dict):
            raise ValueError(f"models[{idx}] must be an object")
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"models[{idx}] has invalid id")
        if model_id in seen:
            raise ValueError(f"Duplicate model id: {model_id}")
        seen.add(model_id)
        result.append(model)
    return result


def get_model_by_id(config_path: Path, model_id: str) -> Dict[str, Any]:
    models = list_models(config_path)
    for model in models:
        if model.get("id") == model_id:
            return model
    known = ", ".join(sorted(str(m.get("id")) for m in models))
    raise KeyError(f"Model id not found in config {config_path}: {model_id} (known: {known})")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_one(model: Dict[str, Any], ids: set) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    mid = model.get("id")
    kind = model.get("kind")
    provider = model.get("provider")
    track = model.get("track")
    if kind not in KINDS:
        errors.append(f"kind must be one of {sorted(KINDS)}, got {kind!r}")
    if provider not in PROVIDERS:
        errors.append(f"provider must be one of {sorted(PROVIDERS)}, got {provider!r}")
    if track not in TRACKS:
        errors.append(f"track must be one of {sorted(TRACKS)}, got {track!r}")
    status = model.get("status", "current")
    if status not in STATUSES:
        errors.append(f"status must be one of {sorted(STATUSES)}, got {status!r}")
    if model.get("python_env", "default") not in PYTHON_ENVS:
        errors.append(f"python_env must be one of {sorted(PYTHON_ENVS)}")
    for flag in ("requires_gpu", "is_fallback"):
        if flag in model and not isinstance(model[flag], bool):
            errors.append(f"{flag} must be true/false")
    for text_key in ("hf_repo", "weights_dir", "notes"):
        if model.get(text_key) is not None and not isinstance(model.get(text_key), str):
            errors.append(f"{text_key} must be a string or null")
    fallback_for = model.get("fallback_for")
    if fallback_for is not None:
        if fallback_for not in ids:
            errors.append(f"fallback_for references unknown model id {fallback_for!r}")
        if not model.get("is_fallback"):
            warnings.append("fallback_for is set but is_fallback is not true")

    if provider == "local_hf" and not model.get("hf_repo") and not model.get("weights_dir"):
        errors.append("local_hf models need hf_repo or weights_dir")
    if provider == "openai_compat":
        if not (model.get("openai_model") or model.get("served_model_name")):
            errors.append("openai_compat models need openai_model or served_model_name")
        if not model.get("openai_base_url") and not isinstance(model.get("serve"), dict):
            errors.append("openai_compat models need openai_base_url or a serve block")
        if (
            model.get("openai_model")
            and model.get("served_model_name")
            and model.get("openai_model") != model.get("served_model_name")
        ):
            warnings.append("openai_model differs from served_model_name; requests use openai_model")
    if provider == "api_over_mcp" and not (model.get("openai_model") or model.get("anthropic_model")):
        errors.append("api_over_mcp models need openai_model or anthropic_model")
    if provider == "gemini_low_cost" and not model.get("gemini_model"):
        errors.append("gemini_low_cost models need gemini_model")
    if provider in {"openai_compat", "local_hf"} and kind == "computer_use_agent" and track == "mediated":
        errors.append("computer_use_agent models cannot use the mediated track")

    serve = model.get("serve")
    if serve is not None:
        if not isinstance(serve, dict):
            errors.append("serve must be an object")
        else:
            port = serve.get("port")
            if port is not None and port != "auto" and not _is_int(port):
                errors.append("serve.port must be an integer or 'auto'")
            tp = serve.get("tensor_parallel")
            if tp is not None and (not _is_int(tp) or tp < 1):
                errors.append("serve.tensor_parallel must be a positive integer")
            util = serve.get("gpu_memory_utilization")
            if util is not None and not (isinstance(util, (int, float)) and 0 < float(util) <= 1):
                errors.append("serve.gpu_memory_utilization must be in (0, 1]")
            for key in sorted(set(serve) - SERVE_KEYS):
                warnings.append(f"unknown serve key {key!r}")
    resources = model.get("resources")
    if resources is not None:
        if not isinstance(resources, dict):
            errors.append("resources must be an object")
        else:
            for key in ("gpus", "cpus"):
                if key in resources and (not _is_int(resources[key]) or resources[key] < 0):
                    errors.append(f"resources.{key} must be a non-negative integer")
            for key in sorted(set(resources) - RESOURCE_KEYS):
                warnings.append(f"unknown resources key {key!r}")
            gpus = resources.get("gpus")
            tp = (serve or {}).get("tensor_parallel") if isinstance(serve, dict) else None
            if _is_int(gpus) and _is_int(tp) and tp > gpus:
                errors.append(f"serve.tensor_parallel={tp} exceeds resources.gpus={gpus}")
    if model.get("runner_defaults") is not None and not isinstance(model.get("runner_defaults"), dict):
        errors.append("runner_defaults must be an object of runner flag -> value")

    allowed = COMMON_KEYS | PROVIDER_KEYS.get(str(provider), set())
    for key in sorted(set(model) - allowed):
        warnings.append(f"unknown key {key!r} for provider {provider!r} (typo?)")
    return [f"{mid}: {msg}" for msg in errors], [f"{mid}: {msg}" for msg in warnings]


def validate_models(models: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings) for a list of registry entries."""
    ids = {str(m.get("id")) for m in models if isinstance(m, dict)}
    errors: List[str] = []
    warnings: List[str] = []
    for model in models:
        e, w = _validate_one(model, ids)
        errors.extend(e)
        warnings.extend(w)
    return errors, warnings


def validate_config(config_path: Path) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """Load + strictly validate a registry file. Structural problems become errors."""
    try:
        models = list_models(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [], [f"{config_path}: {exc}"], []
    errors, warnings = validate_models(models)
    return models, errors, warnings


def resolve_model_path(model: Dict[str, Any], repo_root: Optional[Path] = None, models_dir: Optional[Path] = None) -> str:
    """Local weights directory if present, else the Hugging Face repo id.

    Order: explicit `weights_dir` -> `<models_dir>/<id>` -> `hf_repo`.
    """
    root = repo_root or REPO_ROOT
    weights_dir = model.get("weights_dir")
    if weights_dir:
        path = Path(weights_dir)
        path = path if path.is_absolute() else root / path
        if path.is_dir():
            return str(path)
    base = models_dir or (root / "models")
    candidate = base / str(model.get("id") or "")
    if model.get("id") and candidate.is_dir():
        return str(candidate)
    return str(model.get("hf_repo") or "")


def local_weights_dir(model: Dict[str, Any], repo_root: Optional[Path] = None, models_dir: Optional[Path] = None) -> Optional[Path]:
    resolved = resolve_model_path(model, repo_root=repo_root, models_dir=models_dir)
    path = Path(resolved) if resolved else None
    return path if path is not None and path.is_dir() else None
