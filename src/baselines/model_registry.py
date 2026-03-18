import json
from pathlib import Path
from typing import Any, Dict, List


def load_model_config(config_path: Path) -> Dict[str, Any]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
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
    for model in list_models(config_path):
        if model.get("id") == model_id:
            return model
    raise KeyError(f"Model id not found in config: {model_id}")
