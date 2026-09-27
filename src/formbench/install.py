"""`formbench models install`: download Hugging Face weights into models/<id> (or weights_dir)."""

import socket
from pathlib import Path
from typing import Any, Dict, List, Tuple

from baselines.model_registry import validate_config
from formbench.common import FormbenchError, dump_table, info, warn
from formbench.settings import Settings


def weights_valid(local_dir: Path) -> Tuple[bool, str]:
    if not local_dir.is_dir():
        return False, "missing"
    if not (local_dir / "config.json").is_file():
        return False, "no config.json"
    if not (any(local_dir.glob("*.safetensors")) or any(local_dir.glob("*.bin"))):
        return False, "no *.safetensors / *.bin"
    return True, "ok"


def target_dir(settings: Settings, model: Dict[str, Any]) -> Path:
    if model.get("weights_dir"):
        path = Path(model["weights_dir"])
        return path if path.is_absolute() else settings.root / path
    return settings.models_dir / str(model["id"])


def installable(model: Dict[str, Any]) -> bool:
    """Local weights make sense for in-process (local_hf) and locally served models."""
    if not model.get("hf_repo"):
        return False
    return model.get("provider") == "local_hf" or isinstance(model.get("serve"), dict)


def install_models(settings: Settings, model_ids: List[str], dry_run: bool = False, force: bool = False) -> int:
    models, errors, _ = validate_config(settings.models_config)
    if errors:
        raise FormbenchError("registry has errors: " + "; ".join(errors), hint="make model-check")
    by_id = {m["id"]: m for m in models}
    unknown = [m for m in model_ids if m not in by_id]
    if unknown:
        raise FormbenchError(f"unknown model id(s): {', '.join(unknown)}", hint="make models ALL=1")
    selected = [by_id[m] for m in model_ids] if model_ids else [m for m in models if installable(m) and m.get("status", "current") == "current"]
    rows, todo = [], []
    for model in selected:
        if not installable(model):
            rows.append({"id": model["id"], "target": "-", "state": "remote/API model: nothing to download"})
            continue
        path = target_dir(settings, model)
        valid, state = weights_valid(path)
        rows.append({"id": model["id"], "target": str(path.relative_to(settings.root)) if settings.root in path.parents else str(path), "state": state})
        if force or not valid:
            todo.append((model, path))
    print(dump_table(rows, ["id", "target", "state"]))
    if not todo:
        info("nothing to download")
        return 0
    if dry_run:
        for model, path in todo:
            info(f"would download {model['hf_repo']} -> {path}")
        return 0
    try:
        socket.gethostbyname("huggingface.co")
    except OSError:
        raise FormbenchError("huggingface.co is not resolvable from this node", hint="download on a login node with internet access, or set HF_ENDPOINT")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise FormbenchError(f"huggingface_hub missing: {exc}", hint="make setup")
    failures = 0
    for model, path in todo:
        info(f"downloading {model['hf_repo']} -> {path}")
        path.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(repo_id=model["hf_repo"], local_dir=str(path))
        except Exception as exc:  # noqa: BLE001 - report and continue with the next model
            warn(f"{model['id']}: download failed: {exc}")
            failures += 1
            continue
        valid, state = weights_valid(path)
        (info if valid else warn)(f"{model['id']}: {state}")
        failures += 0 if valid else 1
    return 1 if failures else 0
