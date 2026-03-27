#!/usr/bin/env python3
import argparse
import json
import logging
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REQUIRED_MODEL_KEYS = {"id", "kind", "provider"}
OPTIONAL_PROVIDER_KEYS = {
    "hf_repo",
    "track",
    "requires_gpu",
    "openai_model",
    "openai_base_url",
    "openai_api_key",
    "anthropic_model",
    "is_fallback",
    "fallback_for",
}
MANDATORY_MODEL_FILES = {"config.json"}


def dns_ok(host: str) -> bool:
    try:
        socket.gethostbyname(host)
        return True
    except Exception:
        return False


def setup_logger(logs_dir: Path, verbose: bool) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    logfile = logs_dir / f"model_install_{ts}.log"

    logger = logging.getLogger("model_installer")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    logger.info(f"[INFO] install log: {logfile}")
    return logger


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def model_dir_is_valid(local_dir: Path) -> Tuple[bool, List[str]]:
    if not local_dir.exists() or not local_dir.is_dir():
        return False, ["model directory missing"]
    missing = sorted([name for name in MANDATORY_MODEL_FILES if not (local_dir / name).exists()])
    if missing:
        return False, [f"missing required file: {name}" for name in missing]
    has_weights = any(local_dir.glob("*.safetensors")) or any(local_dir.glob("*.bin"))
    if not has_weights:
        return False, ["missing model weight files (*.safetensors or *.bin)"]
    return True, []


def validate_models(config_path: Path, models: Iterable[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    seen_ids: set = set()
    for i, model in enumerate(models):
        if not isinstance(model, dict):
            errors.append(f"models[{i}] must be an object")
            continue
        missing = sorted([key for key in REQUIRED_MODEL_KEYS if key not in model])
        if missing:
            errors.append(f"models[{i}] missing required keys: {missing}")
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            errors.append(f"models[{i}] has invalid id")
        elif model_id in seen_ids:
            errors.append(f"duplicate model id: {model_id}")
        else:
            seen_ids.add(model_id)
        provider = model.get("provider")
        if provider == "local_hf":
            repo_id = model.get("hf_repo")
            if not isinstance(repo_id, str) or "/" not in repo_id:
                errors.append(f"model '{model_id}' has provider=local_hf but invalid hf_repo")
        for key in model.keys():
            if key not in REQUIRED_MODEL_KEYS and key not in OPTIONAL_PROVIDER_KEYS and key not in {"notes", "enabled"}:
                warnings.append(f"model '{model_id}' has unknown key '{key}' in {config_path.name}")
    return errors, warnings


def install_hf_model(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))


def parse_csv_ids(raw: Optional[str]) -> Optional[set]:
    if not raw:
        return None
    items = [item.strip() for item in raw.split(",")]
    selected = {item for item in items if item}
    return selected if selected else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Install baseline model weights from config with safeguards.")
    parser.add_argument("--config", default="configs/baselines/minimal_models.json", help="Model config path.")
    parser.add_argument("--models-root", default="models", help="Directory for local model snapshots.")
    parser.add_argument("--skip-if-valid", action="store_true", default=True, help="Skip model if local files are valid.")
    parser.add_argument("--include-ids", help="Comma-separated model ids to include.")
    parser.add_argument("--exclude-ids", help="Comma-separated model ids to exclude.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any model fails.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs.")
    parser.add_argument("--logs-dir", default="logs", help="Directory to write installer logs.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve()
    models_root = (repo_root / args.models_root).resolve()
    include_ids = parse_csv_ids(args.include_ids)
    exclude_ids = parse_csv_ids(args.exclude_ids) or set()

    logger = setup_logger((repo_root / args.logs_dir).resolve(), args.verbose)
    logger.info(f"[INFO] config: {config_path}")
    logger.info(f"[INFO] models_root: {models_root}")

    if not config_path.exists():
        logger.error(f"[FAIL] config not found: {config_path}")
        return 1

    cfg = load_config(config_path)
    models_raw = cfg.get("models", [])
    if not isinstance(models_raw, list) or not models_raw:
        logger.error(f"[FAIL] No models in config: {config_path}")
        return 1

    validation_errors, validation_warnings = validate_models(config_path, models_raw)
    if validation_errors:
        logger.error(f"[FAIL] Config validation failed ({len(validation_errors)} issues)")
        for err in validation_errors:
            logger.error(f" - {err}")
        return 1
    for warn in validation_warnings:
        logger.warning(f"[WARN] {warn}")

    models: List[Dict[str, Any]] = []
    for model in models_raw:
        if model.get("enabled", True) is False:
            logger.info(f"[SKIP] {model['id']} disabled by config")
            continue
        if include_ids and model["id"] not in include_ids:
            continue
        if model["id"] in exclude_ids:
            logger.info(f"[SKIP] {model['id']} excluded by --exclude-ids")
            continue
        models.append(model)

    if not models:
        logger.warning("[WARN] no models selected after include/exclude filters")
        return 0

    needs_hf = any(model.get("provider") == "local_hf" for model in models)
    if needs_hf and (not args.dry_run) and not dns_ok("huggingface.co"):
        logger.error("[FAIL] DNS/network for huggingface.co is unavailable.")
        logger.error(f"[INFO] Use internet-enabled node or copy snapshots into: {models_root}")
        return 2

    installed = 0
    skipped = 0
    failed = 0

    for model in models:
        model_id = model["id"]
        kind = model["kind"]
        provider = model["provider"]
        repo_id = model.get("hf_repo")
        target_dir = models_root / model_id

        logger.info(f"[INFO] processing {model_id} ({kind}, provider={provider})")

        if provider != "local_hf":
            logger.info(f"[SKIP] {model_id} is non-local provider ({provider}); no local download needed")
            skipped += 1
            continue

        valid, issues = model_dir_is_valid(target_dir)
        if args.skip_if_valid and valid:
            logger.info(f"[SKIP] {model_id} already valid at {target_dir}")
            skipped += 1
            continue
        if issues and target_dir.exists():
            logger.warning(f"[WARN] {model_id} existing directory is incomplete: {issues}")

        if args.dry_run:
            logger.info(f"[DRY] would download {repo_id} -> {target_dir}")
            skipped += 1
            continue

        try:
            logger.info(f"[INFO] downloading {repo_id} -> {target_dir}")
            install_hf_model(str(repo_id), target_dir)
            valid_after, issues_after = model_dir_is_valid(target_dir)
            if not valid_after:
                raise RuntimeError(f"post-download validation failed: {issues_after}")
            logger.info(f"[OK] installed {model_id}")
            installed += 1
        except Exception as exc:
            logger.error(f"[FAIL] {model_id}: {exc}")
            failed += 1
            if args.strict:
                logger.error("[FAIL] strict mode enabled; stopping on first failure")
                break

    logger.info(f"[DONE] installed={installed} skipped={skipped} failed={failed}")
    if failed > 0 and args.strict:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
