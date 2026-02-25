#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


def run_cmd(cmd: List[str], cwd: Path) -> Tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    output = ((proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")).strip()
    return proc.returncode, output


def check_model_config(config_path: Path) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    errors: List[str] = []
    warnings: List[str] = []
    models_info: List[Dict[str, Any]] = []

    if not config_path.exists():
        return [f"missing config: {config_path}"], warnings, models_info

    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"invalid json in {config_path}: {exc}"], warnings, models_info

    models = cfg.get("models")
    if not isinstance(models, list) or not models:
        return [f"config has no valid 'models' list: {config_path}"], warnings, models_info

    seen_ids = set()
    for idx, model in enumerate(models):
        if not isinstance(model, dict):
            errors.append(f"models[{idx}] must be object")
            continue
        model_id = model.get("id")
        provider = model.get("provider")
        kind = model.get("kind")
        if not isinstance(model_id, str) or not model_id.strip():
            errors.append(f"models[{idx}] invalid id")
            continue
        if model_id in seen_ids:
            errors.append(f"duplicate model id: {model_id}")
            continue
        seen_ids.add(model_id)
        if provider not in {"local_hf", "api_over_mcp"}:
            warnings.append(f"model '{model_id}' uses unknown provider '{provider}'")
        models_info.append({"id": model_id, "provider": provider, "kind": kind, "hf_repo": model.get("hf_repo")})

    return errors, warnings, models_info


def smoke_load_text_model(model_dir: Path) -> Tuple[bool, str]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        return False, f"transformers import failed: {exc}"
    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        # tiny generation to ensure forward path works
        prompt = "Return exactly: OK"
        inputs = tok(prompt, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=3)
        text = tok.decode(out[0], skip_special_tokens=True)
        if not text:
            return False, "empty generation output"
        return True, "text model load+generate ok"
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Safe preflight gate for baseline evaluation.")
    parser.add_argument("--config", default="configs/baselines/minimal_models.json")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--output", default="logs/preflight_report.json")
    parser.add_argument(
        "--smoke-load-text-id",
        default="text_qwen25_3b_instruct",
        help="Model id (kind=text_llm, provider=local_hf) to smoke-load.",
    )
    parser.add_argument(
        "--skip-smoke-load",
        action="store_true",
        help="Skip heavy local model loading smoke test.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_path = (repo_root / args.config).resolve()
    models_root = (repo_root / args.models_root).resolve()

    report: Dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "paths": {
            "repo_root": str(repo_root),
            "config": str(config_path),
            "models_root": str(models_root),
        },
        "checks": {},
        "status": "pass",
        "hard_failures": [],
        "warnings": [],
    }

    # Existing integrity checks
    rc, out = run_cmd([sys.executable, "scripts/verify_baseline_integrity.py"], repo_root)
    report["checks"]["verify_baseline_integrity"] = {"exit_code": rc, "output": out}
    if rc != 0:
        report["hard_failures"].append("verify_baseline_integrity failed")

    rc, out = run_cmd([sys.executable, "scripts/verify_runtime_setup.py"], repo_root)
    report["checks"]["verify_runtime_setup"] = {"exit_code": rc, "output": out}
    if rc != 0:
        report["hard_failures"].append("verify_runtime_setup failed")

    config_errors, config_warnings, models_info = check_model_config(config_path)
    report["checks"]["model_config"] = {"errors": config_errors, "warnings": config_warnings, "models": models_info}
    report["hard_failures"].extend(config_errors)
    report["warnings"].extend(config_warnings)

    # Validate local_hf models have files
    local_model_issues: List[str] = []
    for model in models_info:
        if model.get("provider") != "local_hf":
            continue
        model_dir = models_root / model["id"]
        if not model_dir.exists():
            local_model_issues.append(f"missing model directory: {model_dir}")
            continue
        if not (model_dir / "config.json").exists():
            local_model_issues.append(f"missing config.json for {model['id']}")
        has_weights = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))
        if not has_weights:
            local_model_issues.append(f"missing weights for {model['id']}")
    report["checks"]["local_model_files"] = {"issues": local_model_issues}
    report["hard_failures"].extend(local_model_issues)

    # Optional text model smoke load
    smoke_result: Dict[str, Any] = {"skipped": bool(args.skip_smoke_load)}
    if not args.skip_smoke_load:
        smoke_model_dir = models_root / args.smoke_load_text_id
        ok, detail = smoke_load_text_model(smoke_model_dir)
        smoke_result = {"skipped": False, "model_id": args.smoke_load_text_id, "ok": ok, "detail": detail}
        if not ok:
            report["hard_failures"].append(f"text model smoke load failed: {detail}")
    report["checks"]["text_model_smoke_load"] = smoke_result

    if report["hard_failures"]:
        report["status"] = "fail"

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[INFO] wrote preflight report: {output_path}")
    print(f"[INFO] status: {report['status']}")
    for item in report["warnings"]:
        print(f"[WARN] {item}")
    for item in report["hard_failures"]:
        print(f"[FAIL] {item}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
