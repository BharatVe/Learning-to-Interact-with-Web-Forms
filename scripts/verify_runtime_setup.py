#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


REQUIRED_PYTHON_PACKAGES = [
    "playwright",
    "huggingface_hub",
    "transformers",
    "accelerate",
    "PIL",
    "torchvision",
]


def check_python_packages() -> Tuple[List[str], List[str]]:
    ok: List[str] = []
    missing: List[str] = []
    for module_name in REQUIRED_PYTHON_PACKAGES:
        try:
            spec = importlib.util.find_spec(module_name)
        except Exception:
            spec = None
        if spec is None:
            missing.append(module_name)
        else:
            ok.append(module_name)
    return ok, missing


def check_model_dir(model_dir: Path) -> Tuple[bool, List[str]]:
    problems: List[str] = []
    if not model_dir.exists():
        return False, ["missing directory"]
    if not (model_dir / "config.json").exists():
        problems.append("missing config.json")
    has_weights = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))
    if not has_weights:
        problems.append("missing weight files")
    return len(problems) == 0, problems


def check_model_runtime_compatibility(model_dir: Path) -> Tuple[bool, str]:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return False, "missing config.json"
    try:
        config_payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"invalid config.json: {exc}"
    model_type = str(config_payload.get("model_type") or "")

    try:
        from transformers import AutoConfig
    except Exception as exc:
        return False, f"transformers import failed: {exc}"

    try:
        AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    except Exception as exc:
        return (
            False,
            "runtime_incompatible: "
            f"model_type={model_type or 'unknown'}; error={exc}. "
            "Upgrade transformers to a version that supports this architecture.",
        )
    return True, f"ok:model_type={model_type or 'unknown'}"


def get_playwright_version() -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        text = (result.stdout or result.stderr or "").strip()
        return text if text else "unknown"
    except Exception:
        return "unknown"


def playwright_smoke_check(repo_root: Path, timeout_s: int) -> Tuple[bool, str]:
    env = dict(os.environ)
    env.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(repo_root / ".playwright-browsers"))
    cmd = [
        sys.executable,
        "-c",
        (
            "from playwright.sync_api import sync_playwright; "
            "p=sync_playwright().start(); "
            "b=p.chromium.launch(headless=True); "
            "b.close(); "
            "p.stop(); "
            "print('ok')"
        ),
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(5, timeout_s),
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_s}s"
    except Exception as exc:
        return False, str(exc)
    output = ((proc.stdout or "") + " " + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        return False, output or f"exit code {proc.returncode}"
    if "ok" not in output:
        return False, output or "unexpected empty output"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify baseline runtime setup and model readiness.")
    parser.add_argument("--config", default="configs/baselines/minimal_models.json")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--output", default="logs/setup_report.json")
    parser.add_argument(
        "--skip-playwright-smoke",
        action="store_true",
        help="Skip Playwright startup/browser launch smoke check.",
    )
    parser.add_argument(
        "--playwright-smoke-timeout-s",
        type=int,
        default=30,
        help="Timeout in seconds for Playwright smoke check.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve()
    models_root = (repo_root / args.models_root).resolve()
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "paths": {
            "repo_root": str(repo_root),
            "config": str(config_path),
            "models_root": str(models_root),
        },
        "python": {
            "version": sys.version.split()[0],
            "playwright_version": get_playwright_version(),
        },
        "checks": {},
    }

    py_ok, py_missing = check_python_packages()
    report["checks"]["python_packages"] = {"ok": py_ok, "missing": py_missing}

    node_tools = {name: bool(shutil.which(name)) for name in ["node", "npm", "npx"]}
    report["checks"]["node_tools"] = node_tools

    smoke_check = {"skipped": bool(args.skip_playwright_smoke), "ok": None, "detail": None}
    if not args.skip_playwright_smoke:
        ok, detail = playwright_smoke_check(repo_root, args.playwright_smoke_timeout_s)
        smoke_check = {"skipped": False, "ok": ok, "detail": detail}
    report["checks"]["playwright_smoke"] = smoke_check

    config_errors: List[str] = []
    models_report: List[Dict[str, Any]] = []
    if not config_path.exists():
        config_errors.append(f"missing config: {config_path}")
    else:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        models = cfg.get("models", [])
        if not isinstance(models, list):
            config_errors.append("config 'models' must be a list")
            models = []
        for model in models:
            model_id = str(model.get("id", ""))
            provider = str(model.get("provider", ""))
            kind = str(model.get("kind", ""))
            entry: Dict[str, Any] = {"id": model_id, "provider": provider, "kind": kind}
            if provider == "local_hf":
                valid, issues = check_model_dir(models_root / model_id)
                entry["local_ready"] = valid
                entry["issues"] = issues
                if valid:
                    compat_ok, compat_detail = check_model_runtime_compatibility(models_root / model_id)
                    entry["runtime_compat_ok"] = compat_ok
                    entry["runtime_compat_detail"] = compat_detail
                else:
                    entry["runtime_compat_ok"] = False
                    entry["runtime_compat_detail"] = "skipped_local_not_ready"
            elif provider == "openai_compat":
                issues: List[str] = []
                model_name = str(os.environ.get("OPENAI_MODEL") or model.get("openai_model") or "").strip()
                base_url = str(os.environ.get("OPENAI_BASE_URL") or model.get("openai_base_url") or "").strip()
                if not model_name:
                    issues.append("missing OPENAI_MODEL or model.openai_model")
                if not base_url:
                    issues.append("missing OPENAI_BASE_URL or model.openai_base_url")
                entry["local_ready"] = len(issues) == 0
                entry["issues"] = issues
                entry["runtime_compat_ok"] = len(issues) == 0
                entry["runtime_compat_detail"] = "openai_compat_ready" if not issues else "openai_compat_config_missing"
            else:
                entry["local_ready"] = True
                entry["issues"] = []
                entry["runtime_compat_ok"] = True
                entry["runtime_compat_detail"] = "not_applicable_non_local"
            models_report.append(entry)
    report["checks"]["config_errors"] = config_errors
    report["checks"]["models"] = models_report

    hard_failures: List[str] = []
    if py_missing:
        hard_failures.append(f"missing python packages: {py_missing}")
    if config_errors:
        hard_failures.extend(config_errors)
    if not smoke_check["skipped"] and not smoke_check["ok"]:
        hard_failures.append(f"playwright smoke check failed: {smoke_check['detail']}")
    for model in models_report:
        if model["provider"] == "local_hf" and not model["local_ready"]:
            hard_failures.append(f"model not ready: {model['id']} ({model['issues']})")
            continue
        if model["provider"] == "local_hf" and not bool(model.get("runtime_compat_ok")):
            hard_failures.append(f"model runtime incompatible: {model['id']} ({model.get('runtime_compat_detail')})")

    report["status"] = "pass" if not hard_failures else "fail"
    report["hard_failures"] = hard_failures
    report["warnings"] = []
    if not all(node_tools.values()):
        report["warnings"].append("node/npm/npx not fully available (mcp_server mode will not work)")

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[INFO] wrote setup report: {output_path}")
    print(f"[INFO] status: {report['status']}")
    if hard_failures:
        for item in hard_failures:
            print(f"[FAIL] {item}")
        return 1
    for item in report["warnings"]:
        print(f"[WARN] {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
