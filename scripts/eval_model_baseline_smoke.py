#!/usr/bin/env python3
import argparse
import json
import os
import socket
import shutil
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _trim(text: str, max_chars: int = 240) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def eval_text_model(model_dir: Path, max_new_tokens: int) -> Tuple[bool, Dict[str, Any]]:
    start = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        return False, {"error": f"transformers import failed: {exc}"}

    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        prompt = "Return exactly one word: OK"
        inputs = tok(prompt, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = tok.decode(out[0], skip_special_tokens=True)
        return True, {
            "latency_s": round(time.time() - start, 3),
            "output_preview": _trim(text),
        }
    except Exception as exc:
        return False, {"latency_s": round(time.time() - start, 3), "error": str(exc)}


def eval_vlm_model_load(model_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    start = time.time()
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except Exception as exc:
        return False, {"error": f"transformers import failed: {exc}"}
    try:
        _ = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        _ = AutoModelForVision2Seq.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        return True, {"latency_s": round(time.time() - start, 3), "detail": "load ok"}
    except Exception as exc:
        return False, {"latency_s": round(time.time() - start, 3), "error": str(exc)}


def eval_api_over_mcp() -> Tuple[bool, Dict[str, Any]]:
    node_tools = {name: bool(shutil.which(name)) for name in ["node", "npm", "npx"]}
    api_keys = {
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
    }
    ok = all(node_tools.values()) and any(api_keys.values())
    detail = {
        "node_tools": node_tools,
        "api_keys_present": api_keys,
    }
    if ok:
        detail["detail"] = "runtime prerequisites detected"
    else:
        detail["detail"] = "missing prerequisites for api_over_mcp baseline"
    return ok, detail


def eval_gemini_native(model: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    node_tools = {name: bool(shutil.which(name)) for name in ["node", "npm", "npx"]}
    api_key_present = bool(os.getenv("GEMINI_API_KEY"))
    model_name = str(os.getenv("GEMINI_MODEL") or model.get("gemini_model") or "").strip()
    expected_model = "gemini-2.5-computer-use-preview-10-2025"
    sdk_ok = False
    sdk_error = None
    try:
        from google import genai  # noqa: F401

        sdk_ok = True
    except Exception as exc:
        sdk_error = str(exc)
    model_valid = model_name == expected_model
    ok = all(node_tools.values()) and api_key_present and sdk_ok and bool(model_name) and model_valid
    detail = {
        "node_tools": node_tools,
        "gemini_api_key_present": api_key_present,
        "gemini_model_configured": bool(model_name),
        "gemini_model": model_name or None,
        "expected_gemini_model": expected_model,
        "gemini_model_valid_for_native_computer_use": model_valid,
        "google_genai_import_ok": sdk_ok,
        "google_genai_import_error": sdk_error,
    }
    if ok:
        detail["detail"] = "runtime prerequisites detected for gemini_native"
    else:
        detail["detail"] = "missing prerequisites or wrong model for gemini_native baseline"
    return ok, detail


def eval_gemini_low_cost(model: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    node_tools = {name: bool(shutil.which(name)) for name in ["node", "npm", "npx"]}
    key_file = Path(os.getenv("GEMINI_API_KEY_FILE") or ".secrets/gemini_api_key")
    api_key_present = bool(os.getenv("GEMINI_API_KEY"))
    key_file_present = key_file.exists() and bool(key_file.read_text(encoding="utf-8").strip())
    model_name = str(os.getenv("GEMINI_MODEL") or model.get("gemini_model") or "").strip()
    ok = all(node_tools.values()) and (api_key_present or key_file_present) and bool(model_name)
    detail = {
        "node_tools": node_tools,
        "gemini_api_key_present": api_key_present,
        "gemini_api_key_file_present": key_file_present,
        "gemini_model_configured": bool(model_name),
        "gemini_model": model_name or None,
        "detail": "runtime prerequisites detected for gemini_low_cost"
        if ok
        else "missing prerequisites for gemini_low_cost baseline",
    }
    return ok, detail


def eval_openai_compat(model: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    def endpoint_reachable(base_url: str, timeout_s: float = 3.0) -> Tuple[bool, str]:
        raw = str(base_url or "").strip()
        if not raw:
            return False, "missing_base_url"
        parsed = urllib.parse.urlparse(raw if "://" in raw else f"http://{raw}")
        host = parsed.hostname
        if not host:
            return False, "invalid_base_url_host"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, int(port)), timeout=float(timeout_s)):
                return True, f"tcp_ok:{host}:{port}"
        except Exception as exc:
            return False, f"tcp_unreachable:{host}:{port}:{exc}"

    base_url = str(os.getenv("OPENAI_BASE_URL") or model.get("openai_base_url") or "").strip()
    model_name = str(os.getenv("OPENAI_MODEL") or model.get("openai_model") or "").strip()
    api_key_present = bool(os.getenv("OPENAI_API_KEY") or model.get("openai_api_key"))
    endpoint_ok, endpoint_detail = endpoint_reachable(base_url)
    ok = bool(base_url and model_name and endpoint_ok)
    detail = {
        "base_url_configured": bool(base_url),
        "model_configured": bool(model_name),
        "api_key_present": api_key_present,
        "endpoint_reachable": endpoint_ok,
        "endpoint_detail": endpoint_detail,
        "detail": "openai_compat prerequisites detected"
        if ok
        else "missing OPENAI_BASE_URL/OPENAI_MODEL or endpoint unreachable",
    }
    if base_url:
        detail["base_url"] = base_url
    if model_name:
        detail["model"] = model_name
    return ok, detail


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-evaluate configured baseline models.")
    parser.add_argument("--config", default="configs/baselines/minimal_models.json")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--output", default="logs/model_baseline_smoke.json")
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--include-kinds", default="", help="Comma-separated model kinds to evaluate.")
    parser.add_argument("--exclude-providers", default="", help="Comma-separated providers to defer.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero if any model check fails.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = (repo_root / args.config).resolve()
    models_root = (repo_root / args.models_root).resolve()
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] smoke_config={cfg_path}")
    if not cfg_path.exists():
        print(f"[FAIL] config not found: {cfg_path}")
        return 1

    try:
        cfg = load_config(cfg_path)
    except Exception as exc:
        print(f"[FAIL] failed to load config {cfg_path}: {exc}")
        return 1
    models = cfg.get("models", [])
    if not isinstance(models, list) or not models:
        print(f"[FAIL] no models configured in {cfg_path}")
        return 1

    include_kinds = set(_parse_csv(args.include_kinds))
    exclude_providers = set(_parse_csv(args.exclude_providers))
    if include_kinds:
        print(f"[INFO] include_kinds={sorted(include_kinds)}")
    if exclude_providers:
        print(f"[INFO] exclude_providers={sorted(exclude_providers)}")

    report: Dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "config_path": str(cfg_path),
        "models_root": str(models_root),
        "results": [],
        "summary": {"passed": 0, "failed": 0, "skipped": 0, "total": 0},
    }

    for model in models:
        model_id = str(model.get("id", ""))
        kind = str(model.get("kind", ""))
        provider = str(model.get("provider", ""))
        model_dir = models_root / model_id

        entry: Dict[str, Any] = {
            "id": model_id,
            "kind": kind,
            "provider": provider,
            "status": "skipped",
            "detail": {},
        }

        if include_kinds and kind not in include_kinds:
            entry["status"] = "deferred"
            entry["detail"] = {"reason": f"kind '{kind}' not selected by --include-kinds"}
            report["results"].append(entry)
            continue
        if provider in exclude_providers:
            entry["status"] = "deferred"
            entry["detail"] = {"reason": f"provider '{provider}' excluded by --exclude-providers"}
            report["results"].append(entry)
            continue

        if provider == "local_hf":
            if not model_dir.exists():
                entry["status"] = "failed"
                entry["detail"] = {"error": f"missing model directory: {model_dir}"}
            elif kind == "text_llm":
                ok, detail = eval_text_model(model_dir, args.max_new_tokens)
                entry["status"] = "passed" if ok else "failed"
                entry["detail"] = detail
            elif kind == "vlm":
                ok, detail = eval_vlm_model_load(model_dir)
                entry["status"] = "passed" if ok else "failed"
                entry["detail"] = detail
            else:
                entry["status"] = "skipped"
                entry["detail"] = {"reason": f"unsupported local_hf kind '{kind}'"}
        elif provider == "api_over_mcp":
            ok, detail = eval_api_over_mcp()
            entry["status"] = "passed" if ok else "failed"
            entry["detail"] = detail
        elif provider == "openai_compat":
            ok, detail = eval_openai_compat(model)
            entry["status"] = "passed" if ok else "failed"
            entry["detail"] = detail
        elif provider == "gemini_native":
            ok, detail = eval_gemini_native(model)
            entry["status"] = "passed" if ok else "failed"
            entry["detail"] = detail
        elif provider == "gemini_low_cost":
            ok, detail = eval_gemini_low_cost(model)
            entry["status"] = "passed" if ok else "failed"
            entry["detail"] = detail
        else:
            entry["status"] = "skipped"
            entry["detail"] = {"reason": f"unsupported provider '{provider}'"}

        report["results"].append(entry)

    for row in report["results"]:
        report["summary"]["total"] += 1
        if row["status"] == "passed":
            report["summary"]["passed"] += 1
        elif row["status"] == "failed":
            report["summary"]["failed"] += 1
        elif row["status"] == "deferred":
            report["summary"]["skipped"] += 1
        else:
            report["summary"]["skipped"] += 1

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[INFO] wrote model smoke report: {output_path}")
    print(
        "[INFO] summary: "
        f"passed={report['summary']['passed']} "
        f"failed={report['summary']['failed']} "
        f"skipped={report['summary']['skipped']} "
        f"total={report['summary']['total']}"
    )
    for row in report["results"]:
        print(f"[{row['status'].upper()}] {row['id']} ({row['provider']}/{row['kind']})")

    if args.strict and report["summary"]["failed"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
