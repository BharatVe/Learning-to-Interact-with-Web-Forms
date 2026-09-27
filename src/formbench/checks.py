"""Failsafes: `formbench models check` and `formbench doctor`.

Every check yields a CheckResult(status, name, message, hint). `fail` blocks a run
(unless --skip-checks), `warn` is printed but does not block.
"""

import importlib.util
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from baselines.model_registry import local_weights_dir, validate_config
from formbench.common import dump_table
from formbench.protocols import protocol_for, uses_vllm_python
from formbench.serve import endpoint_for, list_served_models, smoke_chat
from formbench.settings import Settings

OK, WARN, FAIL = "ok", "warn", "FAIL"


@dataclass
class CheckResult:
    status: str
    name: str
    message: str
    hint: str = ""


def _result_table(results: List[CheckResult]) -> str:
    from formbench.common import short

    rows = [{"status": r.status, "check": r.name, "detail": short(r.message + (f"  -> {r.hint}" if r.hint and r.status != OK else ""))} for r in results]
    return dump_table(rows, ["status", "check", "detail"])


def print_results(title: str, results: List[CheckResult]) -> bool:
    print(f"\n== {title}")
    print(_result_table(results))
    return not any(r.status == FAIL for r in results)


def visible_gpus() -> List[Dict[str, Any]]:
    """GPUs visible to this process via nvidia-smi (respects CUDA_VISIBLE_DEVICES)."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            gpus.append({"index": parts[0], "name": parts[1], "memory_gb": round(float(parts[2]) / 1024, 1)})
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() != "":
        wanted = {v.strip() for v in visible.split(",")}
        gpus = [g for g in gpus if g["index"] in wanted]
    return gpus


def _api_key_check(settings: Settings, model: Mapping[str, Any]) -> Optional[CheckResult]:
    provider = model.get("provider")
    if provider == "gemini_low_cost":
        env_name = str(model.get("api_key_env") or "GEMINI_API_KEY")
        if os.environ.get(env_name):
            return CheckResult(OK, "api key", f"{env_name} is set")
        key_file = Path(os.environ.get("GEMINI_API_KEY_FILE") or settings.gemini_api_key_file)
        if key_file.is_file() and key_file.stat().st_size > 0:
            mode = key_file.stat().st_mode & 0o777
            if mode & 0o077:
                return CheckResult(WARN, "api key", f"{key_file} is readable by others (mode {oct(mode)})", f"chmod 600 {key_file}")
            return CheckResult(OK, "api key", f"key file {key_file}")
        return CheckResult(
            FAIL, "api key", f"{env_name} unset and {key_file} missing/empty",
            f"mkdir -p .secrets && chmod 700 .secrets && printf '%s' '<key>' > {key_file} && chmod 600 {key_file}",
        )
    if provider == "api_over_mcp":
        env_name = model.get("api_key_env")
        if env_name:
            ok = bool(os.environ.get(str(env_name)))
            return CheckResult(OK if ok else FAIL, "api key", f"{env_name} {'is set' if ok else 'is not set'}", f"export {env_name}=<key> (never put keys in configs/)")
        have = [name for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if os.environ.get(name)]
        if have:
            return CheckResult(OK, "api key", f"{', '.join(have)} set (runner auto-selects provider)")
        return CheckResult(FAIL, "api key", "neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set", "export one of them")
    if provider == "openai_compat":
        endpoint = endpoint_for(model)
        if not endpoint.managed and endpoint.api_key == "EMPTY":
            return CheckResult(FAIL, "api key", f"remote endpoint {endpoint.base_url} but OPENAI_API_KEY is unset (would send 'EMPTY')", "export OPENAI_API_KEY=<key>")
    return None


def check_model(
    settings: Settings,
    model: Mapping[str, Any],
    registry_errors: Optional[List[str]] = None,
    registry_warnings: Optional[List[str]] = None,
    probe_endpoint: bool = True,
    smoke: bool = False,
    expect_gpus_here: bool = False,
) -> List[CheckResult]:
    mid = str(model.get("id"))
    results: List[CheckResult] = []
    own_errors = [e for e in (registry_errors or []) if e.startswith(mid + ":")]
    own_warnings = [w for w in (registry_warnings or []) if w.startswith(mid + ":")]
    if own_errors:
        results += [CheckResult(FAIL, "registry", e.split(": ", 1)[1], f"fix the entry in {settings.models_config}") for e in own_errors]
    else:
        results.append(CheckResult(OK, "registry", f"{model.get('kind')} / {model.get('provider')} / {model.get('track')} ({model.get('status', 'current')})"))
    results += [CheckResult(WARN, "registry", w.split(": ", 1)[1]) for w in own_warnings]
    if model.get("status") == "legacy":
        results.append(CheckResult(WARN, "status", "legacy model: kept to reproduce earlier experiments"))

    try:
        protocol = protocol_for(model)
        results.append(CheckResult(OK, "protocol", f"{protocol.name} -> {protocol.script}"))
    except Exception as exc:  # noqa: BLE001
        results.append(CheckResult(FAIL, "protocol", str(exc)))

    key_result = _api_key_check(settings, model)
    if key_result:
        results.append(key_result)

    provider = model.get("provider")
    weights = local_weights_dir(dict(model), repo_root=settings.root, models_dir=settings.models_dir)
    if provider == "local_hf":
        if weights and (weights / "config.json").is_file():
            results.append(CheckResult(OK, "weights", str(weights)))
        else:
            results.append(CheckResult(FAIL, "weights", f"no local weights for {mid}", f"make install-models MODEL={mid}"))
    elif provider == "openai_compat" and isinstance(model.get("serve"), dict):
        if weights:
            results.append(CheckResult(OK, "weights", str(weights)))
        else:
            results.append(CheckResult(WARN, "weights", f"no local copy; vLLM will download {model.get('hf_repo')} into HF_HOME at start", f"pre-fetch: make install-models MODEL={mid}"))

    python = settings.vllm_python_bin if uses_vllm_python(model) or (provider == "openai_compat" and isinstance(model.get("serve"), dict)) else settings.python_bin
    results.append(CheckResult(OK if Path(python).exists() else FAIL, "interpreter", str(python), "run `make setup` or set PYTHON_BIN / VLLM_PYTHON_BIN in .env"))

    if provider == "openai_compat":
        endpoint = endpoint_for(model)
        for override in endpoint.overrides:
            results.append(CheckResult(WARN, "env override", override, "unset the variable to use the registry value"))
        if probe_endpoint:
            served = list_served_models(endpoint, timeout=5)
            if served is None:
                if endpoint.managed:
                    results.append(CheckResult(OK, "endpoint", f"{endpoint.base_url} not running (started automatically by eval/matrix)"))
                else:
                    results.append(CheckResult(FAIL, "endpoint", f"{endpoint.base_url} unreachable", "check the URL / VPN, or add a serve block to run it locally"))
            elif endpoint.model_name not in served:
                results.append(CheckResult(FAIL, "endpoint", f"{endpoint.base_url} serves {served}, not {endpoint.model_name!r}", "fix openai_model/served_model_name or stop the other server"))
            else:
                results.append(CheckResult(OK, "endpoint", f"{endpoint.base_url} serves {endpoint.model_name}"))
                if smoke:
                    try:
                        seconds = smoke_chat(endpoint, str(model.get("kind")), timeout=120)
                        results.append(CheckResult(OK, "smoke chat", f"{seconds:.1f}s"))
                    except Exception as exc:  # noqa: BLE001
                        results.append(CheckResult(FAIL, "smoke chat", str(exc)))

    if model.get("requires_gpu"):
        resources = model.get("resources") or {}
        need = int(resources.get("gpus") or (model.get("serve") or {}).get("tensor_parallel") or 1)
        gpus = visible_gpus()
        if not gpus:
            status = FAIL if expect_gpus_here else WARN
            results.append(CheckResult(status, "gpu", f"needs {need} GPU(s); none visible on this node", "run through Slurm: make submit ... (or LOCAL=1 on a GPU node)"))
        elif len(gpus) < need:
            results.append(CheckResult(FAIL if expect_gpus_here else WARN, "gpu", f"needs {need} GPU(s), {len(gpus)} visible", "request more GPUs (resources.gpus) or lower serve.tensor_parallel"))
        else:
            min_vram = resources.get("min_vram_gb")
            smallest = min(g["memory_gb"] for g in gpus[:need])
            if min_vram and smallest < float(min_vram):
                results.append(CheckResult(FAIL, "gpu", f"GPU memory {smallest} GB < min_vram_gb {min_vram}", "use larger GPUs or more tensor parallelism"))
            else:
                results.append(CheckResult(OK, "gpu", f"{len(gpus)} visible ({gpus[0]['name']}, {smallest} GB)"))
    return results


def check_models(settings: Settings, model_ids: List[str], probe_endpoint: bool = True, smoke: bool = False, expect_gpus_here: bool = False) -> bool:
    models, errors, warnings = validate_config(settings.models_config)
    global_errors = [e for e in errors if ":" not in e or not any(e.startswith(str(m.get("id")) + ":") for m in models)]
    if global_errors or not models:
        print_results("registry", [CheckResult(FAIL, "registry", e) for e in global_errors or [f"no models in {settings.models_config}"]])
        return False
    by_id = {str(m["id"]): m for m in models}
    ok = True
    for mid in model_ids:
        if mid not in by_id:
            print_results(mid, [CheckResult(FAIL, "registry", f"unknown model id {mid!r}", "make models  (lists valid ids)")])
            ok = False
            continue
        results = check_model(settings, by_id[mid], errors, warnings, probe_endpoint=probe_endpoint, smoke=smoke, expect_gpus_here=expect_gpus_here)
        ok = print_results(mid, results) and ok
    return ok


def _module_available(python: Path, module: str, env: Optional[Dict[str, str]] = None) -> bool:
    """Is `module` installed for `python`? Uses find_spec (no heavy import, fast on network filesystems)."""
    code = f"import importlib.util, sys; sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"
    try:
        out = subprocess.run([str(python), "-c", code], capture_output=True, timeout=60, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0


def doctor(settings: Settings) -> bool:
    results: List[CheckResult] = []
    import sys

    results.append(CheckResult(OK if sys.version_info >= (3, 9) else FAIL, "python", f"{sys.executable} {sys.version.split()[0]}", "Python >= 3.9 required"))
    results.append(CheckResult(OK if settings.python_bin.exists() else FAIL, "venv", str(settings.python_bin), "make setup"))
    for module, why in [("playwright", "browser automation"), ("flask", "LocalForms site"), ("PIL", "screenshots"), ("numpy", "analysis"), ("matplotlib", "analysis figures")]:
        present = importlib.util.find_spec(module) is not None
        results.append(CheckResult(OK if present else (WARN if module == "matplotlib" else FAIL), f"py:{module}", why if present else f"not importable ({why})", "make setup"))

    vllm_env = dict(os.environ)
    if os.environ.get("MODULE_LD_LIBRARY_PATH"):
        vllm_env["LD_LIBRARY_PATH"] = os.environ["MODULE_LD_LIBRARY_PATH"]
    if settings.vllm_python_bin.exists():
        has_vllm = _module_available(settings.vllm_python_bin, "vllm", vllm_env)
        results.append(CheckResult(OK if has_vllm else WARN, "vllm env", f"{settings.vllm_python_bin} (vllm {'ok' if has_vllm else 'not importable'})", "needed only for locally served models; set MODULES in .env so libpython resolves"))
    else:
        results.append(CheckResult(WARN, "vllm env", f"{settings.vllm_python_bin} missing", "needed only for locally served models (Qwen/OpenCUA); see docs/HPC.md"))

    node = shutil.which("node")
    results.append(CheckResult(OK if node else FAIL, "node", node or "not on PATH", "set MODULES (e.g. nodejs/20.13.1) in .env, or install Node >= 18"))
    mcp_bin = settings.node_tools_dir / "node_modules" / ".bin" / "playwright-mcp"
    results.append(CheckResult(OK if mcp_bin.exists() else FAIL, "playwright-mcp", str(mcp_bin), "make setup"))
    record = settings.playwright_browsers_path / ".mcp-chromium-executable"
    if record.is_file() and Path(record.read_text().strip()).exists():
        results.append(CheckResult(OK, "mcp chromium", record.read_text().strip()))
    else:
        results.append(CheckResult(FAIL, "mcp chromium", f"not installed under {settings.playwright_browsers_path}", "make setup"))

    try:
        settings.cache_root.mkdir(parents=True, exist_ok=True)
        probe = settings.cache_root / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
        usage = shutil.disk_usage(settings.cache_root)
        free_gb = usage.free / 1e9
        results.append(CheckResult(OK if free_gb > 20 else WARN, "cache", f"{settings.cache_root} ({free_gb:.0f} GB free)", "set CACHE_ROOT in .env to a larger workspace"))
    except OSError as exc:
        results.append(CheckResult(FAIL, "cache", f"{settings.cache_root} not writable: {exc}", "set CACHE_ROOT in .env"))

    gpus = visible_gpus()
    results.append(CheckResult(OK if gpus else WARN, "gpu", f"{len(gpus)} visible" + (f" ({gpus[0]['name']})" if gpus else ""), "fine on a login node; GPU models run via `make submit`"))
    results.append(CheckResult(OK, "slurm", "sbatch available" if settings.slurm_available() else "not available (commands run locally)"))

    models, errors, warnings = validate_config(settings.models_config)
    results.append(CheckResult(FAIL if errors else OK, "registry", f"{len(models)} models in {settings.models_config}" + (f"; {len(errors)} error(s)" if errors else ""), "make model-check MODEL=<id>"))
    for warning in warnings:
        results.append(CheckResult(WARN, "registry", warning))

    from formbench.common import discover_form_ids

    forms = discover_form_ids(settings.forms_root)
    answers = [f for f in forms if (settings.answers_root / f / "runs.json").is_file()]
    results.append(CheckResult(OK if forms and len(answers) == len(forms) else FAIL, "dataset", f"{len(forms)} form specs, {len(answers)} with answer sets", "make data"))
    lf_forms = discover_form_ids(settings.localforms_forms_root)
    results.append(CheckResult(OK if len(lf_forms) == len(forms) else WARN, "localforms", f"{len(lf_forms)} LocalForms specs", "make data"))
    refs = sum(1 for f in forms for _ in (settings.reference_root / f / "runs").glob("run_*/tool_trace.jsonl")) if forms else 0
    results.append(CheckResult(OK if refs else WARN, "ideal runs", f"{refs} reference runs with tool_trace.jsonl", "make ideal-runs"))
    return print_results("doctor", results)
