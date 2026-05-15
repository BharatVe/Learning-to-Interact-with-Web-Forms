#!/usr/bin/env python3
import argparse
import json
import shutil
import socket
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baselines import run_opencua_direct_eval as opencua_eval  # noqa: E402
from baselines.model_registry import get_model_by_id  # noqa: E402
from engine.runner import iter_run_specs, resolve_answers_path  # noqa: E402


def _parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _check_command(name: str) -> bool:
    return bool(shutil.which(name))


def _http_get_json(url: str, timeout_s: float) -> Dict[str, Any]:
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=max(1.0, float(timeout_s))) as response:
        raw = response.read().decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("response_not_object")
    return payload


def _endpoint_tcp_reachable(base_url: str, timeout_s: float = 3.0) -> str:
    parsed = urllib.parse.urlparse(base_url if "://" in base_url else f"http://{base_url}")
    host = parsed.hostname
    if not host:
        raise RuntimeError("invalid_base_url_host")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    with socket.create_connection((host, int(port)), timeout=float(timeout_s)):
        return f"{host}:{port}"


def _verify_config(model_cfg: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    expected = {
        "kind": "computer_use_agent",
        "provider": "openai_compat",
        "track": "computer_use_native",
        "server_backend": "vllm",
        "coordinate_type": "qwen25",
    }
    for key, value in expected.items():
        got = str(model_cfg.get(key) or "").strip()
        if got != value:
            errors.append(f"config mismatch: {key}={got!r}, expected {value!r}")

    served_model_name = str(model_cfg.get("served_model_name") or model_cfg.get("openai_model") or "").strip()
    if not served_model_name:
        errors.append("config missing served_model_name/openai_model")
    elif served_model_name != "opencua-32b":
        warnings.append(f"served model name is {served_model_name!r}, expected canonical 'opencua-32b'")

    hf_repo = str(model_cfg.get("hf_repo") or "").strip()
    if not hf_repo:
        errors.append("config missing hf_repo")
    elif hf_repo != "xlangai/OpenCUA-32B":
        warnings.append(f"hf_repo is {hf_repo!r}, expected thesis-primary 'xlangai/OpenCUA-32B'")


def _verify_forms_and_answers(form_ids: Sequence[str], run_indexes: Sequence[int], errors: List[str]) -> None:
    forms_root = REPO_ROOT / "src" / "forms"
    for form_id in form_ids:
        spec_path = forms_root / form_id / "spec.json"
        if not spec_path.exists():
            errors.append(f"missing form spec: {spec_path}")
            continue
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid form spec JSON {spec_path}: {exc}")
            continue
        form_url = str(spec.get("form_url") or spec.get("url") or "").strip()
        if not form_url:
            errors.append(f"form spec missing form_url/url: {spec_path}")

        answers_path = resolve_answers_path(argparse.Namespace(answers_root="data/answers", answers_file="runs.json"), form_id)
        runs = list(iter_run_specs(answers_path))
        if not runs:
            errors.append(f"no answer runs found: {answers_path}")
            continue
        for run_index in run_indexes:
            if run_index < 1 or run_index > len(runs):
                errors.append(f"run index out of range for {form_id}: {run_index} (available {len(runs)})")
                continue
            answers = runs[run_index - 1].get("answers", [])
            if not isinstance(answers, list) or not answers:
                errors.append(f"empty answers for {form_id} run {run_index}")
                continue
            for answer_idx, answer in enumerate(answers, start=1):
                if not isinstance(answer, dict):
                    errors.append(f"{form_id} run {run_index} answer {answer_idx} is not an object")
                    continue
                for key in ("label", "widget_type", "value"):
                    if key not in answer:
                        errors.append(f"{form_id} run {run_index} answer {answer_idx} missing key {key!r}")


def _verify_parser_and_coords(errors: List[str]) -> None:
    action, _ = opencua_eval._parse_opencua_action(
        "pyautogui.click(x=960, y=324)\npyautogui.write('Alice Example')",
        viewport_width=1440,
        viewport_height=900,
        coordinate_type="qwen25",
    )
    if action.get("action") != "type_text":
        errors.append("OpenCUA parser failed to fold click+write into type_text")
    if "target" not in action:
        errors.append("OpenCUA parser produced no target for click+write")

    hotkey_action, _ = opencua_eval._parse_opencua_action(
        "pyautogui.hotkey('ctrl', 'a')",
        viewport_width=1440,
        viewport_height=900,
        coordinate_type="qwen25",
    )
    if hotkey_action != {"action": "press_key", "value": "Control+A"}:
        errors.append(f"OpenCUA hotkey normalization mismatch: {hotkey_action!r}")

    abs_x, abs_y, meta = opencua_eval._qwen25_smart_resize_to_abs(960, 324, 1440, 900)
    if not (0 <= abs_x <= 1440 and 0 <= abs_y <= 900):
        errors.append("coordinate transform produced out-of-bounds coordinates")
    if str(meta.get("coordinate_space") or "") != "qwen25_smart_resize_absolute":
        errors.append("coordinate transform metadata missing qwen25_smart_resize_absolute")


def _verify_prompt_contract(errors: List[str]) -> None:
    prompt = opencua_eval._build_goal_prompt(
        form_url="https://example.test/form",
        remaining_answers=[{"label": "Full name", "value": "Alice Example"}],
        last_result={},
        interaction_map=[{"label": "Full name", "ref": "e1"}],
        page_text="Full name",
        observation_mode="vision_coords",
        recent_history=[],
    )
    if "Interaction map" in prompt or '"ref": "e1"' in prompt:
        errors.append("OpenCUA default prompt includes symbolic interaction-map support")
    if "pyautogui.click" not in prompt or "pyautogui.write" not in prompt:
        errors.append("OpenCUA prompt does not advertise pyautogui-style action outputs")

    ablation_prompt = opencua_eval._build_goal_prompt(
        form_url="https://example.test/form",
        remaining_answers=[{"label": "Full name", "value": "Alice Example"}],
        last_result={},
        interaction_map=[{"label": "Full name", "ref": "e1"}],
        page_text="Full name",
        observation_mode="vision_coords",
        recent_history=[],
        include_symbolic_support=True,
    )
    if "Interaction map" not in ablation_prompt:
        errors.append("OpenCUA symbolic-support ablation prompt did not include interaction map")


def _verify_runtime(base_url: str, require_endpoint: bool, errors: List[str], warnings: List[str]) -> None:
    for command in ("vllm", "playwright-mcp"):
        if not _check_command(command):
            message = f"required command not found on PATH: {command}"
            if require_endpoint:
                errors.append(message)
            else:
                warnings.append(message)

    try:
        endpoint = _endpoint_tcp_reachable(base_url)
    except Exception as exc:
        if require_endpoint:
            errors.append(f"OpenCUA endpoint unreachable at {base_url}: {exc}")
        else:
            warnings.append(f"OpenCUA endpoint not reachable yet at {base_url}: {exc}")
        return

    try:
        payload = _http_get_json(f"{base_url.rstrip('/')}/models", timeout_s=5.0)
    except Exception as exc:
        if require_endpoint:
            errors.append(f"OpenCUA endpoint reachable at {endpoint} but /models failed: {exc}")
        else:
            warnings.append(f"OpenCUA endpoint reachable at {endpoint} but /models failed: {exc}")
        return

    models = payload.get("data")
    if not isinstance(models, list) or not models:
        if require_endpoint:
            errors.append("OpenCUA endpoint /models returned no model list")
        else:
            warnings.append("OpenCUA endpoint /models returned no model list")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify OpenCUA benchmark compatibility before running evaluation.")
    parser.add_argument("--config", default="configs/baselines/track_baseline_models.json")
    parser.add_argument("--model-id", default="computer_use_opencua_32b")
    parser.add_argument("--form-ids", default="conf_interest,event_rsvp,course_feedback,internship_app,workshop_signup")
    parser.add_argument("--run-indexes", default="1,2,3")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--require-endpoint", action="store_true")
    args = parser.parse_args()

    errors: List[str] = []
    warnings: List[str] = []

    config_path = (REPO_ROOT / args.config).resolve()
    if not config_path.exists():
        print(f"[FAIL] config_path does not exist: {config_path}")
        return 1

    model_cfg = get_model_by_id(config_path, args.model_id)
    form_ids = _parse_csv(args.form_ids)
    run_indexes = [int(item) for item in _parse_csv(args.run_indexes)]
    if not form_ids:
        errors.append("no form_ids resolved")
    if not run_indexes:
        errors.append("no run_indexes resolved")

    _verify_config(model_cfg, errors, warnings)
    _verify_forms_and_answers(form_ids, run_indexes, errors)
    _verify_parser_and_coords(errors)
    _verify_prompt_contract(errors)
    _verify_runtime(args.base_url, bool(args.require_endpoint), errors, warnings)

    print(f"[INFO] config_path={config_path}")
    print(f"[INFO] model_id={args.model_id}")
    print(f"[INFO] form_count={len(form_ids)}")
    print(f"[INFO] run_indexes={','.join(str(x) for x in run_indexes)}")
    print(f"[INFO] base_url={args.base_url}")
    if warnings:
        print(f"[WARN] warnings={len(warnings)}")
        for item in warnings:
            print(f"[WARN] {item}")
    if errors:
        print(f"[FAIL] compatibility_errors={len(errors)}")
        for item in errors:
            print(f"[FAIL] {item}")
        return 1
    print("[PASS] OpenCUA compatibility checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
