import argparse
import base64
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines import run_baseline_eval as rbe  # noqa: E402
from baselines.action_schema import validate_low_level_action  # noqa: E402
from baselines.model_registry import get_model_by_id  # noqa: E402
from baselines.prompt_builders import compact_page_text  # noqa: E402
from engine.browser_language import force_english_google_forms_url  # noqa: E402
from engine.runner import iter_run_specs, load_form_spec, resolve_answers_path  # noqa: E402
from engine.trace_logger import TraceLogger  # noqa: E402

DEFAULT_ANSWERS_ROOT = "data/answers"
DEFAULT_DATASET_ROOT = "data/model_baselines"
DEFAULT_CONFIG = "configs/baselines/track_baseline_models.json"
DEFAULT_EXPERIMENT_ID = "track_baseline_opencua_native_v1"
DEFAULT_MAX_STEPS = 64
DEFAULT_TIMEOUT_S = 5400
DEFAULT_API_TIMEOUT_S = 180
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_PROMPT_MODE = "opencua_qwen25_screenshot_v1"
DEFAULT_INTERACTION_PROTOCOL = "human_ui_v1"
DEFAULT_OBSERVATION_MODE = "vision_coords"
DEFAULT_SCORING_MODE = "soft_quality_v1"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_SERVED_MODEL_NAME = "opencua-32b"
DEFAULT_COORDINATE_TYPE = "qwen25"
DEFAULT_MIN_REQUEST_INTERVAL_S = 2.0
DEFAULT_HISTORY_IMAGES = 3
SCHEMA_VERSION = "baseline_eval.v4"
SUMMARY_SCHEMA_VERSION = "baseline_summary.v4"

PY_AUTO_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
CLICK_RE = re.compile(r"pyautogui\.(?P<kind>click|doubleClick)\s*\(\s*x\s*=\s*(?P<x>-?\d+(?:\.\d+)?)\s*,\s*y\s*=\s*(?P<y>-?\d+(?:\.\d+)?)", re.IGNORECASE)
WRITE_RE = re.compile(
    r"pyautogui\.(?:write|typewrite)\s*\(\s*(?:(?:message|text)\s*=\s*)?(?P<quote>['\"])(?P<text>.*?)(?P=quote)",
    re.IGNORECASE,
)
PRESS_RE = re.compile(r"pyautogui\.press\s*\(\s*(?P<quote>['\"])(?P<key>.*?)(?P=quote)", re.IGNORECASE)
HOTKEY_RE = re.compile(r"pyautogui\.hotkey\s*\((?P<args>.*?)\)", re.IGNORECASE)
SCROLL_RE = re.compile(r"pyautogui\.scroll\s*\(\s*(?P<delta>-?\d+)\s*\)", re.IGNORECASE)
WAIT_RE = re.compile(r"(?:time\.)?sleep\s*\(\s*(?P<secs>\d+(?:\.\d+)?)\s*\)", re.IGNORECASE)
SUBMIT_RE = re.compile(r"^(?:submit|finish_and_submit|click_submit)\s*$", re.IGNORECASE)
DONE_RE = re.compile(r"^(?:done|stop|terminate)\s*$", re.IGNORECASE)


def _load_run_answers(answers_path: Path, run_index: int) -> List[Dict[str, Any]]:
    for idx, run_spec in enumerate(iter_run_specs(answers_path), start=1):
        if idx == run_index:
            answers = run_spec.get("answers", [])
            if not isinstance(answers, list):
                raise ValueError(f"Run {run_index} answers must be a list")
            return answers
    raise IndexError(f"Run index out of range: {run_index} for {answers_path}")


def _http_post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url=url, data=body, method="POST")
    for key, value in headers.items():
        request.add_header(key, value)
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=max(1, int(timeout_s))) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        raise RuntimeError(f"opencua_http_error:{exc.code}:{raw}") from exc
    except Exception as exc:
        raise RuntimeError(f"opencua_request_failed:{exc}") from exc

    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"opencua_invalid_json:{exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("opencua_response_not_object")
    return parsed


def _smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280) -> Tuple[int, int]:
    if min(height, width) <= 0:
        raise ValueError("invalid_image_dimensions")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("invalid_aspect_ratio")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _qwen25_smart_resize_to_abs(model_x: float, model_y: float, original_width: int, original_height: int) -> Tuple[int, int, Dict[str, Any]]:
    resized_height, resized_width = _smart_resize(original_height, original_width, factor=28, min_pixels=3136, max_pixels=12845056)
    rel_x = float(model_x) / float(resized_width)
    rel_y = float(model_y) / float(resized_height)
    abs_x = int(max(0.0, min(float(original_width), rel_x * float(original_width))))
    abs_y = int(max(0.0, min(float(original_height), rel_y * float(original_height))))
    meta = {
        "coordinate_space": "qwen25_smart_resize_absolute",
        "resized_width": int(resized_width),
        "resized_height": int(resized_height),
        "model_x": float(model_x),
        "model_y": float(model_y),
        "original_width": int(original_width),
        "original_height": int(original_height),
        "abs_x": int(abs_x),
        "abs_y": int(abs_y),
    }
    return abs_x, abs_y, meta


def _to_norm(abs_x: int, abs_y: int, width: int, height: int) -> Dict[str, int]:
    width = max(1, int(width))
    height = max(1, int(height))
    x = max(0, min(int(width), int(abs_x)))
    y = max(0, min(int(height), int(abs_y)))
    return {
        "x": max(0, min(999, int(round((x / width) * 999)))),
        "y": max(0, min(999, int(round((y / height) * 999)))),
    }


def _extract_openai_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("openai_response_missing_choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError("openai_response_missing_message")
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "\n".join(parts).strip()
    raise RuntimeError("openai_response_missing_text")


def _image_part_for_path(path_value: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return None
    try:
        raw = path.read_bytes()
    except Exception:
        return None
    if not raw:
        return None
    encoded = base64.b64encode(raw).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def _clean_model_text(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if not text:
        return ""
    fenced = PY_AUTO_FENCE_RE.findall(text)
    if fenced:
        text = "\n".join(chunk.strip() for chunk in fenced if chunk.strip()).strip()
    text = text.replace("\r\n", "\n")
    return text.strip()


def _decode_string(value: str) -> str:
    try:
        return bytes(value, "utf-8").decode("unicode_escape")
    except Exception:
        return value


def _split_action_lines(raw_text: str) -> List[str]:
    cleaned = _clean_model_text(raw_text)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip() and not line.strip().startswith("#")]
    return lines


def _normalize_key_name(raw: str) -> str:
    value = str(raw or "").strip().lower()
    mapping = {
        "ctrl": "Control",
        "control": "Control",
        "shift": "Shift",
        "alt": "Alt",
        "enter": "Enter",
        "return": "Enter",
        "tab": "Tab",
        "space": "Space",
        "backspace": "Backspace",
        "delete": "Delete",
        "esc": "Escape",
        "escape": "Escape",
        "up": "ArrowUp",
        "down": "ArrowDown",
        "left": "ArrowLeft",
        "right": "ArrowRight",
        "pagedown": "PageDown",
        "pageup": "PageUp",
        "home": "Home",
        "end": "End",
    }
    if len(value) == 1 and value.isalpha():
        return value.upper()
    return mapping.get(value, raw)


def _parse_opencua_action(raw_text: str, viewport_width: int, viewport_height: int, coordinate_type: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    lines = _split_action_lines(raw_text)
    if not lines:
        raise ValueError("empty_model_output")

    debug: Dict[str, Any] = {"raw_lines": lines}
    joined = "\n".join(lines)

    if SUBMIT_RE.match(lines[0]):
        return {"action": "submit"}, debug
    if DONE_RE.match(lines[0]):
        return {"action": "done"}, debug

    if len(lines) >= 2:
        click_match = CLICK_RE.search(lines[0])
        write_match = WRITE_RE.search(lines[1])
        if click_match and write_match:
            model_x = float(click_match.group("x"))
            model_y = float(click_match.group("y"))
            abs_x, abs_y, coord_meta = _qwen25_smart_resize_to_abs(model_x, model_y, viewport_width, viewport_height)
            target = _to_norm(abs_x, abs_y, viewport_width, viewport_height)
            action = {
                "action": "type_text",
                "target": target,
                "value": _decode_string(write_match.group("text")),
                "clear_before_typing": False,
            }
            debug.update({"parser": "click_then_write", "coordinate_transform": coord_meta})
            return action, debug

    first = lines[0]
    click_match = CLICK_RE.search(first)
    if click_match:
        model_x = float(click_match.group("x"))
        model_y = float(click_match.group("y"))
        abs_x, abs_y, coord_meta = _qwen25_smart_resize_to_abs(model_x, model_y, viewport_width, viewport_height)
        target = _to_norm(abs_x, abs_y, viewport_width, viewport_height)
        action = {"action": "click_mouse", "target": target}
        if str(click_match.group("kind") or "").lower() == "doubleclick":
            action["click_count"] = 2
        debug.update({"parser": "click", "coordinate_transform": coord_meta})
        return action, debug

    write_match = WRITE_RE.search(first)
    if write_match:
        action = {"action": "type_text", "value": _decode_string(write_match.group("text")), "clear_before_typing": False}
        debug.update({"parser": "write_only"})
        return action, debug

    hotkey_match = HOTKEY_RE.search(first)
    if hotkey_match:
        args = [segment.strip().strip("'\"") for segment in hotkey_match.group("args").split(",") if segment.strip()]
        value = "+".join(_normalize_key_name(arg) for arg in args if arg)
        if not value:
            raise ValueError("invalid_hotkey_output")
        debug.update({"parser": "hotkey"})
        return {"action": "press_key", "value": value}, debug

    press_match = PRESS_RE.search(first)
    if press_match:
        debug.update({"parser": "press"})
        return {"action": "press_key", "value": _normalize_key_name(press_match.group("key"))}, debug

    scroll_match = SCROLL_RE.search(first)
    if scroll_match:
        debug.update({"parser": "scroll"})
        return {"action": "scroll", "delta": int(scroll_match.group("delta"))}, debug

    wait_match = WAIT_RE.search(first)
    if wait_match:
        debug.update({"parser": "wait"})
        return {"action": "wait", "delta": int(float(wait_match.group("secs")) * 1000)}, debug

    upper = first.strip().upper()
    if upper == "SUBMIT":
        return {"action": "submit"}, debug
    if upper == "DONE":
        return {"action": "done"}, debug

    raise ValueError(f"unrecognized_opencua_action:{joined[:400]}")


class OpenCUAAdapter:
    def __init__(self, model_cfg: Dict[str, Any], api_timeout_s: int, base_url: str, served_model_name: str, min_request_interval_s: float) -> None:
        self.model_cfg = dict(model_cfg)
        self.api_timeout_s = max(1, int(api_timeout_s))
        self.base_url = str(base_url or os.environ.get("OPENAI_BASE_URL") or model_cfg.get("openai_base_url") or DEFAULT_BASE_URL).rstrip("/")
        self.model = str(served_model_name or os.environ.get("OPENAI_MODEL") or model_cfg.get("openai_model") or model_cfg.get("served_model_name") or DEFAULT_SERVED_MODEL_NAME).strip()
        self.api_key = str(os.environ.get("OPENAI_API_KEY") or model_cfg.get("openai_api_key") or "EMPTY")
        self.min_request_interval_s = max(0.0, float(min_request_interval_s))
        self._last_request_started_at = 0.0
        if not self.base_url:
            raise RuntimeError("openai_base_url_missing")
        if not self.model:
            raise RuntimeError("served_model_name_missing")

    def _pace_requests(self) -> None:
        if self.min_request_interval_s <= 0:
            return
        now = time.perf_counter()
        if self._last_request_started_at > 0:
            wait_s = self.min_request_interval_s - (now - self._last_request_started_at)
            if wait_s > 0:
                time.sleep(wait_s)
        self._last_request_started_at = time.perf_counter()

    def infer(self, messages: List[Dict[str, Any]], max_new_tokens: int) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": int(max_new_tokens),
            "messages": messages,
        }
        self._pace_requests()
        response = _http_post_json(
            url=f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload=payload,
            timeout_s=self.api_timeout_s,
        )
        text = _extract_openai_text(response)
        meta = {
            "provider": "opencua_local",
            "provider_model": self.model,
            "duration_s": round(time.perf_counter() - started, 3),
            "transport": "openai_compat_vllm",
            "base_url": self.base_url,
        }
        return text, meta, response


def _build_goal_prompt(
    form_url: str,
    remaining_answers: List[Dict[str, Any]],
    last_result: Dict[str, Any],
    interaction_map: List[Dict[str, Any]],
    page_text: str,
    observation_mode: str,
    recent_history: List[Dict[str, Any]],
    include_symbolic_support: bool = False,
) -> str:
    validation_feedback = rbe._normalize_validation_feedback(last_result)
    page_text_block = ""
    if observation_mode != "vision_coords":
        page_text_block = f"Visible page text excerpt:\n{compact_page_text(page_text, max_chars=3000)}\n\n"
    interaction_map_block = ""
    if include_symbolic_support:
        interaction_map_block = f"Interaction map (symbolic support only):\n{json.dumps(interaction_map or [], indent=2, ensure_ascii=True)}\n\n"
    return (
        "You are a GUI agent. You are given a task and screenshots of the browser. "
        "You need to perform the next browser action to fill and submit the form.\n"
        "Return the next action only. Prefer a single pyautogui-style statement, or output SUBMIT or DONE. "
        "One or two pyautogui-style action lines are accepted when the model naturally emits a compound action.\n"
        "Allowed action styles:\n"
        "- pyautogui.click(x=..., y=...)\n"
        "- pyautogui.doubleClick(x=..., y=...)\n"
        "- pyautogui.write(message='text')\n"
        "- pyautogui.write('text')\n"
        "- pyautogui.press('enter')\n"
        "- pyautogui.hotkey('ctrl', 'a')\n"
        "- pyautogui.scroll(-600)\n"
        "- time.sleep(1)\n"
        "- SUBMIT\n"
        "- DONE\n"
        "Rules:\n"
        "- Use coordinates from the screenshot, not from the interaction map.\n"
        "- Do not output explanations outside the action text.\n"
        "- Before submitting, double-check the visible form state against the target answers as well as the current screenshot allows.\n"
        "- If the form appears correct and you intend to submit, click the visible Submit button or output SUBMIT.\n"
        "- SUBMIT means you intend to submit the current form state.\n"
        "- DONE means you intend to stop without further browser actions.\n\n"
        f"Current URL:\n{form_url}\n\n"
        f"Remaining answers:\n{json.dumps(remaining_answers, indent=2, ensure_ascii=True)}\n\n"
        f"{interaction_map_block}"
        f"Recent step history:\n{json.dumps(recent_history or [], indent=2, ensure_ascii=True)}\n\n"
        f"Validation feedback:\n{json.dumps(validation_feedback or {}, indent=2, ensure_ascii=True)}\n\n"
        f"Last action result:\n{json.dumps(last_result or {}, indent=2, ensure_ascii=True)}\n\n"
        f"{page_text_block}"
    )


def _build_messages(prompt: str, screenshot_paths: List[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for path in screenshot_paths:
        part = _image_part_for_path(path)
        if part is not None:
            content.append(part)
    content.append({"type": "text", "text": prompt})
    return [
        {"role": "system", "content": "You are a GUI agent operating a browser from screenshots."},
        {"role": "user", "content": content},
    ]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenCUA direct baseline evaluation for one model/form/run.")
    parser.add_argument("--model-id", default="computer_use_opencua_32b")
    parser.add_argument("--form-id", required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--answers-root", default=DEFAULT_ANSWERS_ROOT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--trial-id")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--api-timeout-s", type=int, default=DEFAULT_API_TIMEOUT_S)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--execution-backend", choices=["local", "mcp_server"], default="mcp_server")
    parser.add_argument("--browser-mcp-cmd")
    parser.add_argument("--browser-mcp-timeout-ms", type=int, default=120000)
    parser.add_argument("--browser-init-retries", type=int, default=2)
    parser.add_argument("--browser-init-retry-delay-s", type=float, default=1.5)
    parser.add_argument("--invalid-action-budget", type=int, default=0)
    parser.add_argument("--viewport-width", type=int, default=1440)
    parser.add_argument("--viewport-height", type=int, default=900)
    parser.add_argument("--interaction-protocol", choices=["legacy_semantic_v1", "human_ui_v1"], default=DEFAULT_INTERACTION_PROTOCOL)
    parser.add_argument("--observation-mode", choices=["vision_coords", "vision_coords_text"], default=DEFAULT_OBSERVATION_MODE)
    parser.add_argument("--scoring-mode", choices=["soft_quality_v1", "legacy_binary_v1"], default=DEFAULT_SCORING_MODE)
    parser.add_argument("--disable-action-coercion", action="store_true", default=False)
    parser.add_argument("--retention-window", type=int, default=rbe.DEFAULT_RETENTION_WINDOW)
    parser.add_argument("--run-label")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL)
    parser.add_argument("--served-model-name", default=os.environ.get("OPENAI_MODEL") or DEFAULT_SERVED_MODEL_NAME)
    parser.add_argument("--coordinate-type", default=DEFAULT_COORDINATE_TYPE)
    parser.add_argument("--min-request-interval-s", type=float, default=float(os.environ.get("OPEN_CUA_MIN_REQUEST_INTERVAL_S") or DEFAULT_MIN_REQUEST_INTERVAL_S))
    parser.add_argument("--history-images", type=int, default=DEFAULT_HISTORY_IMAGES)
    parser.add_argument(
        "--include-symbolic-support",
        action="store_true",
        default=False,
        help="Include the benchmark interaction map in the OpenCUA prompt. Default is screenshot-native only.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    args.retention_window = max(0, int(args.retention_window))
    args.history_images = max(1, int(args.history_images))
    run_label = rbe._make_run_label(args.run_label)
    run_started_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    config_path = (ROOT_DIR / args.config).resolve()
    model_cfg = get_model_by_id(config_path, args.model_id)
    if str(model_cfg.get("provider") or "") != "openai_compat":
        raise ValueError(f"run_opencua_direct_eval expects provider=openai_compat: {args.model_id}")

    adapter = OpenCUAAdapter(
        model_cfg=model_cfg,
        api_timeout_s=args.api_timeout_s,
        base_url=args.base_url,
        served_model_name=args.served_model_name,
        min_request_interval_s=args.min_request_interval_s,
    )

    form_spec = load_form_spec(args.form_id, ROOT_DIR / "src" / "forms")
    form_url = force_english_google_forms_url(str(form_spec.get("form_url") or form_spec.get("url") or ""))
    if not form_url:
        raise ValueError(f"Missing form_url in spec for {args.form_id}")

    answers_path = resolve_answers_path(argparse.Namespace(answers_root=args.answers_root, answers_file="runs.json"), args.form_id)
    answers = _load_run_answers(answers_path, args.run_index)
    answer_run_id = f"run_{args.run_index:04d}"
    trial_id = args.trial_id or rbe._make_trial_id()
    paths = rbe._build_trial_paths(args, args.model_id, args.form_id, answer_run_id, trial_id)

    question_states = rbe._build_question_states(answers)
    rbe._write_json(paths["answers_path"], question_states)
    rbe._touch(paths["model_io_path"])
    rbe._touch(paths["step_inputs_path"])
    start_time = time.perf_counter()

    trace = TraceLogger(
        path=paths["trace_path"],
        start_time=start_time,
        validate_mcp_actions=True,
        strict_mcp_validation=True,
        mcp_client=None,
    )

    annotations: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": args.experiment_id,
        "trial_id": trial_id,
        "run_label": run_label,
        "run_started_utc": run_started_utc,
        "model_id": args.model_id,
        "model_kind": "computer_use_agent",
        "track": "computer_use_native",
        "provider": model_cfg.get("provider"),
        "api_provider": "opencua_local",
        "form_id": args.form_id,
        "answer_run_id": answer_run_id,
        "form_url": form_url,
        "execution_backend": args.execution_backend,
        "prompt_mode": DEFAULT_PROMPT_MODE,
        "interaction_protocol": str(args.interaction_protocol),
        "observation_mode": str(args.observation_mode),
        "scoring_mode": str(args.scoring_mode),
        "success": False,
        "submit_success": False,
        "stop_reason": None,
        "failure_category": None,
        "failure_detail": None,
        "question_total": len(question_states),
        "question_correctness": 0,
        "attempted_count": 0,
        "attempted_correctness": 0,
        "verified_count": 0,
        "verified_correctness": 0,
        "action_count": 0,
        "invalid_actions": 0,
        "duration_s": None,
        "model": {"provider": model_cfg.get("provider"), "hf_repo": model_cfg.get("hf_repo")},
        "input_contract": {
            "provides_form_spec": False,
            "provides_dom_dump_upfront": False,
            "provides_answers": True,
            "provides_labels": True,
            "provides_widget_types": bool(args.include_symbolic_support),
            "provides_values": True,
            "provides_interaction_map": bool(args.include_symbolic_support),
        },
        "run_params": {
            "headless": bool(args.headless),
            "timeout_s": args.timeout_s,
            "max_steps": args.max_steps,
            "max_new_tokens": args.max_new_tokens,
            "invalid_action_budget": args.invalid_action_budget,
            "viewport": {"width": args.viewport_width, "height": args.viewport_height},
            "browser_mcp_cmd": args.browser_mcp_cmd,
            "browser_mcp_timeout_ms": args.browser_mcp_timeout_ms,
            "browser_init_retries": args.browser_init_retries,
            "browser_init_retry_delay_s": args.browser_init_retry_delay_s,
            "api_timeout_s": args.api_timeout_s,
            "disable_action_coercion": True,
            "retention_window": int(args.retention_window),
            "run_label": run_label,
            "server_backend": "vllm",
            "served_model_name": adapter.model,
            "coordinate_space": "qwen25_smart_resize_absolute",
            "coordinate_transform": args.coordinate_type,
            "base_url": adapter.base_url,
            "history_images": int(args.history_images),
            "include_symbolic_support": bool(args.include_symbolic_support),
        },
        "artifacts": rbe._artifact_payload(paths),
        "trace": {},
        "environment": {},
        "steps": [],
        "questions": question_states,
        "failure_events": [],
        "soft_violations": [],
        "submit_attempts": [],
    }

    rbe._append_jsonl(
        paths["model_io_path"],
        {
            "phase": "setup",
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "trial_id": trial_id,
            "run_label": run_label,
            "model_id": args.model_id,
            "form_id": args.form_id,
            "answer_run_id": answer_run_id,
            "status": "started",
            "api_provider": "opencua_local",
            "served_model_name": adapter.model,
            "base_url": adapter.base_url,
        },
    )

    execution_session = None
    last_result: Dict[str, Any] = {}
    terminal_screenshot_path: Optional[str] = None
    observation_cache: Dict[int, Dict[str, Any]] = {}
    screenshot_history: List[str] = []

    try:
        execution_session = rbe._make_execution_session(args, paths, trace)
        annotations["environment"] = execution_session.start(form_url) or {}
        observation_cache[0] = execution_session.observe(0)

        initial_observation = observation_cache[0] if isinstance(observation_cache[0], dict) else {}
        initial_screenshot = str(initial_observation.get("screenshot_path") or "").strip()
        initial_interaction_map = initial_observation.get("interaction_map") if isinstance(initial_observation.get("interaction_map"), list) else []
        if (
            args.include_symbolic_support
            and args.interaction_protocol == "human_ui_v1"
            and args.execution_backend == "mcp_server"
            and initial_screenshot
            and len(initial_interaction_map) == 0
        ):
            raise RuntimeError("browser_mcp_preflight_failed: interaction_map_empty_under_human_ui_v1")

        for step_idx in range(args.max_steps):
            elapsed = time.perf_counter() - start_time
            if elapsed >= args.timeout_s:
                annotations["stop_reason"] = "timeout"
                rbe._set_failure(annotations, "timeout", f"timeout after {args.timeout_s}s", step_idx)
                break

            remaining_answers = rbe._serialize_remaining_answers(question_states)
            recent_history = annotations.get("steps", [])[-4:]
            observation = observation_cache.pop(step_idx, None)
            if observation is None:
                observation = execution_session.observe(step_idx)
            if not isinstance(observation, dict):
                observation = {}
            page_text = str(observation.get("page_text") or "")
            screenshot_path = str(observation.get("screenshot_path") or "") or None
            if screenshot_path:
                screenshot_history.append(screenshot_path)
                screenshot_history = screenshot_history[-args.history_images :]
            raw_interaction_map = observation.get("interaction_map") if isinstance(observation.get("interaction_map"), list) else []
            interaction_map = rbe._enrich_interaction_map(raw_interaction_map, remaining_answers)

            prompt = _build_goal_prompt(
                form_url=form_url,
                remaining_answers=remaining_answers,
                last_result=last_result,
                interaction_map=interaction_map,
                page_text=page_text,
                observation_mode=args.observation_mode,
                recent_history=recent_history,
                include_symbolic_support=bool(args.include_symbolic_support),
            )

            step_input_record = {
                "phase": "step_input",
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "step_index": step_idx,
                "form_url": form_url,
                "remaining_answers": remaining_answers,
                "last_result": dict(last_result or {}),
                "page_text_excerpt": compact_page_text(page_text, max_chars=3000),
                "screenshot_path": screenshot_path,
                "screenshot_history": list(screenshot_history),
                "interaction_map": interaction_map if args.include_symbolic_support else [],
                "interaction_map_available_count": len(interaction_map),
                "interaction_map_count": len(interaction_map),
                "interaction_map_prompt_included": bool(args.include_symbolic_support),
                "behavior_nudge": None,
                "prompt_hash": rbe._prompt_hash(prompt),
            }
            rbe._append_jsonl(paths["step_inputs_path"], step_input_record)

            messages = _build_messages(prompt, screenshot_history)
            infer_started = time.perf_counter()
            try:
                raw_output, infer_meta, api_payload = adapter.infer(messages, max_new_tokens=args.max_new_tokens)
                infer_meta["duration_s"] = round(time.perf_counter() - infer_started, 3)
            except Exception as exc:
                infer_error = f"model_inference_failed: {exc}"
                step_record = {
                    "step_index": step_idx,
                    "elapsed_s": round(time.perf_counter() - start_time, 3),
                    "prompt_mode": DEFAULT_PROMPT_MODE,
                    "remaining_answers_before": len(remaining_answers),
                    "page_text_excerpt": page_text[:2000],
                    "screenshot_path": screenshot_path,
                    "raw_model_output": None,
                    "action": None,
                    "warnings": [],
                    "status": "failed",
                    "error": infer_error,
                    "matched_question_id": None,
                    "target_match": None,
                    "execution": None,
                    "verification": None,
                    "progress_made": False,
                "interaction_map_count": len(interaction_map),
                "interaction_map_prompt_included": bool(args.include_symbolic_support),
                "model_inference": {"attempts": [{"attempt": 1, "error": str(exc)}]},
            }
                annotations["steps"].append(step_record)
                rbe._set_failure(annotations, "model_inference_failed", str(exc), step_idx)
                annotations["stop_reason"] = "model_inference_failed"
                break

            step_record: Dict[str, Any] = {
                "step_index": step_idx,
                "elapsed_s": round(time.perf_counter() - start_time, 3),
                "prompt_mode": DEFAULT_PROMPT_MODE,
                "remaining_answers_before": len(remaining_answers),
                "page_text_excerpt": page_text[:2000],
                "screenshot_path": screenshot_path,
                "raw_model_output": raw_output,
                "action": None,
                "warnings": [],
                "status": None,
                "error": None,
                "matched_question_id": None,
                "target_match": None,
                "execution": None,
                "verification": None,
                "progress_made": False,
                "interaction_map_count": len(interaction_map),
                "interaction_map_prompt_included": bool(args.include_symbolic_support),
                "model_inference": {"attempts": [{"attempt": 1, **infer_meta}]},
            }
            io_record: Dict[str, Any] = {
                "phase": "step",
                "step_index": step_idx,
                "elapsed_s": round(time.perf_counter() - start_time, 3),
                "prompt_mode": DEFAULT_PROMPT_MODE,
                "prompt": prompt,
                "remaining_answers": remaining_answers,
                "screenshot_path": screenshot_path,
                "screenshot_history": list(screenshot_history),
                "interaction_map": interaction_map if args.include_symbolic_support else [],
                "interaction_map_available_count": len(interaction_map),
                "interaction_map_prompt_included": bool(args.include_symbolic_support),
                "raw_model_output": raw_output,
                "api_response_excerpt": json.dumps(api_payload, ensure_ascii=True)[:4000],
                "parsed_action": None,
                "warnings": [],
                "error": None,
                "matched_question_id": None,
                "target_match": None,
                "execution": None,
                "verification": None,
                "model_inference": step_record["model_inference"],
            }

            try:
                action_candidate, parser_debug = _parse_opencua_action(raw_output, args.viewport_width, args.viewport_height, args.coordinate_type)
                action, warnings = validate_low_level_action(action_candidate)
                step_record["action"] = action
                step_record["warnings"] = warnings
                step_record["parser_debug"] = parser_debug
                io_record["parsed_action"] = action
                io_record["warnings"] = warnings
                io_record["parser_debug"] = parser_debug
            except Exception as exc:
                annotations["invalid_actions"] += 1
                message = f"model_output_invalid: {exc}"
                step_record["status"] = "failed"
                step_record["error"] = message
                io_record["error"] = message
                io_record["parse_error"] = str(exc)
                rbe._record_soft_violation(annotations, "model_output_invalid", str(exc), step_idx)
                annotations["steps"].append(step_record)
                rbe._append_jsonl(paths["model_io_path"], io_record)
                last_result = {"status": "failed", "error": message, "remaining_answers": len(remaining_answers)}
                if rbe._invalid_action_budget_exhausted(annotations["invalid_actions"], args.invalid_action_budget):
                    annotations["stop_reason"] = "model_output_invalid"
                    break
                continue

            action_name = str(action.get("action") or "")
            annotations["action_count"] += 1

            if action_name == "submit" and len(remaining_answers) > 0:
                detail = json.dumps(
                    {
                        "remaining_question_ids": [str(item.get("question_id") or "") for item in remaining_answers],
                        "remaining_count": len(remaining_answers),
                    },
                    ensure_ascii=True,
                )
                step_record.setdefault("warnings", []).append("premature_submit_with_remaining_answers")
                io_record.setdefault("warnings", []).append("premature_submit_with_remaining_answers")
                step_record["premature_submit_detail"] = json.loads(detail)
                io_record["premature_submit_detail"] = json.loads(detail)
                rbe._record_soft_violation(annotations, "premature_submit", detail, step_idx)

            target = action.get("target") if isinstance(action.get("target"), dict) else {}
            matched_idx, question_state, match_debug = rbe._match_question_state(question_states, target)
            step_record["target_match"] = match_debug
            io_record["target_match"] = match_debug
            if question_state is not None and matched_idx is not None:
                step_record["matched_question_id"] = question_state.get("question_id")
                io_record["matched_question_id"] = question_state.get("question_id")

            execution_payload: Dict[str, Any] = {}
            exec_err: Optional[str] = None
            try:
                if action_name == "move_mouse":
                    execution_payload = execution_session.execute_move_mouse(int(target.get("x")), int(target.get("y")), step_idx)
                    step_record["status"] = "moved"
                    step_record["progress_made"] = True
                elif action_name == "click_mouse":
                    click_count = max(1, int(action_candidate.get("click_count") or 1))
                    for _ in range(click_count):
                        execution_payload = execution_session.execute_click_mouse(int(target.get("x")), int(target.get("y")), step_idx)
                        execution_session.execute_wait(0.15, step_idx)
                    step_record["status"] = "clicked"
                    step_record["progress_made"] = True
                    if question_state is not None:
                        question_state["attempted"] = True
                elif action_name == "type_text":
                    if isinstance(target, dict) and "x" in target and "y" in target:
                        execution_session.execute_click_mouse(int(target.get("x")), int(target.get("y")), step_idx)
                        execution_session.execute_wait(0.2, step_idx)
                    if bool(action_candidate.get("clear_before_typing", False)):
                        execution_session.execute_press_key("Control+A", step_idx)
                        execution_session.execute_press_key("Backspace", step_idx)
                    execution_payload = execution_session.execute_type_text(str(action.get("value") or ""), step_idx)
                    step_record["status"] = "typed"
                    step_record["progress_made"] = True
                    if question_state is not None:
                        question_state["attempted"] = True
                        question_state["attempted_correct"] = rbe._value_matches(question_state.get("value"), action.get("value"))
                elif action_name == "wait":
                    wait_seconds = max(0.25, float(action.get("delta") or 1000) / 1000.0)
                    execution_session.execute_wait(wait_seconds, step_idx)
                    execution_payload = {"status": "waited", "seconds": wait_seconds}
                    step_record["status"] = "waited"
                elif action_name == "scroll":
                    delta = int(action.get("delta") or 600)
                    execution_session.execute_scroll(delta, step_idx)
                    execution_payload = {"status": "scrolled", "delta": delta}
                    step_record["status"] = "scrolled"
                    step_record["progress_made"] = True
                elif action_name == "press_key":
                    key = str(action.get("value") or "Tab")
                    execution_session.execute_press_key(key, step_idx)
                    execution_payload = {"status": "pressed_key", "key": key}
                    step_record["status"] = "pressed_key"
                    step_record["progress_made"] = True
                elif action_name == "submit":
                    submit_attempt = {
                        "step_index": step_idx,
                        "remaining_answer_count_before": len(remaining_answers),
                        "submitted_while_incomplete": len(remaining_answers) > 0,
                    }
                    submit_info, submit_err = execution_session.submit()
                    execution_payload = submit_info
                    submit_attempt["success"] = bool(submit_info.get("success")) if isinstance(submit_info, dict) else False
                    submit_attempt["error"] = submit_err
                    annotations.setdefault("submit_attempts", []).append(submit_attempt)
                    if submit_err:
                        step_record["status"] = "failed"
                        step_record["error"] = f"submission_failed: {submit_err}"
                        rbe._record_soft_violation(annotations, "submission_failed", submit_err, step_idx)
                    elif submit_info.get("success"):
                        step_record["status"] = "submitted"
                        step_record["progress_made"] = True
                        annotations["success"] = True
                        annotations["submit_success"] = True
                        annotations["stop_reason"] = "submitted"
                    else:
                        step_record["status"] = "failed"
                        step_record["error"] = "submission_failed: not confirmed"
                        rbe._record_soft_violation(annotations, "submission_failed", json.dumps(submit_info, ensure_ascii=True), step_idx)
                elif action_name == "done":
                    execution_payload = {"status": "done"}
                    step_record["status"] = "done"
                    annotations["stop_reason"] = "done"
            except Exception as exc:
                exec_err = str(exc)

            step_record["execution"] = execution_payload
            io_record["execution"] = execution_payload
            if exec_err:
                step_record["status"] = "failed"
                step_record["error"] = f"widget_interaction_failed: {exec_err}"
                rbe._record_soft_violation(annotations, "widget_interaction_failed", exec_err, step_idx)

            if step_record.get("status") not in {"submitted", "done", "failed"}:
                verification_rows: List[Dict[str, Any]] = []
                for state in question_states:
                    result = execution_session.verify_entry(state, step_idx)
                    state["last_verification"] = result
                    state["actual_value"] = result.get("actual_value")
                    state["verified"] = bool(result.get("verified"))
                    state["verified_correct"] = bool(result.get("verified")) and rbe._value_matches(state.get("value"), result.get("actual_value"))
                    if state["verified"]:
                        state["attempted"] = True
                    if state["verified_correct"]:
                        state["attempted_correct"] = True
                        state["final_status"] = "correct_verified"
                    elif state.get("attempted_correct"):
                        state["final_status"] = "correct_attempted_only"
                    elif state.get("attempted"):
                        state["final_status"] = "failed"
                    verification_rows.append(
                        {
                            "question_id": state.get("question_id"),
                            "verified": state.get("verified"),
                            "verified_correct": state.get("verified_correct"),
                            "detail": result.get("detail"),
                        }
                    )
                io_record["verification"] = verification_rows

                if question_state is not None and action_name in {"type_text", "click_mouse"}:
                    step_verification = question_state.get("last_verification") or {}
                    step_record["verification"] = step_verification
                    step_record["expected_label"] = question_state.get("label")
                    step_record["expected_value"] = question_state.get("value")
                    step_record["executed_value"] = action.get("value")
                    if question_state.get("verified_correct"):
                        step_record["status"] = "filled"
                        step_record["progress_made"] = True
                    elif question_state.get("verified"):
                        step_record["status"] = "filled_unverified"
                        step_record["error"] = "verification_failed"
                        detail = json.dumps({"expected": question_state.get("value"), "actual": step_verification.get("actual_value")}, ensure_ascii=True)
                        rbe._record_soft_violation(annotations, "verification_failed", detail, step_idx)
                    elif action_name == "type_text" and step_record.get("status") == "typed":
                        step_record["status"] = "filled_unverified"
                        step_record["error"] = f"verification_failed: {step_verification.get('detail')}"
                        rbe._record_soft_violation(annotations, "verification_failed", str(step_verification.get("detail")), step_idx)

            annotations["steps"].append(step_record)
            rbe._append_jsonl(paths["model_io_path"], io_record)
            last_result = {"status": step_record["status"], "error": step_record["error"], "remaining_answers": len(rbe._serialize_remaining_answers(question_states))}

            if annotations["stop_reason"] in {"submitted", "submission_failed", "model_output_invalid", "model_inference_failed", "done"}:
                break

        if annotations["stop_reason"] is None:
            annotations["stop_reason"] = "max_steps_exceeded"
            rbe._set_failure(annotations, "max_steps_exceeded", f"max_steps={args.max_steps}")
    except Exception as exc:
        annotations["stop_reason"] = "environment_error"
        category = rbe._classify_environment_error(str(exc))
        rbe._set_failure(annotations, category, str(exc))
        annotations["run_error"] = str(exc)
        if execution_session is not None:
            try:
                terminal_screenshot_path = execution_session.capture_terminal_screenshot("error.png")
            except Exception:
                terminal_screenshot_path = None
    finally:
        if execution_session is not None and annotations.get("success"):
            try:
                terminal_screenshot_path = execution_session.capture_terminal_screenshot("final.png")
            except Exception:
                terminal_screenshot_path = terminal_screenshot_path
        if execution_session is not None:
            execution_session.close()
        trace_summary = trace.summary()
        trace.close()

    video_path = rbe._finalize_trial_video(paths["artifact_dir"], paths["video_path"])
    if video_path is not None and video_path.exists():
        annotations["artifacts"]["video_path"] = str(video_path)
    if terminal_screenshot_path:
        if annotations.get("success"):
            annotations["artifacts"]["final_screenshot_path"] = terminal_screenshot_path
        else:
            annotations["artifacts"]["error_screenshot_path"] = terminal_screenshot_path

    metrics = rbe._calculate_metrics(question_states)
    annotations.update(metrics)
    if str(args.scoring_mode or "") == "soft_quality_v1":
        soft_metrics = rbe._calculate_soft_quality_metrics(annotations.get("steps", []), annotations, bool(annotations.get("submit_success")))
        annotations.update(soft_metrics)
    annotations["duration_s"] = round(time.perf_counter() - start_time, 3)
    annotations.update(
        rbe._resolve_reference_efficiency(
            form_id=args.form_id,
            answer_run_id=answer_run_id,
            model_duration_s=annotations["duration_s"],
            model_trace_path=paths["trace_path"],
            model_action_count=annotations.get("action_count"),
            prefer_model_action_count=True,
        )
    )
    annotations["trace"] = trace_summary
    run_completed_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    annotations["run_completed_utc"] = run_completed_utc

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment_id": args.experiment_id,
        "trial_id": trial_id,
        "run_label": run_label,
        "run_started_utc": run_started_utc,
        "run_completed_utc": run_completed_utc,
        "model_id": args.model_id,
        "model_kind": "computer_use_agent",
        "track": annotations.get("track"),
        "provider": annotations.get("provider"),
        "api_provider": annotations.get("api_provider"),
        "form_id": args.form_id,
        "answer_run_id": answer_run_id,
        "execution_backend": args.execution_backend,
        "prompt_mode": DEFAULT_PROMPT_MODE,
        "interaction_protocol": str(args.interaction_protocol),
        "observation_mode": str(args.observation_mode),
        "scoring_mode": str(args.scoring_mode),
        "success": bool(annotations["success"]),
        "submit_success": bool(annotations["submit_success"]),
        "stop_reason": annotations["stop_reason"],
        "failure_category": annotations["failure_category"],
        "failure_detail": annotations["failure_detail"],
        "question_total": annotations["question_total"],
        "question_correctness": annotations["question_correctness"],
        "attempted_count": annotations["attempted_count"],
        "attempted_correctness": annotations["attempted_correctness"],
        "verified_count": annotations["verified_count"],
        "verified_correctness": annotations["verified_correctness"],
        "action_count": annotations["action_count"],
        "trace_action_count": annotations.get("trace_action_count"),
        "trace_action_count_source": annotations.get("trace_action_count_source"),
        "invalid_actions": annotations["invalid_actions"],
        "submit_attempt_count": len(annotations.get("submit_attempts") or []),
        "successful_submit_attempt_count": sum(1 for item in (annotations.get("submit_attempts") or []) if item.get("success")),
        "failed_submit_attempt_count": sum(1 for item in (annotations.get("submit_attempts") or []) if not item.get("success")),
        "submitted_while_incomplete_count": sum(1 for item in (annotations.get("submit_attempts") or []) if item.get("submitted_while_incomplete")),
        "duration_s": annotations["duration_s"],
        "reference_available": annotations.get("reference_available"),
        "reference_run_path": annotations.get("reference_run_path"),
        "reference_trace_path": annotations.get("reference_trace_path"),
        "reference_video_path": annotations.get("reference_video_path"),
        "reference_action_count": annotations.get("reference_action_count"),
        "reference_duration_s": annotations.get("reference_duration_s"),
        "action_overhead_ratio": annotations.get("action_overhead_ratio"),
        "time_overhead_ratio": annotations.get("time_overhead_ratio"),
        "action_count_delta": annotations.get("action_count_delta"),
        "duration_delta_s": annotations.get("duration_delta_s"),
        "model_driven_execution": annotations.get("model_driven_execution"),
        "autonomy_step_rate": annotations.get("autonomy_step_rate"),
        "action_diversity": annotations.get("action_diversity"),
        "loop_ratio": annotations.get("loop_ratio"),
        "correction_count": annotations.get("correction_count"),
        "composite_score": annotations.get("composite_score"),
        "server_backend": "vllm",
        "served_model_name": adapter.model,
        "coordinate_space": "qwen25_smart_resize_absolute",
        "coordinate_transform": args.coordinate_type,
        "artifacts": annotations["artifacts"],
    }

    rbe._append_jsonl(
        paths["model_io_path"],
        {
            "phase": "terminal",
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "stop_reason": summary["stop_reason"],
            "failure_category": summary["failure_category"],
            "failure_detail": summary["failure_detail"],
            "success": summary["success"],
            "submit_success": summary["submit_success"],
            "attempted_correctness": summary["attempted_correctness"],
            "verified_correctness": summary["verified_correctness"],
            "trace_action_count": summary.get("trace_action_count"),
            "reference_available": summary.get("reference_available"),
            "reference_action_count": summary.get("reference_action_count"),
            "reference_duration_s": summary.get("reference_duration_s"),
            "action_overhead_ratio": summary.get("action_overhead_ratio"),
            "time_overhead_ratio": summary.get("time_overhead_ratio"),
            "server_backend": summary["server_backend"],
            "served_model_name": summary["served_model_name"],
            "coordinate_space": summary["coordinate_space"],
            "coordinate_transform": summary["coordinate_transform"],
        },
    )

    rbe._write_json(paths["annotations_path"], annotations)
    rbe._write_json(paths["summary_path"], summary)

    manifest_entry = {
        "experiment_id": args.experiment_id,
        "trial_id": trial_id,
        "run_label": run_label,
        "run_started_utc": run_started_utc,
        "run_completed_utc": run_completed_utc,
        "model_id": args.model_id,
        "model_kind": "computer_use_agent",
        "track": annotations.get("track"),
        "provider": annotations.get("provider"),
        "api_provider": annotations.get("api_provider"),
        "form_id": args.form_id,
        "answer_run_id": answer_run_id,
        "success": summary["success"],
        "submit_success": summary["submit_success"],
        "stop_reason": summary["stop_reason"],
        "failure_category": summary["failure_category"],
        "failure_detail": summary["failure_detail"],
        "trace_action_count": summary.get("trace_action_count"),
        "trace_action_count_source": summary.get("trace_action_count_source"),
        "reference_available": summary.get("reference_available"),
        "reference_action_count": summary.get("reference_action_count"),
        "reference_duration_s": summary.get("reference_duration_s"),
        "action_overhead_ratio": summary.get("action_overhead_ratio"),
        "time_overhead_ratio": summary.get("time_overhead_ratio"),
        "server_backend": summary["server_backend"],
        "served_model_name": summary["served_model_name"],
        "coordinate_space": summary["coordinate_space"],
        "coordinate_transform": summary["coordinate_transform"],
        "summary_path": str(paths["summary_path"]),
        "annotations_path": str(paths["annotations_path"]),
        "trace_path": str(paths["trace_path"]),
        "model_io_path": str(paths["model_io_path"]),
        "step_inputs_path": str(paths["step_inputs_path"]),
        "video_path": annotations["artifacts"]["video_path"],
        "artifact_dir": str(paths["artifact_dir"]),
    }
    rbe._append_jsonl(paths["manifest_path"], manifest_entry)
    rbe._update_experiment_indexes(
        experiment_root=paths["experiment_root"],
        manifest_entry=manifest_entry,
        run_label=run_label,
        retention_window=args.retention_window,
    )

    print(f"[INFO] wrote OpenCUA baseline summary: {paths['summary_path']}")
    print(f"[INFO] wrote OpenCUA baseline annotations: {paths['annotations_path']}")
    print(f"[INFO] wrote OpenCUA baseline manifest: {paths['manifest_path']}")
    print(f"[INFO] provider: opencua_local")
    print(f"[INFO] served_model_name: {adapter.model}")
    print(f"[INFO] stop_reason: {summary['stop_reason']}")
    print(f"[INFO] success: {summary['success']}")
    print(f"[INFO] submit_success: {summary['submit_success']}")
    print(f"[INFO] attempted_correctness: {summary['attempted_correctness']}/{summary['question_total']}")
    print(f"[INFO] verified_correctness: {summary['verified_correctness']}/{summary['question_total']}")
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
