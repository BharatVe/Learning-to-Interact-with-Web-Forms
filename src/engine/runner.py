import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from playwright.sync_api import sync_playwright

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from engine import form_filler  # noqa: E402

DEFAULT_DATASET_ROOT = "data/forms"
DEFAULT_TYPE_DELAY = 120
DEFAULT_ACTION_DELAY = 200
DEFAULT_SLOW_MO = 200
DEFAULT_TIMEOUT_MS = 15000

RUN_DIR_PATTERN = re.compile(r"run_(\d{4})$")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate form datasets via Playwright")
    parser.add_argument("--form-id", required=True)
    parser.add_argument("--form-url")
    parser.add_argument("--answers", dest="answers_source")
    parser.add_argument("--answers-source", dest="answers_source")
    parser.add_argument("--answers-json", dest="answers_source")
    parser.add_argument("--video-dir")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--specs-root")
    parser.add_argument("--num-runs", type=int)
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--slow-mo", type=int, default=DEFAULT_SLOW_MO)
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    parser.add_argument("--type-delay", type=int, default=DEFAULT_TYPE_DELAY)
    parser.add_argument("--action-delay", type=int, default=DEFAULT_ACTION_DELAY)
    args = parser.parse_args(argv)
    if not args.answers_source:
        parser.error("--answers is required")
    if args.num_runs is not None and args.num_runs < 1:
        parser.error("--num-runs must be >= 1")
    return args


def load_form_spec(form_id: str, specs_root: Path) -> Dict[str, Any]:
    spec_path = specs_root / form_id / "spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")
    data = json.loads(spec_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Spec file must contain an object: {spec_path}")
    return data


def normalize_run_spec(run_obj: Any, index_label: str) -> Dict[str, Any]:
    if isinstance(run_obj, list):
        answers = run_obj
        metadata = {}
    elif isinstance(run_obj, dict):
        if "answers" in run_obj:
            answers = run_obj.get("answers")
            metadata = {k: v for k, v in run_obj.items() if k != "answers"}
        else:
            raise ValueError(f"Run {index_label} missing answers list")
    else:
        raise ValueError(f"Run {index_label} must be a list or object")
    if not isinstance(answers, list):
        raise ValueError(f"Run {index_label} answers must be a list")
    return {"answers": answers, "metadata": metadata}


def iter_json_runs(path: Path) -> Iterator[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        yield normalize_run_spec(data, "0")
        return
    if not isinstance(data, dict):
        raise ValueError("answers file must be a list or an object containing runs")
    runs = data.get("runs")
    if runs is None:
        runs = data.get("multi_runs")
    if runs is None and "answers" in data:
        yield normalize_run_spec(data, "0")
        return
    if not isinstance(runs, list):
        raise ValueError("runs must be a list in answers file")
    for idx, run in enumerate(runs):
        yield normalize_run_spec(run, str(idx))


def iter_jsonl_runs(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            run_obj = json.loads(stripped)
            yield normalize_run_spec(run_obj, f"line {line_number}")


def iter_run_specs(path: Path) -> Iterator[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return iter_jsonl_runs(path)
    return iter_json_runs(path)


def existing_run_indices(runs_dir: Path) -> List[int]:
    if not runs_dir.exists():
        return []
    indices: List[int] = []
    for entry in runs_dir.iterdir():
        if not entry.is_dir():
            continue
        match = RUN_DIR_PATTERN.match(entry.name)
        if match:
            indices.append(int(match.group(1)))
    return indices


def next_available_index(existing: set, start_index: int) -> int:
    index = max(1, start_index)
    while index in existing:
        index += 1
    return index


def ensure_run_dir(
    runs_dir: Path, run_index: int, skip_existing: bool, existing_indices: set
) -> Tuple[Path, str, int]:
    run_name = f"run_{run_index:04d}"
    run_dir = runs_dir / run_name
    if run_dir.exists():
        existing_indices.add(run_index)
        if skip_existing:
            return ensure_run_dir(runs_dir, run_index + 1, skip_existing, existing_indices)
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    existing_indices.add(run_index)
    return run_dir, run_name, run_index


def finalize_video(run_dir: Path, form_id: str, run_name: str, page_video) -> Optional[Path]:
    raw_video_path = None
    if page_video is not None:
        try:
            raw_video_path = Path(page_video.path())
        except Exception:
            raw_video_path = None
    if raw_video_path is None:
        try:
            candidate_videos = sorted(
                run_dir.glob("*.webm"), key=lambda p: p.stat().st_mtime, reverse=True
            )
            raw_video_path = candidate_videos[0] if candidate_videos else None
        except Exception:
            raw_video_path = None
    if raw_video_path and raw_video_path.exists():
        final_video_path = run_dir / f"{form_id}_{run_name}.webm"
        if raw_video_path != final_video_path:
            if final_video_path.exists():
                final_video_path.unlink()
            try:
                raw_video_path.rename(final_video_path)
                return final_video_path
            except Exception:
                return raw_video_path
        return raw_video_path
    return None


def run_single(
    browser,
    form_id: str,
    form_url: str,
    answers: List[Dict[str, Any]],
    run_dir: Path,
    run_name: str,
    run_label: str,
    args: argparse.Namespace,
    run_metadata: Dict[str, Any],
) -> bool:
    start_time = time.perf_counter()
    actions: List[Dict[str, Any]] = []
    submit_info: Dict[str, Any] = {"success": False, "t_start_s": None, "t_end_s": None}
    submitted = False
    page_video = None
    context = None
    page = None
    captured_exception: Optional[Exception] = None

    answers_path = run_dir / "answers_instance.json"
    answers_path.write_text(json.dumps(answers, indent=2))

    try:
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            record_video_dir=str(run_dir),
        )
        context.add_init_script(form_filler.CURSOR_OVERLAY_SCRIPT)
        page = context.new_page()
        page_video = page.video
        page.set_default_timeout(DEFAULT_TIMEOUT_MS)
        page.goto(form_url, wait_until="domcontentloaded")
        page.wait_for_load_state("networkidle")
        page.mouse.move(10, 10)
        actions = form_filler.fill_form(page, answers, start_time, args.pause_seconds)
        submit_info = form_filler.submit_form(page, start_time)
        submitted = bool(submit_info.get("success"))
        if args.pause_seconds > 0:
            page.wait_for_timeout(int(args.pause_seconds * 1000))
    except Exception as exc:
        captured_exception = exc
        if submit_info.get("t_start_s") is None:
            now = time.perf_counter() - start_time
            submit_info["t_start_s"] = now
            submit_info["t_end_s"] = now
            submit_info["error"] = str(exc)
    finally:
        if page:
            try:
                page.close()
            except Exception:
                pass
        if context:
            try:
                context.close()
            except Exception:
                pass

    video_path = finalize_video(run_dir, form_id, run_name, page_video)

    failure_reason = None
    if not submitted:
        failure_reason = submit_info.get("error")
        if not failure_reason:
            if submit_info.get("submit_clicked") is False:
                failure_reason = "submit_not_clicked"
            else:
                failure_reason = "confirmation_not_detected"

    annotations = {
        "form_id": form_id,
        "run_name": run_name,
        "run_label": run_label,
        "form_url": form_url,
        "video_path": str(video_path) if video_path else None,
        "run_params": {
            "slow_mo": args.slow_mo,
            "type_delay_ms": form_filler.TYPE_DELAY_MS,
            "action_delay_ms": form_filler.ACTION_DELAY_MS,
            "pause_seconds": args.pause_seconds,
        },
        "submitted": submitted,
        "failure_reason": failure_reason,
        "submit": submit_info,
        "actions": actions,
    }
    if run_metadata:
        annotations["run_metadata"] = run_metadata
    annotations_path = run_dir / "annotations.json"
    annotations_path.write_text(json.dumps(annotations, indent=2))

    if captured_exception:
        raise captured_exception
    return submitted


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    specs_root = Path(args.specs_root) if args.specs_root else (SRC_DIR / "forms")
    form_spec = load_form_spec(args.form_id, specs_root)
    form_url = args.form_url or form_spec.get("form_url")
    if not form_url:
        raise ValueError("Form URL not provided and not found in spec")

    answers_path = Path(args.answers_source).resolve()
    run_specs_iter = iter_run_specs(answers_path)

    dataset_root = Path(args.dataset_root).resolve()
    if args.video_dir and args.dataset_root == DEFAULT_DATASET_ROOT:
        dataset_root = Path(args.video_dir).resolve()
    runs_dir = dataset_root / args.form_id / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    existing_indices = set(existing_run_indices(runs_dir))
    start_index = args.start_index or 1
    if start_index < 1:
        raise ValueError("--start-index must be >= 1")
    skip_existing = args.skip_existing or args.resume
    if args.resume and args.start_index is None:
        start_index = next_available_index(existing_indices, 1)

    form_filler.set_type_delay(args.type_delay)
    form_filler.set_action_delay(args.action_delay)

    generated = 0
    current_index = start_index
    unconfirmed_runs: List[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False, slow_mo=args.slow_mo)
        try:
            for spec in run_specs_iter:
                if args.num_runs is not None and generated >= args.num_runs:
                    break
                if skip_existing:
                    current_index = next_available_index(existing_indices, current_index)
                run_dir, run_name, run_index = ensure_run_dir(
                    runs_dir, current_index, skip_existing, existing_indices
                )
                run_label = f"{args.form_id}_{run_name}"
                answers = spec.get("answers", [])
                if not isinstance(answers, list):
                    raise ValueError("Run answers must be a list")
                run_metadata = spec.get("metadata", {})
                submitted = run_single(
                    browser=browser,
                    form_id=args.form_id,
                    form_url=form_url,
                    answers=answers,
                    run_dir=run_dir,
                    run_name=run_name,
                    run_label=run_label,
                    args=args,
                    run_metadata=run_metadata,
                )
                if not submitted:
                    unconfirmed_runs.append(run_name)
                generated += 1
                current_index = run_index + 1
        finally:
            browser.close()

    if unconfirmed_runs:
        joined = ", ".join(unconfirmed_runs)
        print(f"[WARN] Submission not confirmed for runs: {joined}", file=sys.stderr)


if __name__ == "__main__":
    main()
