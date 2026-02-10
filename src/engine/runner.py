import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from engine.form_engine import FormEngine  # noqa: E402
from engine.trace_logger import TraceLogger  # noqa: E402

DEFAULT_DATASET_ROOT = "data/forms"
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
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--specs-root")
    parser.add_argument("--num-runs", type=int)
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-existing-video", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--slow-mo", type=int, default=DEFAULT_SLOW_MO)
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--viewport-height", type=int, default=720)
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    args = parser.parse_args(argv)
    if not args.answers_source:
        parser.error("--answers is required")
    if args.num_runs is not None and args.num_runs < 1:
        parser.error("--num-runs must be >= 1")
    if args.timeout_ms < 1000:
        parser.error("--timeout-ms must be >= 1000")
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
    runs_dir: Path,
    run_index: int,
    skip_existing: bool,
    existing_indices: set,
    skip_existing_video: bool,
    overwrite_existing: bool,
) -> Tuple[Optional[Path], str, int]:
    run_name = f"run_{run_index:04d}"
    run_dir = runs_dir / run_name
    if run_dir.exists():
        existing_indices.add(run_index)
        if overwrite_existing:
            shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir, run_name, run_index
        if skip_existing_video and any(run_dir.rglob("*.webm")):
            return None, run_name, run_index
        if skip_existing:
            return ensure_run_dir(
                runs_dir,
                run_index + 1,
                skip_existing,
                existing_indices,
                skip_existing_video,
                overwrite_existing,
            )
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    existing_indices.add(run_index)
    return run_dir, run_name, run_index


def finalize_video(run_dir: Path, form_id: str, run_name: str) -> Optional[Path]:
    try:
        candidate_videos = sorted(
            run_dir.rglob("*.webm"), key=lambda p: p.stat().st_mtime, reverse=True
        )
    except Exception:
        candidate_videos = []
    if not candidate_videos:
        return None
    raw_video_path = candidate_videos[0]
    final_video_path = run_dir / f"{form_id}_{run_name}.webm"
    if raw_video_path == final_video_path:
        return raw_video_path
    if final_video_path.exists():
        final_video_path.unlink()
    try:
        raw_video_path.rename(final_video_path)
        return final_video_path
    except Exception:
        return raw_video_path


def validate_run_artifacts(
    run_dir: Path,
    video_path: Optional[Path],
    answers_path: Path,
    annotations_path: Path,
    trace_path: Path,
    observations_dir: Path,
) -> None:
    missing: List[str] = []
    if not answers_path.exists():
        missing.append("answers_instance.json")
    if not annotations_path.exists():
        missing.append("annotations.json")
    if not trace_path.exists() or trace_path.stat().st_size == 0:
        missing.append("tool_trace.jsonl (missing or empty)")
    if not observations_dir.exists():
        missing.append("observations/")
    else:
        submit_pre = observations_dir / "submit_pre.png"
        submit_post = observations_dir / "submit_post.png"
        if not submit_pre.exists():
            missing.append("observations/submit_pre.png")
        if not submit_post.exists():
            missing.append("observations/submit_post.png")
    if video_path is None or not video_path.exists() or video_path.stat().st_size == 0:
        missing.append("final video (.webm)")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required artifacts in {run_dir}: {joined}")


def _playwright_import_error() -> RuntimeError:
    return RuntimeError(
        "Playwright is not installed. Run:\n"
        "  python -m playwright install chromium\n"
        "  python -m playwright install --with-deps chromium  # Linux"
    )


def _playwright_browser_error(exc: Exception) -> RuntimeError:
    msg = str(exc)
    return RuntimeError(
        "Playwright browser install appears missing. Run:\n"
        "  python -m playwright install chromium\n"
        "  python -m playwright install --with-deps chromium  # Linux\n"
        f"Original error: {msg}"
    )


def run_single(
    form_id: str,
    form_url: str,
    answers: List[Dict[str, Any]],
    run_dir: Path,
    run_name: str,
    run_label: str,
    args: argparse.Namespace,
    run_metadata: Dict[str, Any],
) -> None:
    answers_path = run_dir / "answers_instance.json"
    answers_path.write_text(json.dumps(answers, indent=2))

    observations_dir = run_dir / "observations"
    observations_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "tool_trace.jsonl"

    start_time = time.perf_counter()
    trace = TraceLogger(trace_path, start_time)

    annotations: Dict[str, Any] = {
        "schema_version": "3.0",
        "form_id": form_id,
        "run_name": run_name,
        "run_label": run_label,
        "form_url": form_url,
        "viewport": {"width": args.viewport_width, "height": args.viewport_height},
        "device_pixel_ratio": None,
        "user_agent": None,
        "locale": None,
        "timezone": None,
        "video_path": None,
        "run_params": {
            "headless": bool(args.headless),
            "slow_mo": args.slow_mo,
            "timeout_ms": args.timeout_ms,
        },
        "actions": [],
        "submit": {
            "success": False,
            "t_start_s": trace.now(),
            "t_end_s": trace.now(),
            "bbox": None,
            "submit_clicked": False,
            "confirmation_method": None,
            "final_url": None,
            "pre_screenshot": None,
            "post_screenshot": None,
        },
        "trace": {
            "tool_trace_path": "tool_trace.jsonl",
            "screenshot_dir": "observations",
        },
        "submitted": False,
        "failure_reason": None,
    }

    errors: List[str] = []
    run_error: Optional[Exception] = None

    page = None
    context = None
    browser = None

    try:
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            raise _playwright_import_error() from exc

        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=args.headless, slow_mo=args.slow_mo)
                context = browser.new_context(
                    viewport={"width": args.viewport_width, "height": args.viewport_height},
                    record_video_dir=str(run_dir),
                    record_video_size={"width": args.viewport_width, "height": args.viewport_height},
                )
                page = context.new_page()
                page.set_default_timeout(args.timeout_ms)
                page.goto(form_url, wait_until="load", timeout=args.timeout_ms)
                env = page.evaluate(
                    "() => ({devicePixelRatio: window.devicePixelRatio || null, userAgent: navigator.userAgent || null, locale: navigator.language || null, timezone: (Intl.DateTimeFormat().resolvedOptions().timeZone || null)})"
                )
                if isinstance(env, dict):
                    annotations["device_pixel_ratio"] = env.get("devicePixelRatio")
                    annotations["user_agent"] = env.get("userAgent")
                    annotations["locale"] = env.get("locale")
                    annotations["timezone"] = env.get("timezone")

                engine = FormEngine(
                    page=page,
                    viewport={"width": args.viewport_width, "height": args.viewport_height},
                    observations_dir=observations_dir,
                    trace=trace,
                    timeout_ms=args.timeout_ms,
                )

                for idx, entry in enumerate(answers):
                    label = entry.get("label") if isinstance(entry, dict) else None
                    if label:
                        print(f"[INFO] Filling step {idx}: {label}")
                    else:
                        print(f"[INFO] Filling step {idx}")
                    action, err = engine.fill_step(entry, idx)
                    annotations["actions"].append(action)
                    if err:
                        errors.append(f"step {idx}: {err}")

                print("[INFO] Submitting form")
                submit_info, submit_err = engine.submit()
                annotations["submit"] = submit_info
                if submit_err:
                    errors.append(f"submit: {submit_err}")
            except PlaywrightError as exc:
                raise _playwright_browser_error(exc) from exc
            except PlaywrightTimeoutError as exc:
                run_error = exc
                errors.append(f"timeout: {exc}")
            except Exception as exc:
                run_error = exc
                errors.append(f"run_error: {exc}")
            finally:
                if page is not None:
                    try:
                        page.close()
                    except Exception:
                        pass
                if context is not None:
                    try:
                        context.close()
                    except Exception:
                        pass
                if browser is not None:
                    try:
                        browser.close()
                    except Exception:
                        pass
    finally:
        trace.close()

    video_path = finalize_video(run_dir, form_id, run_name)
    annotations["video_path"] = str(video_path) if video_path else None

    submitted = bool(annotations.get("submit", {}).get("success"))
    annotations["submitted"] = submitted
    failure_reason = None
    if not submitted:
        if annotations.get("submit", {}).get("submit_clicked") is False:
            failure_reason = "submit_not_clicked"
        else:
            failure_reason = "confirmation_not_detected"
    if run_error is not None and not submitted:
        failure_reason = "run_error"
    annotations["failure_reason"] = failure_reason

    if run_metadata:
        annotations["run_metadata"] = run_metadata

    annotations_path = run_dir / "annotations.json"
    annotations_path.write_text(json.dumps(annotations, indent=2))

    validate_run_artifacts(
        run_dir=run_dir,
        video_path=video_path,
        answers_path=answers_path,
        annotations_path=annotations_path,
        trace_path=trace_path,
        observations_dir=observations_dir,
    )

    screenshot_count = len(list(observations_dir.glob("*.png")))
    print("[INFO] Run complete")
    print(f"[INFO] run_dir: {run_dir}")
    print(f"[INFO] video: {video_path}")
    print(f"[INFO] screenshots: {screenshot_count}")
    print(f"[INFO] trace: {trace_path}")

    if run_error is not None:
        raise RuntimeError(f"Run failed: {run_error}")
    if errors:
        joined = "; ".join(errors)
        raise RuntimeError(f"Run completed with errors: {joined}")


def main(argv: Optional[List[str]] = None) -> bool:
    args = parse_args(argv)
    specs_root = Path(args.specs_root) if args.specs_root else (SRC_DIR / "forms")
    form_spec = load_form_spec(args.form_id, specs_root)
    form_url = args.form_url or form_spec.get("form_url")
    if not form_url:
        raise ValueError("Form URL not provided and not found in spec")

    if os.environ.get("PLAYWRIGHT_SKIP_FFMPEG_INSTALL"):
        print(
            "[WARN] PLAYWRIGHT_SKIP_FFMPEG_INSTALL is set; video recording may fail.",
            file=sys.stderr,
        )

    answers_path = Path(args.answers_source).resolve()
    run_specs_iter = iter_run_specs(answers_path)

    dataset_root = Path(args.dataset_root).resolve()
    runs_dir = dataset_root / args.form_id / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    existing_indices = set(existing_run_indices(runs_dir))
    start_index = args.start_index or 1
    if start_index < 1:
        raise ValueError("--start-index must be >= 1")
    skip_existing = args.skip_existing or args.resume
    if args.resume and args.start_index is None:
        start_index = next_available_index(existing_indices, 1)

    generated = 0
    current_index = start_index

    for spec in run_specs_iter:
        if args.num_runs is not None and generated >= args.num_runs:
            break
        if skip_existing:
            current_index = next_available_index(existing_indices, current_index)
        run_dir, run_name, run_index = ensure_run_dir(
            runs_dir,
            current_index,
            skip_existing,
            existing_indices,
            args.skip_existing_video,
            args.overwrite_existing,
        )
        if run_dir is None:
            print(f"[INFO] Skipping existing video for {run_name}.")
            current_index = run_index + 1
            continue
        run_label = f"{args.form_id}_{run_name}"
        answers = spec.get("answers", [])
        if not isinstance(answers, list):
            raise ValueError("Run answers must be a list")
        run_metadata = spec.get("metadata", {})

        run_single(
            form_id=args.form_id,
            form_url=form_url,
            answers=answers,
            run_dir=run_dir,
            run_name=run_name,
            run_label=run_label,
            args=args,
            run_metadata=run_metadata,
        )

        generated += 1
        current_index = run_index + 1

    return True


if __name__ == "__main__":
    main()
