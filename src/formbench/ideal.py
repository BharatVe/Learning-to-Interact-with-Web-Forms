"""Ideal (scripted Playwright) reference runs: generation and coverage.

Wraps src/engine/runner.py with the settings that produced the committed
reference dataset (official Playwright MCP server, headless, skip runs that
already have a video) and reports which form/run pairs are missing.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from formbench.common import FormbenchError, answer_run_id, discover_form_ids, dump_table, info, quote_cmd, run, short
from formbench.settings import Settings

def roots(settings: Settings, platform: str) -> Tuple[Path, Path, Path]:
    """(specs_root, answers_root, reference_root) for a platform."""
    if platform == "localforms":
        return settings.localforms_forms_root, settings.localforms_answers_root, settings.localforms_reference_root
    if platform != "google":
        raise FormbenchError(f"platform must be google or localforms, got {platform!r}")
    return settings.forms_root, settings.answers_root, settings.reference_root


def answer_count(answers_root: Path, form_id: str) -> int:
    path = answers_root / form_id / "runs.json"
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs") if isinstance(payload, dict) else payload
    return len(runs) if isinstance(runs, list) else 0


def contiguous_ranges(indexes: Sequence[int]) -> List[Tuple[int, int]]:
    """[1,2,3,5,6] -> [(1,3), (5,2)] as (start, count)."""
    ranges: List[Tuple[int, int]] = []
    for idx in sorted(set(indexes)):
        if ranges and ranges[-1][0] + ranges[-1][1] == idx:
            ranges[-1] = (ranges[-1][0], ranges[-1][1] + 1)
        else:
            ranges.append((idx, 1))
    return ranges


def coverage(settings: Settings, platform: str = "google", forms: Optional[List[str]] = None) -> List[Dict[str, object]]:
    specs_root, answers_root, reference_root = roots(settings, platform)
    rows = []
    for form_id in forms or discover_form_ids(specs_root):
        total = answer_count(answers_root, form_id)
        runs_dir = reference_root / form_id / "runs"
        failed = sorted(p.parent.name for p in runs_dir.glob("run_*/failure_manifest.json"))
        have_trace = sorted(p.parent.name for p in runs_dir.glob("run_*/tool_trace.jsonl") if p.parent.name not in failed and p.stat().st_size > 0)
        have_video = sorted(p.parent.name for p in runs_dir.glob("run_*/*.webm"))
        missing = [answer_run_id(i) for i in range(1, total + 1) if answer_run_id(i) not in have_trace]
        rows.append({
            "form_id": form_id, "answer_sets": total, "traces": len(have_trace), "videos": len(have_video),
            "failed": len(failed), "missing": ",".join(m.replace("run_000", "").replace("run_00", "") for m in missing) or "-",
        })
    return rows


def print_coverage(settings: Settings, platform: str, forms: Optional[List[str]] = None) -> None:
    rows = coverage(settings, platform, forms)
    print(dump_table(rows, ["form_id", "answer_sets", "traces", "videos", "failed", "missing"]))
    total = sum(int(r["answer_sets"]) for r in rows)
    traces = sum(int(r["traces"]) for r in rows)
    info(f"{platform}: {traces}/{total} reference runs have a tool trace ({len(rows)} forms)")


def generate(
    settings: Settings, forms: List[str], run_indexes: List[int], platform: str = "google", overwrite: bool = False,
    headed: bool = False, interaction_mode: str = "mcp_server", screenshots: bool = False,
    extra_args: Optional[List[str]] = None, dry_run: bool = False,
) -> int:
    specs_root, answers_root, reference_root = roots(settings, platform)
    failures = 0
    for form_id in forms:
        total = answer_count(answers_root, form_id)
        if total == 0:
            raise FormbenchError(f"no answer sets for {form_id} in {answers_root}", hint="make data")
        wanted = [i for i in run_indexes if i <= total] if run_indexes else list(range(1, total + 1))
        if run_indexes and len(wanted) < len(run_indexes):
            info(f"{form_id}: only {total} answer sets; skipping run indexes > {total}")
        for start, count in contiguous_ranges(wanted):
            argv = [
                str(settings.python_bin), str(settings.root / "src/engine/runner.py"),
                "--form-id", form_id, "--specs-root", str(specs_root), "--answers-root", str(answers_root),
                "--dataset-root", str(reference_root), "--start-index", str(start), "--num-runs", str(count),
                "--interaction-mode", interaction_mode, "--continue-on-run-error",
                "--overwrite-existing" if overwrite else "--skip-existing-video",
            ]
            if not headed:
                argv.append("--headless")
            if screenshots:
                argv.append("--screenshots")
            if platform == "localforms":
                argv += ["--form-url", f"{settings.localforms_base_url}/forms/{form_id}"]
            argv += list(extra_args or [])
            if dry_run:
                print(short(quote_cmd(argv)))
                continue
            code = run(argv, env=settings.subprocess_env(), cwd=settings.root)
            if code != 0:
                failures += 1
    return failures
