#!/usr/bin/env python3
"""Analyze scripted reference form runs and write thesis efficiency inputs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_DATASET_ROOT = Path("data/forms")
DEFAULT_FORMS_ROOT = Path("src/forms")
DEFAULT_OUTPUT_DIR = Path("docs/eval_results/reference_analysis")
DEFAULT_RUNS = "1-6"


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def parse_run_ids(raw: str) -> List[str]:
    run_ids: List[int] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            run_ids.extend(range(int(left), int(right) + 1))
        else:
            run_ids.append(int(part))
    return [f"run_{idx:04d}" for idx in sorted(set(run_ids))]


def discover_form_ids(forms_root: Path = DEFAULT_FORMS_ROOT) -> List[str]:
    return sorted(entry.name for entry in forms_root.iterdir() if entry.is_dir() and (entry / "spec.json").exists())


def trace_stats(trace_path: Path) -> Dict[str, Any]:
    if not trace_path.exists():
        return {
            "trace_exists": False,
            "trace_valid": False,
            "trace_error": "missing_trace",
            "action_count": 0,
            "first_event_time_s": None,
            "last_event_time_s": None,
            "duration_s": None,
            "tool_counts": {},
        }
    tool_counts: Counter[str] = Counter()
    first_t: Optional[float] = None
    last_t: Optional[float] = None
    valid_events = 0
    try:
        lines = trace_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return {
            "trace_exists": True,
            "trace_valid": False,
            "trace_error": f"read_error: {exc}",
            "action_count": 0,
            "first_event_time_s": None,
            "last_event_time_s": None,
            "duration_s": None,
            "tool_counts": {},
        }
    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception as exc:
            return {
                "trace_exists": True,
                "trace_valid": False,
                "trace_error": f"invalid_json_line_{line_number}: {exc}",
                "action_count": valid_events,
                "first_event_time_s": first_t,
                "last_event_time_s": last_t,
                "duration_s": None if first_t is None or last_t is None else round(last_t - first_t, 6),
                "tool_counts": dict(sorted(tool_counts.items())),
            }
        if not isinstance(payload, dict):
            return {
                "trace_exists": True,
                "trace_valid": False,
                "trace_error": f"invalid_record_line_{line_number}",
                "action_count": valid_events,
                "first_event_time_s": first_t,
                "last_event_time_s": last_t,
                "duration_s": None if first_t is None or last_t is None else round(last_t - first_t, 6),
                "tool_counts": dict(sorted(tool_counts.items())),
            }
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        t_s = _optional_float(payload.get("t_s"))
        valid_events += 1
        tool_counts[name] += 1
        if t_s is not None:
            first_t = t_s if first_t is None else min(first_t, t_s)
            last_t = t_s if last_t is None else max(last_t, t_s)
    duration = None if first_t is None or last_t is None else round(last_t - first_t, 6)
    return {
        "trace_exists": True,
        "trace_valid": valid_events > 0,
        "trace_error": "" if valid_events > 0 else "no_valid_events",
        "action_count": valid_events,
        "first_event_time_s": first_t,
        "last_event_time_s": last_t,
        "duration_s": duration,
        "tool_counts": dict(sorted(tool_counts.items())),
    }


def annotations_duration_s(annotations: Dict[str, Any]) -> Optional[float]:
    times: List[float] = []
    actions = annotations.get("actions")
    if isinstance(actions, list):
        for action in actions:
            if not isinstance(action, dict):
                continue
            for key in ["t_start_s", "t_end_s"]:
                value = _optional_float(action.get(key))
                if value is not None:
                    times.append(value)
    submit = annotations.get("submit")
    if isinstance(submit, dict):
        for key in ["t_start_s", "t_end_s"]:
            value = _optional_float(submit.get(key))
            if value is not None:
                times.append(value)
    if not times:
        return None
    return round(max(times) - min(times), 6)


def analyze_reference_run(dataset_root: Path, form_id: str, run_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    run_root = dataset_root / form_id / "runs" / run_id
    annotations_path = run_root / "annotations.json"
    answers_path = run_root / "answers_instance.json"
    trace_path = run_root / "tool_trace.jsonl"
    failure_path = run_root / "failure_manifest.json"
    videos = sorted(run_root.glob("*.webm")) if run_root.exists() else []
    annotations = _read_json(annotations_path) if annotations_path.exists() else {}
    failure = _read_json(failure_path) if failure_path.exists() else {}
    trace = trace_stats(trace_path)
    submit = annotations.get("submit") if isinstance(annotations.get("submit"), dict) else {}
    submit_success = bool(submit.get("success"))
    video_path = str(videos[0]) if videos else ""
    video_available = bool(videos and videos[0].stat().st_size > 0)
    annotations_available = annotations_path.exists() and annotations_path.stat().st_size > 0
    answers_available = answers_path.exists() and answers_path.stat().st_size > 0
    failure_reason = ""
    if failure:
        failure_reason = str(failure.get("message") or failure.get("error_type") or "failure_manifest_present")
    elif not run_root.exists():
        failure_reason = "missing_run_dir"
    elif not annotations_available:
        failure_reason = "missing_annotations"
    elif not answers_available:
        failure_reason = "missing_answers_instance"
    elif not video_available:
        failure_reason = "missing_video"
    elif not trace["trace_valid"]:
        failure_reason = str(trace.get("trace_error") or "invalid_trace")
    elif not submit_success:
        failure_reason = str(annotations.get("failure_reason") or "submit_not_successful")
    usable = bool(annotations_available and answers_available and video_available and trace["trace_valid"] and submit_success)
    row = {
        "form_id": form_id,
        "answer_run_id": run_id,
        "run_path": str(run_root),
        "usable": usable,
        "run_exists": run_root.exists(),
        "annotations_available": annotations_available,
        "answers_instance_available": answers_available,
        "trace_available": bool(trace["trace_exists"]),
        "trace_valid": bool(trace["trace_valid"]),
        "video_available": video_available,
        "submit_success": submit_success,
        "failure_manifest_available": failure_path.exists(),
        "failure_reason": failure_reason,
        "video_path": video_path,
        "action_count": trace["action_count"],
        "first_event_time_s": "" if trace["first_event_time_s"] is None else trace["first_event_time_s"],
        "last_event_time_s": "" if trace["last_event_time_s"] is None else trace["last_event_time_s"],
        "duration_s": "" if trace["duration_s"] is None else trace["duration_s"],
        "annotations_duration_s": "" if annotations_duration_s(annotations) is None else annotations_duration_s(annotations),
        "trace_error": trace["trace_error"],
        "annotations_path": str(annotations_path),
        "answers_instance_path": str(answers_path),
        "trace_path": str(trace_path),
        "failure_manifest_path": str(failure_path) if failure_path.exists() else "",
    }
    breakdown = [
        {
            "form_id": form_id,
            "answer_run_id": run_id,
            "tool_name": name,
            "count": count,
        }
        for name, count in trace["tool_counts"].items()
    ]
    return row, breakdown


def analyze_reference_dataset(dataset_root: Path, forms_root: Path, run_ids: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    breakdown_rows: List[Dict[str, Any]] = []
    for form_id in discover_form_ids(forms_root):
        for run_id in run_ids:
            row, breakdown = analyze_reference_run(dataset_root, form_id, run_id)
            rows.append(row)
            breakdown_rows.extend(breakdown)
    total = len(rows)
    usable = sum(1 for row in rows if row["usable"])
    summary = [
        {
            "target_runs": total,
            "usable_runs": usable,
            "usable_rate": round(usable / total, 6) if total else 0.0,
            "missing_run_dirs": sum(1 for row in rows if not row["run_exists"]),
            "missing_annotations": sum(1 for row in rows if not row["annotations_available"]),
            "missing_answers_instance": sum(1 for row in rows if not row["answers_instance_available"]),
            "missing_traces": sum(1 for row in rows if not row["trace_available"]),
            "invalid_traces": sum(1 for row in rows if row["trace_available"] and not row["trace_valid"]),
            "missing_videos": sum(1 for row in rows if not row["video_available"]),
            "failed_runs": sum(1 for row in rows if row["failure_manifest_available"]),
            "submit_failures": sum(1 for row in rows if row["run_exists"] and row["annotations_available"] and not row["submit_success"]),
        }
    ]
    return rows, breakdown_rows, summary


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#202124}.title{font-size:17px;font-weight:700}.small{font-size:10px;fill:#5f6368}.label{font-weight:700}</style>',
        '<rect width="100%" height="100%" fill="#fff"/>',
    ]


def write_coverage_svg(summary: Dict[str, Any], path: Path) -> None:
    width, height = 760, 180
    left, top, chart_w = 190, 80, 420
    usable = int(summary.get("usable_runs") or 0)
    total = max(1, int(summary.get("target_runs") or 0))
    lines = _svg_header(width, height)
    lines.append('<text class="title" x="24" y="30">Reference Coverage</text>')
    lines.append('<text class="small" x="24" y="52">Usable scripted references for efficiency comparisons</text>')
    lines.append('<text class="label" x="24" y="94">Usable runs</text>')
    lines.append(f'<rect x="{left}" y="{top}" width="{chart_w}" height="22" fill="#eef1f4"/>')
    lines.append(f'<rect x="{left}" y="{top}" width="{chart_w * usable / total:.1f}" height="22" fill="#2563eb"/>')
    lines.append(f'<text x="{left + chart_w + 16}" y="{top + 16}">{usable}/{total} ({usable / total * 100:.1f}%)</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metric_distribution_svg(rows: Sequence[Dict[str, Any]], path: Path, *, key: str, title: str, suffix: str = "") -> None:
    values = sorted(float(row[key]) for row in rows if row.get("usable") and row.get(key) not in ("", None))
    width, height = 920, 260
    left, top, chart_w, chart_h = 70, 70, 760, 120
    lines = _svg_header(width, height)
    lines.append(f'<text class="title" x="24" y="30">{html.escape(title)}</text>')
    if not values:
        lines.append('<text x="24" y="80">No usable reference runs.</text>')
        lines.append("</svg>")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    max_value = max(values) or 1.0
    bins = 12
    counts = [0 for _ in range(bins)]
    for value in values:
        idx = min(bins - 1, int(value / max_value * bins))
        counts[idx] += 1
    max_count = max(counts) or 1
    for idx, count in enumerate(counts):
        x = left + idx * chart_w / bins
        bar_w = chart_w / bins - 4
        bar_h = chart_h * count / max_count
        lines.append(f'<rect x="{x:.1f}" y="{top + chart_h - bar_h:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="#2563eb"/>')
    lines.append(f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#dadce0"/>')
    lines.append(f'<text class="small" x="{left}" y="{top + chart_h + 24}">0{suffix}</text>')
    lines.append(f'<text class="small" x="{left + chart_w - 44}" y="{top + chart_h + 24}">{max_value:.1f}{suffix}</text>')
    lines.append(f'<text class="small" x="24" y="230">n={len(values)}, median={values[len(values)//2]:.1f}{suffix}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tool_mix_svg(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    totals: Counter[str] = Counter()
    for row in rows:
        totals[str(row["tool_name"])] += int(row["count"])
    items = totals.most_common()
    width = 920
    height = 90 + max(1, len(items)) * 30
    left, chart_w = 240, 520
    max_count = max((count for _name, count in items), default=1)
    lines = _svg_header(width, height)
    lines.append('<text class="title" x="24" y="30">Reference Tool Mix</text>')
    for idx, (name, count) in enumerate(items):
        y = 70 + idx * 30
        lines.append(f'<text class="label" x="24" y="{y + 14}">{html.escape(name[:28])}</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{chart_w}" height="16" fill="#eef1f4"/>')
        lines.append(f'<rect x="{left}" y="{y}" width="{chart_w * count / max_count:.1f}" height="16" fill="#059669"/>')
        lines.append(f'<text class="small" x="{left + chart_w + 14}" y="{y + 13}">{count}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(rows: Sequence[Dict[str, Any]], breakdown: Sequence[Dict[str, Any]], summary: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_dir / "reference_runs.csv")
    _write_csv(breakdown, output_dir / "reference_action_breakdown.csv")
    _write_csv(summary, output_dir / "reference_coverage_summary.csv")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    write_coverage_svg(summary[0] if summary else {}, plots_dir / "reference_coverage.svg")
    write_metric_distribution_svg(rows, plots_dir / "reference_action_counts.svg", key="action_count", title="Reference Action Counts")
    write_metric_distribution_svg(rows, plots_dir / "reference_duration_distribution.svg", key="duration_s", title="Reference Duration Distribution", suffix="s")
    write_tool_mix_svg(breakdown, plots_dir / "reference_tool_mix.svg")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze scripted reference dataset artifacts.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--forms-root", default=str(DEFAULT_FORMS_ROOT))
    parser.add_argument("--runs", default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    run_ids = parse_run_ids(args.runs)
    rows, breakdown, summary = analyze_reference_dataset(Path(args.dataset_root), Path(args.forms_root), run_ids)
    write_outputs(rows, breakdown, summary, Path(args.output_dir))
    print(f"[INFO] wrote reference analysis to {args.output_dir}")
    if summary:
        print(f"[INFO] usable_references={summary[0]['usable_runs']}/{summary[0]['target_runs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
