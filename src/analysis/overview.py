"""Experiment overview: one row per (experiment, model) across every experiment on disk.

The core report (analysis.core_report) covers the thesis target study only; this
table gives the same headline metrics for all experiments (Gemini, fill-only,
FormFactory-style, LocalForms, ...), so any run can be compared at a glance.

Outputs: experiment_overview.csv and plots/experiment_overview_accuracy.svg.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional

from analysis.lib import read_json_object

DEFAULT_DATASET_ROOT = Path("data/model_baselines")
DEFAULT_OUTPUT_DIR = Path("docs/eval_results/analysis")
FIELDS = [
    "experiment_id", "model_id", "task_mode", "platform", "trials", "forms", "runs", "success_rate", "submit_rate",
    "scored_accuracy", "questions", "median_action_overhead", "median_duration_s", "top_stop_reason",
    "first_utc", "last_utc",
]
# Reference categorical palette order (light mode); assigned by family, never cycled.
FAMILY_ORDER = ["Qwen Text", "Qwen VLM", "OpenCUA", "Gemini", "FormFactory-style", "Other"]
FAMILY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def model_family(model_id: str) -> str:
    if "formfactory" in model_id:
        return "FormFactory-style"
    if model_id.startswith("text_qwen"):
        return "Qwen Text"
    if model_id.startswith("vlm_qwen"):
        return "Qwen VLM"
    if "opencua" in model_id:
        return "OpenCUA"
    if "gemini" in model_id:
        return "Gemini"
    return "Other"


def iter_summaries(dataset_root: Path) -> Iterable[Dict[str, Any]]:
    """summary.json of every live trial (<experiment>/<model>/<form>/run_XXXX/<trial>/), archive excluded."""
    for path in sorted(dataset_root.glob("*/*/*/run_*/*/summary.json")):
        rel = path.relative_to(dataset_root).parts
        if rel[0].startswith("_") or "_archive" in rel:
            continue
        payload = read_json_object(path)
        if payload:
            payload.setdefault("experiment_id", rel[0])
            payload.setdefault("model_id", rel[1])
            payload.setdefault("form_id", rel[2])
            payload.setdefault("answer_run_id", rel[3])
            yield payload


def _num(value: Any) -> Optional[float]:
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def _rate(values: List[bool]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def _scored(summary: Dict[str, Any]) -> int:
    for key in ("scored_correctness", "verified_correctness"):
        value = _num(summary.get(key))
        if value is not None:
            return int(value)
    return 0


def build_rows(summaries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for s in summaries:
        groups[(s["experiment_id"], s["model_id"])].append(s)
    rows = []
    for (experiment_id, model_id), items in sorted(groups.items()):
        totals = [int(_num(s.get("question_total")) or 0) for s in items]
        scored = [_scored(s) for s in items]
        overheads = [v for v in (_num(s.get("action_overhead_ratio")) for s in items) if v is not None]
        durations = [v for v in (_num(s.get("duration_s")) for s in items) if v is not None]
        stops = Counter(str(s.get("stop_reason") or "unknown") for s in items)
        top_stop, top_count = stops.most_common(1)[0]
        dates = sorted(str(s.get("run_completed_utc") or s.get("run_started_utc")) for s in items if s.get("run_completed_utc") or s.get("run_started_utc"))
        rows.append({
            "experiment_id": experiment_id,
            "model_id": model_id,
            "task_mode": "+".join(sorted({str(s.get("task_mode") or "fill_and_submit") for s in items})),
            "platform": "localforms" if any(str(s.get("form_id", "")).startswith("lf_") for s in items) else "google",
            "trials": len(items),
            "forms": len({s["form_id"] for s in items}),
            "runs": ",".join(sorted({str(int(str(s["answer_run_id"]).replace("run_", ""))) for s in items}, key=int)),
            "success_rate": _rate([bool(s.get("success")) for s in items]),
            "submit_rate": _rate([bool(s.get("submit_success")) for s in items]),
            "scored_accuracy": round(sum(scored) / sum(totals), 4) if sum(totals) else None,
            "questions": sum(totals),
            "median_action_overhead": round(median(overheads), 3) if overheads else None,
            "median_duration_s": round(median(durations), 1) if durations else None,
            "top_stop_reason": f"{top_stop} ({top_count})",
            "first_utc": dates[0][:10] if dates else "",
            "last_utc": dates[-1][:10] if dates else "",
        })
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in FIELDS})


def plot_accuracy(rows: List[Dict[str, Any]], path: Path, min_trials: int = 20) -> Optional[Path]:
    """Horizontal bars: scored accuracy per (experiment, model), colored by model family."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib missing; skipping overview plot (make setup)")
        return None
    data = [r for r in rows if r["trials"] >= min_trials and r["scored_accuracy"] is not None]
    if not data:
        return None
    data.sort(key=lambda r: (r["first_utc"], r["experiment_id"], r["model_id"]))
    families = [model_family(r["model_id"]) for r in data]
    colors = [FAMILY_COLORS[FAMILY_ORDER.index(f)] for f in families]
    fig, ax = plt.subplots(figsize=(12, 0.32 * len(data) + 1.6))
    y = list(range(len(data)))[::-1]
    ax.barh(y, [r["scored_accuracy"] * 100 for r in data], color=colors, height=0.7, edgecolor="#fcfcfb", linewidth=2)
    for yi, r in zip(y, data):
        ax.text(r["scored_accuracy"] * 100 + 0.8, yi, f"{r['scored_accuracy'] * 100:.0f}%  (n={r['trials']})", va="center", fontsize=7.5, color="#52514e")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['experiment_id']}  ·  {r['model_id']}" for r in data], fontsize=7.5, color="#0b0b0b")
    ax.set_xlim(0, 112)
    ax.set_xlabel("scored field accuracy (%)", color="#52514e")
    ax.set_title(f"Scored accuracy by experiment and model (groups with >= {min_trials} trials)", fontsize=11, loc="left", color="#0b0b0b", pad=22)
    ax.grid(axis="x", color="#e7e6e2", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#bdbcb6")
    ax.tick_params(axis="x", colors="#52514e")
    ax.tick_params(axis="y", length=0)
    present = [f for f in FAMILY_ORDER if f in families]
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[FAMILY_ORDER.index(f)]) for f in present]
    ax.legend(handles, present, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(present), frameon=False, fontsize=8)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format=path.suffix.lstrip(".") or "svg", **({"metadata": {"Date": None}} if path.suffix == ".svg" else {"dpi": 110}))
    plt.close(fig)
    return path


def format_table(rows: List[Dict[str, Any]], min_trials: int = 20) -> str:
    shown = [r for r in rows if r["trials"] >= min_trials]
    cols = ["experiment_id", "model_id", "trials", "success_rate", "scored_accuracy", "median_action_overhead"]
    widths = {c: max([len(c)] + [len(str(r.get(c, ""))) for r in shown]) for c in cols}
    lines = ["  ".join(c.ljust(widths[c]) for c in cols)]
    for r in shown:
        lines.append("  ".join(str("" if r.get(c) is None else r.get(c)).ljust(widths[c]) for c in cols))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Per-experiment overview table + accuracy plot.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-id", action="append", default=[], help="restrict to these experiments (repeatable)")
    parser.add_argument("--min-trials", type=int, default=20, help="plot/print only groups with at least this many trials")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    summaries = [s for s in iter_summaries(args.dataset_root) if not args.experiment_id or s["experiment_id"] in args.experiment_id]
    rows = build_rows(summaries)
    csv_path = args.output_dir / "experiment_overview.csv"
    write_csv(rows, csv_path)
    plot = plot_accuracy(rows, args.output_dir / "plots" / "experiment_overview_accuracy.svg", args.min_trials)
    print(f"[INFO] wrote {csv_path} ({len(rows)} experiment/model rows)" + (f" and {plot}" if plot else ""))
    if not args.quiet:
        print(format_table(rows, args.min_trials))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
