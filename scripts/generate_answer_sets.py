#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List


SUPPORTED_Q_TYPES = {"SHORT_TEXT", "PARAGRAPH", "DATE", "TIME", "SINGLE_CHOICE", "MULTI_CHOICE", "DROPDOWN"}
WIDGET_TYPE_MAP = {
    "SHORT_TEXT": "short_text",
    "PARAGRAPH": "paragraph_text",
    "DATE": "date",
    "TIME": "time",
    "SINGLE_CHOICE": "single_choice",
    "MULTI_CHOICE": "multi_choice",
    "DROPDOWN": "dropdown",
}


FIRST_NAMES = ["Avery", "Jordan", "Riley", "Taylor", "Casey", "Morgan", "Sam", "Noah", "Lena", "Priya"]
LAST_NAMES = ["Chen", "Rivera", "Patel", "Kim", "Singh", "Nguyen", "Meyer", "Khan", "Ortiz", "Bauer"]
DOMAINS = ["example.com", "mail.test", "research.edu", "demo.org"]
ORGS = ["TU Dresden", "Stanford University", "Acme Labs", "Open Systems Group", "DataWorks Institute"]
SHORT_TEXT_DEFAULTS = ["alpha", "beta", "gamma", "delta", "omega", "sigma"]
PARAGRAPH_TEMPLATES = [
    "I am interested in this topic and can provide practical insights from recent projects.",
    "My background combines research and implementation, with a focus on reproducible workflows.",
    "I would like to participate and contribute with structured feedback and observations.",
]
TIME_OPTIONS = ["08:30", "09:00", "10:15", "11:30", "13:00", "14:15", "15:30", "16:45"]
GLOBAL_SEED = 0


def _stable_seed(*parts: Any) -> int:
    raw = "|".join([str(GLOBAL_SEED), *(str(p) for p in parts)])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _split_options(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    items = [item.strip() for item in text.split(";")]
    return [item for item in items if item]


def _dict_reader(path: Path) -> Any:
    handle = path.open("r", encoding="utf-8", newline="")
    sample = handle.read(4096)
    handle.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
    except csv.Error:
        dialect = csv.excel
    return handle, csv.DictReader(handle, dialect=dialect)


def _read_questions_by_form(questions_csv: Path) -> Dict[str, List[Dict[str, Any]]]:
    by_form: Dict[str, List[Dict[str, Any]]] = {}
    handle, reader = _dict_reader(questions_csv)
    with handle:
        required_cols = {"form_id", "section_order", "q_order", "q_title", "q_type", "required", "options"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"questions csv missing columns: {sorted(missing)}")
        for row in reader:
            form_id = str(row.get("form_id") or "").strip()
            q_title = str(row.get("q_title") or "").strip()
            q_type = str(row.get("q_type") or "").strip().upper()
            if not form_id or not q_title:
                raise ValueError(f"malformed question row: {row}")
            if q_type not in SUPPORTED_Q_TYPES:
                raise ValueError(f"unsupported q_type '{q_type}' for form_id={form_id} question={q_title}")
            entry = {
                "form_id": form_id,
                "section_order": int(str(row.get("section_order") or "0").strip()),
                "q_order": int(str(row.get("q_order") or "0").strip()),
                "q_title": q_title,
                "q_type": q_type,
                "required": str(row.get("required") or "").strip().lower() in {"1", "true", "t", "yes", "y"},
                "options": _split_options(str(row.get("options") or "")),
            }
            by_form.setdefault(form_id, []).append(entry)
    for form_id, rows in by_form.items():
        rows.sort(key=lambda q: (q["section_order"], q["q_order"], q["q_title"]))
    return by_form


def _pick_name(form_id: str, run_idx: int) -> str:
    seed = _stable_seed("name", form_id, run_idx)
    first = FIRST_NAMES[seed % len(FIRST_NAMES)]
    last = LAST_NAMES[(seed // 7) % len(LAST_NAMES)]
    return f"{first} {last}"


def _pick_email(name: str, form_id: str, run_idx: int) -> str:
    seed = _stable_seed("email", form_id, run_idx)
    domain = DOMAINS[seed % len(DOMAINS)]
    local = name.lower().replace(" ", ".")
    return f"{local}.{run_idx}@{domain}"


def _generate_short_text(form_id: str, run_idx: int, q_idx: int, q_title: str) -> str:
    title = q_title.lower()
    if "email" in title:
        name = _pick_name(form_id, run_idx)
        return _pick_email(name, form_id, run_idx)
    if "name" in title:
        return _pick_name(form_id, run_idx)
    if "university" in title or "organization" in title or "company" in title:
        seed = _stable_seed("org", form_id, run_idx, q_idx)
        return ORGS[seed % len(ORGS)]
    if "number" in title or "count" in title or "guests" in title:
        seed = _stable_seed("count", form_id, run_idx, q_idx)
        return str(1 + (seed % 4))
    seed = _stable_seed("short", form_id, run_idx, q_idx)
    token = SHORT_TEXT_DEFAULTS[seed % len(SHORT_TEXT_DEFAULTS)]
    return f"{token}-{run_idx:02d}-{(seed // 13) % 100:02d}"


def _generate_paragraph(form_id: str, run_idx: int, q_idx: int, q_title: str) -> str:
    seed = _stable_seed("paragraph", form_id, run_idx, q_idx)
    template = PARAGRAPH_TEMPLATES[seed % len(PARAGRAPH_TEMPLATES)]
    return f"{template} ({q_title}; run {run_idx})"


def _generate_date(form_id: str, run_idx: int, q_idx: int) -> str:
    seed = _stable_seed("date", form_id, run_idx, q_idx)
    base = date(2025, 6, 1)
    value = base + timedelta(days=(seed % 180))
    return value.isoformat()


def _generate_time(form_id: str, run_idx: int, q_idx: int) -> str:
    seed = _stable_seed("time", form_id, run_idx, q_idx)
    return TIME_OPTIONS[seed % len(TIME_OPTIONS)]


def _generate_choice(options: List[str], form_id: str, run_idx: int, q_idx: int) -> str:
    if not options:
        raise ValueError("choice question missing options")
    seed = _stable_seed("single_choice", form_id, run_idx, q_idx)
    return options[seed % len(options)]


def _generate_multi_choice(options: List[str], form_id: str, run_idx: int, q_idx: int) -> List[str]:
    if not options:
        raise ValueError("multi-choice question missing options")
    seed = _stable_seed("multi_choice", form_id, run_idx, q_idx)
    rng = random.Random(seed)
    max_take = min(3, len(options))
    take = 1 if max_take == 1 else rng.randint(1, max_take)
    picked = rng.sample(options, k=take)
    picked.sort(key=lambda x: options.index(x))
    return picked


def _build_run_answers(form_id: str, questions: List[Dict[str, Any]], run_idx: int) -> List[Dict[str, Any]]:
    answers: List[Dict[str, Any]] = []
    for q_idx, q in enumerate(questions, start=1):
        q_type = q["q_type"]
        q_title = q["q_title"]
        if q_type == "SHORT_TEXT":
            value: Any = _generate_short_text(form_id, run_idx, q_idx, q_title)
        elif q_type == "PARAGRAPH":
            value = _generate_paragraph(form_id, run_idx, q_idx, q_title)
        elif q_type == "DATE":
            value = _generate_date(form_id, run_idx, q_idx)
        elif q_type == "TIME":
            value = _generate_time(form_id, run_idx, q_idx)
        elif q_type in {"SINGLE_CHOICE", "DROPDOWN"}:
            value = _generate_choice(q["options"], form_id, run_idx, q_idx)
        elif q_type == "MULTI_CHOICE":
            value = _generate_multi_choice(q["options"], form_id, run_idx, q_idx)
        else:
            raise ValueError(f"unsupported q_type {q_type}")
        answers.append(
            {
                "label": q_title,
                "widget_type": WIDGET_TYPE_MAP[q_type],
                "value": value,
                "tags": ["generated", q_type.lower()],
            }
        )
    return answers


def _write_form_outputs(
    form_id: str,
    questions: List[Dict[str, Any]],
    answers_root: Path,
    runs_per_form: int,
    rewrite: bool,
) -> List[Dict[str, str]]:
    form_dir = answers_root / form_id
    form_dir.mkdir(parents=True, exist_ok=True)
    runs_path = form_dir / "runs.json"
    answers_csv_path = form_dir / "answers.csv"
    if (runs_path.exists() or answers_csv_path.exists()) and not rewrite:
        raise FileExistsError(f"answers already exist for '{form_id}'. Re-run with --rewrite.")

    runs_payload = {"form_id": form_id, "description": "Deterministically generated answer sets.", "runs": []}
    csv_rows: List[Dict[str, str]] = []
    for run_idx in range(1, runs_per_form + 1):
        suffix = f"run_{run_idx:04d}"
        run_answers = _build_run_answers(form_id, questions, run_idx)
        runs_payload["runs"].append(
            {
                "suffix": suffix,
                "notes": f"generated_run_{run_idx:04d}",
                "answers": run_answers,
            }
        )
        for q, answer in zip(questions, run_answers):
            raw_value = answer["value"]
            if isinstance(raw_value, list):
                answer_cell = ";".join(str(item) for item in raw_value)
            else:
                answer_cell = str(raw_value)
            csv_rows.append(
                {
                    "form_id": form_id,
                    "run_id": suffix,
                    "q_title": q["q_title"],
                    "q_type": q["q_type"],
                    "answer": answer_cell,
                }
            )

    runs_path.write_text(json.dumps(runs_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    with answers_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["form_id", "run_id", "q_title", "q_type", "answer"])
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    return csv_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic runs.json + answers.csv answer sets for all forms.")
    parser.add_argument("--forms-root", default="src/forms")
    parser.add_argument("--questions-csv", default="From Generator - Questions.csv")
    parser.add_argument("--answers-root", default="data/answers")
    parser.add_argument("--runs-per-form", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260413)
    parser.add_argument("--rewrite", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    global GLOBAL_SEED
    args = _parse_args()
    if args.runs_per_form <= 0:
        raise ValueError("--runs-per-form must be positive")
    repo_root = Path(__file__).resolve().parents[1]
    forms_root = (repo_root / args.forms_root).resolve()
    questions_csv = (repo_root / args.questions_csv).resolve()
    answers_root = (repo_root / args.answers_root).resolve()
    answers_root.mkdir(parents=True, exist_ok=True)

    form_ids = sorted(entry.name for entry in forms_root.iterdir() if entry.is_dir() and (entry / "spec.json").exists())
    if not form_ids:
        raise RuntimeError(f"no form specs found in {forms_root}")
    questions_by_form = _read_questions_by_form(questions_csv)

    all_rows: List[Dict[str, str]] = []
    for form_id in form_ids:
        questions = questions_by_form.get(form_id)
        if not questions:
            raise ValueError(f"missing questions for form '{form_id}' in {questions_csv}")
        all_rows.extend(
            _write_form_outputs(
                form_id=form_id,
                questions=questions,
                answers_root=answers_root,
                runs_per_form=int(args.runs_per_form),
                rewrite=bool(args.rewrite),
            )
        )

    answers_master_path = answers_root / "answers_master.csv"
    with answers_master_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["form_id", "run_id", "q_title", "q_type", "answer"])
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"[INFO] forms_processed={len(form_ids)} runs_per_form={args.runs_per_form}")
    print(f"[INFO] answers_root={answers_root}")
    print(f"[INFO] answers_master={answers_master_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    GLOBAL_SEED = int(args.seed)
