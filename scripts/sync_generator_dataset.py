#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from engine.browser_language import force_english_google_forms_url  # noqa: E402


SUPPORTED_Q_TYPES = {"SHORT_TEXT", "PARAGRAPH", "DATE", "TIME", "SINGLE_CHOICE", "MULTI_CHOICE", "DROPDOWN"}
FORMS_MASTER_COLUMNS = [
    "form_id",
    "section_order",
    "section_title",
    "q_order",
    "q_title",
    "q_type",
    "required",
    "options",
    "help_text",
    "edit_url",
    "published_url",
]


def _dict_reader(path: Path) -> Tuple[Any, csv.DictReader]:
    handle = path.open("r", encoding="utf-8", newline="")
    sample = handle.read(4096)
    handle.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
    except csv.Error:
        dialect = csv.excel
    return handle, csv.DictReader(handle, dialect=dialect)


def _as_bool(raw: str) -> bool:
    text = str(raw or "").strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _as_int(raw: str, field: str, form_id: str, q_title: str) -> int:
    try:
        return int(str(raw or "").strip())
    except Exception as exc:
        raise ValueError(f"invalid integer '{field}' for form_id={form_id} question='{q_title}'") from exc


def _read_forms(forms_csv: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    handle, reader = _dict_reader(forms_csv)
    with handle:
        required_cols = {"form_id", "form_title", "form_description", "active", "edit_url", "published_url"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"forms csv missing columns: {sorted(missing)}")
        for row in reader:
            form_id = str(row.get("form_id") or "").strip()
            if not form_id:
                continue
            published_url = str(row.get("published_url") or "").strip()
            if not published_url.startswith("http"):
                raise ValueError(f"form '{form_id}' has unusable published_url: {published_url!r}")
            rows[form_id] = {
                "form_id": form_id,
                "form_title": str(row.get("form_title") or "").strip(),
                "form_description": str(row.get("form_description") or "").strip(),
                "active": _as_bool(str(row.get("active") or "")),
                "edit_url": str(row.get("edit_url") or "").strip(),
                "published_url": published_url,
            }
    if not rows:
        raise ValueError(f"no forms found in {forms_csv}")
    return rows


def _read_questions(questions_csv: Path, form_rows: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_form: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    handle, reader = _dict_reader(questions_csv)
    with handle:
        required_cols = {"form_id", "section_order", "section_title", "q_order", "q_title", "q_type", "required", "options", "help_text"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"questions csv missing columns: {sorted(missing)}")
        for row in reader:
            form_id = str(row.get("form_id") or "").strip()
            q_title = str(row.get("q_title") or "").strip()
            q_type = str(row.get("q_type") or "").strip().upper()
            if not form_id:
                raise ValueError(f"question row missing form_id: {row}")
            if form_id not in form_rows:
                raise ValueError(f"question references unknown form_id '{form_id}'")
            if not q_title:
                raise ValueError(f"form '{form_id}' has question with empty q_title")
            if q_type not in SUPPORTED_Q_TYPES:
                raise ValueError(f"form '{form_id}' question '{q_title}' has unsupported q_type '{q_type}'")
            item = {
                "form_id": form_id,
                "section_order": _as_int(row.get("section_order", ""), "section_order", form_id, q_title),
                "section_title": str(row.get("section_title") or "").strip(),
                "q_order": _as_int(row.get("q_order", ""), "q_order", form_id, q_title),
                "q_title": q_title,
                "q_type": q_type,
                "required": _as_bool(str(row.get("required") or "")),
                "options": str(row.get("options") or "").strip(),
                "help_text": str(row.get("help_text") or "").strip(),
            }
            if q_type in {"SINGLE_CHOICE", "MULTI_CHOICE", "DROPDOWN"} and not item["options"]:
                raise ValueError(f"form '{form_id}' question '{q_title}' is {q_type} but has empty options")
            by_form[form_id].append(item)
    for form_id in sorted(form_rows.keys()):
        if not by_form.get(form_id):
            raise ValueError(f"form '{form_id}' has no questions in questions csv")
        by_form[form_id].sort(key=lambda r: (r["section_order"], r["q_order"], r["q_title"]))
    return by_form


def _write_form_specs(form_rows: Dict[str, Dict[str, Any]], questions_by_form: Dict[str, List[Dict[str, Any]],], output_forms_root: Path) -> List[str]:
    output_forms_root.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for form_id in sorted(form_rows.keys()):
        form = form_rows[form_id]
        questions = questions_by_form[form_id]
        spec_payload = {
            "form_id": form_id,
            "form_title": form["form_title"],
            "form_description": form["form_description"],
            "active": bool(form["active"]),
            "edit_url": form["edit_url"],
            "published_url": force_english_google_forms_url(form["published_url"]),
            "form_url": force_english_google_forms_url(form["published_url"]),
            "question_count": len(questions),
            "questions": [
                {
                    "section_order": q["section_order"],
                    "section_title": q["section_title"],
                    "q_order": q["q_order"],
                    "q_title": q["q_title"],
                    "q_type": q["q_type"],
                    "required": bool(q["required"]),
                    "options": q["options"],
                    "help_text": q["help_text"],
                }
                for q in questions
            ],
            "source": "From Generator CSV sync",
        }
        spec_dir = output_forms_root / form_id
        spec_dir.mkdir(parents=True, exist_ok=True)
        (spec_dir / "spec.json").write_text(json.dumps(spec_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        written.append(form_id)
    return written


def _write_forms_master(form_rows: Dict[str, Dict[str, Any]], questions_by_form: Dict[str, List[Dict[str, Any]]], forms_master_path: Path) -> int:
    forms_master_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with forms_master_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FORMS_MASTER_COLUMNS)
        writer.writeheader()
        for form_id in sorted(form_rows.keys()):
            form = form_rows[form_id]
            for q in questions_by_form[form_id]:
                writer.writerow(
                    {
                        "form_id": form_id,
                        "section_order": q["section_order"],
                        "section_title": q["section_title"],
                        "q_order": q["q_order"],
                        "q_title": q["q_title"],
                        "q_type": q["q_type"],
                        "required": "TRUE" if q["required"] else "FALSE",
                        "options": q["options"],
                        "help_text": q["help_text"],
                        "edit_url": form["edit_url"],
                        "published_url": form["published_url"],
                    }
                )
                count += 1
    return count


def _prune_stale_form_specs(output_forms_root: Path, keep_form_ids: List[str]) -> List[str]:
    removed: List[str] = []
    keep = set(keep_form_ids)
    if not output_forms_root.exists():
        return removed
    for entry in sorted(output_forms_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in keep:
            continue
        spec_path = entry / "spec.json"
        if spec_path.exists():
            shutil.rmtree(entry)
            removed.append(entry.name)
    return removed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync generator CSVs into src/forms specs and data/specs/forms_master.csv.")
    parser.add_argument("--forms-csv", default="From Generator - Forms.csv")
    parser.add_argument("--questions-csv", default="From Generator - Questions.csv")
    parser.add_argument("--output-forms-root", default="src/forms")
    parser.add_argument("--forms-master", default="data/specs/forms_master.csv")
    parser.add_argument("--prune", action="store_true", default=False, help="Delete stale form spec directories not present in forms csv.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    forms_csv = (repo_root / args.forms_csv).resolve()
    questions_csv = (repo_root / args.questions_csv).resolve()
    output_forms_root = (repo_root / args.output_forms_root).resolve()
    forms_master_path = (repo_root / args.forms_master).resolve()

    if not forms_csv.exists():
        raise FileNotFoundError(f"forms csv not found: {forms_csv}")
    if not questions_csv.exists():
        raise FileNotFoundError(f"questions csv not found: {questions_csv}")

    forms = _read_forms(forms_csv)
    questions_by_form = _read_questions(questions_csv, forms)
    written_form_ids = _write_form_specs(forms, questions_by_form, output_forms_root)
    row_count = _write_forms_master(forms, questions_by_form, forms_master_path)
    removed_form_ids: List[str] = []
    if args.prune:
        removed_form_ids = _prune_stale_form_specs(output_forms_root, written_form_ids)

    print(f"[INFO] forms_synced={len(written_form_ids)} questions_rows={row_count}")
    print(f"[INFO] forms_root={output_forms_root}")
    print(f"[INFO] forms_master={forms_master_path}")
    if removed_form_ids:
        print(f"[INFO] pruned_stale_forms={','.join(removed_form_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
