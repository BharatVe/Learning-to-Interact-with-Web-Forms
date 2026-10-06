"""Shared helpers for the dataset pipeline (generator CSVs -> specs -> answers -> LocalForms)."""

import csv
from pathlib import Path
from typing import Any, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
FORMS_CSV = "data/generator/forms.csv"
QUESTIONS_CSV = "data/generator/questions.csv"

# Google Forms question types exported by the generator sheet <-> engine widget types.
QTYPE_TO_WIDGET = {
    "SHORT_TEXT": "short_text",
    "PARAGRAPH": "paragraph_text",
    "DATE": "date",
    "TIME": "time",
    "SINGLE_CHOICE": "single_choice",
    "MULTI_CHOICE": "multi_choice",
    "DROPDOWN": "dropdown",
}
WIDGET_TO_QTYPE = {widget: qtype for qtype, widget in QTYPE_TO_WIDGET.items()}
SUPPORTED_Q_TYPES = set(QTYPE_TO_WIDGET)


def dict_reader(path: Path) -> Tuple[Any, csv.DictReader]:
    """Open a comma- or tab-separated file; caller closes the returned handle."""
    handle = path.open("r", encoding="utf-8", newline="")
    sample = handle.read(4096)
    handle.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
    except csv.Error:
        dialect = csv.excel
    return handle, csv.DictReader(handle, dialect=dialect)


def split_options(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [item.strip() for item in text.split(";") if item.strip()]


def as_bool(raw: str) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "t", "yes", "y"}


def as_int(raw: str, field: str, form_id: str, q_title: str) -> int:
    try:
        return int(str(raw or "").strip())
    except Exception as exc:
        raise ValueError(f"invalid integer '{field}' for form_id={form_id} question='{q_title}'") from exc


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate)
