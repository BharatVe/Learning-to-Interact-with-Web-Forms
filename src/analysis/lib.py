"""Helpers shared by the analysis modules.

Only helpers whose copies were identical live here. Similar-looking functions that
differ in rounding, error handling or statistics (e.g. the bootstrap variants, the
per-study `aggregate` and CSV writers) stay in their modules on purpose: merging
them would change committed thesis numbers.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_json_object(path: Path) -> Dict[str, Any]:
    """JSON object at `path`, or {} if missing/invalid/not an object."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Object rows of a JSONL file; missing file -> [], bad lines skipped."""
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def rel_to_cwd(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


# Path segments that identify a location inside this repository regardless of where the
# checkout lived when the path was recorded (the HPC workspace has moved before).
_REBASE_MARKERS = ("/data/model_baselines/", "/data/forms_localforms/", "/data/forms/", "/docs/eval_results/")


def resolve_stored_path(raw: Any, root: Path = REPO_ROOT) -> Optional[Path]:
    """Resolve a path recorded in a summary/manifest/annotation.

    Relative paths are taken against the repo root. Absolute paths that no longer exist
    (e.g. written before the workspace moved, or on another machine) are rebased onto
    this checkout via the first known data/docs segment. Returns None for empty input.
    """
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        return root / path
    if path.exists():
        return path
    for marker in _REBASE_MARKERS:
        if marker in text:
            candidate = root / (marker.strip("/") + "/" + text.split(marker, 1)[1])
            if candidate.exists():
                return candidate
    return path
