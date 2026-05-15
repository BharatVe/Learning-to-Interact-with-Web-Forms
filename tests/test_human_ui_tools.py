import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase


REPO_ROOT = Path(__file__).resolve().parents[1]


class HumanUIToolsTests(TestCase):
    def _load_module(self, rel_path: str):
        path = REPO_ROOT / rel_path
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_extract_model_size_parses_underscore_and_hyphen(self):
        mod = self._load_module("scripts/summarize_human_ui_attribution.py")
        self.assertEqual(mod._extract_model_size_b("text_qwen25_7b_instruct"), 7)
        self.assertEqual(mod._extract_model_size_b("vlm-qwen3-30b-a3b"), 30)
        self.assertEqual(mod._extract_model_size_b("model 8b"), 8)
        self.assertIsNone(mod._extract_model_size_b("qwen3_large"))

    def test_validate_answer_sets_detects_missing_runs_and_invalid_option(self):
        script = REPO_ROOT / "scripts" / "validate_answer_sets.py"
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            forms_root = root / "src/forms"
            answers_root = root / "data/answers"
            forms_master = root / "data/specs/forms_master.csv"
            form_dir = forms_root / "demo_form"
            form_dir.mkdir(parents=True, exist_ok=True)
            answers_form_dir = answers_root / "demo_form"
            answers_form_dir.mkdir(parents=True, exist_ok=True)
            forms_master.parent.mkdir(parents=True, exist_ok=True)

            (form_dir / "spec.json").write_text(json.dumps({"form_id": "demo_form", "form_url": "https://example.test/form"}), encoding="utf-8")
            with forms_master.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
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
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "form_id": "demo_form",
                        "section_order": "1",
                        "section_title": "s",
                        "q_order": "1",
                        "q_title": "Pick one",
                        "q_type": "SINGLE_CHOICE",
                        "required": "TRUE",
                        "options": "A;B",
                        "help_text": "",
                        "edit_url": "",
                        "published_url": "https://example.test/form",
                    }
                )
            (answers_form_dir / "runs.json").write_text(
                json.dumps(
                    {
                        "form_id": "demo_form",
                        "runs": [
                            {
                                "suffix": "run_0001",
                                "answers": [{"label": "Pick one", "widget_type": "single_choice", "value": "C"}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--forms-root",
                    str(forms_root),
                    "--answers-root",
                    str(answers_root),
                    "--forms-master",
                    str(forms_master),
                    "--form-ids",
                    "demo_form",
                    "--required-runs",
                    "2",
                    "--strict",
                    "--strict-exact-run-count",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            combined = (proc.stdout or "") + (proc.stderr or "")
            self.assertIn("expected exactly 2 runs", combined)
            self.assertIn("not in options", combined)

    def test_validate_answer_sets_strict_allows_superset_and_all_form_ids(self):
        script = REPO_ROOT / "scripts" / "validate_answer_sets.py"
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            forms_root = root / "src/forms"
            answers_root = root / "data/answers"
            forms_master = root / "data/specs/forms_master.csv"
            forms_master.parent.mkdir(parents=True, exist_ok=True)

            for form_id in ("demo_one", "demo_two"):
                form_dir = forms_root / form_id
                form_dir.mkdir(parents=True, exist_ok=True)
                answers_form_dir = answers_root / form_id
                answers_form_dir.mkdir(parents=True, exist_ok=True)
                (form_dir / "spec.json").write_text(
                    json.dumps({"form_id": form_id, "form_url": f"https://example.test/{form_id}"}),
                    encoding="utf-8",
                )
                (answers_form_dir / "runs.json").write_text(
                    json.dumps(
                        {
                            "form_id": form_id,
                            "runs": [
                                {"suffix": "run_0001", "answers": [{"label": "Name", "widget_type": "short_text", "value": "A"}]},
                                {"suffix": "run_0002", "answers": [{"label": "Name", "widget_type": "short_text", "value": "B"}]},
                                {"suffix": "run_0003", "answers": [{"label": "Name", "widget_type": "short_text", "value": "C"}]},
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            with forms_master.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
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
                    ],
                )
                writer.writeheader()
                for form_id in ("demo_one", "demo_two"):
                    writer.writerow(
                        {
                            "form_id": form_id,
                            "section_order": "1",
                            "section_title": "s",
                            "q_order": "1",
                            "q_title": "Name",
                            "q_type": "SHORT_TEXT",
                            "required": "TRUE",
                            "options": "",
                            "help_text": "",
                            "edit_url": "",
                            "published_url": f"https://example.test/{form_id}",
                        }
                    )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--forms-root",
                    str(forms_root),
                    "--answers-root",
                    str(answers_root),
                    "--forms-master",
                    str(forms_master),
                    "--form-ids",
                    "all",
                    "--required-runs",
                    "2",
                    "--required-run-indexes",
                    "1,2",
                    "--strict",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=(proc.stdout or "") + (proc.stderr or ""))
