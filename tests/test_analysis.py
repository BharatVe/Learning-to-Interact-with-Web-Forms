import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis import lib, overview, registry  # noqa: E402


def _summary(root: Path, exp: str, model: str, form: str, run: int, trial: str, **fields) -> Path:
    path = root / exp / model / form / f"run_{run:04d}" / trial / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"experiment_id": exp, "model_id": model, "form_id": form, "answer_run_id": f"run_{run:04d}", "trial_id": trial}
    payload.update(fields)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class ResolveStoredPathTests(TestCase):
    def test_relative_existing_stale_and_unknown(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "data/model_baselines/exp/m/f/run_0001/t/summary.json"
            target.parent.mkdir(parents=True)
            target.write_text("{}")
            self.assertEqual(lib.resolve_stored_path("data/model_baselines/exp/m/f/run_0001/t/summary.json", root), target)
            self.assertEqual(lib.resolve_stored_path(str(target), root), target)
            stale = "/data/horse/ws/old-workspace/Learning-to-Interact-with-Web-Forms/data/model_baselines/exp/m/f/run_0001/t/summary.json"
            self.assertEqual(lib.resolve_stored_path(stale, root), target)
            self.assertEqual(lib.resolve_stored_path("/nowhere/x.json", root), Path("/nowhere/x.json"))
            self.assertIsNone(lib.resolve_stored_path("", root))
            self.assertIsNone(lib.resolve_stored_path(None, root))

    def test_read_json_object_and_jsonl_are_tolerant(self):
        with TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("[1, 2]")
            self.assertEqual(lib.read_json_object(bad), {})
            self.assertEqual(lib.read_json_object(Path(tmp) / "missing.json"), {})
            jl = Path(tmp) / "x.jsonl"
            jl.write_text('{"a": 1}\nnot json\n\n[1]\n{"b": 2}\n')
            self.assertEqual(lib.load_jsonl(jl), [{"a": 1}, {"b": 2}])
            self.assertEqual(lib.load_jsonl(Path(tmp) / "none.jsonl"), [])


class OverviewTests(TestCase):
    def test_metrics_and_fallbacks(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _summary(root, "exp", "vlm_qwen3_vl_30b_a3b_instruct", "a", 2, "t1", success=True, submit_success=True, question_total=4,
                     scored_correctness=4, action_overhead_ratio=1.0, duration_s=10, stop_reason="submitted", run_completed_utc="2026-07-01T00:00:00Z")
            _summary(root, "exp", "vlm_qwen3_vl_30b_a3b_instruct", "b", 2, "t2", success=False, question_total=6,
                     verified_correctness=3, action_overhead_ratio=3.0, duration_s=30, stop_reason="max_steps", run_completed_utc="2026-07-02T00:00:00Z")
            _summary(root, "exp", "vlm_qwen3_vl_30b_a3b_instruct", "c", 10, "t3", success=False, question_total=0, stop_reason="max_steps")
            _summary(root, "lfexp", "computer_use_opencua_32b_direct_mcp", "lf_a", 2, "t4", task_mode="fill_only_done", question_total=2, scored_correctness=1)
            archived = root / "exp" / "_archive" / "vlm_qwen3_vl_30b_a3b_instruct" / "a" / "run_0002" / "old" / "summary.json"
            archived.parent.mkdir(parents=True)
            archived.write_text(json.dumps({"success": True, "question_total": 100, "scored_correctness": 0}))
            rows = {(r["experiment_id"], r["model_id"]): r for r in overview.build_rows(overview.iter_summaries(root))}
            row = rows[("exp", "vlm_qwen3_vl_30b_a3b_instruct")]
            self.assertEqual(row["trials"], 3)  # archive excluded
            self.assertEqual(row["forms"], 3)
            self.assertEqual(row["runs"], "2,10")
            self.assertAlmostEqual(row["success_rate"], 1 / 3, places=3)
            self.assertAlmostEqual(row["submit_rate"], 1 / 3, places=3)
            self.assertAlmostEqual(row["scored_accuracy"], 7 / 10)  # scored 4 + verified fallback 3 over 10 questions
            self.assertEqual(row["median_action_overhead"], 2.0)
            self.assertEqual(row["top_stop_reason"], "max_steps (2)")
            self.assertEqual((row["first_utc"], row["last_utc"]), ("2026-07-01", "2026-07-02"))
            self.assertEqual(row["platform"], "google")
            lf = rows[("lfexp", "computer_use_opencua_32b_direct_mcp")]
            self.assertEqual((lf["platform"], lf["task_mode"], lf["scored_accuracy"]), ("localforms", "fill_only_done", 0.5))

    def test_main_writes_csv_and_plot(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds"
            for i in range(3):
                _summary(root, "exp", "text_qwen3_30b_a3b_instruct_2507", f"f{i}", 1, f"t{i}", question_total=2, scored_correctness=i % 3)
            out = Path(tmp) / "out"
            self.assertEqual(overview.main(["--dataset-root", str(root), "--output-dir", str(out), "--min-trials", "1", "--quiet"]), 0)
            with (out / "experiment_overview.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["scored_accuracy"], "0.5")
            if importlib.util.find_spec("matplotlib"):
                self.assertTrue((out / "plots" / "experiment_overview_accuracy.svg").is_file())

    def test_family_assignment_is_stable(self):
        self.assertEqual(overview.model_family("vlm_qwen3_vl_30b_a3b_instruct_formfactory_style"), "FormFactory-style")
        self.assertEqual(overview.model_family("text_qwen25_7b_instruct"), "Qwen Text")
        self.assertEqual(overview.model_family("computer_use_gemini_35_flash_lowcost"), "Gemini")
        self.assertEqual(overview.model_family("scads_gemma4_31b_it_api"), "Other")
        self.assertEqual(len(overview.FAMILY_ORDER), len(overview.FAMILY_COLORS))


class StudyRegistryTests(TestCase):
    def test_every_study_command_resolves(self):
        self.assertGreaterEqual(len(registry.STUDIES), 8)
        for study in registry.STUDIES.values():
            for command in study.commands + ([study.check_command] if study.check_command else []):
                if command[0] == "-m":
                    self.assertIsNotNone(importlib.util.find_spec(command[1]), (study.name, command))
                else:
                    self.assertTrue((REPO_ROOT / command[0]).is_file(), (study.name, command))
            for output in study.outputs:
                parent = (REPO_ROOT / output.split("*")[0]).parent
                self.assertTrue(parent.exists(), (study.name, output))

    def test_core_report_config_matches_code_expectations(self):
        cfg = json.loads((REPO_ROOT / "configs/analysis/core_report.json").read_text())
        models = set(cfg["models"]["qwen"]) | {cfg["models"]["opencua_native"], cfg["models"]["opencua_direct_mcp"]}
        self.assertEqual(set(cfg["thesis_model_order"]), models)
        self.assertEqual(set(cfg["thesis_model_labels"]), models)
        self.assertEqual(set(cfg["thesis_colors"]), set(cfg["thesis_model_labels"].values()))
        from analysis import core_report

        self.assertEqual(core_report.TARGET_RUN_IDS[-1], f"run_{cfg['target_run_count']:04d}")


class ProvenanceManifestTests(TestCase):
    """evaluation_additions/manifest.json hashes every thesis input; files must match it."""

    def test_manifest_paths_exist_and_hashes_match(self):
        manifest = json.loads((REPO_ROOT / "evaluation_additions/manifest.json").read_text())
        dataset_dependent = []
        for entry in manifest["paper_inputs"]:
            path = REPO_ROOT / entry["path"]
            if entry["path"].startswith("data/model_baselines/") and not path.exists():
                dataset_dependent.append(entry["path"])  # raw trial data is gitignored; only on the cluster
                continue
            self.assertTrue(path.is_file(), entry["path"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"], entry["path"])
        self.assertLessEqual(len(dataset_dependent), 2)
