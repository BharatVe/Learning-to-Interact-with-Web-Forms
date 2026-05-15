import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baselines import run_baseline_eval as rbe


class ReferenceEfficiencyHelperTests(TestCase):
    def test_resolve_reference_efficiency_uses_matching_reference_run(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference_root = root / "data/forms/conf_interest/runs/run_0001"
            reference_root.mkdir(parents=True, exist_ok=True)
            (reference_root / "annotations.json").write_text(
                json.dumps(
                    {
                        "video_path": str(reference_root / "ref.webm"),
                        "actions": [
                            {"t_end_s": 3.0},
                            {"t_end_s": 7.5},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (reference_root / "tool_trace.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"name": "browser_type", "t_s": 1.0}),
                        json.dumps({"name": "browser_click", "t_s": 2.0}),
                        json.dumps({"name": "browser_press_key", "t_s": 3.0}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            model_trace = root / "trial/tool_trace.jsonl"
            model_trace.parent.mkdir(parents=True, exist_ok=True)
            model_trace.write_text(
                "\n".join(
                    [
                        json.dumps({"name": "browser_type", "t_s": 1.0}),
                        json.dumps({"name": "browser_click", "t_s": 2.0}),
                        json.dumps({"name": "browser_wait_for", "t_s": 4.0}),
                        json.dumps({"name": "browser_take_screenshot", "t_s": 5.0}),
                        json.dumps({"name": "browser_press_key", "t_s": 6.0}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.object(rbe, "ROOT_DIR", root):
                payload = rbe._resolve_reference_efficiency(
                    form_id="conf_interest",
                    answer_run_id="run_0001",
                    model_duration_s=15.0,
                    model_trace_path=model_trace,
                    model_action_count=4,
                )
            self.assertTrue(payload["reference_available"])
            self.assertEqual(payload["reference_action_count"], 3)
            self.assertEqual(payload["reference_duration_s"], 7.5)
            self.assertEqual(payload["trace_action_count"], 5)
            self.assertEqual(payload["trace_action_count_source"], "trace_overrides_summary_field")
            self.assertEqual(payload["action_count_delta"], 2)
            self.assertEqual(payload["duration_delta_s"], 7.5)
            self.assertAlmostEqual(payload["action_overhead_ratio"], 5 / 3, places=6)
            self.assertEqual(payload["time_overhead_ratio"], 2.0)

    def test_resolve_reference_efficiency_can_prefer_model_action_count(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference_root = root / "data/forms/conf_interest/runs/run_0001"
            reference_root.mkdir(parents=True, exist_ok=True)
            (reference_root / "annotations.json").write_text(json.dumps({"actions": [{"t_end_s": 2.0}]}), encoding="utf-8")
            (reference_root / "tool_trace.jsonl").write_text(
                "\n".join([json.dumps({"name": "browser_click", "t_s": 1.0}), json.dumps({"name": "browser_type", "t_s": 2.0})]) + "\n",
                encoding="utf-8",
            )
            model_trace = root / "trial/tool_trace.jsonl"
            model_trace.parent.mkdir(parents=True, exist_ok=True)
            model_trace.write_text(
                "\n".join(json.dumps({"name": "browser_run_code", "t_s": float(i)}) for i in range(20)) + "\n",
                encoding="utf-8",
            )
            with patch.object(rbe, "ROOT_DIR", root):
                payload = rbe._resolve_reference_efficiency(
                    form_id="conf_interest",
                    answer_run_id="run_0001",
                    model_duration_s=4.0,
                    model_trace_path=model_trace,
                    model_action_count=3,
                    prefer_model_action_count=True,
                )
            self.assertEqual(payload["trace_action_count"], 3)
            self.assertEqual(payload["trace_action_count_source"], "summary_field_overrides_trace")
            self.assertEqual(payload["action_count_delta"], 1)
            self.assertAlmostEqual(payload["action_overhead_ratio"], 1.5, places=6)

    def test_missing_reference_is_nonfatal(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_trace = root / "trial/tool_trace.jsonl"
            model_trace.parent.mkdir(parents=True, exist_ok=True)
            model_trace.write_text(json.dumps({"name": "browser_click", "t_s": 1.0}) + "\n", encoding="utf-8")
            with patch.object(rbe, "ROOT_DIR", root):
                payload = rbe._resolve_reference_efficiency(
                    form_id="missing_form",
                    answer_run_id="run_0001",
                    model_duration_s=10.0,
                    model_trace_path=model_trace,
                    model_action_count=1,
                )
            self.assertFalse(payload["reference_available"])
            self.assertIsNone(payload["reference_action_count"])
            self.assertIsNone(payload["reference_duration_s"])
            self.assertIsNone(payload["action_overhead_ratio"])


class ReferenceEfficiencySummaryTests(TestCase):
    def test_summarize_reference_efficiency_outputs_model_aggregates(self):
        script = REPO_ROOT / "scripts" / "summarize_reference_efficiency.py"
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "data/model_baselines"
            experiment_dir = dataset_root / "exp_a"
            experiment_dir.mkdir(parents=True, exist_ok=True)
            summary_path = root / "summaries/trial.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "experiment_id": "exp_a",
                        "trial_id": "trial_1",
                        "model_id": "text_model",
                        "model_kind": "text_llm",
                        "track": "direct_mcp_tool_use",
                        "form_id": "conf_interest",
                        "answer_run_id": "run_0001",
                        "success": True,
                        "submit_success": True,
                        "failure_category": None,
                        "question_total": 2,
                        "attempted_correctness": 2,
                        "verified_correctness": 2,
                        "duration_s": 20.0,
                        "action_count": 5,
                        "trace_action_count": 5,
                        "reference_available": True,
                        "reference_action_count": 4,
                        "reference_duration_s": 10.0,
                        "action_overhead_ratio": 1.25,
                        "time_overhead_ratio": 2.0,
                        "action_count_delta": 1,
                        "duration_delta_s": 10.0,
                    }
                ),
                encoding="utf-8",
            )
            (experiment_dir / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "experiment_id": "exp_a",
                        "trial_id": "trial_1",
                        "model_id": "text_model",
                        "model_kind": "text_llm",
                        "track": "direct_mcp_tool_use",
                        "form_id": "conf_interest",
                        "answer_run_id": "run_0001",
                        "summary_path": str(summary_path),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = root / "logs/reference_efficiency.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset-root",
                    str(dataset_root),
                    "--experiment-id",
                    "exp_a",
                    "--output",
                    str(output_path),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=(proc.stdout or "") + (proc.stderr or ""))
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["trial_count"], 1)
            self.assertEqual(report["per_model"]["text_model"]["median_action_overhead_ratio"], 1.25)
            self.assertEqual(report["per_model_kind"]["text_llm"]["median_time_overhead_ratio"], 2.0)
