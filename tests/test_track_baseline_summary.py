import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase


REPO_ROOT = Path(__file__).resolve().parents[1]


class TrackBaselineSummaryTests(TestCase):
    def _write_summary(self, base: Path, rel: str, payload: dict) -> Path:
        out = base / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload), encoding="utf-8")
        return out

    def test_summarize_track_baseline_outputs_per_track_and_accounting(self):
        script = REPO_ROOT / "scripts" / "summarize_track_baseline.py"
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "data/model_baselines"

            family_a_id = "exp_family_a"
            family_b_id = "exp_family_b"
            family_a_dir = dataset_root / family_a_id
            family_b_dir = dataset_root / family_b_id
            family_a_dir.mkdir(parents=True, exist_ok=True)
            family_b_dir.mkdir(parents=True, exist_ok=True)

            m_text_summary = self._write_summary(
                root,
                "summaries/m_text.json",
                {
                    "model_id": "text_qwen3_30b_a3b_instruct_2507",
                    "model_kind": "text_llm",
                    "track": "direct_mcp_tool_use",
                    "form_id": "conf_interest",
                    "answer_run_id": "run_0001",
                    "success": True,
                    "submit_success": True,
                    "question_total": 2,
                    "verified_correctness": 2,
                    "attempted_correctness": 2,
                    "duration_s": 10.0,
                    "composite_score": 0.9,
                },
            )
            m_vlm_summary = self._write_summary(
                root,
                "summaries/m_vlm.json",
                {
                    "model_id": "vlm_qwen3_vl_30b_a3b_instruct",
                    "model_kind": "vlm",
                    "track": "direct_mcp_tool_use",
                    "form_id": "conf_interest",
                    "answer_run_id": "run_0001",
                    "success": False,
                    "submit_success": False,
                    "question_total": 2,
                    "verified_correctness": 1,
                    "attempted_correctness": 1,
                    "duration_s": 12.0,
                    "failure_category": "max_steps_exceeded",
                },
            )
            d_summary = self._write_summary(
                root,
                "summaries/direct.json",
                {
                    "model_id": "computer_use_mcp_api",
                    "model_kind": "computer_use_agent",
                    "track": "computer_use_native",
                    "form_id": "conf_interest",
                    "answer_run_id": "run_0001",
                    "success": True,
                    "submit_success": True,
                    "question_total": 2,
                    "verified_correctness": 1,
                    "attempted_correctness": 2,
                    "duration_s": 8.0,
                    "failure_category": None,
                },
            )

            (family_a_dir / "manifest.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"summary_path": str(m_text_summary)}),
                        json.dumps({"summary_path": str(m_vlm_summary)}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (family_b_dir / "manifest.jsonl").write_text(
                json.dumps({"summary_path": str(d_summary)}) + "\n",
                encoding="utf-8",
            )

            out_path = root / "logs/track_summary.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset-root",
                    str(dataset_root),
                    "--family-a-experiment-id",
                    family_a_id,
                    "--family-b-experiment-id",
                    family_b_id,
                    "--expected-forms",
                    "1",
                    "--expected-runs-per-form",
                    "1",
                    "--output",
                    str(out_path),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=(proc.stdout or "") + (proc.stderr or ""))
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(report["trial_accounting"]["expected_total"], 3)
            self.assertEqual(report["trial_accounting"]["included_total"], 3)
            self.assertEqual(report["trial_accounting"]["missing_total"], 0)
            self.assertIn("text_llm", report["per_track"])
            self.assertIn("vlm", report["per_track"])
            self.assertIn("computer_use_agent", report["per_track"])
            self.assertEqual(
                report["per_model"]["vlm_qwen3_vl_30b_a3b_instruct"]["failure_categories"],
                {"max_steps_exceeded": 1},
            )

    def test_expected_trial_count_uses_configured_model_count_when_track_missing(self):
        script = REPO_ROOT / "scripts" / "summarize_track_baseline.py"
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "data/model_baselines"
            family_a_id = "exp_family_a"
            family_b_id = "exp_family_b"
            family_a_dir = dataset_root / family_a_id
            family_b_dir = dataset_root / family_b_id
            family_a_dir.mkdir(parents=True, exist_ok=True)
            family_b_dir.mkdir(parents=True, exist_ok=True)

            m_text_summary = self._write_summary(
                root,
                "summaries/m_text.json",
                {
                    "model_id": "text_qwen3_30b_a3b_instruct_2507",
                    "model_kind": "text_llm",
                    "track": "direct_mcp_tool_use",
                    "form_id": "conf_interest",
                    "answer_run_id": "run_0001",
                    "success": True,
                    "submit_success": True,
                    "question_total": 2,
                    "verified_correctness": 2,
                    "attempted_correctness": 2,
                    "duration_s": 10.0,
                },
            )
            m_vlm_summary = self._write_summary(
                root,
                "summaries/m_vlm.json",
                {
                    "model_id": "vlm_qwen3_vl_30b_a3b_instruct",
                    "model_kind": "vlm",
                    "track": "direct_mcp_tool_use",
                    "form_id": "conf_interest",
                    "answer_run_id": "run_0001",
                    "success": True,
                    "submit_success": True,
                    "question_total": 2,
                    "verified_correctness": 2,
                    "attempted_correctness": 2,
                    "duration_s": 12.0,
                },
            )

            (family_a_dir / "manifest.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"summary_path": str(m_text_summary)}),
                        json.dumps({"summary_path": str(m_vlm_summary)}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (family_b_dir / "manifest.jsonl").write_text("", encoding="utf-8")

            config_path = root / "configs/baselines/track_baseline_models.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(
                    {
                        "models": [
                            {"id": "text_qwen3_30b_a3b_instruct_2507"},
                            {"id": "vlm_qwen3_vl_30b_a3b_instruct"},
                            {"id": "computer_use_opencua_32b"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            out_path = root / "logs/track_summary_missing_direct.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset-root",
                    str(dataset_root),
                    "--family-a-experiment-id",
                    family_a_id,
                    "--family-b-experiment-id",
                    family_b_id,
                    "--config-path",
                    str(config_path),
                    "--expected-forms",
                    "1",
                    "--expected-runs-per-form",
                    "1",
                    "--output",
                    str(out_path),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=(proc.stdout or "") + (proc.stderr or ""))
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(report["trial_accounting"]["expected_model_count"], 3)
            self.assertEqual(report["trial_accounting"]["model_count_source"], "config_path")
            self.assertEqual(report["trial_accounting"]["expected_total"], 3)
            self.assertEqual(report["trial_accounting"]["included_total"], 2)
            self.assertEqual(report["trial_accounting"]["missing_total"], 1)
