import json
import sys
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baselines import prompt_builders as pb
from baselines import run_baseline_eval as rbe


class FakeTrace:
    def summary(self):
        return {}

    def close(self):
        return None


class FakeSession:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir
        self.observations_dir = artifact_dir / "observations"

    def start(self, form_url: str):
        return {"url": form_url, "backend": "fake"}

    def observe(self, step_idx: int):
        path = self.observations_dir / f"step_{step_idx:04d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return {"page_text": "Name Submit", "screenshot_path": str(path)}

    def execute_fill(self, entry, step_idx: int):
        return ({"success": True, "step": step_idx, "value": entry.get("value")}, None)

    def verify_entry(self, entry, step_idx: int):
        return {"verified": True, "actual_value": entry.get("value"), "detail": None}

    def submit(self):
        return (
            {
                "success": True,
                "submit_clicked": True,
                "confirmation_method": "text",
                "final_url": "https://example.test/done",
                "pagination_hops": 0,
            },
            None,
        )

    def capture_terminal_screenshot(self, filename: str):
        path = self.artifact_dir / filename
        path.write_bytes(b"png")
        return str(path)

    def close(self):
        return None


class SequenceAdapter:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def infer(self, prompt, max_new_tokens_override=None):
        _ = prompt
        _ = max_new_tokens_override
        idx = self.calls
        self.calls += 1
        if idx >= len(self.outputs):
            return self.outputs[-1]
        return self.outputs[idx]


class BaselineEvalContractTests(TestCase):
    def _write_config(self, repo_root: Path, model_id: str = "text_qwen25_3b_instruct") -> None:
        config_path = repo_root / "configs/baselines/minimal_models.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "id": model_id,
                            "kind": "text_llm",
                            "provider": "local_hf",
                            "hf_repo": "dummy",
                            "track": "mediated",
                            "requires_gpu": False,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    def test_build_trial_paths_uses_flat_layout(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = Namespace(dataset_root="data/model_baselines", experiment_id="exp")
            with patch.object(rbe, "ROOT_DIR", root):
                paths = rbe._build_trial_paths(args, "model", "form", "run_0001", "trial_demo")
            self.assertEqual(paths["summary_path"], root / "data/model_baselines/exp/model/form/run_0001/trial_demo/summary.json")
            self.assertEqual(paths["annotations_path"], root / "data/model_baselines/exp/model/form/run_0001/trial_demo/annotations.json")
            self.assertEqual(paths["video_path"], root / "data/model_baselines/exp/model/form/run_0001/trial_demo/form_trial_demo.webm")
            self.assertEqual(paths["observations_dir"], root / "data/model_baselines/exp/model/form/run_0001/trial_demo/observations")

    def test_match_question_state_accepts_question_id(self):
        states = [
            {"question_id": "q_001", "label": "Full Name", "verified_correct": False},
            {"question_id": "q_002", "label": "Email", "verified_correct": False},
        ]
        idx, state, debug = rbe._match_question_state(states, {"question_id": "q_002"})
        self.assertEqual(idx, 1)
        self.assertEqual(state["label"], "Email")
        self.assertEqual(debug["match_strategy"], "question_id")

    def test_soft_timeout_retry_collects_attempts(self):
        class RetryAdapter:
            def __init__(self):
                self.calls = []

            def infer(self, prompt, max_new_tokens_override=None):
                _ = prompt
                self.calls.append(max_new_tokens_override)
                if len(self.calls) == 1:
                    raise rbe._SoftTimeoutError("model_step_soft_timeout:0.1s")
                return '{"action":"wait"}'

        args = Namespace(max_new_tokens=160, step_retry_max_new_tokens=96, step_soft_timeout_s=0)
        adapter = RetryAdapter()
        output, meta = rbe._infer_with_retry(
            adapter=adapter,
            args=args,
            model_kind="text_llm",
            prompt="x",
            image_path=None,
        )
        self.assertEqual(output, '{"action":"wait"}')
        self.assertEqual(len(meta["attempts"]), 2)
        self.assertTrue(meta["retried"])
        self.assertTrue(meta["attempts"][0]["timed_out"])
        self.assertEqual(adapter.calls, [160, 96])

    def test_parse_args_defaults_balanced_profile(self):
        args = rbe._parse_args(
            [
                "--model-id",
                "text_qwen25_3b_instruct",
                "--model-kind",
                "text_llm",
                "--form-id",
                "event_rsvp",
                "--run-index",
                "1",
            ]
        )
        self.assertEqual(args.max_steps, 36)
        self.assertEqual(args.timeout_s, 1800)
        self.assertEqual(args.max_new_tokens, 224)
        self.assertEqual(int(args.step_soft_timeout_s), 60)
        self.assertEqual(args.step_retry_max_new_tokens, 160)
        self.assertEqual(args.compact_page_text_max_chars, 5000)
        self.assertEqual(args.prompt_profile, "detailed_v1")
        self.assertEqual(args.history_window, 4)
        self.assertTrue(args.fewshot_enabled)
        self.assertEqual(args.fewshot_count, 3)
        self.assertTrue(args.disable_action_coercion)
        self.assertEqual(args.retention_window, 5)
        self.assertEqual(args.inference_backend, "auto")
        self.assertEqual(args.api_timeout_s, 120)
        self.assertEqual(args.browser_init_retries, 2)

    def test_select_inference_backend_for_openai_compat(self):
        model_cfg = {"provider": "openai_compat"}
        self.assertEqual(rbe._select_inference_backend(model_cfg, "auto"), "openai_compat")
        self.assertEqual(rbe._select_inference_backend(model_cfg, "openai_compat"), "openai_compat")
        with self.assertRaises(ValueError):
            rbe._select_inference_backend(model_cfg, "local_hf")

    def test_detailed_prompt_contains_required_blocks(self):
        prompt = pb.build_text_prompt(
            form_url="https://example.test/form",
            remaining_answers=[{"question_id": "q_001", "label": "Name", "widget_type": "short_text", "value": "Olivia"}],
            page_text="Name Email Submit",
            last_result={"status": "failed", "error": "target_not_found", "remaining_answers": 1},
            prompt_profile="detailed_v1",
            visible_field_map=[{"question_id": "q_001", "label": "Name", "widget_type": "short_text", "expected_value": "Olivia"}],
            recent_history=[{"step_index": 0, "status": "failed", "error": "target_not_found"}],
            validation_feedback={"category": "target_not_found", "hint": "use allowed id"},
            fewshot_enabled=True,
            fewshot_count=3,
        )
        self.assertIn("Prompt profile: detailed_v1", prompt)
        self.assertIn("Output schema:", prompt)
        self.assertIn("Canonical few-shot examples:", prompt)
        self.assertIn("Recent step history:", prompt)
        self.assertIn("Validation feedback:", prompt)

    def test_runtime_safe_prompt_profile_marker(self):
        prompt = pb.build_text_prompt(
            form_url="https://example.test/form",
            remaining_answers=[{"question_id": "q_001", "label": "Name", "widget_type": "short_text", "value": "Olivia"}],
            page_text="Name Email Submit",
            last_result={"status": "failed", "error": "target_not_found", "remaining_answers": 1},
            prompt_profile="runtime_safe_v1",
            visible_field_map=[{"question_id": "q_001", "label": "Name", "widget_type": "short_text", "expected_value": "Olivia"}],
            recent_history=[
                {"step_index": 0, "status": "failed", "error": "target_not_found"},
                {"step_index": 1, "status": "failed", "error": "target_not_found"},
                {"step_index": 2, "status": "failed", "error": "target_not_found"},
            ],
            validation_feedback={"category": "target_not_found", "hint": "use allowed id"},
            fewshot_enabled=True,
            fewshot_count=3,
        )
        self.assertIn("Prompt profile: runtime_safe_v1", prompt)
        self.assertIn("Canonical few-shot examples:", prompt)

    def test_recent_history_window_truncation(self):
        rows = [
            {"step_index": 0, "status": "failed", "error": "x", "action": {"action": "wait"}},
            {"step_index": 1, "status": "filled", "error": None, "action": {"action": "type", "target": {"question_id": "q_001"}}},
            {"step_index": 2, "status": "filled", "error": None, "action": {"action": "submit"}},
        ]
        history = rbe._recent_history_from_steps(rows, 2)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["step_index"], 1)
        self.assertEqual(history[1]["step_index"], 2)

    def test_validation_feedback_categorization(self):
        payload = rbe._normalize_validation_feedback({"status": "failed", "error": "model_output_invalid: no_json_object_found"})
        self.assertEqual(payload["category"], "model_output_invalid")
        payload2 = rbe._normalize_validation_feedback({"status": "failed", "error": "target_not_found"})
        self.assertEqual(payload2["category"], "target_not_found")

    def test_make_run_label_uses_slurm_or_na(self):
        with patch.dict("os.environ", {"SLURM_JOB_ID": "2167542"}, clear=True):
            label = rbe._make_run_label(None)
        self.assertIn("_job2167542", label)

        with patch.dict("os.environ", {}, clear=True):
            label2 = rbe._make_run_label(None)
        self.assertIn("_jobna", label2)

    def test_retention_archives_older_trials(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_root = root / "data/model_baselines/exp"
            run_root = exp_root / "model_a" / "form_a" / "run_0001"
            for idx in range(6):
                trial_dir = run_root / f"trial_{idx:02d}"
                trial_dir.mkdir(parents=True, exist_ok=True)
                (trial_dir / "summary.json").write_text(
                    json.dumps({"run_started_utc": f"2026-03-24T10:00:0{idx}Z"}),
                    encoding="utf-8",
                )

            archived = rbe._apply_retention_window(
                experiment_root=exp_root,
                model_id="model_a",
                form_id="form_a",
                answer_run_id="run_0001",
                retention_window=5,
            )
            self.assertEqual(len(archived), 1)
            self.assertTrue((exp_root / "_archive" / "model_a" / "form_a" / "run_0001").exists())

    def test_main_writes_manifest_after_successful_action(self):
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._write_config(repo_root)

            def fake_session_factory(args, paths, trace):
                _ = args
                _ = trace
                return FakeSession(paths["artifact_dir"])

            adapter = SequenceAdapter(
                [
                    json.dumps({"action": "type", "target": {"question_id": "q_001"}, "value": "Olivia Brooks"}),
                    json.dumps({"action": "submit"}),
                ]
            )
            answers = [{"label": "Name", "widget_type": "short_text", "value": "Olivia Brooks"}]
            argv = [
                "--config",
                "configs/baselines/minimal_models.json",
                "--model-id",
                "text_qwen25_3b_instruct",
                "--model-kind",
                "text_llm",
                "--form-id",
                "event_rsvp",
                "--run-index",
                "1",
                "--dataset-root",
                "data/model_baselines",
                "--answers-root",
                "data/answers",
                "--trial-id",
                "trial_demo",
                "--execution-backend",
                "local",
            ]
            with patch.object(rbe, "ROOT_DIR", repo_root), patch.object(rbe, "_ensure_model_runtime_compat", return_value=None), patch.object(
                rbe, "_make_adapter", return_value=adapter
            ), patch.object(rbe, "_make_execution_session", side_effect=fake_session_factory), patch.object(
                rbe, "_load_run_answers", return_value=answers
            ), patch.object(
                rbe, "load_form_spec", return_value={"form_url": "https://example.test/form"}
            ), patch.object(
                rbe, "resolve_answers_path", return_value=repo_root / "data/answers/event_rsvp/runs.json"
            ), patch.object(
                rbe, "TraceLogger", return_value=FakeTrace()
            ), patch.object(
                rbe, "MCPTraceClient", side_effect=RuntimeError("disabled")
            ), patch.object(
                rbe, "_finalize_trial_video", return_value=None
            ):
                rc = rbe.main(argv)

            self.assertEqual(rc, 0)
            summary_path = (
                repo_root
                / "data/model_baselines/baseline_mcp_v1/text_qwen25_3b_instruct/event_rsvp/run_0001/trial_demo/summary.json"
            )
            summary = json.loads(summary_path.read_text())
            self.assertGreater(summary["action_count"], 0)
            manifest_lines = (repo_root / "data/model_baselines/baseline_mcp_v1/manifest.jsonl").read_text().splitlines()
            self.assertEqual(len(manifest_lines), 1)
            manifest = json.loads(manifest_lines[0])
            self.assertEqual(manifest["summary_path"], str(summary_path))
            self.assertEqual(manifest["track"], "mediated")
            self.assertEqual(manifest.get("prompt_profile"), "detailed_v1")
            self.assertEqual(manifest.get("context_package_version"), pb.CONTEXT_PACKAGE_VERSION)
            self.assertEqual(summary.get("prompt_profile"), "detailed_v1")
            self.assertEqual(summary.get("inference_backend"), "local_hf")
            self.assertEqual(summary.get("context_package_version"), pb.CONTEXT_PACKAGE_VERSION)
            self.assertTrue((summary_path.parent / "model_io.jsonl").exists())
            self.assertTrue((summary_path.parent / "step_inputs.jsonl").exists())
            step_rows = [json.loads(line) for line in (summary_path.parent / "step_inputs.jsonl").read_text().splitlines() if line.strip()]
            self.assertGreaterEqual(len(step_rows), 1)
            self.assertEqual(step_rows[0].get("prompt_profile"), "detailed_v1")
            self.assertEqual(step_rows[0].get("context_package_version"), pb.CONTEXT_PACKAGE_VERSION)
            self.assertIn("visible_field_map", step_rows[0])
            self.assertIn("recent_history", step_rows[0])
            self.assertIn("validation_feedback", step_rows[0])
            self.assertIn("fewshot_ids", step_rows[0])
            self.assertTrue(isinstance(step_rows[0].get("prompt_hash"), str) and len(step_rows[0].get("prompt_hash")) == 64)

    def test_disable_action_coercion_surfaces_widget_failure(self):
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._write_config(repo_root)

            def fake_session_factory(args, paths, trace):
                _ = args
                _ = trace
                return FakeSession(paths["artifact_dir"])

            adapter = SequenceAdapter([json.dumps({"action": "select_option", "target": {"question_id": "q_001"}})])
            answers = [{"label": "Name", "widget_type": "short_text", "value": "Olivia Brooks"}]
            argv = [
                "--config",
                "configs/baselines/minimal_models.json",
                "--model-id",
                "text_qwen25_3b_instruct",
                "--model-kind",
                "text_llm",
                "--form-id",
                "event_rsvp",
                "--run-index",
                "1",
                "--dataset-root",
                "data/model_baselines",
                "--answers-root",
                "data/answers",
                "--trial-id",
                "trial_no_coercion",
                "--execution-backend",
                "local",
                "--disable-action-coercion",
                "--max-steps",
                "1",
            ]
            with patch.object(rbe, "ROOT_DIR", repo_root), patch.object(rbe, "_ensure_model_runtime_compat", return_value=None), patch.object(
                rbe, "_make_adapter", return_value=adapter
            ), patch.object(rbe, "_make_execution_session", side_effect=fake_session_factory), patch.object(
                rbe, "_load_run_answers", return_value=answers
            ), patch.object(
                rbe, "load_form_spec", return_value={"form_url": "https://example.test/form"}
            ), patch.object(
                rbe, "resolve_answers_path", return_value=repo_root / "data/answers/event_rsvp/runs.json"
            ), patch.object(
                rbe, "TraceLogger", return_value=FakeTrace()
            ), patch.object(
                rbe, "MCPTraceClient", side_effect=RuntimeError("disabled")
            ), patch.object(
                rbe, "_finalize_trial_video", return_value=None
            ):
                rc = rbe.main(argv)

            self.assertEqual(rc, 1)
            annotations_path = (
                repo_root
                / "data/model_baselines/baseline_mcp_v1/text_qwen25_3b_instruct/event_rsvp/run_0001/trial_no_coercion/annotations.json"
            )
            annotations = json.loads(annotations_path.read_text())
            self.assertTrue(
                any(event.get("type") == "widget_interaction_failed" for event in annotations.get("failure_events", []))
            )
            self.assertIn("incompatible_action_for_widget", annotations["steps"][0].get("error") or "")

    def test_mediated_smoke_two_forms_writes_manifest_entries(self):
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._write_config(repo_root)

            def fake_session_factory(args, paths, trace):
                _ = args
                _ = trace
                return FakeSession(paths["artifact_dir"])

            def adapter_factory(*_args, **_kwargs):
                return SequenceAdapter(
                    [
                        json.dumps({"action": "type", "target": {"question_id": "q_001"}, "value": "Olivia Brooks"}),
                        json.dumps({"action": "submit"}),
                    ]
                )

            answers = [{"label": "Name", "widget_type": "short_text", "value": "Olivia Brooks"}]
            with patch.object(rbe, "ROOT_DIR", repo_root), patch.object(rbe, "_ensure_model_runtime_compat", return_value=None), patch.object(
                rbe, "_make_adapter", side_effect=adapter_factory
            ), patch.object(rbe, "_make_execution_session", side_effect=fake_session_factory), patch.object(
                rbe, "_load_run_answers", return_value=answers
            ), patch.object(
                rbe, "load_form_spec", return_value={"form_url": "https://example.test/form"}
            ), patch.object(
                rbe, "resolve_answers_path", return_value=repo_root / "data/answers/runs.json"
            ), patch.object(
                rbe, "TraceLogger", return_value=FakeTrace()
            ), patch.object(
                rbe, "MCPTraceClient", side_effect=RuntimeError("disabled")
            ), patch.object(
                rbe, "_finalize_trial_video", return_value=None
            ):
                rc1 = rbe.main(
                    [
                        "--config",
                        "configs/baselines/minimal_models.json",
                        "--model-id",
                        "text_qwen25_3b_instruct",
                        "--model-kind",
                        "text_llm",
                        "--form-id",
                        "conf_interest",
                        "--run-index",
                        "1",
                        "--dataset-root",
                        "data/model_baselines",
                        "--answers-root",
                        "data/answers",
                        "--trial-id",
                        "trial_form1",
                        "--execution-backend",
                        "local",
                        "--experiment-id",
                        "pilot_mediated_test",
                    ]
                )
                rc2 = rbe.main(
                    [
                        "--config",
                        "configs/baselines/minimal_models.json",
                        "--model-id",
                        "text_qwen25_3b_instruct",
                        "--model-kind",
                        "text_llm",
                        "--form-id",
                        "event_rsvp",
                        "--run-index",
                        "1",
                        "--dataset-root",
                        "data/model_baselines",
                        "--answers-root",
                        "data/answers",
                        "--trial-id",
                        "trial_form2",
                        "--execution-backend",
                        "local",
                        "--experiment-id",
                        "pilot_mediated_test",
                    ]
                )

            self.assertEqual(rc1, 0)
            self.assertEqual(rc2, 0)
            manifest_path = repo_root / "data/model_baselines/pilot_mediated_test/manifest.jsonl"
            rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["form_id"] for row in rows}, {"conf_interest", "event_rsvp"})

    def test_fallback_metadata_in_summary_and_manifest(self):
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            config_path = repo_root / "configs/baselines/minimal_models.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "vlm_fallback_model",
                                "kind": "text_llm",
                                "provider": "local_hf",
                                "hf_repo": "dummy",
                                "track": "mediated",
                                "requires_gpu": False,
                                "is_fallback": True,
                                "fallback_for": "vlm_primary_model",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            def fake_session_factory(args, paths, trace):
                _ = args
                _ = trace
                return FakeSession(paths["artifact_dir"])

            adapter = SequenceAdapter(
                [
                    json.dumps({"action": "type", "target": {"question_id": "q_001"}, "value": "Olivia Brooks"}),
                    json.dumps({"action": "submit"}),
                ]
            )
            answers = [{"label": "Name", "widget_type": "short_text", "value": "Olivia Brooks"}]

            argv = [
                "--config",
                "configs/baselines/minimal_models.json",
                "--model-id",
                "vlm_fallback_model",
                "--model-kind",
                "text_llm",
                "--form-id",
                "event_rsvp",
                "--run-index",
                "1",
                "--dataset-root",
                "data/model_baselines",
                "--answers-root",
                "data/answers",
                "--trial-id",
                "trial_fallback",
                "--execution-backend",
                "local",
            ]
            with patch.object(rbe, "ROOT_DIR", repo_root), patch.object(rbe, "_ensure_model_runtime_compat", return_value=None), patch.object(
                rbe, "_make_adapter", return_value=adapter
            ), patch.object(rbe, "_make_execution_session", side_effect=fake_session_factory), patch.object(
                rbe, "_load_run_answers", return_value=answers
            ), patch.object(
                rbe, "load_form_spec", return_value={"form_url": "https://example.test/form"}
            ), patch.object(
                rbe, "resolve_answers_path", return_value=repo_root / "data/answers/event_rsvp/runs.json"
            ), patch.object(
                rbe, "TraceLogger", return_value=FakeTrace()
            ), patch.object(
                rbe, "MCPTraceClient", side_effect=RuntimeError("disabled")
            ), patch.object(
                rbe, "_finalize_trial_video", return_value=None
            ):
                rc = rbe.main(argv)

            self.assertEqual(rc, 0)
            summary_path = (
                repo_root
                / "data/model_baselines/baseline_mcp_v1/vlm_fallback_model/event_rsvp/run_0001/trial_fallback/summary.json"
            )
            summary = json.loads(summary_path.read_text())
            self.assertTrue(summary["is_fallback_model"])
            self.assertEqual(summary["fallback_for"], "vlm_primary_model")
            manifest_line = (repo_root / "data/model_baselines/baseline_mcp_v1/manifest.jsonl").read_text().splitlines()[-1]
            manifest = json.loads(manifest_line)
            self.assertTrue(manifest["is_fallback_model"])
            self.assertEqual(manifest["fallback_for"], "vlm_primary_model")


if __name__ == "__main__":
    import unittest

    unittest.main()
