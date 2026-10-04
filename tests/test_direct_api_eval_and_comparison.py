import json
import os
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
from baselines import run_direct_api_eval as dra


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




class DirectProviderTests(TestCase):
    def test_provider_auto_prefers_openai(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "x", "ANTHROPIC_API_KEY": "y"}, clear=True):
            self.assertEqual(dra._select_provider("auto"), "openai")

    def test_provider_auto_falls_back_to_anthropic(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "y"}, clear=True):
            self.assertEqual(dra._select_provider("auto"), "anthropic")

    def test_provider_auto_errors_without_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                dra._select_provider("auto")
        self.assertIn("provider_auto_detect_failed", str(ctx.exception))


class DirectRunnerAndComparisonTests(TestCase):
    def _write_config(self, repo_root: Path) -> None:
        config_path = repo_root / "configs/baselines/minimal_models.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "id": "computer_use_mcp_api",
                            "kind": "computer_use_agent",
                            "provider": "api_over_mcp",
                            "track": "direct_api_tool_use",
                            "requires_gpu": False,
                            "openai_model": "gpt-4.1-mini",
                            "anthropic_model": "claude-3-5-sonnet-latest",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    def test_direct_runner_smoke_with_mocked_api_infer(self):
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._write_config(repo_root)

            answers = [{"label": "Name", "widget_type": "short_text", "value": "Olivia Brooks"}]

            infer_calls = {"count": 0}

            def infer_stub(_adapter, prompt, max_new_tokens):
                _ = prompt
                _ = max_new_tokens
                infer_calls["count"] += 1
                if infer_calls["count"] == 1:
                    return (
                        json.dumps(
                            {
                                "action": "type",
                                "target": {"question_id": "q_001"},
                                "value": "Olivia Brooks",
                            }
                        ),
                        {"provider": "openai", "provider_model": "gpt-4.1-mini"},
                    )
                return (json.dumps({"action": "submit"}), {"provider": "openai", "provider_model": "gpt-4.1-mini"})

            def fake_session_factory(args, paths, trace):
                _ = args
                _ = trace
                return FakeSession(paths["artifact_dir"])

            argv = [
                "--config",
                "configs/baselines/minimal_models.json",
                "--model-id",
                "computer_use_mcp_api",
                "--form-id",
                "event_rsvp",
                "--run-index",
                "1",
                "--dataset-root",
                "data/model_baselines",
                "--answers-root",
                "data/answers",
                "--trial-id",
                "trial_direct",
                "--execution-backend",
                "local",
                "--experiment-id",
                "pilot_direct_test",
                "--provider",
                "auto",
                "--disable-action-coercion",
            ]

            with patch.dict(os.environ, {"OPENAI_API_KEY": "x"}, clear=True), patch.object(
                dra, "ROOT_DIR", repo_root
            ), patch.object(
                rbe, "ROOT_DIR", repo_root
            ), patch.object(
                dra, "load_form_spec", return_value={"form_url": "https://example.test/form"}
            ), patch.object(
                dra, "resolve_answers_path", return_value=repo_root / "data/answers/event_rsvp/runs.json"
            ), patch.object(
                dra, "_load_run_answers", return_value=answers
            ), patch.object(
                dra.rbe, "_make_execution_session", side_effect=fake_session_factory
            ), patch.object(
                dra, "TraceLogger", return_value=FakeTrace()
            ), patch.object(
                dra.rbe, "_finalize_trial_video", return_value=None
            ), patch.object(
                dra.DirectAPIAdapter, "infer", autospec=True, side_effect=infer_stub
            ):
                rc = dra.main(argv)

            self.assertEqual(rc, 0)
            summary_path = (
                repo_root
                / "data/model_baselines/pilot_direct_test/computer_use_mcp_api/event_rsvp/run_0001/trial_direct/summary.json"
            )
            summary = json.loads(summary_path.read_text())
            self.assertTrue(summary["submit_success"])
            self.assertEqual(summary["track"], "direct_api_tool_use")
            self.assertEqual(summary["api_provider"], "openai")
            self.assertIn("attempted_correctness", summary)
            self.assertIn("verified_correctness", summary)
            self.assertTrue((summary_path.parent / "step_inputs.jsonl").exists())
            self.assertTrue(isinstance(summary.get("run_label"), str) and "_job" in summary.get("run_label"))



if __name__ == "__main__":
    import unittest

    unittest.main()
