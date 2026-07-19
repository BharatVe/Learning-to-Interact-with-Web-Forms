import unittest
from unittest import mock
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baselines import run_gemini_low_cost_eval as gemini_low_cost_eval


class GeminiLowCostEvalTests(unittest.TestCase):
    def test_transient_provider_errors_are_classified_as_capacity(self):
        cases = [
            'gemini_interactions_http_error:500:{"error":{"message":"high demand"}}',
            "gemini_interactions_http_error:503:temporarily overloaded",
            "gemini_interactions_http_error:504:deadline exceeded",
            "gemini_interactions_request_failed:The read operation timed out",
            "gemini_interactions_http_error:429:rate limit",
        ]
        for message in cases:
            with self.subTest(message=message):
                self.assertTrue(gemini_low_cost_eval._is_transient_provider_error(message))
                self.assertEqual(gemini_low_cost_eval._provider_failure_category(message), "provider_capacity_error")

    def test_non_transient_errors_remain_model_inference_failures(self):
        message = "unsupported_gemini_action:drag_mouse"
        self.assertFalse(gemini_low_cost_eval._is_transient_provider_error(message))
        self.assertEqual(gemini_low_cost_eval._provider_failure_category(message), "model_inference_failed")

    def test_malformed_tool_call_is_retryable_but_not_capacity(self):
        message = "gemini_interactions_http_error:400:malformed_tool_call"

        self.assertFalse(gemini_low_cost_eval._is_transient_provider_error(message))
        self.assertTrue(gemini_low_cost_eval._is_retryable_provider_error(message))
        self.assertEqual(gemini_low_cost_eval._provider_failure_category(message), "provider_tool_call_error")

    def test_plain_text_done_is_terminal_action(self):
        payload = {"candidates": [{"content": [{"type": "text", "text": "done"}]}]}

        action, meta = gemini_low_cost_eval._extract_action(payload)

        self.assertEqual(action, {"action": "done"})
        self.assertEqual(meta["source"], "text")

    def test_non_done_plain_text_does_not_mask_missing_action(self):
        payload = {"candidates": [{"content": [{"type": "text", "text": "I am finished."}]}]}

        with self.assertRaisesRegex(ValueError, "gemini_response_missing_computer_action"):
            gemini_low_cost_eval._extract_action(payload)

    def test_native_scroll_magnitude_is_preserved(self):
        action = gemini_low_cost_eval._normalize_action(
            "scroll",
            {"x": 500, "y": 450, "direction": "up", "magnitude_in_pixels": 300},
        )

        self.assertEqual(action["delta"], -300)
        self.assertEqual(action["target"], {"x": 500, "y": 450})

    def test_native_type_preserves_press_enter(self):
        action = gemini_low_cost_eval._normalize_action(
            "type",
            {"text": "Example", "press_enter": True},
        )

        self.assertTrue(action["press_enter"])

    def test_tool_preserves_native_computer_use_capabilities(self):
        tool = gemini_low_cost_eval.GeminiLowCostAdapter._computer_use_tool()

        self.assertEqual(tool, {"type": "computer_use", "environment": "browser"})

    def test_initial_request_preserves_historical_payload_shape(self):
        captured = []

        def fake_post(*, url, payload, api_key, timeout_s):
            captured.append(payload)
            return {"id": "interaction_1", "steps": []}

        with mock.patch.object(gemini_low_cost_eval, "_read_api_key", return_value=("secret", "test")), mock.patch.object(
            gemini_low_cost_eval, "_image_data_for_path", return_value="image-data"
        ), mock.patch.object(gemini_low_cost_eval, "_http_post_json", side_effect=fake_post):
            adapter = gemini_low_cost_eval.GeminiLowCostAdapter(
                {"gemini_model": "gemini-3.5-flash"},
                api_timeout_s=10,
            )
            adapter.infer("fill the form", "step.png", 128)

        self.assertEqual(set(captured[0]), {"model", "input", "tools"})
        self.assertEqual(captured[0]["tools"], [{"type": "computer_use", "environment": "browser"}])
        self.assertNotIn("generation_config", captured[0])

    def test_malformed_tool_call_is_retried(self):
        responses = [
            RuntimeError("gemini_interactions_http_error:400:malformed_tool_call"),
            {"id": "interaction_1", "steps": []},
        ]

        with mock.patch.object(gemini_low_cost_eval, "_read_api_key", return_value=("secret", "test")), mock.patch.object(
            gemini_low_cost_eval, "_http_post_json", side_effect=responses
        ), mock.patch.object(gemini_low_cost_eval.time, "sleep"):
            adapter = gemini_low_cost_eval.GeminiLowCostAdapter(
                {"gemini_model": "gemini-3.5-flash", "gemini_max_infer_retries": 1},
                api_timeout_s=10,
            )
            _, meta = adapter.infer("fill the form", None, 128)

        self.assertEqual(meta["retry_count"], 1)
        self.assertIn("malformed_tool_call", meta["retry_errors"][0])

    def test_stateful_request_uses_previous_interaction_and_function_result(self):
        captured = []

        def fake_post(*, url, payload, api_key, timeout_s):
            captured.append(payload)
            return {"id": "interaction_2", "steps": []}

        with mock.patch.object(gemini_low_cost_eval, "_read_api_key", return_value=("secret", "test")), mock.patch.object(
            gemini_low_cost_eval, "_http_post_json", side_effect=fake_post
        ):
            adapter = gemini_low_cost_eval.GeminiLowCostAdapter(
                {"gemini_model": "gemini-3.5-flash"},
                api_timeout_s=10,
            )
            function_result = {
                "type": "function_result",
                "name": "click",
                "call_id": "call_1",
                "result": [{"type": "text", "text": "ok"}],
            }
            _, meta = adapter.infer(
                "unused continuation prompt",
                None,
                128,
                previous_interaction_id="interaction_1",
                function_result=function_result,
            )

        self.assertEqual(captured[0]["previous_interaction_id"], "interaction_1")
        self.assertEqual(captured[0]["input"], [function_result])
        self.assertTrue(meta["stateful_continuation"])
        self.assertEqual(meta["interaction_id"], "interaction_2")

    def test_function_result_contains_execution_remaining_answers_and_image(self):
        with mock.patch.object(gemini_low_cost_eval, "_image_data_for_path", return_value="image-data"):
            result = gemini_low_cost_eval._build_function_result(
                call_id="call_1",
                name="click",
                screenshot_path="step.png",
                execution={"status": "clicked"},
                remaining_answers=[{"label": "Name", "value": "Ada"}],
            )

        self.assertEqual(result["call_id"], "call_1")
        self.assertEqual(result["result"][1]["type"], "image")
        self.assertIn("remaining_answers", result["result"][0]["text"])


if __name__ == "__main__":
    unittest.main()
