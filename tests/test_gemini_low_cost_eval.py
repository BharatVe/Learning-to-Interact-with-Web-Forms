import unittest
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

    def test_plain_text_done_is_terminal_action(self):
        payload = {"candidates": [{"content": [{"type": "text", "text": "done"}]}]}

        action, meta = gemini_low_cost_eval._extract_action(payload)

        self.assertEqual(action, {"action": "done"})
        self.assertEqual(meta["source"], "text")

    def test_non_done_plain_text_does_not_mask_missing_action(self):
        payload = {"candidates": [{"content": [{"type": "text", "text": "I am finished."}]}]}

        with self.assertRaisesRegex(ValueError, "gemini_response_missing_computer_action"):
            gemini_low_cost_eval._extract_action(payload)


if __name__ == "__main__":
    unittest.main()
