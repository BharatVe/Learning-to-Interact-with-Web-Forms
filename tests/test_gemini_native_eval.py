import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baselines import run_gemini_native_computer_use_eval as gemini_eval


def _load_smoke_module():
    script_path = REPO_ROOT / "scripts" / "eval_model_baseline_smoke.py"
    spec = importlib.util.spec_from_file_location("eval_model_baseline_smoke", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed_to_load_eval_model_baseline_smoke")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class GeminiNativeEvalTests(TestCase):
    def test_native_function_calls_are_mapped(self):
        action, safety_events, source = gemini_eval._resolve_action_and_safety(
            [
                {
                    "name": "type_text_at",
                    "args": {"x": 371, "y": 470, "text": "Alice", "press_enter": True, "clear_before_typing": True},
                }
            ],
            "",
        )
        self.assertEqual(action["action"], "type_text")
        self.assertEqual(action["target"]["x"], 371)
        self.assertEqual(action["target"]["y"], 470)
        self.assertEqual(action["value"], "Alice")
        self.assertTrue(action["press_enter"])
        self.assertTrue(action["clear_before_typing"])
        self.assertEqual(safety_events, [])
        self.assertEqual(source["source"], "function_call")

    def test_safety_decision_inside_native_function_call_is_logged(self):
        action, safety_events, _source = gemini_eval._resolve_action_and_safety(
            [
                {
                    "name": "click_at",
                    "args": {
                        "x": 60,
                        "y": 100,
                        "safety_decision": {
                            "decision": "require_confirmation",
                            "explanation": "Cookie banner interaction",
                        },
                    },
                }
            ],
            "",
        )
        self.assertEqual(action["action"], "click_mouse")
        self.assertEqual(len(safety_events), 1)
        self.assertEqual(safety_events[0]["decision"], "auto_allow")

    def test_require_confirmation_function_call_auto_allow(self):
        action, safety_events, source = gemini_eval._resolve_action_and_safety(
            [
                {
                    "name": "require_confirmation",
                    "args": {
                        "reason": "click may navigate away",
                        "action": {"action": "click_mouse", "target": {"x": 400, "y": 220}},
                    },
                }
            ],
            "",
        )
        self.assertEqual(action["action"], "click_mouse")
        self.assertEqual(action["target"]["x"], 400)
        self.assertEqual(action["target"]["y"], 220)
        self.assertEqual(len(safety_events), 1)
        self.assertEqual(safety_events[0]["decision"], "auto_allow")
        self.assertEqual(source["source"], "function_call")

    def test_text_json_require_confirmation_is_logged(self):
        action, safety_events, source = gemini_eval._resolve_action_and_safety(
            [],
            '{"action":"type_text","value":"Alice","require_confirmation":true}',
        )
        normalized, _warnings = gemini_eval.validate_low_level_action(action)
        self.assertEqual(normalized["action"], "type_text")
        self.assertEqual(normalized["value"], "Alice")
        self.assertEqual(len(safety_events), 1)
        self.assertEqual(safety_events[0]["decision"], "auto_allow")
        self.assertEqual(source["source"], "text_json")


class GeminiSmokeGateTests(TestCase):
    def test_gemini_gate_fails_without_key(self):
        smoke = _load_smoke_module()
        with patch.object(smoke.shutil, "which", return_value="/usr/bin/true"), patch.dict(os.environ, {}, clear=True):
            ok, detail = smoke.eval_gemini_native({"gemini_model": "gemini-2.5-computer-use-preview-10-2025"})
        self.assertFalse(ok)
        self.assertFalse(detail["gemini_api_key_present"])

    def test_gemini_gate_passes_with_key_and_sdk(self):
        smoke = _load_smoke_module()
        fake_google = types.ModuleType("google")
        fake_genai = types.ModuleType("google.genai")
        fake_google.genai = fake_genai

        with patch.object(smoke.shutil, "which", return_value="/usr/bin/true"), patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-2.5-computer-use-preview-10-2025"},
            clear=True,
        ), patch.dict(
            sys.modules,
            {"google": fake_google, "google.genai": fake_genai},
            clear=False,
        ):
            ok, detail = smoke.eval_gemini_native({"gemini_model": "gemini-2.5-computer-use-preview-10-2025"})

        self.assertTrue(ok)
        self.assertTrue(detail["gemini_api_key_present"])
        self.assertTrue(detail["google_genai_import_ok"])
        self.assertEqual(detail["gemini_model"], "gemini-2.5-computer-use-preview-10-2025")
        self.assertTrue(detail["gemini_model_valid_for_native_computer_use"])

    def test_gemini_gate_fails_for_wrong_model(self):
        smoke = _load_smoke_module()
        fake_google = types.ModuleType("google")
        fake_genai = types.ModuleType("google.genai")
        fake_google.genai = fake_genai

        with patch.object(smoke.shutil, "which", return_value="/usr/bin/true"), patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-3-flash-preview"},
            clear=True,
        ), patch.dict(
            sys.modules,
            {"google": fake_google, "google.genai": fake_genai},
            clear=False,
        ):
            ok, detail = smoke.eval_gemini_native({"gemini_model": "gemini-3-flash-preview"})

        self.assertFalse(ok)
        self.assertFalse(detail["gemini_model_valid_for_native_computer_use"])
