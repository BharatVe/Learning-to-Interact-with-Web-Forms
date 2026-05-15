from pathlib import Path
from unittest import TestCase

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baselines import run_opencua_direct_eval as opencua_eval


class OpenCUAParserTests(TestCase):
    def test_click_then_write_becomes_type_text(self):
        action, debug = opencua_eval._parse_opencua_action(
            "```python\npyautogui.click(x=960, y=324)\npyautogui.write('Alice')\n```",
            viewport_width=1920,
            viewport_height=1080,
            coordinate_type="qwen25",
        )
        self.assertEqual(action["action"], "type_text")
        self.assertEqual(action["value"], "Alice")
        self.assertIn("target", action)
        self.assertGreaterEqual(action["target"]["x"], 0)
        self.assertLessEqual(action["target"]["x"], 999)
        self.assertEqual(debug["parser"], "click_then_write")
        self.assertEqual(debug["coordinate_transform"]["coordinate_space"], "qwen25_smart_resize_absolute")

    def test_click_then_keyword_write_becomes_type_text(self):
        action, debug = opencua_eval._parse_opencua_action(
            "pyautogui.click(x=960, y=324)\npyautogui.write(message='Sam Bauer')",
            viewport_width=1920,
            viewport_height=1080,
            coordinate_type="qwen25",
        )
        self.assertEqual(action["action"], "type_text")
        self.assertEqual(action["value"], "Sam Bauer")
        self.assertEqual(action["clear_before_typing"], False)
        self.assertIn("target", action)
        self.assertEqual(debug["parser"], "click_then_write")

    def test_keyword_write_only_becomes_type_text(self):
        for output in ("pyautogui.write(message='Sam Bauer')", 'pyautogui.typewrite(text="Sam Bauer")'):
            with self.subTest(output=output):
                action, debug = opencua_eval._parse_opencua_action(
                    output,
                    viewport_width=1440,
                    viewport_height=900,
                    coordinate_type="qwen25",
                )
                self.assertEqual(action, {"action": "type_text", "value": "Sam Bauer", "clear_before_typing": False})
                self.assertEqual(debug["parser"], "write_only")

    def test_hotkey_is_normalized(self):
        action, debug = opencua_eval._parse_opencua_action(
            "pyautogui.hotkey('ctrl', 'a')",
            viewport_width=1440,
            viewport_height=900,
            coordinate_type="qwen25",
        )
        self.assertEqual(action, {"action": "press_key", "value": "Control+A"})
        self.assertEqual(debug["parser"], "hotkey")

    def test_submit_and_done_literals(self):
        submit_action, _ = opencua_eval._parse_opencua_action("SUBMIT", 1440, 900, "qwen25")
        done_action, _ = opencua_eval._parse_opencua_action("DONE", 1440, 900, "qwen25")
        self.assertEqual(submit_action, {"action": "submit"})
        self.assertEqual(done_action, {"action": "done"})


class OpenCUACoordinateTests(TestCase):
    def test_qwen25_coordinate_transform_is_in_bounds(self):
        abs_x, abs_y, meta = opencua_eval._qwen25_smart_resize_to_abs(960, 324, 1920, 1080)
        self.assertGreaterEqual(abs_x, 0)
        self.assertLessEqual(abs_x, 1920)
        self.assertGreaterEqual(abs_y, 0)
        self.assertLessEqual(abs_y, 1080)
        self.assertEqual(meta["coordinate_space"], "qwen25_smart_resize_absolute")
        self.assertGreater(meta["resized_width"], 0)
        self.assertGreater(meta["resized_height"], 0)

    def test_norm_conversion_bounds(self):
        coords = opencua_eval._to_norm(1920, 1080, 1920, 1080)
        self.assertEqual(coords["x"], 999)
        self.assertEqual(coords["y"], 999)


class OpenCUAPromptContractTests(TestCase):
    def test_prompt_is_screenshot_native_by_default(self):
        prompt = opencua_eval._build_goal_prompt(
            form_url="https://example.test/form",
            remaining_answers=[{"label": "Name", "value": "Alice"}],
            last_result={},
            interaction_map=[{"label": "Name", "ref": "e1"}],
            page_text="Name",
            observation_mode="vision_coords",
            recent_history=[],
        )
        self.assertNotIn("Interaction map", prompt)
        self.assertNotIn('"ref": "e1"', prompt)
        self.assertIn("pyautogui.click", prompt)
        self.assertIn("Before submitting, double-check the visible form state", prompt)
        self.assertIn("click the visible Submit button or output SUBMIT", prompt)
        self.assertIn("SUBMIT means you intend to submit", prompt)

    def test_prompt_can_include_symbolic_support_for_ablation(self):
        prompt = opencua_eval._build_goal_prompt(
            form_url="https://example.test/form",
            remaining_answers=[{"label": "Name", "value": "Alice"}],
            last_result={},
            interaction_map=[{"label": "Name", "ref": "e1"}],
            page_text="Name",
            observation_mode="vision_coords",
            recent_history=[],
            include_symbolic_support=True,
        )
        self.assertIn("Interaction map", prompt)
        self.assertIn('"ref": "e1"', prompt)
