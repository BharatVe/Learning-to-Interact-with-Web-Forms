from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import sys

from PIL import Image

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

    def test_ruler_overlay_preserves_dimensions_coordinates_and_raw_image(self):
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "raw.png"
            output = Path(temp_dir) / "model_input.png"
            Image.new("RGB", (1440, 900), color=(240, 240, 240)).save(source)
            raw_before = source.read_bytes()
            transform_before = opencua_eval._qwen25_smart_resize_to_abs(700, 420, 1440, 900)

            metadata = opencua_eval._add_pixel_ruler_overlay(str(source), output)

            self.assertEqual(source.read_bytes(), raw_before)
            self.assertTrue(output.exists())
            self.assertNotEqual(output.read_bytes(), raw_before)
            with Image.open(source) as raw_image, Image.open(output) as model_image:
                self.assertEqual(model_image.size, raw_image.size)
            transform_after = opencua_eval._qwen25_smart_resize_to_abs(700, 420, 1440, 900)
            self.assertEqual(transform_after, transform_before)
            self.assertEqual(metadata["width"], 1440)
            self.assertEqual(metadata["height"], 900)
            self.assertTrue(metadata["coordinate_dimensions_unchanged"])


class OpenCUALoopDetectionTests(TestCase):
    def test_recent_same_action_signature_count_counts_trailing_repeats(self):
        repeated = {"action": {"action": "click_mouse", "target": {"x": 330, "y": 987}}}
        history = [
            {"action": {"action": "type_text", "target": {"x": 343, "y": 353}, "value": "Taylor"}},
            repeated,
            repeated,
            repeated,
            repeated,
        ]
        self.assertEqual(opencua_eval._recent_same_action_signature_count(history), 4)

    def test_recent_same_action_signature_count_stops_at_different_action(self):
        history = [
            {"action": {"action": "click_mouse", "target": {"x": 330, "y": 987}}},
            {"action": {"action": "click_mouse", "target": {"x": 345, "y": 885}}},
            {"action": {"action": "click_mouse", "target": {"x": 330, "y": 987}}},
        ]
        self.assertEqual(opencua_eval._recent_same_action_signature_count(history), 1)

    def test_verified_fill_complete_requires_every_field(self):
        self.assertTrue(opencua_eval._verified_fill_complete([{"verified_correct": True}, {"verified_correct": True}]))
        self.assertFalse(opencua_eval._verified_fill_complete([{"verified_correct": True}, {"verified_correct": False}]))
        self.assertFalse(opencua_eval._verified_fill_complete([]))


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
        self.assertIn("Return only the next GUI action", prompt)
        self.assertIn("Do not output a script", prompt)
        self.assertIn("top-to-bottom order", prompt)
        self.assertIn("If an action does not change the visible state", prompt)
        self.assertIn("scroll instead of clicking an unrelated area", prompt)
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

    def test_fill_only_prompt_forbids_submission_and_uses_done(self):
        prompt = opencua_eval._build_goal_prompt(
            form_url="https://example.test/form",
            remaining_answers=[{"label": "Name", "value": "Alice"}],
            last_result={},
            interaction_map=[],
            page_text="Name",
            observation_mode="vision_coords",
            recent_history=[],
            task_mode="fill_only_done",
        )
        self.assertIn("fill the form without submitting it", prompt)
        self.assertIn("never click Submit and never output SUBMIT", prompt)
        self.assertIn("When every target answer is correctly filled, output DONE", prompt)
        self.assertNotIn("- SUBMIT\n", prompt)
        self.assertNotIn("click the visible Submit button or output SUBMIT", prompt)

    def test_ruler_prompt_is_present_only_when_enabled(self):
        common = dict(
            form_url="https://example.test/form",
            remaining_answers=[{"label": "Name", "widget_type": "short_text", "value": "Alice"}],
            last_result={},
            interaction_map=[],
            page_text="Name",
            observation_mode="vision_coords",
            recent_history=[],
            task_mode="fill_only_done",
        )
        without_ruler = opencua_eval._build_goal_prompt(**common)
        with_ruler = opencua_eval._build_goal_prompt(**common, ruler_overlay=True)
        self.assertNotIn("labeled pixel rulers", without_ruler)
        self.assertIn("labeled pixel rulers", with_ruler)
        self.assertIn("absolute click coordinates", with_ruler)

    def test_input_contract_discloses_widget_types_and_ruler(self):
        contract = opencua_eval._build_input_contract(
            include_symbolic_support=False,
            ruler_overlay=True,
        )
        self.assertTrue(contract["provides_widget_types"])
        self.assertTrue(contract["provides_visual_ruler"])
        self.assertFalse(contract["provides_interaction_map"])

    def test_formfactory_style_cli_flags(self):
        args = opencua_eval._parse_args(
            [
                "--form-id",
                "conf_interest",
                "--run-index",
                "2",
                "--fill-only-done",
                "--formfactory-style",
                "--ruler-overlay",
            ]
        )
        self.assertTrue(args.fill_only_done)
        self.assertTrue(args.formfactory_style)
        self.assertTrue(args.ruler_overlay)
