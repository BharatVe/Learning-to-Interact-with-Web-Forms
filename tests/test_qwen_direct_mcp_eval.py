from pathlib import Path
from unittest import TestCase

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baselines import run_qwen_direct_mcp_eval as qwen_direct_mcp_eval


class QwenDirectMCPEvalTests(TestCase):
    def test_mcp_tools_convert_to_openai_tools(self):
        tools = qwen_direct_mcp_eval._mcp_tools_to_openai_tools(
            [
                {
                    "name": "browser_click",
                    "description": "Click the page",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                        "required": ["x", "y"],
                    },
                },
                {
                    "name": "browser_run_code",
                    "description": "Run arbitrary page code",
                    "inputSchema": {"type": "object", "properties": {"code": {"type": "string"}}},
                },
            ]
        )
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "browser_click")
        self.assertEqual(tools[0]["function"]["parameters"]["required"], ["ref"])
        self.assertEqual(set(tools[0]["function"]["parameters"]["properties"]), {"ref"})

    def test_select_option_is_hidden_without_real_select_control(self):
        tools = qwen_direct_mcp_eval._mcp_tools_to_openai_tools(
            [
                {"name": "browser_click", "description": "Click", "inputSchema": {}},
                {"name": "browser_select_option", "description": "Select", "inputSchema": {}},
            ]
        )
        filtered = qwen_direct_mcp_eval._filter_tools_for_visible_controls(
            tools,
            [{"ref": "e111", "tag": "input", "role": "combobox", "valid_mcp_tools": ["browser_type", "browser_click"]}],
        )
        self.assertEqual([tool["function"]["name"] for tool in filtered], ["browser_click"])

    def test_select_option_is_visible_for_real_select_control(self):
        tools = qwen_direct_mcp_eval._mcp_tools_to_openai_tools(
            [
                {"name": "browser_click", "description": "Click", "inputSchema": {}},
                {"name": "browser_select_option", "description": "Select", "inputSchema": {}},
            ]
        )
        filtered = qwen_direct_mcp_eval._filter_tools_for_visible_controls(
            tools,
            [{"ref": "e7", "tag": "select", "role": "", "valid_mcp_tools": ["browser_select_option"]}],
        )
        self.assertEqual([tool["function"]["name"] for tool in filtered], ["browser_click", "browser_select_option"])

    def test_parse_openai_response_extracts_tool_calls(self):
        payload = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "Using tools.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "browser_click",
                                    "arguments": "{\"x\": 12, \"y\": 30}",
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        parsed = qwen_direct_mcp_eval._parse_openai_response(payload)
        self.assertEqual(parsed["text"], "Using tools.")
        self.assertEqual(parsed["finish_reason"], "tool_calls")
        self.assertEqual(parsed["tool_calls"][0]["name"], "browser_click")
        self.assertEqual(parsed["tool_calls"][0]["arguments"], {"x": 12, "y": 30})

    def test_parse_openai_response_recovers_text_tool_call(self):
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            "<tool_call>\n"
                            "{\"name\":\"browser_click\",\"arguments\":{\"x\":12,\"y\":30}}\n"
                            "</tool_call>"
                        ),
                    },
                }
            ]
        }
        parsed = qwen_direct_mcp_eval._parse_openai_response(payload)
        self.assertEqual(parsed["tool_calls"][0]["name"], "browser_click")
        self.assertEqual(parsed["tool_calls"][0]["arguments"], {"x": 12, "y": 30})
        self.assertEqual(parsed["tool_calls"][0]["source"], "assistant_text_tool_call")

    def test_parse_openai_response_ignores_malformed_text_tool_call(self):
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "<tool_call>{\"name\":\"browser_click\",\"arguments\":{\"x\":12</tool_call>"},
                }
            ]
        }
        parsed = qwen_direct_mcp_eval._parse_openai_response(payload)
        self.assertEqual(parsed["tool_calls"], [])

    def test_done_text_accepts_done_and_stop(self):
        self.assertTrue(qwen_direct_mcp_eval._done_text("DONE finished"))
        self.assertTrue(qwen_direct_mcp_eval._done_text("stop no further progress"))
        self.assertTrue(qwen_direct_mcp_eval._done_text("The form is complete. DONE"))
        self.assertFalse(qwen_direct_mcp_eval._done_text("browser_click"))

    def test_observation_prompt_includes_accessibility_snapshot(self):
        prompt = qwen_direct_mcp_eval._observation_prompt(
            form_url="https://example.test/form",
            remaining_answers=[{"question_id": "q1", "label": "Name", "value": "Alice"}],
            page_text="Name\nSubmit",
            url="https://example.test/form",
            step_idx=0,
            accessibility_snapshot="- textbox \"Name\" [ref=e1]",
        )
        self.assertIn("Latest Playwright MCP accessibility snapshot", prompt)
        self.assertIn("[ref=e1]", prompt)
        self.assertIn("use refs from the snapshot", prompt)
        self.assertIn("DONE means you have observed a form submission confirmation page", prompt)
        self.assertIn("Before submitting, double-check the visible form state", prompt)
        self.assertIn("Do not use DONE to say the form is ready to submit", prompt)

    def test_observation_prompt_includes_control_contract(self):
        prompt = qwen_direct_mcp_eval._observation_prompt(
            form_url="https://example.test/form",
            remaining_answers=[{"question_id": "q1", "label": "Time", "value": "10"}],
            page_text="Time",
            url="https://example.test/form",
            step_idx=0,
            control_contract=[
                {
                    "ref": "e198",
                    "label": "Hour",
                    "tag": "input",
                    "type": "number",
                    "role": "combobox",
                    "valid_mcp_tools": ["browser_type", "browser_click"],
                }
            ],
        )
        self.assertIn("Visible Google Forms control/tool compatibility metadata", prompt)
        self.assertIn('"ref": "e198"', prompt)
        self.assertIn('"browser_type"', prompt)
        self.assertIn("does not choose answers", prompt)

    def test_assistant_history_omits_null_tool_calls(self):
        message = qwen_direct_mcp_eval._assistant_history_message({"text": "DONE", "tool_calls": []})
        self.assertEqual(message, {"role": "assistant", "content": "DONE"})
        self.assertNotIn("tool_calls", message)

    def test_assistant_history_includes_real_tool_calls(self):
        message = qwen_direct_mcp_eval._assistant_history_message(
            {
                "text": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "browser_click",
                        "arguments": {"x": 12, "y": 30},
                    }
                ],
            }
        )
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["content"], "")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "browser_click")
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"], "{\"x\": 12, \"y\": 30}")

    def test_normalize_tool_arguments_removes_non_documented_click_fields(self):
        normalized = qwen_direct_mcp_eval._normalize_tool_arguments(
            "browser_click",
            {"element": "Submit button", "ref": "e12", "x": 100, "y": 200},
        )
        self.assertEqual(normalized, {"ref": "e12"})

    def test_normalize_fill_form_keeps_only_ref_and_value(self):
        normalized = qwen_direct_mcp_eval._normalize_tool_arguments(
            "browser_fill_form",
            {"fields": [{"name": "Email", "type": "textbox", "ref": "e3", "value": "a@example.test"}]},
        )
        self.assertEqual(normalized, {"fields": [{"ref": "e3", "value": "a@example.test"}]})

    def test_select_option_guard_rejects_non_select_ref(self):
        class FakeEngine:
            def _run_code(self, code, purpose, step_ref):
                self.last = (code, purpose, step_ref)
                return {"found": True, "ref": "e198", "tag": "input", "type": "number", "role": "combobox"}

        error = qwen_direct_mcp_eval._validate_tool_call_for_execution(
            engine=FakeEngine(),
            name="browser_select_option",
            arguments={"ref": "e198", "values": ["10"]},
            step_idx=4,
        )
        self.assertEqual(error["error"], "incompatible_mcp_tool_for_ref")
        self.assertEqual(error["required_tag"], "select")
        self.assertEqual(error["actual_tag"], "input")

    def test_select_option_guard_allows_select_ref(self):
        class FakeEngine:
            def _run_code(self, code, purpose, step_ref):
                return {"found": True, "ref": "e7", "tag": "select", "type": "", "role": ""}

        error = qwen_direct_mcp_eval._validate_tool_call_for_execution(
            engine=FakeEngine(),
            name="browser_select_option",
            arguments={"ref": "e7", "values": ["United States"]},
            step_idx=0,
        )
        self.assertIsNone(error)

    def test_valid_tools_for_google_forms_controls(self):
        self.assertEqual(
            qwen_direct_mcp_eval._valid_tools_for_control({"tag": "input", "type": "number", "role": "combobox"}),
            ["browser_type", "browser_click"],
        )
        self.assertEqual(
            qwen_direct_mcp_eval._valid_tools_for_control({"tag": "div", "role": "radio"}),
            ["browser_click", "browser_check"],
        )
        self.assertEqual(
            qwen_direct_mcp_eval._valid_tools_for_control({"tag": "select", "role": ""}),
            ["browser_select_option"],
        )

    def test_control_looks_like_submit(self):
        self.assertTrue(
            qwen_direct_mcp_eval._control_looks_like_submit(
                {"ref": "e1", "label": "Submit", "role": "button"},
                {"ref": "e1"},
            )
        )
        self.assertTrue(
            qwen_direct_mcp_eval._control_looks_like_submit(
                {"ref": "e2", "label": "Senden", "role": "button"},
                {"ref": "e2"},
            )
        )
        self.assertFalse(
            qwen_direct_mcp_eval._control_looks_like_submit(
                {"ref": "e3", "label": "Full Name", "role": "textbox"},
                {"ref": "e3"},
            )
        )

    def test_control_by_ref(self):
        controls = [{"ref": "e1", "label": "Name"}, {"ref": "e2", "label": "Submit"}]
        self.assertEqual(qwen_direct_mcp_eval._control_by_ref(controls, "e2"), {"ref": "e2", "label": "Submit"})
        self.assertIsNone(qwen_direct_mcp_eval._control_by_ref(controls, "e9"))
