from pathlib import Path
from unittest import TestCase

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baselines import run_opencua_direct_mcp_eval as opencua_direct_mcp_eval
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

    def test_model_visible_tools_exclude_internal_playwright_tools(self):
        exposed = [
            {"name": "browser_snapshot", "description": "Snapshot", "inputSchema": {}},
            {"name": "browser_click", "description": "Click", "inputSchema": {}},
            {"name": "browser_type", "description": "Type", "inputSchema": {}},
            {"name": "browser_fill_form", "description": "Fill", "inputSchema": {}},
            {"name": "browser_check", "description": "Check", "inputSchema": {}},
            {"name": "browser_uncheck", "description": "Uncheck", "inputSchema": {}},
            {"name": "browser_select_option", "description": "Select", "inputSchema": {}},
            {"name": "browser_wait_for", "description": "Wait", "inputSchema": {}},
            {"name": "browser_press_key", "description": "Press", "inputSchema": {}},
            {"name": "browser_navigate", "description": "Navigate", "inputSchema": {}},
            {"name": "browser_run_code", "description": "Run arbitrary page code", "inputSchema": {}},
            {"name": "browser_close", "description": "Close", "inputSchema": {}},
            {"name": "browser_take_screenshot", "description": "Screenshot", "inputSchema": {}},
        ]
        tools = qwen_direct_mcp_eval._mcp_tools_to_openai_tools(exposed)
        visible_names = {tool["function"]["name"] for tool in tools}
        self.assertEqual(visible_names, qwen_direct_mcp_eval.DIRECT_MCP_MODEL_TOOLS)
        self.assertNotIn("browser_run_code", visible_names)
        self.assertNotIn("browser_navigate", visible_names)
        self.assertNotIn("browser_close", visible_names)
        self.assertNotIn("browser_take_screenshot", visible_names)

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
        self.assertEqual(parsed["tool_call_transport"], "native_tool_calls")
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
        self.assertEqual(parsed["tool_call_transport"], "text_tool_call_fallback")
        self.assertEqual(parsed["tool_calls"][0]["name"], "browser_click")
        self.assertEqual(parsed["tool_calls"][0]["arguments"], {"x": 12, "y": 30})
        self.assertEqual(parsed["tool_calls"][0]["source"], "assistant_text_tool_call")

    def test_parse_openai_response_recovers_relaxed_opencua_text_tool_call(self):
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            "<tool_call>\n"
                            '  "name": "browser_type",\n'
                            '  "arguments": {\n'
                            '    "text": "Morgan Bauer",\n'
                            '    "ref": "e36"\n'
                            "  }\n"
                            "</tool_call>"
                        ),
                    },
                }
            ]
        }
        parsed = qwen_direct_mcp_eval._parse_openai_response(payload)
        self.assertEqual(parsed["tool_call_transport"], "text_tool_call_fallback")
        self.assertEqual(parsed["tool_calls"][0]["name"], "browser_type")
        self.assertEqual(parsed["tool_calls"][0]["arguments"], {"text": "Morgan Bauer", "ref": "e36"})
        self.assertEqual(parsed["tool_calls"][0]["source"], "assistant_text_tool_call")

    def test_relaxed_opencua_tool_call_normalizes_to_documented_arguments(self):
        parsed = qwen_direct_mcp_eval._parse_raw_mcp_tool_calls(
            "<tool_call>\n"
            '  "name": "browser_type",\n'
            '  "arguments": {"text": "Morgan Bauer", "ref": "e36", "x": 100, "y": 200}\n'
            "</tool_call>"
        )
        self.assertEqual(len(parsed), 1)
        normalized = qwen_direct_mcp_eval._normalize_tool_arguments(parsed[0]["name"], parsed[0]["arguments"])
        self.assertEqual(normalized, {"ref": "e36", "text": "Morgan Bauer"})

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
        self.assertEqual(parsed["tool_call_transport"], "none")

    def test_parse_openai_response_ignores_malformed_relaxed_text_tool_call(self):
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            "<tool_call>\n"
                            '  "name": "browser_type",\n'
                            '  "arguments": {"text": "Morgan Bauer", "ref": "e36"\n'
                            "</tool_call>"
                        ),
                    },
                }
            ]
        }
        parsed = qwen_direct_mcp_eval._parse_openai_response(payload)
        self.assertEqual(parsed["tool_calls"], [])
        self.assertEqual(parsed["tool_call_transport"], "none")

    def test_call_model_can_disable_native_openai_tools(self):
        captured = {}

        def fake_post(url, headers, payload, timeout_s):
            captured["payload"] = payload
            return {"choices": [{"message": {"content": "STOP"}}]}

        original = qwen_direct_mcp_eval._http_post_json
        qwen_direct_mcp_eval._http_post_json = fake_post
        try:
            parsed = qwen_direct_mcp_eval._call_model(
                base_url="http://localhost:8000/v1",
                api_key="EMPTY",
                model="opencua-32b",
                messages=[{"role": "user", "content": "use tools"}],
                tools=[{"type": "function", "function": {"name": "browser_click", "parameters": {}}}],
                native_tool_calls=False,
                max_new_tokens=16,
                timeout_s=1,
            )
        finally:
            qwen_direct_mcp_eval._http_post_json = original
        self.assertEqual(parsed["text"], "STOP")
        self.assertNotIn("tools", captured["payload"])
        self.assertNotIn("tool_choice", captured["payload"])

    def test_call_model_sends_native_tools_when_enabled(self):
        captured = {}

        def fake_post(url, headers, payload, timeout_s):
            captured["payload"] = payload
            return {"choices": [{"message": {"content": "STOP"}}]}

        original = qwen_direct_mcp_eval._http_post_json
        qwen_direct_mcp_eval._http_post_json = fake_post
        try:
            qwen_direct_mcp_eval._call_model(
                base_url="http://localhost:8000/v1",
                api_key="EMPTY",
                model="qwen",
                messages=[{"role": "user", "content": "use tools"}],
                tools=[{"type": "function", "function": {"name": "browser_click", "parameters": {}}}],
                native_tool_calls=True,
                max_new_tokens=16,
                timeout_s=1,
            )
        finally:
            qwen_direct_mcp_eval._http_post_json = original
        self.assertIn("tools", captured["payload"])
        self.assertEqual(captured["payload"]["tool_choice"], "auto")

    def test_strict_direct_mcp_does_not_parse_pyautogui_as_tool(self):
        payload = {"choices": [{"message": {"content": "pyautogui.click(x=10, y=20)"}}]}
        parsed = qwen_direct_mcp_eval._parse_openai_response(payload)
        self.assertEqual(parsed["tool_calls"], [])
        self.assertEqual(parsed["tool_call_transport"], "none")

    def test_computer_use_agent_messages_include_screenshot(self):
        image_path = REPO_ROOT / "tests" / "fixtures_opencua_direct_mcp.png"
        image_path.write_bytes(b"png")
        try:
            messages = qwen_direct_mcp_eval._build_messages(
                model_kind="computer_use_agent",
                observation_text="snapshot",
                screenshot_path=image_path,
                history=[],
            )
        finally:
            image_path.unlink(missing_ok=True)
        self.assertIsInstance(messages[-1]["content"], list)
        self.assertEqual(messages[-1]["content"][0]["type"], "text")
        self.assertEqual(messages[-1]["content"][1]["type"], "image_url")

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
        self.assertIn("Do not output pyautogui", prompt)
        self.assertIn("textual fallback call", prompt)

    def test_fill_only_done_prompt_replaces_submit_terminal_condition(self):
        prompt = qwen_direct_mcp_eval._observation_prompt(
            form_url="https://example.test/form",
            remaining_answers=[{"question_id": "q1", "label": "Name", "value": "Alice"}],
            page_text="Name\nSubmit",
            url="https://example.test/form",
            step_idx=0,
            fill_only_done=True,
        )
        self.assertIn("Fill-only terminal condition", prompt)
        self.assertIn("never click Submit", prompt)
        self.assertIn("reply with plain text DONE", prompt)
        self.assertNotIn("observed a form submission confirmation page", prompt)

    def test_opencua_direct_mcp_wrapper_sets_computer_use_kind(self):
        captured = {}

        def fake_main(argv):
            captured["argv"] = argv
            return 0

        original = opencua_direct_mcp_eval.run_qwen_direct_mcp_eval.main
        opencua_direct_mcp_eval.run_qwen_direct_mcp_eval.main = fake_main
        try:
            status = opencua_direct_mcp_eval.main(["--model-id", "computer_use_opencua_32b_direct_mcp"])
        finally:
            opencua_direct_mcp_eval.run_qwen_direct_mcp_eval.main = original
        self.assertEqual(status, 0)
        self.assertEqual(captured["argv"][:2], ["--model-kind", "computer_use_agent"])

    def test_parse_args_accepts_fill_only_done(self):
        args = qwen_direct_mcp_eval._parse_args(
            [
                "--model-id",
                "text_qwen3_30b_a3b_instruct_2507",
                "--model-kind",
                "text_llm",
                "--form-id",
                "bug_report",
                "--run-index",
                "2",
                "--fill-only-done",
            ]
        )
        self.assertTrue(args.fill_only_done)

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

    def test_observation_prompt_omits_browser_check_when_not_visible(self):
        prompt = qwen_direct_mcp_eval._observation_prompt(
            form_url="https://example.test/form",
            remaining_answers=[{"question_id": "q1", "label": "Choice", "value": "A"}],
            page_text="Choice",
            url="https://example.test/form",
            step_idx=0,
            model_visible_tool_names=["browser_snapshot", "browser_click", "browser_type"],
            control_contract=[
                {
                    "ref": "e1",
                    "label": "A",
                    "tag": "div",
                    "role": "radio",
                    "valid_mcp_tools": ["browser_click"],
                }
            ],
        )
        self.assertNotIn("browser_check", prompt)
        self.assertIn("For checkboxes and radio buttons use browser_click", prompt)

    def test_filter_control_contract_tools_intersects_visible_tools(self):
        filtered = qwen_direct_mcp_eval._filter_control_contract_tools(
            [
                {
                    "ref": "e1",
                    "label": "A",
                    "valid_mcp_tools": ["browser_click", "browser_check"],
                }
            ],
            {"browser_click"},
        )
        self.assertEqual(filtered[0]["valid_mcp_tools"], ["browser_click"])

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

    def test_compact_history_keeps_complete_recent_tool_turns(self):
        history = [
            {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "old"},
            {"role": "assistant", "tool_calls": [{"id": "call_2"}]},
            {"role": "tool", "tool_call_id": "call_2", "content": "recent"},
            {"role": "assistant", "tool_calls": [{"id": "call_3"}]},
            {"role": "tool", "tool_call_id": "call_3", "content": "latest"},
        ]
        compacted = qwen_direct_mcp_eval._compact_history(history, 2)
        self.assertEqual(compacted[0]["role"], "user")
        self.assertIn("1 earlier browser-action turn", compacted[0]["content"])
        self.assertEqual(compacted[1:], history[2:])

    def test_compact_history_zero_preserves_full_history(self):
        history = [{"role": "assistant", "content": "DONE"}]
        self.assertEqual(qwen_direct_mcp_eval._compact_history(history, 0), history)

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
