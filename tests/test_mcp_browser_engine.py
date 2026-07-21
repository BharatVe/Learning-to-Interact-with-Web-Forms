import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from engine.mcp_browser_engine import MCPBrowserEngine


class FakeMCPClient:
    def __init__(self):
        self.calls = []

    def call_tool(self, name, args):
        self.calls.append((name, dict(args)))
        if name == "browser_take_screenshot":
            Path(args["filename"]).write_bytes(b"png")
        return {"ok": True}


class FakeTrace:
    def __init__(self):
        self.events = []

    def log_event(self, name, args, step_ref=None, ok=True, error=None, extra=None):
        self.events.append({"name": name, "args": args, "step_ref": step_ref, "ok": ok, "error": error, "extra": extra})

    def now(self):
        return 0.0


class MCPBrowserEngineTests(TestCase):
    def test_capture_screenshot_uses_absolute_filename_and_creates_parent(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mcp = FakeMCPClient()
            trace = FakeTrace()
            engine = MCPBrowserEngine(
                mcp_client=mcp,
                trace=trace,
                observations_dir=root / "observations",
                timeout_ms=15000,
                type_delay_ms=0,
                action_delay_ms=0,
                take_screenshots=True,
            )
            path = root / "nested" / "terminal" / "error.png"
            captured = engine.capture_screenshot(path, step_ref=7)
            self.assertEqual(captured, str(path.resolve()))
            self.assertTrue(path.exists())
            self.assertEqual(mcp.calls[-1][0], "browser_take_screenshot")
            self.assertTrue(Path(mcp.calls[-1][1]["filename"]).is_absolute())
            self.assertEqual(Path(mcp.calls[-1][1]["filename"]), path.resolve())

    def test_dropdown_verifier_reads_selected_option_not_trigger_text(self):
        engine = MCPBrowserEngine.__new__(MCPBrowserEngine)
        engine.timeout_ms = 15000
        code = engine._build_verify_step_code(
            {"label": "Issue category", "widget_type": "dropdown", "value": "Network"}
        )
        self.assertIn('el.getAttribute("aria-selected")', code)
        self.assertIn('selectedRoleLabels(container, "option")', code)
        self.assertNotIn('trigger.innerText({ timeout: 1000 })', code)
