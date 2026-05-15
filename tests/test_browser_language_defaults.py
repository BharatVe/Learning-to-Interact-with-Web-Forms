import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baselines import run_baseline_eval as rbe
from engine.browser_language import DEFAULT_BROWSER_LOCALE, force_english_google_forms_url
from engine import runner


class BrowserLanguageDefaultTests(TestCase):
    def test_google_forms_url_gets_english_language_param(self):
        url = "https://docs.google.com/forms/d/e/abc/viewform?usp=sf_link"
        self.assertEqual(
            force_english_google_forms_url(url),
            "https://docs.google.com/forms/d/e/abc/viewform?usp=sf_link&hl=en",
        )

    def test_non_google_forms_url_is_unchanged(self):
        url = "https://example.test/form?hl=de"
        self.assertEqual(force_english_google_forms_url(url), url)

    def test_baseline_default_mcp_command_writes_english_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = rbe._default_browser_mcp_command(1280, 720, Path(tmp), True, 120000)
            config_path = Path(command[command.index("--config") + 1])
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["browser"]["contextOptions"]["locale"], DEFAULT_BROWSER_LOCALE)
            self.assertEqual(payload["browser"]["contextOptions"]["extraHTTPHeaders"]["Accept-Language"], "en-US,en;q=0.9")
            self.assertIn("--lang=en-US", payload["browser"]["launchOptions"]["args"])

    def test_dataset_default_mcp_command_writes_english_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                browser_mcp_timeout_ms=120000,
                viewport_width=1280,
                viewport_height=720,
                headless=True,
            )
            command = runner._default_browser_mcp_command(args, Path(tmp))
            config_path = Path(command[command.index("--config") + 1])
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["browser"]["contextOptions"]["locale"], DEFAULT_BROWSER_LOCALE)
