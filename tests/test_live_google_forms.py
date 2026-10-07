"""Opt-in live check of widget reading/scoring against the real Google Forms (no submission).

Google Forms markup changes over time; a silent change already caused dropdowns to be
scored 0% (see docs/eval_results/interaction_failure_analysis/DROPDOWN_FAILURE_ANALYSIS.md).
Run before a large evaluation campaign:

    FORMBENCH_LIVE=1 scripts/env.sh .venv/bin/python -m unittest tests.test_live_google_forms -v

It opens a form, fills ONE page-1 dropdown with the scripted MCP engine, verifies it the
way model trials are scored, and closes the browser without submitting.
"""

import json
import os
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


@unittest.skipUnless(os.environ.get("FORMBENCH_LIVE") == "1", "live browser test; set FORMBENCH_LIVE=1")
class LiveDropdownScoringTests(unittest.TestCase):
    FORM_ID = "bug_report"

    def test_dropdown_fill_verify_and_score(self):
        from baselines import run_baseline_eval as rbe
        from engine import runner
        from engine.browser_language import force_english_google_forms_url
        from engine.mcp_browser_engine import MCPBrowserEngine
        from engine.mcp_trace_client import MCPClient
        from engine.trace_logger import TraceLogger

        spec = json.loads((REPO_ROOT / "src/forms" / self.FORM_ID / "spec.json").read_text())
        first_section = min(int(q.get("section_order", 1) or 1) for q in spec["questions"])
        question = next(q for q in spec["questions"] if q.get("q_type") == "DROPDOWN" and int(q.get("section_order", 1) or 1) == first_section)
        answers = json.loads((REPO_ROOT / "data/answers" / self.FORM_ID / "runs.json").read_text())["runs"][0]["answers"]
        entry = next(a for a in answers if a["label"] == question["q_title"])
        options = [o.strip() for o in str(question.get("options", "")).split(";") if o.strip()]
        wrong = next(o for o in options if o != entry["value"])

        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            args = runner.parse_args(["--form-id", self.FORM_ID, "--headless"])
            client = MCPClient(command=runner._default_browser_mcp_command(args, run_dir), timeout_ms=120000,
                               required_tools=["browser_navigate", "browser_run_code", "browser_close"], env=runner._node_env())
            engine = MCPBrowserEngine(client, TraceLogger(run_dir / "trace.jsonl", time.time(), validate_mcp_actions=False),
                                      run_dir, 15000, 0, 100, False)
            try:
                engine.navigate(force_english_google_forms_url(spec["form_url"]))
                before = engine.verify_entry(entry, 0)
                _, fill_error = engine.fill_step(entry, 1)
                after = engine.verify_entry(entry, 2)
            finally:
                engine.close()
                client.close()

        self.assertFalse(rbe._value_matches(entry["value"], before["actual_value"]), before)  # placeholder is not an answer
        self.assertIsNone(fill_error)
        self.assertEqual(after["actual_value"], entry["value"])  # exact selected label, not the option list
        self.assertTrue(after["verified"] and rbe._value_matches(entry["value"], after["actual_value"]))
        self.assertFalse(rbe._value_matches(wrong, after["actual_value"]))


if __name__ == "__main__":
    unittest.main()
