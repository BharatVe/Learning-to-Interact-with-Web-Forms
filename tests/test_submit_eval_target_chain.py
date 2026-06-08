import json
import tempfile
from pathlib import Path
from unittest import TestCase

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import submit_eval_target_chain as chain


class SubmitEvalTargetChainTests(TestCase):
    def test_target_run_indexes_derives_six_runs_for_300_trials_and_50_forms(self):
        forms = [f"form_{idx:02d}" for idx in range(50)]
        self.assertEqual(chain._target_run_indexes(300, forms), [1, 2, 3, 4, 5, 6])

    def test_observed_pairs_count_unique_model_form_runs_across_experiments(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            for experiment in ["exp_a", "exp_b"]:
                trial = dataset / experiment / "text_qwen3_30b_a3b_instruct_2507" / "conf_interest" / "run_0001" / f"trial_{experiment}"
                trial.mkdir(parents=True)
                (trial / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            original = chain.DATASET_ROOT
            chain.DATASET_ROOT = dataset
            try:
                observed = chain._observed_pairs()
            finally:
                chain.DATASET_ROOT = original
        self.assertEqual(observed["text_qwen3_30b_a3b_instruct_2507"], {("conf_interest", "run_0001")})

    def test_missing_forms_are_model_specific(self):
        forms = ["conf_interest", "event_rsvp"]
        observed = {
            "text_qwen3_30b_a3b_instruct_2507": {("conf_interest", "run_0001")},
            "vlm_qwen3_vl_30b_a3b_instruct": {("event_rsvp", "run_0001")},
            "computer_use_opencua_32b_direct_mcp": {("conf_interest", "run_0001")},
        }
        self.assertEqual(chain._missing_forms_for_qwen_run(1, forms, observed), ["conf_interest", "event_rsvp"])
        self.assertEqual(chain._missing_forms_for_opencua_direct_mcp_run(1, forms, observed), ["event_rsvp"])

