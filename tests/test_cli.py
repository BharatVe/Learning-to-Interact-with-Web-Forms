"""End-to-end CLI tests: `python -m formbench <command>` as users (and the Makefile) call it.

Everything runs against temp DATASET_ROOT / LOGS_DIR / REPORTS_DIR, with LOCAL=1 and no
GPUs, models or network; commands that would start models use --dry-run.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

REPO_ROOT = Path(__file__).resolve().parents[1]


class CLITestCase(TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.env = {
            k: v for k, v in os.environ.items()
            if not k.startswith(("OPENAI_", "GEMINI_", "SLURM_")) and k not in {"MODULES"}
        }
        self.env.update({
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "LOCAL": "1",
            "DATASET_ROOT": str(self.tmp / "ds"),
            "LOGS_DIR": str(self.tmp / "logs"),
            "REPORTS_DIR": str(self.tmp / "reports"),
            "GEMINI_API_KEY_FILE": str(self.tmp / "no_key"),
            "PYTHON_BIN": sys.executable,
        })

    def tearDown(self):
        self._tmp.cleanup()

    def fb(self, *args, expect=0, env=None):
        proc = subprocess.run(
            [sys.executable, "-m", "formbench", *args], cwd=str(REPO_ROOT), capture_output=True, text=True,
            env={**self.env, **(env or {})}, timeout=600,
        )
        out = proc.stdout + proc.stderr
        if expect is not None:
            self.assertEqual(proc.returncode, expect, f"formbench {' '.join(args)} -> {proc.returncode}\n{out[-3000:]}")
        return out

    def write_trial(self, exp="exp_cli", model="computer_use_gemini_35_flash_lowcost", form="conf_interest", run=1, trial="trial_1", **fields):
        d = self.tmp / "ds" / exp / model / form / f"run_{run:04d}" / trial
        d.mkdir(parents=True, exist_ok=True)
        summary = {"experiment_id": exp, "model_id": model, "form_id": form, "answer_run_id": f"run_{run:04d}", "trial_id": trial,
                   "success": True, "submit_success": False, "question_total": 7, "scored_correctness": 5, "verified_correctness": 5,
                   "stop_reason": "filled_without_submit", "duration_s": 12.5, "run_completed_utc": "2026-07-01T00:00:00Z"}
        summary.update(fields)
        (d / "summary.json").write_text(json.dumps(summary))
        (d / "step_inputs.jsonl").write_text(json.dumps({"step_index": 0, "prompt": "fill"}) + "\n")
        (d / "model_io.jsonl").write_text(json.dumps({"phase": "step", "step_index": 0, "raw_model_output": "click", "parsed_action": {"action": "click"}}) + "\n")
        return d


class HelpAndListingTests(CLITestCase):
    def test_help_for_every_command(self):
        self.assertIn("<command>", self.fb("--help"))
        for command in (["setup"], ["doctor"], ["models", "list"], ["models", "check"], ["models", "serve"], ["models", "install"],
                        ["data", "build"], ["data", "check"], ["data", "relpaths"], ["forms", "serve"], ["ideal"], ["eval"],
                        ["matrix"], ["submit"], ["experiments"], ["report"], ["inspect"]):
            self.assertIn("usage:", self.fb(*command, "--help"))

    def test_models_list_hides_legacy_by_default(self):
        out = self.fb("models", "list")
        self.assertIn("computer_use_gemini_35_flash_lowcost", out)
        self.assertNotIn("text_qwen25_7b_instruct", out)
        self.assertIn("legacy model(s) hidden", out)
        self.assertIn("text_qwen25_7b_instruct", self.fb("models", "list", "--all"))

    def test_experiments_lists_all_manifests(self):
        out = self.fb("experiments")
        for name in ("smoke", "fill_only_done_30", "track_baseline_pilot", "localforms_opencua_direct_mcp", "qwen3vl_interface_comparison"):
            self.assertIn(name, out)


class ModelCheckTests(CLITestCase):
    def test_api_model_passes_with_key_and_fails_without(self):
        self.assertIn("api key", self.fb("models", "check", "computer_use_gemini_35_flash_lowcost", env={"GEMINI_API_KEY": "k"}))
        out = self.fb("models", "check", "computer_use_gemini_35_flash_lowcost", expect=1)
        self.assertIn("FAIL", out)
        self.assertIn(".secrets", out)  # fix hint

    def test_unknown_model(self):
        self.assertIn("unknown model id", self.fb("models", "check", "gpt-17", expect=1))

    def test_env_override_is_reported(self):
        out = self.fb("models", "check", "text_qwen3_30b_a3b_instruct_2507", env={"OPENAI_MODEL": "something-else"})
        self.assertIn("env override", out)
        self.assertIn("OPENAI_MODEL=something-else", out)

    def test_install_dry_run(self):
        out = self.fb("models", "install", "text_qwen25_7b_instruct", "--dry-run")
        self.assertIn("text_qwen25_7b_instruct", out)


class EvalMatrixSubmitTests(CLITestCase):
    def test_eval_dry_run_builds_runner_command(self):
        out = self.fb("eval", "computer_use_gemini_35_flash_lowcost", "bug_report", "3", "--set", "max_steps=4", "--dry-run", "--skip-checks")
        self.assertIn("src/baselines/run_gemini_low_cost_eval.py", out)
        self.assertIn("--form-id bug_report --run-index 3", out)
        self.assertIn("--max-steps 4", out)

    def test_matrix_dry_run_for_served_model_shows_server(self):
        out = self.fb("matrix", "fill_only_done_30", "--cohort", "qwen", "--models", "vlm_qwen3_vl_30b_a3b_instruct", "--forms", "conf_interest",
                      "--dry-run", "--skip-checks", "--experiment-suffix", "_cli")
        self.assertIn("# server: ", out)
        self.assertIn("vllm.entrypoints.openai.api_server", out)
        self.assertIn("qwen_direct_mcp_fill_only_done_30_seed20260709_r2_step32_cli", out)
        self.assertEqual(out.count("src/baselines/run_qwen_direct_mcp_eval.py"), 1)

    def test_matrix_dry_run_preflight_reports_but_does_not_block(self):
        out = self.fb("matrix", "smoke", "--cohort", "gemini", "--dry-run")
        self.assertIn("api key", out)
        self.assertIn("a real run would stop here", out)

    def test_real_run_blocked_by_failing_preflight(self):
        out = self.fb("matrix", "smoke", "--cohort", "gemini", expect=1)
        self.assertIn("model preflight failed", out)
        self.assertFalse((self.tmp / "ds").exists())

    def test_selection_errors(self):
        self.assertIn("give an experiment manifest or --models", self.fb("matrix", expect=1))
        self.assertIn("experiment manifest not found", self.fb("matrix", "nope", expect=1))
        self.assertIn("unknown form id", self.fb("matrix", "--models", "computer_use_gemini_35_flash_lowcost", "--forms", "zzz", "--dry-run", expect=1))
        self.assertIn("would merge several cohorts", self.fb("matrix", "fill_only_done_30", "--experiment-id", "x", "--dry-run", expect=1))
        self.assertIn("--set expects key=value", self.fb("eval", "computer_use_gemini_35_flash_lowcost", "conf_interest", "--set", "oops", "--dry-run", "--skip-checks", expect=1))
        self.assertIn("does not support the localforms platform",
                      self.fb("eval", "computer_use_gemini_35_flash_lowcost", "conf_interest", "--platform", "localforms", "--dry-run", "--skip-checks", expect=1))

    def test_submit_dry_run_writes_job_scripts(self):
        out = self.fb("submit", "track_baseline_pilot", "--split", "cohort", "--chain", "afterok", "--dry-run")
        self.assertIn("#SBATCH --gres=gpu:2", out)
        self.assertIn("#SBATCH --gres=gpu:4", out)
        self.assertIn("matrix track_baseline_pilot --cohort family_b_native", out)
        self.assertEqual(len(list((self.tmp / "logs" / "slurm" / "jobs").glob("*.sbatch"))), 2)

    def test_submit_refuses_without_slurm(self):
        self.assertIn("sbatch not found", self.fb("submit", "smoke", expect=1))


class DataIdealTests(CLITestCase):
    def test_data_check_passes_on_committed_dataset(self):
        out = self.fb("data", "check")
        self.assertIn("answer-set validation passed", out)
        self.assertIn("integrity checks passed", out)

    def test_ideal_status_and_dry_run(self):
        out = self.fb("ideal", "--status", "--forms", "conf_interest")
        self.assertIn("conf_interest", out)
        out = self.fb("ideal", "--forms", "conf_interest", "--runs", "1-2", "--platform", "localforms", "--dry-run")
        self.assertIn("--form-id lf_conf_interest", out)
        self.assertIn("--start-index 1 --num-runs 2", out)
        self.assertIn("/forms/lf_conf_interest", out)

    def test_relpaths_is_dry_by_default(self):
        self.assertIn("would rewrite", self.fb("data", "relpaths"))


class ReportInspectTests(CLITestCase):
    def test_report_on_selected_experiments_writes_overview_and_tracker(self):
        self.write_trial()
        self.write_trial(form="event_rsvp", trial="trial_2", success=False, scored_correctness=1)
        out = self.fb("report", "--experiments", "exp_cli")
        self.assertIn("experiment_overview.csv", out)
        overview = (self.tmp / "reports" / "experiment_overview.csv").read_text()
        self.assertIn("exp_cli,computer_use_gemini_35_flash_lowcost", overview)
        self.assertIn(",0.4286,", overview)  # (5 + 1) / 14
        self.assertTrue((self.tmp / "reports" / "metrics.csv").is_file())

    def test_full_report_on_empty_dataset(self):
        (self.tmp / "ds").mkdir()
        self.fb("report")
        for name in ("thesis_model_summary.csv", "experiment_overview.csv", "latest_analysis.md", "metrics.csv"):
            self.assertTrue((self.tmp / "reports" / name).is_file(), name)

    def test_studies_listing_and_unknown_study(self):
        out = self.fb("report", "--list-studies")
        for name in ("core_report", "form_length_position", "opencua_ruler", "interaction_failure"):
            self.assertIn(name, out)
        self.assertIn("unknown study", self.fb("report", "--study", "nope", expect=1))
        self.assertIn("has no check mode", self.fb("report", "--study", "core_report", "--check", expect=1))

    def test_inspect_trial_summary_steps_media(self):
        trial = self.write_trial()
        out = self.fb("inspect", str(trial), "--steps", "all", "--media")
        self.assertIn("filled_without_submit", out)
        self.assertIn("STEP 0", out)
        self.assertIn('"raw_model_output": "click"', out)
        self.assertIn("ideal reference run", out)
        out = self.fb("inspect", "exp_cli/computer_use_gemini_35_flash_lowcost/conf_interest/run_0001")  # newest trial under a run dir
        self.assertIn("trial_1", out)
        self.assertIn("trial not found", self.fb("inspect", "no/such/trial", expect=1))


class DoctorTests(CLITestCase):
    def test_doctor_runs_and_reports_dataset(self):
        out = self.fb("doctor", expect=None)
        self.assertIn("== doctor", out)
        self.assertIn("50 form specs, 50 with answer sets", out)
        self.assertIn("registry", out)
