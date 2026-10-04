import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baselines.model_registry import validate_config, validate_models  # noqa: E402
from formbench import common, matrix, protocols, serve, slurm  # noqa: E402
from formbench.settings import load_settings, parse_env_file  # noqa: E402


def _settings(**env):
    base = {"LOCAL": "1"}
    base.update(env)
    return load_settings(environ=base)


class RegistryTests(TestCase):
    def test_repo_registry_is_valid(self):
        models, errors, warnings = validate_config(REPO_ROOT / "configs" / "models.json")
        self.assertGreater(len(models), 5)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_schema_catches_typos_and_bad_references(self):
        models = [
            {"id": "a", "kind": "vlm", "provider": "openai_compat", "track": "direct_mcp_tool_use", "served_model_name": "x",
             "serve": {"port": "auto", "tensor_parallel": 4}, "resources": {"gpus": 2}, "opnai_model": "typo"},
            {"id": "b", "kind": "vlm", "provider": "local_hf", "track": "mediated", "hf_repo": "r", "is_fallback": True, "fallback_for": "missing"},
            {"id": "c", "kind": "robot", "provider": "gemini_low_cost", "track": "proprietary_computer_use_low_cost"},
        ]
        errors, warnings = validate_models(models)
        joined = "\n".join(errors)
        self.assertIn("a: serve.tensor_parallel=4 exceeds resources.gpus=2", joined)
        self.assertIn("b: fallback_for references unknown model id 'missing'", joined)
        self.assertIn("c: kind must be one of", joined)
        self.assertIn("c: gemini_low_cost models need gemini_model", joined)
        self.assertTrue(any("opnai_model" in w for w in warnings))


class ProtocolCommandTests(TestCase):
    """Generated runner commands must match what the replaced shell matrices passed."""

    def setUp(self):
        self.settings = _settings()
        self.registry = {m["id"]: m for m in json.loads((REPO_ROOT / "configs/models.json").read_text())["models"]}

    def _argv(self, model_id, cohort_args=None, platform="google", budget=None, endpoint=None):
        model = self.registry[model_id]
        protocol = protocols.protocol_for(model)
        args = protocols.resolve_runner_args(protocol, model, cohort_args or {}, {}, budget)
        spec = protocols.TrialSpec(model, protocol, "exp", "conf_interest", 2, "trial_x", "label", args, {}, platform)
        argv, env = protocols.build_trial_command(self.settings, spec, endpoint)
        return argv, env

    @staticmethod
    def _flag(argv, name):
        return argv[argv.index(name) + 1]

    def test_qwen_direct_mcp_matches_old_matrix_defaults(self):
        argv, _ = self._argv("vlm_qwen3_vl_30b_a3b_instruct", {"max_steps": 32, "fill_only_done": True})
        self.assertTrue(argv[1].endswith("src/baselines/run_qwen_direct_mcp_eval.py"))
        self.assertEqual(self._flag(argv, "--model-kind"), "vlm")
        self.assertEqual(self._flag(argv, "--api-timeout-s"), "300")
        self.assertEqual(self._flag(argv, "--timeout-s"), "1800")
        self.assertEqual(self._flag(argv, "--max-steps"), "32")
        self.assertEqual(self._flag(argv, "--max-new-tokens"), "1024")
        self.assertIn("--fill-only-done", argv)
        self.assertIn("--headless", argv)

    def test_opencua_mcp_uses_model_runner_defaults_and_vllm_python(self):
        argv, _ = self._argv("computer_use_opencua_32b_direct_mcp")
        self.assertEqual(self._flag(argv, "--timeout-s"), "9000")
        self.assertEqual(self._flag(argv, "--model-kind"), "computer_use_agent")
        self.assertTrue(argv[0].endswith(".venv-opencua/bin/python"))

    def test_mediated_budget_profile_and_kind_tokens(self):
        argv, _ = self._argv("vlm_qwen3_vl_30b_a3b_instruct_formfactory_style", {"observation_mode": "vision_only_coords_v1"}, budget="large_qwen3")
        self.assertEqual(self._flag(argv, "--max-steps"), "128")
        self.assertEqual(self._flag(argv, "--timeout-s"), "10800")
        self.assertEqual(self._flag(argv, "--max-new-tokens"), "160")
        self.assertEqual(self._flag(argv, "--prompt-profile"), "runtime_safe_v1")
        self.assertIn("--no-fewshot-enabled", argv)
        self.assertIn("--require-gpu", argv)

    def test_opencua_native_gets_endpoint_flags(self):
        endpoint = serve.Endpoint("computer_use_opencua_32b", "http://127.0.0.1:18001/v1", "opencua-32b", "EMPTY", True)
        argv, env = self._argv("computer_use_opencua_32b", {"formfactory_style": True, "ruler_overlay": True}, endpoint=endpoint)
        self.assertEqual(self._flag(argv, "--base-url"), "http://127.0.0.1:18001/v1")
        self.assertEqual(self._flag(argv, "--coordinate-type"), "qwen25")
        self.assertIn("--ruler-overlay", argv)
        self.assertEqual(env["OPENAI_BASE_URL"], "http://127.0.0.1:18001/v1")
        self.assertEqual(env["OPEN_CUA_MIN_REQUEST_INTERVAL_S"], "2.0")

    def test_localforms_rewrites_form_and_roots(self):
        argv, _ = self._argv("computer_use_opencua_32b_direct_mcp", platform="localforms")
        self.assertEqual(self._flag(argv, "--form-id"), "lf_conf_interest")
        self.assertEqual(self._flag(argv, "--form-url"), "http://127.0.0.1:5000/forms/lf_conf_interest")
        self.assertTrue(self._flag(argv, "--forms-root").endswith("src/forms_localforms"))

    def test_localforms_rejected_for_unsupported_protocol(self):
        with self.assertRaises(common.FormbenchError):
            self._argv("computer_use_gemini_35_flash_lowcost", platform="localforms")


class EndpointTests(TestCase):
    def test_auto_port_inside_slurm_and_env_override_is_reported(self):
        model = {"id": "m", "provider": "openai_compat", "openai_model": "served", "serve": {"port": "auto"}}
        with mock.patch.dict("os.environ", {"SLURM_JOB_ID": "2310575"}, clear=False):
            endpoint = serve.endpoint_for(model, environ={"OPENAI_MODEL": "other"})
            self.assertEqual(endpoint.base_url, f"http://127.0.0.1:{18000 + 2310575 % 20000}/v1")
        self.assertTrue(endpoint.managed)
        self.assertEqual(endpoint.model_name, "other")
        self.assertTrue(any("OPENAI_MODEL" in o for o in endpoint.overrides))


class MatrixTests(TestCase):
    def test_skip_completed_modes(self):
        with TemporaryDirectory() as tmp:
            settings = _settings(DATASET_ROOT=tmp)
            done = Path(tmp) / "other_exp" / "m1" / "conf_interest" / "run_0002" / "trial_1" / "summary.json"
            done.parent.mkdir(parents=True)
            done.write_text("{}")
            cohort = matrix.Cohort(name="c", experiment_id="this_exp", models=["m1"], forms=["conf_interest"], run_indexes=[2])
            self.assertFalse(matrix.trial_completed(settings, cohort, "m1", "conf_interest", 2))
            cohort.skip_completed = "any"
            self.assertTrue(matrix.trial_completed(settings, cohort, "m1", "conf_interest", 2))
            cohort.skip_completed = "none"
            self.assertFalse(matrix.trial_completed(settings, cohort, "m1", "conf_interest", 2))

    def test_every_manifest_loads(self):
        settings = _settings()
        names = [name for name, _ in matrix.list_experiments(settings)]
        self.assertIn("smoke", names)
        registry = {m["id"] for m in json.loads((REPO_ROOT / "configs/models.json").read_text())["models"]}
        for name in names:
            experiment = matrix.load_experiment(settings, name)
            for cohort in experiment.cohorts:
                self.assertTrue(set(cohort.models) <= registry, (name, cohort.models))
                for model_id in cohort.models:
                    model = next(m for m in json.loads((REPO_ROOT / "configs/models.json").read_text())["models"] if m["id"] == model_id)
                    protocol = protocols.protocol_for(model)
                    protocols.resolve_runner_args(protocol, model, cohort.args, {}, cohort.budget_profile)

    def test_parse_overrides_and_run_indexes(self):
        self.assertEqual(matrix.parse_overrides(["max_steps=4", "fill-only-done=true", "note=hi"]), {"max_steps": 4, "fill_only_done": True, "note": "hi"})
        self.assertEqual(common.parse_run_indexes("1-3,5"), [1, 2, 3, 5])
        self.assertEqual(common.parse_run_indexes([2, 2]), [2])
        with self.assertRaises(ValueError):
            common.parse_run_indexes("0")

    def test_args_to_flags_negations(self):
        flags = common.args_to_flags({"a_b": 1, "on": True, "off": False, "fewshot_enabled": False, "skip": None}, {"fewshot_enabled": "no_fewshot_enabled"})
        self.assertEqual(flags, ["--a-b", "1", "--on", "--no-fewshot-enabled"])


class SlurmTests(TestCase):
    def test_resources_take_max_over_models_and_script_renders(self):
        settings = _settings(SLURM_ACCOUNT="p_test", SLURM_EXCLUDE="i8033")
        res = slurm.job_resources(settings, ["text_qwen3_30b_a3b_instruct_2507", "computer_use_opencua_32b"])
        self.assertEqual((res["gpus"], res["cpus"], res["mem"]), (4, 16, "180G"))
        job = slurm.JobPlan(name="fb-x", formbench_args=["matrix", "smoke"], models=[], resources=res)
        script = slurm.render_script(settings, job)
        self.assertIn("#SBATCH --gres=gpu:4", script)
        self.assertIn("#SBATCH --account=p_test", script)
        self.assertIn("#SBATCH --exclude=i8033", script)
        self.assertIn("-m formbench matrix smoke", script)


class SettingsTests(TestCase):
    def test_env_file_parsing(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text('# c\nexport A=1\nB="two words"\nC=x # trailing\nbad line\n')
            self.assertEqual(parse_env_file(path), {"A": "1", "B": "two words", "C": "x"})

    def test_localforms_port_auto(self):
        self.assertEqual(load_settings(environ={"LOCALFORMS_PORT": "auto"}).localforms_port, 5000)
        self.assertEqual(load_settings(environ={"LOCALFORMS_PORT": "auto", "SLURM_JOB_ID": "2331078"}).localforms_port, 38000 + 2331078 % 9000)
        self.assertEqual(load_settings(environ={"LOCALFORMS_PORT": "6123", "SLURM_JOB_ID": "5"}).localforms_port, 6123)

    def test_process_env_overrides_file(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / ".env").write_text("LOCALFORMS_PORT=6000\nCACHE_ROOT=cache\n")
            settings = load_settings(environ={"LOCALFORMS_PORT": "7000"}, root=Path(tmp))
            self.assertEqual(settings.localforms_port, 7000)
            self.assertEqual(settings.cache_root, Path(tmp).resolve() / "cache")


class QwenRunParamsTests(TestCase):
    def test_direct_mcp_runner_records_run_params(self):
        source = (REPO_ROOT / "src/baselines/run_qwen_direct_mcp_eval.py").read_text()
        self.assertIn('"run_params": {', source)
        self.assertIn("rbe._update_experiment_indexes(", source)
