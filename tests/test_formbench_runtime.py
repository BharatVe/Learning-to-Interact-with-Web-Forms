"""Runtime behaviour of formbench: server lifecycle, failsafes, matrix execution, Slurm, ideal runs.

No GPUs, models or network: vLLM is replaced by a stub OpenAI-compatible HTTP server and
runner scripts by a stub that writes summary.json, so the real orchestration code runs.
"""

import json
import os
import socket
import stat
import sys
import textwrap
import threading
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from formbench import checks, common, data, ideal, install, matrix, protocols, serve, slurm  # noqa: E402
from formbench.settings import load_settings  # noqa: E402

PY = sys.executable


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def fake_openai_server(served_names):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"data": [{"id": name} for name in served_names]})

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self._send({"choices": [{"message": {"content": "{}"}}]})

    server = HTTPServer(("127.0.0.1", free_port()), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


STUB_VLLM = textwrap.dedent(
    """\
    #!{py}
    # Stand-in for `python -m vllm.entrypoints.openai.api_server ...`: serves /v1/models + chat.
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    args = sys.argv[1:]
    if "--crash" in " ".join(args) or "crash" in args[args.index("--served-model-name") + 1]:
        print("CUDA out of memory (stub)"); sys.exit(3)
    name = args[args.index("--served-model-name") + 1]
    port = int(args[args.index("--port") + 1])
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, payload):
            body = json.dumps(payload).encode(); self.send_response(200)
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_GET(self): self._send({{"data": [{{"id": name}}]}})
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0)); self._send({{"choices": [{{"message": {{"content": "ok"}}}}]}})
    print("argv:", json.dumps(args), flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()
    """
)

STUB_RUNNER = textwrap.dedent(
    """\
    # Stand-in for a src/baselines runner: writes summary.json where the real runners do.
    import argparse, json, os, sys
    from pathlib import Path
    p = argparse.ArgumentParser()
    for flag in ("--config", "--model-id", "--model-kind", "--form-id", "--run-index", "--trial-id", "--experiment-id",
                 "--run-label", "--dataset-root", "--max-steps"):
        p.add_argument(flag)
    a, _ = p.parse_known_args()
    outcome = os.environ.get("STUB_OUTCOME_" + a.model_id.upper(), "success")
    if outcome == "crash":
        print("Traceback: stub runner crashed", file=sys.stderr); sys.exit(2)
    root = Path(a.dataset_root or "data/model_baselines")
    d = root / a.experiment_id / a.model_id / a.form_id / f"run_{int(a.run_index):04d}" / a.trial_id
    d.mkdir(parents=True, exist_ok=True)
    summary = {"success": outcome == "success", "stop_reason": outcome, "failure_category": None if outcome == "success" else outcome,
               "failure_detail": "", "openai_base_url": os.environ.get("OPENAI_BASE_URL"), "max_steps": a.max_steps}
    (d / "summary.json").write_text(json.dumps(summary))
    sys.exit(0 if outcome == "success" else 1)
    """
)


class RuntimeEnv:
    """Temp registry + dataset/logs dirs + stub executables, wrapped in Settings."""

    def __init__(self, tmp: Path, models):
        self.tmp = tmp
        (tmp / "models.json").write_text(json.dumps({"models": models}))
        stub = tmp / "stub_vllm"
        stub.write_text(STUB_VLLM.format(py=PY))
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        (tmp / "stub_runner.py").write_text(STUB_RUNNER)
        self.env = {
            "LOCAL": "1",
            "MODELS_CONFIG": str(tmp / "models.json"),
            "DATASET_ROOT": str(tmp / "ds"),
            "LOGS_DIR": str(tmp / "logs"),
            "REPORTS_DIR": str(tmp / "reports"),
            "VLLM_PYTHON_BIN": str(stub),
            "PYTHON_BIN": PY,
        }
        self.settings = load_settings(environ=self.env)


def _served_model(mid, name, port, **extra):
    model = {
        "id": mid, "kind": "text_llm", "provider": "openai_compat", "track": "direct_mcp_tool_use", "requires_gpu": False,
        "openai_model": name, "served_model_name": name, "hf_repo": "org/x",
        "serve": {"port": port, "tensor_parallel": 1, "startup_attempts": 50, "startup_sleep_s": 0.2},
    }
    model.update(extra)
    return model


class VLLMLifecycleTests(TestCase):
    def test_start_ready_warmup_and_stop(self):
        with TemporaryDirectory() as tmp:
            port = free_port()
            rt = RuntimeEnv(Path(tmp), [_served_model("m", "served-m", port)])
            model = json.loads((Path(tmp) / "models.json").read_text())["models"][0]
            endpoint = serve.endpoint_for(model, environ={})
            self.assertTrue(endpoint.managed)
            with serve.VLLMServer(rt.settings, model, endpoint, log_dir=Path(tmp) / "logs") as server:
                self.assertFalse(server.reused)
                self.assertEqual(serve.list_served_models(endpoint), ["served-m"])
                self.assertGreater(endpoint.warmup_s, -1)
                pid = server.process.pid
            self.assertIsNone(serve.list_served_models(endpoint, timeout=1))
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
            log = (Path(tmp) / "logs" / "m-vllm-local.log").read_text()
            self.assertIn('"--served-model-name", "served-m"', log)

    def test_crash_before_ready_raises_with_log_tail(self):
        with TemporaryDirectory() as tmp:
            rt = RuntimeEnv(Path(tmp), [_served_model("m", "crash-model", free_port())])
            model = json.loads((Path(tmp) / "models.json").read_text())["models"][0]
            with self.assertRaises(common.FormbenchError) as ctx:
                serve.VLLMServer(rt.settings, model, serve.endpoint_for(model, environ={}), log_dir=Path(tmp) / "logs").start()
            self.assertIn("exited with code 3", str(ctx.exception))
            self.assertIn("out of memory", str(ctx.exception))

    def test_reuses_running_server_and_detects_port_conflicts(self):
        with TemporaryDirectory() as tmp, fake_openai_server(["served-m"]) as port:
            rt = RuntimeEnv(Path(tmp), [_served_model("m", "served-m", port), _served_model("other", "other-name", port)])
            models = json.loads((Path(tmp) / "models.json").read_text())["models"]
            ok = serve.VLLMServer(rt.settings, models[0], serve.endpoint_for(models[0], environ={}))
            ok.start()
            self.assertTrue(ok.reused)
            self.assertIsNone(ok.process)
            clash = serve.VLLMServer(rt.settings, models[1], serve.endpoint_for(models[1], environ={}))
            with self.assertRaises(common.FormbenchError) as ctx:
                clash.start()
            self.assertIn("port in use", str(ctx.exception))

    def test_smoke_chat_text_and_vlm(self):
        with fake_openai_server(["x"]) as port:
            endpoint = serve.Endpoint("m", f"http://127.0.0.1:{port}/v1", "x", "EMPTY", False)
            self.assertGreaterEqual(serve.smoke_chat(endpoint, "text_llm"), 0)
            self.assertGreaterEqual(serve.smoke_chat(endpoint, "vlm"), 0)


class VLLMCommandParityTests(TestCase):
    """build_vllm_command must emit what run_qwen_vllm_server.sh / run_opencua_vllm_server.sh did."""

    def setUp(self):
        self.settings = load_settings(environ={"LOCAL": "1", "VLLM_PYTHON_BIN": "/opt/vllm/bin/python"})
        self.registry = {m["id"]: m for m in json.loads((REPO_ROOT / "configs/models.json").read_text())["models"]}

    def _cmd(self, mid):
        model = self.registry[mid]
        with mock.patch.dict(os.environ, {"MODULE_LD_LIBRARY_PATH": "/mod/lib"}, clear=False):
            return serve.build_vllm_command(self.settings, model, serve.endpoint_for(model, environ={}))

    def test_qwen_vl_flags(self):
        argv, env = self._cmd("vlm_qwen3_vl_30b_a3b_instruct")
        joined = " ".join(argv)
        self.assertEqual(argv[:3], ["/opt/vllm/bin/python", "-m", "vllm.entrypoints.openai.api_server"])
        for flag in ("--tensor-parallel-size 2", "--gpu-memory-utilization 0.92", "--max-model-len 32768", "--trust-remote-code",
                     "--disable-custom-all-reduce", "--enable-auto-tool-choice", "--tool-call-parser hermes",
                     "--generation-config vllm", "--served-model-name vlm-qwen3-vl-30b-a3b-instruct"):
            self.assertIn(flag, joined)
        self.assertEqual(argv[argv.index("--limit-mm-per-prompt") + 1], '{"image":1,"video":0}')
        self.assertEqual(env["LD_LIBRARY_PATH"], "/mod/lib")
        self.assertEqual(env["NCCL_P2P_DISABLE"], "1")

    def test_opencua_flags(self):
        argv, _ = self._cmd("computer_use_opencua_32b")
        joined = " ".join(argv)
        self.assertIn("--tensor-parallel-size 4", joined)
        self.assertIn("--gpu-memory-utilization 0.9 ", joined + " ")
        self.assertIn("--model xlangai/OpenCUA-32B", joined)
        self.assertNotIn("--enable-auto-tool-choice", joined)
        self.assertNotIn("--generation-config", joined)


class FailsafeCheckTests(TestCase):
    def _status(self, results, name):
        return [r.status for r in results if r.name == name]

    def test_gemini_key_states(self):
        model = {"id": "g", "kind": "computer_use_agent", "provider": "gemini_low_cost", "track": "proprietary_computer_use_low_cost", "gemini_model": "x"}
        with TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("GEMINI_API_KEY_FILE", None)
            key = Path(tmp) / "key"
            settings = load_settings(environ={"LOCAL": "1", "GEMINI_API_KEY_FILE": str(key)})
            self.assertEqual(self._status(checks.check_model(settings, model), "api key"), [checks.FAIL])
            key.write_text("secret")
            key.chmod(0o644)
            self.assertEqual(self._status(checks.check_model(settings, model), "api key"), [checks.WARN])
            key.chmod(0o600)
            self.assertEqual(self._status(checks.check_model(settings, model), "api key"), [checks.OK])
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}):
                key.unlink()
                self.assertEqual(self._status(checks.check_model(settings, model), "api key"), [checks.OK])

    def test_remote_api_without_key_fails_and_local_weights_missing_fails(self):
        settings = load_settings(environ={"LOCAL": "1", "MODELS_DIR": "/nonexistent"})
        api = {"id": "a", "kind": "computer_use_agent", "provider": "api_over_mcp", "track": "direct_api_tool_use", "openai_model": "m", "api_key_env": "FB_TEST_KEY"}
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FB_TEST_KEY", None)
            self.assertEqual(self._status(checks.check_model(settings, api), "api key"), [checks.FAIL])
        local = {"id": "l", "kind": "text_llm", "provider": "local_hf", "track": "mediated", "hf_repo": "org/none", "weights_dir": "/nonexistent/w"}
        self.assertEqual(self._status(checks.check_model(settings, local), "weights"), [checks.FAIL])

    def test_endpoint_states_and_env_override(self):
        settings = load_settings(environ={"LOCAL": "1"})
        closed = free_port()
        remote = {"id": "r", "kind": "text_llm", "provider": "openai_compat", "track": "direct_mcp_tool_use", "openai_model": "x", "openai_base_url": f"http://127.0.0.1:{closed}/v1"}
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            self.assertEqual(self._status(checks.check_model(settings, remote), "endpoint"), [checks.FAIL])
        managed = _served_model("m", "served-m", closed)
        self.assertEqual(self._status(checks.check_model(settings, managed), "endpoint"), [checks.OK])  # will be started
        with fake_openai_server(["other"]) as port:
            wrong = dict(remote, openai_base_url=f"http://127.0.0.1:{port}/v1")
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
                self.assertEqual(self._status(checks.check_model(settings, wrong), "endpoint"), [checks.FAIL])
                with mock.patch.dict(os.environ, {"OPENAI_MODEL": "other"}):
                    results = checks.check_model(settings, wrong, smoke=True)
                    self.assertEqual(self._status(results, "endpoint"), [checks.OK])
                    self.assertEqual(self._status(results, "env override"), [checks.WARN])
                    self.assertEqual(self._status(results, "smoke chat"), [checks.OK])

    def test_gpu_requirements(self):
        settings = load_settings(environ={"LOCAL": "1"})
        model = _served_model("m", "x", free_port(), requires_gpu=True, resources={"gpus": 2, "min_vram_gb": 40})
        with mock.patch.object(checks, "visible_gpus", return_value=[]):
            self.assertEqual(self._status(checks.check_model(settings, model, probe_endpoint=False), "gpu"), [checks.WARN])
            self.assertEqual(self._status(checks.check_model(settings, model, probe_endpoint=False, expect_gpus_here=True), "gpu"), [checks.FAIL])
        one = [{"index": "0", "name": "A100", "memory_gb": 80.0}]
        with mock.patch.object(checks, "visible_gpus", return_value=one):
            self.assertEqual(self._status(checks.check_model(settings, model, probe_endpoint=False, expect_gpus_here=True), "gpu"), [checks.FAIL])
        small = [{"index": str(i), "name": "A100", "memory_gb": 39.5} for i in range(2)]
        with mock.patch.object(checks, "visible_gpus", return_value=small):
            self.assertEqual(self._status(checks.check_model(settings, model, probe_endpoint=False), "gpu"), [checks.FAIL])
        big = [{"index": str(i), "name": "A100", "memory_gb": 80.0} for i in range(2)]
        with mock.patch.object(checks, "visible_gpus", return_value=big):
            self.assertEqual(self._status(checks.check_model(settings, model, probe_endpoint=False), "gpu"), [checks.OK])

    def test_check_models_rejects_unknown_ids(self):
        settings = load_settings(environ={"LOCAL": "1"})
        with mock.patch("sys.stdout", new_callable=StringIO) as out:
            self.assertFalse(checks.check_models(settings, ["no_such_model"], probe_endpoint=False))
        self.assertIn("unknown model id", out.getvalue())


class MatrixExecutionTests(TestCase):
    """Real MatrixRunner with a stub runner script (and stub vLLM for served models)."""

    def _runner(self, rt, cohort_kw, protocol_name="direct_mcp", **kw):
        stub = protocols.Protocol(name=protocol_name, script=str(rt.tmp / "stub_runner.py"), description="stub",
                                  defaults={"max_steps": 7}, passes_model_kind=True)
        experiment = matrix.Experiment(id="exp", description="t", cohorts=[matrix.Cohort(**cohort_kw)])
        runner = matrix.MatrixRunner(rt.settings, experiment, **kw)
        return runner, mock.patch.object(matrix, "protocol_for", return_value=stub), mock.patch("formbench.analysis_hooks.post_matrix")

    def test_runs_trials_records_outcomes_and_skips_on_rerun(self):
        with TemporaryDirectory() as tmp:
            api = {"id": "api_model", "kind": "computer_use_agent", "provider": "gemini_low_cost", "track": "proprietary_computer_use_low_cost", "gemini_model": "g"}
            rt = RuntimeEnv(Path(tmp), [api])
            cohort = dict(name="c", experiment_id="exp1", models=["api_model"], forms=["conf_interest", "event_rsvp"], run_indexes=[1, 2])
            runner, p1, p2 = self._runner(rt, cohort, overrides={"max_steps": 3})
            with p1, p2 as post, mock.patch.dict(os.environ, {"STUB_OUTCOME_API_MODEL": "success"}):
                self.assertEqual(runner.run(), 0)
            self.assertEqual(len(runner.outcomes), 4)
            self.assertTrue(all(o.exit_code == 0 and o.success for o in runner.outcomes))
            post.assert_called_once()
            summary = next((Path(tmp) / "ds" / "exp1" / "api_model" / "conf_interest" / "run_0001").glob("*/summary.json"))
            self.assertEqual(json.loads(summary.read_text())["max_steps"], "3")  # --set override reached the runner
            report = json.loads(next((Path(tmp) / "logs" / "matrix").glob("exp-*.json")).read_text())
            self.assertEqual(len(report["outcomes"]), 4)
            again, p1, p2 = self._runner(rt, cohort)
            with p1, p2 as post_again:
                again.run()
            self.assertEqual((len(again.outcomes), again.skipped), (0, 4))
            post_again.assert_not_called()

    def test_runner_failure_is_recorded_and_fail_fast_stops(self):
        with TemporaryDirectory() as tmp:
            api = {"id": "api_model", "kind": "computer_use_agent", "provider": "gemini_low_cost", "track": "proprietary_computer_use_low_cost", "gemini_model": "g"}
            rt = RuntimeEnv(Path(tmp), [api])
            cohort = dict(name="c", experiment_id="exp2", models=["api_model"], forms=["conf_interest", "event_rsvp"], run_indexes=[1])
            with mock.patch.dict(os.environ, {"STUB_OUTCOME_API_MODEL": "timeout"}):
                runner, p1, p2 = self._runner(rt, cohort)
                with p1, p2:
                    self.assertEqual(runner.run(), 0)  # benchmark failures do not fail the job by default
                self.assertEqual([o.exit_code for o in runner.outcomes], [1, 1])
                self.assertEqual(runner.outcomes[0].failure_category, "timeout")
                cohort["experiment_id"] = "exp3"
                strict, p1, p2 = self._runner(rt, cohort, fail_fast=True)
                with p1, p2, self.assertRaises(common.FormbenchError):
                    strict.run()
                self.assertEqual(len(strict.outcomes), 1)

    def test_runner_crash_without_summary_fails_the_job(self):
        with TemporaryDirectory() as tmp:
            api = {"id": "api_model", "kind": "computer_use_agent", "provider": "gemini_low_cost", "track": "proprietary_computer_use_low_cost", "gemini_model": "g"}
            rt = RuntimeEnv(Path(tmp), [api])
            cohort = dict(name="c", experiment_id="exp_crash", models=["api_model"], forms=["conf_interest"], run_indexes=[1])
            runner, p1, p2 = self._runner(rt, cohort)
            with p1, p2 as post, mock.patch.dict(os.environ, {"STUB_OUTCOME_API_MODEL": "crash"}):
                self.assertEqual(runner.run(), 1)
            self.assertFalse(runner.outcomes[0].summary_written)
            post.assert_not_called()  # nothing to summarise

    def test_server_start_failure_marks_trials_and_continues_with_next_model(self):
        with TemporaryDirectory() as tmp:
            broken = _served_model("broken", "crash-model", free_port())
            api = {"id": "api_model", "kind": "computer_use_agent", "provider": "gemini_low_cost", "track": "proprietary_computer_use_low_cost", "gemini_model": "g"}
            rt = RuntimeEnv(Path(tmp), [broken, api])
            cohort = dict(name="c", experiment_id="exp_srv", models=["broken", "api_model"], forms=["conf_interest", "event_rsvp"], run_indexes=[1])
            runner, p1, p2 = self._runner(rt, cohort)
            with p1, p2 as post, mock.patch.dict(os.environ, {"STUB_OUTCOME_API_MODEL": "success"}):
                self.assertEqual(runner.run(), 1)  # infrastructure failure -> non-zero job exit
            by_model = {}
            for o in runner.outcomes:
                by_model.setdefault(o.model_id, []).append(o)
            self.assertEqual([o.stop_reason for o in by_model["broken"]], ["server_start_failed"] * 2)
            self.assertTrue(all(o.success for o in by_model["api_model"]))
            post.assert_called_once()
            report = json.loads(next((Path(tmp) / "logs" / "matrix").glob("exp-*.json")).read_text())
            self.assertEqual(report["server_failures"][0]["model_id"], "broken")
            self.assertIn("exited with code 3", report["server_failures"][0]["error"])
            strict, p1, p2 = self._runner(rt, dict(cohort, experiment_id="exp_srv2"), fail_fast=True)
            with p1, p2, self.assertRaises(common.FormbenchError):
                strict.run()

    def test_vlm_fallback_runs_after_timeout(self):
        with TemporaryDirectory() as tmp:
            primary = {"id": "vlm_primary", "kind": "vlm", "provider": "local_hf", "track": "mediated", "hf_repo": "org/a"}
            backup = {"id": "vlm_backup", "kind": "vlm", "provider": "local_hf", "track": "mediated", "hf_repo": "org/b", "is_fallback": True, "fallback_for": "vlm_primary"}
            rt = RuntimeEnv(Path(tmp), [primary, backup])
            cohort = dict(name="c", experiment_id="exp4", models=["vlm_primary"], forms=["conf_interest"], run_indexes=[1])
            with mock.patch.dict(os.environ, {"STUB_OUTCOME_VLM_PRIMARY": "timeout", "STUB_OUTCOME_VLM_BACKUP": "success"}):
                runner, p1, p2 = self._runner(rt, cohort, protocol_name="mediated")
                with p1, p2:
                    runner.run()
            self.assertEqual([(o.model_id, o.fallback_for) for o in runner.outcomes], [("vlm_primary", None), ("vlm_backup", "vlm_primary")])
            self.assertTrue(runner.outcomes[1].success)

    def test_managed_server_started_for_trials_and_stopped(self):
        with TemporaryDirectory() as tmp:
            port = free_port()
            rt = RuntimeEnv(Path(tmp), [_served_model("served", "served-name", port)])
            cohort = dict(name="c", experiment_id="exp5", models=["served"], forms=["conf_interest"], run_indexes=[1])
            runner, p1, p2 = self._runner(rt, cohort)
            with p1, p2, mock.patch.dict(os.environ, {"STUB_OUTCOME_SERVED": "success"}):
                runner.run()
            summary = json.loads(next((Path(tmp) / "ds" / "exp5").rglob("summary.json")).read_text())
            self.assertEqual(summary["openai_base_url"], f"http://127.0.0.1:{port}/v1")
            self.assertIsNone(serve.list_served_models(serve.Endpoint("s", f"http://127.0.0.1:{port}/v1", "x", "EMPTY", True), timeout=1))

    def test_should_fallback_rules(self):
        vlm, text = {"kind": "vlm"}, {"kind": "text_llm"}
        out = lambda code, cat=None, det=None: matrix.TrialOutcome("e", "m", "f", 1, "t", code, 0.0, failure_category=cat, failure_detail=det)  # noqa: E731
        self.assertTrue(matrix._should_fallback(vlm, out(1, "timeout")))
        self.assertTrue(matrix._should_fallback(vlm, out(1, "other", "CUDA out of memory")))
        self.assertFalse(matrix._should_fallback(vlm, out(1, "loop_stall_terminal")))
        self.assertFalse(matrix._should_fallback(vlm, out(0, "timeout")))
        self.assertFalse(matrix._should_fallback(text, out(1, "timeout")))

    def test_cohort_validation_errors(self):
        settings = load_settings(environ={"LOCAL": "1"})
        bad = [
            ({"models": ["m"]}, "no experiment_id"),
            ({"experiment_id": "e"}, "lists no models"),
            ({"experiment_id": "e", "models": ["m"], "platform": "mars"}, "platform"),
            ({"experiment_id": "e", "models": ["m"], "skip_completed": "maybe"}, "skip_completed"),
            ({"experiment_id": "e", "models": ["m"], "forms": ["not_a_form"]}, "unknown form"),
        ]
        for raw, message in bad:
            with self.assertRaises(common.FormbenchError) as ctx:
                matrix._build_cohort(settings, raw, {}, "c")
            self.assertIn(message, str(ctx.exception))
        cohort = matrix._build_cohort(settings, {"experiment_id": "e", "models": "a,b", "forms": "all", "forms_limit": 3, "run_indexes": "1-2"}, {"args": {"x": 1}}, "c")
        self.assertEqual((cohort.models, len(cohort.forms), cohort.run_indexes, cohort.args), (["a", "b"], 3, [1, 2], {"x": 1}))

    def test_filters_and_missing_manifest(self):
        settings = load_settings(environ={"LOCAL": "1"})
        experiment = matrix.load_experiment(settings, "fill_only_done_30")
        only = matrix.apply_cohort_filters(experiment, ["qwen"], ["vlm_qwen3_vl_30b_a3b_instruct"])
        self.assertEqual([c.models for c in only.cohorts], [["vlm_qwen3_vl_30b_a3b_instruct"]])
        with self.assertRaises(common.FormbenchError):
            matrix.apply_cohort_filters(experiment, ["nope"], [])
        with self.assertRaises(common.FormbenchError) as ctx:
            matrix.load_experiment(settings, "does_not_exist")
        self.assertIn("available:", ctx.exception.hint)


class SlurmPlanningTests(TestCase):
    def test_split_modes_and_dry_run_submission(self):
        settings = load_settings(environ={"LOCAL": "1"})
        experiment = matrix.load_experiment(settings, "track_baseline_pilot")
        self.assertEqual(len(slurm.plan_jobs(settings, experiment, "none", ["track_baseline_pilot"])), 1)
        self.assertEqual(len(slurm.plan_jobs(settings, experiment, "cohort", ["track_baseline_pilot"])), 2)
        self.assertEqual(len(slurm.plan_jobs(settings, experiment, "model", ["track_baseline_pilot"])), 3)
        per_run = slurm.plan_jobs(settings, experiment, "run", ["track_baseline_pilot"])
        self.assertEqual(len(per_run), 9)  # 3 models x runs 1-3
        self.assertIn("--runs", per_run[0].formbench_args)
        with self.assertRaises(common.FormbenchError):
            slurm.plan_jobs(settings, experiment, "weird", [])
        with TemporaryDirectory() as tmp:
            settings = load_settings(environ={"LOCAL": "1", "LOGS_DIR": tmp})
            jobs = slurm.plan_jobs(settings, experiment, "cohort", ["track_baseline_pilot"])
            with mock.patch("sys.stdout", new_callable=StringIO) as out:
                self.assertEqual(slurm.submit(settings, jobs, chain="afterok", after="123", dry_run=True), [])
            text = out.getvalue()
            self.assertIn("--dependency=afterok:123", text)
            self.assertIn("--dependency=afterok:<job0>", text)
            self.assertEqual(len(list((Path(tmp) / "slurm" / "jobs").glob("*.sbatch"))), 2)

    def test_adhoc_split_has_one_models_flag_and_time_override(self):
        settings = load_settings(environ={"LOCAL": "1"})
        experiment = matrix.adhoc_experiment(settings, "val", ["text_qwen3_30b_a3b_instruct_2507", "computer_use_opencua_32b"], "conf_interest", "1")
        base = ["--models", "text_qwen3_30b_a3b_instruct_2507,computer_use_opencua_32b", "--forms", "conf_interest", "--experiment-id", "val"]
        jobs = slurm.plan_jobs(settings, experiment, "model", base, time_limit="02:00:00")
        self.assertEqual(len(jobs), 2)
        for job, model in zip(jobs, ["text_qwen3_30b_a3b_instruct_2507", "computer_use_opencua_32b"]):
            self.assertEqual(job.formbench_args.count("--models"), 1)
            self.assertEqual(job.formbench_args[job.formbench_args.index("--models") + 1], model)
            self.assertNotIn("--cohort", job.formbench_args)
            self.assertEqual(job.resources["time"], "02:00:00")
        self.assertEqual(jobs[1].resources["gpus"], 4)
        runs = slurm.plan_jobs(settings, experiment, "run", base + ["--runs", "1"])
        self.assertTrue(all(j.formbench_args.count("--runs") == 1 for j in runs))
        manifest_jobs = slurm.plan_jobs(settings, matrix.load_experiment(settings, "smoke"), "model", ["smoke"])
        self.assertTrue(all("--cohort" in j.formbench_args for j in manifest_jobs))
        with self.assertRaises(common.FormbenchError):
            slurm.plan_jobs(settings, experiment, "model", base, time_limit="two hours")

    def test_resource_parsing(self):
        self.assertEqual(slurm._mem_mb("120G"), 120 * 1024)
        self.assertEqual(slurm._mem_mb("512M"), 512)
        self.assertEqual(slurm._time_s("1-02:00:00"), 26 * 3600)
        self.assertEqual(slurm._time_s("30:00"), 1800)
        with self.assertRaises(common.FormbenchError):
            slurm._mem_mb("lots")
        cpu = slurm.cpu_job("x", ["ideal"])
        self.assertEqual(cpu.resources["gpus"], 0)
        self.assertNotIn("--gres", slurm.render_script(load_settings(environ={"LOCAL": "1"}), cpu))


class IdealAndDataTests(TestCase):
    def _settings(self, tmp):
        root = Path(tmp)
        for form, runs in (("f1", 3), ("f2", 2)):
            (root / "specs" / form).mkdir(parents=True)
            (root / "specs" / form / "spec.json").write_text("{}")
            (root / "answers" / form).mkdir(parents=True)
            (root / "answers" / form / "runs.json").write_text(json.dumps({"runs": [{"answers": []}] * runs}))
        for run in ("run_0001", "run_0003"):
            d = root / "ref" / "f1" / "runs" / run
            d.mkdir(parents=True)
            (d / "tool_trace.jsonl").write_text('{"name": "browser_navigate"}\n')
        failed = root / "ref" / "f2" / "runs" / "run_0001"
        failed.mkdir(parents=True)
        (failed / "tool_trace.jsonl").write_text("")
        (failed / "failure_manifest.json").write_text("{}")
        return load_settings(environ={"LOCAL": "1", "FORMS_ROOT": str(root / "specs"), "ANSWERS_ROOT": str(root / "answers"), "REFERENCE_ROOT": str(root / "ref")})

    def test_coverage_counts_only_successful_traces(self):
        with TemporaryDirectory() as tmp:
            rows = {r["form_id"]: r for r in ideal.coverage(self._settings(tmp))}
            self.assertEqual((rows["f1"]["traces"], rows["f1"]["missing"]), (2, "2"))
            self.assertEqual((rows["f2"]["traces"], rows["f2"]["failed"], rows["f2"]["missing"]), (0, 1, "1,2"))

    def test_generate_dry_run_plans_contiguous_ranges(self):
        with TemporaryDirectory() as tmp:
            settings = self._settings(tmp)
            self.assertEqual(ideal.contiguous_ranges([5, 1, 2, 3, 7, 6]), [(1, 3), (5, 3)])
            with mock.patch("sys.stdout", new_callable=StringIO) as out:
                self.assertEqual(ideal.generate(settings, ["f1"], [1, 3], dry_run=True), 0)
            lines = out.getvalue().strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("--start-index 1 --num-runs 1", lines[0])
            self.assertIn("--interaction-mode mcp_server", lines[0])
            self.assertIn("--skip-existing-video", lines[0])
            self.assertIn("--headless", lines[0])
            with mock.patch("sys.stdout", new_callable=StringIO) as out:
                ideal.generate(settings, ["f2"], [], overwrite=True, headed=True, dry_run=True)
            self.assertIn("--overwrite-existing", out.getvalue())
            self.assertNotIn("--headless", out.getvalue())
            with self.assertRaises(common.FormbenchError):
                ideal.generate(settings, ["missing_form"], [], dry_run=True)

    def test_relpaths_dry_run_and_apply(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp) / "data" / "forms" / "f" / "runs" / "run_0001"
            run.mkdir(parents=True)
            ann = run / "annotations.json"
            ann.write_text(json.dumps({"video_path": "/data/horse/ws/a-b/Learning-to-Interact-with-Web-Forms/data/forms/f/runs/run_0001/x.webm"}))
            settings = load_settings(environ={"LOCAL": "1"}, root=Path(tmp))
            before = ann.read_text()
            data.relpaths(settings, apply=False)
            self.assertEqual(ann.read_text(), before)
            data.relpaths(settings, apply=True)
            self.assertEqual(json.loads(ann.read_text())["video_path"], "data/forms/f/runs/run_0001/x.webm")

    def test_install_dry_run_and_weight_validation(self):
        with TemporaryDirectory() as tmp:
            w = Path(tmp) / "w"
            self.assertEqual(install.weights_valid(w), (False, "missing"))
            w.mkdir()
            (w / "config.json").write_text("{}")
            self.assertFalse(install.weights_valid(w)[0])
            (w / "model.safetensors").write_text("x")
            self.assertEqual(install.weights_valid(w), (True, "ok"))
            model = {"id": "lh", "kind": "text_llm", "provider": "local_hf", "track": "mediated", "hf_repo": "org/model", "weights_dir": str(Path(tmp) / "absent")}
            (Path(tmp) / "models.json").write_text(json.dumps({"models": [model]}))
            settings = load_settings(environ={"LOCAL": "1", "MODELS_CONFIG": str(Path(tmp) / "models.json")})
            with mock.patch("sys.stdout", new_callable=StringIO):
                self.assertEqual(install.install_models(settings, ["lh"], dry_run=True), 0)
                with self.assertRaises(common.FormbenchError):
                    install.install_models(settings, ["nope"], dry_run=True)
            self.assertFalse(install.installable({"provider": "gemini_low_cost", "hf_repo": None}))


class LocalFormsSiteTests(TestCase):
    def test_site_starts_serves_a_form_and_stops(self):
        from formbench.forms_site import LocalFormsSite, site_reachable

        with TemporaryDirectory() as tmp:
            settings = load_settings(environ={"LOCAL": "1", "LOCALFORMS_PORT": str(free_port()), "LOGS_DIR": tmp, "PYTHON_BIN": PY})
            with LocalFormsSite(settings) as site:
                self.assertIsNotNone(site.process)
                with urllib.request.urlopen(f"{settings.localforms_base_url}/forms/lf_conf_interest", timeout=10) as response:
                    html = response.read().decode()
                self.assertIn('role="listitem"', html)
            self.assertFalse(site_reachable(settings, timeout=1))
