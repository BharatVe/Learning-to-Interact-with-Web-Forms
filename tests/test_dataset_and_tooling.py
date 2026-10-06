"""Dataset pipeline reproducibility, scripts/env.sh behaviour, Makefile wiring, shell syntax."""

import filecmp
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dataset import common as dcommon  # noqa: E402

ENV = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}


def run_module(*args):
    proc = subprocess.run([sys.executable, "-m", *args], cwd=str(REPO_ROOT), env=ENV, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise AssertionError(f"{args} failed:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def assert_same_tree(test, left: Path, right: Path):
    cmp = filecmp.dircmp(str(left), str(right))
    stack = [cmp]
    while stack:
        cur = stack.pop()
        test.assertEqual(cur.left_only + cur.right_only, [], f"{cur.left} vs {cur.right}")
        _, mismatch, errors = filecmp.cmpfiles(cur.left, cur.right, cur.common_files, shallow=False)
        test.assertEqual(mismatch + errors, [], f"{cur.left}: differing files")
        stack.extend(cur.subdirs.values())


class DatasetReproducibilityTests(TestCase):
    """Regenerating from data/generator/*.csv must reproduce the committed dataset byte-for-byte."""

    def test_specs_from_generator_csv(self):
        with TemporaryDirectory() as tmp:
            run_module("dataset.sync_specs", "--output-forms-root", f"{tmp}/forms", "--forms-master", f"{tmp}/forms_master.csv")
            assert_same_tree(self, Path(tmp) / "forms", REPO_ROOT / "src" / "forms")
            self.assertTrue(filecmp.cmp(f"{tmp}/forms_master.csv", REPO_ROOT / "data/specs/forms_master.csv", shallow=False))

    def test_answers_seed_zero_reproduces_committed_and_seed_matters(self):
        with TemporaryDirectory() as tmp:
            run_module("dataset.answers", "--answers-root", f"{tmp}/a0")
            assert_same_tree(self, Path(tmp) / "a0", REPO_ROOT / "data" / "answers")
            run_module("dataset.answers", "--answers-root", f"{tmp}/a1", "--seed", "1", "--runs-per-form", "2")
            runs = json.loads((Path(tmp) / "a1" / "conf_interest" / "runs.json").read_text())["runs"]
            committed = json.loads((REPO_ROOT / "data/answers/conf_interest/runs.json").read_text())["runs"]
            self.assertEqual(len(runs), 2)
            self.assertNotEqual(runs[0]["answers"], committed[0]["answers"])

    def test_answers_refuse_overwrite_without_rewrite(self):
        with TemporaryDirectory() as tmp:
            run_module("dataset.answers", "--answers-root", f"{tmp}/a", "--runs-per-form", "1")
            proc = subprocess.run([sys.executable, "-m", "dataset.answers", "--answers-root", f"{tmp}/a", "--runs-per-form", "1"],
                                  cwd=str(REPO_ROOT), env=ENV, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("--rewrite", proc.stderr)

    def test_localforms_regeneration(self):
        with TemporaryDirectory() as tmp:
            t = Path(tmp)
            run_module("dataset.localforms", "--site-dir", str(t / "site"), "--forms-out", str(t / "forms"), "--answers-out", str(t / "answers"))
            assert_same_tree(self, t / "forms", REPO_ROOT / "src/forms_localforms")
            assert_same_tree(self, t / "answers", REPO_ROOT / "data/answers_localforms")
            assert_same_tree(self, t / "site" / "templates", REPO_ROOT / "evaluation_additions/formfactory_import/site/templates")
            self.assertTrue(filecmp.cmp(t / "site" / "app.py", REPO_ROOT / "evaluation_additions/formfactory_import/site/app.py", shallow=False))

    def test_validation_and_integrity_pass_on_committed_data(self):
        self.assertIn("[PASS]", run_module("dataset.validate", "--strict"))
        self.assertIn("[PASS]", run_module("dataset.integrity"))

    def test_common_helpers(self):
        self.assertEqual(dcommon.split_options(" a; b ;;c "), ["a", "b", "c"])
        self.assertEqual(dcommon.split_options(None), [])
        self.assertTrue(dcommon.as_bool("Yes") and not dcommon.as_bool("0"))
        self.assertEqual(dcommon.as_int(" 7 ", "f", "form", "q"), 7)
        with self.assertRaises(ValueError):
            dcommon.as_int("x", "q_order", "form", "q")
        self.assertEqual(dcommon.WIDGET_TO_QTYPE[dcommon.QTYPE_TO_WIDGET["DROPDOWN"]], "DROPDOWN")


class EnvScriptTests(TestCase):
    """scripts/env.sh: .env loading, precedence, cache/LD handling."""

    def _run(self, dotenv: str, extra_env=None, prepare=None):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "scripts").mkdir()
        shutil.copy(REPO_ROOT / "scripts" / "env.sh", root / "scripts" / "env.sh")
        (root / ".env").write_text(dotenv.replace("{root}", str(root)))
        if prepare:
            prepare(root)
        env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"), "LD_LIBRARY_PATH": "/orig/lib"}
        env.update(extra_env or {})
        # exec `env` directly (not `bash -c`) with stdin detached: bash sources ~/.bashrc when stdin is a socket.
        proc = subprocess.run(["bash", str(root / "scripts" / "env.sh"), "env"], env=env, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        values = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
        return root, values

    def test_dotenv_parsing_and_precedence(self):
        root, v = self._run('# comment\nexport A_ONE=1\nB_TWO="two words"\nC_THREE=\'q\'\nbad-key=x\nD_SHELL=from_file\n', {"D_SHELL": "from_shell"})
        self.assertEqual((v["A_ONE"], v["B_TWO"], v["C_THREE"], v["D_SHELL"]), ("1", "two words", "q", "from_shell"))
        self.assertNotIn("bad-key", v)

    def test_caches_pythonpath_and_ld(self):
        root, v = self._run("CACHE_ROOT=cachedir\n")
        self.assertEqual(v["CACHE_ROOT"], f"{root}/cachedir")
        self.assertEqual(v["HF_HOME"], f"{root}/cachedir/hf")
        self.assertEqual(v["PLAYWRIGHT_BROWSERS_PATH"], f"{root}/cachedir/playwright")
        self.assertTrue(v["PYTHONPATH"].startswith(f"{root}/src"))
        self.assertTrue(v["PATH"].startswith(f"{root}/.node-tools/node_modules/.bin"), v["PATH"][:200])
        self.assertEqual(v["LD_LIBRARY_PATH"], "/orig/lib")
        self.assertEqual(v["MODULE_LD_LIBRARY_PATH"], "/orig/lib")
        self.assertTrue((root / "cachedir" / "pip").is_dir())

    def test_chromium_record_used_only_if_executable_exists(self):
        def stale(root):
            (root / "c" / "playwright").mkdir(parents=True)
            (root / "c" / "playwright" / ".mcp-chromium-executable").write_text("/gone/chrome\n")

        _, v = self._run("CACHE_ROOT=c\n", prepare=stale)
        self.assertNotIn("PLAYWRIGHT_MCP_CHROMIUM_EXECUTABLE", v)

        def valid(root):
            stale(root)
            exe = root / "chrome"
            exe.write_text("#!/bin/sh\n")
            exe.chmod(0o755)
            (root / "c" / "playwright" / ".mcp-chromium-executable").write_text(f"{exe}\n")

        root, v = self._run("CACHE_ROOT=c\n", prepare=valid)
        self.assertEqual(v["PLAYWRIGHT_MCP_CHROMIUM_EXECUTABLE"], f"{root}/chrome")


class MakefileTests(TestCase):
    def _make(self, *args):
        proc = subprocess.run(["make", "-n", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return " ".join(proc.stdout.replace("\\\n", " ").split())  # join continuation lines, collapse spaces

    def test_variables_become_cli_flags(self):
        out = self._make("eval", "MODEL=m1", "FORM=f1", "RUN=3", 'SET=max_steps=4 fill_only=true', "DRY_RUN=1", "PLATFORM=localforms")
        self.assertIn("-m formbench eval m1 f1 3", out)
        self.assertIn("--platform localforms", out)
        self.assertIn("--set max_steps=4 --set fill_only=true", out)
        self.assertIn("--dry-run", out)
        out = self._make("matrix", "EXPERIMENT=fill_only_done_30", "COHORT=qwen", "MODELS=a,b", "RUNS=1-3", "SUFFIX=_r2", "SKIP_CHECKS=1")
        for part in ("matrix fill_only_done_30", "--models a,b", "--runs 1-3", "--cohort qwen", "--experiment-suffix _r2", "--skip-checks"):
            self.assertIn(part, out)
        self.assertNotIn("--dry-run", out)
        out = self._make("submit", "EXPERIMENT=smoke", "SPLIT=model", "CHAIN=afterok")
        self.assertIn("submit smoke", out)
        self.assertIn("--split model --chain afterok", out)
        self.assertIn("report --study core_report --check", self._make("study", "NAME=core_report", "CHECK=1"))
        self.assertIn("ideal --forms a,b --runs 1-2 --platform google --overwrite --submit", self._make("ideal-runs", "FORMS=a,b", "RUNS=1-2", "PLATFORM=google", "OVERWRITE=1", "SUBMIT=1"))
        self.assertIn("models list --all", self._make("models", "ALL=1"))

    def test_help_lists_every_target(self):
        proc = subprocess.run(["make", "-C", str(REPO_ROOT), "help"], capture_output=True, text=True, timeout=60)
        targets = [line.split()[1] for line in proc.stdout.splitlines() if line.startswith("  make ") and "Examples" not in line and "=" not in line.split()[1]]
        for target in ("setup", "doctor", "test", "models", "model-check", "serve-model", "install-models", "data", "data-check",
                       "serve-forms", "ideal-runs", "ideal-status", "experiments", "eval", "matrix", "submit", "report",
                       "reference-report", "studies", "study", "inspect"):
            self.assertIn(target, targets)


class ShellSyntaxTests(TestCase):
    def test_all_shell_scripts_parse(self):
        for script in sorted((REPO_ROOT / "scripts").glob("*.sh")):
            proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"{script}: {proc.stderr}")
