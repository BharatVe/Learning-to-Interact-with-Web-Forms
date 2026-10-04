"""Site settings: repo paths, interpreters, caches, cluster options.

Precedence: process environment > `.env` file at the repo root > defaults below.
Nothing here is cluster-specific; HPC details (modules, cache location, Slurm
account) belong in `.env` (see `.env.example`).
"""

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"


def parse_env_file(path: Path) -> Dict[str, str]:
    """Parse a minimal KEY=VALUE file (comments, blank lines, optional quotes, `export`)."""
    values: Dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def localforms_port(value: str, environ: Mapping[str, str]) -> int:
    """'auto': per-job port inside Slurm (concurrent jobs on one node must not collide), else 5000."""
    if str(value).strip().lower() != "auto":
        return int(value)
    job = str(environ.get("SLURM_JOB_ID") or "")
    return 38000 + int(job) % 9000 if job.isdigit() else 5000


def _resolve(root: Path, value: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return path if path.is_absolute() else (root / path)


@dataclass
class Settings:
    root: Path
    python_bin: Path
    vllm_python_bin: Path
    cache_root: Path
    models_config: Path
    models_dir: Path
    dataset_root: Path
    reference_root: Path
    forms_root: Path
    answers_root: Path
    localforms_forms_root: Path
    localforms_answers_root: Path
    localforms_reference_root: Path
    localforms_host: str
    localforms_port: int
    node_tools_dir: Path
    logs_dir: Path
    reports_dir: Path
    modules: str
    gemini_api_key_file: Path
    slurm_account: str
    slurm_partition: str
    slurm_exclude: str
    slurm_extra_args: List[str] = field(default_factory=list)
    force_local: bool = False
    raw: Dict[str, str] = field(default_factory=dict)

    @property
    def playwright_browsers_path(self) -> Path:
        return self.cache_root / "playwright"

    @property
    def localforms_base_url(self) -> str:
        return f"http://{self.localforms_host}:{self.localforms_port}"

    def slurm_available(self) -> bool:
        return not self.force_local and shutil.which("sbatch") is not None

    def in_slurm_job(self) -> bool:
        return bool(os.environ.get("SLURM_JOB_ID"))

    def cache_env(self) -> Dict[str, str]:
        """Cache locations exported to every subprocess (setdefault semantics)."""
        hf_home = self.cache_root / "hf"
        return {
            "XDG_CACHE_HOME": str(self.cache_root / "xdg"),
            "HF_HOME": str(hf_home),
            "TRANSFORMERS_CACHE": str(hf_home / "transformers"),
            "PIP_CACHE_DIR": str(self.cache_root / "pip"),
            "UV_CACHE_DIR": str(self.cache_root / "uv"),
            "PLAYWRIGHT_BROWSERS_PATH": str(self.playwright_browsers_path),
        }

    def subprocess_env(self, extra: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
        """Environment for runner subprocesses: caches, PYTHONPATH, node tools on PATH."""
        env = dict(os.environ)
        for key, value in self.cache_env().items():
            env.setdefault(key, value)
        env["PYTHONUNBUFFERED"] = "1"
        env["LOCALFORMS_PORT"] = str(self.localforms_port)
        py_path = [str(SRC_DIR)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(py_path))
        path_parts = [str(self.node_tools_dir / "node_modules" / ".bin"), str(self.vllm_python_bin.parent)]
        env["PATH"] = os.pathsep.join(path_parts + [env.get("PATH", "")])
        executable_record = self.playwright_browsers_path / ".mcp-chromium-executable"
        if "PLAYWRIGHT_MCP_CHROMIUM_EXECUTABLE" not in env and executable_record.is_file():
            recorded = executable_record.read_text(encoding="utf-8").strip()
            if recorded and Path(recorded).exists():  # stale after a workspace move -> make setup
                env["PLAYWRIGHT_MCP_CHROMIUM_EXECUTABLE"] = recorded
        if extra:
            env.update({k: str(v) for k, v in extra.items()})
        return env


def load_settings(environ: Optional[Mapping[str, str]] = None, root: Optional[Path] = None) -> Settings:
    root = (root or REPO_ROOT).resolve()
    file_values = parse_env_file(root / ".env")
    env = dict(file_values)
    env.update(dict(os.environ if environ is None else environ))

    def get(key: str, default: str) -> str:
        value = env.get(key)
        return default if value is None or value == "" else value

    extra = get("SLURM_EXTRA_ARGS", "")
    return Settings(
        root=root,
        python_bin=_resolve(root, get("PYTHON_BIN", ".venv/bin/python")),
        vllm_python_bin=_resolve(root, get("VLLM_PYTHON_BIN", ".venv-opencua/bin/python")),
        cache_root=_resolve(root, get("CACHE_ROOT", ".runtime-cache")),
        models_config=_resolve(root, get("MODELS_CONFIG", "configs/models.json")),
        models_dir=_resolve(root, get("MODELS_DIR", "models")),
        dataset_root=_resolve(root, get("DATASET_ROOT", "data/model_baselines")),
        reference_root=_resolve(root, get("REFERENCE_ROOT", "data/forms")),
        forms_root=_resolve(root, get("FORMS_ROOT", "src/forms")),
        answers_root=_resolve(root, get("ANSWERS_ROOT", "data/answers")),
        localforms_forms_root=_resolve(root, get("LOCALFORMS_FORMS_ROOT", "src/forms_localforms")),
        localforms_answers_root=_resolve(root, get("LOCALFORMS_ANSWERS_ROOT", "data/answers_localforms")),
        localforms_reference_root=_resolve(root, get("LOCALFORMS_REFERENCE_ROOT", "data/forms_localforms")),
        localforms_host=get("LOCALFORMS_HOST", "127.0.0.1"),
        localforms_port=localforms_port(get("LOCALFORMS_PORT", "auto"), env),
        node_tools_dir=_resolve(root, get("NODE_TOOLS_DIR", ".node-tools")),
        logs_dir=_resolve(root, get("LOGS_DIR", "logs")),
        reports_dir=_resolve(root, get("REPORTS_DIR", "reports")),
        modules=get("MODULES", ""),
        gemini_api_key_file=_resolve(root, get("GEMINI_API_KEY_FILE", ".secrets/gemini_api_key")),
        slurm_account=get("SLURM_ACCOUNT", ""),
        slurm_partition=get("SLURM_PARTITION", ""),
        slurm_exclude=get("SLURM_EXCLUDE", ""),
        slurm_extra_args=extra.split() if extra else [],
        force_local=get("LOCAL", "0") == "1",
        raw=env,
    )
