"""Render and submit Slurm jobs for experiments (replaces the per-model .sbatch files).

Resources come from the registry `resources` block (max over the models in a job,
since a job runs its models one after another). Account/partition/excludes come
from `.env`. With `--split`, one job per cohort/model/run index is created and
optionally chained with `--dependency=afterok:<previous>`.
"""

import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from baselines.model_registry import get_model_by_id
from formbench.common import FormbenchError, info, utc_now
from formbench.matrix import Experiment
from formbench.settings import Settings

DEFAULT_RESOURCES = {"gpus": 0, "cpus": 4, "mem": "16G", "time": "04:00:00"}
SPLIT_MODES = {"none", "cohort", "model", "run"}


def _mem_mb(value: str) -> int:
    match = re.fullmatch(r"(\d+)\s*([KMGT]?)B?", str(value).strip().upper())
    if not match:
        raise FormbenchError(f"cannot parse memory {value!r} (use e.g. 120G)")
    number, unit = int(match.group(1)), match.group(2) or "M"
    return number * {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[unit]


def _time_s(value: str) -> int:
    text = str(value).strip()
    if not re.fullmatch(r"(\d+-)?\d+(:\d+){0,2}", text):
        raise FormbenchError(f"cannot parse time {value!r} (use e.g. 04:00:00 or 1-00:00:00)")
    days = 0
    if "-" in text:
        day_part, text = text.split("-", 1)
        days = int(day_part)
    parts = [int(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts[-3:]
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def job_resources(settings: Settings, model_ids: Sequence[str]) -> Dict[str, Any]:
    res = dict(DEFAULT_RESOURCES)
    for model_id in model_ids:
        model = get_model_by_id(settings.models_config, model_id)
        mres = dict(model.get("resources") or {})
        res["gpus"] = max(int(res["gpus"]), int(mres.get("gpus", 0)))
        res["cpus"] = max(int(res["cpus"]), int(mres.get("cpus", 0)))
        if mres.get("mem") and _mem_mb(mres["mem"]) > _mem_mb(res["mem"]):
            res["mem"] = mres["mem"]
        if mres.get("time") and _time_s(mres["time"]) > _time_s(res["time"]):
            res["time"] = mres["time"]
    return res


@dataclass
class JobPlan:
    name: str
    formbench_args: List[str]
    models: List[str]
    resources: Dict[str, Any] = field(default_factory=dict)
    script_path: Optional[Path] = None


VALUE_FLAGS = {"--models", "--runs", "--cohort"}


def merge_flags(base: List[str], extra: List[str]) -> List[str]:
    """Append `extra` to `base`, dropping any value flag in `base` that `extra` sets again."""
    override = {extra[i] for i in range(len(extra) - 1) if extra[i] in VALUE_FLAGS}
    merged: List[str] = []
    skip = False
    for i, token in enumerate(base):
        if skip:
            skip = False
            continue
        if token in override:
            skip = True
            continue
        merged.append(token)
    return merged + extra


def plan_jobs(settings: Settings, experiment: Experiment, split: str, base_args: List[str], time_limit: Optional[str] = None) -> List[JobPlan]:
    if split not in SPLIT_MODES:
        raise FormbenchError(f"--split must be one of {sorted(SPLIT_MODES)}")
    if time_limit:
        _time_s(time_limit)  # validate early
    jobs: List[JobPlan] = []
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", experiment.id)[:40]
    from_manifest = bool(base_args) and not base_args[0].startswith("-")

    def add(suffix: str, extra: List[str], models: List[str]) -> None:
        if not from_manifest:  # ad-hoc runs have a single implicit cohort
            extra = [tok for i, tok in enumerate(extra) if tok != "--cohort" and (i == 0 or extra[i - 1] != "--cohort")]
        name = f"fb-{safe}" + (f"-{suffix}" if suffix else "")
        resources = job_resources(settings, models)
        if time_limit:
            resources["time"] = time_limit
        jobs.append(JobPlan(name=name, formbench_args=["matrix"] + merge_flags(base_args, extra), models=models, resources=resources))

    if split == "none":
        models = sorted({m for c in experiment.cohorts for m in c.models})
        add("", [], models)
        return jobs
    for cohort in experiment.cohorts:
        if split == "cohort":
            add(cohort.name, ["--cohort", cohort.name], cohort.models)
            continue
        for model_id in cohort.models:
            if split == "model":
                add(f"{cohort.name}-{model_id}"[:60], ["--cohort", cohort.name, "--models", model_id], [model_id])
                continue
            for run_index in cohort.run_indexes:
                add(f"{model_id}-r{run_index}"[:60], ["--cohort", cohort.name, "--models", model_id, "--runs", str(run_index)], [model_id])
    return jobs


def cpu_job(name: str, formbench_args: List[str], resources: Optional[Mapping[str, Any]] = None) -> JobPlan:
    res = dict(DEFAULT_RESOURCES)
    res.update({"cpus": 4, "mem": "16G", "time": "24:00:00"})
    res.update(dict(resources or {}))
    return JobPlan(name=name, formbench_args=formbench_args, models=[], resources=res)


def render_script(settings: Settings, job: JobPlan) -> str:
    res = job.resources
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job.name}",
        f"#SBATCH --output={settings.logs_dir}/slurm/%x-%j.out",
        f"#SBATCH --error={settings.logs_dir}/slurm/%x-%j.err",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={res['cpus']}",
        f"#SBATCH --mem={res['mem']}",
        f"#SBATCH --time={res['time']}",
    ]
    if int(res["gpus"]) > 0:
        lines.append(f"#SBATCH --gres=gpu:{res['gpus']}")
    if settings.slurm_account:
        lines.append(f"#SBATCH --account={settings.slurm_account}")
    if settings.slurm_partition:
        lines.append(f"#SBATCH --partition={settings.slurm_partition}")
    if settings.slurm_exclude:
        lines.append(f"#SBATCH --exclude={settings.slurm_exclude}")
    matrix_cmd = " ".join(shlex.quote(a) for a in [str(settings.root / "scripts" / "env.sh"), str(settings.python_bin), "-m", "formbench"] + job.formbench_args)
    lines += [
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(str(settings.root))}",
        'echo "[INFO] job_id=${SLURM_JOB_ID:-na} host=$(hostname) start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"',
        "nvidia-smi || true" if int(res["gpus"]) > 0 else "",
        "export LOCAL=1  # already inside the allocation: run here, do not resubmit",
        matrix_cmd,
        'echo "[INFO] end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"',
        "",
    ]
    return "\n".join(line for line in lines if line is not None)


def submit(settings: Settings, jobs: List[JobPlan], chain: Optional[str], after: Optional[str], dry_run: bool) -> List[str]:
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    out_dir = settings.logs_dir / "slurm" / "jobs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (settings.logs_dir / "slurm").mkdir(parents=True, exist_ok=True)
    job_ids: List[str] = []
    previous = after
    for idx, job in enumerate(jobs):
        job.script_path = out_dir / f"{job.name}-{stamp}-{idx:02d}.sbatch"
        job.script_path.write_text(render_script(settings, job), encoding="utf-8")
        cmd = ["sbatch", "--parsable"] + settings.slurm_extra_args
        if previous:
            cmd.append(f"--dependency={chain or 'afterok'}:{previous}")
        cmd.append(str(job.script_path))
        res = job.resources
        info(f"job {job.name}: gpus={res['gpus']} cpus={res['cpus']} mem={res['mem']} time={res['time']} models={','.join(job.models)}")
        if dry_run:
            print(f"# {' '.join(shlex.quote(c) for c in cmd)}")
            print(job.script_path.read_text(encoding="utf-8"))
            previous = f"<job{idx}>" if chain else after
            continue
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(settings.root))
        if result.returncode != 0:
            raise FormbenchError(f"sbatch failed: {result.stderr.strip() or result.stdout.strip()}", hint="check SLURM_ACCOUNT / SLURM_PARTITION in .env")
        job_id = result.stdout.strip().split(";")[0]
        job_ids.append(job_id)
        info(f"submitted {job.name} as job {job_id} (logs: {settings.logs_dir}/slurm/{job.name}-{job_id}.out)")
        previous = job_id if chain else after
    return job_ids
