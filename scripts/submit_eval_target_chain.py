#!/usr/bin/env python3
"""Submit dependent SLURM jobs toward the active unique form-run target."""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
FORMS_ROOT = ROOT / "src" / "forms"
DATASET_ROOT = ROOT / "data" / "model_baselines"
QWEN_MODELS = {
    "text_qwen3_30b_a3b_instruct_2507",
    "vlm_qwen3_vl_30b_a3b_instruct",
}
OPENCUA_MODEL = "computer_use_opencua_32b"
OPENCUA_DIRECT_MCP_MODEL = "computer_use_opencua_32b_direct_mcp"
DEFAULT_TARGET_TRIALS = 300


def _form_ids() -> List[str]:
    return sorted(entry.name for entry in FORMS_ROOT.iterdir() if entry.is_dir() and (entry / "spec.json").exists())


def _parse_run_indexes(value: str) -> List[int]:
    indexes: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            indexes.extend(range(int(start_s), int(end_s) + 1))
        else:
            indexes.append(int(part))
    return sorted(dict.fromkeys(indexes))


def _observed_pairs() -> Dict[str, Set[Tuple[str, str]]]:
    observed: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for summary in sorted(DATASET_ROOT.glob("**/summary.json")):
        parts = summary.relative_to(DATASET_ROOT).parts
        if len(parts) < 5:
            continue
        _experiment, model_id, form_id, answer_run_id, _trial_id = parts[:5]
        observed[model_id].add((form_id, answer_run_id))
    return observed


def _target_run_indexes(target_trials: int, forms: Sequence[str], explicit_run_indexes: Optional[Sequence[int]] = None) -> List[int]:
    if explicit_run_indexes:
        return sorted(dict.fromkeys(int(idx) for idx in explicit_run_indexes))
    if not forms:
        return []
    run_count = max(1, math.ceil(max(1, int(target_trials)) / len(forms)))
    return list(range(1, run_count + 1))


def _missing_forms_for_qwen_run(run_index: int, forms: Sequence[str], observed: Dict[str, Set[Tuple[str, str]]]) -> List[str]:
    answer_run_id = f"run_{run_index:04d}"
    missing = []
    for form_id in forms:
        if any((form_id, answer_run_id) not in observed.get(model_id, set()) for model_id in QWEN_MODELS):
            missing.append(form_id)
    return missing


def _missing_forms_for_opencua_run(run_index: int, forms: Sequence[str], observed: Dict[str, Set[Tuple[str, str]]]) -> List[str]:
    answer_run_id = f"run_{run_index:04d}"
    return [form_id for form_id in forms if (form_id, answer_run_id) not in observed.get(OPENCUA_MODEL, set())]


def _missing_forms_for_opencua_direct_mcp_run(run_index: int, forms: Sequence[str], observed: Dict[str, Set[Tuple[str, str]]]) -> List[str]:
    answer_run_id = f"run_{run_index:04d}"
    return [form_id for form_id in forms if (form_id, answer_run_id) not in observed.get(OPENCUA_DIRECT_MCP_MODEL, set())]


def _submit(
    *,
    script: str,
    env_updates: Dict[str, str],
    dependency_job_id: Optional[str],
    dependency_type: str,
    exclude: str,
    dry_run: bool,
) -> Optional[str]:
    cmd = ["sbatch"]
    if dependency_job_id:
        cmd.append(f"--dependency={dependency_type}:{dependency_job_id}")
    if exclude:
        cmd.append(f"--exclude={exclude}")
    cmd.append(script)
    printable_env = " ".join(f"{key}={value}" for key, value in sorted(env_updates.items()))
    print(f"[PLAN] {printable_env} {' '.join(cmd)}")
    if dry_run:
        return dependency_job_id
    env = os.environ.copy()
    env.update(env_updates)
    result = subprocess.run(cmd, cwd=ROOT, env=env, check=True, text=True, capture_output=True)
    print(result.stdout.strip())
    match = re.search(r"Submitted batch job\s+(\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse sbatch job id from: {result.stdout!r}")
    return match.group(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Queue missing evaluation target batches with SLURM dependencies.")
    parser.add_argument("--run-indexes", default="", help="Optional explicit run indexes to consider, e.g. 1,3-6")
    parser.add_argument("--target-trials", type=int, default=DEFAULT_TARGET_TRIALS, help="Unique form-run target per model/interface condition")
    parser.add_argument("--qwen-after-job", default="", help="Submit first Qwen job after this SLURM job id")
    parser.add_argument("--opencua-after-job", default="", help="Submit first OpenCUA job after this SLURM job id")
    parser.add_argument("--opencua-direct-mcp-after-job", default="", help="Submit first OpenCUA direct-MCP job after this SLURM job id")
    parser.add_argument("--tracks", choices=["all", "qwen", "opencua", "opencua-direct-mcp"], default="all")
    parser.add_argument("--dependency-type", choices=["afterok", "afterany"], default="afterok")
    parser.add_argument("--exclude", default="i8033", help="Comma-separated node list to exclude when submitting GPU jobs")
    parser.add_argument("--submit", action="store_true", help="Actually call sbatch. Without this, only print the plan.")
    parser.add_argument("--date-stamp", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    args = parser.parse_args()

    forms = _form_ids()
    observed = _observed_pairs()
    explicit_run_indexes = _parse_run_indexes(args.run_indexes) if args.run_indexes.strip() else None
    run_indexes = _target_run_indexes(args.target_trials, forms, explicit_run_indexes)
    dry_run = not args.submit

    if args.tracks in {"all", "qwen"}:
        qwen_dep = args.qwen_after_job.strip() or None
        for run_index in run_indexes:
            missing_forms = _missing_forms_for_qwen_run(run_index, forms, observed)
            if not missing_forms:
                continue
            experiment_id = f"qwen_direct_mcp_target300_run{run_index}_{args.date_stamp}"
            qwen_dep = _submit(
                script="scripts/slurm_qwen_direct_mcp.sbatch",
                env_updates={
                    "EXPERIMENT_ID": experiment_id,
                    "CONFIG_PATH": "configs/baselines/track_baseline_models.json",
                    "FORM_IDS": ",".join(missing_forms),
                    "RUN_INDEXES": str(run_index),
                    "DIRECT_MCP_MAX_STEPS": "128",
                    "DIRECT_MCP_TIMEOUT_S": "9000",
                    "DIRECT_MCP_TEXT_MAX_NEW_TOKENS": "1024",
                    "DIRECT_MCP_VLM_MAX_NEW_TOKENS": "1024",
                    "SKIP_COMPLETED": "1",
                    "FAIL_ON_TRIAL_FAILURE": "0",
                },
                dependency_job_id=qwen_dep,
                dependency_type=args.dependency_type,
                exclude=args.exclude,
                dry_run=dry_run,
            )

    if args.tracks in {"all", "opencua"}:
        opencua_dep = args.opencua_after_job.strip() or None
        for run_index in run_indexes:
            missing_forms = _missing_forms_for_opencua_run(run_index, forms, observed)
            if not missing_forms:
                continue
            experiment_id = f"opencua_control_guidance_target300_run{run_index}_{args.date_stamp}"
            opencua_dep = _submit(
                script="scripts/slurm_opencua_direct.sbatch",
                env_updates={
                    "DIRECT_EXPERIMENT_ID": experiment_id,
                    "CONFIG_PATH": "configs/baselines/track_baseline_models.json",
                    "FORM_IDS": ",".join(missing_forms),
                    "RUN_INDEXES": str(run_index),
                    "DIRECT_MAX_STEPS": "128",
                    "DIRECT_TIMEOUT_S": "3600",
                    "DIRECT_MAX_NEW_TOKENS": "96",
                    "DIRECT_API_TIMEOUT_S": "180",
                    "OPENCUA_GPU_MEMORY_UTILIZATION": "0.85",
                    "OPENCUA_MAX_MODEL_LEN": "16384",
                    "FAIL_ON_TRIAL_FAILURE": "0",
                },
                dependency_job_id=opencua_dep,
                dependency_type=args.dependency_type,
                exclude=args.exclude,
                dry_run=dry_run,
            )

    if args.tracks in {"all", "opencua-direct-mcp"}:
        opencua_direct_dep = args.opencua_direct_mcp_after_job.strip() or None
        for run_index in run_indexes:
            missing_forms = _missing_forms_for_opencua_direct_mcp_run(run_index, forms, observed)
            if not missing_forms:
                continue
            experiment_id = f"opencua_direct_mcp_tools_target300_run{run_index}_{args.date_stamp}"
            opencua_direct_dep = _submit(
                script="scripts/slurm_opencua_direct_mcp.sbatch",
                env_updates={
                    "EXPERIMENT_ID": experiment_id,
                    "CONFIG_PATH": "configs/baselines/track_baseline_models.json",
                    "MODEL_ID": OPENCUA_DIRECT_MCP_MODEL,
                    "FORM_IDS": ",".join(missing_forms),
                    "RUN_INDEXES": str(run_index),
                    "DIRECT_MCP_MAX_STEPS": "128",
                    "DIRECT_MCP_TIMEOUT_S": "9000",
                    "DIRECT_MCP_MAX_NEW_TOKENS": "1024",
                    "SKIP_COMPLETED": "1",
                    "FAIL_ON_TRIAL_FAILURE": "0",
                },
                dependency_job_id=opencua_direct_dep,
                dependency_type=args.dependency_type,
                exclude=args.exclude,
                dry_run=dry_run,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
