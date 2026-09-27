"""`formbench data`: the dataset pipeline (all steps deterministic and idempotent).

    generator CSVs (data/generator/*.csv)
      -> src/forms/<id>/spec.json + data/specs/forms_master.csv   (dataset.sync_specs, only with --from-csv)
      -> data/answers/<id>/runs.json                               (dataset.answers, seed 0 = committed sets)
      -> src/forms_localforms + data/answers_localforms + Flask site (dataset.localforms)
      -> validation                                                (dataset.validate + dataset.integrity)
"""

from typing import List

from formbench.common import run
from formbench.settings import Settings


def _module(settings: Settings, module: str, *args: str) -> List[str]:
    return [str(settings.python_bin), "-m", module, *args]


def build(settings: Settings, from_csv: bool = False, seed: int = 0, runs_per_form: int = 10, prune: bool = False) -> int:
    env = settings.subprocess_env()
    steps: List[List[str]] = []
    if from_csv:
        steps.append(_module(settings, "dataset.sync_specs", *(["--prune"] if prune else [])))
    steps.append(_module(settings, "dataset.answers", "--rewrite", "--seed", str(seed), "--runs-per-form", str(runs_per_form)))
    steps.append(_module(settings, "dataset.localforms"))
    for argv in steps:
        code = run(argv, env=env, cwd=settings.root)
        if code != 0:
            return code
    return check(settings, runs_per_form)


def check(settings: Settings, runs_per_form: int = 10) -> int:
    env = settings.subprocess_env()
    code = run(_module(settings, "dataset.validate", "--strict", "--required-runs", str(runs_per_form)), env=env, cwd=settings.root)
    code2 = run(_module(settings, "dataset.integrity"), env=env, cwd=settings.root)
    return code or code2
