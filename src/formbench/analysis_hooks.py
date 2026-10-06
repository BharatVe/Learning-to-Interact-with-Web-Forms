"""Post-run summaries written after a matrix finishes (what the shell matrices did at exit)."""

import subprocess
from typing import TYPE_CHECKING

from formbench.common import info, warn
from formbench.settings import Settings

if TYPE_CHECKING:  # pragma: no cover
    from formbench.matrix import Experiment


def _run_quiet(settings: Settings, argv: list) -> None:
    info("$ " + " ".join(argv))
    code = subprocess.run(argv, env=settings.subprocess_env(), cwd=str(settings.root)).returncode
    if code != 0:
        warn(f"post-run summary failed (exit {code}); trial data is unaffected")


def post_matrix(settings: Settings, experiment: "Experiment", update_tracker: bool = True) -> None:
    from formbench.protocols import protocol_for

    python = str(settings.python_bin)
    seen = set()
    for cohort in experiment.cohorts:
        exp_id = cohort.experiment_id
        if exp_id in seen:
            continue
        seen.add(exp_id)
        _run_quiet(settings, [
            python, "-m", "analysis.reference_efficiency", "--dataset-root", str(settings.dataset_root),
            "--experiment-id", exp_id, "--output", str(settings.logs_dir / f"{exp_id}_reference_efficiency_summary.json"),
        ])
        mediated = [m for m in cohort.models if protocol_for(_model(settings, m)).name == "mediated"]
        if mediated:
            protocol_value = str(cohort.args.get("interaction_protocol") or "human_ui_v1")
            _run_quiet(settings, [
                python, "-m", "analysis.human_ui_attribution", "--dataset-root", str(settings.dataset_root),
                "--experiment-id", exp_id, "--interaction-protocol", protocol_value,
                "--expected-forms", str(len(cohort.forms)), "--expected-runs-per-form", str(len(cohort.run_indexes)),
                "--expected-models", str(len(mediated)),
                "--output", str(settings.logs_dir / f"{exp_id}_human_ui_attribution.json"),
            ])
        if update_tracker:
            _run_quiet(settings, [python, "-m", "analysis.tracker", "--dataset-root", str(settings.dataset_root), "--output-dir", str(settings.reports_dir), "--experiment-id", exp_id])


def _model(settings: Settings, model_id: str) -> dict:
    from baselines.model_registry import get_model_by_id

    return get_model_by_id(settings.models_config, model_id)
