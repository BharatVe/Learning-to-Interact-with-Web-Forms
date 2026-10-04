"""`formbench report`: analytics as CSV tables + SVG plots.

    make report             current analytics into reports/ (gitignored): core thesis report,
                            all-experiment overview, results tracker
    make reference-report   ideal (scripted reference) run summary into reports/reference/
    make studies / study    list / regenerate one committed thesis study (analysis.registry),
                            incl. core_report / reference / results_tracker under docs/eval_results

`make report` never touches committed files: docs/eval_results/** are thesis records
(hashed in evaluation_additions/manifest.json, read by other studies) and change only
through `make study NAME=...`.
"""

from pathlib import Path
from typing import List, Optional

from formbench.common import FormbenchError, dump_table, info, run
from formbench.settings import Settings


def _python(settings: Settings, *args: str) -> List[str]:
    return [str(settings.python_bin), *args]


def _run_all(settings: Settings, commands: List[List[str]]) -> int:
    env = settings.subprocess_env({"MPLCONFIGDIR": str(settings.cache_root / "matplotlib")})
    for argv in commands:
        code = run(argv, env=env, cwd=settings.root)
        if code != 0:
            return code
    return 0


def run_core(settings: Settings, experiments: Optional[List[str]] = None, output_dir: Optional[str] = None, extra: Optional[List[str]] = None) -> int:
    out = Path(output_dir) if output_dir else settings.reports_dir
    exp_flags: List[str] = []
    for exp in experiments or []:
        exp_flags += ["--experiment-id", exp]
    commands = []
    if not experiments:
        # The core report's cohorts are fixed by configs/analysis/core_report.json, so it always reads everything.
        commands.append(_python(settings, "-m", "analysis.core_report", "--dataset-root", str(settings.dataset_root), "--output-dir", str(out), *(extra or [])))
    commands.append(_python(settings, "-m", "analysis.overview", "--dataset-root", str(settings.dataset_root), "--output-dir", str(out), *exp_flags))
    tracker_out = out
    commands.append(_python(settings, "-m", "analysis.tracker", "--dataset-root", str(settings.dataset_root), "--output-dir", str(tracker_out), *exp_flags))
    code = _run_all(settings, commands)
    if code == 0:
        info(f"analytics written to {out} (overview: experiment_overview.csv, plots/); tracker: {tracker_out}/metrics.csv")
    return code


def run_reference(settings: Settings) -> int:
    out = settings.reports_dir / "reference"
    return _run_all(settings, [_python(settings, "-m", "analysis.reference", "--dataset-root", str(settings.reference_root),
                                       "--forms-root", str(settings.forms_root), "--output-dir", str(out))])


def print_studies() -> None:
    from analysis.registry import STUDIES

    rows = [{"name": s.name, "check": "yes" if s.check_command else "-", "description": s.description} for s in STUDIES.values()]
    print(dump_table(rows, ["name", "check", "description"]))
    info("regenerate with: make study NAME=<name>   (then `git diff --stat` shows what changed)")


def run_study(settings: Settings, name: str, extra: Optional[List[str]] = None, check: bool = False) -> int:
    from analysis.registry import STUDIES

    if name not in STUDIES:
        raise FormbenchError(f"unknown study {name!r}", hint="make studies")
    study = STUDIES[name]
    if check:
        if not study.check_command:
            raise FormbenchError(f"study {name} has no check mode", hint="regenerate it and inspect `git diff`")
        return _run_all(settings, [_python(settings, *study.check_command)])
    commands = [_python(settings, *cmd) for cmd in study.commands]
    if extra:
        commands[-1] += extra
    if study.check_command and study.check_command not in study.commands:
        commands.append(_python(settings, *study.check_command))
    code = _run_all(settings, commands)
    if code == 0:
        info(f"study {name} regenerated: {', '.join(study.outputs)}")
    return code
