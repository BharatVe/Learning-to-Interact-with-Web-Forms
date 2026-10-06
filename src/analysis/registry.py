"""Thesis studies that `make study NAME=<name>` regenerates.

Each study is a list of commands (python module or script + args, run from the repo
root) and the paths it writes. Outputs are committed, so `git diff --stat` after a
re-run shows exactly what changed.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Study:
    name: str
    description: str
    commands: List[List[str]]
    outputs: List[str]
    check_command: List[str] = field(default_factory=list)


_IFA = "docs/eval_results/interaction_failure_analysis"

STUDIES: Dict[str, Study] = {
    study.name: study
    for study in [
        Study(
            "core_report",
            "Core thesis report (Qwen text/VL + OpenCUA native/MCP, 50 forms x runs 1-6): tables + thesis plots.",
            [["-m", "analysis.core_report", "--output-dir", "docs/eval_results/analysis"]],
            ["docs/eval_results/analysis/"],
        ),
        Study(
            "reference",
            "Ideal (scripted Playwright) reference runs: coverage, action breakdown, durations.",
            [["-m", "analysis.reference", "--output-dir", "docs/eval_results/reference_analysis"]],
            ["docs/eval_results/reference_analysis/"],
        ),
        Study(
            "form_length_position",
            "Accuracy by form length and question position (fill-only cohorts, dropdowns excluded).",
            [["-m", "analysis.studies.form_length_position"]],
            ["docs/eval_results/form_length_position_analysis/"],
        ),
        Study(
            "formfactory_qwen3vl",
            "Qwen3-VL: direct Playwright MCP vs FormFactory-style screenshot/coordinates, submit and fill-only.",
            [["-m", "analysis.studies.formfactory_qwen3vl"]],
            ["data/model_baseline_exports/formfactory_qwen3vl_*", "docs/eval_results/FORMFACTORY_QWEN3VL_RESULTS.md"],
        ),
        Study(
            "opencua_ruler",
            "OpenCUA FormFactory-style condition with vs without the pixel-ruler overlay (paired forms/fields).",
            [["-m", "analysis.studies.opencua_ruler"]],
            ["evaluation_additions/opencua_ruler_comparison/"],
            check_command=["-m", "analysis.studies.opencua_ruler", "--check"],
        ),
        Study(
            "localforms_comparison",
            "Platform comparison: OpenCUA direct MCP on LocalForms vs the same forms on Google Forms.",
            [["-m", "analysis.studies.localforms_comparison"]],
            ["data/localforms_comparison_analysis/"],
        ),
        Study(
            "fill_only_done_50_export",
            "Canonical 50-form fill-only/DONE export (Gemini, Qwen text/VL, OpenCUA MCP) used by later analyses.",
            [["-m", "analysis.studies.fill_only_done_50_export"]],
            ["data/model_baseline_exports/fill_only_done_50_20260714/"],
        ),
        Study(
            "missing_fill_only_additions",
            "Field/trial exports for the fill-only runs added after the 50-form export.",
            [["-m", "analysis.studies.missing_fill_only_additions"]],
            ["evaluation_additions/missing_fill_only_runs/"],
            check_command=["-m", "analysis.studies.missing_fill_only_additions", "--check"],
        ),
        Study(
            "interaction_failure",
            "Action/failure analysis incl. the dropdown selected-state audit and the HTML report artifact.",
            [
                [f"{_IFA}/analyze_action_failures.py", "--project-root", ".",
                 "--trials-csv", "data/model_baseline_exports/fill_only_done_50_20260714/trials.csv",
                 "--baseline-actions-csv", "docs/eval_results/analysis/model_action_trial_counts.csv",
                 "--reference-runs-csv", "docs/eval_results/reference_analysis/reference_runs.csv",
                 "--reference-actions-csv", "docs/eval_results/reference_analysis/reference_action_breakdown.csv",
                 "--output-dir", f"{_IFA}/data"],
                [f"{_IFA}/audit_dropdown_selected_state.py", "--project-root", ".",
                 "--field-outcomes", f"{_IFA}/data/field_outcomes.csv", "--output-dir", f"{_IFA}/data"],
                [f"{_IFA}/build_report_artifact.py"],
            ],
            [f"{_IFA}/data/", f"{_IFA}/artifact.json", f"{_IFA}/report.html"],
            check_command=[f"{_IFA}/validate_outputs.py"],
        ),
    ]
}
