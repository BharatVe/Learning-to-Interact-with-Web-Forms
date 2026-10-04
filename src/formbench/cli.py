"""formbench: one entry point for setup, data, ideal runs, model evals and analytics.

Run `python -m formbench <command> --help` (or the matching `make` target) for details.
"""

import argparse
import subprocess
import sys
from typing import List, Optional

from formbench.common import FormbenchError, dump_table, fail, info, parse_csv_list, parse_run_indexes, warn
from formbench.settings import Settings, load_settings


# ----------------------------------------------------------------------------- helpers

def _add_selection(p: argparse.ArgumentParser, experiment_optional: bool = True) -> None:
    p.add_argument("experiment", nargs="?" if experiment_optional else None, help="experiment manifest name in configs/experiments/ (omit for an ad-hoc run)")
    p.add_argument("--models", help="comma-separated model ids (filters a manifest, or defines an ad-hoc run)")
    p.add_argument("--forms", help="comma-separated form ids or 'all' (overrides the manifest)")
    p.add_argument("--runs", help="answer-set indexes, e.g. 2 or 1-3 or 1,4 (overrides the manifest)")
    p.add_argument("--cohort", action="append", default=[], help="only run this cohort (name or experiment_id); repeatable")
    p.add_argument("--experiment-id", help="output experiment id for an ad-hoc run (default: adhoc_<model>_<date>)")
    p.add_argument("--experiment-suffix", help="append to every cohort's experiment_id, e.g. _rerun1 (fresh replication instead of resuming)")
    p.add_argument("--platform", choices=["google", "localforms"], help="form platform (default google)")
    p.add_argument("--budget", help="budget profile for mediated models (balanced|large|xlarge|large_qwen3)")
    p.add_argument("--skip-completed", choices=["experiment", "any", "none"], help="skip form/run pairs that already have a summary.json")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="override a runner flag, e.g. --set max_steps=32 (repeatable)")


def _selection_argv(args: argparse.Namespace) -> List[str]:
    """Re-serialise selection flags (for Slurm job scripts)."""
    out: List[str] = [args.experiment] if getattr(args, "experiment", None) else []
    for flag in ("models", "forms", "runs", "experiment_id", "experiment_suffix", "platform", "budget", "skip_completed"):
        value = getattr(args, flag, None)
        if value:
            out += ["--" + flag.replace("_", "-"), str(value)]
    for cohort in args.cohort:
        out += ["--cohort", cohort]
    for pair in args.overrides:
        out += ["--set", pair]
    return out


def _load_selection(settings: Settings, args: argparse.Namespace):
    from formbench.matrix import Cohort, Experiment, adhoc_experiment, apply_cohort_filters, load_experiment, _resolve_forms

    models = parse_csv_list(args.models)
    if args.experiment:
        experiment = load_experiment(settings, args.experiment)
        experiment = apply_cohort_filters(experiment, args.cohort, models)
        cohorts = []
        for cohort in experiment.cohorts:
            data = dict(cohort.__dict__)
            if args.forms:
                data["forms"] = _resolve_forms(settings, args.forms)
            if args.runs:
                data["run_indexes"] = parse_run_indexes(args.runs)
            if args.platform:
                data["platform"] = args.platform
            if args.budget:
                data["budget_profile"] = args.budget
            if args.skip_completed:
                data["skip_completed"] = args.skip_completed
            if args.experiment_id:
                if len(experiment.cohorts) > 1:
                    raise FormbenchError("--experiment-id would merge several cohorts into one folder", hint="use --experiment-suffix, or select one cohort with --cohort")
                data["experiment_id"] = args.experiment_id
            if getattr(args, "experiment_suffix", None):
                data["experiment_id"] = data["experiment_id"] + args.experiment_suffix
            cohorts.append(Cohort(**data))
        return Experiment(id=experiment.id, description=experiment.description, cohorts=cohorts, source=experiment.source)
    if not models:
        raise FormbenchError("give an experiment manifest or --models", hint="make experiments   (lists manifests)   |   make models   (lists model ids)")
    from formbench.common import utc_now

    exp_id = args.experiment_id or f"adhoc_{models[0] if len(models) == 1 else 'multi'}_{utc_now().strftime('%Y%m%d')}"
    return adhoc_experiment(
        settings, exp_id, models, args.forms or "conf_interest", args.runs or "1",
        platform=args.platform or "google", skip_completed=args.skip_completed or "experiment", budget_profile=args.budget,
    )


def _preflight(settings: Settings, experiment, dry_run: bool, skip: bool) -> None:
    from formbench.checks import check_models

    if skip:
        warn("--skip-checks: model preflight skipped")
        return
    model_ids = sorted({m for c in experiment.cohorts for m in c.models})
    ok = check_models(settings, model_ids, probe_endpoint=True, expect_gpus_here=not dry_run)
    if not ok and not dry_run:
        raise FormbenchError("model preflight failed (see table above)", hint="fix the FAIL rows, submit to Slurm with `make submit`, or pass --skip-checks")
    if not ok:
        warn("preflight has failures; a real run would stop here")


# ----------------------------------------------------------------------------- commands

def cmd_setup(settings: Settings, args: argparse.Namespace) -> int:
    argv = ["bash", str(settings.root / "scripts" / "setup.sh")]
    if args.with_extra:
        argv += ["--with", args.with_extra]
    if args.skip_browsers:
        argv.append("--skip-browsers")
    return subprocess.run(argv, cwd=str(settings.root)).returncode


def cmd_doctor(settings: Settings, args: argparse.Namespace) -> int:
    from formbench.checks import doctor

    return 0 if doctor(settings) else 1


def cmd_models_list(settings: Settings, args: argparse.Namespace) -> int:
    from baselines.model_registry import local_weights_dir, validate_config
    from formbench.protocols import protocol_for

    models, errors, _ = validate_config(settings.models_config)
    rows = []
    for m in models:
        if m.get("status") == "legacy" and not args.all:
            continue
        try:
            protocol = protocol_for(m).name
        except FormbenchError:
            protocol = "?"
        needs = []
        if m.get("requires_gpu"):
            needs.append(f"{(m.get('resources') or {}).get('gpus', '?')} GPU")
        if m.get("provider") in {"gemini_low_cost", "api_over_mcp"}:
            needs.append("API key")
        weights = "local" if local_weights_dir(m, settings.root, settings.models_dir) else ("remote" if m.get("provider") in {"gemini_low_cost", "api_over_mcp"} else "download")
        rows.append({"id": m["id"], "kind": m.get("kind"), "protocol": protocol, "needs": ", ".join(needs) or "-", "weights": weights, "status": m.get("status", "current")})
    print(dump_table(rows, ["id", "kind", "protocol", "needs", "weights", "status"]))
    hidden = sum(1 for m in models if m.get("status") == "legacy") if not args.all else 0
    if hidden:
        info(f"{hidden} legacy model(s) hidden; show with --all (make models ALL=1)")
    if errors:
        warn(f"{len(errors)} registry error(s); run `make model-check MODEL=<id>`")
    return 0


def cmd_models_check(settings: Settings, args: argparse.Namespace) -> int:
    from baselines.model_registry import list_models
    from formbench.checks import check_models

    ids = parse_csv_list(",".join(args.model)) if args.model else []
    if args.all or not ids:
        ids = [m["id"] for m in list_models(settings.models_config) if args.all or m.get("status", "current") == "current"]
    return 0 if check_models(settings, ids, probe_endpoint=True, smoke=args.smoke, expect_gpus_here=args.here) else 1


def cmd_models_serve(settings: Settings, args: argparse.Namespace) -> int:
    import time

    from baselines.model_registry import get_model_by_id
    from formbench.serve import VLLMServer, endpoint_for

    model = get_model_by_id(settings.models_config, args.model)
    endpoint = endpoint_for(model)
    if not endpoint.managed:
        raise FormbenchError(f"{args.model} has no local serve block (endpoint {endpoint.base_url or 'n/a'})")
    with VLLMServer(settings, model, endpoint) as server:
        info(f"serving {args.model} at {endpoint.base_url} as {endpoint.model_name!r}; Ctrl-C to stop")
        info(f"in another shell: OPENAI_BASE_URL={endpoint.base_url} make eval MODEL={args.model} ...  (reused automatically)")
        try:
            while server.process is None or server.process.poll() is None:
                time.sleep(5)
        except KeyboardInterrupt:
            info("stopping")
    return 0


def cmd_models_install(settings: Settings, args: argparse.Namespace) -> int:
    from formbench.install import install_models

    return install_models(settings, parse_csv_list(",".join(args.model)) if args.model else [], dry_run=args.dry_run, force=args.force)


def cmd_data_build(settings: Settings, args: argparse.Namespace) -> int:
    from formbench import data

    return data.build(settings, from_csv=args.from_csv, seed=args.seed, runs_per_form=args.runs_per_form, prune=args.prune)


def cmd_data_check(settings: Settings, args: argparse.Namespace) -> int:
    from formbench import data

    return data.check(settings)


def cmd_data_relpaths(settings: Settings, args: argparse.Namespace) -> int:
    from formbench import data

    return data.relpaths(settings, apply=args.apply)


def cmd_forms_serve(settings: Settings, args: argparse.Namespace) -> int:
    from formbench.forms_site import serve_command

    info(f"LocalForms at {settings.localforms_base_url} (Ctrl-C to stop)")
    return subprocess.run(serve_command(settings), env=settings.subprocess_env(), cwd=str(settings.root)).returncode


def cmd_ideal(settings: Settings, args: argparse.Namespace) -> int:
    from contextlib import ExitStack

    from formbench import ideal
    from formbench.common import discover_form_ids

    platform = args.platform or "google"
    specs_root, _, _ = ideal.roots(settings, platform)
    forms = discover_form_ids(specs_root) if not args.forms or args.forms == "all" else parse_csv_list(args.forms)
    if platform == "localforms":
        forms = [f if f.startswith("lf_") else f"lf_{f}" for f in forms]
    if args.status:
        ideal.print_coverage(settings, platform, forms)
        return 0
    if args.submit:
        from formbench.slurm import cpu_job, submit

        job_args = ["ideal", "--platform", platform, "--forms", ",".join(forms)]
        if args.runs:
            job_args += ["--runs", args.runs]
        if args.overwrite:
            job_args.append("--overwrite")
        submit(settings, [cpu_job(f"fb-ideal-{platform}", job_args)], chain=None, after=None, dry_run=args.dry_run)
        return 0
    if settings.slurm_available() and not settings.in_slurm_job() and not args.dry_run:
        warn("running on a login node; long generations may be killed. Use SUBMIT=1 (make ideal-runs SUBMIT=1) to run as a Slurm job.")
    with ExitStack() as stack:
        if platform == "localforms" and not args.dry_run:
            from formbench.forms_site import LocalFormsSite

            stack.enter_context(LocalFormsSite(settings))
        failures = ideal.generate(
            settings, forms, parse_run_indexes(args.runs) if args.runs else [], platform=platform, overwrite=args.overwrite,
            headed=args.headed, interaction_mode=args.interaction_mode, screenshots=args.screenshots, dry_run=args.dry_run,
        )
    if not args.dry_run:
        ideal.print_coverage(settings, platform, forms)
    return 1 if failures else 0


def cmd_matrix(settings: Settings, args: argparse.Namespace) -> int:
    from formbench.matrix import MatrixRunner, parse_overrides

    experiment = _load_selection(settings, args)
    _preflight(settings, experiment, args.dry_run, args.skip_checks)
    runner = MatrixRunner(settings, experiment, parse_overrides(args.overrides), dry_run=args.dry_run, fail_fast=args.fail_fast, update_tracker=not args.no_tracker)
    return runner.run()


def cmd_eval(settings: Settings, args: argparse.Namespace) -> int:
    args.experiment = None
    args.models = args.model
    args.forms = args.form
    args.runs = str(args.run)
    args.cohort = []
    args.skip_completed = "none"
    args.budget = getattr(args, "budget", None)
    if args.submit:
        args.experiment_id = args.experiment_id or None
        return cmd_submit(settings, args)
    return cmd_matrix(settings, args)


def cmd_submit(settings: Settings, args: argparse.Namespace) -> int:
    from formbench.slurm import plan_jobs, submit

    experiment = _load_selection(settings, args)
    if not settings.slurm_available() and not args.dry_run:
        raise FormbenchError("sbatch not found (or LOCAL=1)", hint="run locally with `make matrix ...` instead")
    base = _selection_argv(args)
    if not args.experiment and not args.experiment_id:
        base += ["--experiment-id", experiment.cohorts[0].experiment_id]  # pin the dated id for every job
    if getattr(args, "skip_checks", False):
        base.append("--skip-checks")
    jobs = plan_jobs(settings, experiment, getattr(args, "split", "none") or "none", base)
    submit(settings, jobs, chain=getattr(args, "chain", None), after=getattr(args, "after", None), dry_run=args.dry_run)
    return 0


def cmd_experiments(settings: Settings, args: argparse.Namespace) -> int:
    from formbench.matrix import list_experiments, load_experiment

    rows = []
    for name, description in list_experiments(settings):
        try:
            exp = load_experiment(settings, name)
            trials = sum(len(c.models) * len(c.forms) * len(c.run_indexes) for c in exp.cohorts)
            rows.append({"name": name, "cohorts": len(exp.cohorts), "trials": trials, "description": description[:90]})
        except FormbenchError as exc:
            rows.append({"name": name, "cohorts": "!", "trials": "!", "description": str(exc)[:90]})
    print(dump_table(rows, ["name", "cohorts", "trials", "description"]))
    return 0


def cmd_report(settings: Settings, args: argparse.Namespace) -> int:
    from formbench import report

    if args.list_studies:
        report.print_studies()
        return 0
    if args.study:
        return report.run_study(settings, args.study, extra=args.extra, check=args.check)
    if args.reference:
        return report.run_reference(settings)
    return report.run_core(settings, experiments=parse_csv_list(args.experiments), output_dir=args.output_dir, extra=args.extra)


def cmd_inspect(settings: Settings, args: argparse.Namespace) -> int:
    from formbench import inspect_trial

    return inspect_trial.inspect(settings, args.trial, steps=args.steps, media=args.media)


# ----------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="formbench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    p = sub.add_parser("setup", help="create venv, install Python/Node deps and Chromium (idempotent)")
    p.add_argument("--with", dest="with_extra", choices=["localhf", "vllm", "all"], help="also install in-process HF deps and/or the vLLM env (.venv-opencua)")
    p.add_argument("--skip-browsers", action="store_true")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("doctor", help="check the environment and print fixes")
    p.set_defaults(func=cmd_doctor)

    models = sub.add_parser("models", help="list / check / serve / install models").add_subparsers(dest="models_cmd", required=True, metavar="<action>")
    p = models.add_parser("list", help="registered models")
    p.add_argument("--all", action="store_true", help="include legacy models")
    p.set_defaults(func=cmd_models_list)
    p = models.add_parser("check", help="preflight: registry, keys, weights, endpoint, GPUs")
    p.add_argument("model", nargs="*", help="model id(s); default: all current models")
    p.add_argument("--all", action="store_true", help="include legacy models")
    p.add_argument("--smoke", action="store_true", help="send a tiny chat request to running endpoints")
    p.add_argument("--here", action="store_true", help="treat missing GPUs on this node as an error")
    p.set_defaults(func=cmd_models_check)
    p = models.add_parser("serve", help="start a registry model's vLLM server in the foreground")
    p.add_argument("model")
    p.set_defaults(func=cmd_models_serve)
    p = models.add_parser("install", help="download local weights into models/<id>")
    p.add_argument("model", nargs="*", help="model id(s); default: all current models with local weights")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="re-download even if the directory exists")
    p.set_defaults(func=cmd_models_install)

    data = sub.add_parser("data", help="dataset pipeline").add_subparsers(dest="data_cmd", required=True, metavar="<action>")
    p = data.add_parser("build", help="regenerate answers + LocalForms (and specs with --from-csv), then validate")
    p.add_argument("--from-csv", action="store_true", help="also re-sync src/forms specs from data/generator/*.csv")
    p.add_argument("--prune", action="store_true", help="with --from-csv: delete specs not in the CSV")
    p.add_argument("--seed", type=int, default=0, help="answer seed (0 reproduces the committed answer sets)")
    p.add_argument("--runs-per-form", type=int, default=10)
    p.set_defaults(func=cmd_data_build)
    p = data.add_parser("check", help="validate specs, answer sets and committed reference traces")
    p.set_defaults(func=cmd_data_check)
    p = data.add_parser("relpaths", help="make absolute workspace paths in reference runs repo-relative (dry run by default)")
    p.add_argument("--apply", action="store_true", help="rewrite the files")
    p.set_defaults(func=cmd_data_relpaths)

    forms = sub.add_parser("forms", help="LocalForms site").add_subparsers(dest="forms_cmd", required=True, metavar="<action>")
    p = forms.add_parser("serve", help="serve the LocalForms Flask site in the foreground")
    p.set_defaults(func=cmd_forms_serve)

    p = sub.add_parser("ideal", help="generate scripted Playwright reference ('ideal') runs")
    p.add_argument("--forms", help="comma-separated form ids or 'all' (default all)")
    p.add_argument("--runs", help="answer-set indexes, e.g. 1-10 (default: every answer set)")
    p.add_argument("--platform", choices=["google", "localforms"], default="google")
    p.add_argument("--overwrite", action="store_true", help="regenerate runs that already exist (default: skip runs with a video)")
    p.add_argument("--headed", action="store_true", help="show the browser")
    p.add_argument("--screenshots", action="store_true", help="save per-step screenshots")
    p.add_argument("--interaction-mode", choices=["mcp_server", "local"], default="mcp_server", help="mcp_server produced the committed dataset")
    p.add_argument("--status", action="store_true", help="only print reference-run coverage")
    p.add_argument("--submit", action="store_true", help="run as a CPU Slurm job")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_ideal)

    p = sub.add_parser("eval", help="run ONE trial: model x form x answer-set")
    p.add_argument("model")
    p.add_argument("form")
    p.add_argument("run", nargs="?", type=int, default=1)
    p.add_argument("--experiment-id", help="output experiment id (default adhoc_<model>_<date>)")
    p.add_argument("--platform", choices=["google", "localforms"])
    p.add_argument("--budget")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--dry-run", action="store_true", help="print the server and runner commands only")
    p.add_argument("--skip-checks", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--no-tracker", action="store_true", help="do not update docs/eval_results tracker")
    p.add_argument("--submit", action="store_true", help="run as a Slurm job with the model's resources")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("matrix", help="run an experiment: forms x runs x models (locally / inside a job)")
    _add_selection(p)
    p.add_argument("--dry-run", action="store_true", help="print server + runner commands, run nothing")
    p.add_argument("--skip-checks", action="store_true", help="skip the model preflight")
    p.add_argument("--fail-fast", action="store_true", help="stop at the first runner error")
    p.add_argument("--no-tracker", action="store_true", help="do not update docs/eval_results tracker afterwards")
    p.set_defaults(func=cmd_matrix)

    p = sub.add_parser("submit", help="submit an experiment to Slurm (resources from the registry)")
    _add_selection(p)
    p.add_argument("--split", choices=["none", "cohort", "model", "run"], default="none", help="one job per cohort/model/run index")
    p.add_argument("--chain", choices=["afterok", "afterany"], help="make each job depend on the previous one")
    p.add_argument("--after", help="first job depends on this job id")
    p.add_argument("--skip-checks", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="write and print job scripts without submitting")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("experiments", help="list experiment manifests")
    p.set_defaults(func=cmd_experiments)

    p = sub.add_parser("report", help="analytics: CSV tables + SVG plots")
    p.add_argument("--experiments", help="restrict the core report to these experiment ids")
    p.add_argument("--output-dir", help="default reports/ (gitignored)")
    p.add_argument("--study", help="re-run a named thesis study (see --list-studies)")
    p.add_argument("--list-studies", action="store_true")
    p.add_argument("--check", action="store_true", help="with --study: verify committed outputs are up to date instead of rewriting")
    p.add_argument("--reference", action="store_true", help="ideal-run (reference dataset) summary + plots")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="extra args passed to the study script (after --)")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("inspect", help="show one trial: summary, step inputs, media")
    p.add_argument("trial", help="trial directory, or experiment/model/form/run_XXXX/trial_id under data/model_baselines")
    p.add_argument("--steps", nargs="?", const="first", help="print step inputs/outputs: first 3 (default), 'all', or e.g. 0,4,7")
    p.add_argument("--media", action="store_true", help="list screenshots/videos")
    p.set_defaults(func=cmd_inspect)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "extra", None) and args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]
    settings = load_settings()
    try:
        return int(args.func(settings, args) or 0)
    except FormbenchError as exc:
        fail(str(exc))
        if exc.hint:
            print(f"       fix: {exc.hint}", file=sys.stderr)
        return 1
    except KeyError as exc:
        fail(str(exc).strip("'\""))
        return 1
    except KeyboardInterrupt:
        warn("interrupted")
        return 130
