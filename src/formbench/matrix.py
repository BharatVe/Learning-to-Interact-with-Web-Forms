"""Experiment manifests and the forms x runs x models trial matrix.

Replaces the per-model shell matrices. An experiment manifest
(`configs/experiments/<name>.json`) holds one or more cohorts; each cohort is
one output experiment_id with a model list, forms, run indexes and runner args:

    {
      "id": "fill_only_done_50_completion_20260713",
      "description": "...",
      "defaults": {"platform": "google", "run_indexes": [2], "skip_completed": "experiment"},
      "cohorts": [
        {"name": "gemini", "experiment_id": "gemini_..._step32", "models": ["computer_use_gemini_35_flash_lowcost"],
         "forms": ["alumni_checkin", "..."], "args": {"max_steps": 32, "fill_only": true},
         "env": {"GEMINI_MAX_INFER_RETRIES": "8"}}
      ]
    }

`forms` is a list or "all" (optionally with `forms_offset` / `forms_limit`).
`skip_completed`: "experiment" (a summary.json for model/form/run exists in this
experiment), "any" (in any experiment - the old shell default), or "none".
"""

import json
import os
import subprocess
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from baselines.model_registry import get_model_by_id, list_models
from formbench.common import (
    FormbenchError, answer_run_id, discover_form_ids, fail, info, new_run_label, new_trial_id,
    parse_run_indexes, quote_cmd, read_json, short, utc_now, warn, write_json,
)
from formbench.protocols import TrialSpec, build_trial_command, protocol_for, resolve_runner_args, uses_vllm_python
from formbench.settings import Settings

EXPERIMENTS_DIR = Path("configs/experiments")
SKIP_MODES = {"experiment", "any", "none"}
PLATFORMS = {"google", "localforms"}
FALLBACK_CATEGORIES = {"model_inference_failed", "environment_error", "timeout"}
FALLBACK_MARKERS = ("out of memory", "snapshotforai", "timed out")


@dataclass
class Cohort:
    name: str
    experiment_id: str
    models: List[str]
    forms: List[str]
    run_indexes: List[int]
    platform: str = "google"
    skip_completed: str = "experiment"
    args: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    budget_profile: Optional[str] = None
    fallback: bool = True


@dataclass
class Experiment:
    id: str
    description: str
    cohorts: List[Cohort]
    source: Optional[Path] = None


def list_experiments(settings: Settings) -> List[Tuple[str, str]]:
    root = settings.root / EXPERIMENTS_DIR
    rows = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = read_json(path)
            rows.append((path.stem, str(payload.get("description", ""))))
        except (OSError, ValueError) as exc:
            rows.append((path.stem, f"<invalid: {exc}>"))
    return rows


def _resolve_forms(settings: Settings, value: Any, offset: int = 0, limit: int = 0) -> List[str]:
    if value in (None, "", "all"):
        forms = discover_form_ids(settings.forms_root)
    elif isinstance(value, str):
        forms = [f.strip() for f in value.split(",") if f.strip()]
    elif isinstance(value, list):
        forms = [str(f).strip() for f in value if str(f).strip()]
    else:
        raise FormbenchError(f"forms must be a list or 'all', got {value!r}")
    if offset:
        forms = forms[offset:]
    if limit:
        forms = forms[:limit]
    known = set(discover_form_ids(settings.forms_root))
    unknown = [f for f in forms if f.replace("lf_", "", 1) not in known]
    if unknown:
        raise FormbenchError(f"unknown form id(s): {', '.join(unknown)}", hint=f"valid ids are the folders in {settings.forms_root}")
    return forms


def _build_cohort(settings: Settings, raw: Mapping[str, Any], defaults: Mapping[str, Any], fallback_name: str) -> Cohort:
    merged: Dict[str, Any] = dict(defaults)
    for key, value in raw.items():
        if key in {"args", "env"}:
            merged[key] = {**dict(defaults.get(key) or {}), **dict(value or {})}
        else:
            merged[key] = value
    experiment_id = str(merged.get("experiment_id") or "").strip()
    if not experiment_id:
        raise FormbenchError(f"cohort {raw.get('name', fallback_name)!r} has no experiment_id")
    models = merged.get("models") or []
    if isinstance(models, str):
        models = [m.strip() for m in models.split(",") if m.strip()]
    if not models:
        raise FormbenchError(f"cohort {experiment_id} lists no models")
    platform = str(merged.get("platform") or "google")
    if platform not in PLATFORMS:
        raise FormbenchError(f"platform must be one of {sorted(PLATFORMS)}, got {platform!r}")
    skip = str(merged.get("skip_completed") or "experiment")
    if skip not in SKIP_MODES:
        raise FormbenchError(f"skip_completed must be one of {sorted(SKIP_MODES)}, got {skip!r}")
    run_indexes = parse_run_indexes(merged.get("run_indexes", 1))
    if not run_indexes:
        raise FormbenchError(f"cohort {experiment_id} has no run_indexes")
    return Cohort(
        name=str(merged.get("name") or fallback_name),
        experiment_id=experiment_id,
        models=[str(m) for m in models],
        forms=_resolve_forms(settings, merged.get("forms", "all"), int(merged.get("forms_offset") or 0), int(merged.get("forms_limit") or 0)),
        run_indexes=run_indexes,
        platform=platform,
        skip_completed=skip,
        args=dict(merged.get("args") or {}),
        env={k: str(v) for k, v in dict(merged.get("env") or {}).items()},
        budget_profile=merged.get("budget_profile"),
        fallback=bool(merged.get("fallback", True)),
    )


def load_experiment(settings: Settings, name_or_path: str) -> Experiment:
    path = Path(name_or_path)
    if not path.suffix:
        path = settings.root / EXPERIMENTS_DIR / f"{name_or_path}.json"
    elif not path.is_absolute():
        path = settings.root / path
    if not path.is_file():
        available = ", ".join(name for name, _ in list_experiments(settings))
        raise FormbenchError(f"experiment manifest not found: {path}", hint=f"available: {available}")
    payload = read_json(path)
    defaults = dict(payload.get("defaults") or {})
    cohorts_raw = payload.get("cohorts") or []
    if not cohorts_raw:
        raise FormbenchError(f"{path} defines no cohorts")
    cohorts = [_build_cohort(settings, raw, defaults, f"cohort{i + 1}") for i, raw in enumerate(cohorts_raw)]
    return Experiment(id=str(payload.get("id") or path.stem), description=str(payload.get("description", "")), cohorts=cohorts, source=path)


def adhoc_experiment(
    settings: Settings, experiment_id: str, models: List[str], forms: Any, run_indexes: Any,
    platform: str = "google", skip_completed: str = "experiment", budget_profile: Optional[str] = None,
) -> Experiment:
    raw = {"experiment_id": experiment_id, "models": models, "forms": forms, "run_indexes": run_indexes,
           "platform": platform, "skip_completed": skip_completed, "budget_profile": budget_profile}
    return Experiment(id=experiment_id, description="ad-hoc", cohorts=[_build_cohort(settings, raw, {}, "adhoc")])


def apply_cohort_filters(experiment: Experiment, cohorts: List[str], models: List[str]) -> Experiment:
    selected = []
    for cohort in experiment.cohorts:
        if cohorts and cohort.name not in cohorts and cohort.experiment_id not in cohorts:
            continue
        if models:
            kept = [m for m in cohort.models if m in models]
            if not kept:
                continue
            cohort = Cohort(**{**cohort.__dict__, "models": kept})
        selected.append(cohort)
    if not selected:
        names = ", ".join(f"{c.name} ({', '.join(c.models)})" for c in experiment.cohorts)
        raise FormbenchError("filters matched no cohort/model", hint=f"cohorts in {experiment.id}: {names}")
    return Experiment(id=experiment.id, description=experiment.description, cohorts=selected, source=experiment.source)


def _form_dir_id(cohort: Cohort, form_id: str) -> str:
    return f"lf_{form_id}" if cohort.platform == "localforms" and not form_id.startswith("lf_") else form_id


def completed_pairs(dataset_root: Path, model_id: str, experiment_id: Optional[str] = None) -> set:
    """{(form_dir, run_id)} with a summary.json for this model, in one experiment or (None) any.

    One directory walk per model instead of one glob per form/run: on Lustre with ~80
    experiments this cuts planning a 1,200-trial manifest from ~40 s to about a second.
    """
    import os

    done = set()
    experiments = [experiment_id] if experiment_id else [e.name for e in os.scandir(dataset_root) if e.is_dir() and not e.name.startswith("_")] if dataset_root.is_dir() else []
    for exp in experiments:
        model_dir = dataset_root / exp / model_id
        if not model_dir.is_dir():
            continue
        for form in os.scandir(model_dir):
            if not form.is_dir():
                continue
            for run in os.scandir(form.path):
                if not run.is_dir() or not run.name.startswith("run_") or (form.name, run.name) in done:
                    continue
                if any(os.path.isfile(os.path.join(t.path, "summary.json")) for t in os.scandir(run.path) if t.is_dir()):
                    done.add((form.name, run.name))
    return done


def missing_references(settings: Settings, cohort: Cohort) -> List[Tuple[str, int]]:
    """(form, run) pairs of a cohort without a usable ideal reference run (trace present, no failure).

    Trials on those pairs still run, but get no efficiency-vs-ideal metrics.
    """
    root = settings.localforms_reference_root if cohort.platform == "localforms" else settings.reference_root
    missing = []
    for form_id in cohort.forms:
        for run_index in cohort.run_indexes:
            run_dir = root / _form_dir_id(cohort, form_id) / "runs" / answer_run_id(run_index)
            trace = run_dir / "tool_trace.jsonl"
            if not trace.is_file() or trace.stat().st_size == 0 or (run_dir / "failure_manifest.json").exists():
                missing.append((form_id, run_index))
    return missing


def trial_completed(settings: Settings, cohort: Cohort, model_id: str, form_id: str, run_index: int) -> bool:
    if cohort.skip_completed == "none":
        return False
    scope = None if cohort.skip_completed == "any" else cohort.experiment_id
    return (_form_dir_id(cohort, form_id), answer_run_id(run_index)) in completed_pairs(settings.dataset_root, model_id, scope)


@dataclass
class TrialOutcome:
    experiment_id: str
    model_id: str
    form_id: str
    run_index: int
    trial_id: str
    exit_code: int
    duration_s: float
    success: Optional[bool] = None
    stop_reason: Optional[str] = None
    failure_category: Optional[str] = None
    failure_detail: Optional[str] = None
    fallback_for: Optional[str] = None
    summary_written: bool = False


def _trial_env(settings: Settings, model: Mapping[str, Any], extra: Mapping[str, str]) -> Dict[str, str]:
    env = settings.subprocess_env(extra)
    module_ld = os.environ.get("MODULE_LD_LIBRARY_PATH")
    if module_ld:
        env.setdefault("NODE_LD_LIBRARY_PATH_FOR_MCP", module_ld)
        if uses_vllm_python(model):
            env["LD_LIBRARY_PATH"] = module_ld
    return env


def _read_summary(settings: Settings, cohort: Cohort, model_id: str, form_id: str, run_index: int, trial_id: str) -> Dict[str, Any]:
    path = settings.dataset_root / cohort.experiment_id / model_id / _form_dir_id(cohort, form_id) / answer_run_id(run_index) / trial_id / "summary.json"
    try:
        payload = read_json(path)
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def _should_fallback(model: Mapping[str, Any], outcome: TrialOutcome) -> bool:
    if model.get("kind") != "vlm" or outcome.exit_code == 0:
        return False
    if outcome.failure_category in FALLBACK_CATEGORIES:
        return True
    text = f"{outcome.failure_category or ''} {outcome.failure_detail or ''}".lower()
    return any(marker in text for marker in FALLBACK_MARKERS)


class MatrixRunner:
    def __init__(self, settings: Settings, experiment: Experiment, overrides: Optional[Mapping[str, Any]] = None,
                 dry_run: bool = False, fail_fast: bool = False, update_tracker: bool = True):
        self.settings = settings
        self.experiment = experiment
        self.overrides = dict(overrides or {})
        self.dry_run = dry_run
        self.fail_fast = fail_fast
        self.update_tracker = update_tracker
        self.registry = {str(m["id"]): m for m in list_models(settings.models_config)}
        self.outcomes: List[TrialOutcome] = []
        self.skipped = 0
        self.server_failures: List[Dict[str, Any]] = []

    def model(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self.registry:
            return get_model_by_id(self.settings.models_config, model_id)  # raises with known ids
        return self.registry[model_id]

    def fallback_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        for model in self.registry.values():
            if model.get("is_fallback") and model.get("fallback_for") == model_id:
                return model
        return None

    def spec(self, cohort: Cohort, model: Dict[str, Any], form_id: str, run_index: int) -> TrialSpec:
        protocol = protocol_for(model)
        args = resolve_runner_args(protocol, model, cohort.args, self.overrides, cohort.budget_profile)
        trial_id = "trial_<utc>" if self.dry_run else new_trial_id()
        run_label = "<utc>_job<id>" if self.dry_run else new_run_label()
        return TrialSpec(model=model, protocol=protocol, experiment_id=cohort.experiment_id, form_id=form_id,
                         run_index=run_index, trial_id=trial_id, run_label=run_label, args=args, env=cohort.env,
                         platform=cohort.platform)

    def pending(self, cohort: Cohort, model_id: str) -> List[Tuple[str, int]]:
        todo = []
        done: set = set()
        if cohort.skip_completed != "none":
            scope = None if cohort.skip_completed == "any" else cohort.experiment_id
            done = completed_pairs(self.settings.dataset_root, model_id, scope)
        for form_id in cohort.forms:
            for run_index in cohort.run_indexes:
                if (_form_dir_id(cohort, form_id), answer_run_id(run_index)) in done:
                    self.skipped += 1
                    continue
                todo.append((form_id, run_index))
        return todo

    def run_trial(self, cohort: Cohort, model: Dict[str, Any], form_id: str, run_index: int, endpoint: Any, fallback_for: Optional[str] = None) -> TrialOutcome:
        spec = self.spec(cohort, model, form_id, run_index)
        argv, extra_env = build_trial_command(self.settings, spec, endpoint)
        if self.dry_run:
            print("  " + short(quote_cmd(argv[1:])))
            return TrialOutcome(cohort.experiment_id, model["id"], form_id, run_index, spec.trial_id, 0, 0.0)
        info(f"trial model={model['id']} form={form_id} run={run_index} trial={spec.trial_id}" + (f" (fallback for {fallback_for})" if fallback_for else ""))
        started = time.time()
        code = subprocess.run(argv, env=_trial_env(self.settings, model, extra_env), cwd=str(self.settings.root)).returncode
        summary = _read_summary(self.settings, cohort, model["id"], form_id, run_index, spec.trial_id)
        outcome = TrialOutcome(
            cohort.experiment_id, model["id"], form_id, run_index, spec.trial_id, code, round(time.time() - started, 1),
            success=summary.get("success"), stop_reason=summary.get("stop_reason"),
            failure_category=summary.get("failure_category"), failure_detail=summary.get("failure_detail"),
            fallback_for=fallback_for, summary_written=bool(summary),
        )
        status = "ok" if code == 0 else f"exit={code}"
        info(f"trial done {status} success={outcome.success} stop_reason={outcome.stop_reason} ({outcome.duration_s}s)")
        self.outcomes.append(outcome)
        if code != 0 and self.fail_fast:
            raise FormbenchError(f"trial failed (exit {code}) and --fail-fast is set")
        return outcome

    def run_model(self, cohort: Cohort, model: Dict[str, Any]) -> None:
        from formbench.serve import VLLMServer, endpoint_for

        todo = self.pending(cohort, model["id"])
        header = f"cohort={cohort.name} experiment={cohort.experiment_id} model={model['id']} protocol={protocol_for(model).name}"
        if not todo:
            info(f"{header}: all {len(cohort.forms) * len(cohort.run_indexes)} trials already completed, skipping")
            return
        endpoint = endpoint_for(model) if model.get("provider") == "openai_compat" else None
        if self.dry_run:
            print(f"\n# {header}: {len(todo)} trial(s)")
            if endpoint is not None and endpoint.managed:
                from formbench.serve import build_vllm_command

                argv, _ = build_vllm_command(self.settings, model, endpoint)
                print("# server: " + short(quote_cmd(argv)))
            elif endpoint is not None:
                print(f"# endpoint: {endpoint.base_url} ({endpoint.model_name})")
            for form_id, run_index in todo:
                self.run_trial(cohort, model, form_id, run_index, endpoint)
            return
        info(f"{header}: {len(todo)} trial(s)")
        with ExitStack() as stack:
            if endpoint is not None and endpoint.managed:
                try:
                    stack.enter_context(VLLMServer(self.settings, model, endpoint))
                except FormbenchError as exc:
                    if self.fail_fast:
                        raise
                    # One model's server failing must not cost the other models in this job their trials.
                    first_line = str(exc).splitlines()[0]
                    fail(f"{header}: server did not start; {len(todo)} trial(s) marked server_start_failed, continuing\n{exc}")
                    if exc.hint:
                        print(f"       fix: {exc.hint}", flush=True)
                    self.server_failures.append({"experiment_id": cohort.experiment_id, "model_id": model["id"], "error": first_line})
                    for form_id, run_index in todo:
                        self.outcomes.append(TrialOutcome(
                            cohort.experiment_id, model["id"], form_id, run_index, "", -1, 0.0, success=False,
                            stop_reason="server_start_failed", failure_category="server_start_failed", failure_detail=first_line,
                        ))
                    return
            fallback = self.fallback_model(model["id"]) if cohort.fallback else None
            for form_id, run_index in todo:
                outcome = self.run_trial(cohort, model, form_id, run_index, endpoint)
                if fallback is not None and _should_fallback(model, outcome):
                    if fallback.get("provider") == "openai_compat" and isinstance(fallback.get("serve"), dict):
                        warn(f"fallback {fallback['id']} needs its own server; not started mid-matrix. Run it as a separate cohort.")
                    else:
                        self.run_trial(cohort, fallback, form_id, run_index, None, fallback_for=model["id"])

    def run(self) -> int:
        from formbench.forms_site import LocalFormsSite

        started = utc_now()
        for cohort in self.experiment.cohorts:
            gaps = missing_references(self.settings, cohort)
            if gaps:
                runs = ",".join(str(r) for r in sorted({r for _, r in gaps}))
                platform = f" PLATFORM={cohort.platform}" if cohort.platform != "google" else ""
                warn(
                    f"cohort {cohort.name}: {len(gaps)}/{len(cohort.forms) * len(cohort.run_indexes)} form/run pairs have no ideal "
                    f"reference run, so those trials get no efficiency-vs-ideal metrics (e.g. {gaps[0][0]} run {gaps[0][1]}). "
                    f"Generate them first with: make ideal-runs RUNS={runs}{platform}"
                )
        with ExitStack() as stack:
            if not self.dry_run and any(c.platform == "localforms" for c in self.experiment.cohorts):
                stack.enter_context(LocalFormsSite(self.settings))
            for cohort in self.experiment.cohorts:
                for model_id in cohort.models:
                    self.run_model(cohort, self.model(model_id))
        if self.dry_run:
            print(f"\n# dry run: {self.skipped} trial(s) would be skipped as already completed")
            return 0
        return self.finish(started)

    def finish(self, started: Any) -> int:
        from formbench.analysis_hooks import post_matrix

        completed = [o for o in self.outcomes if o.summary_written]
        succeeded = [o for o in self.outcomes if o.success]
        crashed = [o for o in self.outcomes if o.exit_code != 0 and not o.summary_written and o.stop_reason != "server_start_failed"]
        unserved = [o for o in self.outcomes if o.stop_reason == "server_start_failed"]
        info(
            f"trials={len(self.outcomes)} completed={len(completed)} task_success={len(succeeded)} "
            f"runner_crashes={len(crashed)} server_start_failed={len(unserved)} skipped={self.skipped}"
        )
        for o in crashed:
            warn(f"runner crashed without a summary: model={o.model_id} form={o.form_id} run={o.run_index} exit={o.exit_code}")
        report = {
            "experiment": self.experiment.id,
            "source": str(self.experiment.source) if self.experiment.source else None,
            "started_utc": started.isoformat(),
            "finished_utc": utc_now().isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "overrides": self.overrides,
            "skipped": self.skipped,
            "server_failures": self.server_failures,
            "outcomes": [o.__dict__ for o in self.outcomes],
        }
        out = self.settings.logs_dir / "matrix" / f"{self.experiment.id}-{started.strftime('%Y%m%dT%H%M%SZ')}.json"
        write_json(out, report)
        info(f"matrix report: {out}")
        if completed:
            post_matrix(self.settings, self.experiment, update_tracker=self.update_tracker)
        # Task failures are benchmark results (exit 0); infrastructure failures fail the job so Slurm shows them.
        return 1 if self.server_failures or crashed else 0


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    """--set max_steps=32 --set fill_only_done=true -> {"max_steps": 32, "fill_only_done": True}."""
    result: Dict[str, Any] = {}
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        if not sep or not key.strip():
            raise FormbenchError(f"--set expects key=value, got {pair!r}")
        key = key.strip().replace("-", "_")
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        result[key] = value
    return result
