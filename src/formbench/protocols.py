"""Evaluation protocols: which runner script a model uses and with which flags.

A protocol is picked from the registry entry's (provider, track). Runner flags are
merged in this order, later wins:

    protocol defaults < budget profile < provider/kind defaults
      < model `runner_defaults` < experiment cohort `args` < CLI `--set` overrides

The defaults below reproduce the per-protocol defaults of the shell matrix
scripts they replace (run_model_baseline_matrix.sh, run_qwen_direct_mcp_matrix.sh,
run_opencua_direct_matrix.sh, run_opencua_direct_mcp_matrix.sh,
run_gemini_low_cost_matrix.sh, run_track_baseline_matrix.sh).
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from formbench.common import FormbenchError
from formbench.settings import Settings

# Mediated (benchmark-action) budget presets, from run_model_baseline_matrix.sh.
MEDIATED_BUDGETS: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "max_steps": 24, "timeout_s": 900, "max_new_tokens": 128, "step_soft_timeout_s": 30,
        "step_retry_max_new_tokens": 96, "compact_page_text_max_chars": 5000,
        "browser_mcp_timeout_ms": 180000, "prompt_profile": "detailed_v1",
    },
    "large": {
        "max_steps": 56, "timeout_s": 3600, "max_new_tokens": 384, "step_soft_timeout_s": 180,
        "step_retry_max_new_tokens": 256, "compact_page_text_max_chars": 9000,
        "browser_mcp_timeout_ms": 240000, "prompt_profile": "detailed_v1",
    },
    "xlarge": {
        "max_steps": 72, "timeout_s": 5400, "max_new_tokens": 512, "step_soft_timeout_s": 240,
        "step_retry_max_new_tokens": 320, "compact_page_text_max_chars": 12000,
        "browser_mcp_timeout_ms": 300000, "prompt_profile": "detailed_v1",
    },
    "large_qwen3": {
        "max_steps": 128, "timeout_s": 10800, "max_new_tokens": 640, "step_soft_timeout_s": 600,
        "step_retry_max_new_tokens": 384, "compact_page_text_max_chars": 12000,
        "browser_mcp_timeout_ms": 600000, "prompt_profile": "runtime_safe_v1",
    },
}


@dataclass(frozen=True)
class Protocol:
    name: str
    script: str
    description: str
    defaults: Dict[str, Any]
    passes_model_kind: bool = False
    supports_localforms: bool = False
    negations: Dict[str, str] = field(default_factory=dict)
    env_defaults: Dict[str, str] = field(default_factory=dict)
    kind_defaults: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict)
    budget_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    default_budget: Optional[str] = None

    def budget(self, name: Optional[str]) -> Dict[str, Any]:
        if not self.budget_profiles:
            if name:
                raise FormbenchError(f"protocol {self.name} has no budget profiles (got {name!r})")
            return {}
        key = name or self.default_budget
        if key not in self.budget_profiles:
            raise FormbenchError(
                f"unknown budget profile {key!r} for protocol {self.name}",
                hint=f"choose one of: {', '.join(sorted(self.budget_profiles))}",
            )
        return dict(self.budget_profiles[key])


_COMMON_BROWSER = {"execution_backend": "mcp_server", "headless": True, "browser_mcp_timeout_ms": 600000}

PROTOCOLS: Dict[str, Protocol] = {
    "mediated": Protocol(
        name="mediated",
        script="src/baselines/run_baseline_eval.py",
        description="Benchmark-action protocol (text/VLM emit JSON actions executed by the harness); also the FormFactory-style visual condition.",
        passes_model_kind=True,
        defaults={
            **_COMMON_BROWSER,
            "inference_backend": "auto", "api_timeout_s": 240, "invalid_action_budget": 0,
            "idle_step_threshold": 4, "idle_nudge_max": 3, "control_level": "high_level",
            "interaction_protocol": "human_ui_v1", "observation_mode": "vision_coords",
            "scoring_mode": "soft_quality_v1", "verification_scope": "target_only", "history_window": 2,
            "fewshot_enabled": False, "fewshot_count": 1, "browser_init_retries": 2,
            "browser_init_retry_delay_s": 1.5, "retention_window": 5, "disable_action_coercion": True,
            "fill_only_done": False,
        },
        negations={"fewshot_enabled": "no_fewshot_enabled", "disable_action_coercion": "enable_action_coercion"},
        kind_defaults={("openai_compat", "text_llm"): {"max_new_tokens": 128}, ("openai_compat", "vlm"): {"max_new_tokens": 160}},
        budget_profiles=MEDIATED_BUDGETS,
        default_budget="balanced",
    ),
    "direct_mcp": Protocol(
        name="direct_mcp",
        script="src/baselines/run_qwen_direct_mcp_eval.py",
        description="Model calls raw Playwright MCP tools directly (Qwen text/VL, OpenCUA MCP ablation).",
        passes_model_kind=True,
        supports_localforms=True,
        defaults={
            "api_timeout_s": 300, "timeout_s": 1800, "max_steps": 128, "max_new_tokens": 1024, "history_turns": 0,
            "browser_mcp_timeout_ms": 600000, "headless": True, "fill_only_done": False,
        },
    ),
    "opencua_native": Protocol(
        name="opencua_native",
        script="src/baselines/run_opencua_direct_eval.py",
        description="Native computer-use: screenshot -> coordinate actions (OpenCUA-32B), incl. FormFactory-style/ruler variants.",
        defaults={
            **_COMMON_BROWSER,
            "api_timeout_s": 180, "max_new_tokens": 96, "max_steps": 128, "timeout_s": 7200,
            "interaction_protocol": "human_ui_v1", "observation_mode": "vision_coords",
            "scoring_mode": "soft_quality_v1", "retention_window": 5, "history_images": 3,
            "disable_action_coercion": True, "fill_only_done": False, "formfactory_style": False,
            "ruler_overlay": False,
        },
        env_defaults={"OPEN_CUA_MIN_REQUEST_INTERVAL_S": "2.0"},
    ),
    "gemini_low_cost": Protocol(
        name="gemini_low_cost",
        script="src/baselines/run_gemini_low_cost_eval.py",
        description="Gemini native computer use through the low-token Interactions API.",
        defaults={
            **_COMMON_BROWSER,
            "api_timeout_s": 300, "max_new_tokens": 128, "max_steps": 48, "timeout_s": 3600,
            "interaction_protocol": "human_ui_v1", "observation_mode": "vision_coords",
            "scoring_mode": "soft_quality_v1", "retention_window": 5, "disable_action_coercion": True,
            "include_controls": False, "fill_only": False,
        },
        env_defaults={
            "GEMINI_MAX_INFER_RETRIES": "4", "GEMINI_RETRY_DELAY_S": "30",
            "GEMINI_RETRY_BACKOFF": "2", "GEMINI_RETRY_MAX_DELAY_S": "240",
        },
    ),
    "direct_api": Protocol(
        name="direct_api",
        script="src/baselines/run_direct_api_eval.py",
        description="Hosted chat API (OpenAI-compatible or Anthropic) emitting browser actions executed over MCP.",
        defaults={
            **_COMMON_BROWSER,
            "provider": "auto", "api_timeout_s": 180, "max_new_tokens": 96, "max_steps": 128,
            "timeout_s": 5400, "retention_window": 5, "disable_action_coercion": True,
        },
    ),
}


def protocol_for(model: Mapping[str, Any]) -> Protocol:
    provider = model.get("provider")
    track = model.get("track")
    if provider == "gemini_low_cost":
        return PROTOCOLS["gemini_low_cost"]
    if provider == "api_over_mcp":
        return PROTOCOLS["direct_api"]
    if provider == "openai_compat" and track == "direct_mcp_tool_use":
        return PROTOCOLS["direct_mcp"]
    if provider == "openai_compat" and track == "computer_use_native":
        return PROTOCOLS["opencua_native"]
    if provider in {"openai_compat", "local_hf"} and track in {"mediated", "formfactory_style_visual"}:
        return PROTOCOLS["mediated"]
    raise FormbenchError(
        f"no evaluation protocol for model {model.get('id')!r} (provider={provider!r}, track={track!r})",
        hint="see docs/MODELS.md for supported provider/track combinations",
    )


def uses_vllm_python(model: Mapping[str, Any]) -> bool:
    return model.get("python_env") == "vllm"


def python_for(settings: Settings, model: Mapping[str, Any]) -> str:
    return str(settings.vllm_python_bin if uses_vllm_python(model) else settings.python_bin)


def resolve_runner_args(
    protocol: Protocol,
    model: Mapping[str, Any],
    cohort_args: Optional[Mapping[str, Any]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    budget_profile: Optional[str] = None,
) -> Dict[str, Any]:
    args: Dict[str, Any] = dict(protocol.defaults)
    args.update(protocol.budget(budget_profile))
    args.update(protocol.kind_defaults.get((str(model.get("provider")), str(model.get("kind"))), {}))
    if protocol.name == "mediated" and model.get("requires_gpu"):
        args["require_gpu"] = True
    args.update(dict(model.get("runner_defaults") or {}))
    args.update(dict(cohort_args or {}))
    args.update(dict(overrides or {}))
    return args


@dataclass
class TrialSpec:
    model: Dict[str, Any]
    protocol: Protocol
    experiment_id: str
    form_id: str
    run_index: int
    trial_id: str
    run_label: str
    args: Dict[str, Any]
    env: Dict[str, str]
    platform: str = "google"


def build_trial_command(settings: Settings, spec: TrialSpec, endpoint: Optional[Any] = None) -> Tuple[List[str], Dict[str, str]]:
    """Return (argv, extra_env) for one trial. `endpoint` is a serve.Endpoint for openai_compat models."""
    from formbench.common import args_to_flags

    model = spec.model
    protocol = spec.protocol
    argv: List[str] = [python_for(settings, model), str(settings.root / protocol.script)]
    argv += ["--config", str(settings.models_config), "--model-id", str(model["id"])]
    if protocol.passes_model_kind:
        argv += ["--model-kind", str(model["kind"])]
    form_id = spec.form_id
    if spec.platform == "localforms":
        if not protocol.supports_localforms:
            raise FormbenchError(
                f"protocol {protocol.name} (model {model['id']}) does not support the localforms platform",
                hint="LocalForms runs are supported for direct_mcp models only",
            )
        form_id = form_id if form_id.startswith("lf_") else f"lf_{form_id}"
        argv += [
            "--forms-root", str(settings.localforms_forms_root),
            "--form-url", f"{settings.localforms_base_url}/forms/{form_id}",
            "--answers-root", str(settings.localforms_answers_root),
        ]
    argv += ["--form-id", form_id, "--run-index", str(spec.run_index), "--trial-id", spec.trial_id]
    argv += ["--experiment-id", spec.experiment_id, "--run-label", spec.run_label]
    default_dataset = settings.root / "data" / "model_baselines"
    if settings.dataset_root.resolve() != default_dataset.resolve():
        argv += ["--dataset-root", str(settings.dataset_root)]

    args = dict(spec.args)
    extra_env: Dict[str, str] = dict(protocol.env_defaults)
    if endpoint is not None:
        extra_env.update({"OPENAI_BASE_URL": endpoint.base_url, "OPENAI_MODEL": endpoint.model_name, "OPENAI_API_KEY": endpoint.api_key})
        if protocol.name == "opencua_native":
            args.setdefault("base_url", endpoint.base_url)
            args.setdefault("served_model_name", endpoint.model_name)
            args.setdefault("coordinate_type", model.get("coordinate_type") or "qwen25")
            extra_env["OPENCUA_SERVED_MODEL_NAME"] = endpoint.model_name
        if protocol.name == "mediated":
            extra_env.update(endpoint.baseline_env())
    elif protocol.name == "mediated" and model.get("provider") == "local_hf":
        extra_env["BASELINE_SERVING_MODE"] = "local_hf_trial_local"
    extra_env.update({k: str(v) for k, v in spec.env.items()})
    # Explicit env from the caller's shell wins over protocol defaults (e.g. GEMINI_* retries).
    for key in list(protocol.env_defaults):
        if key in os.environ and key not in spec.env:
            extra_env[key] = os.environ[key]
    argv += args_to_flags(args, protocol.negations)
    return argv, extra_env
