# Models and experiments

All models live in one registry, `configs/models.json`. An entry says what the model is
(`kind`), how to reach it (`provider`, endpoint or `serve` block), which evaluation
protocol it uses (`track`), and what hardware it needs (`resources`). The `id` is also
the output folder name: `data/model_baselines/<experiment>/<id>/…`.

```bash
make models                 # current models (ALL=1 adds legacy ones)
make model-check MODEL=<id> # validate one entry and everything it needs
```

## How a model is run

`provider` + `track` select the protocol, i.e. the runner script and its default flags:

| Protocol | provider / track | Runner | Used for |
|---|---|---|---|
| `direct_mcp` | `openai_compat` / `direct_mcp_tool_use` | `src/baselines/run_qwen_direct_mcp_eval.py` | Model calls Playwright MCP tools directly (Qwen text/VL, OpenCUA MCP). Supports `PLATFORM=localforms` |
| `opencua_native` | `openai_compat` / `computer_use_native` | `src/baselines/run_opencua_direct_eval.py` | Screenshot → pyautogui-style coordinate actions (OpenCUA). `formfactory_style`, `ruler_overlay` flags |
| `mediated` | `openai_compat` or `local_hf` / `mediated`, `formfactory_style_visual` | `src/baselines/run_baseline_eval.py` | Benchmark-action JSON protocol; FormFactory-style visual condition. Budget profiles |
| `gemini_low_cost` | `gemini_low_cost` | `src/baselines/run_gemini_low_cost_eval.py` | Gemini native computer use (Interactions API) |
| `direct_api` | `api_over_mcp` | `src/baselines/run_direct_api_eval.py` | Hosted OpenAI-compatible / Anthropic API emitting browser actions |

Runner flags are merged in this order, later wins:

```
protocol defaults < budget profile (mediated) < provider/kind defaults
  < model "runner_defaults" < experiment cohort "args" < command line SET="k=v"
```

`DRY_RUN=1` on `eval`, `matrix` or `submit` prints the final server and runner commands.

## Registry fields

| Field | Required for | Meaning |
|---|---|---|
| `id` | all | Unique; output folder name |
| `kind` | all | `text_llm`, `vlm` or `computer_use_agent` |
| `provider` | all | `openai_compat`, `local_hf`, `gemini_low_cost`, `api_over_mcp` |
| `track` | all | Selects the protocol (table above) |
| `status` | | `current` (default) or `legacy` (hidden from `make models`, kept to reproduce old runs) |
| `hf_repo` | `local_hf`, served models | Hugging Face repo id; downloaded by `make install-models` or by vLLM at start |
| `weights_dir` | | Local weights folder if not `models/<id>` (lets several ids share weights) |
| `requires_gpu` | | Enforced by preflight and `--require-gpu` |
| `resources` | | `{gpus, cpus, mem, time, min_vram_gb}`: Slurm request and GPU preflight |
| `openai_model` / `served_model_name` | `openai_compat` | Name sent in requests / name vLLM serves (normally equal) |
| `openai_base_url` | `openai_compat` without `serve` | Remote or externally started endpoint |
| `serve` | locally served `openai_compat` | vLLM settings: `port` (`"auto"` = per-job port), `tensor_parallel`, `gpu_memory_utilization`, `max_model_len`, `tool_call_parser`, `enable_auto_tool_choice`, `generation_config`, `limit_mm_per_prompt`, `model_impl`, `chat_template`, `extra_args`, `env`, `startup_attempts`, `startup_sleep_s` |
| `python_env` | | `vllm` runs the runner with the vLLM interpreter (OpenCUA) |
| `coordinate_type` | OpenCUA | Coordinate convention for native actions (`qwen25`) |
| `runner_defaults` | | Per-model runner flags, e.g. `{"timeout_s": 9000}` |
| `gemini_model`, `pricing`, `api_key_env` | `gemini_low_cost` | Model name, cost table, key variable (default `GEMINI_API_KEY`, or `.secrets/gemini_api_key`) |
| `openai_model` / `anthropic_model`, `openai_base_url`, `openai_extra_body`, `api_key_env` | `api_over_mcp` | Hosted API model and credentials variable |
| `is_fallback`, `fallback_for` | | Run this model when the primary VLM times out or runs out of memory |

The schema is enforced: wrong types, unknown providers or tracks, a
`tensor_parallel` larger than `resources.gpus`, or a dangling `fallback_for` are
errors. Unknown keys are reported as likely typos.

## Adding or swapping a model

Most new models need **only a registry entry**, with no code changes. The work is
picking the right `provider` + `track` (which decides the runner), then checking and
trialling the entry.

### Step 1: decide what kind of model it is

| Your model | `provider` / `track` | `kind` | Code changes? |
|---|---|---|---|
| Open-weights chat model with tool calling (Qwen, Llama, Mistral, …), served on our GPUs | `openai_compat` / `direct_mcp_tool_use` + `serve` block | `text_llm` or `vlm` | No |
| Same, but already served elsewhere (another job, a group server) | `openai_compat` / `direct_mcp_tool_use` + `openai_base_url` (no `serve`) | `text_llm` or `vlm` | No |
| Hosted OpenAI-compatible or Anthropic API (e.g. a university LLM service) | `api_over_mcp` / `direct_api_tool_use` | `computer_use_agent` | No |
| Gemini computer-use model | `gemini_low_cost` / `proprietary_computer_use_low_cost` | `computer_use_agent` | No (`gemini_model` picks the version) |
| Screenshot → pyautogui-style computer-use model (OpenCUA family) | `openai_compat` / `computer_use_native` | `computer_use_agent` | No if it emits OpenCUA-style `pyautogui.*` actions; otherwise the action parser in `run_opencua_direct_eval.py` needs extending |
| Small model run in-process with transformers (benchmark-action protocol) | `local_hf` / `mediated` | `text_llm` or `vlm` | No |
| A new kind of agent/API not covered above | new provider | | Yes, see "Adding a new protocol" below |

**Swapping** a checkpoint within a family (for example a newer Qwen VL): copy the existing entry
and change `id`, `hf_repo`, `openai_model`/`served_model_name`, and `resources`/`serve` if the
size changes. Keep `track`, so the comparison uses the same protocol.

### Step 2: add the entry to `configs/models.json`

**Locally served model** (vLLM is started and stopped by `eval`/`matrix`):

```json
{
  "id": "vlm_new_model", "status": "current", "kind": "vlm",
  "provider": "openai_compat", "track": "direct_mcp_tool_use", "hf_repo": "Org/New-VL-Model",
  "requires_gpu": true, "openai_model": "vlm-new-model", "served_model_name": "vlm-new-model",
  "serve": {"port": "auto", "tensor_parallel": 2, "tool_call_parser": "hermes", "limit_mm_per_prompt": "{\"image\":1,\"video\":0}"},
  "resources": {"gpus": 2, "cpus": 12, "mem": "120G", "time": "24:00:00"}
}
```

Sizing: `tensor_parallel` must not exceed `resources.gpus`. Roughly, weights in bf16 need 2 GB per
billion parameters, plus room for the KV cache (a 30B model fits on 2 × A100-40GB). Set
`tool_call_parser` to the parser vLLM documents for the model family. Large checkpoints on the
cluster filesystem can take 15–45 min to load, and `startup_attempts` (default 420 × 10 s) covers
that.

**Hosted OpenAI-compatible API** (no GPU). This is the SCADS/TUD:AI Gemma 4 endpoint used in a
pilot, kept here as a ready-made example:

```json
{
  "id": "scads_gemma4_31b_it_api", "status": "current", "kind": "computer_use_agent",
  "provider": "api_over_mcp", "track": "direct_api_tool_use", "hf_repo": "google/gemma-4-31B-it",
  "requires_gpu": false, "openai_model": "google/gemma-4-31B-it", "openai_base_url": "https://llm.scads.ai/v1",
  "openai_extra_body": {"disable_fallbacks": true}, "api_key_env": "OPENAI_API_KEY",
  "resources": {"gpus": 0, "cpus": 6, "mem": "32G", "time": "12:00:00"}
}
```

Never put API keys in the registry. Export the variable named in `api_key_env`
(`export OPENAI_API_KEY=…`), or for Gemini use `.secrets/gemini_api_key` (mode 600; gitignored).
The preflight refuses to start a remote API model without its key.

### Step 3: check, trial, then scale up

```bash
make model-check MODEL=<id>                 # fix every FAIL row; WARN rows are informational
make install-models MODEL=<id>              # optional: pre-fetch weights (served / local_hf models)
make eval MODEL=<id> FORM=conf_interest RUN=1 SET="max_steps=8" DRY_RUN=1    # inspect the exact commands
make eval MODEL=<id> FORM=conf_interest RUN=1 SET="max_steps=8" SUBMIT=1 TIME=02:00:00   # cluster
make inspect TRIAL=adhoc_<id>_<date>/<id>/conf_interest/run_0001 STEPS=all  # did it behave sensibly?
```

Then add the id to a cohort's `models` list in an experiment manifest (or create one, see below),
and run `make submit EXPERIMENT=<name> DRY_RUN=1` before submitting for real.
`make matrix EXPERIMENT=<manifest> MODELS=<id>` filters an existing manifest to just your model.

### Adding a new protocol (only for a new kind of agent)

If the model needs a different interaction loop or API than every protocol above:

1. **Runner:** add `src/baselines/run_<name>_eval.py`. Copy the closest existing runner, since
   they share argument names (`--config --model-id --form-id --run-index --trial-id
   --experiment-id …`). Write outputs with `rbe._build_trial_paths`,
   `rbe._update_experiment_indexes` and the usual `summary.json`/`annotations.json`
   fields (`success`, `question_total`, `scored_correctness`, `stop_reason`, `run_params`), so
   analytics and skip-completed logic work unchanged. Reuse `baselines/common.py` for HTTP and
   answer loading.
2. **Registry schema:** add the provider to `PROVIDERS` and its allowed keys to `PROVIDER_KEYS`
   (plus any required-key rule) in `src/baselines/model_registry.py`, and a track to `TRACKS`.
3. **Protocol:** add a `Protocol` (script, default flags) to `PROTOCOLS` and a branch in
   `protocol_for()` in `src/formbench/protocols.py`.
4. **Failsafes:** if it uses a new kind of credential, add it to `_api_key_check()` in
   `src/formbench/checks.py`.
5. **Tests:** add a command-building case to `tests/test_formbench.py` (`ProtocolCommandTests`)
   and run `make test`.

## Environment overrides

Runners read `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_API_KEY`, `GEMINI_MODEL`,
`GEMINI_*RETRY*` and similar variables, and these **override the registry**.
`make model-check` lists every active override as a WARN row, so a stale export in your
shell cannot silently send requests to the wrong model. `eval`/`matrix` set
`OPENAI_*` themselves for served models.

## Check failures and fixes

| Check | Typical failure | Fix |
|---|---|---|
| registry | `serve.tensor_parallel=4 exceeds resources.gpus=2` | Align the two, or raise `resources.gpus` |
| api key | `GEMINI_API_KEY unset and .secrets/gemini_api_key missing` | Create the key file (`chmod 600`) or export the variable |
| api key | `remote endpoint … but OPENAI_API_KEY is unset` | `export OPENAI_API_KEY=…` |
| weights | `no local weights` (local_hf) | `make install-models MODEL=<id>` |
| transformers (`SMOKE=1`) | architecture not loadable | `make setup WITH=localhf`, or upgrade transformers |
| endpoint | `… serves ['x'], not 'y'` | Fix `openai_model`, or stop the other server on that port |
| endpoint | unreachable (remote) | Check URL/VPN, or add a `serve` block to run it locally |
| gpu | needs N GPU(s), none visible | Run through `make submit` (or on a GPU node with `LOCAL=1`) |
| interpreter | `.venv-opencua/bin/python` missing | `make setup WITH=vllm`, or set `VLLM_PYTHON_BIN` in `.env` |

## Experiment manifests

`configs/experiments/<name>.json` records an experiment completely, so any run can be
reproduced by name:

```json
{
  "id": "fill_only_done_30",
  "description": "…",
  "defaults": {"forms": ["…"], "run_indexes": [2], "skip_completed": "none"},
  "cohorts": [
    {"name": "qwen", "experiment_id": "qwen_direct_mcp_fill_only_done_30_seed20260709_r2_step32",
     "models": ["text_qwen3_30b_a3b_instruct_2507", "vlm_qwen3_vl_30b_a3b_instruct"],
     "args": {"max_steps": 32, "fill_only_done": true}},
    {"name": "gemini", "experiment_id": "…", "models": ["computer_use_gemini_35_flash_lowcost"],
     "args": {"max_steps": 32, "fill_only": true}, "env": {"GEMINI_MAX_INFER_RETRIES": "8"}}
  ]
}
```

| Key | Meaning |
|---|---|
| `experiment_id` | Output folder; re-running a manifest resumes it |
| `models` | Registry ids (one server per model, started once per cohort) |
| `forms` | List, or `"all"` (+ `forms_offset` / `forms_limit`) |
| `run_indexes` | `[2]`, `"1-10"`, `"1,3"`: which answer sets |
| `platform` | `google` (default) or `localforms` |
| `skip_completed` | `experiment` (default): skip pairs already done in this experiment; `any`: done in any experiment; `none`: always run |
| `args` / `env` | Runner flags and environment for this cohort |
| `budget_profile` | Mediated models: `balanced`, `large`, `xlarge`, `large_qwen3` |
| `fallback` | Use registered fallback models (default true) |

The committed manifests reproduce the thesis experiments from their recorded parameters
(`run_params` in the trial annotations and the Slurm logs). `make experiments` lists them.
A new study needs a new manifest: copy one, give each cohort a new `experiment_id`,
and run `make submit EXPERIMENT=<name> DRY_RUN=1` first.

## Legacy models

Entries with `"status": "legacy"` reproduce earlier experiments (the benchmark-action
"mediated" track with vLLM or in-process transformers, FP8 and 7B variants, the first
direct-API pilot). Where an older config reused a current id for a different setup, the
legacy entry has a suffix (`_mediated`, `_localhf`) and points `weights_dir` at the shared
weights. Results already on disk keep their original folder names.
