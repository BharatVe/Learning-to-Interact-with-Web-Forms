# Short commands for the whole workflow. Every target runs inside scripts/env.sh
# (loads .env, HPC modules, caches) and calls the formbench CLI, so
# `make <target>` and `scripts/env.sh .venv/bin/python -m formbench ...` are equivalent.
# Run `make help` for the list; variables are passed as `make eval MODEL=... FORM=...`.

SHELL := /bin/bash
.DEFAULT_GOAL := help
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PY ?= $(ROOT)/.venv/bin/python
FB := $(ROOT)/scripts/env.sh $(PY) -m formbench

# Optional variables (empty = default)
MODEL ?=
MODELS ?=
FORM ?= conf_interest
FORMS ?=
RUN ?= 1
RUNS ?=
EXPERIMENT ?=
EXP_ID ?=
SUFFIX ?=
COHORT ?=
PLATFORM ?=
BUDGET ?=
SET ?=
SPLIT ?= none
CHAIN ?=
EXPERIMENTS ?=
NAME ?=
TRIAL ?=
STEPS ?=
WITH ?=

flag = $(if $(2),$(1) $(2),)
bool = $(if $(filter 1 yes true,$(2)),$(1),)
SET_FLAGS := $(foreach kv,$(SET),--set $(kv))
SELECT := $(EXPERIMENT) $(call flag,--models,$(or $(MODELS),$(MODEL))) $(call flag,--forms,$(FORMS)) \
  $(call flag,--runs,$(RUNS)) $(call flag,--cohort,$(COHORT)) $(call flag,--experiment-id,$(EXP_ID)) \
  $(call flag,--experiment-suffix,$(SUFFIX)) $(call flag,--platform,$(PLATFORM)) $(call flag,--budget,$(BUDGET)) $(SET_FLAGS) $(call bool,--dry-run,$(DRY_RUN))

.PHONY: help setup doctor test models model-check serve-model install-models data data-check serve-forms \
  ideal-runs ideal-status experiments eval matrix submit report study studies reference-report inspect clean-logs

help: ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-16s %s\n", $$1, $$2}'
	@echo; echo "  Examples:"; \
	echo "    make eval MODEL=computer_use_gemini_35_flash_lowcost FORM=conf_interest RUN=1 SET=\"max_steps=4\""; \
	echo "    make matrix EXPERIMENT=fill_only_done_30 DRY_RUN=1"; \
	echo "    make submit EXPERIMENT=track_baseline_pilot SPLIT=model CHAIN=afterok"

## --- setup ------------------------------------------------------------------
setup: ## create .venv, install deps, Playwright MCP + Chromium (WITH=localhf|vllm|all)
	@FORMBENCH_ENV_READY= bash $(ROOT)/scripts/setup.sh $(call flag,--with,$(WITH))

doctor: ## check environment, dataset and registry; prints fixes
	@$(FB) doctor

test: ## run the unit tests
	@$(ROOT)/scripts/env.sh $(PY) -m unittest discover -s tests -t . $(if $(V),-v,)

## --- models -----------------------------------------------------------------
models: ## list registered models (ALL=1 includes legacy)
	@$(FB) models list $(call bool,--all,$(ALL))

model-check: ## preflight a model: registry, keys, weights, endpoint, GPUs (MODEL=, SMOKE=1)
	@$(FB) models check $(MODEL) $(call bool,--smoke,$(SMOKE)) $(call bool,--all,$(ALL))

serve-model: ## start a model's vLLM server in the foreground (MODEL=)
	@$(FB) models serve $(MODEL)

install-models: ## download local weights into models/ (MODEL= optional, DRY_RUN=1)
	@$(FB) models install $(MODEL) $(call bool,--dry-run,$(DRY_RUN))

## --- dataset + ideal runs -----------------------------------------------------
data: ## rebuild answers + LocalForms (FROM_CSV=1 also re-syncs specs) and validate
	@$(FB) data build $(call bool,--from-csv,$(FROM_CSV))

data-check: ## validate specs, answer sets and reference traces
	@$(FB) data check

serve-forms: ## serve the LocalForms site (http://127.0.0.1:$$LOCALFORMS_PORT)
	@$(FB) forms serve

ideal-runs: ## regenerate scripted Playwright reference runs (FORMS= RUNS= PLATFORM= OVERWRITE=1 SUBMIT=1)
	@$(FB) ideal $(call flag,--forms,$(FORMS)) $(call flag,--runs,$(RUNS)) $(call flag,--platform,$(PLATFORM)) \
	  $(call bool,--overwrite,$(OVERWRITE)) $(call bool,--headed,$(HEADED)) $(call bool,--submit,$(SUBMIT)) $(call bool,--dry-run,$(DRY_RUN))

ideal-status: ## reference-run coverage per form (PLATFORM=)
	@$(FB) ideal --status $(call flag,--platform,$(PLATFORM))

## --- evaluation ---------------------------------------------------------------
experiments: ## list experiment manifests in configs/experiments/
	@$(FB) experiments

eval: ## one trial (MODEL= FORM= RUN= PLATFORM= SET="k=v ..." DRY_RUN=1 SUBMIT=1)
	@$(FB) eval $(MODEL) $(FORM) $(RUN) $(call flag,--experiment-id,$(EXP_ID)) $(call flag,--platform,$(PLATFORM)) \
	  $(call flag,--budget,$(BUDGET)) $(SET_FLAGS) $(call bool,--dry-run,$(DRY_RUN)) $(call bool,--submit,$(SUBMIT))

matrix: ## run an experiment here (EXPERIMENT= or MODELS= FORMS= RUNS=; COHORT= DRY_RUN=1)
	@$(FB) matrix $(SELECT) $(call bool,--skip-checks,$(SKIP_CHECKS))

submit: ## submit an experiment to Slurm (same selection vars; SPLIT=cohort|model|run CHAIN=afterok)
	@$(FB) submit $(SELECT) --split $(SPLIT) $(call flag,--chain,$(CHAIN)) $(call flag,--after,$(AFTER))

## --- analytics ----------------------------------------------------------------
report: ## core analytics: CSV tables + SVG plots in docs/eval_results/analysis (EXPERIMENTS=)
	@$(FB) report $(call flag,--experiments,$(EXPERIMENTS))

reference-report: ## ideal-run dataset summary + plots in docs/eval_results/reference_analysis
	@$(FB) report --reference

studies: ## list thesis studies that `make study NAME=` can regenerate
	@$(FB) report --list-studies

study: ## regenerate one thesis study's outputs (NAME=)
	@$(FB) report --study $(NAME)

inspect: ## show one trial: summary (+ STEPS=all|0,3 and MEDIA=1)
	@$(FB) inspect $(TRIAL) $(if $(STEPS),--steps $(STEPS),) $(call bool,--media,$(MEDIA))

clean-logs: ## delete local logs/ (never touches data/)
	@rm -rf $(ROOT)/logs && echo "[OK] removed logs/"
