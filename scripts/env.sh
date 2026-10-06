#!/usr/bin/env bash
# Run a command inside the project environment:  scripts/env.sh <command> [args...]
#
#  - loads .env (see .env.example) without overriding variables already set
#  - `module load $MODULES` when an environment-modules/Lmod system is present
#    (HPC: Node for Playwright MCP, Python 3.12 runtime for the vLLM venv)
#  - keeps the module LD_LIBRARY_PATH in MODULE_LD_LIBRARY_PATH and restores the
#    original one: the main .venv (system Python) breaks with the module OpenSSL,
#    while Node/vLLM subprocesses need it (formbench passes it to them explicitly)
#  - points every cache (HF, pip, uv, XDG, Playwright browsers) at $CACHE_ROOT
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/.env" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"
    [ -z "$line" ] || [ "${line:0:1}" = "#" ] && continue
    line="${line#export }"
    key="${line%%=*}"
    value="${line#*=}"
    [ "$key" = "$line" ] && continue
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    value="${value%\"}"; value="${value#\"}"; value="${value%\'}"; value="${value#\'}"
    if [ -z "${!key+x}" ]; then
      export "$key=$value"
    fi
  done <"$ROOT_DIR/.env"
fi

ORIGINAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
if [ -n "${MODULES:-}" ] && [ -z "${FORMBENCH_MODULES_LOADED:-}" ]; then
  if ! type module >/dev/null 2>&1; then
    for init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
      # shellcheck disable=SC1090
      [ -f "$init" ] && source "$init" && break
    done
  fi
  if type module >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    module load $MODULES
    export FORMBENCH_MODULES_LOADED=1
  else
    echo "[WARN] MODULES is set but no 'module' command is available; skipping module load" >&2
  fi
fi
export MODULE_LD_LIBRARY_PATH="${MODULE_LD_LIBRARY_PATH:-${LD_LIBRARY_PATH-}}"
export NODE_LD_LIBRARY_PATH_FOR_MCP="${NODE_LD_LIBRARY_PATH_FOR_MCP:-$MODULE_LD_LIBRARY_PATH}"
if [ -n "$ORIGINAL_LD_LIBRARY_PATH" ]; then
  export LD_LIBRARY_PATH="$ORIGINAL_LD_LIBRARY_PATH"
else
  unset LD_LIBRARY_PATH
fi

CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/.runtime-cache}"
case "$CACHE_ROOT" in /*) ;; *) CACHE_ROOT="$ROOT_DIR/$CACHE_ROOT" ;; esac
export CACHE_ROOT
mkdir -p "$CACHE_ROOT"/{xdg,hf,pip,uv,playwright}
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CACHE_ROOT/playwright}"
if [ -z "${PLAYWRIGHT_MCP_CHROMIUM_EXECUTABLE:-}" ] && [ -s "$PLAYWRIGHT_BROWSERS_PATH/.mcp-chromium-executable" ]; then
  recorded="$(cat "$PLAYWRIGHT_BROWSERS_PATH/.mcp-chromium-executable")"
  # Ignore a stale record (e.g. after the workspace moved); `make setup` re-resolves it.
  if [ -x "$recorded" ]; then
    export PLAYWRIGHT_MCP_CHROMIUM_EXECUTABLE="$recorded"
  fi
fi

export PATH="$ROOT_DIR/.node-tools/node_modules/.bin:$PATH"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

exec "$@"
