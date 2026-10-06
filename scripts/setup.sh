#!/usr/bin/env bash
# Idempotent project setup (no sudo; everything project-local).
#
#   scripts/setup.sh [--with localhf|vllm|all] [--skip-browsers]
#
#   .venv            core Python env (requirements.txt [+ requirements-localhf.txt])
#   .venv-opencua    vLLM env for locally served models (requirements-vllm.txt), --with vllm
#   .node-tools      pinned @playwright/mcp (npm ci from the committed lockfile)
#   $CACHE_ROOT/playwright  Chromium for Python Playwright and for Playwright MCP
#
# Runs inside scripts/env.sh, so MODULES / CACHE_ROOT from .env apply.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${FORMBENCH_ENV_READY:-}" ]; then
  FORMBENCH_ENV_READY=1 exec "$ROOT_DIR/scripts/env.sh" bash "$0" "$@"
fi
cd "$ROOT_DIR"

WITH=""
SKIP_BROWSERS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --with) WITH="${2:-}"; shift 2 ;;
    --with=*) WITH="${1#*=}"; shift ;;
    --with-vllm) WITH="vllm"; shift ;;
    --skip-browsers) SKIP_BROWSERS=1; shift ;;
    *) echo "[FAIL] unknown option: $1" >&2; exit 2 ;;
  esac
done

step() { echo; echo "== $*"; }

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
case "$PYTHON_BIN" in /*) ;; *) PYTHON_BIN="$ROOT_DIR/$PYTHON_BIN" ;; esac
BASE_PYTHON="${BASE_PYTHON:-/usr/bin/python3}"
[ -x "$BASE_PYTHON" ] || BASE_PYTHON="$(command -v python3)"

step "core Python env ($PYTHON_BIN)"
if [ ! -x "$PYTHON_BIN" ]; then
  # Use the original library path: the system Python must not see module-provided OpenSSL.
  "$BASE_PYTHON" -m venv "$(dirname "$(dirname "$PYTHON_BIN")")"
fi
"$PYTHON_BIN" -m pip install --quiet --upgrade pip
REQ="requirements.txt"
if [ "$WITH" = "localhf" ] || [ "$WITH" = "all" ]; then
  REQ="requirements-localhf.txt"
fi
"$PYTHON_BIN" -m pip install --quiet -r "$REQ"
echo "[OK] installed $REQ"

if [ "$WITH" = "vllm" ] || [ "$WITH" = "all" ]; then
  VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-$ROOT_DIR/.venv-opencua/bin/python}"
  case "$VLLM_PYTHON_BIN" in /*) ;; *) VLLM_PYTHON_BIN="$ROOT_DIR/$VLLM_PYTHON_BIN" ;; esac
  step "vLLM env ($VLLM_PYTHON_BIN)"
  if [ ! -x "$VLLM_PYTHON_BIN" ]; then
    VLLM_BASE_PYTHON="${VLLM_BASE_PYTHON:-$(command -v python3.12 || true)}"
    if [ -z "$VLLM_BASE_PYTHON" ]; then
      echo "[FAIL] python3.12 not found; add a Python 3.12 module to MODULES in .env or set VLLM_BASE_PYTHON" >&2
      exit 1
    fi
    LD_LIBRARY_PATH="${MODULE_LD_LIBRARY_PATH:-${LD_LIBRARY_PATH-}}" "$VLLM_BASE_PYTHON" -m venv "$(dirname "$(dirname "$VLLM_PYTHON_BIN")")"
  fi
  LD_LIBRARY_PATH="${MODULE_LD_LIBRARY_PATH:-${LD_LIBRARY_PATH-}}" "$VLLM_PYTHON_BIN" -m pip install --quiet --upgrade pip
  LD_LIBRARY_PATH="${MODULE_LD_LIBRARY_PATH:-${LD_LIBRARY_PATH-}}" "$VLLM_PYTHON_BIN" -m pip install --quiet -r requirements-vllm.txt
  echo "[OK] installed requirements-vllm.txt"
fi

step "Playwright MCP (Node)"
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "[FAIL] node/npm not found. On HPC set MODULES (e.g. 'release/25.06 GCCcore/13.3.0 nodejs/20.13.1') in .env; locally install Node >= 18." >&2
  exit 1
fi
if [ ! -x .node-tools/node_modules/.bin/playwright-mcp ]; then
  LD_LIBRARY_PATH="${MODULE_LD_LIBRARY_PATH:-${LD_LIBRARY_PATH-}}" npm ci --prefix .node-tools --no-audit --no-fund
fi
echo "[OK] $(LD_LIBRARY_PATH="${MODULE_LD_LIBRARY_PATH:-${LD_LIBRARY_PATH-}}" .node-tools/node_modules/.bin/playwright-mcp --version 2>/dev/null || echo 'playwright-mcp installed')"

if [ "$SKIP_BROWSERS" = "0" ]; then
  step "Chromium (Python Playwright + Playwright MCP) -> $PLAYWRIGHT_BROWSERS_PATH"
  "$PYTHON_BIN" -m playwright install chromium
  LD_LIBRARY_PATH="${MODULE_LD_LIBRARY_PATH:-${LD_LIBRARY_PATH-}}" \
    PLAYWRIGHT_MCP_INSTALL_LOG="${LOGS_DIR:-$ROOT_DIR/logs}/playwright-mcp-install.log" \
    bash scripts/ensure_playwright_mcp_runtime.sh
fi

step "doctor"
"$PYTHON_BIN" -m formbench doctor || true
echo
echo "[OK] setup complete. Next: make test && make models"
