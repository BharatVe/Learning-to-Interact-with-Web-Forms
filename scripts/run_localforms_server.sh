#!/usr/bin/env bash
# Starts the FormFactory-style LocalForms Flask app (evaluation_additions/formfactory_import/site).
# Mirrors the readiness/env pattern already used for scripts/run_opencua_vllm_server.sh.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${LOCALFORMS_HOST:-127.0.0.1}"
PORT="${LOCALFORMS_PORT:-5000}"
# .venv (bare system Python 3.9) is ABI-incompatible with the OpenSSL brought in
# by `module load release/25.06 GCCcore/13.3.0 ...` on compute nodes (hashlib
# import fails: OPENSSL_3.4.0 not found). .venv-opencua's Python is the
# module-provided 3.12.3 interpreter and is already used under that same module
# load for the rest of the OpenCUA job, so it is the ABI-compatible choice here.
PYTHON_BIN="${LOCALFORMS_PYTHON_BIN:-$ROOT_DIR/.venv-opencua/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[FAIL] localforms python interpreter missing: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" -c "import flask" || {
  echo "[FAIL] flask not importable in $PYTHON_BIN" >&2
  exit 1
}

echo "[INFO] localforms_host=$HOST"
echo "[INFO] localforms_port=$PORT"
exec "$PYTHON_BIN" evaluation_additions/formfactory_import/site/app.py --host "$HOST" --port "$PORT"
