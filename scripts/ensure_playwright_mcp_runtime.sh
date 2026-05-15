#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

if ! command -v node >/dev/null 2>&1; then
  echo "[FAIL] node is required to verify the Playwright MCP browser runtime" >&2
  exit 1
fi

if [ ! -d "$ROOT_DIR/.node-tools/node_modules" ]; then
  echo "[FAIL] missing node tool dependencies: $ROOT_DIR/.node-tools/node_modules" >&2
  exit 1
fi

CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/.runtime-cache}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CACHE_ROOT/playwright}"
mkdir -p "$PLAYWRIGHT_BROWSERS_PATH"

EXECUTABLE_RECORD="$PLAYWRIGHT_BROWSERS_PATH/.mcp-chromium-executable"
INSTALL_LOG="${PLAYWRIGHT_MCP_INSTALL_LOG:-/tmp/playwright-mcp-install-${USER:-unknown}.log}"
FALLBACK_BROWSERS_PATH="${PLAYWRIGHT_MCP_FALLBACK_BROWSERS_PATH:-$(dirname "$ROOT_DIR")/cache/playwright}"

resolve_runtime_json() {
  node <<'NODE'
const fs = require('fs');
const path = require('path');

const root = process.cwd();
const searchRoot = path.join(root, '.node-tools', 'node_modules');
const mcpPkg = require.resolve('@playwright/mcp/package.json', { paths: [searchRoot] });
const mcpDir = path.dirname(mcpPkg);
const playwrightPkg = require.resolve('playwright/package.json', { paths: [mcpDir] });
const playwrightMain = require.resolve('playwright', { paths: [mcpDir] });
const playwrightDir = path.dirname(playwrightPkg);
const playwright = require(playwrightMain);
const executablePath = playwright.chromium.executablePath();

process.stdout.write(JSON.stringify({
  mcpPackage: mcpPkg,
  playwrightPackage: playwrightPkg,
  playwrightVersion: require(playwrightPkg).version,
  playwrightCli: path.join(playwrightDir, 'cli.js'),
  executablePath,
  executableExists: fs.existsSync(executablePath),
  browsersPath: process.env.PLAYWRIGHT_BROWSERS_PATH || ''
}));
NODE
}

runtime_json="$(resolve_runtime_json)"
playwright_version="$(node -e 'const data=JSON.parse(process.argv[1]); process.stdout.write(data.playwrightVersion)' "$runtime_json")"
playwright_cli="$(node -e 'const data=JSON.parse(process.argv[1]); process.stdout.write(data.playwrightCli)' "$runtime_json")"
chromium_executable="$(node -e 'const data=JSON.parse(process.argv[1]); process.stdout.write(data.executablePath)' "$runtime_json")"
chromium_exists="$(node -e 'const data=JSON.parse(process.argv[1]); process.stdout.write(String(data.executableExists))' "$runtime_json")"

echo "[INFO] playwright_mcp_runtime playwright_version=$playwright_version browsers_path=$PLAYWRIGHT_BROWSERS_PATH expected_chromium=$chromium_executable exists=$chromium_exists"

copy_runtime_from_fallback() {
  expected_dir="$1"
  if [ -z "$FALLBACK_BROWSERS_PATH" ] || [ ! -d "$FALLBACK_BROWSERS_PATH" ]; then
    return 0
  fi
  mkdir -p "$PLAYWRIGHT_BROWSERS_PATH"
  if [ -n "$expected_dir" ] && [ -d "$FALLBACK_BROWSERS_PATH/$expected_dir" ] && [ ! -e "$PLAYWRIGHT_BROWSERS_PATH/$expected_dir" ]; then
    echo "[INFO] copying_playwright_mcp_runtime from=$FALLBACK_BROWSERS_PATH to=$PLAYWRIGHT_BROWSERS_PATH expected_dir=$expected_dir"
    cp -a "$FALLBACK_BROWSERS_PATH/$expected_dir" "$PLAYWRIGHT_BROWSERS_PATH/" || true
  fi
  for sibling_path in "$FALLBACK_BROWSERS_PATH"/chromium_headless_shell-* "$FALLBACK_BROWSERS_PATH"/ffmpeg-* "$FALLBACK_BROWSERS_PATH"/.links; do
    if [ -e "$sibling_path" ]; then
      sibling="$(basename "$sibling_path")"
      if [ ! -e "$PLAYWRIGHT_BROWSERS_PATH/$sibling" ]; then
        echo "[INFO] copying_playwright_mcp_runtime_asset from=$sibling_path to=$PLAYWRIGHT_BROWSERS_PATH/$sibling"
        cp -a "$sibling_path" "$PLAYWRIGHT_BROWSERS_PATH/" || true
      fi
    fi
  done
}

expected_dir="$(basename "$(dirname "$(dirname "$chromium_executable")")")"
copy_runtime_from_fallback "$expected_dir"

if [ "$chromium_exists" != "true" ]; then
  runtime_json="$(resolve_runtime_json)"
  chromium_executable="$(node -e 'const data=JSON.parse(process.argv[1]); process.stdout.write(data.executablePath)' "$runtime_json")"
  chromium_exists="$(node -e 'const data=JSON.parse(process.argv[1]); process.stdout.write(String(data.executableExists))' "$runtime_json")"
fi

if [ "$chromium_exists" != "true" ]; then
  echo "[INFO] installing_playwright_mcp_runtime cli=$playwright_cli log=$INSTALL_LOG"
  PLAYWRIGHT_BROWSERS_PATH="$PLAYWRIGHT_BROWSERS_PATH" \
    node "$playwright_cli" install chromium ffmpeg >"$INSTALL_LOG" 2>&1 || {
      echo "[FAIL] Playwright MCP browser install failed" >&2
      tail -n 160 "$INSTALL_LOG" >&2 || true
      exit 1
    }
fi

runtime_json="$(resolve_runtime_json)"
chromium_executable="$(node -e 'const data=JSON.parse(process.argv[1]); process.stdout.write(data.executablePath)' "$runtime_json")"
chromium_exists="$(node -e 'const data=JSON.parse(process.argv[1]); process.stdout.write(String(data.executableExists))' "$runtime_json")"

if [ "$chromium_exists" != "true" ]; then
  echo "[FAIL] Playwright MCP Chromium executable is still missing after install: $chromium_executable" >&2
  tail -n 160 "$INSTALL_LOG" >&2 || true
  exit 1
fi

printf '%s\n' "$chromium_executable" >"$EXECUTABLE_RECORD"
echo "[INFO] playwright_mcp_runtime_ready executable=$chromium_executable record=$EXECUTABLE_RECORD"
