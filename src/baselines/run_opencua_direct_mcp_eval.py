"""Run OpenCUA through the strict Playwright MCP tool interface."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines import run_qwen_direct_mcp_eval


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--model-kind" not in args:
        args = ["--model-kind", "computer_use_agent", *args]
    return run_qwen_direct_mcp_eval.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
