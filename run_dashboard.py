#!/usr/bin/env python3
"""Q23 entrypoint: launch the Streamlit dashboard.

Behavior:
- Prefer modular dashboard app if present: src/q23/dashboard/app.py
- Fallback to legacy monolithic dashboard in project root: Q23_dashboard.py
- Runs Streamlit via subprocess so it's environment/CLI friendly

Usage:
  python run_dashboard.py
  python run_dashboard.py -- --server.port 8502

Note: anything after `--` is passed directly to `streamlit run`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _pick_app_path(root: Path) -> Path:
    p_mod = root / "src" / "q23" / "dashboard" / "app.py"
    if p_mod.exists():
        return p_mod

    p_legacy = root / "Q23_dashboard.py"
    if p_legacy.exists():
        return p_legacy

    p_legacy2 = Path.cwd() / "Q23_dashboard.py"
    if p_legacy2.exists():
        return p_legacy2

    raise FileNotFoundError(
        "Could not find a dashboard entry. Expected either src/q23/dashboard/app.py or Q23_dashboard.py"
    )


def main(argv: List[str]) -> int:
    if shutil.which("streamlit") is None:
        print("[error] streamlit is not installed or not on PATH for this environment.")
        print("        Install inside your conda env:")
        print("          pip install streamlit 'altair<5' standard-imghdr 'click==8.0.4'")
        return 2

    root = _project_root()
    os.chdir(root)

    # ensure q23 imports resolve inside streamlit process too
    src_path = str(root / "src")
    os.environ["PYTHONPATH"] = f"{src_path}:{os.environ.get('PYTHONPATH', '')}"

    passthrough: List[str] = []
    if "--" in argv:
        i = argv.index("--")
        passthrough = argv[i + 1 :]

    app_path = _pick_app_path(root)

    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    cmd = ["streamlit", "run", str(app_path), *passthrough]
    print("[run]", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
