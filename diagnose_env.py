#!/usr/bin/env python3
"""
Q23 Environment Diagnostic

Checks:
- Python version / executable
- Conda environment
- Key environment variables
- Streamlit availability + version
- qnt package availability
- xarray / pandas / numpy versions
- PYTHONPATH sanity
"""

from __future__ import annotations

import os
import sys
import shutil
import platform
from importlib.util import find_spec


def header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def kv(k: str, v) -> None:
    print(f"{k:<28}: {v}")


def check_import(name: str):
    spec = find_spec(name)
    return "OK" if spec is not None else "NOT FOUND"


def main() -> int:
    header("Q23 ENVIRONMENT DIAGNOSTIC")

    # ------------------------------------------------------------------
    header("Python")
    kv("sys.executable", sys.executable)
    kv("Python version", sys.version.replace("\n", " "))
    kv("Platform", platform.platform())

    # ------------------------------------------------------------------
    header("Conda / Virtual Env")
    kv("CONDA_DEFAULT_ENV", os.environ.get("CONDA_DEFAULT_ENV"))
    kv("CONDA_PREFIX", os.environ.get("CONDA_PREFIX"))

    # ------------------------------------------------------------------
    header("Key Environment Variables")
    keys = [
        "API_KEY",
        "Q23_PROJECT_DIR",
        "Q23_CONDA_ENV",
        "Q23_CONDA_SH",
        "PYTHONPATH",
    ]
    for k in keys:
        v = os.environ.get(k)
        if k == "API_KEY":
            v = "SET" if v else "NOT SET"
        kv(k, v)

    # ------------------------------------------------------------------
    header("Executable Availability")
    kv("python", shutil.which("python"))
    kv("pip", shutil.which("pip"))
    kv("streamlit", shutil.which("streamlit"))

    # ------------------------------------------------------------------
    header("Package Imports")
    pkgs = [
        "streamlit",
        "qnt",
        "xarray",
        "pandas",
        "numpy",
    ]
    for p in pkgs:
        kv(p, check_import(p))

    # ------------------------------------------------------------------
    header("Package Versions")
    try:
        import streamlit
        kv("streamlit.__version__", streamlit.__version__)
    except Exception as e:
        kv("streamlit.__version__", f"ERROR ({e})")

    try:
        import xarray
        kv("xarray.__version__", xarray.__version__)
    except Exception as e:
        kv("xarray.__version__", f"ERROR ({e})")

    try:
        import pandas
        kv("pandas.__version__", pandas.__version__)
    except Exception as e:
        kv("pandas.__version__", f"ERROR ({e})")

    try:
        import numpy
        kv("numpy.__version__", numpy.__version__)
    except Exception as e:
        kv("numpy.__version__", f"ERROR ({e})")

    try:
        import qnt
        kv("qnt.__version__", getattr(qnt, "__version__", "unknown"))
    except Exception as e:
        kv("qnt.__version__", f"ERROR ({e})")

    # ------------------------------------------------------------------
    header("Summary")
    problems = []

    if not os.environ.get("API_KEY"):
        problems.append("API_KEY is not set")

    if shutil.which("streamlit") is None:
        problems.append("streamlit executable not found")

    if find_spec("qnt") is None:
        problems.append("qnt package not importable")

    if problems:
        print("⚠️  Issues detected:")
        for p in problems:
            print(f" - {p}")
        return 2

    print("✅ Environment looks sane for Q23.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
