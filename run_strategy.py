#!/usr/bin/env python3
"""
Q23 entrypoint: runs one or more strategies (local).

Reads toggles from environment:
  Q23_RUN_V4=true/false
  Q23_RUN_NASNYS1010=true/false
  Q23_RUN_Q23LSNEW3=true/false
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y", "on")


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _ensure_env() -> None:
    """
    If launched outside the .command runner, try to load .env (KEY=VALUE)
    with a tiny parser (no external dependencies).
    """
    root = _project_root()
    env_path = root / ".env"

    # Ensure cwd stable
    os.chdir(root)

    # Ensure imports stable
    src_path = str(root / "src")
    os.environ["PYTHONPATH"] = f"{src_path}:{os.environ.get('PYTHONPATH', '')}"

    if os.environ.get("API_KEY", "").strip():
        return
    if not env_path.exists():
        return

    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def _require_api_key() -> None:
    api_key = os.environ.get("API_KEY", "").strip()
    if not api_key:
        print(
            "\nERROR: API_KEY is not set.\n"
            "Put it in /Users/svenbenson/Q23_QUANT_SYSTEM/.env as:\n"
            "  API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx\n",
            file=sys.stderr,
        )
        raise SystemExit(2)


@dataclass
class RunResult:
    name: str
    rc: int
    message: str = ""


def _run_v4(tag: Optional[str] = None) -> RunResult:
    from q23.strategy.engine import StrategyEngine

    eng = StrategyEngine()
    artifacts = eng.run(tag=tag)

    outdir = None
    if isinstance(artifacts, dict):
        outdir = artifacts.get("output_dir")
    else:
        outdir = getattr(artifacts, "output_dir", None)

    msg = f"v4 done. output_dir={outdir}" if outdir else "v4 done."
    return RunResult(name="v4", rc=0, message=msg)


def _run_legacy_stub(name: str) -> RunResult:
    return RunResult(name=name, rc=0, message="(legacy not wired yet)")


def main(argv: Optional[list[str]] = None) -> int:
    _ensure_env()
    _require_api_key()

    tag = os.environ.get("Q23_TAG", "").strip() or None

    run_v4 = _str2bool(os.environ.get("Q23_RUN_V4", "true"))
    run_nasnys1010 = _str2bool(os.environ.get("Q23_RUN_NASNYS1010", "false"))
    run_q23lsnew3 = _str2bool(os.environ.get("Q23_RUN_Q23LSNEW3", "false"))

    print("")
    print("Q23 strategy runner")
    print(f"  tag: {tag}")
    print(f"  Q23_RUN_V4: {run_v4}")
    print(f"  Q23_RUN_NASNYS1010: {run_nasnys1010}")
    print(f"  Q23_RUN_Q23LSNEW3: {run_q23lsnew3}")
    print("")

    rc = 0

    if run_v4:
        print(">> Running v4 modular strategy...")
        r = _run_v4(tag=tag)
        print(f">> {r.name}: {r.message}")
        rc = max(rc, r.rc)

    if run_nasnys1010:
        print(">> Running NASNYS1010 (legacy)...")
        r = _run_legacy_stub("NASNYS1010")
        print(f">> {r.name}: {r.message}")
        rc = max(rc, r.rc)

    if run_q23lsnew3:
        print(">> Running Q23LSNEW3 (legacy)...")
        r = _run_legacy_stub("Q23LSNEW3")
        print(f">> {r.name}: {r.message}")
        rc = max(rc, r.rc)

    print("")
    print("Done.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
