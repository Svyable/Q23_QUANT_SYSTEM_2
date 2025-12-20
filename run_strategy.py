#!/usr/bin/env python3
"""
Q23 entrypoint: runs one or more strategies (local).

Configuration sources (in order of priority):
  1. enabled_strategies.json - Managed by Strategy Warehouse in dashboard (RECOMMENDED)
  2. Environment variables - Q23_RUN_<strategy_id>=true/false (DEPRECATED, kept for backward compatibility)

The enabled_strategies.json file is the PRIMARY and RECOMMENDED way to enable/disable strategies.
It is created and updated by the Strategy Warehouse page in the dashboard.

NOTE: The .env file is ONLY used for API_KEY and other environment variables (Q23_CONDA_ENV, etc.).
Strategy enabling should be done via enabled_strategies.json, NOT via .env file.

Environment variables (Q23_RUN_<strategy_id>) can still override JSON config if explicitly set,
but this is deprecated. Use the Strategy Warehouse dashboard instead.

Freshness Check:
  By default, strategies with a run from today (YYYYMMDD_* tag) are skipped to save time.
  Use Q23_FORCE_RERUN=true to force re-running all strategies regardless of freshness.

Benchmark Auto-Run:
  Benchmarks are automatically run after regular strategies to ensure fresh comparison data.
  Use Q23_SKIP_BENCHMARKS=true to skip benchmark auto-run.
  Use Q23_AUTO_RUN_BENCHMARKS=false to disable benchmark auto-run entirely.

Debug Mode:
  Use Q23_DEBUG_RUNS=true for detailed logging of freshness checks and run decisions.

Legacy example (DEPRECATED - use Strategy Warehouse instead):
  Q23_RUN_V4=true
  Q23_RUN_nasnys_v4=true
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Configuration
# =============================================================================

def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y", "on")


def _project_root() -> Path:
    return Path(__file__).resolve().parent


# Debug mode for detailed logging
DEBUG_RUNS = _str2bool(os.environ.get("Q23_DEBUG_RUNS", "false"))


def _debug(msg: str) -> None:
    """Print debug message if debug mode is enabled."""
    if DEBUG_RUNS:
        print(f"  [DEBUG] {msg}")


# =============================================================================
# Tag Parsing and Normalization
# =============================================================================

def _parse_tag_date(tag: str) -> Optional[datetime]:
    """Parse date from tag, handling multiple formats.
    
    Supported formats:
    - YYYYMMDD_HHMMSS (e.g., 20251219_112716)
    - YYYYMMDD (e.g., 20251219)
    - YYYY-MM-DD_HHMMSS (e.g., 2025-12-19_112716)
    - YYYY-MM-DD (e.g., 2025-12-19)
    
    Args:
        tag: Tag string to parse
        
    Returns:
        datetime if successfully parsed, None otherwise
    """
    if not tag:
        return None
    
    # Try each format
    formats = [
        ("%Y%m%d_%H%M%S", 15),   # 20251219_112716
        ("%Y%m%d", 8),           # 20251219
        ("%Y-%m-%d_%H%M%S", 17), # 2025-12-19_112716
        ("%Y-%m-%d", 10),        # 2025-12-19
    ]
    
    for fmt, length in formats:
        try:
            # Only try to parse the expected length
            parse_str = tag[:length] if len(tag) >= length else tag
            return datetime.strptime(parse_str, fmt)
        except (ValueError, IndexError):
            continue
    
    # Try to extract just the date portion (first 8 digits if present)
    digits = ''.join(c for c in tag[:15] if c.isdigit())
    if len(digits) >= 8:
        try:
            return datetime.strptime(digits[:8], "%Y%m%d")
        except ValueError:
            pass
    
    return None


def _normalize_tag(tag: Optional[str] = None) -> str:
    """Normalize tag to standard format: YYYYMMDD_HHMMSS.
    
    Args:
        tag: Existing tag to normalize (None = generate new)
        
    Returns:
        Normalized tag string
    """
    if tag:
        tag_date = _parse_tag_date(tag)
        if tag_date:
            return tag_date.strftime("%Y%m%d_%H%M%S")
    
    # Generate new tag with current timestamp
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _is_tag_from_today(tag: str, today: datetime) -> bool:
    """Check if tag represents a run from today.
    
    Args:
        tag: Tag string to check
        today: Today's datetime
        
    Returns:
        True if tag's date matches today's date
    """
    tag_date = _parse_tag_date(tag)
    if tag_date is None:
        _debug(f"Could not parse tag date: {tag}")
        return False
    
    is_today = tag_date.date() == today.date()
    _debug(f"Tag '{tag}' date={tag_date.date()}, today={today.date()}, is_today={is_today}")
    return is_today


def _get_tag_age_days(tag: str) -> Optional[int]:
    """Get age of tag in days.
    
    Args:
        tag: Tag string to check
        
    Returns:
        Number of days since tag date, or None if unparseable
    """
    tag_date = _parse_tag_date(tag)
    if tag_date is None:
        return None
    return (datetime.now().date() - tag_date.date()).days


# =============================================================================
# Environment and API Key
# =============================================================================

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
            "Put it in /Users/svenbenson/Q23_QUANT_SYSTEM 2/.env as:\n"
            "  API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx\n",
            file=sys.stderr,
        )
        raise SystemExit(2)


# =============================================================================
# Strategy Running
# =============================================================================

@dataclass
class RunResult:
    name: str
    rc: int
    message: str = ""


def _run_legacy_v4(tag: Optional[str] = None) -> RunResult:
    """Run legacy StrategyEngine (backward compatibility)."""
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


def _run_strategy(strategy_id: str, tag: Optional[str] = None) -> RunResult:
    """Run a registered strategy from StrategyRegistry."""
    try:
        from q23.strategies.registry import StrategyRegistry
        
        strategy = StrategyRegistry.get_instance(strategy_id)
        artifacts = strategy.run(tag=tag, write_outputs=True)
        
        output_dir = strategy.get_output_dir()
        msg = f"{strategy_id} done. output_dir={output_dir}"
        return RunResult(name=strategy_id, rc=0, message=msg)
    except Exception as e:
        error_msg = f"Failed to run {strategy_id}: {str(e)}"
        print(f"ERROR: {error_msg}", file=sys.stderr)
        import traceback
        if DEBUG_RUNS:
            traceback.print_exc()
        return RunResult(name=strategy_id, rc=1, message=error_msg)


# =============================================================================
# Freshness Detection
# =============================================================================

def _has_fresh_run(strategy_id: str, today: datetime) -> Tuple[bool, Optional[str]]:
    """Check if strategy has a run from today.
    
    Uses robust date parsing to handle multiple tag formats.
    
    Args:
        strategy_id: The strategy ID to check
        today: Today's datetime
    
    Returns:
        Tuple of (is_fresh, latest_tag) where:
        - is_fresh: True if latest tag is from today
        - latest_tag: The most recent tag found (or None if no tags)
    """
    try:
        from q23.dashboard.core import discover_strategy_tags
        
        tags = discover_strategy_tags(strategy_id)
        if not tags:
            _debug(f"{strategy_id}: No tags found")
            return False, None
        
        latest_tag = tags[0]  # Tags are sorted by date, newest first
        is_fresh = _is_tag_from_today(latest_tag, today)
        
        if DEBUG_RUNS:
            age = _get_tag_age_days(latest_tag)
            age_str = f"{age} days old" if age is not None else "unknown age"
            _debug(f"{strategy_id}: latest_tag={latest_tag} ({age_str}), is_fresh={is_fresh}")
        
        return is_fresh, latest_tag
    except Exception as e:
        _debug(f"{strategy_id}: Error checking freshness: {e}")
        # If we can't check, assume not fresh (will run)
        return False, None


def _get_freshness_status(
    strategy_ids: List[str],
    today: datetime,
) -> Dict[str, Tuple[bool, Optional[str]]]:
    """Get freshness status for all strategies.
    
    Args:
        strategy_ids: List of strategy IDs to check
        today: Today's datetime
    
    Returns:
        Dict mapping strategy_id -> (is_fresh, latest_tag)
    """
    status = {}
    for strategy_id in strategy_ids:
        status[strategy_id] = _has_fresh_run(strategy_id, today)
    return status


# =============================================================================
# Strategy Configuration
# =============================================================================

def _get_env_var_name(strategy_id: str) -> str:
    """Convert strategy_id to environment variable name (Q23_RUN_<strategy_id>)."""
    return f"Q23_RUN_{strategy_id.upper()}"


def _get_enabled_strategies_config() -> Dict[str, bool]:
    """
    Load enabled strategies from JSON config.
    
    The config file is managed by the Strategy Warehouse in the dashboard.
    Returns empty dict if file doesn't exist.
    """
    config_path = _project_root() / "src" / "q23" / "strategies" / "enabled_strategies.json"
    
    if not config_path.exists():
        return {}
    
    try:
        data = json.loads(config_path.read_text())
        return data.get("enabled", {})
    except Exception as e:
        print(f"WARNING: Could not load enabled_strategies.json: {e}", file=sys.stderr)
        return {}


def _get_strategy_enabled_status(strategy_id: str, json_config: Dict[str, bool]) -> bool:
    """
    Check if a strategy is enabled.
    
    Priority:
    1. Environment variable (if explicitly set) - DEPRECATED, kept for backward compatibility
    2. JSON config (enabled_strategies.json) - RECOMMENDED
    3. Default (False)
    
    Note: Environment variable override is deprecated. Use Strategy Warehouse dashboard instead.
    """
    env_var = _get_env_var_name(strategy_id)
    env_value = os.environ.get(env_var, "").strip()
    
    # If env var is explicitly set, use it (but warn about deprecation)
    if env_value:
        print(
            f"WARNING: Using deprecated environment variable {env_var} for strategy enabling. "
            f"Please use Strategy Warehouse dashboard (enabled_strategies.json) instead.",
            file=sys.stderr,
        )
        return _str2bool(env_value)
    
    # Otherwise use JSON config (recommended)
    return json_config.get(strategy_id, False)


# =============================================================================
# Benchmark Auto-Run
# =============================================================================

def _get_benchmark_strategies() -> List[str]:
    """Get list of all benchmark strategy IDs."""
    try:
        from q23.strategies.registry import StrategyRegistry
        all_ids = StrategyRegistry.list_ids()
        return [s for s in all_ids if s.startswith("benchmark_")]
    except Exception:
        return []


def _get_regular_strategies() -> List[str]:
    """Get list of all non-benchmark strategy IDs."""
    try:
        from q23.strategies.registry import StrategyRegistry
        all_ids = StrategyRegistry.list_ids()
        return [s for s in all_ids if not s.startswith("benchmark_")]
    except Exception:
        return []


def _ensure_benchmarks_fresh(
    today: datetime,
    required_benchmarks: Optional[List[str]] = None,
    force_rerun: bool = False,
) -> int:
    """Ensure benchmark strategies are fresh.
    
    Runs any benchmarks that don't have a run from today.
    
    Args:
        today: Today's datetime
        required_benchmarks: List of benchmark IDs to ensure (None = all)
        force_rerun: Force re-run all benchmarks regardless of freshness
    
    Returns:
        Number of benchmarks that failed
    """
    if required_benchmarks is None:
        required_benchmarks = _get_benchmark_strategies()
    
    if not required_benchmarks:
        _debug("No benchmark strategies found")
        return 0
    
    print("")
    print("===== Benchmark Auto-Run =====")
    
    failed = 0
    ran = 0
    skipped = 0
    
    for bench_id in required_benchmarks:
        is_fresh, latest_tag = _has_fresh_run(bench_id, today)
        
        if is_fresh and not force_rerun:
            _debug(f"{bench_id}: Fresh (tag={latest_tag}), skipping")
            skipped += 1
            continue
        
        status = "stale" if latest_tag else "missing"
        print(f">> Auto-running benchmark {bench_id} ({status}: {latest_tag or 'no data'})")
        
        result = _run_strategy(bench_id, tag=None)
        if result.rc != 0:
            print(f">> {bench_id}: FAILED - {result.message}")
            failed += 1
        else:
            print(f">> {bench_id}: {result.message}")
            ran += 1
    
    print("")
    if ran > 0 or skipped > 0:
        print(f"  Benchmarks: {ran} ran, {skipped} skipped (fresh), {failed} failed")
    else:
        print("  No benchmarks needed to run")
    
    return failed


# =============================================================================
# Main Entry Point
# =============================================================================

def main(argv: Optional[list[str]] = None) -> int:
    global DEBUG_RUNS
    
    _ensure_env()
    _require_api_key()

    # Configuration
    tag = os.environ.get("Q23_TAG", "").strip() or None
    force_rerun = _str2bool(os.environ.get("Q23_FORCE_RERUN", "false"))
    DEBUG_RUNS = _str2bool(os.environ.get("Q23_DEBUG_RUNS", "false"))
    auto_run_benchmarks = _str2bool(os.environ.get("Q23_AUTO_RUN_BENCHMARKS", "true"))
    skip_benchmarks = _str2bool(os.environ.get("Q23_SKIP_BENCHMARKS", "false"))
    
    today = datetime.now()
    today_prefix = today.strftime("%Y%m%d")

    # Discover available strategies from registry
    all_strategies = []
    try:
        from q23.strategies.registry import StrategyRegistry
        all_strategies = StrategyRegistry.list_ids()
    except Exception as e:
        print(f"WARNING: Could not discover strategies: {e}", file=sys.stderr)

    # Separate regular strategies from benchmarks
    regular_strategies = [s for s in all_strategies if not s.startswith("benchmark_")]
    benchmark_strategies = [s for s in all_strategies if s.startswith("benchmark_")]

    # Load JSON config from Strategy Warehouse
    json_config = _get_enabled_strategies_config()
    config_source = "enabled_strategies.json" if json_config else "environment variables"
    
    # Check legacy StrategyEngine flag (backward compatibility)
    run_legacy_v4 = _str2bool(os.environ.get("Q23_RUN_V4", "false"))

    # Check flags for each registered strategy (JSON config + env vars)
    strategy_flags = {}
    for strategy_id in regular_strategies:
        strategy_flags[strategy_id] = _get_strategy_enabled_status(strategy_id, json_config)

    # Get freshness status for regular strategies
    freshness_status = _get_freshness_status(regular_strategies, today)

    # Print status header
    print("")
    print("=" * 60)
    print("Q23 Strategy Runner")
    print("=" * 60)
    print(f"  Config source: {config_source}")
    print(f"  Tag: {tag or '(auto-generated)'}")
    print(f"  Force rerun: {force_rerun}")
    print(f"  Today: {today_prefix} ({today.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"  Debug mode: {DEBUG_RUNS}")
    print(f"  Auto-run benchmarks: {auto_run_benchmarks and not skip_benchmarks}")
    print(f"  Q23_RUN_V4 (legacy): {run_legacy_v4}")
    print("")
    
    # Print strategy status with freshness info
    print("  Regular Strategies:")
    strategies_to_run = []
    strategies_to_skip = []
    
    for strategy_id in regular_strategies:
        enabled = strategy_flags.get(strategy_id, False)
        is_fresh, latest_tag = freshness_status.get(strategy_id, (False, None))
        
        if not enabled:
            status_str = "disabled"
        elif force_rerun:
            status_str = "enabled (force rerun)"
            strategies_to_run.append(strategy_id)
        elif is_fresh:
            status_str = f"enabled (fresh: {latest_tag} - skipping)"
            strategies_to_skip.append(strategy_id)
        elif latest_tag:
            age = _get_tag_age_days(latest_tag)
            age_str = f", {age}d old" if age is not None else ""
            status_str = f"enabled (stale: {latest_tag}{age_str} - will run)"
            strategies_to_run.append(strategy_id)
        else:
            status_str = "enabled (no data - will run)"
            strategies_to_run.append(strategy_id)
        
        print(f"    {strategy_id}: {status_str}")
    
    # Print benchmark status summary
    if benchmark_strategies:
        print("")
        print(f"  Benchmarks: {len(benchmark_strategies)} available")
        if auto_run_benchmarks and not skip_benchmarks:
            print("    (will auto-run after regular strategies)")
        elif skip_benchmarks:
            print("    (skipped: Q23_SKIP_BENCHMARKS=true)")
        else:
            print("    (auto-run disabled: Q23_AUTO_RUN_BENCHMARKS=false)")
    
    print("")
    
    # Summary
    if strategies_to_skip and not force_rerun:
        print(f"  Skipping {len(strategies_to_skip)} fresh strateg{'y' if len(strategies_to_skip) == 1 else 'ies'}")
    if strategies_to_run:
        print(f"  Running {len(strategies_to_run)} strateg{'y' if len(strategies_to_run) == 1 else 'ies'}")
    print("")

    rc = 0

    # Run legacy StrategyEngine if enabled
    if run_legacy_v4:
        print(">> Running legacy v4 StrategyEngine...")
        r = _run_legacy_v4(tag=tag)
        print(f">> {r.name}: {r.message}")
        if r.rc != 0:
            rc = max(rc, r.rc)

    # Run registered strategies if enabled and not fresh (or force_rerun)
    print("===== Running Regular Strategies =====")
    ran_count = 0
    
    for strategy_id, enabled in strategy_flags.items():
        if not enabled:
            continue
            
        is_fresh, latest_tag = freshness_status.get(strategy_id, (False, None))
        
        # Skip fresh strategies unless force_rerun is set
        if is_fresh and not force_rerun:
            print(f">> Skipping {strategy_id} (fresh run from today: {latest_tag})")
            continue
        
        print(f">> Running {strategy_id}...")
        r = _run_strategy(strategy_id, tag=tag)
        print(f">> {r.name}: {r.message}")
        ran_count += 1
        if r.rc != 0:
            rc = max(rc, r.rc)
    
    if ran_count == 0 and not strategies_to_run:
        print(">> No regular strategies to run")

    # Auto-run benchmarks after regular strategies
    if auto_run_benchmarks and not skip_benchmarks and benchmark_strategies:
        bench_failures = _ensure_benchmarks_fresh(
            today=today,
            required_benchmarks=benchmark_strategies,
            force_rerun=force_rerun,
        )
        if bench_failures > 0:
            print(f"WARNING: {bench_failures} benchmark(s) failed")
            # Don't fail overall run for benchmark failures
            # They're not critical for strategy execution

    # Warn if nothing was enabled
    if not run_legacy_v4 and not any(strategy_flags.values()):
        print("")
        print("WARNING: No strategies enabled.")
        print("")
        print("To enable strategies:")
        print("  1. Use the Strategy Warehouse in the dashboard (RECOMMENDED)")
        print("     This updates enabled_strategies.json automatically")
        print("")
        print("  2. Manually edit enabled_strategies.json:")
        print(f"     File: {_project_root() / 'src' / 'q23' / 'strategies' / 'enabled_strategies.json'}")
        print("     Set \"enabled\": { \"<strategy_id>\": true }")
        print("")
        print("  (Legacy: Environment variables Q23_RUN_<strategy_id>=true are deprecated)")
        print("")
        print("Available strategies:")
        for strategy_id in regular_strategies:
            print(f"  - {strategy_id}")
        print("")
        print("Or run legacy engine: Q23_RUN_V4=true (deprecated)")
    
    # Hint about force rerun if all were skipped
    elif strategies_to_skip and not strategies_to_run and not force_rerun:
        print("")
        print("All enabled strategies have fresh data from today.")
        print("To force re-run: Q23_FORCE_RERUN=true")

    print("")
    print("=" * 60)
    print("Done.")
    print("=" * 60)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
