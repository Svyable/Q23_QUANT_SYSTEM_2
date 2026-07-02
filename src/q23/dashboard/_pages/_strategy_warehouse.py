"""
Strategy Warehouse

A comprehensive strategy management dashboard providing:
- Overview of all registered strategies with key metrics
- Enable/disable controls for run_strategy.py
- Strategy comparison (performance, config diff)
- Run history viewer
- Performance metrics from latest runs

This is the central hub for managing the Q23 strategy ecosystem.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from q23.dashboard.core import (
    discover_available_strategies,
    discover_strategy_tags,
    get_strategy_display_name,
    get_strategy_configs,
    load_strategy_dashboard_data,
)


# =============================================================================
# ENABLED STRATEGIES CONFIGURATION
# =============================================================================

def _get_enabled_config_path() -> Path:
    """Get path to enabled_strategies.json."""
    return Path("src/q23/strategies/enabled_strategies.json")


def _load_enabled_strategies() -> Dict[str, bool]:
    """Load enabled strategies from JSON config."""
    config_path = _get_enabled_config_path()
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text())
            return data.get("enabled", {})
        except Exception:
            pass
    return {}


def _auto_enable_benchmarks_with_runs(enabled: Dict[str, bool]) -> Dict[str, bool]:
    """Auto-enable benchmarks that have runs.
    
    Benchmarks with any runs are automatically enabled to ensure they're
    available for comparison and auto-run. This respects user's explicit
    settings but ensures benchmarks with data are enabled by default.
    
    Args:
        enabled: Current enabled status dict
        
    Returns:
        Updated enabled dict with benchmarks auto-enabled if they have runs
    """
    all_strategies = discover_available_strategies(include_benchmarks=True)
    benchmarks = [s for s in all_strategies if s.startswith("benchmark_")]
    
    updated = enabled.copy()
    auto_enabled_count = 0
    
    for bench_id in benchmarks:
        # Check if benchmark has any runs
        tags = discover_strategy_tags(bench_id)
        if tags:
            # Has runs - auto-enable it (even if explicitly disabled)
            # This ensures benchmarks with data are available for comparison
            if updated.get(bench_id, False) != True:
                updated[bench_id] = True
                auto_enabled_count += 1
        else:
            # No runs - default to disabled if not explicitly set
            if bench_id not in updated:
                updated[bench_id] = False
    
    # Save if we auto-enabled any benchmarks
    if auto_enabled_count > 0:
        try:
            _save_enabled_strategies(updated)
        except Exception:
            pass  # Don't fail if save fails
    
    return updated


def _save_enabled_strategies(enabled: Dict[str, bool], default_on_startup: Optional[str] = None) -> bool:
    """Save enabled strategies to JSON config."""
    config_path = _get_enabled_config_path()
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "enabled": enabled,
            "default_on_startup": default_on_startup,
            "last_updated": datetime.now().isoformat(),
        }
        config_path.write_text(json.dumps(data, indent=2))
        return True
    except Exception as e:
        st.error(f"Failed to save config: {e}")
        return False


def _is_strategy_enabled(strategy_id: str) -> bool:
    """Check if a strategy is enabled."""
    enabled = _load_enabled_strategies()
    return enabled.get(strategy_id, False)


# =============================================================================
# METRICS LOADING
# =============================================================================

def _load_run_metrics(strategy_id: str, tag: str) -> Dict[str, Any]:
    """Load performance metrics from a strategy run."""
    try:
        data = load_strategy_dashboard_data(strategy_id, tag)
        
        if data.diag is None or data.diag.empty:
            return {}
        
        diag = data.diag
        
        # Calculate basic metrics
        port_ret = diag.get("port_ret", pd.Series(dtype=float))
        if port_ret.empty:
            return {}
        
        # Cumulative return
        cum_ret = (1 + port_ret).prod() - 1
        
        # Annualized return (CAGR)
        n_days = len(port_ret)
        if n_days > 0:
            cagr = (1 + cum_ret) ** (252 / n_days) - 1
        else:
            cagr = 0
        
        # Volatility (annualized)
        vol = port_ret.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = cagr / vol if vol > 0 else 0
        
        # Max drawdown
        cum_wealth = (1 + port_ret).cumprod()
        peak = cum_wealth.expanding().max()
        drawdown = (cum_wealth - peak) / peak
        max_dd = drawdown.min()
        
        # Win rate
        win_rate = (port_ret > 0).mean()
        
        return {
            "cum_ret": f"{cum_ret:.1%}",
            "cagr": f"{cagr:.1%}",
            "vol": f"{vol:.1%}",
            "sharpe": f"{sharpe:.2f}",
            "max_dd": f"{max_dd:.1%}",
            "win_rate": f"{win_rate:.1%}",
            "n_days": n_days,
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# COMPARISON CHARTS
# =============================================================================

def _render_comparison_charts(strategy_ids: List[str]) -> None:
    """Render comparison charts for selected strategies."""
    if not strategy_ids:
        return
    
    st.markdown("#### Cumulative Performance Comparison")
    
    # Load data for each strategy
    perf_data = {}
    for sid in strategy_ids:
        tags = discover_strategy_tags(sid)
        if not tags:
            continue
        
        try:
            data = load_strategy_dashboard_data(sid, tags[0])
            if data.diag is not None and not data.diag.empty:
                port_ret = data.diag.get("port_ret", pd.Series(dtype=float))
                if not port_ret.empty:
                    cum_ret = (1 + port_ret).cumprod()
                    perf_data[get_strategy_display_name(sid)] = cum_ret
        except Exception:
            continue
    
    if not perf_data:
        st.info("No performance data available for selected strategies.")
        return
    
    # Create comparison DataFrame
    df = pd.DataFrame(perf_data)
    
    # Plot
    st.line_chart(df)
    
    # Stats table
    st.markdown("#### Comparison Metrics")
    
    stats = []
    for sid in strategy_ids:
        tags = discover_strategy_tags(sid)
        if tags:
            metrics = _load_run_metrics(sid, tags[0])
            metrics["Strategy"] = get_strategy_display_name(sid)
            stats.append(metrics)
    
    if stats:
        stats_df = pd.DataFrame(stats)
        cols = ["Strategy"] + [c for c in stats_df.columns if c != "Strategy"]
        stats_df = stats_df[cols]
        # Ensure all values are strings to avoid Arrow type errors
        stats_df = stats_df.astype(str)
        st.dataframe(stats_df, hide_index=True)


def _render_config_diff(left_id: str, right_id: str) -> None:
    """Render configuration diff between two strategies."""
    if left_id == right_id:
        st.info("Select two different strategies to compare.")
        return
    
    configs = get_strategy_configs()
    
    left_cfg = configs.get(left_id)
    right_cfg = configs.get(right_id)
    
    if not left_cfg or not right_cfg:
        st.warning("Could not load configs for comparison.")
        return
    
    # Build comparison table
    params = [
        ("Display Name", "display_name"),
        ("Version", "version"),
        ("Factors", "factors"),
        ("Min Date", "min_date"),
        ("Exchanges", "exchanges"),
        ("Long Only", "long_only"),
        ("Long Seats", "long_seats"),
        ("Short Seats", "short_seats"),
        ("Top N", "topn"),
        ("Max Position", "max_pos"),
        ("Min Position", "min_pos"),
        ("Target Vol", "target_vol"),
        ("Lev Cap", "lev_cap"),
        ("Lev Min", "lev_min"),
    ]
    
    diff_data = []
    for label, attr in params:
        left_val = getattr(left_cfg, attr, "N/A")
        right_val = getattr(right_cfg, attr, "N/A")
        
        # Format lists
        if isinstance(left_val, (list, tuple)):
            left_val = len(left_val) if attr == "factors" else ", ".join(map(str, left_val))
        if isinstance(right_val, (list, tuple)):
            right_val = len(right_val) if attr == "factors" else ", ".join(map(str, right_val))
        
        # Format floats
        if isinstance(left_val, float):
            left_val = f"{left_val:.3f}"
        if isinstance(right_val, float):
            right_val = f"{right_val:.3f}"
        
        # Check if different
        is_diff = str(left_val) != str(right_val)
        
        diff_data.append({
            "Parameter": label,
            get_strategy_display_name(left_id): left_val,
            get_strategy_display_name(right_id): right_val,
            "Different": "⚠️" if is_diff else "✓",
        })
    
    df = pd.DataFrame(diff_data)
    # Ensure all values are strings to avoid Arrow type errors
    df = df.astype(str)
    st.dataframe(df, hide_index=True)


def _render_run_history(strategy_id: str) -> None:
    """Render run history for a strategy."""
    tags = discover_strategy_tags(strategy_id)
    
    if not tags:
        st.info(f"No runs found for {get_strategy_display_name(strategy_id)}.")
        return
    
    history_data = []
    for tag in tags[:20]:  # Limit to most recent 20
        try:
            tag_time = pd.to_datetime(tag)
            time_str = tag_time.strftime("%Y-%m-%d %H:%M")
        except:
            time_str = tag
        
        metrics = _load_run_metrics(strategy_id, tag)
        
        history_data.append({
            "Tag": tag,
            "Time": time_str,
            "Sharpe": metrics.get("sharpe", "-"),
            "CAGR": metrics.get("cagr", "-"),
            "MaxDD": metrics.get("max_dd", "-"),
            "Days": metrics.get("n_days", "-"),
        })
    
    df = pd.DataFrame(history_data)
    # Ensure all values are strings to avoid Arrow type errors
    df = df.astype(str)
    st.dataframe(df, hide_index=True)


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_strategy_warehouse() -> None:
    """Render the Strategy Warehouse page."""
    st.header("📦 Strategy Warehouse")
    
    st.markdown(
        """
        Central hub for managing all Q23 strategies. View metrics, compare performance,
        configure which strategies run on startup, and explore run history.
        """
    )
    
    st.divider()
    
    # =========================================================================
    # STRATEGY OVERVIEW TABLE
    # =========================================================================
    st.subheader("📊 Registered Strategies")
    
    strategies = discover_available_strategies(include_benchmarks=True)
    
    if not strategies:
        st.warning("No strategies registered. Create one in the Live Strategy Lab.")
        return
    
    # Load enabled status and auto-enable benchmarks with runs
    enabled_config = _load_enabled_strategies()
    enabled_config = _auto_enable_benchmarks_with_runs(enabled_config)
    
    # Build overview data
    warehouse_data = []
    for sid in strategies:
        configs = get_strategy_configs()
        config = configs.get(sid)
        
        if not config:
            continue
        
        tags = discover_strategy_tags(sid)
        latest_tag = tags[0] if tags else None
        
        # Load metrics from latest run
        metrics = _load_run_metrics(sid, latest_tag) if latest_tag else {}
        
        warehouse_data.append({
            "Strategy": config.display_name,
            "ID": sid,
            "Version": config.version,
            "Factors": len(config.factors),
            "Style": "Long" if config.long_only else "L/S",
            "Runs": len(tags),
            "Latest": latest_tag[:10] if latest_tag else "Never",
            "Sharpe": metrics.get("sharpe", "-"),
            "CAGR": metrics.get("cagr", "-"),
            "MaxDD": metrics.get("max_dd", "-"),
            "Enabled": enabled_config.get(sid, False),
        })
    
    if not warehouse_data:
        st.info("No strategy data available.")
        return
    
    df = pd.DataFrame(warehouse_data)
    
    # Convert metrics columns to strings to avoid Arrow type errors with mixed types
    for col in ["Sharpe", "CAGR", "MaxDD", "Latest"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Editable table with enable/disable checkboxes
    st.markdown("**Enable strategies for `run_strategy.py`:**")
    
    edited_df = st.data_editor(
        df,
        column_config={
            "Enabled": st.column_config.CheckboxColumn(
                "Enabled",
                help="Enable this strategy in run_strategy.py",
                default=False,
            ),
            "Strategy": st.column_config.TextColumn("Strategy", disabled=True),
            "ID": st.column_config.TextColumn("ID", disabled=True),
            "Version": st.column_config.TextColumn("v", disabled=True, width="small"),
            "Factors": st.column_config.NumberColumn("Factors", disabled=True, width="small"),
            "Style": st.column_config.TextColumn("Style", disabled=True, width="small"),
            "Runs": st.column_config.NumberColumn("Runs", disabled=True, width="small"),
            "Latest": st.column_config.TextColumn("Latest Run", disabled=True),
            "Sharpe": st.column_config.TextColumn("Sharpe", disabled=True, width="small"),
            "CAGR": st.column_config.TextColumn("CAGR", disabled=True, width="small"),
            "MaxDD": st.column_config.TextColumn("MaxDD", disabled=True, width="small"),
        },
        width='stretch',
        hide_index=True,
        key="warehouse_table",
    )
    
    # Save button
    col_save, col_info = st.columns([1, 3])
    with col_save:
        if st.button("💾 Save Enable/Disable Settings", type="primary", help="Save strategy enable/disable settings to run_strategy.py"):
            with st.spinner("💾 Saving settings..."):
                new_enabled = {}
                # Use itertuples() instead of iterrows() for better performance
                for row in edited_df.itertuples(index=False):
                    new_enabled[row.ID] = bool(row.Enabled)

                if _save_enabled_strategies(new_enabled):
                    st.success("Settings saved!")
                else:
                    st.error("Failed to save settings.")
    
    with col_info:
        # Use vectorized operation instead of iterrows() for better performance
        enabled_count = int(edited_df["Enabled"].sum())
        st.caption(f"{enabled_count} strategies enabled for startup runs")
    
    st.divider()
    
    # =========================================================================
    # STRATEGY COMPARISON
    # =========================================================================
    st.subheader("📈 Strategy Comparison")
    
    compare_strategies = st.multiselect(
        "Select strategies to compare",
        options=strategies,
        format_func=get_strategy_display_name,
        max_selections=4,
        key="warehouse_compare",
    )
    
    if compare_strategies:
        _render_comparison_charts(compare_strategies)
    else:
        st.caption("Select 2-4 strategies to compare their performance.")
    
    st.divider()
    
    # =========================================================================
    # CONFIG DIFF VIEWER
    # =========================================================================
    st.subheader("🔍 Configuration Diff")
    
    col1, col2 = st.columns(2)
    with col1:
        left_strategy = st.selectbox(
            "Strategy A",
            options=strategies,
            format_func=get_strategy_display_name,
            key="warehouse_diff_left",
        )
    with col2:
        right_strategy = st.selectbox(
            "Strategy B",
            options=strategies,
            format_func=get_strategy_display_name,
            index=min(1, len(strategies) - 1),
            key="warehouse_diff_right",
        )
    
    if left_strategy and right_strategy:
        _render_config_diff(left_strategy, right_strategy)
    
    st.divider()
    
    # =========================================================================
    # RUN HISTORY
    # =========================================================================
    st.subheader("📜 Run History")
    
    selected_for_history = st.selectbox(
        "Select strategy to view run history",
        options=strategies,
        format_func=get_strategy_display_name,
        key="warehouse_history",
    )
    
    if selected_for_history:
        _render_run_history(selected_for_history)
    
    st.divider()
    
    # =========================================================================
    # QUICK ACTIONS
    # =========================================================================
    st.subheader("⚡ Quick Actions")
    
    col_a1, col_a2, col_a3 = st.columns(3)
    
    with col_a1:
        if st.button("🔄 Refresh Data", key="warehouse_refresh"):
            st.cache_data.clear()
            st.rerun()
    
    with col_a2:
        if st.button("✅ Enable All", key="warehouse_enable_all"):
            all_enabled = {sid: True for sid in strategies}
            _save_enabled_strategies(all_enabled)
            st.rerun()
    
    with col_a3:
        if st.button("❌ Disable All", key="warehouse_disable_all"):
            all_disabled = {sid: False for sid in strategies}
            _save_enabled_strategies(all_disabled)
            st.rerun()
