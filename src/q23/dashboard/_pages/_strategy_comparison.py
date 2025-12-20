from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from q23.dashboard.core import (
    DashboardData,
    discover_available_strategies,
    discover_strategy_tags,
    load_strategy_dashboard_data,
    get_strategy_display_name,
    get_strategy_configs,
    artifacts_to_dashboard_data,
)
from q23.dashboard.analytics.performance import compute_comprehensive_performance
from q23.dashboard.analytics.viz import setup_plot_style
from q23.dashboard.components.styles import (
    STRATEGY_COLORS,
    BENCHMARK_COLORS,
    BENCHMARK_STRATEGY_IDS,
    build_strategy_color_map,
    inject_multiselect_colors,
)


def _compute_return_series(data: DashboardData) -> pd.Series:
    if data.diag is not None and not data.diag.empty:
        if "port_ret" in data.diag.columns:
            return data.diag["port_ret"].fillna(0.0)
        elif "active_ret" in data.diag.columns:
            return data.diag["active_ret"].fillna(0.0)
    return pd.Series(dtype=float)


def build_comparison_table(
    strategy_data: Dict[str, DashboardData],
    tc_bps_map: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    rows = []
    for sid, data in strategy_data.items():
        tc_bps = tc_bps_map.get(sid, 10.0) if tc_bps_map else 10.0
        perf = compute_comprehensive_performance(
            data.weights, returns=None, diag=data.diag, tc_bps=tc_bps
        )
        rows.append({
            "Strategy": get_strategy_display_name(sid),
            "Ann. Return": f"{perf['annual_return']:.2%}",
            "Volatility": f"{perf['annual_vol']:.2%}",
            "Sharpe": f"{perf['sharpe']:.3f}",
            "Sortino": f"{perf['sortino']:.3f}",
            "Calmar": f"{perf['calmar']:.3f}" if np.isfinite(perf['calmar']) else "—",
            "Max DD": f"{perf['max_drawdown']:.2%}",
            "Win Rate": f"{perf['win_rate']:.1%}",
            "Positions": f"{perf['n_positions']:.0f}",
            "Avg Turnover": f"{perf['avg_turnover']:.2%}",
        })
    return pd.DataFrame(rows)


def create_multi_equity_chart(
    strategy_data: Dict[str, DashboardData],
    color_map: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Create multi-equity cumulative return chart.
    
    Args:
        strategy_data: Dict of strategy_id -> DashboardData
        color_map: Optional pre-built color map for consistent coloring.
                   If None, builds one from strategy_data keys.
        figsize: Figure size tuple
    
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    # Build color map if not provided (ensures consistency with UI elements)
    if color_map is None:
        color_map = build_strategy_color_map(list(strategy_data.keys()))
    
    for sid, data in strategy_data.items():
        ret_series = _compute_return_series(data)
        if ret_series.empty:
            continue
        
        cumret = (1.0 + ret_series).cumprod() - 1.0
        label = get_strategy_display_name(sid)
        
        # Get color from the centralized map
        color = color_map.get(sid, STRATEGY_COLORS[0])
        
        # Use different styling for benchmarks vs strategies
        if sid in BENCHMARK_STRATEGY_IDS:
            linestyle = "--"  # Dashed line for benchmarks
            linewidth = 2.0
            alpha = 0.8
        else:
            linestyle = "-"  # Solid line for strategies
            linewidth = 2.5
            alpha = 1.0
        
        ax.plot(
            cumret.index, 
            cumret.values, 
            label=label, 
            linewidth=linewidth, 
            color=color,
            linestyle=linestyle,
            alpha=alpha,
        )

    ax.axhline(y=0, color='white', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_title("Cumulative Returns Comparison", fontsize=14, fontweight='bold', pad=20)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    return fig


def compute_return_correlations(strategy_data: Dict[str, DashboardData]) -> pd.DataFrame:
    returns_dict = {}
    for sid, data in strategy_data.items():
        ret_series = _compute_return_series(data)
        if not ret_series.empty:
            returns_dict[get_strategy_display_name(sid)] = ret_series

    if len(returns_dict) < 2:
        return pd.DataFrame()

    returns_df = pd.DataFrame(returns_dict)
    corr_matrix = returns_df.corr()
    return corr_matrix


def compute_rolling_correlation(
    strategy_data: Dict[str, DashboardData],
    window: int = 63,
) -> pd.DataFrame:
    returns_dict = {}
    for sid, data in strategy_data.items():
        ret_series = _compute_return_series(data)
        if not ret_series.empty:
            returns_dict[get_strategy_display_name(sid)] = ret_series

    if len(returns_dict) < 2:
        return pd.DataFrame()

    returns_df = pd.DataFrame(returns_dict).dropna()
    names = list(returns_df.columns)

    rolling_corr = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair = f"{names[i]} vs {names[j]}"
            rolling_corr[pair] = returns_df[names[i]].rolling(window).corr(returns_df[names[j]])

    return pd.DataFrame(rolling_corr)


def render_strategy_comparison(
    current_strategy: str,
    date_range: Tuple[str, str],
) -> None:
    st.subheader("Strategy Comparison")

    available_strategies = discover_available_strategies()
    
    # Use centralized benchmark IDs
    regular_strategies = [s for s in available_strategies if s not in BENCHMARK_STRATEGY_IDS]
    
    # Add benchmark toggle
    if "q23_include_benchmarks" not in st.session_state:
        st.session_state.q23_include_benchmarks = bool(
            st.session_state.get("admin_default_include_benchmarks", False)
        )
    include_benchmarks = st.checkbox(
        "Include Market Benchmarks",
        value=bool(st.session_state.q23_include_benchmarks),
        key="q23_include_benchmarks",
        help="Include NYSE and NASDAQ equal-weighted and market-cap weighted benchmarks",
    )
    
    if len(regular_strategies) < 2 and not include_benchmarks:
        st.info("At least 2 strategies required for comparison. Only 1 registered.")
        return

    st.markdown(
        """
        Compare multiple strategies side-by-side. Select strategies to compare
        and see performance metrics, equity curves, and correlations.
        """
    )
    
    # Build color map for all available strategies BEFORE rendering multiselect
    # This ensures consistent colors regardless of selection order
    all_available = regular_strategies + (BENCHMARK_STRATEGY_IDS if include_benchmarks else [])
    color_map = build_strategy_color_map(all_available)
    
    # Inject CSS for multiselect tag colors (must be before the multiselect)
    inject_multiselect_colors(color_map, get_strategy_display_name)

    comparison_strategies = st.multiselect(
        "Strategies to Compare",
        options=regular_strategies,
        default=regular_strategies[:min(2, len(regular_strategies))],
        format_func=lambda x: get_strategy_display_name(x),
    )
    
    # Add benchmarks if enabled
    if include_benchmarks:
        benchmark_selection = st.multiselect(
            "Benchmarks to Include",
            options=BENCHMARK_STRATEGY_IDS,
            default=BENCHMARK_STRATEGY_IDS,  # Include all by default
            format_func=lambda x: get_strategy_display_name(x),
        )
        comparison_strategies.extend(benchmark_selection)

    if len(comparison_strategies) < 2:
        st.warning("Select at least 2 strategies/benchmarks to compare.")
        return

    strategy_data: Dict[str, DashboardData] = {}
    strategy_configs = get_strategy_configs()
    tc_bps_map = {}

    for sid in comparison_strategies:
        # Handle benchmarks differently (they're computed on-demand)
        if sid in BENCHMARK_STRATEGY_IDS:
            try:
                from q23.strategies.registry import StrategyRegistry
                benchmark_strategy = StrategyRegistry.get_instance(sid)
                
                # Compute benchmark on-demand
                start_date, end_date = date_range
                artifacts = benchmark_strategy.run(
                    min_date=start_date,
                    max_date=end_date,
                    write_outputs=False,
                )
                
                # Convert artifacts to DashboardData
                data = artifacts_to_dashboard_data(artifacts, sid)
                
                # Apply date range filtering
                if data.weights is not None and not data.weights.empty:
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    mask = (data.weights.index >= start_dt) & (data.weights.index <= end_dt)
                    data.weights = data.weights.loc[mask].copy() if mask.any() else data.weights
                    
                    if data.diag is not None and not data.diag.empty:
                        diag_mask = (data.diag.index >= start_dt) & (data.diag.index <= end_dt)
                        data.diag = data.diag.loc[diag_mask].copy() if diag_mask.any() else data.diag
                
                if data.weights is None or data.weights.empty:
                    st.warning(f"Could not compute benchmark '{sid}'. Market cap data may be unavailable.")
                    continue
                
                # Check if benchmark has any non-zero weights (market cap benchmarks might fail)
                if data.weights.abs().sum().sum() < 1e-6:
                    st.warning(f"Benchmark '{sid}' computed but has zero weights. Market cap data may be unavailable.")
                    continue
                
                strategy_data[sid] = data
                if sid in strategy_configs:
                    tc_bps_map[sid] = strategy_configs[sid].tc_bps
            except Exception as e:
                st.warning(f"Error computing benchmark '{sid}': {e}")
                continue
        else:
            # Regular strategy - load from files
            tags = discover_strategy_tags(sid)
            if not tags:
                st.warning(f"No output tags found for strategy '{sid}'")
                continue

            tag = tags[0]
            data = load_strategy_dashboard_data(sid, tag, date_range=date_range)
            
            if data.weights is None or data.weights.empty:
                st.warning(f"No data loaded for strategy '{sid}'")
                continue

            strategy_data[sid] = data
            if sid in strategy_configs:
                tc_bps_map[sid] = strategy_configs[sid].tc_bps

    if len(strategy_data) < 2:
        st.error("Could not load data for at least 2 strategies.")
        return

    st.markdown("### Performance Comparison")
    comparison_df = build_comparison_table(strategy_data, tc_bps_map)
    st.dataframe(comparison_df, height=200)

    st.markdown("---")

    st.markdown("### Cumulative Returns")
    # Pass the color_map to ensure chart colors match multiselect tag colors
    fig = create_multi_equity_chart(strategy_data, color_map=color_map)
    st.pyplot(fig)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Return Correlations")
        corr_matrix = compute_return_correlations(strategy_data)
        if not corr_matrix.empty:
            st.dataframe(
                corr_matrix.style.format("{:.3f}").background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
            )
        else:
            st.info("Not enough data for correlation matrix")

    with col2:
        st.markdown("### Strategy Details")
        details = []
        for sid, data in strategy_data.items():
            if sid in strategy_configs:
                cfg = strategy_configs[sid]
                details.append({
                    "Strategy": get_strategy_display_name(sid),
                    "Version": cfg.version,
                    "Factors": len(cfg.factors),
                    "Min Date": cfg.min_date,
                    "TopN": cfg.topn,
                    "Target Vol": f"{cfg.target_vol:.0%}",
                    "TC (bps)": cfg.tc_bps,
                })
        if details:
            st.dataframe(pd.DataFrame(details), height=200)

    st.markdown("---")

    st.markdown("### Rolling Correlation")
    window = st.slider("Rolling Window (days)", 21, 252, 63, 21)
    rolling_corr = compute_rolling_correlation(strategy_data, window=window)
    if not rolling_corr.empty:
        st.line_chart(rolling_corr.tail(500), height=300)
    else:
        st.info("Not enough data for rolling correlation")

    st.markdown("---")

    st.markdown("### Drawdown Comparison")
    dd_data = {}
    for sid, data in strategy_data.items():
        ret_series = _compute_return_series(data)
        if not ret_series.empty:
            eq = (1.0 + ret_series).cumprod()
            dd = eq / eq.cummax() - 1.0
            dd_data[get_strategy_display_name(sid)] = dd

    if dd_data:
        dd_df = pd.DataFrame(dd_data)
        st.line_chart(dd_df.tail(500), height=300)

        max_dds = {}
        for name, dd_series in dd_data.items():
            max_dds[name] = f"{dd_series.min():.2%}"
        st.markdown("**Maximum Drawdowns:**")
        st.json(max_dds)
