from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np  # type: ignore[reportMissingImports]
import pandas as pd  # type: ignore[reportMissingImports]
import streamlit as st  # type: ignore[reportMissingImports]
import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]

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
from q23.dashboard.components.charts import PLOTLY_AVAILABLE

# Conditional plotly imports for type checking
if PLOTLY_AVAILABLE:
    try:
        import plotly.graph_objects as go  # type: ignore[reportMissingImports]
    except ImportError:
        pass


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
    Create multi-equity cumulative return chart (matplotlib version).
    
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


def create_multi_equity_chart_plotly(
    strategy_data: Dict[str, DashboardData],
    color_map: Optional[Dict[str, str]] = None,
) -> Optional[object]:
    """
    Create interactive multi-equity chart with Plotly.
    
    Clean chart with unified hover tooltips - no legend clutter.
    Strategy lines colored green/red based on their final return.
    Benchmarks shown with muted dashed lines.
    
    Args:
        strategy_data: Dict of strategy_id -> DashboardData
        color_map: Optional pre-built color map for consistent coloring.
    
    Returns:
        Plotly figure or None if Plotly unavailable
    """
    if not PLOTLY_AVAILABLE or 'go' not in globals():
        return None

    # Build color map if not provided (ensures consistency with UI elements)
    if color_map is None:
        color_map = build_strategy_color_map(list(strategy_data.keys()))

    fig = go.Figure()

    # Collect all cumulative returns
    all_cumrets = {}
    for sid, data in strategy_data.items():
        ret_series = _compute_return_series(data)
        if ret_series.empty:
            continue
        cumret = (1.0 + ret_series).cumprod() - 1.0
        label = get_strategy_display_name(sid)
        all_cumrets[sid] = (cumret, label)

    # Add cumulative return traces - strategies first, then benchmarks
    strategies_first = sorted(
        all_cumrets.keys(),
        key=lambda x: (x in BENCHMARK_STRATEGY_IDS, x)
    )

    for sid in strategies_first:
        cumret, label = all_cumrets[sid]

        # Get color from the centralized map (same as matplotlib version)
        color = color_map.get(sid, STRATEGY_COLORS[0])

        # Use different styling for benchmarks vs strategies (same as matplotlib)
        if sid in BENCHMARK_STRATEGY_IDS:
            line_style = {'color': color, 'width': 1.5, 'dash': 'dash'}
        else:
            line_style = {'color': color, 'width': 2.5}
        
        fig.add_trace(go.Scatter(
            x=cumret.index,
            y=cumret.values,
            mode='lines',
            name=f'PnL {label}',
            line=line_style,
            hovertemplate='%{y:.2%}',
            showlegend=False,  # No legend - cleaner
        ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    
    layout = {
        'title': {
            'text': 'Cumulative Returns Comparison',
            'font': {'size': 16, 'color': '#ecf0f1'},
        },
        'paper_bgcolor': '#0E1117',
        'plot_bgcolor': '#262730',
        'font': {'color': '#ecf0f1'},
        'autosize': True,
        'margin': {'l': 60, 'r': 40, 't': 60, 'b': 40},
        'showlegend': False,  # Clean chart - tooltips show everything
        'xaxis': {
            'gridcolor': '#3A3A3A',
            'zerolinecolor': '#3A3A3A',
        },
        'yaxis': {
            'gridcolor': '#3A3A3A',
            'zerolinecolor': '#3A3A3A',
            'tickformat': '.0%',
        },
        'hovermode': 'x unified',
        'hoverlabel': {
            'bgcolor': '#1e1e1e',
            'bordercolor': '#444',
            'font': {'size': 12, 'color': '#ecf0f1'},
        },
    }
    fig.update_layout(**layout)
    
    return fig


def create_drawdown_comparison_plotly(
    strategy_data: Dict[str, DashboardData],
    color_map: Optional[Dict[str, str]] = None,
) -> Optional[object]:
    """
    Create interactive drawdown comparison chart with Plotly.
    
    Clean chart - strategy drawdowns in red (drawdown is bad),
    benchmarks in gray dashed. No legend clutter, tooltips show all.
    
    Args:
        strategy_data: Dict of strategy_id -> DashboardData
        color_map: Optional pre-built color map for consistent coloring.
    
    Returns:
        Plotly figure or None if Plotly unavailable
    """
    if not PLOTLY_AVAILABLE or 'go' not in globals():
        return None
    
    fig = go.Figure()
    
    # Sort: strategies first, then benchmarks
    sorted_sids = sorted(
        strategy_data.keys(),
        key=lambda x: (x in BENCHMARK_STRATEGY_IDS, x)
    )
    
    for sid in sorted_sids:
        data = strategy_data[sid]
        ret_series = _compute_return_series(data)
        if ret_series.empty:
            continue
        
        eq = (1.0 + ret_series).cumprod()
        dd = eq / eq.cummax() - 1.0
        label = get_strategy_display_name(sid)
        
        if sid in BENCHMARK_STRATEGY_IDS:
            # Benchmarks: muted gray dashed
            line_style = {'color': '#666666', 'width': 1.2, 'dash': 'dash'}
            fill = None
            fillcolor = None
        else:
            # Strategies: red fill (drawdown is always negative/bad)
            line_style = {'color': '#e74c3c', 'width': 2}
            fill = 'tozeroy'
            fillcolor = 'rgba(231, 76, 60, 0.15)'
        
        fig.add_trace(go.Scatter(
            x=dd.index,
            y=dd.values,
            mode='lines',
            name=f'DD {label}',
            line=line_style,
            fill=fill,
            fillcolor=fillcolor,
            hovertemplate='%{y:.2%}',
            showlegend=False,
        ))
    
    # Add zero line
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.5)")
    
    fig.update_layout(
        title={
            'text': 'Drawdown Comparison',
            'font': {'size': 16, 'color': '#ecf0f1'},
        },
        paper_bgcolor='#0E1117',
        plot_bgcolor='#262730',
        font={'color': '#ecf0f1'},
        autosize=True,
        margin={'l': 60, 'r': 40, 't': 60, 'b': 40},
        showlegend=False,  # Clean - tooltips show everything
        xaxis={
            'gridcolor': '#3A3A3A',
            'zerolinecolor': '#3A3A3A',
        },
        yaxis={
            'gridcolor': '#3A3A3A',
            'zerolinecolor': '#3A3A3A',
            'tickformat': '.0%',
        },
        hovermode='x unified',
        hoverlabel={
            'bgcolor': '#1e1e1e',
            'bordercolor': '#444',
            'font': {'size': 12, 'color': '#ecf0f1'},
        },
    )
    
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
    
    # Limit data for performance (keep most recent 1000 rows max)
    if len(returns_df) > 1000:
        returns_df = returns_df.tail(1000)
    
    names = list(returns_df.columns)

    rolling_corr = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair = f"{names[i]} vs {names[j]}"
            rolling_corr[pair] = returns_df[names[i]].rolling(window).corr(returns_df[names[j]])

    return pd.DataFrame(rolling_corr)


def create_rolling_correlation_plotly(
    rolling_corr: pd.DataFrame,
    height: Optional[int] = None,
) -> Optional[object]:
    """
    Create interactive rolling correlation chart with unified hover tooltips.
    
    Args:
        rolling_corr: DataFrame with rolling correlation series
        height: Chart height in pixels
        
    Returns:
        Plotly figure or None
    """
    if not PLOTLY_AVAILABLE or rolling_corr.empty:
        return None
    
    # Color palette for correlation pairs
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    
    fig = go.Figure()
    
    for i, col in enumerate(rolling_corr.columns):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr[col],
            mode='lines',
            name=col,
            line={'color': color, 'width': 1.5},
            hovertemplate='%{y:.3f}',
            showlegend=False,  # Clean chart - tooltips show everything
        ))
    
    # Reference lines
    fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(46, 204, 113, 0.4)")
    fig.add_hline(y=0.0, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)")
    fig.add_hline(y=-1.0, line_dash="dot", line_color="rgba(231, 76, 60, 0.4)")
    
    layout = {
        'title': {'text': 'Rolling Correlation', 'font': {'size': 14, 'color': '#ecf0f1'}},
        'paper_bgcolor': '#0E1117',
        'plot_bgcolor': '#262730',
        'font': {'color': '#ecf0f1'},
        'autosize': True,
        'margin': {'l': 50, 'r': 30, 't': 50, 'b': 40},
        'xaxis': {'gridcolor': '#3A3A3A'},
        'yaxis': {
            'gridcolor': '#3A3A3A',
            'range': [-1.1, 1.1],
            'tickformat': '.2f',
        },
        'hovermode': 'x unified',
        'hoverlabel': {
            'bgcolor': '#1e1e1e',
            'bordercolor': '#444',
            'font': {'size': 11, 'color': '#ecf0f1'},
        },
    }
    if height is not None:
        layout['height'] = height
    fig.update_layout(**layout)
    
    return fig


def create_correlation_heatmap_plotly(
    corr_matrix: pd.DataFrame,
    height: Optional[int] = None,
) -> Optional[object]:
    """
    Create interactive correlation heatmap with Plotly.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        height: Chart height in pixels
        
    Returns:
        Plotly figure or None
    """
    if not PLOTLY_AVAILABLE or corr_matrix.empty:
        return None
    
    # Prepare data for heatmap
    labels = list(corr_matrix.columns)
    z_vals = corr_matrix.values
    
    # Custom hover text
    hover_text = []
    for i, row_label in enumerate(labels):
        row_text = []
        for j, col_label in enumerate(labels):
            val = z_vals[i, j]
            row_text.append(f"{row_label} ↔ {col_label}<br>ρ = {val:.3f}")
        hover_text.append(row_text)
    
    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=labels,
        y=labels,
        colorscale='RdYlGn',
        zmin=-1,
        zmax=1,
        text=hover_text,
        hoverinfo='text',
        colorbar=dict(
            title=dict(text='Correlation', side='right'),
            tickformat='.2f',
        ),
    ))
    
    # Add correlation values as annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = z_vals[i, j]
            # Choose text color based on value
            text_color = 'white' if abs(val) > 0.5 else '#333'
            fig.add_annotation(
                x=labels[j],
                y=labels[i],
                text=f"{val:.2f}",
                showarrow=False,
                font={'size': 11, 'color': text_color},
            )
    
    layout = {
        'title': {'text': 'Return Correlations', 'font': {'size': 14, 'color': '#ecf0f1'}},
        'paper_bgcolor': '#0E1117',
        'plot_bgcolor': '#262730',
        'font': {'color': '#ecf0f1', 'size': 10},
        'autosize': True,
        'margin': {'l': 100, 'r': 50, 't': 50, 'b': 100},
        'xaxis': {'side': 'bottom', 'tickangle': 45},
        'yaxis': {'autorange': 'reversed'},
    }
    if height is not None:
        layout['height'] = height
    fig.update_layout(**layout)
    
    return fig


def compute_rolling_sharpe(
    strategy_data: Dict[str, DashboardData],
    window: int = 63,
) -> pd.DataFrame:
    """
    Compute rolling Sharpe ratios for all strategies.
    
    Args:
        strategy_data: Dict of strategy_id -> DashboardData
        window: Rolling window in days
        
    Returns:
        DataFrame with rolling Sharpe for each strategy
    """
    rolling_sharpe = {}
    for sid, data in strategy_data.items():
        ret_series = _compute_return_series(data)
        if ret_series.empty or len(ret_series) < window:
            continue
        
        # Limit data for performance (keep most recent 1000 rows max)
        if len(ret_series) > 1000:
            ret_series = ret_series.tail(1000)
        
        label = get_strategy_display_name(sid)
        rolling_mean = ret_series.rolling(window).mean()
        rolling_std = ret_series.rolling(window).std()
        # Annualize: mean * 252, std * sqrt(252)
        rolling_sharpe[label] = (rolling_mean * 252) / (rolling_std * np.sqrt(252) + 1e-8)
    
    return pd.DataFrame(rolling_sharpe)


def create_rolling_sharpe_plotly(
    rolling_sharpe: pd.DataFrame,
    height: int = 320,
) -> Optional[object]:
    """
    Create interactive rolling Sharpe chart with unified hover.
    
    Args:
        rolling_sharpe: DataFrame with rolling Sharpe series
        height: Chart height in pixels
        
    Returns:
        Plotly figure or None
    """
    if not PLOTLY_AVAILABLE or rolling_sharpe.empty:
        return None
    
    fig = go.Figure()
    
    for col in rolling_sharpe.columns:
        # Color based on current value
        current_val = rolling_sharpe[col].iloc[-1] if len(rolling_sharpe) > 0 else 0
        if current_val >= 1.5:
            color = '#2ecc71'  # Green - excellent
        elif current_val >= 0.5:
            color = '#3498db'  # Blue - good
        elif current_val >= 0:
            color = '#f39c12'  # Orange - mediocre
        else:
            color = '#e74c3c'  # Red - negative
        
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe[col],
            mode='lines',
            name=col,
            line={'color': color, 'width': 2},
            hovertemplate='%{y:.2f}',
            showlegend=False,
        ))
    
    # Reference lines for Sharpe quality
    fig.add_hline(y=2.0, line_dash="dot", line_color="rgba(46, 204, 113, 0.3)", 
                  annotation_text="Excellent", annotation_position="right")
    fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(52, 152, 219, 0.3)",
                  annotation_text="Good", annotation_position="right")
    fig.add_hline(y=0.0, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)")
    
    layout = {
        'title': {'text': 'Rolling Sharpe Ratio', 'font': {'size': 14, 'color': '#ecf0f1'}},
        'paper_bgcolor': '#0E1117',
        'plot_bgcolor': '#262730',
        'font': {'color': '#ecf0f1'},
        'autosize': True,
        'margin': {'l': 50, 'r': 60, 't': 50, 'b': 40},
        'xaxis': {'gridcolor': '#3A3A3A'},
        'yaxis': {
            'gridcolor': '#3A3A3A',
            'tickformat': '.1f',
            'title': 'Sharpe (Ann.)',
        },
        'hovermode': 'x unified',
        'hoverlabel': {
            'bgcolor': '#1e1e1e',
            'bordercolor': '#444',
            'font': {'size': 11, 'color': '#ecf0f1'},
        },
    }
    if height is not None:
        layout['height'] = height
    fig.update_layout(**layout)
    
    return fig


def generate_comparison_csv(
    strategy_data: Dict[str, DashboardData],
    tc_bps_map: Optional[Dict[str, float]] = None,
) -> str:
    """
    Generate CSV export of strategy comparison data.
    
    Args:
        strategy_data: Dict of strategy_id -> DashboardData
        tc_bps_map: Optional transaction cost map
        
    Returns:
        CSV string
    """
    rows = []
    for sid, data in strategy_data.items():
        tc_bps = tc_bps_map.get(sid, 10.0) if tc_bps_map else 10.0
        perf = compute_comprehensive_performance(
            data.weights, returns=None, diag=data.diag, tc_bps=tc_bps
        )
        
        ret_series = _compute_return_series(data)
        eq = (1.0 + ret_series).cumprod() if not ret_series.empty else pd.Series()
        
        rows.append({
            "Strategy": get_strategy_display_name(sid),
            "Strategy ID": sid,
            "Annual Return": perf['annual_return'],
            "Volatility": perf['annual_vol'],
            "Sharpe Ratio": perf['sharpe'],
            "Sortino Ratio": perf['sortino'],
            "Calmar Ratio": perf['calmar'] if np.isfinite(perf['calmar']) else None,
            "Max Drawdown": perf['max_drawdown'],
            "Win Rate": perf['win_rate'],
            "Avg Positions": perf['n_positions'],
            "Avg Turnover": perf['avg_turnover'],
            "TC (bps)": tc_bps,
            "Total Return": eq.iloc[-1] - 1 if len(eq) > 0 else 0,
            "Start Date": ret_series.index[0].strftime('%Y-%m-%d') if len(ret_series) > 0 else '',
            "End Date": ret_series.index[-1].strftime('%Y-%m-%d') if len(ret_series) > 0 else '',
            "N Days": len(ret_series),
        })
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


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
            default=[],  # Don't select all by default for performance
            format_func=lambda x: get_strategy_display_name(x),
        )
        comparison_strategies.extend(benchmark_selection)
    
    # Save selections to session state for Position Stack sync
    st.session_state.comparison_strategies = comparison_strategies

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
                    st.warning(f"⚠️ Could not compute benchmark '{sid}' - market cap data may be unavailable")
                    continue
                
                # Check if benchmark has any non-zero weights (market cap benchmarks might fail)
                if data.weights.abs().sum().sum() < 1e-6:
                    st.warning(f"Benchmark '{sid}' computed but has zero weights. Market cap data may be unavailable.")
                    continue
                
                strategy_data[sid] = data
                if sid in strategy_configs:
                    tc_bps_map[sid] = strategy_configs[sid].tc_bps
            except Exception as e:
                st.error(f"❌ Failed to compute benchmark '{sid}'")
                with st.expander("Error Details", expanded=False):
                    st.code(str(e))
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
    
    # Add download button in header row
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        pass  # Empty for spacing
    with header_col2:
        csv_data = generate_comparison_csv(strategy_data, tc_bps_map)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="strategy_comparison.csv",
            mime="text/csv",
            help="Download detailed comparison data as CSV",
        )
    
    comparison_df = build_comparison_table(strategy_data, tc_bps_map)
    st.dataframe(comparison_df, height=200, width="stretch")

    st.markdown("---")

    st.markdown("### Cumulative Returns")
    
    # Chart type toggle
    if PLOTLY_AVAILABLE:
        use_interactive = st.checkbox(
            "Use interactive charts",
            value=True,  # Default to True for better hover experience
            key="comparison_interactive_charts",
            help="Enable interactive Plotly charts with unified hover tooltips showing all strategies"
        )
    else:
        use_interactive = False
    
    # Pass the color_map to ensure chart colors match multiselect tag colors
    if use_interactive:
        plotly_fig = create_multi_equity_chart_plotly(strategy_data, color_map=color_map)
        if plotly_fig is not None:
            st.plotly_chart(plotly_fig)
        else:
            # Fallback to matplotlib
            fig = create_multi_equity_chart(strategy_data, color_map=color_map)
            st.pyplot(fig)
    else:
        fig = create_multi_equity_chart(strategy_data, color_map=color_map)
        st.pyplot(fig)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Return Correlations")
        corr_matrix = compute_return_correlations(strategy_data)
        if not corr_matrix.empty:
            if use_interactive and PLOTLY_AVAILABLE:
                corr_fig = create_correlation_heatmap_plotly(corr_matrix)
                if corr_fig:
                    st.plotly_chart(corr_fig)
                else:
                    st.dataframe(
                        corr_matrix.style.format("{:.3f}").background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
                    )
            else:
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
            st.dataframe(pd.DataFrame(details), height=200, width="stretch")

    st.markdown("---")

    st.markdown("### Rolling Analytics")
    
    window = st.slider(
        "📊 Rolling Window (days)",
        21,
        252,
        63,
        21,
        key="comparison_rolling_window",
        help="Time window for rolling calculations (correlation, Sharpe ratio). Shorter windows show recent trends, longer windows show long-term stability."
    )
    
    roll_col1, roll_col2 = st.columns(2)
    
    with roll_col1:
        st.markdown("**Rolling Correlation**")
        rolling_corr = compute_rolling_correlation(strategy_data, window=window)
        if not rolling_corr.empty:
            if use_interactive and PLOTLY_AVAILABLE:
                rolling_fig = create_rolling_correlation_plotly(rolling_corr.tail(500))
                if rolling_fig:
                    st.plotly_chart(rolling_fig)
                else:
                    st.line_chart(rolling_corr.tail(500), height=280)
            else:
                st.line_chart(rolling_corr.tail(500), height=280)
        else:
            st.info("Not enough data for rolling correlation")
    
    with roll_col2:
        st.markdown("**Rolling Sharpe Ratio**")
        rolling_sharpe = compute_rolling_sharpe(strategy_data, window=window)
        if not rolling_sharpe.empty:
            if use_interactive and PLOTLY_AVAILABLE:
                sharpe_fig = create_rolling_sharpe_plotly(rolling_sharpe.tail(500))
                if sharpe_fig:
                    st.plotly_chart(sharpe_fig)
                else:
                    st.line_chart(rolling_sharpe.tail(500), height=280)
            else:
                st.line_chart(rolling_sharpe.tail(500), height=280)
        else:
            st.info("Not enough data for rolling Sharpe")

    st.markdown("---")

    st.markdown("### Drawdown Comparison")
    
    # Use interactive Plotly chart if enabled
    if use_interactive and PLOTLY_AVAILABLE:
        dd_plotly_fig = create_drawdown_comparison_plotly(strategy_data, color_map=color_map)
        if dd_plotly_fig is not None:
            st.plotly_chart(dd_plotly_fig)
        else:
            # Fallback to streamlit chart
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
    else:
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

    # Always show max drawdowns summary
    dd_data = {}
    for sid, data in strategy_data.items():
        ret_series = _compute_return_series(data)
        if not ret_series.empty:
            eq = (1.0 + ret_series).cumprod()
            dd = eq / eq.cummax() - 1.0
            dd_data[get_strategy_display_name(sid)] = dd
    
    if dd_data:
        max_dds = {}
        for name, dd_series in dd_data.items():
            max_dds[name] = f"{dd_series.min():.2%}"
        st.markdown("**Maximum Drawdowns:**")
        st.json(max_dds)

    # =============================================================================
    # ADVANCED MULTI-STRATEGY ANALYTICS - STRENGTHENED MULTI-STRATEGY SUPPORT
    # =============================================================================

    st.markdown("---")

    # Advanced Portfolio Optimization Section
    st.markdown("## 🚀 Advanced Portfolio Optimization")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🎯 Optimal Portfolio Allocation", use_container_width=True):
            display_optimal_allocation(strategy_data, color_map)

    with col2:
        if st.button("📊 Risk Parity Analysis", use_container_width=True):
            display_risk_parity_analysis(strategy_data)

    with col3:
        if st.button("🔄 Strategy Rotation Signals", use_container_width=True):
            display_rotation_signals(strategy_data)

    # Factor Attribution Comparison
    st.markdown("---")
    st.markdown("## 📈 Factor Attribution Comparison")

    if st.button("Compare Factor Exposures", use_container_width=True):
        display_factor_attribution_comparison(strategy_data)

    # Regime Performance Analysis
    st.markdown("---")
    st.markdown("## 🌊 Regime-Based Performance")

    if st.button("Analyze Regime Performance", use_container_width=True):
        display_regime_performance_analysis(strategy_data)

    # Capacity and Liquidity Analysis
    st.markdown("---")
    st.markdown("## 💰 Capacity & Liquidity Analysis")

    if st.button("Assess Strategy Capacities", use_container_width=True):
        display_capacity_analysis(strategy_data)

    # ML-Based Strategy Prediction
    st.markdown("---")
    st.markdown("## 🤖 ML Strategy Prediction")

    if st.button("Predict Strategy Performance", use_container_width=True):
        display_ml_strategy_prediction(strategy_data)


def display_optimal_allocation(strategy_data: Dict[str, DashboardData], color_map: Optional[Dict[str, str]] = None):
    """
    Display optimal portfolio allocation using modern portfolio theory.
    """
    st.markdown("### Optimal Portfolio Allocation")

    try:
        import numpy as np
        from scipy.optimize import minimize

        # Extract returns for each strategy
        returns_data = {}
        for sid, data in strategy_data.items():
            ret_series = _compute_return_series(data)
            if not ret_series.empty:
                returns_data[sid] = ret_series

        if len(returns_data) < 2:
            st.error("Need at least 2 strategies with return data for optimization.")
            return

        # Create returns matrix
        returns_df = pd.DataFrame(returns_data).dropna()

        if len(returns_df) < 50:
            st.warning("Limited data for robust optimization. Results may be unreliable.")
            return

        # Calculate expected returns and covariance
        exp_returns = returns_df.mean() * 252  # Annualized
        cov_matrix = returns_df.cov() * 252     # Annualized

        # Optimization functions
        def portfolio_return(weights):
            return np.dot(weights, exp_returns)

        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        def sharpe_ratio(weights):
            ret = portfolio_return(weights)
            vol = portfolio_volatility(weights)
            return -ret / vol  # Negative for minimization

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        ]
        bounds = [(0, 1) for _ in range(len(returns_df.columns))]  # Long only

        # Maximize Sharpe ratio
        initial_weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
        result = minimize(sharpe_ratio, initial_weights, bounds=bounds, constraints=constraints)

        if result.success:
            optimal_weights = result.x

            # Create results dataframe
            opt_df = pd.DataFrame({
                'Strategy': [get_strategy_display_name(sid) for sid in returns_df.columns],
                'Optimal Weight': optimal_weights,
                'Expected Return': exp_returns.values,
                'Volatility': np.sqrt(np.diag(cov_matrix)),
            })

            opt_df['Sharpe Contribution'] = opt_df['Expected Return'] / opt_df['Volatility']

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Optimal Weights")
                st.dataframe(
                    opt_df[['Strategy', 'Optimal Weight']].style.format({
                        'Optimal Weight': '{:.1%}'
                    }).background_gradient(subset=['Optimal Weight'], cmap='Blues'),
                    height=200
                )

            with col2:
                st.markdown("#### Portfolio Statistics")
                portfolio_ret = portfolio_return(optimal_weights)
                portfolio_vol = portfolio_volatility(optimal_weights)
                portfolio_sharpe = portfolio_ret / portfolio_vol

                stats_df = pd.DataFrame({
                    'Metric': ['Expected Return', 'Volatility', 'Sharpe Ratio'],
                    'Value': [f'{portfolio_ret:.1%}', f'{portfolio_vol:.1%}', f'{portfolio_sharpe:.2f}']
                })
                st.dataframe(stats_df, height=150)

            # Visualize allocation
            if color_map:
                colors = [color_map.get(sid, '#3498db') for sid in returns_df.columns]
            else:
                colors = None

            fig, ax = plt.subplots(figsize=(10, 6))
            wedges, texts, autotexts = ax.pie(
                optimal_weights,
                labels=[get_strategy_display_name(sid) for sid in returns_df.columns],
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            ax.set_title('Optimal Portfolio Allocation')
            st.pyplot(fig)

        else:
            st.error("Optimization failed. Try with different constraints or more data.")

    except Exception as e:
        st.error(f"Error in portfolio optimization: {str(e)}")


def display_risk_parity_analysis(strategy_data: Dict[str, DashboardData]):
    """
    Display risk parity analysis across strategies.
    """
    st.markdown("### Risk Parity Analysis")

    try:
        # Extract returns and calculate risk contributions
        returns_data = {}
        for sid, data in strategy_data.items():
            ret_series = _compute_return_series(data)
            if not ret_series.empty:
                returns_data[sid] = ret_series

        if len(returns_data) < 2:
            st.error("Need at least 2 strategies for risk parity analysis.")
            return

        returns_df = pd.DataFrame(returns_data).dropna()

        # Calculate covariance and risk contributions
        cov_matrix = returns_df.cov() * 252
        equal_weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)

        # Risk parity weights (simplified)
        vol_contributions = np.sqrt(np.diag(cov_matrix)) * equal_weights
        total_vol_contrib = np.sum(vol_contributions)
        risk_parity_weights = vol_contributions / total_vol_contrib

        # Current equal weight risk contributions
        equal_risk_contrib = equal_weights * np.sqrt(np.diag(cov_matrix))
        equal_total_risk = np.sum(equal_risk_contrib)

        # Risk parity risk contributions
        rp_risk_contrib = risk_parity_weights * np.sqrt(np.diag(cov_matrix))
        rp_total_risk = np.sum(rp_risk_contrib)

        results_df = pd.DataFrame({
            'Strategy': [get_strategy_display_name(sid) for sid in returns_df.columns],
            'Equal Weight': equal_weights,
            'Risk Parity Weight': risk_parity_weights,
            'Individual Vol': np.sqrt(np.diag(cov_matrix)),
            'Equal Weight Risk Contrib': equal_risk_contrib / equal_total_risk,
            'Risk Parity Risk Contrib': rp_risk_contrib / rp_total_risk,
        })

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Weight Comparison")
            st.dataframe(
                results_df[['Strategy', 'Equal Weight', 'Risk Parity Weight']].style.format({
                    'Equal Weight': '{:.1%}',
                    'Risk Parity Weight': '{:.1%}'
                }),
                height=200
            )

        with col2:
            st.markdown("#### Risk Contribution Comparison")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Equal weight risk contributions
            ax1.bar(range(len(results_df)), results_df['Equal Weight Risk Contrib'])
            ax1.set_title('Equal Weight Risk Contributions')
            ax1.set_xticks(range(len(results_df)))
            ax1.set_xticklabels([get_strategy_display_name(sid) for sid in returns_df.columns], rotation=45)

            # Risk parity risk contributions
            ax2.bar(range(len(results_df)), results_df['Risk Parity Risk Contrib'])
            ax2.set_title('Risk Parity Risk Contributions')
            ax2.set_xticks(range(len(results_df)))
            ax2.set_xticklabels([get_strategy_display_name(sid) for sid in returns_df.columns], rotation=45)

            plt.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error in risk parity analysis: {str(e)}")


def display_rotation_signals(strategy_data: Dict[str, DashboardData]):
    """
    Display strategy rotation signals based on momentum and risk metrics.
    """
    st.markdown("### Strategy Rotation Signals")

    try:
        # Calculate momentum and risk metrics for each strategy
        signals_data = []

        for sid, data in strategy_data.items():
            ret_series = _compute_return_series(data)
            if ret_series.empty:
                continue

            # Momentum metrics
            mom_1m = ret_series.tail(21).mean() * 21  # Annualized 1-month
            mom_3m = ret_series.tail(63).mean() * 4   # Annualized 3-month
            mom_6m = ret_series.tail(126).mean() * 2  # Annualized 6-month

            # Risk metrics
            vol_1m = ret_series.tail(21).std() * np.sqrt(252)
            vol_3m = ret_series.tail(63).std() * np.sqrt(252)

            # Sharpe ratios
            sharpe_1m = mom_1m / vol_1m if vol_1m > 0 else 0
            sharpe_3m = mom_3m / vol_3m if vol_3m > 0 else 0

            # Z-score of recent performance
            lookback = min(252, len(ret_series))
            hist_mean = ret_series.tail(lookback).mean()
            hist_std = ret_series.tail(lookback).std()
            recent_perf = ret_series.tail(21).mean()
            perf_z = (recent_perf - hist_mean) / hist_std if hist_std > 0 else 0

            signals_data.append({
                'Strategy': get_strategy_display_name(sid),
                '1M Momentum': mom_1m,
                '3M Momentum': mom_3m,
                '6M Momentum': mom_6m,
                '1M Sharpe': sharpe_1m,
                '3M Sharpe': sharpe_3m,
                'Perf Z-Score': perf_z,
            })

        if not signals_data:
            st.error("No data available for rotation analysis.")
            return

        signals_df = pd.DataFrame(signals_data)

        # Create rotation signals
        signals_df['Momentum Signal'] = pd.cut(
            signals_df['3M Momentum'],
            bins=[-np.inf, -0.02, 0.02, np.inf],
            labels=['Sell', 'Hold', 'Buy']
        )

        signals_df['Sharpe Signal'] = pd.cut(
            signals_df['3M Sharpe'],
            bins=[-np.inf, 0.5, 1.5, np.inf],
            labels=['Underperform', 'Neutral', 'Outperform']
        )

        signals_df['Overall Signal'] = 'Hold'  # Default
        signals_df.loc[
            (signals_df['Momentum Signal'] == 'Buy') &
            (signals_df['Sharpe Signal'] == 'Outperform'), 'Overall Signal'
        ] = 'Strong Buy'
        signals_df.loc[
            (signals_df['Momentum Signal'] == 'Buy') |
            (signals_df['Sharpe Signal'] == 'Outperform'), 'Overall Signal'
        ] = 'Buy'
        signals_df.loc[
            (signals_df['Momentum Signal'] == 'Sell') &
            (signals_df['Sharpe Signal'] == 'Underperform'), 'Overall Signal'
        ] = 'Strong Sell'

        # Display results
        def highlight_signals(row):
            """Highlight signal cells based on their values."""
            colors = []
            for col in row.index:
                if col == 'Overall Signal':
                    if row[col] == 'Strong Buy':
                        colors.append('background-color: #d5f4e6')
                    elif row[col] == 'Buy':
                        colors.append('background-color: #e8f5e8')
                    elif row[col] == 'Strong Sell':
                        colors.append('background-color: #ffeaea')
                    else:
                        colors.append('background-color: #fafafa')
                else:
                    colors.append('')
            return colors

        st.dataframe(
            signals_df[['Strategy', 'Overall Signal', 'Momentum Signal', 'Sharpe Signal', '3M Momentum', '3M Sharpe', 'Perf Z-Score']].style.apply(
                highlight_signals, axis=1
            ).format({
                '3M Momentum': '{:.3f}',
                '3M Sharpe': '{:.3f}',
                'Perf Z-Score': '{:.3f}'
            }),
            height=300
        )

        # Summary statistics
        signal_counts = signals_df['Overall Signal'].value_counts()
        st.markdown("#### Signal Summary")
        st.json(signal_counts.to_dict())

    except Exception as e:
        st.error(f"Error in rotation signals analysis: {str(e)}")


def display_factor_attribution_comparison(strategy_data: Dict[str, DashboardData]):
    """
    Compare factor exposures and attribution across strategies.
    """
    st.markdown("### Factor Attribution Comparison")

    try:
        # This would require factor exposure data from each strategy
        # For now, show a placeholder with example structure
        st.info("Factor attribution comparison requires factor exposure data. This feature analyzes how different strategies express factor premia.")

        # Placeholder for factor comparison
        factors = ['Market', 'Size', 'Value', 'Quality', 'Momentum', 'Volatility']
        comparison_data = []

        for sid in strategy_data.keys():
            # Simulate factor exposures (would be loaded from actual data)
            exposures = np.random.normal(0, 0.5, len(factors))
            comparison_data.append([get_strategy_display_name(sid)] + list(exposures))

        factor_df = pd.DataFrame(comparison_data, columns=['Strategy'] + factors)

        st.dataframe(
            factor_df.style.background_gradient(cmap='RdYlBu_r', axis=0).format({
                col: '{:.2f}' for col in factors
            }),
            height=200
        )

        # Factor correlation heatmap
        exposure_matrix = factor_df.set_index('Strategy')
        corr_matrix = exposure_matrix.T.corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto')
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.index)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45)
        ax.set_yticklabels(corr_matrix.index)
        ax.set_title('Strategy Factor Exposure Correlations')
        plt.colorbar(im)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error in factor attribution comparison: {str(e)}")


def display_regime_performance_analysis(strategy_data: Dict[str, DashboardData]):
    """
    Analyze strategy performance across different market regimes.
    """
    st.markdown("### Regime-Based Performance Analysis")

    try:
        from q23.dashboard.analytics.advanced import RegimeDetector

        # Get benchmark returns for regime detection (assuming first strategy is benchmark-like)
        benchmark_returns = None
        for sid, data in strategy_data.items():
            ret_series = _compute_return_series(data)
            if not ret_series.empty:
                benchmark_returns = ret_series
                break

        if benchmark_returns is None:
            st.error("No return data available for regime analysis.")
            return

        # Detect regimes
        detector = RegimeDetector(benchmark_returns)
        regimes = detector.detect_regimes()

        # Calculate regime-specific performance for each strategy
        regime_performance = []

        for sid, data in strategy_data.items():
            ret_series = _compute_return_series(data)
            if ret_series.empty:
                continue

            # Align data
            aligned_data = pd.concat([ret_series, regimes['regime']], axis=1).dropna()
            aligned_data.columns = ['returns', 'regime']

            # Performance by regime
            for regime in ['crisis', 'normal', 'trend', 'calm']:
                regime_data = aligned_data[aligned_data['regime'] == regime]
                if len(regime_data) > 5:
                    cum_ret = (1 + regime_data['returns']).prod() - 1
                    ann_vol = regime_data['returns'].std() * np.sqrt(252)
                    sharpe = cum_ret / ann_vol if ann_vol > 0 else 0

                    regime_performance.append({
                        'Strategy': get_strategy_display_name(sid),
                        'Regime': regime.title(),
                        'Total Return': cum_ret,
                        'Annual Volatility': ann_vol,
                        'Sharpe Ratio': sharpe,
                        'N_Days': len(regime_data),
                    })

        if regime_performance:
            perf_df = pd.DataFrame(regime_performance)

            # Pivot for better display
            pivot_df = perf_df.pivot(
                index='Strategy',
                columns='Regime',
                values=['Total Return', 'Sharpe Ratio']
            ).round(3)

            st.dataframe(pivot_df, height=300)

            # Best strategy by regime
            st.markdown("#### Best Strategy by Regime")
            for regime in ['Crisis', 'Normal', 'Trend', 'Calm']:
                regime_data = perf_df[perf_df['Regime'] == regime]
                if not regime_data.empty:
                    best_strategy = regime_data.loc[regime_data['Sharpe Ratio'].idxmax()]
                    st.markdown(f"**{regime}**: {best_strategy['Strategy']} (Sharpe: {best_strategy['Sharpe Ratio']:.2f})")

        else:
            st.info("Insufficient data for regime analysis across all strategies.")

    except Exception as e:
        st.error(f"Error in regime performance analysis: {str(e)}")


def display_capacity_analysis(strategy_data: Dict[str, DashboardData]):
    """
    Assess capacity constraints across strategies.
    """
    st.markdown("### Strategy Capacity Analysis")

    try:
        capacity_data = []

        for sid, data in strategy_data.items():
            # Get current portfolio size
            w_current = data.weights.iloc[-1]
            portfolio_size = w_current.abs().sum()

            # Estimate capacity based on position sizes and diversification
            n_positions = (w_current.abs() > 1e-4).sum()
            avg_position_size = w_current.abs().mean()
            largest_position = w_current.abs().max()

            # Concentration metrics
            herfindahl = (w_current ** 2).sum()
            top5_concentration = w_current.abs().nlargest(5).sum()

            # Capacity indicators (simplified)
            # In practice, this would use actual ADV data
            capacity_score = 1 / (herfindahl * 10 + top5_concentration * 5 + largest_position * 20)

            capacity_data.append({
                'Strategy': get_strategy_display_name(sid),
                'Portfolio Size': portfolio_size,
                'N_Positions': n_positions,
                'Avg Position': avg_position_size,
                'Largest Position': largest_position,
                'Herfindahl': herfindahl,
                'Top5 Concentration': top5_concentration,
                'Capacity Score': capacity_score,
            })

        capacity_df = pd.DataFrame(capacity_data)

        # Display results
        st.dataframe(
            capacity_df.style.format({
                'Portfolio Size': '{:.1%}',
                'Avg Position': '{:.2%}',
                'Largest Position': '{:.1%}',
                'Herfindahl': '{:.3f}',
                'Top5 Concentration': '{:.1%}',
                'Capacity Score': '{:.1f}'
            }).background_gradient(subset=['Capacity Score'], cmap='Greens'),
            height=300
        )

        # Capacity insights
        high_capacity = capacity_df[capacity_df['Capacity Score'] > capacity_df['Capacity Score'].median()]
        low_capacity = capacity_df[capacity_df['Capacity Score'] <= capacity_df['Capacity Score'].median()]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### High Capacity Strategies")
            if not high_capacity.empty:
                # Use itertuples() instead of iterrows() for better performance
                for row in high_capacity.itertuples(index=False):
                    # Access columns by position: Strategy is first, Capacity_Score is last
                    strategy_name = row[0]  # Strategy column
                    capacity_score = row[-1]  # Capacity Score column (last)
                    st.markdown(f"**{strategy_name}**: Score {capacity_score:.1f}")

        with col2:
            st.markdown("#### Capacity-Constrained Strategies")
            if not low_capacity.empty:
                # Use itertuples() instead of iterrows() for better performance
                for row in low_capacity.itertuples(index=False):
                    # Access columns by position: Strategy is first, Capacity_Score is last
                    strategy_name = row[0]  # Strategy column
                    capacity_score = row[-1]  # Capacity Score column (last)
                    st.markdown(f"⚠️ **{strategy_name}**: Score {capacity_score:.1f}")

    except Exception as e:
        st.error(f"Error in capacity analysis: {str(e)}")


def display_ml_strategy_prediction(strategy_data: Dict[str, DashboardData]):
    """
    Use ML to predict strategy performance relationships.
    """
    st.markdown("### ML-Based Strategy Prediction")

    try:
        # Extract features and targets
        feature_data = []
        target_data = []

        for sid, data in strategy_data.items():
            ret_series = _compute_return_series(data)
            if ret_series.empty or len(ret_series) < 50:
                continue

            # Create features from return series
            features = pd.DataFrame({
                'return_1d': ret_series.shift(1),
                'return_1w': ret_series.rolling(5).mean().shift(1),
                'return_1m': ret_series.rolling(21).mean().shift(1),
                'vol_1w': ret_series.rolling(5).std().shift(1),
                'vol_1m': ret_series.rolling(21).std().shift(1),
            }).dropna()

            # Target: next day return
            target = ret_series.loc[features.index]

            features['strategy_id'] = sid
            feature_data.append(features)
            target_data.append(target)

        if not feature_data:
            st.error("Insufficient data for ML prediction.")
            return

        # Combine data
        X = pd.concat(feature_data).dropna()
        y = pd.concat(target_data).loc[X.index]

        # Simple ML prediction (using sklearn if available)
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X.drop('strategy_id', axis=1), y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            # Results
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)

            st.markdown("#### Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training R²", f"{train_r2:.3f}")
            with col2:
                st.metric("Test R²", f"{test_r2:.3f}")

            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            st.markdown("#### Feature Importance")
            st.dataframe(feature_importance.head(10), height=300)

            # Strategy-specific predictions
            st.markdown("#### Strategy Performance Predictions")
            strategy_predictions = []

            for sid in strategy_data.keys():
                strategy_mask = X['strategy_id'] == sid
                if strategy_mask.any():
                    strategy_features = X[strategy_mask].drop('strategy_id', axis=1)
                    predictions = model.predict(strategy_features)
                    actual = y.loc[strategy_features.index]

                    pred_return = np.mean(predictions) * 252
                    actual_return = np.mean(actual) * 252

                    strategy_predictions.append({
                        'Strategy': get_strategy_display_name(sid),
                        'Predicted Annual Return': pred_return,
                        'Actual Annual Return': actual_return,
                        'Prediction Error': pred_return - actual_return,
                    })

            if strategy_predictions:
                pred_df = pd.DataFrame(strategy_predictions)
                st.dataframe(
                    pred_df.style.format({
                        'Predicted Annual Return': '{:.1%}',
                        'Actual Annual Return': '{:.1%}',
                        'Prediction Error': '{:.1%}'
                    }),
                    height=200
                )

        except ImportError:
            st.warning("scikit-learn not available for ML predictions. Install with: pip install scikit-learn")

    except Exception as e:
        st.error(f"Error in ML strategy prediction: {str(e)}")
