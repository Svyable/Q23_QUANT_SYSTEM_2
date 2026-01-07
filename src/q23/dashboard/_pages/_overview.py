"""
Overview Page

Professional strategy overview with KPIs, performance charts, portfolio snapshot,
calendar heatmap, and PM insights panel.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from io import BytesIO
import hashlib
import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from q23.dashboard.core import DashboardData
from q23.dashboard.components.styles import (
    inject_overview_css,
    style_long_weights_gradient,
    style_short_weights_gradient,
)
from q23.dashboard.components.insights import (
    PMInsightEngine,
    PMInsight,
    InsightSeverity,
    render_insights_panel,
)
from q23.dashboard.components.charts import (
    PLOTLY_AVAILABLE,
    create_time_series_chart,
    create_risk_gauge,
    create_sparkline,
    PM_COLORS,
)
from q23.dashboard.analytics import (
    summary_exposure,
    top_positions,
    period_windows,
    select_return_series,
    create_exposure_time_series,
    drawdown,
    create_calendar_heatmap,
    compute_tc_comparison,
)
from q23.dashboard.analytics.performance import (
    compute_comprehensive_performance,
    format_performance_table,
)
from q23.dashboard.analytics.viz import setup_plot_style
from q23.shared.config import cfg


# =============================================================================
# BENCHMARK COMPARISON
# =============================================================================

BENCHMARK_OPTIONS = {
    # Exchange-based benchmarks (all constituents on exchange)
    "benchmark_nys_ew": "NYSE Equal-Weight",
    "benchmark_nys_mc": "NYSE Market-Cap",
    "benchmark_nas_ew": "NASDAQ Equal-Weight",
    "benchmark_nas_mc": "NASDAQ Market-Cap",
    # Index-based benchmarks (specific index constituents)
    "benchmark_sp500_ew": "S&P 500 Equal-Weight",
    "benchmark_sp500_mc": "S&P 500 Market-Cap",
    "benchmark_nas100_ew": "NASDAQ-100 Equal-Weight",
    "benchmark_nas100_mc": "NASDAQ-100 Market-Cap",
}

BENCHMARK_COLORS = {
    # Exchange-based (grays/purples)
    "NYSE Equal-Weight": "#95a5a6",        # Gray
    "NYSE Market-Cap": "#7f8c8d",          # Dark Gray
    "NASDAQ Equal-Weight": "#9b59b6",      # Purple
    "NASDAQ Market-Cap": "#8e44ad",        # Dark Purple
    # Index-based (blues/teals)
    "S&P 500 Equal-Weight": "#3498db",     # Blue
    "S&P 500 Market-Cap": "#2980b9",       # Dark Blue
    "NASDAQ-100 Equal-Weight": "#1abc9c",  # Teal
    "NASDAQ-100 Market-Cap": "#16a085",    # Dark Teal
}


# =============================================================================
# CACHING HELPERS
# =============================================================================

def _compute_df_hash(df: Optional[pd.DataFrame]) -> str:
    """Compute a hash for a DataFrame for caching purposes."""
    if df is None or df.empty:
        return "empty"
    # Use shape and a sample of data for fast hashing
    shape_str = f"{df.shape}"
    first_row = str(df.iloc[0].values[:10]) if len(df) > 0 else ""
    last_row = str(df.iloc[-1].values[:10]) if len(df) > 0 else ""
    idx_str = f"{df.index[0]}_{df.index[-1]}" if len(df) > 0 else ""
    combined = f"{shape_str}_{first_row}_{last_row}_{idx_str}"
    return hashlib.md5(combined.encode()).hexdigest()[:16]


@st.cache_data(ttl=300, show_spinner=False)
def _cached_compute_performance(
    weights_hash: str,
    diag_hash: str,
    tc_bps: float,
    _weights: pd.DataFrame,
    _diag: Optional[pd.DataFrame],
) -> Dict:
    """
    Cached wrapper for compute_comprehensive_performance.
    
    The hash parameters are used for cache invalidation.
    The underscore-prefixed parameters are the actual data (not hashed by Streamlit).
    """
    return compute_comprehensive_performance(
        _weights, returns=None, diag=_diag, tc_bps=tc_bps
    )


@st.cache_data(ttl=300, show_spinner=False)
def _cached_generate_insights(
    weights_hash: str,
    diag_hash: str,
    exposure_hash: str,
    _weights: pd.DataFrame,
    _ret_series: Optional[pd.Series],
    _diag: Optional[pd.DataFrame],
    _exposure: Optional[pd.DataFrame],
) -> List[PMInsight]:
    """Cached wrapper for insight generation."""
    engine = PMInsightEngine()
    return engine.analyze_portfolio(
        weights=_weights,
        returns=_ret_series,
        diag=_diag,
        exposure=_exposure,
    )


def _render_benchmark_selector(
    date_range: Tuple[str, str],
) -> Dict[str, pd.Series]:
    """Render benchmark comparison selector and load selected benchmarks.
    
    Args:
        date_range: Tuple of (start_date, end_date) strings
    
    Returns:
        Dictionary mapping benchmark display names to return series
    """
    from q23.dashboard.core import (
        discover_strategy_tags,
        load_benchmark_for_comparison,
    )
    
    # Check which benchmarks have saved runs
    available = {
        k: v for k, v in BENCHMARK_OPTIONS.items() 
        if discover_strategy_tags(k)
    }
    
    if not available:
        st.caption("💡 No benchmark data available. Run benchmarks first to enable comparison.")
        return {}
    
    # Create expander for benchmark selection
    with st.expander("📊 Compare to Benchmarks", expanded=False):
        selected = st.multiselect(
            "Select benchmarks to overlay",
            options=list(available.keys()),
            format_func=lambda x: available[x],
            default=[],
            help="Overlay benchmark index returns on performance charts",
            key="benchmark_comparison_select",
        )
        
        if selected:
            st.caption(f"Selected: {', '.join([available[s] for s in selected])}")
    
    # Load selected benchmark data
    benchmark_returns: Dict[str, pd.Series] = {}
    for bench_id in selected:
        ret_series = load_benchmark_for_comparison(bench_id, date_range)
        if ret_series is not None and not ret_series.empty:
            benchmark_returns[BENCHMARK_OPTIONS[bench_id]] = ret_series
    
    return benchmark_returns


def render_overview_page(
    data: DashboardData,
    tag: str,
    date_range: Tuple[str, str],
    strategy_display_name: str,
) -> None:
    """
    Render the Overview page with professional KPIs and charts.
    
    Args:
        data: Dashboard data bundle
        tag: Current run tag
        date_range: (start_date, end_date) tuple
        strategy_display_name: Human-readable strategy name
    """
    # Inject CSS
    inject_overview_css()
    
    # Compute hashes for caching
    weights_hash = _compute_df_hash(data.weights)
    diag_hash = _compute_df_hash(data.diag)
    exposure_hash = _compute_df_hash(data.exposure)
    
    # Compute performance metrics (cached)
    tc_bps = float(getattr(cfg.strategy, "TC_BASIS_POINTS", 10.0))
    perf = _cached_compute_performance(
        weights_hash, diag_hash, tc_bps,
        _weights=data.weights, _diag=data.diag
    )
    
    # Generate PM insights (cached)
    ret_series = select_return_series(data.diag)
    insights = _cached_generate_insights(
        weights_hash, diag_hash, exposure_hash,
        _weights=data.weights,
        _ret_series=ret_series,
        _diag=data.diag,
        _exposure=data.exposure,
    )
    
    # Render strategy header
    _render_strategy_header(perf, strategy_display_name, tag, date_range)
    
    # Show critical alerts banner if any
    _render_critical_alerts_banner(insights)
    
    # Benchmark comparison selector
    benchmark_returns = _render_benchmark_selector(date_range)
    
    # Render KPIs
    _render_kpis(perf)
    
    # Render performance charts (with optional benchmark overlay)
    _render_performance_charts(data, benchmark_returns=benchmark_returns)
    
    # Render calendar heatmap
    _render_calendar_heatmap(ret_series)
    
    # Render risk metrics
    _render_risk_metrics(perf)
    
    # Render win/loss profile
    _render_winloss_profile(perf)
    
    # Render portfolio snapshot and trading activity
    _render_portfolio_snapshot(data, perf, ret_series=ret_series)
    
    st.markdown("---")
    
    # Render charts and positions
    _render_exposure_and_positions(data)
    
    st.markdown("---")
    
    # Expandable sections (including TC sensitivity)
    _render_expandable_sections(data, perf, tc_bps, tag, insights, ret_series)


def _render_critical_alerts_banner(insights: List[PMInsight]) -> None:
    """Show a banner for critical and alert-level insights."""
    critical = [i for i in insights if i.severity == InsightSeverity.CRITICAL]
    alerts = [i for i in insights if i.severity == InsightSeverity.ALERT]
    
    if critical:
        for insight in critical:
            st.error(f"🚨 **{insight.title}**: {insight.message}")
            if insight.recommendation:
                st.caption(f"💡 {insight.recommendation}")
    elif alerts:
        # Show first 2 alerts
        for insight in alerts[:2]:
            st.warning(f"⚠️ **{insight.title}**: {insight.message}")
    elif not insights:
        # No issues - show healthy status
        st.success("✅ Portfolio looks healthy - no critical issues detected")


def _generate_stats_csv(perf: Dict, strategy_name: str, tag: str, date_range: Tuple[str, str]) -> bytes:
    """Generate CSV bytes for stats download."""
    # Create a comprehensive stats DataFrame
    stats_data = {
        "Metric": [],
        "Value": [],
        "Category": [],
    }
    
    # Performance metrics
    perf_metrics = [
        ("Annual Return", f"{perf['annual_return']:.4f}", "Performance"),
        ("Sharpe Ratio", f"{perf['sharpe']:.4f}", "Performance"),
        ("Sortino Ratio", f"{perf['sortino']:.4f}", "Performance"),
        ("Calmar Ratio", f"{perf['calmar']:.4f}" if np.isfinite(perf['calmar']) else "N/A", "Performance"),
        ("Annual Volatility", f"{perf['annual_vol']:.4f}", "Performance"),
        ("Max Drawdown", f"{perf['max_drawdown']:.4f}", "Risk"),
        ("Current Drawdown", f"{perf['current_drawdown']:.4f}", "Risk"),
        ("Worst Day", f"{perf['worst_day']:.4f}", "Risk"),
        ("Worst Month", f"{perf['worst_month']:.4f}", "Risk"),
        ("Win Rate", f"{perf['win_rate']:.4f}", "Win/Loss"),
        ("Profit Factor", f"{perf['profit_factor']:.4f}", "Win/Loss"),
        ("Avg Win", f"{perf['avg_win']:.6f}", "Win/Loss"),
        ("Avg Loss", f"{perf['avg_loss']:.6f}", "Win/Loss"),
        ("Positions", f"{perf['n_positions']:.0f}", "Portfolio"),
        ("Gross Exposure", f"{perf['gross_exposure']:.4f}", "Portfolio"),
        ("Net Exposure", f"{perf['net_exposure']:.4f}", "Portfolio"),
        ("Long Exposure", f"{perf['long_exposure']:.4f}", "Portfolio"),
        ("Short Exposure", f"{perf['short_exposure']:.4f}", "Portfolio"),
        ("Avg Daily Turnover", f"{perf['avg_turnover']:.4f}", "Trading"),
        ("Top 5 Concentration", f"{perf['top5_concentration']:.4f}", "Trading"),
        ("Est Annual TC Drag", f"{perf['est_annual_tc_drag']:.4f}", "Trading"),
        ("Net Return After TC", f"{perf['net_return_after_tc']:.4f}", "Trading"),
    ]
    
    for metric, value, category in perf_metrics:
        stats_data["Metric"].append(metric)
        stats_data["Value"].append(value)
        stats_data["Category"].append(category)
    
    df = pd.DataFrame(stats_data)
    
    # Add metadata rows
    meta_rows = pd.DataFrame({
        "Metric": ["Strategy", "Run Tag", "Date Range", "Export Date"],
        "Value": [strategy_name, tag, f"{date_range[0]} to {date_range[1]}", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")],
        "Category": ["Metadata", "Metadata", "Metadata", "Metadata"],
    })
    df = pd.concat([meta_rows, df], ignore_index=True)
    
    return df.to_csv(index=False).encode('utf-8')


def _render_strategy_header(
    perf: Dict,
    strategy_display_name: str,
    tag: str,
    date_range: Tuple[str, str],
) -> None:
    """Render the strategy header with performance badge and download button."""
    sharpe = perf['sharpe']
    if sharpe >= 2.0:
        badge_class, badge_text = "badge-excellent", "Excellent"
    elif sharpe >= 1.0:
        badge_class, badge_text = "badge-good", "Good"
    elif sharpe >= 0.5:
        badge_class, badge_text = "badge-fair", "Fair"
    else:
        badge_class, badge_text = "badge-poor", "Needs Work"
    
    # Header row with title and download button
    col_title, col_download = st.columns([4, 1])
    
    with col_title:
        header_html = f"""
        <div class="strategy-header-container">
            <div class="strategy-title">Strategy Performance Overview</div>
            <div class="strategy-meta">
                <strong>{strategy_display_name}</strong> &nbsp;|&nbsp; 
                Tag: <code>{tag}</code> &nbsp;|&nbsp; 
                {date_range[0]} to {date_range[1]}
                <span class="perf-badge {badge_class}">{badge_text}</span>
            </div>
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)
    
    with col_download:
        # Generate CSV for download
        csv_data = _generate_stats_csv(perf, strategy_display_name, tag, date_range)
        filename = f"{strategy_display_name.replace(' ', '_')}_{tag}_stats.csv"
        
        st.download_button(
            label="📥 Download Stats",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            key="overview_download_stats",
            help="Download all performance metrics as CSV",
        )


def _render_kpis(perf: Dict) -> None:
    """Render primary KPI cards."""
    st.markdown(
        '<div class="section-title"><span class="icon">📊</span> Key Performance Indicators</div>',
        unsafe_allow_html=True
    )
    
    sharpe = perf['sharpe']
    ret_class = "positive" if perf['annual_return'] > 0 else "negative"
    sharpe_class = "positive" if sharpe >= 1.0 else ("neutral" if sharpe >= 0.5 else "negative")
    
    kpi_html = f"""
    <div class="kpi-row">
        <div class="kpi-card primary">
            <div class="kpi-label">Annual Return</div>
            <div class="kpi-value {ret_class}">{perf['annual_return']:.2%}</div>
        </div>
        <div class="kpi-card primary">
            <div class="kpi-label">Sharpe Ratio</div>
            <div class="kpi-value {sharpe_class}">{sharpe:.3f}</div>
        </div>
        <div class="kpi-card primary">
            <div class="kpi-label">Sortino Ratio</div>
            <div class="kpi-value {sharpe_class}">{perf['sortino']:.3f}</div>
        </div>
        <div class="kpi-card primary">
            <div class="kpi-label">Volatility</div>
            <div class="kpi-value neutral">{perf['annual_vol']:.2%}</div>
        </div>
        <div class="kpi-card primary">
            <div class="kpi-label">Calmar Ratio</div>
            <div class="kpi-value {sharpe_class}">{'%.3f' % perf['calmar'] if np.isfinite(perf['calmar']) else '—'}</div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)


def _render_performance_charts(
    data: DashboardData,
    benchmark_returns: Optional[Dict[str, pd.Series]] = None,
) -> None:
    """Render cumulative return and drawdown charts with optional benchmark overlay.
    
    Args:
        data: Dashboard data bundle
        benchmark_returns: Optional dict mapping benchmark names to return series
    """
    st.markdown(
        '<div class="section-title"><span class="icon">📈</span> Performance</div>',
        unsafe_allow_html=True
    )
    
    ret_series = select_return_series(data.diag)
    if ret_series is not None and not ret_series.empty:
        # Chart type toggle (only show if Plotly is available)
        if PLOTLY_AVAILABLE:
            use_interactive = st.checkbox(
                "Use interactive charts",
                value=True,  # Default to True for better UX with hover tooltips
                key="overview_interactive_charts",
                help="Enable interactive Plotly charts with zoom/pan and detailed hover tooltips"
            )
        else:
            use_interactive = False
        
        chart_left, chart_right = st.columns(2)
        
        with chart_left:
            if use_interactive:
                _render_cumulative_return_chart_plotly(
                    ret_series, 
                    benchmark_returns,
                    diag=data.diag,
                )
            else:
                _render_cumulative_return_chart(ret_series, benchmark_returns)
        
        with chart_right:
            if use_interactive:
                _render_drawdown_chart_plotly(
                    ret_series, 
                    benchmark_returns,
                    diag=data.diag,
                )
            else:
                _render_drawdown_chart(ret_series, benchmark_returns)
    else:
        st.info("No return series available for performance chart")


def _render_cumulative_return_chart(
    ret_series: pd.Series,
    benchmark_returns: Optional[Dict[str, pd.Series]] = None,
) -> None:
    """Render cumulative return chart with optional benchmark overlays.
    
    Args:
        ret_series: Strategy return series
        benchmark_returns: Optional dict mapping benchmark names to return series
    """
    cumret = (1.0 + ret_series).cumprod() - 1.0
    
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    
    # Plot strategy with fill
    ax.fill_between(
        cumret.index, 0, cumret.values,
        where=(cumret.values >= 0), alpha=0.3, color='#2ecc71'
    )
    ax.fill_between(
        cumret.index, 0, cumret.values,
        where=(cumret.values < 0), alpha=0.3, color='#e74c3c'
    )
    ax.plot(cumret.index, cumret.values, linewidth=2, color='#3498db', label='Strategy')
    
    # Plot benchmark overlays
    if benchmark_returns:
        for name, bench_ret in benchmark_returns.items():
            bench_cumret = (1.0 + bench_ret).cumprod() - 1.0
            # Align to strategy dates
            bench_cumret = bench_cumret.reindex(cumret.index, method='ffill')
            color = BENCHMARK_COLORS.get(name, '#95a5a6')
            ax.plot(
                bench_cumret.index, bench_cumret.values,
                linewidth=1.5, linestyle='--', color=color,
                label=name, alpha=0.8
            )
        ax.legend(loc='upper left', fontsize=7, framealpha=0.8)
    
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_title("Cumulative Return", fontsize=11, fontweight='bold', pad=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_drawdown_chart(
    ret_series: pd.Series,
    benchmark_returns: Optional[Dict[str, pd.Series]] = None,
) -> None:
    """Render drawdown chart with optional benchmark overlays.
    
    Args:
        ret_series: Strategy return series
        benchmark_returns: Optional dict mapping benchmark names to return series
    """
    eq = (1.0 + ret_series).cumprod()
    dd = eq / eq.cummax() - 1.0
    
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    
    # Plot strategy drawdown with fill
    ax.fill_between(dd.index, 0, dd.values, alpha=0.4, color='#e74c3c')
    ax.plot(dd.index, dd.values, linewidth=1.5, color='#c0392b', label='Strategy')
    
    # Plot benchmark drawdowns
    if benchmark_returns:
        for name, bench_ret in benchmark_returns.items():
            bench_eq = (1.0 + bench_ret).cumprod()
            bench_dd = bench_eq / bench_eq.cummax() - 1.0
            # Align to strategy dates
            bench_dd = bench_dd.reindex(dd.index, method='ffill')
            color = BENCHMARK_COLORS.get(name, '#95a5a6')
            ax.plot(
                bench_dd.index, bench_dd.values,
                linewidth=1.5, linestyle='--', color=color,
                label=name, alpha=0.8
            )
        ax.legend(loc='lower left', fontsize=7, framealpha=0.8)
    
    ax.axhline(y=0, color='white', linestyle='-', alpha=0.5, linewidth=1)
    ax.set_title("Drawdown", fontsize=11, fontweight='bold', pad=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_cumulative_return_chart_plotly(
    ret_series: pd.Series,
    benchmark_returns: Optional[Dict[str, pd.Series]] = None,
    diag: Optional[pd.DataFrame] = None,
) -> None:
    """Render interactive cumulative return chart with Plotly and benchmark overlays.
    
    Features Quantiacs-style unified hover tooltip showing:
    - Strategy PnL (green when positive, red when negative)
    - Benchmark PnL (if selected)
    - Underwater (drawdown)
    - Long/Short/Net exposure
    
    Args:
        ret_series: Strategy return series
        benchmark_returns: Optional dict mapping benchmark names to return series
        diag: Optional diagnostics DataFrame with exposure data
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Plotly not available")
        return
    
    cumret = (1.0 + ret_series).cumprod() - 1.0
    
    # Compute drawdown for underwater display
    eq = (1.0 + ret_series).cumprod()
    underwater = eq / eq.cummax() - 1.0
    
    # Determine strategy color based on final return (green = positive, red = negative)
    final_return = cumret.iloc[-1] if len(cumret) > 0 else 0
    strategy_color = '#2ecc71' if final_return >= 0 else '#e74c3c'
    strategy_fill = 'rgba(46, 204, 113, 0.2)' if final_return >= 0 else 'rgba(231, 76, 60, 0.2)'
    
    fig = go.Figure()
    
    # Add strategy line with area fill - dynamic green/red based on performance
    fig.add_trace(go.Scatter(
        x=cumret.index,
        y=cumret.values,
        mode='lines',
        name='PnL Strategy',
        line={'color': strategy_color, 'width': 2.5},
        fill='tozeroy',
        fillcolor=strategy_fill,
        hovertemplate='%{y:.2%}',
        showlegend=False,  # Hide from legend, keep in tooltip
    ))
    
    # Add benchmark overlays (hidden from legend but visible in tooltip)
    if benchmark_returns:
        bench_colors = ['#3498db', '#9b59b6', '#1abc9c', '#f39c12', '#95a5a6', '#e67e22']
        for i, (name, bench_ret) in enumerate(benchmark_returns.items()):
            bench_cumret = (1.0 + bench_ret).cumprod() - 1.0
            bench_cumret = bench_cumret.reindex(cumret.index, method='ffill')
            color = BENCHMARK_COLORS.get(name, bench_colors[i % len(bench_colors)])
            fig.add_trace(go.Scatter(
                x=bench_cumret.index,
                y=bench_cumret.values,
                mode='lines',
                name=f'PnL {name}',
                line={'color': color, 'width': 1.5, 'dash': 'dash'},
                hovertemplate='%{y:.2%}',
                showlegend=False,
            ))
    
    # Add underwater trace (drawdown) - visible line, shows in tooltip
    fig.add_trace(go.Scatter(
        x=underwater.index,
        y=underwater.values,
        mode='lines',
        name='Underwater',
        line={'color': '#95a5a6', 'width': 1, 'dash': 'dot'},
        hovertemplate='%{y:.2%}',
        showlegend=False,
    ))
    
    # Add exposure traces if diag data is available
    if diag is not None and not diag.empty:
        diag_aligned = diag.reindex(cumret.index, method='ffill')
        
        if 'gross_exposure' in diag_aligned.columns and 'net_exposure' in diag_aligned.columns:
            gross = diag_aligned['gross_exposure'].fillna(1.0)
            net = diag_aligned['net_exposure'].fillna(1.0)
            long_exp = (gross + net) / 2
            short_exp = (gross - net) / 2
            
            # These are invisible lines - only appear in hover tooltip
            fig.add_trace(go.Scatter(
                x=long_exp.index,
                y=long_exp.values,
                mode='lines',
                name='Long',
                line={'color': 'rgba(0,0,0,0)', 'width': 0},  # Invisible line
                hovertemplate='%{y:.2f}',
                showlegend=False,
            ))
            
            fig.add_trace(go.Scatter(
                x=short_exp.index,
                y=short_exp.values,
                mode='lines',
                name='Short',
                line={'color': 'rgba(0,0,0,0)', 'width': 0},
                hovertemplate='%{y:.2f}',
                showlegend=False,
            ))
            
            fig.add_trace(go.Scatter(
                x=net.index,
                y=net.values,
                mode='lines',
                name='Bias',
                line={'color': 'rgba(0,0,0,0)', 'width': 0},
                hovertemplate='%{y:.2f}',
                showlegend=False,
            ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    
    fig.update_layout(
        title={'text': 'Cumulative Return', 'font': {'size': 14, 'color': '#ecf0f1'}},
        paper_bgcolor='#0E1117',
        plot_bgcolor='#262730',
        font={'color': '#ecf0f1'},
        height=320,
        margin={'l': 60, 'r': 20, 't': 40, 'b': 40},
        showlegend=False,  # Hide legend entirely - tooltips still work
        xaxis={'gridcolor': '#3A3A3A', 'zerolinecolor': '#3A3A3A'},
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
    
    st.plotly_chart(fig, use_container_width=True)


def _render_drawdown_chart_plotly(
    ret_series: pd.Series,
    benchmark_returns: Optional[Dict[str, pd.Series]] = None,
    diag: Optional[pd.DataFrame] = None,
) -> None:
    """Render interactive drawdown chart with Plotly and benchmark overlays.
    
    Clean chart with unified hover tooltip showing all metrics.
    No legend clutter - tooltips show everything on hover.
    
    Args:
        ret_series: Strategy return series
        benchmark_returns: Optional dict mapping benchmark names to return series
        diag: Optional diagnostics DataFrame with additional metrics
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Plotly not available")
        return
    
    eq = (1.0 + ret_series).cumprod()
    dd = eq / eq.cummax() - 1.0
    
    fig = go.Figure()
    
    # Add strategy drawdown with red fill (drawdown is always negative)
    fig.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values,
        mode='lines',
        name='DD Strategy',
        line={'color': '#e74c3c', 'width': 2},
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.3)',
        hovertemplate='%{y:.2%}',
        showlegend=False,
    ))
    
    # Add benchmark drawdowns (no legend, visible in tooltip)
    if benchmark_returns:
        bench_colors = ['#3498db', '#9b59b6', '#1abc9c', '#f39c12', '#95a5a6', '#e67e22']
        for i, (name, bench_ret) in enumerate(benchmark_returns.items()):
            bench_eq = (1.0 + bench_ret).cumprod()
            bench_dd = bench_eq / bench_eq.cummax() - 1.0
            bench_dd = bench_dd.reindex(dd.index, method='ffill')
            color = BENCHMARK_COLORS.get(name, bench_colors[i % len(bench_colors)])
            fig.add_trace(go.Scatter(
                x=bench_dd.index,
                y=bench_dd.values,
                mode='lines',
                name=f'DD {name}',
                line={'color': color, 'width': 1.2, 'dash': 'dash'},
                hovertemplate='%{y:.2%}',
                showlegend=False,
            ))
    
    # Add additional metrics from diag - invisible lines, only in tooltip
    if diag is not None and not diag.empty:
        diag_aligned = diag.reindex(dd.index, method='ffill')
        
        if 'n_positions' in diag_aligned.columns:
            n_pos = diag_aligned['n_positions'].fillna(0)
            fig.add_trace(go.Scatter(
                x=n_pos.index,
                y=n_pos.values,
                mode='lines',
                name='Positions',
                line={'color': 'rgba(0,0,0,0)', 'width': 0},  # Invisible
                yaxis='y2',
                hovertemplate='%{y:.0f}',
                showlegend=False,
            ))
        
        if 'turnover' in diag_aligned.columns:
            turnover = diag_aligned['turnover'].fillna(0)
            fig.add_trace(go.Scatter(
                x=turnover.index,
                y=turnover.values,
                mode='lines',
                name='Turnover',
                line={'color': 'rgba(0,0,0,0)', 'width': 0},
                yaxis='y2',
                hovertemplate='%{y:.1%}',
                showlegend=False,
            ))
    
    # Add zero line
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.5)")
    
    fig.update_layout(
        title={'text': 'Drawdown', 'font': {'size': 14, 'color': '#ecf0f1'}},
        paper_bgcolor='#0E1117',
        plot_bgcolor='#262730',
        font={'color': '#ecf0f1'},
        height=320,
        margin={'l': 60, 'r': 20, 't': 40, 'b': 40},
        showlegend=False,  # No legend - cleaner chart, tooltips still work
        xaxis={'gridcolor': '#3A3A3A', 'zerolinecolor': '#3A3A3A'},
        yaxis={
            'gridcolor': '#3A3A3A',
            'zerolinecolor': '#3A3A3A',
            'tickformat': '.0%',
        },
        yaxis2={
            'overlaying': 'y',
            'side': 'right',
            'showgrid': False,
            'visible': False,  # Hide secondary axis
        },
        hovermode='x unified',
        hoverlabel={
            'bgcolor': '#1e1e1e',
            'bordercolor': '#444',
            'font': {'size': 12, 'color': '#ecf0f1'},
        },
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_calendar_heatmap(ret_series: pd.Series) -> None:
    """Render calendar heatmap of daily returns."""
    st.markdown(
        '<div class="section-title"><span class="icon">📅</span> Calendar Heatmap (Recent Year)</div>',
        unsafe_allow_html=True
    )
    
    if ret_series is None or ret_series.empty:
        st.info("No return data available for calendar heatmap")
        return
    
    # Use last ~380 days (about 1.5 years of trading days)
    recent = ret_series.tail(380)
    
    if len(recent) < 50:
        st.info("Not enough data for calendar heatmap (need at least 50 days)")
        return
    
    # Try Plotly heatmap first
    if PLOTLY_AVAILABLE:
        _render_calendar_heatmap_plotly(recent)
    else:
        # Fallback to matplotlib
        try:
            fig = create_calendar_heatmap(recent, title="Daily Returns Heatmap")
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Could not render calendar heatmap: {str(e)}")


def _render_calendar_heatmap_plotly(ret_series: pd.Series) -> None:
    """Render interactive calendar heatmap using Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    
    # Prepare data - pivot by week and day
    df = pd.DataFrame({'return': ret_series})
    df['date'] = df.index
    df['year'] = df['date'].apply(lambda x: x.year)
    df['week'] = df['date'].apply(lambda x: x.isocalendar()[1])
    df['weekday'] = df['date'].apply(lambda x: x.weekday())
    df['month'] = df['date'].apply(lambda x: x.strftime('%b'))
    
    # Create hover text
    df['hover'] = df.apply(
        lambda row: f"{row['date'].strftime('%Y-%m-%d')}<br>Return: {row['return']:.2%}",
        axis=1
    )
    
    # Get the most recent year
    most_recent_year = df['year'].max()
    df_year = df[df['year'] == most_recent_year]
    
    if len(df_year) < 20:
        # Fall back to all data if current year has too little
        df_year = df.tail(252)  # About 1 year
    
    # Create pivot table for heatmap
    # Week on x-axis, weekday on y-axis
    pivot_data = df_year.pivot_table(
        values='return',
        index='weekday',
        columns='week',
        aggfunc='mean'
    )
    
    weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # Limit to actual weekdays (0-4)
    pivot_data = pivot_data.loc[pivot_data.index.isin([0, 1, 2, 3, 4])]
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values * 100,  # Convert to percentage
        x=[f"W{w}" for w in pivot_data.columns],
        y=[weekday_labels[i] for i in pivot_data.index],
        colorscale=[
            [0, '#e74c3c'],      # Red for negative
            [0.5, '#f5f5f5'],    # White for zero
            [1, '#2ecc71'],      # Green for positive
        ],
        zmid=0,
        colorbar=dict(
            title=dict(text="Return %", side="right"),
            ticksuffix="%",
        ),
        hovertemplate="Week %{x}<br>%{y}<br>Return: %{z:.2f}%<extra></extra>",
    ))
    
    fig.update_layout(
        title={'text': f"Daily Returns Heatmap ({most_recent_year})", 'font': {'size': 14, 'color': '#ecf0f1'}},
        paper_bgcolor='#0E1117',
        plot_bgcolor='#262730',
        font={'color': '#ecf0f1'},
        height=200,
        margin={'l': 50, 'r': 20, 't': 50, 'b': 30},
        xaxis={'showgrid': False},
        yaxis={'showgrid': False},
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_tc_sensitivity(data: DashboardData, ret_series: pd.Series) -> None:
    """Render TC sensitivity analysis section."""
    if ret_series is None or data.weights is None or len(ret_series) < 20:
        return
    
    with st.expander("📊 Transaction Cost Sensitivity", expanded=False):
        st.markdown("""
        **How Sharpe ratio varies with different transaction cost assumptions.**
        Use this to stress-test the strategy's viability under different cost scenarios.
        """)
        
        try:
            tc_comparison = compute_tc_comparison(
                data.weights,
                ret_series,
                tc_bps_values=[0, 5, 10, 15, 20, 30, 50],
            )
            
            # Format for display
            display_df = tc_comparison.copy()
            display_df["annual_return"] = display_df["annual_return"].apply(lambda x: f"{x:.2%}")
            display_df["annual_vol"] = display_df["annual_vol"].apply(lambda x: f"{x:.2%}")
            display_df["sharpe"] = display_df["sharpe"].apply(lambda x: f"{x:.3f}")
            display_df["max_drawdown"] = display_df["max_drawdown"].apply(lambda x: f"{x:.2%}")
            display_df["annual_tc_drag"] = display_df["annual_tc_drag"].apply(lambda x: f"{x:.2%}")
            display_df.columns = ["TC (bps)", "Ann Return", "Ann Vol", "Sharpe", "Max DD", "TC Drag"]
            
            # Style the dataframe
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Quick insight
            sharpe_at_10 = tc_comparison[tc_comparison["tc_bps"] == 10]["sharpe"].values[0]
            sharpe_at_30 = tc_comparison[tc_comparison["tc_bps"] == 30]["sharpe"].values[0]
            
            if sharpe_at_10 > 1.0 and sharpe_at_30 > 0.5:
                st.success("✅ Strategy is robust to transaction costs")
            elif sharpe_at_10 > 0.5:
                st.info("ℹ️ Strategy is moderately sensitive to transaction costs")
            else:
                st.warning("⚠️ Strategy is highly sensitive to transaction costs")
                
        except Exception as e:
            st.warning(f"Could not compute TC sensitivity: {str(e)}")


def _render_risk_metrics(perf: Dict) -> None:
    """Render risk and drawdown metrics with visual risk gauge."""
    st.markdown(
        '<div class="section-title"><span class="icon">⚠️</span> Risk & Drawdown</div>',
        unsafe_allow_html=True
    )
    
    # Split into metrics and gauge
    col_metrics, col_gauge = st.columns([3, 1])
    
    with col_metrics:
        risk_html = f"""
        <div class="kpi-row">
            <div class="kpi-card risk">
                <div class="kpi-label">Max Drawdown</div>
                <div class="kpi-value negative">{perf['max_drawdown']:.2%}</div>
            </div>
            <div class="kpi-card risk">
                <div class="kpi-label">Current Drawdown</div>
                <div class="kpi-value negative">{perf['current_drawdown']:.2%}</div>
            </div>
            <div class="kpi-card risk">
                <div class="kpi-label">Worst Day</div>
                <div class="kpi-value negative">{perf['worst_day']:.2%}</div>
            </div>
            <div class="kpi-card risk">
                <div class="kpi-label">Worst Month</div>
                <div class="kpi-value negative">{perf['worst_month']:.2%}</div>
            </div>
        </div>
        """
        st.markdown(risk_html, unsafe_allow_html=True)
    
    with col_gauge:
        # Render risk gauge if Plotly is available
        if PLOTLY_AVAILABLE:
            current_dd = abs(perf['current_drawdown'])
            fig = create_risk_gauge(
                value=current_dd,
                title="DD Risk",
                min_val=0,
                max_val=0.25,
                thresholds=[0.05, 0.10, 0.20],
            )
            if fig is not None:
                st.plotly_chart(fig, width='stretch')
        else:
            # Fallback: simple text-based indicator
            dd_level = abs(perf['current_drawdown'])
            if dd_level < 0.05:
                st.success("Risk: LOW")
            elif dd_level < 0.10:
                st.warning("Risk: MODERATE")
            elif dd_level < 0.20:
                st.error("Risk: HIGH")
            else:
                st.error("Risk: CRITICAL")


def _render_winloss_profile(perf: Dict) -> None:
    """Render win/loss profile metrics."""
    st.markdown(
        '<div class="section-title"><span class="icon">🎯</span> Win/Loss Profile</div>',
        unsafe_allow_html=True
    )
    
    win_class = "positive" if perf['win_rate'] >= 0.5 else "negative"
    pf_class = "positive" if perf['profit_factor'] >= 1.0 else "negative"
    
    winloss_html = f"""
    <div class="kpi-row">
        <div class="kpi-card win">
            <div class="kpi-label">Win Rate</div>
            <div class="kpi-value {win_class}">{perf['win_rate']:.1%}</div>
        </div>
        <div class="kpi-card win">
            <div class="kpi-label">Profit Factor</div>
            <div class="kpi-value {pf_class}">{perf['profit_factor']:.2f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Avg Win</div>
            <div class="kpi-value positive">{perf['avg_win']:.3%}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Avg Loss</div>
            <div class="kpi-value negative">{perf['avg_loss']:.3%}</div>
        </div>
    </div>
    """
    st.markdown(winloss_html, unsafe_allow_html=True)


def _render_portfolio_snapshot(
    data: DashboardData,
    perf: Dict,
    ret_series: Optional[pd.Series] = None,
) -> None:
    """Render portfolio snapshot and trading activity with sparklines."""
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown(
            '<div class="section-title"><span class="icon">💼</span> Portfolio Snapshot</div>',
            unsafe_allow_html=True
        )
        
        snapshot_html = f"""
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-label">As of Date</div>
                <div class="kpi-value highlight">{str(data.last_dt.date())}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Positions</div>
                <div class="kpi-value neutral">{perf['n_positions']:.0f}</div>
            </div>
        </div>
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-label">Gross Exposure</div>
                <div class="kpi-value neutral">{perf['gross_exposure']:.1%}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Net Exposure</div>
                <div class="kpi-value neutral">{perf['net_exposure']:.1%}</div>
            </div>
        </div>
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-label">Long</div>
                <div class="kpi-value positive">{perf['long_exposure']:.1%}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Short</div>
                <div class="kpi-value negative">{perf['short_exposure']:.1%}</div>
            </div>
        </div>
        """
        st.markdown(snapshot_html, unsafe_allow_html=True)
    
    with col_right:
        st.markdown(
            '<div class="section-title"><span class="icon">🔄</span> Trading Activity</div>',
            unsafe_allow_html=True
        )
        
        tc_class = "negative" if perf['est_annual_tc_drag'] > 0.01 else "neutral"
        net_class = "positive" if perf['net_return_after_tc'] > 0 else "negative"
        
        trading_html = f"""
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-label">Avg Daily Turnover</div>
                <div class="kpi-value neutral">{perf['avg_turnover']:.2%}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Top 5 Concentration</div>
                <div class="kpi-value neutral">{perf['top5_concentration']:.1%}</div>
            </div>
        </div>
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-label">Est. Annual TC Drag</div>
                <div class="kpi-value {tc_class}">{perf['est_annual_tc_drag']:.2%}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Net of TC</div>
                <div class="kpi-value {net_class}">{perf['net_return_after_tc']:.2%}</div>
            </div>
        </div>
        """
        st.markdown(trading_html, unsafe_allow_html=True)
        
        # Period returns with sparklines
        perf_windows = period_windows(data.diag, data.last_dt)
        if perf_windows:
            wtd = perf_windows.get("wtd", float("nan"))
            mtd = perf_windows.get("mtd", float("nan"))
            ytd = perf_windows.get("ytd", float("nan"))

            wtd_class = "positive" if wtd > 0 else "negative" if wtd < 0 else "neutral"
            mtd_class = "positive" if mtd > 0 else "negative" if mtd < 0 else "neutral"
            ytd_class = "positive" if ytd > 0 else "negative" if ytd < 0 else "neutral"

            period_html = f"""
            <div class="kpi-row">
                <div class="kpi-card">
                    <div class="kpi-label">WTD</div>
                    <div class="kpi-value {wtd_class}">{wtd:.2%}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">MTD</div>
                    <div class="kpi-value {mtd_class}">{mtd:.2%}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">YTD</div>
                    <div class="kpi-value {ytd_class}">{ytd:.2%}</div>
                </div>
            </div>
            """
            st.markdown(period_html, unsafe_allow_html=True)
            
            # Add sparkline for recent returns trend
            if ret_series is not None and len(ret_series) >= 20 and PLOTLY_AVAILABLE:
                recent_rets = ret_series.tail(30)
                cumret = (1 + recent_rets).cumprod() - 1
                fig = create_sparkline(cumret, height=50, width=300)
                if fig is not None:
                    st.caption("30-day return trend:")
                    st.plotly_chart(fig, width='stretch')


def _render_exposure_and_positions(data: DashboardData) -> None:
    """Render exposure time series and top positions."""
    left, right = st.columns([1.2, 0.8])
    
    with left:
        st.markdown(
            '<div class="section-title"><span class="icon">📈</span> Exposure & Turnover Time Series</div>',
            unsafe_allow_html=True
        )
        if data.diag is not None and not data.diag.empty:
            fig = create_exposure_time_series(data.diag, title="")
            st.pyplot(fig)
        else:
            st.info("No portfolio diagnostics available")
    
    with right:
        st.markdown(
            '<div class="section-title"><span class="icon">🏆</span> Top Positions</div>',
            unsafe_allow_html=True
        )
        long_df, short_df = top_positions(data.w_last, n=10)
        
        lc, rc = st.columns(2)
        with lc:
            st.markdown("**Longs**")
            if not long_df.empty:
                styled_long = style_long_weights_gradient(long_df, column="weight")
                st.dataframe(styled_long, height=450, width='stretch')
            else:
                st.caption("No long positions")
        with rc:
            st.markdown("**Shorts**")
            if not short_df.empty:
                styled_short = style_short_weights_gradient(short_df, column="weight")
                st.dataframe(styled_short, height=450, width='stretch')
            else:
                st.caption("No short positions")


def _render_expandable_sections(
    data: DashboardData,
    perf: Dict,
    tc_bps: float,
    tag: str,
    insights: List[PMInsight],
    ret_series: Optional[pd.Series] = None,
) -> None:
    """Render expandable sections for detailed tables, metadata, insights, and TC analysis."""
    # PM Insights Panel
    with st.expander("🔔 PM Insights & Alerts", expanded=len(insights) > 0):
        render_insights_panel(insights, st)
    
    # TC Sensitivity Analysis (migrated from Performance page)
    _render_tc_sensitivity(data, ret_series)
    
    with st.expander("📋 Detailed Performance Table", expanded=False):
        perf_table = format_performance_table(perf)
        
        def color_values(val):
            try:
                if isinstance(val, str) and '%' in val:
                    num = float(val.replace('%', ''))
                    if num > 0:
                        return 'color: #2ecc71; font-weight: 600'
                    elif num < 0:
                        return 'color: #e74c3c; font-weight: 600'
            except:
                pass
            return ''
        
        styled_table = perf_table.style.map(color_values).set_properties(**{
            'background-color': '#1e1e1e',
            'color': '#ecf0f1',
            'border': '1px solid #333',
        }).set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#2d2d2d'),
                ('color', '#ecf0f1'),
                ('font-weight', 'bold'),
                ('text-align', 'left'),
            ]},
            {'selector': 'tr:nth-of-type(even)', 'props': [('background-color', '#252525')]},
            {'selector': 'tr:hover', 'props': [('background-color', '#2a2a2a')]},
        ])
        st.dataframe(styled_table, height=600, width='stretch')
    
    with st.expander("🔧 Run Metadata", expanded=False):
        if data.meta:
            meta_display = {
                "Tag": data.tag,
                "Base Name": data.base_name,
                "Timestamp": data.meta.get("timestamp", "—"),
                "Factors": ", ".join(data.meta.get("factors", [])) if data.meta.get("factors") else "—",
                "TC Basis Points": data.meta.get("tc_bps", tc_bps),
            }
            st.json(meta_display)
        else:
            st.info("No metadata available")
