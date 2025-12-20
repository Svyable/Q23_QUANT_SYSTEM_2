from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple

from q23.dashboard.analytics.advanced import (
    RegimeDetector,
    TailRiskAnalyzer,
    ConvexityAnalyzer,
    AlphaDecayAnalyzer,
    CapacityEstimator,
)
from q23.dashboard.analytics.scenarios import (
    ScenarioAnalyzer,
    FactorTimingAnalyzer,
    CrossSectionalMomentumTracker,
    RankICAnalyzer,
)
from q23.dashboard.analytics import select_return_series
from q23.dashboard.core import DashboardData
from q23.dashboard.components.charts import PLOTLY_AVAILABLE

# Constants
REGIME_COLORS: Dict[str, str] = {
    "crisis": "#e74c3c",  # Red
    "normal": "#2ecc71",  # Green
    "trend": "#9b59b6",   # Purple
    "calm": "#3498db",     # Blue
}

REGIME_ORDER = ["crisis", "normal", "trend", "calm"]


def _create_regime_timeline_chart(
    regimes: pd.DataFrame,
    ret_series: pd.Series,
) -> Optional["go.Figure"]:
    """Create interactive regime timeline chart with Plotly."""
    if not PLOTLY_AVAILABLE:
        return None
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None
    
    # Create figure with subplots - FIX: use shared_xaxes (plural)
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,
        subplot_titles=("Market Regime Timeline", "Underlying Signals"),
        shared_xaxes=True,  # Fixed: was shared_xaxis
    )
    # Add cumulative returns line first
    cumret = (1 + ret_series).cumprod() - 1
    cumret_aligned = cumret.reindex(regimes.index).ffill().fillna(0)
    y_min = cumret_aligned.min() * 1.1 if len(cumret_aligned) > 0 else -0.5
    y_max = cumret_aligned.max() * 1.1 if len(cumret_aligned) > 0 else 0.5
    
    # Add colored background shapes for each regime period
    _add_regime_background_shapes(fig, regimes, y_min, y_max)
    
    # Add cumulative returns line on top
    fig.add_trace(
        go.Scatter(
            x=cumret_aligned.index,
            y=cumret_aligned.values,
            mode="lines",
            name="Cumulative Returns",
            line=dict(color="#f39c12", width=2.5),
            hovertemplate="<b>Cumulative Return</b><br>%{x}<br>%{y:.2%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    
    # Add legend entries for regimes (invisible traces just for legend)
    _add_regime_legend_entries(fig, regimes)
    
    # Add underlying signals in second subplot
    _add_signal_traces(fig, regimes)
    
    # Update layout
    _update_regime_chart_layout(fig)
    
    return fig


def _add_regime_background_shapes(
    fig: "go.Figure",
    regimes: pd.DataFrame,
    y_min: float,
    y_max: float,
) -> None:
    """Add colored background rectangles for each regime period."""
    dates = regimes.index
    current_regime = None
    regime_start = None
    
    for date in dates:
        regime = regimes.loc[date, "regime"]
        if regime != current_regime:
            # End previous regime
            if current_regime is not None and regime_start is not None:
                fig.add_shape(
                    type="rect",
                    x0=regime_start,
                    x1=date,
                    y0=y_min,
                    y1=y_max,
                    fillcolor=REGIME_COLORS[current_regime],
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                    row=1,
                    col=1,
                )
            # Start new regime
            current_regime = regime
            regime_start = date
    
    # Close last regime
    if current_regime is not None and regime_start is not None and len(dates) > 0:
        fig.add_shape(
            type="rect",
            x0=regime_start,
            x1=dates[-1],
            y0=y_min,
            y1=y_max,
            fillcolor=REGIME_COLORS[current_regime],
            opacity=0.3,
            layer="below",
            line_width=0,
            row=1,
            col=1,
        )


def _add_regime_legend_entries(fig: "go.Figure", regimes: pd.DataFrame) -> None:
    """Add invisible traces for regime legend entries."""
    import plotly.graph_objects as go
    
    for regime_name in REGIME_ORDER:
        if regime_name in regimes["regime"].values:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=REGIME_COLORS[regime_name]),
                    name=regime_name.capitalize(),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )


def _add_signal_traces(fig: "go.Figure", regimes: pd.DataFrame) -> None:
    """Add underlying signal traces to the second subplot."""
    import plotly.graph_objects as go
    
    fig.add_trace(
        go.Scatter(
            x=regimes.index,
            y=regimes["vol_z"].values,
            mode="lines",
            name="Volatility Z-Score",
            line=dict(color="#e74c3c", width=1.5),
            hovertemplate="<b>Vol Z-Score</b><br>%{x}<br>%{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    
    fig.add_trace(
        go.Scatter(
            x=regimes.index,
            y=regimes["trend_corr"].values,
            mode="lines",
            name="Trend Correlation",
            line=dict(color="#3498db", width=1.5),
            hovertemplate="<b>Trend Corr</b><br>%{x}<br>%{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )


def _update_regime_chart_layout(fig: "go.Figure") -> None:
    """Update chart layout with consistent styling."""
    fig.update_layout(
        height=600,
        title="Market Regime Analysis Over Time",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#262730",
        font=dict(color="#FAFAFA", size=11),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#3A3A3A",
            borderwidth=1,
        ),
        hovermode="x unified",
        margin=dict(l=60, r=40, t=80, b=40),
    )
    
    fig.update_xaxes(
        gridcolor="#3A3A3A",
        zerolinecolor="#3A3A3A",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Regime / Cumulative Returns",
        gridcolor="#3A3A3A",
        zerolinecolor="#3A3A3A",
        tickformat=".0%",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        gridcolor="#3A3A3A",
        zerolinecolor="#3A3A3A",
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="Signal Values",
        gridcolor="#3A3A3A",
        zerolinecolor="#3A3A3A",
        row=2,
        col=1,
    )


def _calculate_performance_by_regime(
    regimes: pd.DataFrame,
    ret_series: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """Calculate performance metrics for each regime."""
    aligned_returns = ret_series.reindex(regimes.index).fillna(0)
    regime_perf = {}
    
    for regime_name in REGIME_COLORS.keys():
        regime_mask = regimes["regime"] == regime_name
        if regime_mask.sum() > 0:
            regime_rets = aligned_returns[regime_mask]
            if len(regime_rets) > 0:
                total_ret = (1 + regime_rets).prod() - 1
                n_days = len(regime_rets)
                ann_ret = (1 + total_ret) ** (252 / n_days) - 1 if n_days > 0 else 0.0
                ann_vol = regime_rets.std() * np.sqrt(252)
                sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
                win_rate = (regime_rets > 0).mean()
                
                regime_perf[regime_name] = {
                    "Total Return": total_ret,
                    "Ann. Return": ann_ret,
                    "Ann. Volatility": ann_vol,
                    "Sharpe Ratio": sharpe,
                    "Win Rate": win_rate,
                    "Days": regime_mask.sum(),
                }
    
    return regime_perf


def _calculate_regime_duration_stats(
    regimes: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """Calculate duration statistics for each regime."""
    regime_duration_stats = {}
    
    for regime_name in REGIME_COLORS.keys():
        regime_mask = regimes["regime"] == regime_name
        if regime_mask.sum() > 0:
            # Find consecutive periods
            regime_periods = (regime_mask != regime_mask.shift()).cumsum()
            period_lengths = regime_periods[regime_mask].value_counts()
            if len(period_lengths) > 0:
                regime_duration_stats[regime_name] = {
                    "Avg Duration (days)": period_lengths.mean(),
                    "Median Duration (days)": period_lengths.median(),
                    "Max Duration (days)": period_lengths.max(),
                    "Min Duration (days)": period_lengths.min(),
                    "Number of Periods": len(period_lengths),
                }
    
    return regime_duration_stats


def _render_performance_charts(regime_perf: Dict[str, Dict[str, float]]) -> None:
    """Render performance comparison charts by regime."""
    if not regime_perf:
        return
    
    perf_df = pd.DataFrame(regime_perf).T
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Annualized Returns by Regime**")
        if PLOTLY_AVAILABLE:
            import plotly.graph_objects as go
            fig_bar = go.Figure()
            for regime_name in perf_df.index:
                fig_bar.add_trace(go.Bar(
                    x=[regime_name.capitalize()],
                    y=[perf_df.loc[regime_name, "Ann. Return"]],
                    marker_color=REGIME_COLORS[regime_name],
                    name=regime_name.capitalize(),
                    text=[f"{perf_df.loc[regime_name, 'Ann. Return']:.2%}"],
                    textposition="outside",
                ))
            fig_bar.update_layout(
                height=300,
                paper_bgcolor="#0E1117",
                plot_bgcolor="#262730",
                font=dict(color="#FAFAFA"),
                showlegend=False,
                yaxis=dict(tickformat=".0%", gridcolor="#3A3A3A"),
                xaxis=dict(gridcolor="#3A3A3A"),
            )
            st.plotly_chart(fig_bar, width='stretch')
        else:
            st.bar_chart(perf_df[["Ann. Return"]])
    
    with col2:
        st.markdown("**Sharpe Ratio by Regime**")
        if PLOTLY_AVAILABLE:
            import plotly.graph_objects as go
            fig_bar2 = go.Figure()
            for regime_name in perf_df.index:
                fig_bar2.add_trace(go.Bar(
                    x=[regime_name.capitalize()],
                    y=[perf_df.loc[regime_name, "Sharpe Ratio"]],
                    marker_color=REGIME_COLORS[regime_name],
                    name=regime_name.capitalize(),
                    text=[f"{perf_df.loc[regime_name, 'Sharpe Ratio']:.2f}"],
                    textposition="outside",
                ))
            fig_bar2.update_layout(
                height=300,
                paper_bgcolor="#0E1117",
                plot_bgcolor="#262730",
                font=dict(color="#FAFAFA"),
                showlegend=False,
                yaxis=dict(gridcolor="#3A3A3A"),
                xaxis=dict(gridcolor="#3A3A3A"),
            )
            st.plotly_chart(fig_bar2, width='stretch')
        else:
            st.bar_chart(perf_df[["Sharpe Ratio"]])


def _render_duration_stats(
    regimes: pd.DataFrame,
    regime_duration_stats: Dict[str, Dict[str, float]],
) -> None:
    """Render regime duration statistics and transitions."""
    if not regime_duration_stats:
        return
    
    duration_df = pd.DataFrame(regime_duration_stats).T
    duration_df_display = duration_df.copy()
    for col in duration_df.columns:
        if "days" in col.lower() or "Duration" in col:
            duration_df_display[col] = duration_df[col].apply(lambda x: f"{x:.1f}")
        else:
            duration_df_display[col] = duration_df[col].apply(lambda x: f"{x:.0f}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(duration_df_display, height=200)
    
    with col2:
        regime_changes_count = (regimes["regime"] != regimes["regime"].shift()).sum()
        st.metric("Total Regime Changes", regime_changes_count)
        
        # Most common transitions
        transitions = []
        for i in range(1, len(regimes)):
            prev_regime = regimes["regime"].iloc[i-1]
            curr_regime = regimes["regime"].iloc[i]
            if prev_regime != curr_regime:
                transitions.append(f"{prev_regime} → {curr_regime}")
        
        if transitions:
            transition_counts = pd.Series(transitions).value_counts()
            st.markdown("**Most Common Transitions:**")
            for trans, count in transition_counts.head(5).items():
                st.text(f"{trans}: {count} times")


def render_regime_analysis_page(data: DashboardData) -> None:
    """Render the regime analysis page with enhanced visualizations."""
    st.subheader("⚡ Regime Analysis & Detection")

    ret_series = select_return_series(data.diag)

    if ret_series is None or ret_series.empty:
        st.warning("No return series available for regime analysis")
        return

    # Default to 30-day window
    lookback = st.slider("Detection Lookback (days)", 10, 500, 30, 5)

    detector = RegimeDetector(ret_series, lookback=int(lookback))
    regimes = detector.detect_regimes()

    st.markdown("### Market Regime Over Time")
    
    # Create time series chart with color-coded regime areas
    fig = _create_regime_timeline_chart(regimes, ret_series)
    if fig is not None:
        st.plotly_chart(fig, width='stretch')
    else:
        # Fallback to simple line chart if Plotly not available
        regime_numeric = regimes["regime"].map({
            "crisis": 1,
            "normal": 2,
            "trend": 3,
            "calm": 4,
        }).fillna(2)
        st.line_chart(regime_numeric, height=400)
        st.warning("Plotly not available. Install plotly for enhanced visualizations.")

    st.divider()

    # Regime statistics
    st.markdown("### Regime Statistics")
    regime_counts = regimes["regime"].value_counts()
    regime_pct = regimes["regime"].value_counts(normalize=True) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    for i, (regime_name, count) in enumerate(regime_counts.items()):
        pct = regime_pct.get(regime_name, 0)
        color = REGIME_COLORS.get(regime_name, "#95a5a6")
        with [col1, col2, col3, col4][i % 4]:
            st.metric(
                regime_name.capitalize(),
                f"{count} days ({pct:.1f}%)",
                delta=None,
            )

    st.divider()

    # Additional metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Volatility Z-Score Over Time**")
        st.line_chart(regimes[["vol_z"]], height=250)

    with col2:
        st.markdown("**Trend Correlation Over Time**")
        st.line_chart(regimes[["trend_corr"]], height=250)

    st.divider()
    
    # Performance by Regime
    st.markdown("### Performance by Regime")
    regime_perf = _calculate_performance_by_regime(regimes, ret_series)
    
    if regime_perf:
        perf_df = pd.DataFrame(regime_perf).T
        perf_df_display = perf_df.copy()
        perf_df_display["Total Return"] = perf_df_display["Total Return"].apply(lambda x: f"{x:.2%}")
        perf_df_display["Ann. Return"] = perf_df_display["Ann. Return"].apply(lambda x: f"{x:.2%}")
        perf_df_display["Ann. Volatility"] = perf_df_display["Ann. Volatility"].apply(lambda x: f"{x:.2%}")
        perf_df_display["Sharpe Ratio"] = perf_df_display["Sharpe Ratio"].apply(lambda x: f"{x:.2f}")
        perf_df_display["Win Rate"] = perf_df_display["Win Rate"].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(perf_df_display, height=200)
        
        # Visualize performance metrics
        _render_performance_charts(regime_perf)
    
    st.divider()
    
    # Regime transitions and duration analysis
    st.markdown("### Regime Duration & Transitions")
    regime_duration_stats = _calculate_regime_duration_stats(regimes)
    _render_duration_stats(regimes, regime_duration_stats)
    
    st.divider()
    
    # Additional metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Volatility Z-Score Over Time**")
        st.line_chart(regimes[["vol_z"]], height=250)

    with col2:
        st.markdown("**Trend Correlation Over Time**")
        st.line_chart(regimes[["trend_corr"]], height=250)

    st.divider()
    
    # Show recent regime details
    st.markdown("### Recent Regime Details")
    st.dataframe(regimes.tail(100), height=400)


def render_tail_risk_page(data: DashboardData):
    st.subheader("📉 Tail Risk & Stress Analysis")

    ret_series = select_return_series(data.diag)

    if ret_series is None or ret_series.empty:
        st.warning("No return series available")
        return

    analyzer = TailRiskAnalyzer(ret_series)
    tail_metrics = analyzer.compute_tail_metrics()

    st.markdown("### Tail Risk Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("VaR 95%", f"{tail_metrics.get('var_95', 0):.2%}")
    col2.metric("CVaR 95%", f"{tail_metrics.get('cvar_95', 0):.2%}")
    col3.metric("VaR 99%", f"{tail_metrics.get('var_99', 0):.2%}")
    col4.metric("CVaR 99%", f"{tail_metrics.get('cvar_99', 0):.2%}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Worst Day", f"{tail_metrics.get('worst_day', 0):.2%}")
    col6.metric("Worst Week", f"{tail_metrics.get('worst_week', 0):.2%}")
    col7.metric("Skewness", f"{tail_metrics.get('skewness', 0):.2f}")
    col8.metric("Kurtosis", f"{tail_metrics.get('kurtosis', 0):.2f}")

    st.divider()
    st.markdown("### Rolling Tail Risk")

    window = st.slider("Rolling Window", 60, 500, 252, 10)
    rolling_tail = analyzer.rolling_tail_risk(window=int(window))

    if not rolling_tail.empty:
        st.line_chart(rolling_tail[["var_95", "cvar_95"]], height=300)

    st.divider()
    st.markdown("### Stress Test Scenarios")

    if data.weights is not None and not data.weights.empty:
        scenario_analyzer = ScenarioAnalyzer(data.weights, data.weights)
        scenarios = scenario_analyzer.stress_test_scenarios()

        if not scenarios.empty:
            st.dataframe(scenarios, height=300)
        else:
            st.info("No overlapping scenario periods in data")


def render_convexity_page(data: DashboardData):
    st.subheader("🎯 Convexity & Gamma Analysis")

    if data.weights is None or data.weights.empty:
        st.warning("No weights available")
        return

    st.markdown(
        """
        **Convexity analysis** reveals option-like characteristics in your portfolio.
        Positive gamma = profit accelerates with favorable moves.
        """
    )

    returns_proxy = pd.DataFrame(
        np.random.randn(len(data.weights), data.weights.shape[1]) * 0.01,
        index=data.weights.index,
        columns=data.weights.columns,
    )

    analyzer = ConvexityAnalyzer(data.weights, returns_proxy)

    st.markdown("### Gamma Exposure (Proxy)")
    gamma = analyzer.compute_gamma_exposure()

    st.line_chart(gamma.tail(500), height=300)

    st.divider()
    st.markdown("### Option-Like Payoff Analysis")

    payoff = analyzer.option_like_payoff_analysis()

    col1, col2, col3 = st.columns(3)
    upside_beta = payoff["upside_beta"].iloc[0] if not payoff.empty else 0.0
    downside_beta = payoff["downside_beta"].iloc[0] if not payoff.empty else 0.0
    asymmetry = payoff["asymmetry"].iloc[0] if not payoff.empty else 0.0

    col1.metric("Upside Beta", f"{upside_beta:.2f}")
    col2.metric("Downside Beta", f"{downside_beta:.2f}")
    col3.metric("Asymmetry", f"{asymmetry:.2f}", delta="Higher is better")


def render_alpha_decay_page(data: DashboardData):
    st.subheader("⏱️ Alpha Decay & Optimal Holding Period")

    if data.factor_weights is None or data.factor_weights.empty:
        st.warning("No factor weights available")
        return

    st.markdown(
        """
        **Alpha decay** measures how quickly predictive power deteriorates.
        Find the optimal holding period before signal quality fades.
        """
    )

    max_horizon = st.slider("Max Horizon (days)", 5, 42, 21, 1)

    scores_proxy = data.factor_weights.cumsum(axis=0)
    returns_proxy = data.factor_weights.diff().fillna(0.0)

    analyzer = AlphaDecayAnalyzer(scores_proxy, returns_proxy)

    ic_decay = analyzer.compute_ic_decay(max_horizon=int(max_horizon))

    if not ic_decay.empty:
        st.markdown("### IC Decay Curve")
        avg_ic = ic_decay.mean(axis=0)
        st.line_chart(avg_ic, height=300)

        optimal = analyzer.optimal_holding_period(ic_decay)
        st.success(f"**Optimal Holding Period**: {optimal} days")

        st.divider()
        st.markdown("### IC Decay Heatmap (Recent 100 days)")
        st.dataframe(ic_decay.tail(100), height=400)
    else:
        st.info("Not enough data for IC decay analysis")


def render_capacity_page(data: DashboardData):
    st.subheader("💰 Strategy Capacity Estimation")

    if data.weights is None or data.weights.empty:
        st.warning("No weights available")
        return

    st.markdown(
        """
        **Capacity estimation** determines maximum AUM before market impact degrades performance.
        Based on ADV (Average Daily Volume) constraints.
        """
    )

    max_adv_pct = st.slider("Max % of ADV", 1.0, 20.0, 5.0, 0.5) / 100.0
    days_to_build = st.slider("Days to build position", 1, 10, 5, 1)

    volumes_proxy = pd.DataFrame(
        np.random.uniform(1e6, 1e9, (len(data.weights), data.weights.shape[1])),
        index=data.weights.index,
        columns=data.weights.columns,
    )

    estimator = CapacityEstimator(data.weights, volumes=volumes_proxy)

    capacity = estimator.estimate_capacity(
        max_adv_pct=max_adv_pct, days_to_build=int(days_to_build)
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Estimated Capacity",
        f"${capacity.get('estimated_capacity_usd', 0) / 1e6:.1f}M"
        if np.isfinite(capacity.get("estimated_capacity_usd", 0))
        else "∞",
    )
    col2.metric("Current Gross", f"{capacity.get('current_gross', 0):.2f}")
    col3.metric("Capacity Multiple", f"{capacity.get('capacity_multiple', 0):.1f}x")

    if "most_constrained_stock" in capacity:
        st.info(f"🔴 Most constrained: **{capacity['most_constrained_stock']}**")


# A/B Testing page deprecated - use Strategy Comparison page instead
# This function is kept for backward compatibility but should not be called
def render_ab_testing_page(data: DashboardData):
    st.subheader("🔬 A/B Strategy Comparison (Deprecated)")
    
    st.info(
        """
        **This feature has been deprecated.** 
        
        Please use the **Strategy Comparison** page instead, which provides:
        - Multi-strategy comparison with real data
        - Performance metrics, equity curves, and correlations
        - Rolling correlation analysis
        - Drawdown comparisons
        
        Navigate to **Strategy Comparison** from the main page menu.
        """
    )
    
    if st.button("Go to Strategy Comparison"):
        st.session_state.page = "Strategy Comparison"
        st.rerun()


def render_factor_timing_page(data: DashboardData):
    st.subheader("⏰ Factor Timing Skill")

    if data.factor_weights is None or data.factor_weights.empty:
        st.warning("No factor weights available")
        return

    st.markdown(
        """
        **Factor timing skill** measures your ability to allocate to factors before they perform.
        High timing correlation = good predictive ability.
        """
    )

    factor_returns_proxy = data.factor_weights.diff().fillna(0.0)

    analyzer = FactorTimingAnalyzer(data.factor_weights, factor_returns_proxy)

    timing_results = analyzer.timing_skill_test()

    if not timing_results.empty:
        st.markdown("### Timing Skill by Factor")
        st.dataframe(timing_results, height=400)

        avg_timing_corr = timing_results["timing_corr"].mean()
        avg_hit_rate = timing_results["hit_rate"].mean()

        col1, col2 = st.columns(2)
        col1.metric("Avg Timing Correlation", f"{avg_timing_corr:.3f}")
        col2.metric("Avg Hit Rate", f"{avg_hit_rate:.1%}")
    else:
        st.info("Not enough data for timing analysis")


def render_rotation_page(data: DashboardData):
    st.subheader("🔄 Cross-Sectional Rotation Velocity")

    if data.weights is None or data.weights.empty:
        st.warning("No weights available")
        return

    st.markdown(
        """
        **Rotation velocity** tracks how quickly you're shifting between stocks.
        High velocity = active rebalancing. Low velocity = sticky positions.
        """
    )

    window = st.slider("Rotation Window", 5, 63, 21, 1)

    tracker = CrossSectionalMomentumTracker(data.weights)

    velocity = tracker.rotation_velocity(window=int(window))

    st.markdown("### Rotation Velocity")
    st.line_chart(velocity.tail(500), height=300)

    avg_velocity = velocity.mean()
    st.metric("Average Velocity", f"{avg_velocity:.2f}")


def render_rank_ic_page(data: DashboardData):
    st.subheader("📊 Rank IC & Quintile Analysis")

    if data.factor_weights is None or data.factor_weights.empty:
        st.warning("No factor weights available")
        return

    st.markdown(
        """
        **Rank IC** is more robust than Pearson IC because it's insensitive to outliers.
        **Quintile analysis** shows monotonic relationship between signal and returns.
        """
    )

    scores_proxy = data.factor_weights.cumsum(axis=0)
    returns_proxy = data.factor_weights.diff().fillna(0.0)

    rank_ic = RankICAnalyzer.compute_rank_ic(scores_proxy, returns_proxy)

    st.markdown("### Rank IC Over Time")
    st.line_chart(rank_ic.tail(500), height=300)

    st.divider()
    st.markdown("### Quintile Analysis")

    quintiles = RankICAnalyzer.quintile_analysis(scores_proxy, returns_proxy, n_quantiles=5)

    if not quintiles.empty:
        avg_quintile = quintiles.mean(axis=0)
        
        # Create colored bar chart for quintiles (gradient from low to high)
        if PLOTLY_AVAILABLE:
            import plotly.graph_objects as go
            
            # Color gradient: Q1 (low) = red, Q5 (high) = green
            quintile_colors = [
                "#e74c3c",  # Q1 - Red (low)
                "#f39c12",  # Q2 - Orange
                "#f1c40f",  # Q3 - Yellow
                "#2ecc71",  # Q4 - Green
                "#27ae60",  # Q5 - Dark Green (high)
            ]
            
            fig = go.Figure()
            for i, (quintile, value) in enumerate(avg_quintile.items()):
                fig.add_trace(go.Bar(
                    x=[quintile],
                    y=[value],
                    marker_color=quintile_colors[i] if i < len(quintile_colors) else "#3498db",
                    name=quintile,
                    text=[f"{value:.3f}"],
                    textposition="outside",
                    hovertemplate=f"<b>{quintile}</b><br>Avg IC: %{{y:.3f}}<extra></extra>",
                ))
            
            fig.update_layout(
                title={"text": "Average IC by Quintile", "font": {"size": 14, "color": "#FAFAFA"}},
                paper_bgcolor="#0E1117",
                plot_bgcolor="#262730",
                font={"color": "#FAFAFA", "size": 11},
                height=350,
                showlegend=False,
                xaxis={
                    "title": "Quintile",
                    "gridcolor": "#3A3A3A",
                },
                yaxis={
                    "title": "Average IC",
                    "gridcolor": "#3A3A3A",
                    "zerolinecolor": "#3A3A3A",
                },
                margin={"l": 60, "r": 40, "t": 50, "b": 40},
            )
            
            st.plotly_chart(fig, width='stretch')
        else:
            st.bar_chart(avg_quintile)

        st.caption("Q1 = lowest signal, Q5 = highest signal. Should be monotonically increasing.")
    else:
        st.info("Not enough data for quintile analysis")
