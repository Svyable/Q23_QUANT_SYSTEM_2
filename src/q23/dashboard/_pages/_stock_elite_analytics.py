"""
Stock Elite Analytics Page

Institutional-grade quantitative analytics for individual stocks including:
- Regime-conditional performance
- Tail risk analysis
- Beta and sensitivity metrics
- Return distribution analysis
- Signal quality assessment
- Risk decomposition

PM Best Practices:
- All metrics annualized where appropriate
- Statistical significance clearly indicated
- Sample size warnings
- Exportable data
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from q23.dashboard.core import DashboardData
from q23.dashboard.analytics import select_return_series
from q23.dashboard.analytics.stock_analytics import (
    get_available_stocks,
    get_current_holdings,
    compute_stock_returns_from_weights,
)
from q23.dashboard.analytics.stock_elite_analytics import (
    compute_stock_regime_performance,
    get_regime_exposure_summary,
    compute_stock_tail_metrics,
    compute_rolling_tail_risk,
    compute_stock_beta_analysis,
    compute_rolling_beta,
    compute_stock_distribution_stats,
    compute_qq_data,
    compute_stock_signal_quality,
    compute_stock_risk_decomposition,
    compute_elite_scorecard,
    EliteStockMetrics,
)
from q23.dashboard.components.charts import (
    PLOTLY_AVAILABLE,
    create_time_series_chart,
    get_plotly_layout,
    PM_COLORS,
)

if PLOTLY_AVAILABLE:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

# Regime colors (consistent with elite_analytics.py)
REGIME_COLORS = {
    "crisis": "#e74c3c",
    "normal": "#2ecc71",
    "trend": "#9b59b6",
    "calm": "#3498db",
}


def render_stock_elite_analytics_page(data: DashboardData) -> None:
    """
    Render the Stock Elite Analytics page.
    
    Args:
        data: Dashboard data bundle
    """
    st.subheader("🔬 Stock Elite Analytics")
    st.caption("Institutional-grade quantitative analysis for individual stocks")
    
    if data.weights is None or data.weights.empty:
        st.error("No weights data available for stock analysis.")
        return
    
    # Get available stocks
    available_stocks = get_available_stocks(data.weights)
    current_holdings = get_current_holdings(data.weights)
    
    if not available_stocks:
        st.warning("No stocks found in portfolio weights.")
        return
    
    # Stock selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_stock = st.selectbox(
            "Select Stock for Elite Analysis",
            options=available_stocks,
            help="Choose a stock for in-depth quantitative analysis"
        )
    
    with col2:
        current_only = st.checkbox(
            "Current holdings",
            value=False,
            help="Show only current portfolio holdings"
        )
        if current_only and selected_stock not in current_holdings:
            selected_stock = current_holdings[0] if current_holdings else None
    
    if not selected_stock:
        st.info("No stock selected.")
        return
    
    # Get portfolio returns
    portfolio_returns = select_return_series(data.diag)
    
    # Compute stock returns
    stock_returns = compute_stock_returns_from_weights(
        data.weights,
        portfolio_returns if portfolio_returns is not None else pd.Series(dtype=float),
        selected_stock,
    )
    
    # Get stock weights
    stock_weights = data.weights[selected_stock].fillna(0.0)
    
    # Render elite scorecard
    _render_elite_scorecard(selected_stock, stock_returns, portfolio_returns, stock_weights)
    
    st.divider()
    
    # Data quality notice
    st.caption(
        "ℹ️ Returns approximated from portfolio data. Metrics are estimates. "
        "For precise analysis, enable stock returns artifact saving."
    )
    
    # Tabs for elite analytics
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Regime",
        "📉 Tail Risk",
        "📈 Beta",
        "📋 Distribution",
        "🎯 Signal Quality",
        "⚖️ Risk Decomp",
    ])
    
    with tab1:
        _render_regime_tab(selected_stock, stock_returns, portfolio_returns, stock_weights)
    
    with tab2:
        _render_tail_risk_tab(selected_stock, stock_returns)
    
    with tab3:
        _render_beta_tab(selected_stock, stock_returns, portfolio_returns)
    
    with tab4:
        _render_distribution_tab(selected_stock, stock_returns)
    
    with tab5:
        _render_signal_quality_tab(selected_stock, stock_weights, stock_returns)
    
    with tab6:
        _render_risk_decomposition_tab(
            selected_stock, stock_returns, portfolio_returns, stock_weights
        )


def _render_elite_scorecard(
    symbol: str,
    stock_returns: Optional[pd.Series],
    portfolio_returns: Optional[pd.Series],
    weights: pd.Series,
) -> None:
    """Render the elite metrics scorecard."""
    
    st.markdown("### Elite Scorecard")
    
    if stock_returns is None or stock_returns.empty:
        st.warning("Insufficient data for elite scorecard.")
        return
    
    # Compute scorecard
    scorecard = compute_elite_scorecard(stock_returns, portfolio_returns, weights)
    
    # Display in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Risk-Adjusted**")
        st.metric("Sharpe", f"{scorecard.sharpe:.2f}")
        st.metric("Sortino", f"{scorecard.sortino:.2f}")
        st.metric("Calmar", f"{scorecard.calmar:.2f}")
    
    with col2:
        st.markdown("**Tail Risk**")
        st.metric("VaR 95%", f"{scorecard.var_95:.2%}")
        st.metric("CVaR 95%", f"{scorecard.cvar_95:.2%}")
        st.metric("Max DD", f"{scorecard.max_drawdown:.2%}")
    
    with col3:
        st.markdown("**Distribution**")
        st.metric("Skewness", f"{scorecard.skewness:.2f}")
        st.metric("Kurtosis", f"{scorecard.kurtosis:.2f}")
        st.metric("Worst Day", f"{scorecard.worst_day:.2%}")
    
    with col4:
        st.markdown("**Statistical**")
        # T-stat significance indicator
        sig = "✓" if scorecard.p_value < 0.05 else "✗"
        st.metric("T-Stat", f"{scorecard.t_stat:.2f} {sig}")
        st.metric("P-Value", f"{scorecard.p_value:.4f}")
        st.metric("N Obs", f"{scorecard.n_observations:,}")


def _render_regime_tab(
    symbol: str,
    stock_returns: Optional[pd.Series],
    portfolio_returns: Optional[pd.Series],
    weights: pd.Series,
) -> None:
    """Render regime analysis tab."""
    
    st.markdown("### Regime-Conditional Performance")
    st.caption("Stock performance broken down by market regime")
    
    if stock_returns is None or stock_returns.empty:
        st.warning("Insufficient stock return data for regime analysis.")
        return
    
    if portfolio_returns is None or portfolio_returns.empty:
        st.warning("Portfolio returns needed for regime detection.")
        return
    
    # Lookback slider
    lookback = st.slider(
        "Regime Detection Lookback (days)",
        min_value=10,
        max_value=252,
        value=30,
        step=5,
        help="Window for regime detection algorithm"
    )
    
    # Compute regime performance
    regime_data = compute_stock_regime_performance(
        stock_returns, portfolio_returns, lookback
    )
    
    if not regime_data or "regime_performance" not in regime_data:
        st.info("Insufficient data for regime analysis with current lookback.")
        return
    
    regime_perf = regime_data["regime_performance"]
    
    # Performance table
    if regime_perf:
        rows = []
        for regime_name, perf in regime_perf.items():
            rows.append({
                "Regime": regime_name.capitalize(),
                "Days": perf.n_days,
                "Total Return": f"{perf.total_return:.2%}",
                "Ann. Return": f"{perf.ann_return:.2%}",
                "Ann. Vol": f"{perf.ann_vol:.2%}",
                "Sharpe": f"{perf.sharpe:.2f}",
                "Win Rate": f"{perf.win_rate:.1%}",
            })
        
        perf_df = pd.DataFrame(rows)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Regime timeline chart
    if PLOTLY_AVAILABLE and "regime_timeline" in regime_data:
        st.markdown("### Regime Timeline")
        
        timeline = regime_data["regime_timeline"]
        
        # Create cumulative returns
        cum_ret = (1 + stock_returns).cumprod() - 1
        cum_ret = cum_ret.reindex(timeline.index).ffill().fillna(0)
        
        fig = go.Figure()
        
        # Add cumulative returns line
        fig.add_trace(
            go.Scatter(
                x=cum_ret.index,
                y=cum_ret.values,
                mode="lines",
                name="Cumulative Return",
                line=dict(color=PM_COLORS["accent"], width=2),
            )
        )
        
        # Add regime background shapes
        y_min = cum_ret.min() * 1.1 if len(cum_ret) > 0 else -0.5
        y_max = cum_ret.max() * 1.1 if len(cum_ret) > 0 else 0.5
        
        _add_regime_backgrounds(fig, timeline, y_min, y_max)
        
        fig.update_layout(
            **get_plotly_layout(title="Stock Returns by Market Regime", height=400),
            showlegend=True,
        )
        fig.update_yaxes(tickformat=".0%")
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Regime exposure summary
    st.markdown("### Regime Exposure")
    exposure = get_regime_exposure_summary(weights, portfolio_returns, lookback)
    
    if not exposure.empty:
        # Format for display
        exposure_display = exposure.copy()
        exposure_display["avg_weight"] = exposure_display["avg_weight"].apply(lambda x: f"{x:.2%}")
        exposure_display["max_weight"] = exposure_display["max_weight"].apply(lambda x: f"{x:.2%}")
        exposure_display["pct_time_held"] = exposure_display["pct_time_held"].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(exposure_display, use_container_width=True, hide_index=True)


def _add_regime_backgrounds(
    fig: "go.Figure",
    timeline: pd.DataFrame,
    y_min: float,
    y_max: float,
) -> None:
    """Add colored background shapes for regimes."""
    if "regime" not in timeline.columns:
        return
    
    current_regime = None
    regime_start = None
    dates = timeline.index
    
    for date in dates:
        regime = timeline.loc[date, "regime"]
        if regime != current_regime:
            if current_regime is not None and regime_start is not None:
                fig.add_shape(
                    type="rect",
                    x0=regime_start,
                    x1=date,
                    y0=y_min,
                    y1=y_max,
                    fillcolor=REGIME_COLORS.get(current_regime, "#808080"),
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )
            current_regime = regime
            regime_start = date
    
    # Final regime
    if current_regime is not None and regime_start is not None and len(dates) > 0:
        fig.add_shape(
            type="rect",
            x0=regime_start,
            x1=dates[-1],
            y0=y_min,
            y1=y_max,
            fillcolor=REGIME_COLORS.get(current_regime, "#808080"),
            opacity=0.2,
            layer="below",
            line_width=0,
        )


def _render_tail_risk_tab(
    symbol: str,
    stock_returns: Optional[pd.Series],
) -> None:
    """Render tail risk analysis tab."""
    
    st.markdown("### Tail Risk Analysis")
    st.caption("Extreme event risk metrics")
    
    if stock_returns is None or stock_returns.empty:
        st.warning("Insufficient data for tail risk analysis.")
        return
    
    # Compute tail metrics
    tail_metrics = compute_stock_tail_metrics(stock_returns)
    
    if "error" in tail_metrics:
        st.warning(tail_metrics["error"])
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("VaR 95%", f"{tail_metrics.get('var_95', 0):.2%}")
        st.metric("VaR 99%", f"{tail_metrics.get('var_99', 0):.2%}")
    
    with col2:
        st.metric("CVaR 95%", f"{tail_metrics.get('cvar_95', 0):.2%}")
        st.metric("CVaR 99%", f"{tail_metrics.get('cvar_99', 0):.2%}")
    
    with col3:
        st.metric("Skewness", f"{tail_metrics.get('skewness', 0):.2f}")
        st.metric("Kurtosis", f"{tail_metrics.get('kurtosis', 0):.2f}")
    
    with col4:
        st.metric("Worst Day", f"{tail_metrics.get('worst_day', 0):.2%}")
        st.metric("Worst Week", f"{tail_metrics.get('worst_week', 0):.2%}")
    
    st.divider()
    
    # Additional metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Gain/Loss Ratio", f"{tail_metrics.get('gain_loss_ratio', 0):.2f}")
    
    with col2:
        st.metric("Profit Factor", f"{tail_metrics.get('profit_factor', 0):.2f}")
    
    with col3:
        st.metric("Max Consecutive Losses", f"{tail_metrics.get('max_consecutive_losses', 0)}")
    
    recovery = tail_metrics.get('recovery_days', 0)
    if recovery == -1:
        st.warning("⚠️ Worst drawdown has not yet recovered")
    elif recovery > 0:
        st.info(f"Recovery from worst drawdown: {recovery} days")
    
    st.divider()
    
    # Rolling tail risk chart
    st.markdown("### Rolling Tail Risk")
    
    window = st.slider("Rolling Window (days)", 21, 252, 63, 21)
    
    rolling_tail = compute_rolling_tail_risk(stock_returns, window=window)
    
    if not rolling_tail.empty and PLOTLY_AVAILABLE:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.6, 0.4],
            subplot_titles=("Rolling VaR & CVaR", "Rolling Skewness & Kurtosis"),
            shared_xaxes=True,
            vertical_spacing=0.12,
        )
        
        # VaR/CVaR
        fig.add_trace(
            go.Scatter(
                x=rolling_tail.index,
                y=rolling_tail["var_95"],
                name="VaR 95%",
                line=dict(color=PM_COLORS["negative"]),
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=rolling_tail.index,
                y=rolling_tail["cvar_95"],
                name="CVaR 95%",
                line=dict(color=PM_COLORS["warning"], dash="dash"),
            ),
            row=1, col=1
        )
        
        # Skew/Kurt
        fig.add_trace(
            go.Scatter(
                x=rolling_tail.index,
                y=rolling_tail["skewness"],
                name="Skewness",
                line=dict(color=PM_COLORS["accent"]),
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=rolling_tail.index,
                y=rolling_tail["kurtosis"],
                name="Kurtosis",
                line=dict(color=PM_COLORS["neutral"]),
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            **get_plotly_layout(title="", height=500),
            showlegend=True,
        )
        fig.update_yaxes(tickformat=".1%", row=1, col=1)
        
        st.plotly_chart(fig, use_container_width=True)


def _render_beta_tab(
    symbol: str,
    stock_returns: Optional[pd.Series],
    portfolio_returns: Optional[pd.Series],
) -> None:
    """Render beta and sensitivity analysis tab."""
    
    st.markdown("### Beta & Sensitivity Analysis")
    st.caption("Systematic risk and convexity metrics")
    
    if stock_returns is None or stock_returns.empty:
        st.warning("Insufficient stock data for beta analysis.")
        return
    
    if portfolio_returns is None or portfolio_returns.empty:
        st.warning("Portfolio returns needed for beta analysis.")
        return
    
    # Compute beta analysis
    beta_data = compute_stock_beta_analysis(stock_returns, portfolio_returns)
    
    if "error" in beta_data:
        st.warning(beta_data["error"])
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Beta", f"{beta_data.get('beta', 0):.2f}")
        st.metric("R-Squared", f"{beta_data.get('r_squared', 0):.1%}")
    
    with col2:
        st.metric("Alpha (ann.)", f"{beta_data.get('alpha', 0):.2%}")
        st.metric("Correlation", f"{beta_data.get('correlation', 0):.2f}")
    
    with col3:
        st.metric("Upside Beta", f"{beta_data.get('upside_beta', 0):.2f}")
        st.metric("Downside Beta", f"{beta_data.get('downside_beta', 0):.2f}")
    
    with col4:
        asymmetry = beta_data.get('asymmetry', 0)
        delta_color = "normal" if asymmetry >= 0 else "inverse"
        st.metric(
            "Asymmetry",
            f"{asymmetry:.2f}",
            delta="Convex" if asymmetry > 0.1 else ("Concave" if asymmetry < -0.1 else "Neutral"),
            delta_color=delta_color
        )
        st.metric("Info Ratio", f"{beta_data.get('information_ratio', 0):.2f}")
    
    st.divider()
    
    # Additional metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Tracking Error", f"{beta_data.get('tracking_error', 0):.2%}")
    
    with col2:
        corr_stab = beta_data.get('correlation_stability')
        if corr_stab is not None:
            st.metric("Correlation Stability", f"{corr_stab:.2f}")
    
    st.caption(f"Based on {beta_data.get('n_observations', 0):,} overlapping observations")
    
    st.divider()
    
    # Rolling beta chart
    st.markdown("### Rolling Beta")
    
    window = st.slider("Rolling Window", 21, 252, 63, 21, key="beta_window")
    
    rolling = compute_rolling_beta(stock_returns, portfolio_returns, window)
    
    if not rolling.empty and PLOTLY_AVAILABLE:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.6, 0.4],
            subplot_titles=("Rolling Beta", "Rolling Correlation & R²"),
            shared_xaxes=True,
            vertical_spacing=0.12,
        )
        
        # Beta
        fig.add_trace(
            go.Scatter(
                x=rolling.index,
                y=rolling["rolling_beta"],
                name="Beta",
                line=dict(color=PM_COLORS["accent"], width=2),
            ),
            row=1, col=1
        )
        
        # Add beta=1 reference line
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)
        
        # Correlation and R-squared
        fig.add_trace(
            go.Scatter(
                x=rolling.index,
                y=rolling["rolling_correlation"],
                name="Correlation",
                line=dict(color=PM_COLORS["positive"]),
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=rolling.index,
                y=rolling["rolling_r_squared"],
                name="R²",
                line=dict(color=PM_COLORS["neutral"], dash="dash"),
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            **get_plotly_layout(title="", height=500),
            showlegend=True,
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    with st.expander("Interpretation Guide"):
        st.markdown("""
        - **Beta > 1**: Stock amplifies portfolio moves
        - **Beta < 1**: Stock dampens portfolio moves
        - **Upside Beta > Downside Beta**: Convex payoff (desirable)
        - **High R²**: Stock moves are well-explained by portfolio
        - **High Correlation Stability**: Relationship is consistent over time
        """)


def _render_distribution_tab(
    symbol: str,
    stock_returns: Optional[pd.Series],
) -> None:
    """Render distribution analysis tab."""
    
    st.markdown("### Return Distribution Analysis")
    st.caption("Statistical properties and normality assessment")
    
    if stock_returns is None or stock_returns.empty:
        st.warning("Insufficient data for distribution analysis.")
        return
    
    # Compute distribution stats
    dist_stats = compute_stock_distribution_stats(stock_returns)
    
    if "error" in dist_stats:
        st.warning(dist_stats["error"])
        return
    
    # Key moments
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean (daily)", f"{dist_stats.get('mean', 0):.4f}")
        st.metric("Ann. Return", f"{dist_stats.get('ann_return', 0):.2%}")
    
    with col2:
        st.metric("Std Dev (daily)", f"{dist_stats.get('std', 0):.4f}")
        st.metric("Ann. Vol", f"{dist_stats.get('ann_vol', 0):.2%}")
    
    with col3:
        st.metric("Skewness", f"{dist_stats.get('skewness', 0):.2f}")
        skew_interp = "Left tail" if dist_stats.get('skewness', 0) < -0.5 else (
            "Right tail" if dist_stats.get('skewness', 0) > 0.5 else "Symmetric"
        )
        st.caption(skew_interp)
    
    with col4:
        st.metric("Kurtosis", f"{dist_stats.get('kurtosis', 0):.2f}")
        kurt_interp = "Fat tails" if dist_stats.get('kurtosis', 0) > 1 else (
            "Thin tails" if dist_stats.get('kurtosis', 0) < -1 else "Normal-like"
        )
        st.caption(kurt_interp)
    
    st.divider()
    
    # Normality tests
    st.markdown("### Normality Tests")
    
    col1, col2 = st.columns(2)
    
    with col1:
        jb_p = dist_stats.get('jarque_bera_pvalue')
        is_normal_jb = dist_stats.get('is_normal_jb')
        status_jb = "✓ Normal" if is_normal_jb else "✗ Non-normal"
        st.metric(
            "Jarque-Bera",
            f"p={jb_p:.4f}" if jb_p is not None else "N/A",
            delta=status_jb,
            delta_color="normal" if is_normal_jb else "inverse"
        )
    
    with col2:
        shap_p = dist_stats.get('shapiro_pvalue')
        is_normal_shap = dist_stats.get('is_normal_shapiro')
        status_shap = "✓ Normal" if is_normal_shap else "✗ Non-normal"
        st.metric(
            "Shapiro-Wilk",
            f"p={shap_p:.4f}" if shap_p is not None else "N/A",
            delta=status_shap,
            delta_color="normal" if is_normal_shap else "inverse"
        )
    
    # T-test for mean
    st.markdown("### Statistical Significance")
    
    t_stat = dist_stats.get('t_stat', 0)
    t_p = dist_stats.get('t_pvalue', 1)
    is_sig = dist_stats.get('mean_significant', False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("T-Statistic", f"{t_stat:.2f}")
    with col2:
        sig_text = "✓ Significant" if is_sig else "✗ Not significant"
        st.metric("P-Value", f"{t_p:.4f}", delta=sig_text, delta_color="normal" if is_sig else "off")
    
    st.caption(f"H₀: Mean return = 0 (N = {dist_stats.get('n_observations', 0):,})")
    
    st.divider()
    
    # Histogram and QQ plot
    if PLOTLY_AVAILABLE:
        st.markdown("### Distribution Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Return Histogram vs Normal**")
            
            hist_vals = dist_stats.get('hist_values', [])
            hist_edges = dist_stats.get('hist_edges', [])
            normal_x = dist_stats.get('normal_x', [])
            normal_y = dist_stats.get('normal_y', [])
            
            if hist_vals and hist_edges:
                fig = go.Figure()
                
                # Histogram
                fig.add_trace(
                    go.Bar(
                        x=[(hist_edges[i] + hist_edges[i+1])/2 for i in range(len(hist_vals))],
                        y=hist_vals,
                        name="Actual",
                        marker_color=PM_COLORS["accent"],
                        opacity=0.7,
                    )
                )
                
                # Normal overlay
                if normal_x and normal_y:
                    fig.add_trace(
                        go.Scatter(
                            x=normal_x,
                            y=normal_y,
                            name="Normal",
                            line=dict(color=PM_COLORS["negative"], width=2),
                        )
                    )
                
                fig.update_layout(
                    **get_plotly_layout(title="", height=350),
                    showlegend=True,
                    xaxis_title="Daily Return",
                    yaxis_title="Density",
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Q-Q Plot vs Normal**")
            
            theoretical, sample = compute_qq_data(stock_returns)
            
            if len(theoretical) > 0:
                fig = go.Figure()
                
                # QQ points
                fig.add_trace(
                    go.Scatter(
                        x=theoretical,
                        y=sample,
                        mode="markers",
                        name="Q-Q",
                        marker=dict(color=PM_COLORS["accent"], size=5),
                    )
                )
                
                # 45-degree line
                min_val = min(theoretical.min(), sample.min())
                max_val = max(theoretical.max(), sample.max())
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        name="Normal",
                        line=dict(color=PM_COLORS["negative"], dash="dash"),
                    )
                )
                
                fig.update_layout(
                    **get_plotly_layout(title="", height=350),
                    showlegend=True,
                    xaxis_title="Theoretical Quantiles",
                    yaxis_title="Sample Quantiles",
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Percentiles
    st.divider()
    st.markdown("### Percentile Distribution")
    
    percentiles = dist_stats.get('percentiles', {})
    if percentiles:
        pct_df = pd.DataFrame([percentiles]).T
        pct_df.columns = ["Value"]
        pct_df.index = [f"{k.upper()}" for k in pct_df.index]
        pct_df["Value"] = pct_df["Value"].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(pct_df.T, use_container_width=True)


def _render_signal_quality_tab(
    symbol: str,
    weights: pd.Series,
    stock_returns: Optional[pd.Series],
) -> None:
    """Render signal quality analysis tab."""
    
    st.markdown("### Signal Quality Assessment")
    st.caption("How well does position size predict forward returns?")
    
    if weights is None or weights.empty:
        st.warning("No weight data available.")
        return
    
    if stock_returns is None or stock_returns.empty:
        st.warning("Insufficient return data for signal analysis.")
        return
    
    # Max horizon slider
    max_horizon = st.slider(
        "Max Forward Horizon (days)",
        min_value=5,
        max_value=63,
        value=21,
        step=1,
    )
    
    # Compute signal quality
    signal_data = compute_stock_signal_quality(weights, stock_returns, max_horizon)
    
    if "error" in signal_data:
        st.warning(signal_data["error"])
        return
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        optimal = signal_data.get('optimal_horizon', 1)
        st.metric("Optimal Holding Period", f"{optimal} days")
    
    with col2:
        hit_rate = signal_data.get('hit_rate')
        if hit_rate is not None:
            st.metric("Hit Rate", f"{hit_rate:.1%}")
        else:
            st.metric("Hit Rate", "N/A")
    
    with col3:
        rolling_ic = signal_data.get('rolling_ic_mean')
        if rolling_ic is not None:
            st.metric("Avg Rolling IC", f"{rolling_ic:.3f}")
        else:
            st.metric("Avg Rolling IC", "N/A")
    
    # Correct vs incorrect returns
    col1, col2 = st.columns(2)
    
    with col1:
        avg_correct = signal_data.get('avg_ret_correct')
        if avg_correct is not None:
            st.metric("Avg Return (Correct)", f"{avg_correct:.4f}")
    
    with col2:
        avg_incorrect = signal_data.get('avg_ret_incorrect')
        if avg_incorrect is not None:
            st.metric("Avg Return (Incorrect)", f"{avg_incorrect:.4f}")
    
    st.divider()
    
    # IC by horizon
    ic_by_horizon = signal_data.get('ic_by_horizon', {})
    
    if ic_by_horizon and PLOTLY_AVAILABLE:
        st.markdown("### IC Decay Curve")
        
        horizons = sorted(ic_by_horizon.keys())
        pearson_ic = [ic_by_horizon[h]['pearson_ic'] for h in horizons]
        rank_ic = [ic_by_horizon[h]['rank_ic'] for h in horizons]
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=horizons,
                y=pearson_ic,
                name="Pearson IC",
                mode="lines+markers",
                line=dict(color=PM_COLORS["accent"]),
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=horizons,
                y=rank_ic,
                name="Rank IC",
                mode="lines+markers",
                line=dict(color=PM_COLORS["positive"], dash="dash"),
            )
        )
        
        # Mark optimal horizon
        fig.add_vline(
            x=signal_data.get('optimal_horizon', 1),
            line_dash="dot",
            line_color="gray",
            annotation_text="Optimal",
        )
        
        fig.update_layout(
            **get_plotly_layout(title="Information Coefficient by Horizon", height=400),
            xaxis_title="Forward Horizon (days)",
            yaxis_title="IC",
            showlegend=True,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # IC table
        ic_df = pd.DataFrame({
            "Horizon": horizons,
            "Pearson IC": [f"{ic:.3f}" for ic in pearson_ic],
            "Rank IC": [f"{ic:.3f}" for ic in rank_ic],
        })
        
        st.dataframe(ic_df, use_container_width=True, hide_index=True)
    
    # Rolling IC chart
    rolling_ic_series = signal_data.get('rolling_ic_series')
    
    if rolling_ic_series is not None and not rolling_ic_series.empty and PLOTLY_AVAILABLE:
        st.divider()
        st.markdown("### Rolling IC Over Time")
        
        fig = create_time_series_chart(
            rolling_ic_series.to_frame("Rolling IC"),
            title="21-Day Rolling Weight-Return IC",
        )
        
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    with st.expander("Interpretation Guide"):
        st.markdown("""
        - **IC > 0.05**: Weak but potentially useful signal
        - **IC > 0.10**: Good predictive power
        - **IC > 0.20**: Strong signal (rare in practice)
        - **Hit Rate > 55%**: Better than random
        - **Optimal Horizon**: Hold period that maximizes IC
        """)


def _render_risk_decomposition_tab(
    symbol: str,
    stock_returns: Optional[pd.Series],
    portfolio_returns: Optional[pd.Series],
    weights: pd.Series,
) -> None:
    """Render risk decomposition analysis tab."""
    
    st.markdown("### Risk Decomposition")
    st.caption("Systematic vs idiosyncratic risk breakdown")
    
    if stock_returns is None or stock_returns.empty:
        st.warning("Insufficient stock data for risk decomposition.")
        return
    
    if portfolio_returns is None or portfolio_returns.empty:
        st.warning("Portfolio returns needed for risk decomposition.")
        return
    
    # Compute risk decomposition
    risk_data = compute_stock_risk_decomposition(
        stock_returns, portfolio_returns, weights
    )
    
    if "error" in risk_data:
        st.warning(risk_data["error"])
        return
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Total Risk**")
        st.metric("Total Vol", f"{risk_data.get('total_risk', 0):.2%}")
    
    with col2:
        st.markdown("**Systematic**")
        st.metric("Systematic Vol", f"{risk_data.get('systematic_risk', 0):.2%}")
        st.metric("% of Total", f"{risk_data.get('systematic_pct', 0):.1f}%")
    
    with col3:
        st.markdown("**Idiosyncratic**")
        st.metric("Idiosyncratic Vol", f"{risk_data.get('idiosyncratic_risk', 0):.2%}")
        st.metric("% of Total", f"{risk_data.get('idiosyncratic_pct', 0):.1f}%")
    
    st.divider()
    
    # Pie chart for risk decomposition
    if PLOTLY_AVAILABLE:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Risk Breakdown**")
            
            sys_pct = risk_data.get('systematic_pct', 50)
            idio_pct = risk_data.get('idiosyncratic_pct', 50)
            
            fig = go.Figure(
                data=[go.Pie(
                    labels=["Systematic", "Idiosyncratic"],
                    values=[sys_pct, idio_pct],
                    hole=0.4,
                    marker_colors=[PM_COLORS["accent"], PM_COLORS["neutral"]],
                )]
            )
            
            fig.update_layout(
                **get_plotly_layout(title="", height=300),
                showlegend=True,
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Portfolio Contribution**")
            
            st.metric("Beta", f"{risk_data.get('beta', 0):.2f}")
            st.metric("R²", f"{risk_data.get('r_squared', 0):.1%}")
            st.metric("Avg Weight", f"{risk_data.get('avg_weight', 0):.2%}")
            st.metric("Marginal VaR Contrib", f"{risk_data.get('marginal_var_contribution', 0):.4f}")
    
    st.divider()
    
    # Diversification benefit
    st.markdown("### Diversification Analysis")
    
    div_ratio = risk_data.get('diversification_ratio', 1.0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Diversification Ratio",
            f"{div_ratio:.2f}",
            delta="Good diversifier" if div_ratio > 1.2 else (
                "Concentrator" if div_ratio < 0.8 else "Neutral"
            ),
            delta_color="normal" if div_ratio > 1.0 else "inverse"
        )
    
    with col2:
        st.caption(f"Based on {risk_data.get('n_observations', 0):,} observations")
    
    # Interpretation
    with st.expander("Interpretation Guide"):
        st.markdown("""
        **Risk Components:**
        - **Systematic Risk**: Risk explained by portfolio/market movements (beta-driven)
        - **Idiosyncratic Risk**: Stock-specific risk not explained by the portfolio
        
        **Diversification Ratio:**
        - **> 1.0**: Stock provides diversification benefit
        - **< 1.0**: Stock adds concentrated risk
        
        **Marginal VaR Contribution:**
        - Impact on portfolio VaR from adding marginal position
        - Higher = more risk contribution per unit weight
        """)
