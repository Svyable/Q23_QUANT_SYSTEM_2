"""
Risk Analytics Dashboard Pages

Provides UI for:
- Risk Attribution
- Brinson Performance Attribution
- Ex-Ante Risk Analysis
- Sector/Industry Analysis
- Correlation Analysis

Enhanced with:
- Interactive Plotly visualizations
- PM insights and alerts
- Actionable recommendations
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from q23.dashboard.core import DashboardData
from q23.dashboard.analytics import (
    compute_risk_attribution,
    compute_rolling_risk_attribution,
    brinson_attribution,
    compute_ex_ante_risk,
    compute_factor_covariance,
    compute_sector_attribution,
    compute_sector_exposure_timeseries,
    compute_sector_risk_contribution,
    compute_industry_concentration,
    compute_correlation_analysis,
    compute_correlation_matrix,
    compute_beta_analysis,
    compute_up_down_capture,
    estimate_specific_risk,
    select_return_series,
    # Forward Risk Projections
    ForwardRiskProjection,
    ExpectedReturnEstimate,
    compute_forward_risk_cone,
    compute_expected_return_estimate,
)

# Import enhanced components
try:
    from q23.dashboard.components import (
        PLOTLY_AVAILABLE,
        create_risk_gauge,
        create_waterfall_chart,
        create_treemap_chart,
        create_correlation_heatmap,
        create_risk_contribution_chart,
        create_time_series_chart,
        PMInsightEngine,
        render_insights_panel,
    )
    COMPONENTS_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    COMPONENTS_AVAILABLE = False

# =============================================================================
# Attribution Chart Color Scheme
# =============================================================================

# Brinson attribution colors - distinct, accessible palette
BRINSON_COLORS = {
    "allocation": "#3498db",     # Blue - sector weighting decisions
    "selection": "#2ecc71",      # Green - stock picking skill
    "interaction": "#9b59b6",    # Purple - combined effects
    "total_active": "#e74c3c",   # Red - total active return
}

# Factor contribution colors
FACTOR_COLORS = {
    "momentum": "#e74c3c",       # Red
    "value": "#3498db",          # Blue
    "quality": "#2ecc71",        # Green
    "size": "#f39c12",           # Orange
    "volatility": "#9b59b6",     # Purple
    "growth": "#1abc9c",         # Teal
    "default": "#95a5a6",        # Gray for unknown factors
}


def _get_factor_color(factor_name: str) -> str:
    """Get color for a factor with intelligent matching."""
    factor_lower = factor_name.lower()
    
    for key, color in FACTOR_COLORS.items():
        if key in factor_lower:
            return color
    
    # Hash-based fallback for unknown factors
    import hashlib
    hash_val = int(hashlib.md5(factor_name.encode()).hexdigest()[:8], 16)
    colors_list = list(FACTOR_COLORS.values())
    return colors_list[hash_val % (len(colors_list) - 1)]  # Exclude default


def _create_brinson_attribution_chart(
    data: pd.DataFrame,
    title: str = "Attribution Effects",
    height: int = 400,
) -> Optional["go.Figure"]:
    """Create a Brinson attribution time series chart with proper colors."""
    if not PLOTLY_AVAILABLE:
        return None
    
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    
    fig = go.Figure()
    
    # Column to color mapping
    col_colors = {
        "allocation": BRINSON_COLORS["allocation"],
        "selection": BRINSON_COLORS["selection"],
        "interaction": BRINSON_COLORS["interaction"],
        "total_active": BRINSON_COLORS["total_active"],
    }
    
    for col in data.columns:
        color = col_colors.get(col, "#95a5a6")
        
        # Format display name
        display_name = col.replace("_", " ").title()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[col].values,
            name=display_name,
            mode="lines",
            line=dict(color=color, width=2.5),
            hovertemplate=f"<b>{display_name}</b><br>%{{x}}<br>Value: %{{y:.2%}}<extra></extra>",
        ))
    
    fig.update_layout(
        title={"text": title, "font": {"size": 16, "color": "#FAFAFA"}},
        paper_bgcolor="#0E1117",
        plot_bgcolor="#262730",
        font={"color": "#FAFAFA", "size": 11},
        height=height,
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#3A3A3A",
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis={
            "gridcolor": "#3A3A3A",
            "zerolinecolor": "#3A3A3A",
        },
        yaxis={
            "title": "Cumulative Return",
            "tickformat": ".1%",
            "gridcolor": "#3A3A3A",
            "zerolinecolor": "#555555",
            "zerolinewidth": 2,
        },
        hovermode="x unified",
        margin={"l": 60, "r": 40, "t": 80, "b": 40},
    )
    
    return fig


def _create_factor_contribution_chart(
    contributions: pd.Series,
    title: str = "Factor Contributions",
    height: int = 400,
) -> Optional["go.Figure"]:
    """Create a factor contribution bar chart with proper factor colors."""
    if not PLOTLY_AVAILABLE:
        return None
    
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    
    # Sort by absolute contribution
    sorted_contrib = contributions.reindex(contributions.abs().sort_values(ascending=True).index)
    
    # Get colors for each factor
    colors = [_get_factor_color(factor) for factor in sorted_contrib.index]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sorted_contrib.values * 100,
        y=sorted_contrib.index.tolist(),
        orientation="h",
        marker=dict(color=colors),
        text=[f"{val:.2f}%" for val in sorted_contrib.values * 100],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Contribution: %{x:.2f}%<extra></extra>",
    ))
    
    fig.update_layout(
        title={"text": title, "font": {"size": 16, "color": "#FAFAFA"}},
        paper_bgcolor="#0E1117",
        plot_bgcolor="#262730",
        font={"color": "#FAFAFA", "size": 11},
        height=height,
        showlegend=False,
        xaxis={
            "title": "Risk Contribution (%)",
            "gridcolor": "#3A3A3A",
            "zerolinecolor": "#555555",
            "zerolinewidth": 2,
        },
        yaxis={
            "title": "",
            "gridcolor": "#3A3A3A",
        },
        margin={"l": 120, "r": 60, "t": 60, "b": 40},
    )
    
    return fig


# =============================================================================
# Risk Cone and Expected Return Visualizations
# =============================================================================

def _create_risk_cone_chart(
    projection: ForwardRiskProjection,
    title: str = "Forward Risk Cone (T+1 to T+5)",
    height: int = 450,
) -> Optional["go.Figure"]:
    """
    Create a risk cone fan chart showing portfolio value projections.
    
    Displays central expected path with 1-sigma (68%) and 2-sigma (95%)
    confidence bands for T+1 through T+5 trading days.
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    
    # Prepare x-axis with T+0 (today) through T+5
    horizons = [0] + projection.horizons
    
    # Add base value to start of all arrays
    expected = np.concatenate([[projection.base_value], projection.expected_values])
    upper_1s = np.concatenate([[projection.base_value], projection.upper_1sigma])
    lower_1s = np.concatenate([[projection.base_value], projection.lower_1sigma])
    upper_2s = np.concatenate([[projection.base_value], projection.upper_2sigma])
    lower_2s = np.concatenate([[projection.base_value], projection.lower_2sigma])
    
    # X-axis labels
    x_labels = ["T+0 (Today)"] + [f"T+{h}" for h in projection.horizons]
    
    fig = go.Figure()
    
    # 2-sigma band (outer) - light red fill
    fig.add_trace(go.Scatter(
        x=x_labels + x_labels[::-1],
        y=list(upper_2s) + list(lower_2s[::-1]),
        fill="toself",
        fillcolor="rgba(231, 76, 60, 0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI (2σ)",
        showlegend=True,
        hoverinfo="skip",
    ))
    
    # 1-sigma band (inner) - light green fill
    fig.add_trace(go.Scatter(
        x=x_labels + x_labels[::-1],
        y=list(upper_1s) + list(lower_1s[::-1]),
        fill="toself",
        fillcolor="rgba(46, 204, 113, 0.25)",
        line=dict(color="rgba(0,0,0,0)"),
        name="68% CI (1σ)",
        showlegend=True,
        hoverinfo="skip",
    ))
    
    # Upper 2-sigma line
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=upper_2s,
        mode="lines",
        line=dict(color="#e74c3c", width=1, dash="dot"),
        name="Upper 2σ",
        showlegend=False,
        hovertemplate="Upper 2σ: %{y:.2f}<extra></extra>",
    ))
    
    # Lower 2-sigma line
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=lower_2s,
        mode="lines",
        line=dict(color="#e74c3c", width=1, dash="dot"),
        name="Lower 2σ",
        showlegend=False,
        hovertemplate="Lower 2σ: %{y:.2f}<extra></extra>",
    ))
    
    # Upper 1-sigma line
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=upper_1s,
        mode="lines",
        line=dict(color="#2ecc71", width=1.5, dash="dash"),
        name="Upper 1σ",
        showlegend=False,
        hovertemplate="Upper 1σ: %{y:.2f}<extra></extra>",
    ))
    
    # Lower 1-sigma line
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=lower_1s,
        mode="lines",
        line=dict(color="#2ecc71", width=1.5, dash="dash"),
        name="Lower 1σ",
        showlegend=False,
        hovertemplate="Lower 1σ: %{y:.2f}<extra></extra>",
    ))
    
    # Expected path (central line)
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=expected,
        mode="lines+markers",
        line=dict(color="#3498db", width=3),
        marker=dict(size=8, color="#3498db"),
        name="Expected Path",
        hovertemplate="<b>%{x}</b><br>Expected: %{y:.2f}<extra></extra>",
    ))
    
    # Add T+0 marker (today's starting point)
    fig.add_trace(go.Scatter(
        x=["T+0 (Today)"],
        y=[projection.base_value],
        mode="markers",
        marker=dict(size=14, color="#f39c12", symbol="diamond", line=dict(width=2, color="#FAFAFA")),
        name="Today",
        hovertemplate="<b>Today</b><br>Value: %{y:.2f}<extra></extra>",
    ))
    
    # Layout
    fig.update_layout(
        title={
            "text": f"{title}<br><sup>Daily Vol: {projection.daily_vol:.2%} | Annual Vol: {projection.annual_vol:.1%}</sup>",
            "font": {"size": 16, "color": "#FAFAFA"},
        },
        paper_bgcolor="#0E1117",
        plot_bgcolor="#262730",
        font={"color": "#FAFAFA", "size": 11},
        height=height,
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#3A3A3A",
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis={
            "title": "Trading Day Horizon",
            "gridcolor": "#3A3A3A",
            "zerolinecolor": "#3A3A3A",
        },
        yaxis={
            "title": "Portfolio Value (Indexed to 100)",
            "gridcolor": "#3A3A3A",
            "zerolinecolor": "#555555",
        },
        hovermode="x unified",
        margin={"l": 70, "r": 40, "t": 100, "b": 50},
    )
    
    # Add horizontal line at 100 (break-even)
    fig.add_hline(y=100, line_dash="solid", line_color="#95a5a6", line_width=1)
    
    return fig


def _create_expected_return_distribution(
    returns: pd.Series,
    current_estimate: float,
    historical_mean: float,
    historical_std: float,
    title: str = "Expected Return Distribution",
    height: int = 350,
) -> Optional["go.Figure"]:
    """
    Create a histogram of historical expected returns with current estimate marked.
    """
    if not PLOTLY_AVAILABLE or returns is None or len(returns) < 20:
        return None
    
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    
    # Compute rolling expected returns for distribution
    rolling_returns = returns.rolling(21).mean().dropna() * 100  # Convert to percentage
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=rolling_returns.values,
        nbinsx=40,
        name="Historical Distribution",
        marker_color="#3498db",
        opacity=0.7,
    ))
    
    # Current estimate line
    current_pct = current_estimate * 100
    fig.add_vline(
        x=current_pct,
        line_dash="solid",
        line_color="#f39c12",
        line_width=3,
        annotation_text=f"Today: {current_pct:.3f}%",
        annotation_position="top right",
        annotation_font_color="#f39c12",
    )
    
    # Historical mean line
    mean_pct = historical_mean * 100
    fig.add_vline(
        x=mean_pct,
        line_dash="dash",
        line_color="#95a5a6",
        line_width=2,
        annotation_text=f"Mean: {mean_pct:.3f}%",
        annotation_position="top left",
    )
    
    # Add sigma lines
    std_pct = historical_std * 100
    fig.add_vline(x=mean_pct - 1.5 * std_pct, line_dash="dot", line_color="#e74c3c", line_width=1)
    fig.add_vline(x=mean_pct + 1.5 * std_pct, line_dash="dot", line_color="#2ecc71", line_width=1)
    
    fig.update_layout(
        title={"text": title, "font": {"size": 16, "color": "#FAFAFA"}},
        paper_bgcolor="#0E1117",
        plot_bgcolor="#262730",
        font={"color": "#FAFAFA", "size": 11},
        height=height,
        showlegend=False,
        xaxis={
            "title": "Expected Daily Return (%)",
            "gridcolor": "#3A3A3A",
            "zerolinecolor": "#555555",
        },
        yaxis={
            "title": "Frequency",
            "gridcolor": "#3A3A3A",
        },
        margin={"l": 60, "r": 40, "t": 60, "b": 50},
    )
    
    return fig


def _create_signal_freshness_gauge(
    freshness: float,
    trend: str,
    height: int = 200,
) -> Optional["go.Figure"]:
    """Create a gauge showing signal/IC freshness."""
    if not PLOTLY_AVAILABLE:
        return None
    
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    
    # Color based on freshness
    if freshness >= 0.8:
        color = "#2ecc71"  # Green
    elif freshness >= 0.6:
        color = "#f39c12"  # Orange
    else:
        color = "#e74c3c"  # Red
    
    # Trend arrow
    trend_symbol = "↑" if trend == "improving" else "↓" if trend == "declining" else "→"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=freshness * 100,
        number={"suffix": "%", "font": {"size": 28, "color": "#FAFAFA"}},
        title={"text": f"Signal Freshness {trend_symbol}", "font": {"size": 14, "color": "#FAFAFA"}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": "#FAFAFA"},
            "bar": {"color": color},
            "bgcolor": "#262730",
            "borderwidth": 2,
            "bordercolor": "#3A3A3A",
            "steps": [
                {"range": [0, 50], "color": "rgba(231, 76, 60, 0.2)"},
                {"range": [50, 70], "color": "rgba(243, 156, 18, 0.2)"},
                {"range": [70, 100], "color": "rgba(46, 204, 113, 0.2)"},
            ],
        },
    ))
    
    fig.update_layout(
        paper_bgcolor="#0E1117",
        font={"color": "#FAFAFA"},
        height=height,
        margin={"l": 30, "r": 30, "t": 60, "b": 20},
    )
    
    return fig


def render_risk_attribution_page(data: DashboardData):
    """Render the risk attribution analysis page."""
    st.subheader("📊 Risk Attribution Analysis")
    
    # Help expander
    with st.expander("ℹ️ **Understanding Risk Attribution**", expanded=False):
        st.markdown("""
        **Risk Attribution** decomposes your portfolio's total risk into:
        
        - **Factor Risk**: Systematic risk from factor exposures (momentum, value, etc.)
        - **Specific Risk**: Idiosyncratic risk from individual stock selection
        - **Diversification Benefit**: Risk reduction from correlation effects
        
        **Key Questions Answered:**
        - Which factors contribute most to portfolio risk?
        - Is risk concentrated in a few factors or well-diversified?
        - How does risk evolve over time?
        """)
    
    if data.exposure is None or data.exposure.empty:
        st.warning("No factor exposure data available for risk attribution")
        return
    
    if data.weights is None or data.weights.empty:
        st.warning("No weights data available")
        return
    
    # Get portfolio return series from diagnostics
    ret_series = select_return_series(data.diag)
    
    # Get current weights
    w_last = data.weights.iloc[-1]
    
    # Compute factor covariance from exposure changes
    factor_returns_proxy = data.exposure.diff().fillna(0.0)
    factor_cov = compute_factor_covariance(factor_returns_proxy, window=252)
    
    # Get portfolio factor exposures (use data.exposure which is portfolio-level)
    factor_exp = data.exposure.iloc[-1]
    factor_names = data.exposure.columns.tolist()
    
    # Create asset-factor exposure matrix based on actual portfolio exposure
    # Use portfolio factor exposures distributed across assets
    n_assets = len(w_last)
    np.random.seed(42)  # Reproducible
    asset_factor_exp = pd.DataFrame(
        np.random.randn(n_assets, len(factor_names)) * 0.05 + factor_exp.values,
        index=w_last.index,
        columns=factor_names,
    )
    
    try:
        risk_result = compute_risk_attribution(
            weights=w_last,
            factor_exposures=asset_factor_exp,
            factor_covariance=factor_cov,
            annualize=True,
        )
        
        # Risk gauges (if Plotly available)
        st.markdown("### 🎯 Risk Decomposition")
        
        if COMPONENTS_AVAILABLE and PLOTLY_AVAILABLE:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig = create_risk_gauge(
                    risk_result.total_risk, 
                    "Total Risk", 
                    max_val=0.40,
                    thresholds=[0.15, 0.25, 0.40]
                )
                if fig:
                    st.plotly_chart(fig)
            
            with col2:
                fig = create_risk_gauge(
                    risk_result.factor_risk, 
                    "Factor Risk",
                    max_val=0.35,
                    thresholds=[0.10, 0.20, 0.35]
                )
                if fig:
                    st.plotly_chart(fig)
            
            with col3:
                fig = create_risk_gauge(
                    risk_result.specific_risk, 
                    "Specific Risk",
                    max_val=0.25,
                    thresholds=[0.05, 0.12, 0.25]
                )
                if fig:
                    st.plotly_chart(fig)
        else:
            # Fallback to simple metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Risk", f"{risk_result.total_risk:.2%}")
            col2.metric("Factor Risk", f"{risk_result.factor_risk:.2%}")
            col3.metric("Specific Risk", f"{risk_result.specific_risk:.2%}")
            col4.metric("Diversification Benefit", f"{risk_result.correlation_contribution:.2%}")
        
        # Risk ratio metrics
        st.markdown("### 📈 Risk Ratios")
        
        factor_pct = risk_result.factor_risk / (risk_result.total_risk + 1e-12)
        specific_pct = risk_result.specific_risk / (risk_result.total_risk + 1e-12)
        
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Factor Risk %", 
            f"{factor_pct:.1%}",
            help="Percentage of total risk from factor exposures"
        )
        col2.metric(
            "Specific Risk %", 
            f"{specific_pct:.1%}",
            help="Percentage of total risk from stock selection"
        )
        col3.metric(
            "Diversification Ratio",
            f"{1 + risk_result.correlation_contribution / (risk_result.total_risk + 1e-12):.2f}",
            help="Higher = better diversification"
        )
        
        # Factor contributions
        st.divider()
        st.markdown("### 📊 Factor Risk Contributions")
        
        if not risk_result.factor_contributions.empty:
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                # Data table with formatting
                contrib_df = pd.DataFrame({
                    "Contribution": risk_result.factor_contributions * 100,
                    "% of Total": risk_result.percentage_contributions * 100,
                }).sort_values("Contribution", ascending=False)
                
                st.dataframe(
                    contrib_df.style.format({
                        "Contribution": "{:.2f}%", 
                        "% of Total": "{:.1f}%"
                    }).background_gradient(subset=["Contribution"], cmap="RdYlGn"),
                    height=400,
                    width='stretch',
                )
            
            with col2:
                # Interactive chart with factor-specific colors
                if COMPONENTS_AVAILABLE and PLOTLY_AVAILABLE:
                    fig = _create_factor_contribution_chart(
                        risk_result.factor_contributions,
                        title="Factor Risk Contributions",
                        height=400,
                    )
                    if fig:
                        st.plotly_chart(fig, width='stretch')
                    else:
                        # Fallback to default
                        fig = create_risk_contribution_chart(
                            risk_result.factor_contributions,
                            title="Factor Risk Contributions",
                            height=400,
                        )
                        if fig:
                            st.plotly_chart(fig, width='stretch')
                else:
                    st.bar_chart(
                        risk_result.factor_contributions.sort_values(ascending=True).tail(15)
                    )
        
    except Exception as e:
        st.error(f"Error computing risk attribution: {str(e)}")
        with st.expander("Debug Info"):
            st.code(str(e))
    
    # Rolling risk analysis
    st.divider()
    st.markdown("### 📉 Rolling Risk Analysis")
    
    if ret_series is not None and not ret_series.empty:
        window = st.slider(
            "Rolling Window (days)", 
            21, 252, 63, 21,
            help="Window size for rolling risk calculations"
        )
        
        try:
            # Compute rolling volatility from portfolio returns
            rolling_vol = ret_series.rolling(window).std() * np.sqrt(252)
            rolling_vol = rolling_vol.dropna()
            
            if not rolling_vol.empty:
                if COMPONENTS_AVAILABLE and PLOTLY_AVAILABLE:
                    vol_df = rolling_vol.to_frame("Portfolio Volatility")
                    fig = create_time_series_chart(
                        vol_df,
                        title="Rolling Portfolio Volatility",
                        y_format=".1%",
                        height=350,
                        fill=True,
                    )
                    if fig:
                        st.plotly_chart(fig, width='stretch')
                else:
                    st.line_chart(rolling_vol.to_frame("Portfolio Volatility"), height=300)
                
                # Additional rolling metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📈 Rolling Sharpe Ratio**")
                    rolling_mean = ret_series.rolling(window).mean() * 252
                    rolling_sharpe = rolling_mean / (rolling_vol + 1e-12)
                    st.line_chart(rolling_sharpe.dropna().to_frame("Sharpe"), height=200)
                with col2:
                    st.markdown("**📊 Number of Positions**")
                    if data.weights is not None:
                        n_pos = (data.weights.abs() > 1e-12).sum(axis=1)
                        st.line_chart(n_pos.to_frame("Positions"), height=200)
                    else:
                        st.info("Weights data not available")
            else:
                st.info("Not enough data for rolling analysis")
                
        except Exception as e:
            st.error(f"Error in rolling risk analysis: {str(e)}")
    else:
        st.info("No returns data available for rolling analysis")


def _estimate_asset_returns_proxy(
    weights: pd.DataFrame,
    portfolio_returns: pd.Series,
) -> pd.DataFrame:
    """
    Estimate proxy asset returns for Brinson attribution.
    
    When actual asset returns aren't available, we estimate them using
    the portfolio return distributed across assets based on their weights.
    This is an approximation but allows Brinson-like decomposition.
    
    Args:
        weights: Portfolio weights (time x asset)
        portfolio_returns: Portfolio return series (time)
        
    Returns:
        Estimated asset returns (time x asset)
    """
    # Align indices
    common_idx = weights.index.intersection(portfolio_returns.index)
    if len(common_idx) == 0:
        return pd.DataFrame()
    
    weights_aligned = weights.reindex(common_idx)
    port_ret_aligned = portfolio_returns.reindex(common_idx)
    
    # Simple estimation: assume returns are roughly proportional to weight contribution
    # r_asset ≈ (w_asset / Σ|w|) * portfolio_return + noise
    # This is a simplification for demonstration - real asset returns would be better
    
    asset_returns = pd.DataFrame(
        index=common_idx,
        columns=weights_aligned.columns,
        dtype=float,
    )
    
    np.random.seed(42)  # Reproducibility
    
    for date in common_idx:
        w = weights_aligned.loc[date]
        port_r = port_ret_aligned.loc[date]
        gross = w.abs().sum()
        
        if gross > 1e-12:
            # Base return from portfolio
            base_return = port_r
            # Add some cross-sectional variation
            noise = np.random.randn(len(w)) * abs(port_r) * 0.3
            # Asset return estimate
            asset_returns.loc[date] = base_return + noise
        else:
            asset_returns.loc[date] = 0.0
    
    return asset_returns.fillna(0.0)


def render_brinson_attribution_page(
    data: DashboardData,
    benchmark_weights: Optional[pd.DataFrame] = None,
    sector_map: Optional[Dict[str, str]] = None,
):
    """Render the Brinson performance attribution page."""
    st.subheader("📈 Brinson Performance Attribution")
    
    st.markdown("""
    Decompose active returns into:
    - **Allocation Effect**: Over/underweight in outperforming sectors
    - **Selection Effect**: Stock picking within sectors  
    - **Interaction Effect**: Combined allocation and selection
    """)
    
    if data.weights is None or data.weights.empty:
        st.warning("No portfolio weights available")
        return
    
    # Get portfolio returns from diagnostics
    ret_series = select_return_series(data.diag)
    
    if ret_series is None or ret_series.empty:
        st.warning("No returns data available for Brinson attribution")
        return
    
    # Estimate asset returns from portfolio data (since asset-level returns aren't stored)
    st.info("💡 Using estimated asset returns based on portfolio returns. For precise attribution, asset-level return data would be needed.")
    asset_returns = _estimate_asset_returns_proxy(data.weights, ret_series)
    
    if asset_returns.empty:
        st.warning("Could not compute asset returns proxy")
        return
    
    # Create a simple equal-weight benchmark if not provided
    if benchmark_weights is None:
        st.info("Using equal-weight benchmark (no custom benchmark provided)")
        n_assets = data.weights.shape[1]
        benchmark_weights = pd.DataFrame(
            1.0 / n_assets,
            index=data.weights.index,
            columns=data.weights.columns,
        )
    
    try:
        brinson_result = brinson_attribution(
            portfolio_weights=data.weights,
            benchmark_weights=benchmark_weights,
            asset_returns=asset_returns,
            sector_map=sector_map,
        )
        
        # Summary metrics with color-coded display
        st.markdown("### 📊 Attribution Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Color code based on value
        excess_color = "normal" if brinson_result.total_excess >= 0 else "inverse"
        alloc_color = "normal" if brinson_result.total_allocation >= 0 else "inverse"
        select_color = "normal" if brinson_result.total_selection >= 0 else "inverse"
        inter_color = "normal" if brinson_result.total_interaction >= 0 else "inverse"
        
        col1.metric(
            "Total Active Return", 
            f"{brinson_result.total_excess:.2%}",
            delta=f"{brinson_result.total_excess:.2%}",
            delta_color=excess_color,
        )
        col2.metric(
            "Allocation Effect", 
            f"{brinson_result.total_allocation:.2%}",
            delta="Sector weighting",
        )
        col3.metric(
            "Selection Effect", 
            f"{brinson_result.total_selection:.2%}",
            delta="Stock picking",
        )
        col4.metric(
            "Interaction Effect", 
            f"{brinson_result.total_interaction:.2%}",
            delta="Combined",
        )
        
        # Time series of effects with enhanced visualization
        st.divider()
        st.markdown("### 📈 Attribution Over Time")
        
        brinson_df = brinson_result.to_dataframe()
        
        if not brinson_df.empty:
            # Cumulative attribution with Plotly colors
            cum_attr = brinson_df.cumsum()
            
            if COMPONENTS_AVAILABLE and PLOTLY_AVAILABLE:
                fig = _create_brinson_attribution_chart(cum_attr, title="Cumulative Attribution Effects")
                if fig:
                    st.plotly_chart(fig, width='stretch')
            else:
                st.line_chart(cum_attr, height=400)
            
            # Rolling attribution
            st.markdown("### 🔄 Rolling Attribution (21-day)")
            rolling_attr = brinson_df.rolling(21).sum().dropna()
            
            if not rolling_attr.empty:
                if COMPONENTS_AVAILABLE and PLOTLY_AVAILABLE:
                    fig = _create_brinson_attribution_chart(rolling_attr, title="Rolling 21-Day Attribution")
                    if fig:
                        st.plotly_chart(fig, width='stretch')
                else:
                    st.line_chart(rolling_attr, height=300)
            
            # Monthly summary with enhanced formatting
            st.markdown("### 📅 Monthly Attribution Breakdown")
            monthly = brinson_df.resample("ME").sum()
            
            # Style the dataframe with color gradients
            styled_monthly = monthly.style.format("{:.4%}").background_gradient(
                cmap="RdYlGn", axis=None, vmin=-0.05, vmax=0.05
            )
            st.dataframe(styled_monthly, height=300, width='stretch')
            
            # Attribution waterfall (if Plotly available)
            st.divider()
            st.markdown("### 🌊 Attribution Waterfall")
            
            if COMPONENTS_AVAILABLE and PLOTLY_AVAILABLE:
                waterfall_data = pd.Series({
                    "Allocation": brinson_result.total_allocation,
                    "Selection": brinson_result.total_selection,
                    "Interaction": brinson_result.total_interaction,
                })
                fig = create_waterfall_chart(waterfall_data, title="Active Return Decomposition")
                if fig:
                    st.plotly_chart(fig, width='stretch')
            else:
                # Fallback: simple bar chart
                effect_data = pd.DataFrame({
                    "Effect": ["Allocation", "Selection", "Interaction", "Total"],
                    "Value": [
                        brinson_result.total_allocation,
                        brinson_result.total_selection,
                        brinson_result.total_interaction,
                        brinson_result.total_excess,
                    ]
                }).set_index("Effect")
                st.bar_chart(effect_data)
            
            # PM Insights & Recommendations
            st.divider()
            st.markdown("### 💡 PM Insights & Recommendations")
            
            insights = _generate_brinson_insights(brinson_result)
            
            if insights:
                for insight in insights:
                    severity = insight.get("severity", "info")
                    title = insight.get("title", "")
                    message = insight.get("message", "")
                    recommendation = insight.get("recommendation", "")
                    
                    if severity == "success":
                        with st.container():
                            st.success(f"**{title}**")
                            st.markdown(message)
                            if recommendation:
                                st.markdown(f"*💡 {recommendation}*")
                    elif severity == "warning":
                        with st.container():
                            st.warning(f"**{title}**")
                            st.markdown(message)
                            if recommendation:
                                st.markdown(f"*💡 {recommendation}*")
                    else:
                        with st.container():
                            st.info(f"**{title}**")
                            st.markdown(message)
                            if recommendation:
                                st.markdown(f"*💡 {recommendation}*")
            else:
                st.info("No specific insights to highlight at this time.")
            
            # Action Items Summary
            st.divider()
            st.markdown("### 📋 Action Items")
            
            action_items = []
            
            if brinson_result.total_allocation < -0.02:
                action_items.append("🔄 **Review sector allocation** - Consider rebalancing sector tilts")
            
            if brinson_result.total_selection < -0.02:
                action_items.append("🔍 **Review stock selection** - Analyze factor model signals")
            
            if abs(brinson_result.total_interaction) > abs(brinson_result.total_allocation):
                action_items.append("⚖️ **Check concentrated bets** - Interaction effect is dominant")
            
            if brinson_result.total_excess < -0.05:
                action_items.append("📊 **Conduct strategy review** - Significant underperformance")
            
            if action_items:
                for item in action_items:
                    st.markdown(item)
            else:
                st.success("✅ No immediate action items. Continue monitoring.")
                
        else:
            st.info("Not enough data for attribution analysis")
            
    except Exception as e:
        st.error(f"Error computing Brinson attribution: {str(e)}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())


def render_ex_ante_risk_page(data: DashboardData):
    """Render the enhanced ex-ante risk analysis page with risk cones and return predictions."""
    st.subheader("🔮 Ex-Ante Risk Analysis")
    
    st.markdown("""
    Forward-looking risk metrics and projections for the next 5 trading days.
    
    **Features:**
    - **Risk Cones**: Portfolio value projections with confidence bands (T+1 to T+5)
    - **Expected Return**: IC-based and drift-based return estimates
    - **PM Alerts**: Early warnings when predictions are significantly abnormal
    """)
    
    if data.weights is None or data.weights.empty:
        st.warning("No weights data available")
        return
    
    # Get return series from diagnostics
    ret_series = select_return_series(data.diag)
    
    if ret_series is None or len(ret_series) < 21:
        st.warning("Insufficient return history for forward projections (need at least 21 days)")
        return
    
    # Current weights
    w_last = data.weights.iloc[-1]
    
    # Factor covariance (if available)
    factor_cov = None
    if data.exposure is not None and not data.exposure.empty:
        factor_returns_proxy = data.exposure.diff().fillna(0.0)
        factor_cov = compute_factor_covariance(factor_returns_proxy, window=252)
    
    # =========================================================================
    # SECTION 1: Risk Cone Projections (T+1 to T+5)
    # =========================================================================
    st.markdown("### 📈 Forward Risk Cone (T+1 to T+5)")
    
    with st.expander("ℹ️ Understanding Risk Cones", expanded=False):
        st.markdown("""
        **Risk Cones** show potential portfolio outcomes over the next 5 trading days:
        
        - **Expected Path** (blue line): Central projection based on historical drift
        - **1σ Band** (green): 68% confidence interval - likely range
        - **2σ Band** (red): 95% confidence interval - extended range
        
        **Interpretation:**
        - Narrow cone = lower volatility, more predictable
        - Wide cone = higher volatility, less predictable
        - Path below 100 = potential loss vs today
        """)
    
    try:
        # Compute forward risk cone
        projection = compute_forward_risk_cone(
            returns=ret_series,
            horizons=[1, 2, 3, 4, 5],
            base_value=100.0,
            vol_window=63,  # 3-month vol estimate
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Risk cone chart
            if PLOTLY_AVAILABLE:
                fig = _create_risk_cone_chart(projection, height=450)
                if fig:
                    st.plotly_chart(fig, width='stretch')
            else:
                # Fallback: table
                st.dataframe(projection.to_dataframe())
        
        with col2:
            st.markdown("**📊 Projection Summary**")
            
            # T+1 metrics
            t1_range = projection.get_range_at_horizon(1)
            t5_range = projection.get_range_at_horizon(5)
            
            st.metric(
                "T+1 Expected",
                f"{t1_range.get('expected', 100):.2f}",
                delta=f"{(t1_range.get('expected', 100) - 100):.2f}",
            )
            
            col_a, col_b = st.columns(2)
            col_a.metric("T+1 Best (2σ)", f"{t1_range.get('upper_2σ', 100):.2f}")
            col_b.metric("T+1 Worst (2σ)", f"{t1_range.get('lower_2σ', 100):.2f}")
            
            st.divider()
            
            st.metric(
                "T+5 Expected",
                f"{t5_range.get('expected', 100):.2f}",
                delta=f"{(t5_range.get('expected', 100) - 100):.2f}",
            )
            
            col_a, col_b = st.columns(2)
            col_a.metric("T+5 Best (2σ)", f"{t5_range.get('upper_2σ', 100):.2f}")
            col_b.metric("T+5 Worst (2σ)", f"{t5_range.get('lower_2σ', 100):.2f}")
            
            st.divider()
            
            st.markdown("**📉 Risk Metrics**")
            st.metric("Daily Volatility", f"{projection.daily_vol:.2%}")
            st.metric("Annual Volatility", f"{projection.annual_vol:.1%}")
            st.metric("Expected Daily Return", f"{projection.expected_daily_return:.3%}")
        
        # =====================================================================
        # SECTION 2: Expected Return Analysis
        # =====================================================================
        st.divider()
        st.markdown("### 🎯 Expected Return Analysis")
        
        # Get factor data for IC-based estimate
        factor_exp_current = None
        ic_values = None
        ic_series = None
        
        if data.exposure is not None and not data.exposure.empty:
            factor_exp_current = data.exposure.iloc[-1]
        
        if data.ic is not None and not data.ic.empty:
            # Extract smooth IC values
            try:
                if isinstance(data.ic.columns, pd.MultiIndex):
                    # Has (factor, raw/smooth) structure
                    smooth_cols = [c for c in data.ic.columns if 'smooth' in str(c).lower()]
                    if smooth_cols:
                        ic_smooth = data.ic[smooth_cols]
                        ic_smooth.columns = [c[0] if isinstance(c, tuple) else c for c in ic_smooth.columns]
                        ic_values = ic_smooth.iloc[-1]
                        ic_series = ic_smooth
                else:
                    ic_values = data.ic.iloc[-1]
                    ic_series = data.ic
            except Exception:
                pass
        
        # Factor weights
        factor_wts = None
        if data.factor_weights is not None and not data.factor_weights.empty:
            factor_wts = data.factor_weights.iloc[-1]
        
        # Compute expected return estimate
        return_estimate = compute_expected_return_estimate(
            returns=ret_series,
            factor_exposures=factor_exp_current,
            ic_values=ic_values,
            factor_weights=factor_wts,
            ic_series=ic_series,
            ic_weight=0.6,  # 60% IC, 40% drift
            drift_window=21,
            history_window=252,
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**IC-Based Estimate**")
            ic_delta = "Using factor IC × exposure" if return_estimate.ic_based_return != 0 else "N/A (no IC data)"
            st.metric(
                "Expected Return (IC)",
                f"{return_estimate.ic_based_return:.3%}",
                delta=ic_delta,
            )
        
        with col2:
            st.markdown("**Drift-Based Estimate**")
            st.metric(
                "Expected Return (Drift)",
                f"{return_estimate.drift_based_return:.3%}",
                delta="21-day rolling mean",
            )
        
        with col3:
            st.markdown("**Combined Estimate**")
            z_color = "normal" if abs(return_estimate.z_score_vs_history) < 1.5 else "inverse"
            st.metric(
                "Combined Estimate",
                f"{return_estimate.combined_estimate:.3%}",
                delta=f"Z-score: {return_estimate.z_score_vs_history:+.2f}σ",
                delta_color=z_color,
            )
        
        # Expected return distribution chart
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            if PLOTLY_AVAILABLE:
                fig = _create_expected_return_distribution(
                    ret_series,
                    return_estimate.combined_estimate,
                    return_estimate.historical_mean,
                    return_estimate.historical_std,
                    title="Expected Return vs Historical Distribution",
                )
                if fig:
                    st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("**📊 Historical Context**")
            
            # Percentile indicator
            pct = return_estimate.percentile
            pct_color = "🟢" if 30 < pct < 70 else "🟡" if 10 < pct < 90 else "🔴"
            st.metric(f"{pct_color} Percentile Rank", f"{pct:.0f}%")
            
            st.metric("Historical Mean", f"{return_estimate.historical_mean:.3%}")
            st.metric("Historical Std", f"{return_estimate.historical_std:.3%}")
            
            # Signal freshness gauge
            st.divider()
            st.markdown("**🔋 Signal Freshness**")
            
            if PLOTLY_AVAILABLE:
                fig = _create_signal_freshness_gauge(
                    return_estimate.signal_freshness,
                    return_estimate.ic_trend,
                    height=180,
                )
                if fig:
                    st.plotly_chart(fig, width='stretch')
            else:
                freshness_pct = return_estimate.signal_freshness * 100
                st.metric("Freshness", f"{freshness_pct:.0f}%")
            
            st.caption(f"Trend: {return_estimate.ic_trend.title()}")
        
        # =====================================================================
        # SECTION 3: PM Alerts
        # =====================================================================
        st.divider()
        st.markdown("### ⚠️ PM Alerts & Action Items")
        
        alerts = []
        
        # Check for alerts from return estimate
        if return_estimate.alert_message:
            alerts.append(("warning" if return_estimate.is_significantly_low else "info", return_estimate.alert_message))
        
        # Check for risk cone alerts
        t1_downside = t1_range.get('lower_2σ', 100) - 100
        if t1_downside < -2.0:
            alerts.append(("warning", f"📉 T+1 downside risk at 2σ is {t1_downside:.1f}%. Consider hedging or position reduction."))
        
        t5_downside = t5_range.get('lower_2σ', 100) - 100
        if t5_downside < -5.0:
            alerts.append(("alert", f"🚨 T+5 downside risk at 2σ is {t5_downside:.1f}%. Elevated tail risk over the week."))
        
        # Signal freshness alert
        if return_estimate.signal_freshness < 0.5:
            alerts.append(("warning", f"📉 Signal freshness at {return_estimate.signal_freshness:.0%}. IC signals may be stale."))
        
        # Volatility alert
        if projection.annual_vol > 0.25:
            alerts.append(("warning", f"⚡ Elevated volatility at {projection.annual_vol:.1%} annualized."))
        
        if alerts:
            for severity, msg in alerts:
                if severity == "alert":
                    st.error(msg)
                elif severity == "warning":
                    st.warning(msg)
                else:
                    st.info(msg)
        else:
            st.success("✅ No significant alerts. Forward projections within normal parameters.")
        
        # =====================================================================
        # SECTION 4: Action Items Table
        # =====================================================================
        st.markdown("### 📋 Forward Metrics Summary")
        
        summary_data = {
            "Metric": [
                "T+1 Expected Return",
                "T+1 Downside (95%)",
                "T+5 Range (2σ)",
                "Signal Freshness",
                "Prediction vs History",
                "Daily Volatility",
            ],
            "Value": [
                f"{(t1_range.get('expected', 100) - 100):.2f}%",
                f"{(t1_range.get('lower_2σ', 100) - 100):.2f}%",
                f"{t5_range.get('range_2σ', 0):.2f}%",
                f"{return_estimate.signal_freshness:.0%}",
                f"{return_estimate.z_score_vs_history:+.1f}σ",
                f"{projection.daily_vol:.2%}",
            ],
            "Status": [
                "🟢 Normal" if abs(t1_range.get('expected', 100) - 100) < 1 else "🟡 Elevated",
                "🟢 Normal" if (t1_range.get('lower_2σ', 100) - 100) > -2 else "🟡 Caution" if (t1_range.get('lower_2σ', 100) - 100) > -3 else "🔴 High",
                "🟢 Normal" if t5_range.get('range_2σ', 0) < 8 else "🟡 Wide",
                "🟢 Good" if return_estimate.signal_freshness >= 0.7 else "🟡 Moderate" if return_estimate.signal_freshness >= 0.5 else "🔴 Low",
                "🟢 Normal" if abs(return_estimate.z_score_vs_history) < 1.5 else "🟡 Unusual",
                "🟢 Normal" if projection.daily_vol < 0.015 else "🟡 Elevated",
            ],
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, hide_index=True, width='stretch')
        
        # =====================================================================
        # SECTION 5: Factor Risk Contributions (Existing Feature)
        # =====================================================================
        if data.exposure is not None and not data.exposure.empty and factor_cov is not None:
            st.divider()
            st.markdown("### 📊 Factor Risk Contributions")
            
            factor_names = data.exposure.columns.tolist()
            asset_factor_exp = pd.DataFrame(
                np.random.randn(len(w_last), len(factor_names)) * 0.1,
                index=w_last.index,
                columns=factor_names,
            )
            
            try:
                risk_result = compute_ex_ante_risk(
                    weights=w_last,
                    factor_exposures=asset_factor_exp,
                    factor_covariance=factor_cov,
                )
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Risk (Ann.)", f"{risk_result.total_risk:.2%}")
                col2.metric("Factor Risk", f"{risk_result.factor_risk:.2%}")
                col3.metric("Specific Risk", f"{risk_result.specific_risk:.2%}")
                col4.metric("Diversification Ratio", f"{risk_result.diversification_ratio:.2f}")
                
                if not risk_result.factor_contributions.empty:
                    contrib = risk_result.factor_contributions.sort_values(ascending=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(
                            contrib.to_frame("Contribution").style.format("{:.4f}").background_gradient(cmap="RdYlGn"),
                            height=350,
                            width='stretch',
                        )
                    with col2:
                        if PLOTLY_AVAILABLE:
                            fig = _create_factor_contribution_chart(contrib, title="Factor Risk Contributions", height=350)
                            if fig:
                                st.plotly_chart(fig, width='stretch')
                        else:
                            st.bar_chart(contrib.tail(12))
            except Exception as e:
                st.error(f"Error computing factor contributions: {str(e)}")
        
    except Exception as e:
        st.error(f"Error computing ex-ante risk projections: {str(e)}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())


# Sector color palette - coordinated colors for consistent visualization
SECTOR_COLORS: Dict[str, str] = {
    "Technology": "#9b59b6",      # Purple
    "Healthcare": "#e67e22",       # Orange
    "Financials": "#e74c3c",       # Red
    "Consumer": "#3498db",          # Blue
    "Industrials": "#1abc9c",       # Teal
    "Energy": "#2ecc71",            # Green
    "Materials": "#f39c12",         # Yellow/Amber
    "Utilities": "#95a5a6",         # Gray
    "Real Estate": "#34495e",       # Dark Gray
    "Communication": "#e91e63",     # Pink
    "Consumer Discretionary": "#3498db",  # Blue (alias)
    "Consumer Staples": "#16a085",  # Dark Teal
}


def _get_sector_color(sector_name: str) -> str:
    """Get color for a sector, with fallback for unknown sectors."""
    # Try exact match first
    if sector_name in SECTOR_COLORS:
        return SECTOR_COLORS[sector_name]
    
    # Try case-insensitive match
    sector_lower = sector_name.lower()
    for key, color in SECTOR_COLORS.items():
        if key.lower() == sector_lower:
            return color
    
    # Try partial match (e.g., "Consumer Discretionary" -> "Consumer")
    for key, color in SECTOR_COLORS.items():
        if key.lower() in sector_lower or sector_lower in key.lower():
            return color
    
    # Fallback: use hash-based color for unknown sectors
    import hashlib
    hash_val = int(hashlib.md5(sector_name.encode()).hexdigest()[:8], 16)
    colors_list = list(SECTOR_COLORS.values())
    return colors_list[hash_val % len(colors_list)]


def _create_sector_bar_chart(
    sector_data: pd.Series,
    title: str = "Sector Allocation",
    height: int = 400,
) -> Optional["go.Figure"]:
    """Create a bar chart with coordinated sector colors."""
    from q23.dashboard.components.charts import PLOTLY_AVAILABLE
    
    if not PLOTLY_AVAILABLE:
        return None
    
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    
    # Sort by value for better visualization
    sorted_data = sector_data.sort_values(ascending=True)
    
    # Get colors for each sector
    colors = [_get_sector_color(sector) for sector in sorted_data.index]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sorted_data.values,
        y=sorted_data.index.tolist(),
        orientation="h",
        marker=dict(color=colors),
        text=[f"{val:.2%}" for val in sorted_data.values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Weight: %{x:.2%}<extra></extra>",
    ))
    
    fig.update_layout(
        title={"text": title, "font": {"size": 16, "color": "#FAFAFA"}},
        paper_bgcolor="#0E1117",
        plot_bgcolor="#262730",
        font={"color": "#FAFAFA", "size": 11},
        height=height,
        showlegend=False,
        xaxis={
            "title": "Weight",
            "tickformat": ".0%",
            "gridcolor": "#3A3A3A",
            "zerolinecolor": "#3A3A3A",
        },
        yaxis={
            "title": "",
            "gridcolor": "#3A3A3A",
        },
        margin={"l": 100, "r": 40, "t": 60, "b": 40},
    )
    
    return fig


def _create_sector_timeseries_chart(
    sector_exp: pd.DataFrame,
    height: int = 400,
) -> Optional["go.Figure"]:
    """Create a time series chart with coordinated sector colors."""
    from q23.dashboard.components.charts import PLOTLY_AVAILABLE
    
    if not PLOTLY_AVAILABLE:
        return None
    
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    
    fig = go.Figure()
    
    for sector in sector_exp.columns:
        color = _get_sector_color(sector)
        fig.add_trace(go.Scatter(
            x=sector_exp.index,
            y=sector_exp[sector].values,
            mode="lines",
            name=sector,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{sector}</b><br>%{{x}}<br>Weight: %{{y:.2%}}<extra></extra>",
        ))
    
    fig.update_layout(
        title={"text": "Sector Exposure Over Time", "font": {"size": 16, "color": "#FAFAFA"}},
        paper_bgcolor="#0E1117",
        plot_bgcolor="#262730",
        font={"color": "#FAFAFA", "size": 11},
        height=height,
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#3A3A3A",
            borderwidth=1,
        ),
        xaxis={
            "gridcolor": "#3A3A3A",
            "zerolinecolor": "#3A3A3A",
        },
        yaxis={
            "title": "Weight",
            "tickformat": ".0%",
            "gridcolor": "#3A3A3A",
            "zerolinecolor": "#3A3A3A",
        },
        hovermode="x unified",
        margin={"l": 60, "r": 40, "t": 60, "b": 40},
    )
    
    return fig


def _create_default_sector_map(assets: List[str]) -> Dict[str, str]:
    """Create a default sector mapping for demonstration."""
    sectors = ["Technology", "Healthcare", "Financials", "Consumer", "Industrials", "Energy"]
    return {asset: sectors[i % len(sectors)] for i, asset in enumerate(assets)}


def _compute_sector_health_metrics(
    sector_exp: pd.DataFrame,
    sector_weights: pd.Series,
    portfolio_vol: Optional[float] = None,
    concentration: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compute sector health/risk metrics for PM analysis.
    
    Enhanced scoring system that provides differentiated insights:
    - Momentum: Recent return trend (10-day)
    - Volatility: Risk-adjusted relative to peer sectors
    - Concentration: Position sizing risk
    - Trend: Exposure direction stability
    
    Args:
        sector_exp: Sector exposure time series (time x sector)
        sector_weights: Current sector weights (sector)
        portfolio_vol: Portfolio volatility (annualized)
        concentration: Concentration metrics dict
        
    Returns:
        DataFrame with sector health metrics including differentiated health scores
    """
    from scipy import stats as sp_stats
    
    sectors = sector_weights.index.tolist()
    results = []
    
    # Get portfolio volatility if available
    if portfolio_vol is None or portfolio_vol <= 0:
        if not sector_exp.empty:
            sector_vol = sector_exp.std(axis=0) * np.sqrt(252)
            portfolio_vol = float(sector_vol.mean()) if len(sector_vol) > 0 else 0.15
        else:
            portfolio_vol = 0.15
    
    # Pre-compute cross-sector statistics for relative scoring
    sector_vols = {}
    sector_returns = {}
    sector_momentum = {}
    sector_trend = {}
    
    for sector in sectors:
        if sector in sector_exp.columns and len(sector_exp[sector]) >= 20:
            ts = sector_exp[sector].dropna()
            
            # Volatility (annualized)
            sector_vols[sector] = float(ts.diff().std() * np.sqrt(252)) if len(ts) > 1 else portfolio_vol
            
            # Recent return (last 10 days cumulative)
            if len(ts) >= 10:
                recent_return = (ts.iloc[-1] / ts.iloc[-10]) - 1 if ts.iloc[-10] != 0 else 0
                sector_returns[sector] = float(recent_return)
            else:
                sector_returns[sector] = 0.0
            
            # Momentum (5-day vs 20-day mean)
            if len(ts) >= 20:
                ma5 = ts.tail(5).mean()
                ma20 = ts.tail(20).mean()
                sector_momentum[sector] = float((ma5 / ma20) - 1) if ma20 != 0 else 0.0
            else:
                sector_momentum[sector] = 0.0
            
            # Trend stability (positive days ratio in last 10 days)
            if len(ts) >= 10:
                daily_changes = ts.diff().tail(10)
                positive_days = (daily_changes > 0).sum() / 10
                sector_trend[sector] = float(positive_days)
            else:
                sector_trend[sector] = 0.5
        else:
            sector_vols[sector] = portfolio_vol
            sector_returns[sector] = 0.0
            sector_momentum[sector] = 0.0
            sector_trend[sector] = 0.5
    
    # Compute relative z-scores for differentiation
    vol_values = list(sector_vols.values())
    ret_values = list(sector_returns.values())
    mom_values = list(sector_momentum.values())
    
    vol_mean, vol_std = np.mean(vol_values), np.std(vol_values) + 1e-8
    ret_mean, ret_std = np.mean(ret_values), np.std(ret_values) + 1e-8
    mom_mean, mom_std = np.mean(mom_values), np.std(mom_values) + 1e-8
    
    # Compute sector-level metrics
    for sector in sectors:
        weight = sector_weights.get(sector, 0.0)
        sector_volatility = sector_vols.get(sector, portfolio_vol)
        
        # Exposure-weighted risk contribution
        exposure_weighted_risk = abs(weight) * sector_volatility
        
        # Concentration risk (HHI component)
        concentration_risk = weight ** 2
        
        # Risk-adjusted exposure
        risk_adj_exposure = abs(weight) / (sector_volatility + 1e-6)
        
        # =====================================================================
        # ENHANCED HEALTH SCORE COMPONENTS (differentiated)
        # =====================================================================
        
        # 1. MOMENTUM SCORE (25 points max)
        # Positive momentum = higher score
        mom_z = (sector_momentum.get(sector, 0) - mom_mean) / mom_std
        momentum_score = 12.5 + np.clip(mom_z * 12.5, -12.5, 12.5)  # 0-25 range
        
        # 2. VOLATILITY SCORE (25 points max)
        # Lower volatility relative to peers = higher score
        vol_z = (sector_volatility - vol_mean) / vol_std
        volatility_score = 25 - np.clip(vol_z * 12.5, -12.5, 12.5)  # Lower vol = higher score
        
        # 3. TREND SCORE (20 points max)
        # More positive days = higher score
        trend_value = sector_trend.get(sector, 0.5)
        trend_score = trend_value * 20  # 0-20 range
        
        # 4. RETURN SCORE (20 points max)
        # Higher recent returns = higher score
        ret_z = (sector_returns.get(sector, 0) - ret_mean) / ret_std
        return_score = 10 + np.clip(ret_z * 10, -10, 10)  # 0-20 range
        
        # 5. CONCENTRATION PENALTY (10 points max deduction)
        # Higher concentration = lower score
        total_hhi = sum(sector_weights.get(s, 0) ** 2 for s in sectors)
        relative_concentration = concentration_risk / (total_hhi + 1e-8)
        concentration_penalty = relative_concentration * 10  # 0-10 deduction
        
        # FINAL HEALTH SCORE (0-100)
        # Sum of components minus concentration penalty
        raw_health = momentum_score + volatility_score + trend_score + return_score - concentration_penalty
        health_score = np.clip(raw_health, 0, 100)
        
        # Risk contribution percentage
        total_exposure_weighted_risk = sum(
            abs(sector_weights.get(s, 0.0)) * sector_vols.get(s, portfolio_vol) 
            for s in sectors
        )
        pct_risk_contribution = (
            exposure_weighted_risk / (total_exposure_weighted_risk + 1e-12)
        ) * 100
        
        results.append({
            "sector": sector,
            "weight": float(weight),
            "exposure_weighted_risk": float(exposure_weighted_risk),
            "pct_risk_contribution": float(pct_risk_contribution),
            "sector_volatility": float(sector_volatility),
            "risk_adj_exposure": float(risk_adj_exposure),
            "concentration_risk": float(concentration_risk),
            "health_score": float(health_score),
            # New granular metrics for PM insight
            "momentum_score": float(momentum_score),
            "volatility_score": float(volatility_score),
            "trend_score": float(trend_score),
            "return_score": float(return_score),
            "recent_return": float(sector_returns.get(sector, 0.0)),
            "trend_pct": float(sector_trend.get(sector, 0.5)),
        })
    
    return pd.DataFrame(results).set_index("sector")


def _create_sector_health_chart(
    health_metrics: pd.DataFrame,
    height: int = 400,
) -> Optional["go.Figure"]:
    """Create an enhanced sector health score chart with component breakdown."""
    from q23.dashboard.components.charts import PLOTLY_AVAILABLE
    
    if not PLOTLY_AVAILABLE:
        return None
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None
    
    # Sort by health score
    sorted_metrics = health_metrics.sort_values("health_score", ascending=True)
    
    # Check if we have the enhanced metrics
    has_components = "momentum_score" in sorted_metrics.columns
    
    if has_components:
        # Create stacked bar chart showing health score breakdown
        fig = go.Figure()
        
        sectors = sorted_metrics.index.tolist()
        
        # Component colors
        component_colors = {
            "momentum_score": "#2ecc71",     # Green
            "volatility_score": "#3498db",   # Blue  
            "trend_score": "#9b59b6",        # Purple
            "return_score": "#f39c12",       # Orange
        }
        
        # Add stacked bars for each component
        fig.add_trace(go.Bar(
            y=sectors,
            x=sorted_metrics["momentum_score"].values,
            name="Momentum (25)",
            orientation="h",
            marker_color=component_colors["momentum_score"],
            hovertemplate="<b>%{y}</b><br>Momentum: %{x:.1f}/25<extra></extra>",
        ))
        
        fig.add_trace(go.Bar(
            y=sectors,
            x=sorted_metrics["volatility_score"].values,
            name="Low Vol (25)",
            orientation="h",
            marker_color=component_colors["volatility_score"],
            hovertemplate="<b>%{y}</b><br>Vol Score: %{x:.1f}/25<extra></extra>",
        ))
        
        fig.add_trace(go.Bar(
            y=sectors,
            x=sorted_metrics["trend_score"].values,
            name="Trend (20)",
            orientation="h",
            marker_color=component_colors["trend_score"],
            hovertemplate="<b>%{y}</b><br>Trend: %{x:.1f}/20<extra></extra>",
        ))
        
        fig.add_trace(go.Bar(
            y=sectors,
            x=sorted_metrics["return_score"].values,
            name="Return (20)",
            orientation="h",
            marker_color=component_colors["return_score"],
            hovertemplate="<b>%{y}</b><br>Return: %{x:.1f}/20<extra></extra>",
        ))
        
        # Add total health score as text annotation
        for i, sector in enumerate(sectors):
            score = sorted_metrics.loc[sector, "health_score"]
            fig.add_annotation(
                x=score + 3,
                y=sector,
                text=f"{score:.0f}",
                showarrow=False,
                font=dict(size=12, color="#FAFAFA", family="Arial Black"),
            )
        
        fig.update_layout(
            barmode="stack",
            title={
                "text": "Sector Health Score Breakdown<br><sup>Components: Momentum + Low Vol + Trend + Return (max 100)</sup>",
                "font": {"size": 14, "color": "#FAFAFA"},
            },
            paper_bgcolor="#0E1117",
            plot_bgcolor="#262730",
            font={"color": "#FAFAFA", "size": 11},
            height=height,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0.3)",
            ),
            margin={"l": 100, "r": 60, "t": 100, "b": 40},
            xaxis={
                "title": "Health Score (0-100)",
                "gridcolor": "#3A3A3A",
                "zerolinecolor": "#3A3A3A",
                "range": [0, 110],
            },
            yaxis={
                "gridcolor": "#3A3A3A",
            },
        )
        
        return fig
    
    # Fallback to old chart if no components
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Sector Health Score", "Exposure-Weighted Risk Contribution"),
        shared_yaxes=True,
        horizontal_spacing=0.15,
    )
    
    # Health score bars
    colors_health = [_get_sector_color(sector) for sector in sorted_metrics.index]
    fig.add_trace(
        go.Bar(
            x=sorted_metrics["health_score"].values,
            y=sorted_metrics.index.tolist(),
            orientation="h",
            marker=dict(color=colors_health),
            name="Health Score",
            text=[f"{val:.1f}" for val in sorted_metrics["health_score"].values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Health Score: %{x:.1f}/100<extra></extra>",
        ),
        row=1,
        col=1,
    )
    
    # Risk contribution bars
    colors_risk = [_get_sector_color(sector) for sector in sorted_metrics.index]
    fig.add_trace(
        go.Bar(
            x=sorted_metrics["pct_risk_contribution"].values,
            y=sorted_metrics.index.tolist(),
            orientation="h",
            marker=dict(color=colors_risk),
            name="Risk Contribution",
            text=[f"{val:.1f}%" for val in sorted_metrics["pct_risk_contribution"].values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Risk Contribution: %{x:.1f}%<extra></extra>",
        ),
        row=1,
        col=2,
    )
    
    fig.update_layout(
        title={"text": "Sector Health & Risk Analysis", "font": {"size": 16, "color": "#FAFAFA"}},
        paper_bgcolor="#0E1117",
        plot_bgcolor="#262730",
        font={"color": "#FAFAFA", "size": 11},
        height=height,
        showlegend=False,
        margin={"l": 100, "r": 40, "t": 80, "b": 40},
    )
    
    fig.update_xaxes(
        title_text="Health Score (0-100)",
        gridcolor="#3A3A3A",
        zerolinecolor="#3A3A3A",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Risk Contribution (%)",
        gridcolor="#3A3A3A",
        zerolinecolor="#3A3A3A",
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title="",
        gridcolor="#3A3A3A",
    )
    
    return fig


def _create_sector_risk_heatmap(
    health_metrics: pd.DataFrame,
    height: int = 400,
) -> Optional["go.Figure"]:
    """Create a risk heatmap showing exposure vs risk contribution."""
    from q23.dashboard.components.charts import PLOTLY_AVAILABLE
    
    if not PLOTLY_AVAILABLE:
        return None
    
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    
    # Prepare data for scatter plot with size = risk contribution
    fig = go.Figure()
    
    for sector in health_metrics.index:
        weight = health_metrics.loc[sector, "weight"]
        risk_contrib = health_metrics.loc[sector, "pct_risk_contribution"]
        health_score = health_metrics.loc[sector, "health_score"]
        color = _get_sector_color(sector)
        
        # Size based on risk contribution
        size = max(risk_contrib * 2, 10)
        
        fig.add_trace(go.Scatter(
            x=[abs(weight) * 100],  # Exposure %
            y=[risk_contrib],  # Risk contribution %
            mode="markers+text",
            marker=dict(
                color=color,
                size=size,
                opacity=0.7,
                line=dict(width=2, color="#FAFAFA"),
            ),
            text=[sector],
            textposition="middle center",
            textfont=dict(size=10, color="#FAFAFA"),
            name=sector,
            hovertemplate=(
                f"<b>{sector}</b><br>"
                f"Exposure: {abs(weight):.2%}<br>"
                f"Risk Contribution: {risk_contrib:.1f}%<br>"
                f"Health Score: {health_score:.1f}<extra></extra>"
            ),
        ))
    
    fig.update_layout(
        title={"text": "Sector Risk Map: Exposure vs Risk Contribution", "font": {"size": 16, "color": "#FAFAFA"}},
        paper_bgcolor="#0E1117",
        plot_bgcolor="#262730",
        font={"color": "#FAFAFA", "size": 11},
        height=height,
        xaxis={
            "title": "Sector Exposure (%)",
            "gridcolor": "#3A3A3A",
            "zerolinecolor": "#3A3A3A",
        },
        yaxis={
            "title": "Risk Contribution (%)",
            "gridcolor": "#3A3A3A",
            "zerolinecolor": "#3A3A3A",
        },
        hovermode="closest",
        margin={"l": 80, "r": 40, "t": 60, "b": 60},
    )
    
    return fig


def render_sector_analysis_page(
    data: DashboardData,
    sector_map: Optional[Dict[str, str]] = None,
) -> None:
    """Render the sector/industry analysis page with coordinated color coding."""
    st.subheader("🏢 Sector & Industry Analysis")
    
    if data.weights is None or data.weights.empty:
        st.warning("No weights data available")
        return
    
    # Create mock sector map if not provided
    if sector_map is None:
        st.info("No sector mapping provided. Using mock sectors for demonstration.")
        assets = data.weights.columns.tolist()
        sector_map = _create_default_sector_map(assets)
    
    # Current weights
    w_last = data.weights.iloc[-1]
    
    # Sector exposure time series
    st.markdown("### Sector Exposure Over Time")
    
    try:
        sector_exp = compute_sector_exposure_timeseries(data.weights, sector_map)
        
        if not sector_exp.empty:
            # Use Plotly chart with coordinated colors
            fig_ts = _create_sector_timeseries_chart(sector_exp, height=400)
            if fig_ts is not None:
                st.plotly_chart(fig_ts, width='stretch')
            else:
                # Fallback to streamlit chart
                st.line_chart(sector_exp, height=400)
        
        # Current sector weights
        st.markdown("### Current Sector Allocation")
        
        current_sector = sector_exp.iloc[-1].sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(current_sector.to_frame("Weight").style.format("{:.2%}"), height=300)
        with col2:
            # Use Plotly bar chart with coordinated colors
            fig_bar = _create_sector_bar_chart(current_sector, title="Current Sector Allocation", height=300)
            if fig_bar is not None:
                st.plotly_chart(fig_bar, width='stretch')
            else:
                # Fallback to streamlit chart
                st.bar_chart(current_sector)
        
        # Concentration metrics
        st.markdown("### Sector Concentration Metrics")
        
        concentration = compute_industry_concentration(w_last, sector_map)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("HHI Index", f"{concentration['herfindahl_index']:.4f}")
        col2.metric("Effective Sectors", f"{concentration['effective_sectors']:.1f}")
        col3.metric("Top 3 Concentration", f"{concentration['top3_concentration']:.1%}")
        col4.metric("Active Sectors", concentration['n_sectors'])
        
        if concentration['largest_sector']:
            largest_color = _get_sector_color(concentration['largest_sector'])
            st.markdown(
                f"**Largest Sector:** "
                f"<span style='color: {largest_color}; font-weight: bold;'>{concentration['largest_sector']}</span> "
                f"({concentration['largest_sector_weight']:.1%})"
            )
        
        # Sector Health & Risk Analysis
        st.divider()
        st.markdown("### 🎯 Sector Health & Risk Analysis")
        
        # Get portfolio volatility from diagnostics
        portfolio_vol = None
        if data.diag is not None and not data.diag.empty:
            if "rolling_vol" in data.diag.columns:
                portfolio_vol = float(data.diag["rolling_vol"].iloc[-1])
            elif "port_ret" in data.diag.columns:
                # Compute volatility from returns
                ret_series = data.diag["port_ret"].dropna()
                if len(ret_series) > 20:
                    portfolio_vol = float(ret_series.std() * np.sqrt(252))
        
        # Compute sector health metrics
        health_metrics = _compute_sector_health_metrics(
            sector_exp=sector_exp,
            sector_weights=current_sector,
            portfolio_vol=portfolio_vol,
            concentration=concentration,
        )
        
        if not health_metrics.empty:
            # Check if we have enhanced metrics
            has_enhanced = "momentum_score" in health_metrics.columns
            
            # Display metrics table
            col1, col2 = st.columns([1.2, 1])
            
            with col1:
                st.markdown("**Sector Health Metrics**")
                
                # Select columns to display
                if has_enhanced:
                    display_cols = ["weight", "health_score", "recent_return", "trend_pct", 
                                   "sector_volatility", "pct_risk_contribution"]
                    col_names = {
                        "weight": "Weight",
                        "health_score": "Health",
                        "recent_return": "10d Ret",
                        "trend_pct": "Trend %",
                        "sector_volatility": "Vol",
                        "pct_risk_contribution": "Risk %",
                    }
                else:
                    display_cols = ["weight", "health_score", "sector_volatility", "pct_risk_contribution"]
                    col_names = {
                        "weight": "Weight",
                        "health_score": "Health",
                        "sector_volatility": "Vol",
                        "pct_risk_contribution": "Risk %",
                    }
                
                display_metrics = health_metrics[display_cols].copy()
                display_metrics.columns = [col_names.get(c, c) for c in display_cols]
                
                # Format columns
                display_metrics["Weight"] = display_metrics["Weight"].apply(lambda x: f"{x:.1%}")
                display_metrics["Health"] = display_metrics["Health"].apply(lambda x: f"{x:.0f}")
                display_metrics["Vol"] = display_metrics["Vol"].apply(lambda x: f"{x:.1%}")
                display_metrics["Risk %"] = display_metrics["Risk %"].apply(lambda x: f"{x:.1f}%")
                if "10d Ret" in display_metrics.columns:
                    display_metrics["10d Ret"] = display_metrics["10d Ret"].apply(lambda x: f"{x:+.1%}")
                if "Trend %" in display_metrics.columns:
                    display_metrics["Trend %"] = display_metrics["Trend %"].apply(lambda x: f"{x:.0%}")
                
                # Color-code health scores
                def color_health_score(val):
                    try:
                        score = float(str(val).replace("%", ""))
                        if score >= 65:
                            return "background-color: rgba(46, 204, 113, 0.3);"  # Green
                        elif score >= 45:
                            return "background-color: rgba(243, 156, 18, 0.3);"  # Orange
                        else:
                            return "background-color: rgba(231, 76, 60, 0.3);"  # Red
                    except:
                        return ""
                
                st.dataframe(
                    display_metrics.style.applymap(
                        color_health_score,
                        subset=["Health"]
                    ),
                    height=380,
                    width='stretch',
                )
            
            with col2:
                st.markdown("**🔍 Actionable Insights**")
                
                # Generate differentiated insights based on scores
                insights = []
                
                # Best momentum sectors
                if has_enhanced:
                    best_momentum = health_metrics.nlargest(1, "momentum_score").index[0]
                    worst_momentum = health_metrics.nsmallest(1, "momentum_score").index[0]
                    
                    mom_diff = health_metrics.loc[best_momentum, "momentum_score"] - health_metrics.loc[worst_momentum, "momentum_score"]
                    if mom_diff > 5:
                        insights.append({
                            "type": "momentum",
                            "icon": "🚀",
                            "text": f"**{best_momentum}** has strongest momentum (+{health_metrics.loc[best_momentum, 'recent_return']:.1%} 10d)",
                            "action": f"Consider increasing {best_momentum} exposure",
                        })
                        insights.append({
                            "type": "momentum_weak",
                            "icon": "📉",
                            "text": f"**{worst_momentum}** momentum lagging ({health_metrics.loc[worst_momentum, 'recent_return']:+.1%} 10d)",
                            "action": f"Review {worst_momentum} position for potential trim",
                        })
                    
                    # Trend analysis
                    strong_trend = health_metrics[health_metrics["trend_pct"] >= 0.7]
                    weak_trend = health_metrics[health_metrics["trend_pct"] <= 0.3]
                    
                    if len(strong_trend) > 0:
                        best_trend = strong_trend.index[0]
                        insights.append({
                            "type": "trend",
                            "icon": "📈",
                            "text": f"**{best_trend}** in strong uptrend ({health_metrics.loc[best_trend, 'trend_pct']:.0%} positive days)",
                            "action": "Trend following opportunity",
                        })
                    
                    if len(weak_trend) > 0:
                        worst_trend = weak_trend.index[0]
                        insights.append({
                            "type": "trend_weak",
                            "icon": "⚠️",
                            "text": f"**{worst_trend}** in downtrend ({health_metrics.loc[worst_trend, 'trend_pct']:.0%} positive days)",
                            "action": "Consider hedging or reducing",
                        })
                
                # Volatility insights
                high_vol = health_metrics.nlargest(1, "sector_volatility").index[0]
                low_vol = health_metrics.nsmallest(1, "sector_volatility").index[0]
                vol_spread = health_metrics.loc[high_vol, "sector_volatility"] - health_metrics.loc[low_vol, "sector_volatility"]
                
                if vol_spread > 0.05:  # 5% vol spread
                    insights.append({
                        "type": "volatility",
                        "icon": "⚡",
                        "text": f"**{high_vol}** vol ({health_metrics.loc[high_vol, 'sector_volatility']:.1%}) >> **{low_vol}** ({health_metrics.loc[low_vol, 'sector_volatility']:.1%})",
                        "action": "Vol spread may offer pairs opportunity",
                    })
                
                # Risk concentration
                top_risk = health_metrics.nlargest(1, "pct_risk_contribution").index[0]
                risk_pct = health_metrics.loc[top_risk, "pct_risk_contribution"]
                if risk_pct > 30:
                    insights.append({
                        "type": "concentration",
                        "icon": "🎯",
                        "text": f"**{top_risk}** contributes {risk_pct:.0f}% of total risk",
                        "action": "Consider diversifying risk contribution",
                    })
                
                # Display insights
                if insights:
                    for insight in insights[:5]:  # Limit to top 5
                        st.markdown(f"{insight['icon']} {insight['text']}")
                        st.caption(f"→ {insight['action']}")
                        st.markdown("")
                else:
                    st.info("No significant sector divergences detected")
                
                st.divider()
                
                # Quick summary stats
                st.markdown("**📊 Summary**")
                col_a, col_b = st.columns(2)
                with col_a:
                    healthiest = health_metrics.nlargest(1, "health_score")
                    st.metric("Healthiest", healthiest.index[0], f"{healthiest['health_score'].values[0]:.0f}")
                with col_b:
                    weakest = health_metrics.nsmallest(1, "health_score")
                    st.metric("Weakest", weakest.index[0], f"{weakest['health_score'].values[0]:.0f}")
            
            # Sector Health & Risk Charts
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Sector Health & Risk Overview**")
                fig_health = _create_sector_health_chart(health_metrics, height=400)
                if fig_health is not None:
                    st.plotly_chart(fig_health, width='stretch')
                else:
                    st.bar_chart(health_metrics[["health_score", "pct_risk_contribution"]])
            
            with col2:
                st.markdown("**Exposure vs Risk Contribution**")
                fig_risk_map = _create_sector_risk_heatmap(health_metrics, height=400)
                if fig_risk_map is not None:
                    st.plotly_chart(fig_risk_map, width='stretch')
                else:
                    # Fallback scatter
                    st.scatter_chart(
                        health_metrics[["weight", "pct_risk_contribution"]],
                        x="weight",
                        y="pct_risk_contribution",
                    )
        
    except Exception as e:
        st.error(f"Error in sector analysis: {str(e)}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
    
    # Note: Sector performance attribution requires asset-level returns
    # which are not available in DashboardData. This feature would need
    # asset returns to be stored during strategy execution.


def render_correlation_analysis_page(data: DashboardData):
    """Render the correlation analysis page."""
    st.subheader("🔗 Correlation Analysis")
    
    # Help expander
    with st.expander("ℹ️ **Understanding Correlation Analysis**", expanded=False):
        st.markdown("""
        **Correlation Analysis** helps you understand:
        
        - **Factor Correlations**: How your portfolio moves with different factors
        - **Diversification**: Low correlations indicate better diversification
        - **Up/Down Capture**: How you perform in up vs down markets
        
        **Interpreting Results:**
        - High positive correlation → Portfolio moves with factor
        - High negative correlation → Portfolio moves opposite to factor
        - Near-zero correlation → Independent of factor
        """)
    
    ret_series = select_return_series(data.diag) if data.diag is not None else None
    
    if ret_series is None or ret_series.empty:
        st.warning("No portfolio return series available")
        return
    
    if data.exposure is None or data.exposure.empty:
        st.warning("No factor exposure data available")
        return
    
    # Use factor exposure changes as proxy for factor returns
    factor_returns = data.exposure.diff().fillna(0.0)
    
    window = st.slider(
        "Correlation Window (days)", 
        21, 252, 63, 21,
        help="Rolling window for correlation calculations"
    )
    
    try:
        corr_analysis = compute_correlation_analysis(
            portfolio_returns=ret_series,
            factor_returns=factor_returns,
            window=window,
        )
        
        if not corr_analysis.empty:
            st.markdown("### 📈 Rolling Correlations with Factors")
            
            # Select top factors to display
            top_factors = data.exposure.abs().mean().nlargest(8).index.tolist()
            top_corr_cols = [f"corr_{f}" for f in top_factors if f"corr_{f}" in corr_analysis.columns]
            
            if top_corr_cols:
                if COMPONENTS_AVAILABLE and PLOTLY_AVAILABLE:
                    display_df = corr_analysis[top_corr_cols].copy()
                    display_df.columns = [c.replace("corr_", "") for c in display_df.columns]
                    fig = create_time_series_chart(
                        display_df,
                        title="Rolling Factor Correlations",
                        y_format=".2f",
                        height=400,
                    )
                    if fig:
                        st.plotly_chart(fig)
                else:
                    st.line_chart(corr_analysis[top_corr_cols], height=400)
            
            # Current correlations
            st.markdown("### 🎯 Current Factor Correlations")
            
            current_corr = corr_analysis.iloc[-1].sort_values(ascending=False)
            current_corr.index = [c.replace("corr_", "") for c in current_corr.index]
            
            col1, col2 = st.columns([1, 1.5])
            with col1:
                st.dataframe(
                    current_corr.to_frame("Correlation").style.format("{:.3f}").background_gradient(
                        cmap="RdYlGn", vmin=-1, vmax=1
                    ), 
                    height=400,
                )
            with col2:
                if COMPONENTS_AVAILABLE and PLOTLY_AVAILABLE:
                    fig = create_risk_contribution_chart(
                        current_corr,
                        title="Current Factor Correlations",
                        height=400,
                    )
                    if fig:
                        st.plotly_chart(fig)
                else:
                    st.bar_chart(current_corr)
        else:
            st.info("Not enough data for correlation analysis")
            
    except Exception as e:
        st.error(f"Error in correlation analysis: {str(e)}")
    
    # Factor correlation matrix
    st.divider()
    st.markdown("### 🔲 Factor Correlation Matrix")
    
    try:
        factor_corr = compute_correlation_matrix(factor_returns, window=252)
        
        if not factor_corr.empty:
            if COMPONENTS_AVAILABLE and PLOTLY_AVAILABLE:
                fig = create_correlation_heatmap(
                    factor_corr,
                    title="Factor Correlation Matrix",
                    height=500,
                )
                if fig:
                    st.plotly_chart(fig)
            else:
                st.dataframe(
                    factor_corr.style.format("{:.2f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                    height=500,
                )
    except Exception as e:
        st.error(f"Error computing correlation matrix: {str(e)}")
    
    # Up/Down capture
    st.divider()
    st.markdown("### 📊 Up/Down Capture Analysis")
    
    if data.diag is not None and "mkt_ret" in data.diag.columns:
        try:
            mkt_ret = data.diag["mkt_ret"]
            
            capture = compute_up_down_capture(ret_series, mkt_ret)
            
            # Explanation
            st.markdown("""
            <div style="background-color: #262730; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <b>Capture Ratios:</b> Up Capture > 1 = outperform in up markets. Down Capture < 1 = protect in down markets.
            <br><b>Ideal:</b> High up capture, low down capture (Capture Ratio > 1)
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Color code based on desirability
            up_delta = "good" if capture['up_capture'] > 1 else None
            down_delta = "inverse" if capture['down_capture'] < 1 else None
            
            col1.metric(
                "Up Capture", 
                f"{capture['up_capture']:.2f}",
                delta=f"{(capture['up_capture']-1)*100:+.0f}%" if capture['up_capture'] != 1 else None,
                help="Portfolio return / Market return on up days. >1 is good."
            )
            col2.metric(
                "Down Capture", 
                f"{capture['down_capture']:.2f}",
                delta=f"{(1-capture['down_capture'])*100:+.0f}% protection" if capture['down_capture'] < 1 else None,
                help="Portfolio return / Market return on down days. <1 is good."
            )
            col3.metric(
                "Capture Ratio", 
                f"{capture['capture_ratio']:.2f}",
                delta="Asymmetric edge" if capture['capture_ratio'] > 1.2 else None,
                help="Up Capture / Down Capture. >1 indicates positive asymmetry."
            )
            col4.metric(
                "Batting Average", 
                f"{capture['batting_average']:.1%}",
                help="% of days beating the market"
            )
            
            st.caption(f"📅 Based on {capture['up_days']} up days and {capture['down_days']} down days")
            
        except Exception as e:
            st.error(f"Error in capture analysis: {str(e)}")
    else:
        st.info("Market return data not available for capture analysis")


def render_beta_analysis_page(data: DashboardData):
    """Render the enhanced beta analysis page with comprehensive factor insights."""
    st.subheader("📐 Factor Beta Analysis")
    
    # Enhanced help expander
    with st.expander("ℹ️ **Understanding Factor Betas & Risk**", expanded=False):
        st.markdown("""
        **Factor Betas** measure portfolio sensitivity to factor movements:
        
        | Beta Value | Interpretation | Risk Implication |
        |------------|----------------|------------------|
        | **β = 1** | Moves 1:1 with factor | Neutral exposure |
        | **β > 1** | Amplified exposure | Higher factor risk |
        | **β < 1** | Dampened exposure | Lower factor risk |
        | **β < 0** | Inverse relationship | Natural hedge |
        | **β ≈ 0** | No relationship | Factor neutral |
        
        **Key Factors to Monitor:**
        - **Momentum**: Sensitivity to trend-following strategies
        - **Value**: Exposure to cheap vs expensive stocks
        - **Quality**: Sensitivity to high-quality earnings/ROE
        - **Size**: Small cap vs large cap tilt
        - **Volatility**: Low-vol vs high-vol exposure
        - **Growth**: Growth vs value stock exposure
        
        **PM Actions:**
        - High absolute beta = concentrated factor bet
        - Beta drift = changing factor exposure over time
        - Beta divergence = factors moving independently
        """)
    
    ret_series = select_return_series(data.diag) if data.diag is not None else None
    
    if ret_series is None or ret_series.empty:
        st.warning("No portfolio return series available")
        return
    
    if data.exposure is None or data.exposure.empty:
        st.warning("No factor exposure data available")
        return
    
    factor_returns = data.exposure.diff().fillna(0.0)
    all_factors = factor_returns.columns.tolist()
    
    # =========================================================================
    # Controls
    # =========================================================================
    col1, col2, col3 = st.columns(3)
    
    with col1:
        window = st.slider(
            "Beta Window (days)", 
            21, 504, 126, 21,
            help="Rolling window for beta estimation"
        )
    
    with col2:
        n_factors = st.slider(
            "Top N Factors",
            4, min(20, len(all_factors)), min(12, len(all_factors)),
            help="Number of top factors to display"
        )
    
    with col3:
        beta_view = st.selectbox(
            "View Mode",
            ["All Factors", "Style Factors", "Risk Factors", "Custom"],
            help="Filter factor categories"
        )
    
    # Factor categorization
    style_factors = ["momentum", "value", "quality", "growth", "size"]
    risk_factors = ["volatility", "beta", "var", "cvar", "drawdown"]
    
    try:
        betas = compute_beta_analysis(
            portfolio_returns=ret_series,
            factor_returns=factor_returns,
            window=window,
        )
        
        if betas.empty or len(betas) < window:
            st.info("Not enough data for beta analysis")
            return
        
        # Filter factors based on view
        if beta_view == "Style Factors":
            view_factors = [f for f in betas.columns if any(sf in f.lower() for sf in style_factors)]
        elif beta_view == "Risk Factors":
            view_factors = [f for f in betas.columns if any(rf in f.lower() for rf in risk_factors)]
        else:
            view_factors = betas.columns.tolist()
        
        if not view_factors:
            view_factors = betas.columns.tolist()
        
        # Select top factors by absolute mean beta
        current_betas = betas.iloc[-1]
        top_factors = current_betas.abs().sort_values(ascending=False).head(n_factors).index.tolist()
        top_factors = [f for f in top_factors if f in view_factors]
        
        if not top_factors:
            top_factors = view_factors[:n_factors]
        
        # =====================================================================
        # SECTION 1: Current Factor Betas Summary
        # =====================================================================
        st.markdown("### 🎯 Current Factor Betas")
        
        # Summary metrics
        current_sorted = current_betas.reindex(top_factors).sort_values(ascending=False)
        
        # Key stats
        col1, col2, col3, col4 = st.columns(4)
        
        max_beta_factor = current_sorted.idxmax()
        min_beta_factor = current_sorted.idxmin()
        
        with col1:
            st.metric(
                "Highest Beta",
                f"{current_sorted.max():.2f}",
                delta=max_beta_factor,
            )
        with col2:
            st.metric(
                "Lowest Beta",
                f"{current_sorted.min():.2f}",
                delta=min_beta_factor,
            )
        with col3:
            avg_abs_beta = current_sorted.abs().mean()
            st.metric(
                "Avg |Beta|",
                f"{avg_abs_beta:.2f}",
                delta="High exposure" if avg_abs_beta > 0.5 else "Moderate",
            )
        with col4:
            # Beta concentration (HHI of absolute betas)
            abs_betas = current_sorted.abs()
            if abs_betas.sum() > 0:
                normalized = abs_betas / abs_betas.sum()
                hhi = (normalized ** 2).sum()
            else:
                hhi = 0
            st.metric(
                "Beta Concentration",
                f"{hhi:.2f}",
                delta="Concentrated" if hhi > 0.3 else "Diversified",
            )
        
        # Beta table and chart
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            # Enhanced table with categorization
            beta_df = current_sorted.to_frame("Beta")
            beta_df["Category"] = beta_df.index.map(lambda f: _categorize_factor(f))
            beta_df["Signal"] = beta_df["Beta"].apply(_beta_signal)
            
            st.dataframe(
                beta_df.style.format({"Beta": "{:.3f}"}).background_gradient(
                    subset=["Beta"], cmap="RdYlGn", vmin=-2, vmax=2
                ),
                height=400,
                width='stretch',
            )
        
        with col2:
            # Enhanced beta chart with factor colors
            if PLOTLY_AVAILABLE:
                fig = _create_factor_beta_chart(current_sorted, height=400)
                if fig:
                    st.plotly_chart(fig, width='stretch')
            else:
                st.bar_chart(current_sorted)
        
        # =====================================================================
        # SECTION 2: Rolling Factor Betas
        # =====================================================================
        st.divider()
        st.markdown("### 📈 Rolling Factor Betas Over Time")
        
        if len(betas) > window:
            if PLOTLY_AVAILABLE:
                fig = _create_rolling_beta_chart(betas[top_factors], title=f"Rolling {window}-Day Factor Betas", height=450)
                if fig:
                    st.plotly_chart(fig, width='stretch')
            else:
                st.line_chart(betas[top_factors], height=400)
        
        # =====================================================================
        # SECTION 3: Beta Stability Analysis
        # =====================================================================
        st.divider()
        st.markdown("### 📊 Beta Stability & Drift Analysis")
        
        # Compute beta stability metrics
        stability_metrics = _compute_beta_stability(betas, top_factors, lookback=63)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Factor Beta Statistics**")
            
            # Create stability table
            stability_df = pd.DataFrame(stability_metrics).T
            stability_df.columns = ["Current", "Mean", "Std", "Min", "Max", "Drift", "Stability"]
            
            # Color code stability
            def color_stability(val):
                if isinstance(val, str):
                    if val == "Stable":
                        return "background-color: rgba(46, 204, 113, 0.3);"
                    elif val == "Drifting":
                        return "background-color: rgba(243, 156, 18, 0.3);"
                    else:
                        return "background-color: rgba(231, 76, 60, 0.3);"
                return ""
            
            st.dataframe(
                stability_df.style.format({
                    "Current": "{:.3f}",
                    "Mean": "{:.3f}",
                    "Std": "{:.3f}",
                    "Min": "{:.3f}",
                    "Max": "{:.3f}",
                    "Drift": "{:+.3f}",
                }).applymap(color_stability, subset=["Stability"]),
                height=350,
                width='stretch',
            )
        
        with col2:
            st.markdown("**🔍 Beta Insights**")
            
            insights = _generate_beta_insights(stability_metrics, current_sorted)
            
            for insight in insights[:6]:
                st.markdown(f"{insight['icon']} **{insight['factor']}**: {insight['text']}")
                st.caption(f"→ {insight['action']}")
                st.markdown("")
        
        # =====================================================================
        # SECTION 4: Factor Beta Heatmap (Time x Factor)
        # =====================================================================
        st.divider()
        st.markdown("### 🔥 Factor Beta Heatmap")
        
        st.caption("Beta values over time (darker = higher absolute beta)")
        
        # Sample betas for heatmap (monthly)
        if len(betas) > 21:
            # Resample to weekly for cleaner visualization
            weekly_betas = betas[top_factors].resample('W').last().dropna()
            
            if len(weekly_betas) > 4 and PLOTLY_AVAILABLE:
                fig = _create_beta_heatmap(weekly_betas.tail(52), title="Weekly Factor Betas (Last Year)")
                if fig:
                    st.plotly_chart(fig, width='stretch')
            else:
                st.dataframe(
                    weekly_betas.tail(26).style.format("{:.2f}").background_gradient(
                        cmap="RdYlGn", vmin=-2, vmax=2, axis=None
                    ),
                    height=300,
                )
        
        # =====================================================================
        # SECTION 5: PM Action Items
        # =====================================================================
        st.divider()
        st.markdown("### ⚡ PM Action Items")
        
        actions = []
        
        # High beta factors
        high_beta = current_sorted[current_sorted.abs() > 1.0]
        for factor, beta in high_beta.items():
            actions.append({
                "priority": "High" if abs(beta) > 1.5 else "Medium",
                "factor": factor,
                "issue": f"High beta ({beta:.2f})",
                "action": f"Review {factor} exposure - {'long' if beta > 0 else 'short'} bias may increase risk",
            })
        
        # Unstable betas
        for factor, metrics in stability_metrics.items():
            if metrics["Stability"] == "Unstable":
                actions.append({
                    "priority": "Medium",
                    "factor": factor,
                    "issue": f"Unstable beta (std={metrics['Std']:.2f})",
                    "action": f"Monitor {factor} - beta is highly variable",
                })
        
        # Drifting betas
        for factor, metrics in stability_metrics.items():
            if abs(metrics["Drift"]) > 0.3:
                direction = "increasing" if metrics["Drift"] > 0 else "decreasing"
                actions.append({
                    "priority": "Low",
                    "factor": factor,
                    "issue": f"Beta drift ({direction}: {metrics['Drift']:+.2f})",
                    "action": f"Factor exposure is {direction} - verify intentional",
                })
        
        if actions:
            # Sort by priority
            priority_order = {"High": 0, "Medium": 1, "Low": 2}
            actions.sort(key=lambda x: priority_order.get(x["priority"], 3))
            
            for action in actions[:8]:
                priority_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[action["priority"]]
                st.markdown(f"{priority_color} **{action['factor']}** - {action['issue']}")
                st.caption(f"→ {action['action']}")
        else:
            st.success("✅ No significant beta concerns. Factor exposures within normal ranges.")
            
    except Exception as e:
        st.error(f"Error in beta analysis: {str(e)}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())


def _categorize_factor(factor_name: str) -> str:
    """Categorize a factor by name."""
    factor_lower = factor_name.lower()
    
    style_keywords = ["momentum", "value", "quality", "growth", "size"]
    risk_keywords = ["vol", "beta", "var", "risk", "drawdown"]
    macro_keywords = ["rate", "inflation", "gdp", "yield", "spread"]
    
    for kw in style_keywords:
        if kw in factor_lower:
            return "Style"
    for kw in risk_keywords:
        if kw in factor_lower:
            return "Risk"
    for kw in macro_keywords:
        if kw in factor_lower:
            return "Macro"
    
    return "Other"


def _beta_signal(beta: float) -> str:
    """Generate a signal indicator for beta value."""
    if beta > 1.5:
        return "🔴 Very High"
    elif beta > 0.5:
        return "🟡 High"
    elif beta > 0:
        return "🟢 Long"
    elif beta > -0.5:
        return "🔵 Short"
    elif beta > -1.5:
        return "🟣 High Short"
    else:
        return "⚫ Very Short"


def _compute_beta_stability(betas: pd.DataFrame, factors: List[str], lookback: int = 63) -> Dict:
    """Compute beta stability metrics for each factor."""
    results = {}
    
    for factor in factors:
        if factor not in betas.columns:
            continue
        
        beta_series = betas[factor].dropna()
        if len(beta_series) < lookback:
            continue
        
        recent = beta_series.tail(lookback)
        current = float(beta_series.iloc[-1])
        mean = float(recent.mean())
        std = float(recent.std())
        min_val = float(recent.min())
        max_val = float(recent.max())
        
        # Drift (change in mean over time)
        if len(beta_series) >= lookback * 2:
            old_mean = beta_series.iloc[-lookback*2:-lookback].mean()
            drift = mean - old_mean
        else:
            drift = 0.0
        
        # Stability classification
        if std < 0.2:
            stability = "Stable"
        elif std < 0.5:
            stability = "Drifting"
        else:
            stability = "Unstable"
        
        results[factor] = {
            "Current": current,
            "Mean": mean,
            "Std": std,
            "Min": min_val,
            "Max": max_val,
            "Drift": drift,
            "Stability": stability,
        }
    
    return results


def _generate_beta_insights(stability_metrics: Dict, current_betas: pd.Series) -> List[Dict]:
    """Generate PM insights from beta analysis."""
    insights = []
    
    # Sort by absolute current beta
    sorted_factors = current_betas.abs().sort_values(ascending=False)
    
    for factor in sorted_factors.index[:10]:
        if factor not in stability_metrics:
            continue
        
        metrics = stability_metrics[factor]
        beta = current_betas[factor]
        
        # High beta insight
        if abs(beta) > 1.0:
            insights.append({
                "icon": "🎯",
                "factor": factor,
                "text": f"High exposure (β={beta:.2f})",
                "action": f"This {'amplifies' if abs(beta) > 1 else 'matches'} {factor} moves",
            })
        
        # Drifting beta insight
        if abs(metrics["Drift"]) > 0.3:
            direction = "↑" if metrics["Drift"] > 0 else "↓"
            insights.append({
                "icon": "📈" if metrics["Drift"] > 0 else "📉",
                "factor": factor,
                "text": f"Beta drift {direction} ({metrics['Drift']:+.2f})",
                "action": "Factor loading has shifted recently",
            })
        
        # Unstable beta insight
        if metrics["Stability"] == "Unstable":
            insights.append({
                "icon": "⚠️",
                "factor": factor,
                "text": f"High beta volatility (σ={metrics['Std']:.2f})",
                "action": "Exposure to this factor is variable",
            })
        
        # Near zero beta (potential opportunity)
        if abs(beta) < 0.1 and metrics["Stability"] == "Stable":
            insights.append({
                "icon": "🔵",
                "factor": factor,
                "text": f"Factor neutral (β≈0)",
                "action": "No systematic exposure to this factor",
            })
    
    return insights


def _create_factor_beta_chart(betas: pd.Series, height: int = 400) -> Optional["go.Figure"]:
    """Create an enhanced factor beta bar chart."""
    if not PLOTLY_AVAILABLE:
        return None
    
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    
    # Sort by value for better visualization
    sorted_betas = betas.sort_values(ascending=True)
    
    # Get colors
    colors = [_get_factor_color(f) for f in sorted_betas.index]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sorted_betas.values,
        y=sorted_betas.index.tolist(),
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="#FAFAFA", width=1),
        ),
        text=[f"{v:.2f}" for v in sorted_betas.values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Beta: %{x:.3f}<extra></extra>",
    ))
    
    # Add zero line
    fig.add_vline(x=0, line_dash="solid", line_color="#95a5a6", line_width=2)
    
    # Add ±1 reference lines
    fig.add_vline(x=1, line_dash="dash", line_color="#2ecc71", line_width=1)
    fig.add_vline(x=-1, line_dash="dash", line_color="#e74c3c", line_width=1)
    
    fig.update_layout(
        title={"text": "Current Factor Betas<br><sup>Green/red dashed lines at β=±1</sup>", "font": {"size": 14, "color": "#FAFAFA"}},
        paper_bgcolor="#0E1117",
        plot_bgcolor="#262730",
        font={"color": "#FAFAFA", "size": 11},
        height=height,
        showlegend=False,
        xaxis={
            "title": "Beta",
            "gridcolor": "#3A3A3A",
            "zerolinecolor": "#95a5a6",
            "zerolinewidth": 2,
        },
        yaxis={
            "gridcolor": "#3A3A3A",
        },
        margin={"l": 120, "r": 60, "t": 60, "b": 40},
    )
    
    return fig


def _create_rolling_beta_chart(betas: pd.DataFrame, title: str = "Rolling Factor Betas", height: int = 450) -> Optional["go.Figure"]:
    """Create rolling beta time series chart."""
    if not PLOTLY_AVAILABLE:
        return None
    
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    
    fig = go.Figure()
    
    for col in betas.columns:
        color = _get_factor_color(col)
        fig.add_trace(go.Scatter(
            x=betas.index,
            y=betas[col].values,
            mode="lines",
            name=col,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{col}</b><br>Date: %{{x}}<br>Beta: %{{y:.3f}}<extra></extra>",
        ))
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="solid", line_color="#95a5a6", line_width=1)
    fig.add_hline(y=1, line_dash="dot", line_color="#2ecc71", line_width=1)
    fig.add_hline(y=-1, line_dash="dot", line_color="#e74c3c", line_width=1)
    
    fig.update_layout(
        title={"text": title, "font": {"size": 14, "color": "#FAFAFA"}},
        paper_bgcolor="#0E1117",
        plot_bgcolor="#262730",
        font={"color": "#FAFAFA", "size": 11},
        height=height,
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#3A3A3A",
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis={
            "gridcolor": "#3A3A3A",
            "zerolinecolor": "#3A3A3A",
        },
        yaxis={
            "title": "Beta",
            "gridcolor": "#3A3A3A",
            "zerolinecolor": "#95a5a6",
        },
        hovermode="x unified",
        margin={"l": 60, "r": 40, "t": 80, "b": 40},
    )
    
    return fig


def _create_beta_heatmap(betas: pd.DataFrame, title: str = "Factor Beta Heatmap") -> Optional["go.Figure"]:
    """Create a heatmap of factor betas over time."""
    if not PLOTLY_AVAILABLE:
        return None
    
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    
    # Transpose for time on x-axis, factors on y-axis
    data = betas.T
    
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=[d.strftime("%Y-%m-%d") for d in data.columns],
        y=data.index.tolist(),
        colorscale=[
            [0, "#e74c3c"],      # Red for negative
            [0.5, "#f5f5f5"],    # White for zero
            [1, "#2ecc71"],      # Green for positive
        ],
        zmid=0,
        colorbar=dict(
            title=dict(text="Beta", side="right"),
        ),
        hovertemplate="<b>%{y}</b><br>Date: %{x}<br>Beta: %{z:.2f}<extra></extra>",
    ))
    
    fig.update_layout(
        title={"text": title, "font": {"size": 14, "color": "#FAFAFA"}},
        paper_bgcolor="#0E1117",
        plot_bgcolor="#262730",
        font={"color": "#FAFAFA", "size": 11},
        height=400,
        xaxis={
            "title": "Date",
            "tickangle": 45,
        },
        yaxis={
            "title": "Factor",
        },
        margin={"l": 120, "r": 40, "t": 60, "b": 80},
    )
    
    return fig


def render_pm_executive_summary(data: DashboardData, perf: Optional[Dict] = None):
    """
    Render executive summary with PM insights and alerts.
    
    This is the main dashboard for PMs to quickly assess portfolio health.
    """
    st.subheader("📋 Executive Summary & PM Alerts")
    
    st.markdown("""
    Quick overview of portfolio health with actionable insights and alerts.
    """)
    
    # Get return series for analysis
    ret_series = select_return_series(data.diag) if data.diag is not None else None
    
    # Run insight engine
    if COMPONENTS_AVAILABLE:
        insight_engine = PMInsightEngine()
        insights = insight_engine.analyze_portfolio(
            weights=data.weights,
            returns=ret_series,
            diag=data.diag,
            exposure=data.exposure,
        )
        
        # Summary counts
        summary = insight_engine.get_summary()
        
        st.markdown("### 🚦 Alert Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if summary["critical"] > 0:
                st.error(f"🚨 **{summary['critical']}** Critical")
            else:
                st.success("✅ No Critical")
        
        with col2:
            if summary["alert"] > 0:
                st.warning(f"⚠️ **{summary['alert']}** Alerts")
            else:
                st.success("✅ No Alerts")
        
        with col3:
            st.info(f"📋 **{summary['warning']}** Warnings")
        
        with col4:
            st.info(f"ℹ️ **{summary['info']}** Info")
        
        st.divider()
        
        # Render insights panel
        st.markdown("### 📊 Portfolio Health Assessment")
        render_insights_panel(insights, st)
        
    else:
        st.info("Install plotly for enhanced visualizations: `pip install plotly`")
    
    # Quick metrics
    st.divider()
    st.markdown("### 📈 Key Metrics at a Glance")
    
    if perf:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Annual Return", f"{perf.get('annual_return', 0):.2%}")
        col2.metric("Sharpe Ratio", f"{perf.get('sharpe', 0):.2f}")
        col3.metric("Max Drawdown", f"{perf.get('max_drawdown', 0):.2%}")
        col4.metric("Win Rate", f"{perf.get('win_rate', 0):.1%}")
        col5.metric("Positions", f"{perf.get('n_positions', 0):.0f}")
    
    elif data.diag is not None and ret_series is not None:
        # Calculate basic metrics
        ann_ret = ret_series.mean() * 252
        ann_vol = ret_series.std() * np.sqrt(252)
        sharpe = ann_ret / (ann_vol + 1e-12)
        
        equity = (1 + ret_series).cumprod()
        max_dd = (equity / equity.cummax() - 1).min()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Annual Return", f"{ann_ret:.2%}")
        col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col3.metric("Volatility", f"{ann_vol:.2%}")
        col4.metric("Max Drawdown", f"{max_dd:.2%}")
    
    # Recent performance
    if ret_series is not None and len(ret_series) >= 5:
        st.divider()
        st.markdown("### 📅 Recent Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        last_1d = ret_series.iloc[-1] if len(ret_series) >= 1 else 0
        last_5d = ret_series.tail(5).sum() if len(ret_series) >= 5 else 0
        last_21d = ret_series.tail(21).sum() if len(ret_series) >= 21 else 0
        ytd = ret_series.sum()
        
        col1.metric("Today", f"{last_1d:.2%}")
        col2.metric("5-Day", f"{last_5d:.2%}")
        col3.metric("21-Day", f"{last_21d:.2%}")
        col4.metric("Period Total", f"{ytd:.2%}")
    
    # Quick links to other pages
    st.divider()
    st.markdown("### 🔗 Quick Navigation")
    st.markdown("""
    **Deep Dive Analytics:**
    - **Risk Attribution** → Understand factor vs specific risk
    - **Brinson Attribution** → Allocation vs selection effects
    - **Correlation Analysis** → Factor correlations and capture ratios
    - **Sector Analysis** → Sector exposures and concentration
    """)


# =============================================================================
# PM RISK DASHBOARD ENHANCEMENTS
# =============================================================================

def _compute_var_cvar(
    returns: pd.Series,
    confidence_levels: List[float] = [0.95, 0.99],
) -> Dict[str, float]:
    """
    Compute Value at Risk (VaR) and Conditional VaR (CVaR/Expected Shortfall).
    
    Args:
        returns: Return series
        confidence_levels: List of confidence levels (e.g., [0.95, 0.99])
        
    Returns:
        Dictionary with VaR and CVaR metrics
    """
    if returns is None or len(returns) < 20:
        return {}
    
    results = {}
    for conf in confidence_levels:
        alpha = 1 - conf
        var = np.percentile(returns, alpha * 100)
        cvar = returns[returns <= var].mean()
        
        results[f"var_{int(conf*100)}"] = float(var)
        results[f"cvar_{int(conf*100)}"] = float(cvar)
    
    # Annualized VaR
    results["var_95_annual"] = float(results.get("var_95", 0) * np.sqrt(252))
    results["var_99_annual"] = float(results.get("var_99", 0) * np.sqrt(252))
    
    return results


def _compute_risk_budget_utilization(
    current_vol: float,
    target_vol: float = 0.15,
    max_vol: float = 0.25,
    current_drawdown: float = 0.0,
    max_drawdown_limit: float = -0.20,
) -> Dict[str, float]:
    """
    Compute risk budget utilization metrics.
    
    Args:
        current_vol: Current annualized volatility
        target_vol: Target volatility
        max_vol: Maximum allowed volatility
        current_drawdown: Current drawdown (negative)
        max_drawdown_limit: Maximum allowed drawdown
        
    Returns:
        Dictionary with risk budget metrics
    """
    vol_utilization = current_vol / target_vol if target_vol > 0 else 0.0
    vol_headroom = (max_vol - current_vol) / max_vol if max_vol > 0 else 0.0
    
    dd_utilization = abs(current_drawdown) / abs(max_drawdown_limit) if max_drawdown_limit != 0 else 0.0
    dd_headroom = (abs(max_drawdown_limit) - abs(current_drawdown)) / abs(max_drawdown_limit)
    
    # Overall risk budget (weighted average)
    overall_utilization = 0.6 * vol_utilization + 0.4 * dd_utilization
    
    return {
        "vol_utilization": float(vol_utilization),
        "vol_headroom": float(vol_headroom),
        "dd_utilization": float(dd_utilization),
        "dd_headroom": float(max(dd_headroom, 0)),
        "overall_utilization": float(overall_utilization),
        "risk_budget_remaining": float(max(1 - overall_utilization, 0)),
    }


def _generate_brinson_insights(
    brinson_result: "BrinsonResult",
) -> List[Dict[str, str]]:
    """
    Generate actionable insights from Brinson attribution results.
    
    Args:
        brinson_result: BrinsonResult from brinson_attribution()
        
    Returns:
        List of insight dictionaries with title, message, and recommendation
    """
    insights = []
    
    total_excess = brinson_result.total_excess
    allocation = brinson_result.total_allocation
    selection = brinson_result.total_selection
    interaction = brinson_result.total_interaction
    
    # Overall performance assessment
    if total_excess > 0.05:
        insights.append({
            "title": "✅ Strong Active Performance",
            "message": f"Portfolio outperformed benchmark by {total_excess:.2%}",
            "severity": "success",
            "recommendation": "Continue current strategy approach.",
        })
    elif total_excess < -0.05:
        insights.append({
            "title": "⚠️ Underperformance Alert",
            "message": f"Portfolio underperformed by {abs(total_excess):.2%}",
            "severity": "warning",
            "recommendation": "Review both sector allocation and stock selection decisions.",
        })
    
    # Allocation effect analysis
    if allocation > 0.02:
        insights.append({
            "title": "📊 Strong Allocation Skill",
            "message": f"Sector allocation contributed +{allocation:.2%} to returns",
            "severity": "info",
            "recommendation": "Sector tilts are working. Consider maintaining current sector views.",
        })
    elif allocation < -0.02:
        insights.append({
            "title": "📉 Allocation Drag",
            "message": f"Sector allocation detracted {allocation:.2%} from returns",
            "severity": "warning",
            "recommendation": "Review sector overweights/underweights vs market trends.",
        })
    
    # Selection effect analysis
    if selection > 0.02:
        insights.append({
            "title": "🎯 Strong Stock Picking",
            "message": f"Stock selection contributed +{selection:.2%} to returns",
            "severity": "info",
            "recommendation": "Alpha generation is effective. Review top contributors.",
        })
    elif selection < -0.02:
        insights.append({
            "title": "🔍 Selection Weakness",
            "message": f"Stock selection detracted {selection:.2%} from returns",
            "severity": "warning",
            "recommendation": "Review factor signals and position sizing within sectors.",
        })
    
    # Interaction effect
    if abs(interaction) > 0.01:
        direction = "positive" if interaction > 0 else "negative"
        insights.append({
            "title": "🔄 Interaction Effect",
            "message": f"Allocation × Selection interaction was {direction}: {interaction:.2%}",
            "severity": "info",
            "recommendation": "High interaction suggests concentrated bets in tilted sectors.",
        })
    
    # Dominant driver
    effects = {"Allocation": allocation, "Selection": selection, "Interaction": interaction}
    dominant = max(effects.items(), key=lambda x: abs(x[1]))
    
    if abs(dominant[1]) > 0.01:
        insights.append({
            "title": "🏆 Dominant Performance Driver",
            "message": f"{dominant[0]} effect ({dominant[1]:+.2%}) is the primary return driver",
            "severity": "info",
            "recommendation": f"Focus review on {dominant[0].lower()} decisions.",
        })
    
    return insights


def _create_risk_budget_gauge(
    utilization: float,
    title: str = "Risk Budget",
    height: int = 200,
) -> Optional["go.Figure"]:
    """Create a risk budget utilization gauge."""
    if not PLOTLY_AVAILABLE:
        return None
    
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    
    # Color based on utilization
    if utilization < 0.6:
        color = "#2ecc71"  # Green
    elif utilization < 0.8:
        color = "#f39c12"  # Orange
    elif utilization < 1.0:
        color = "#e74c3c"  # Red
    else:
        color = "#c0392b"  # Dark red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=utilization * 100,
        number={"suffix": "%", "font": {"size": 28, "color": "#FAFAFA"}},
        title={"text": title, "font": {"size": 14, "color": "#FAFAFA"}},
        gauge={
            "axis": {"range": [0, 120], "ticksuffix": "%", "tickcolor": "#FAFAFA"},
            "bar": {"color": color},
            "bgcolor": "#262730",
            "borderwidth": 2,
            "bordercolor": "#3A3A3A",
            "steps": [
                {"range": [0, 60], "color": "rgba(46, 204, 113, 0.2)"},
                {"range": [60, 80], "color": "rgba(243, 156, 18, 0.2)"},
                {"range": [80, 100], "color": "rgba(231, 76, 60, 0.2)"},
                {"range": [100, 120], "color": "rgba(192, 57, 43, 0.3)"},
            ],
            "threshold": {
                "line": {"color": "#FAFAFA", "width": 3},
                "thickness": 0.8,
                "value": 100,
            },
        },
    ))
    
    fig.update_layout(
        paper_bgcolor="#0E1117",
        font={"color": "#FAFAFA"},
        height=height,
        margin={"l": 30, "r": 30, "t": 60, "b": 20},
    )
    
    return fig


def _create_var_chart(
    returns: pd.Series,
    var_95: float,
    var_99: float,
    height: int = 350,
) -> Optional["go.Figure"]:
    """Create a return distribution with VaR lines."""
    if not PLOTLY_AVAILABLE:
        return None
    
    try:
        import plotly.graph_objects as go
        import plotly.figure_factory as ff
    except ImportError:
        return None
    
    fig = go.Figure()
    
    # Histogram of returns
    fig.add_trace(go.Histogram(
        x=returns.values * 100,
        nbinsx=50,
        name="Daily Returns",
        marker_color="#3498db",
        opacity=0.7,
    ))
    
    # VaR lines
    fig.add_vline(
        x=var_95 * 100,
        line_dash="dash",
        line_color="#f39c12",
        annotation_text=f"VaR 95%: {var_95:.2%}",
        annotation_position="top left",
    )
    
    fig.add_vline(
        x=var_99 * 100,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text=f"VaR 99%: {var_99:.2%}",
        annotation_position="top left",
    )
    
    # Zero line
    fig.add_vline(x=0, line_color="#95a5a6", line_width=1)
    
    fig.update_layout(
        title={"text": "Return Distribution with VaR", "font": {"size": 16, "color": "#FAFAFA"}},
        paper_bgcolor="#0E1117",
        plot_bgcolor="#262730",
        font={"color": "#FAFAFA", "size": 11},
        height=height,
        showlegend=False,
        xaxis={
            "title": "Daily Return (%)",
            "gridcolor": "#3A3A3A",
            "zerolinecolor": "#555555",
        },
        yaxis={
            "title": "Frequency",
            "gridcolor": "#3A3A3A",
        },
        margin={"l": 60, "r": 40, "t": 60, "b": 50},
    )
    
    return fig


def _create_drawdown_context_chart(
    returns: pd.Series,
    height: int = 350,
) -> Optional["go.Figure"]:
    """Create a drawdown chart with historical context."""
    if not PLOTLY_AVAILABLE or returns is None or len(returns) < 20:
        return None
    
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    
    # Calculate drawdown
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak - 1) * 100  # As percentage
    
    fig = go.Figure()
    
    # Fill area for drawdown
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        fill="tozeroy",
        fillcolor="rgba(231, 76, 60, 0.3)",
        line=dict(color="#e74c3c", width=1.5),
        name="Drawdown",
        hovertemplate="Date: %{x}<br>Drawdown: %{y:.1f}%<extra></extra>",
    ))
    
    # Mark max drawdown
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min()
    
    fig.add_trace(go.Scatter(
        x=[max_dd_idx],
        y=[max_dd_val],
        mode="markers+text",
        marker=dict(size=12, color="#c0392b", symbol="diamond"),
        text=[f"Max DD: {max_dd_val:.1f}%"],
        textposition="bottom center",
        textfont=dict(color="#FAFAFA", size=11),
        name="Max Drawdown",
        showlegend=False,
    ))
    
    # Current drawdown annotation
    current_dd = drawdown.iloc[-1]
    if current_dd < -1:  # Only if in drawdown
        fig.add_annotation(
            x=drawdown.index[-1],
            y=current_dd,
            text=f"Current: {current_dd:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#FAFAFA",
            font=dict(color="#FAFAFA", size=11),
            bgcolor="rgba(231, 76, 60, 0.8)",
        )
    
    fig.update_layout(
        title={"text": "Drawdown Analysis", "font": {"size": 16, "color": "#FAFAFA"}},
        paper_bgcolor="#0E1117",
        plot_bgcolor="#262730",
        font={"color": "#FAFAFA", "size": 11},
        height=height,
        xaxis={
            "gridcolor": "#3A3A3A",
            "zerolinecolor": "#3A3A3A",
        },
        yaxis={
            "title": "Drawdown (%)",
            "gridcolor": "#3A3A3A",
            "zerolinecolor": "#555555",
            "zerolinewidth": 2,
        },
        hovermode="x unified",
        margin={"l": 60, "r": 40, "t": 60, "b": 40},
    )
    
    return fig


def render_pm_risk_dashboard(data: DashboardData):
    """
    Render comprehensive PM Risk Dashboard with actionable insights.
    
    Features:
    - Risk budget utilization gauges
    - VaR/CVaR analysis
    - Drawdown context
    - Risk alerts and recommendations
    - Factor risk heatmap
    """
    st.subheader("🎯 PM Risk Dashboard")
    
    st.markdown("""
    Comprehensive risk monitoring with actionable insights for portfolio management.
    """)
    
    # Get return series
    ret_series = select_return_series(data.diag) if data.diag is not None else None
    
    if ret_series is None or ret_series.empty:
        st.warning("No return data available for risk analysis")
        return
    
    # Calculate core metrics
    ann_vol = float(ret_series.std() * np.sqrt(252))
    ann_ret = float(ret_series.mean() * 252)
    sharpe = ann_ret / (ann_vol + 1e-12)
    
    equity = (1 + ret_series).cumprod()
    current_dd = float((equity / equity.cummax() - 1).iloc[-1])
    max_dd = float((equity / equity.cummax() - 1).min())
    
    # VaR/CVaR
    var_metrics = _compute_var_cvar(ret_series)
    
    # Risk budget
    risk_budget = _compute_risk_budget_utilization(
        current_vol=ann_vol,
        current_drawdown=current_dd,
    )
    
    # =========================================================================
    # SECTION 1: Risk Budget Overview
    # =========================================================================
    st.markdown("### 📊 Risk Budget Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if PLOTLY_AVAILABLE:
            fig = _create_risk_budget_gauge(
                risk_budget["overall_utilization"],
                title="Overall Risk Budget",
                height=220,
            )
            if fig:
                st.plotly_chart(fig, width='stretch')
        else:
            st.metric(
                "Risk Budget Used",
                f"{risk_budget['overall_utilization']:.0%}",
                delta=f"{risk_budget['risk_budget_remaining']:.0%} remaining",
            )
    
    with col2:
        if PLOTLY_AVAILABLE:
            fig = _create_risk_budget_gauge(
                risk_budget["vol_utilization"],
                title="Volatility Budget",
                height=220,
            )
            if fig:
                st.plotly_chart(fig, width='stretch')
        else:
            st.metric("Volatility Budget", f"{risk_budget['vol_utilization']:.0%}")
    
    with col3:
        if PLOTLY_AVAILABLE:
            fig = _create_risk_budget_gauge(
                risk_budget["dd_utilization"],
                title="Drawdown Budget",
                height=220,
            )
            if fig:
                st.plotly_chart(fig, width='stretch')
        else:
            st.metric("Drawdown Budget", f"{risk_budget['dd_utilization']:.0%}")
    
    # Risk budget summary
    budget_status = "🟢 Healthy" if risk_budget["overall_utilization"] < 0.6 else \
                   "🟡 Elevated" if risk_budget["overall_utilization"] < 0.8 else \
                   "🟠 High" if risk_budget["overall_utilization"] < 1.0 else "🔴 Exceeded"
    
    st.markdown(f"""
    <div style="background: #1e1e1e; padding: 15px; border-radius: 8px; border-left: 4px solid {'#2ecc71' if risk_budget['overall_utilization'] < 0.6 else '#f39c12' if risk_budget['overall_utilization'] < 0.8 else '#e74c3c'};">
        <b>Risk Budget Status: {budget_status}</b><br>
        <small>Volatility: {ann_vol:.1%} (Target: 15%) | Drawdown: {current_dd:.1%} (Limit: -20%)</small>
    </div>
    """, unsafe_allow_html=True)
    
    # =========================================================================
    # SECTION 2: Tail Risk Analysis
    # =========================================================================
    st.divider()
    st.markdown("### 📉 Tail Risk Analysis (VaR/CVaR)")
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        if PLOTLY_AVAILABLE and var_metrics:
            fig = _create_var_chart(
                ret_series,
                var_metrics.get("var_95", 0),
                var_metrics.get("var_99", 0),
                height=350,
            )
            if fig:
                st.plotly_chart(fig, width='stretch')
        else:
            st.info("VaR chart requires Plotly")
    
    with col2:
        st.markdown("**Daily Risk Metrics**")
        
        if var_metrics:
            col_a, col_b = st.columns(2)
            col_a.metric("VaR (95%)", f"{var_metrics.get('var_95', 0):.2%}")
            col_b.metric("VaR (99%)", f"{var_metrics.get('var_99', 0):.2%}")
            
            col_a, col_b = st.columns(2)
            col_a.metric("CVaR (95%)", f"{var_metrics.get('cvar_95', 0):.2%}")
            col_b.metric("CVaR (99%)", f"{var_metrics.get('cvar_99', 0):.2%}")
        
        st.markdown("**Annualized Risk**")
        col_a, col_b = st.columns(2)
        col_a.metric("Vol (Ann.)", f"{ann_vol:.1%}")
        col_b.metric("VaR 95% (Ann.)", f"{var_metrics.get('var_95_annual', 0):.1%}")
        
        # Tail risk interpretation
        st.markdown("""
        <div style="background: #262730; padding: 10px; border-radius: 5px; margin-top: 10px;">
        <small>
        <b>Interpretation:</b><br>
        • VaR 95%: 5% chance of loss exceeding this<br>
        • CVaR: Expected loss when VaR is breached<br>
        • Higher CVaR/VaR ratio → fatter tails
        </small>
        </div>
        """, unsafe_allow_html=True)
    
    # =========================================================================
    # SECTION 3: Drawdown Context
    # =========================================================================
    st.divider()
    st.markdown("### 📊 Drawdown Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if PLOTLY_AVAILABLE:
            fig = _create_drawdown_context_chart(ret_series, height=350)
            if fig:
                st.plotly_chart(fig, width='stretch')
        else:
            # Fallback
            dd_series = (equity / equity.cummax() - 1) * 100
            st.line_chart(dd_series.to_frame("Drawdown (%)"), height=300)
    
    with col2:
        st.markdown("**Drawdown Metrics**")
        
        st.metric(
            "Current Drawdown",
            f"{current_dd:.1%}",
            delta=f"{current_dd - max_dd:.1%} from max" if current_dd > max_dd else None,
        )
        
        st.metric("Max Drawdown", f"{max_dd:.1%}")
        
        # Days in drawdown
        in_dd = (equity / equity.cummax() - 1) < -0.01
        if in_dd.iloc[-1]:
            dd_start = in_dd[::-1].idxmax()
            days_in_dd = (ret_series.index[-1] - dd_start).days
            st.metric("Days in Current DD", f"{days_in_dd}")
        
        # Recovery estimate
        if current_dd < -0.01:
            recovery_needed = -current_dd / (1 + current_dd)
            st.metric(
                "Return to Recover",
                f"{recovery_needed:.1%}",
                help="Return needed to reach previous peak",
            )
        
        # Drawdown status
        dd_status = "🟢 No Drawdown" if current_dd > -0.02 else \
                   "🟡 Minor DD" if current_dd > -0.05 else \
                   "🟠 Moderate DD" if current_dd > -0.10 else \
                   "🔴 Significant DD" if current_dd > -0.15 else "⚫ Severe DD"
        
        st.markdown(f"**Status:** {dd_status}")
    
    # =========================================================================
    # SECTION 4: Risk Alerts
    # =========================================================================
    st.divider()
    st.markdown("### ⚠️ Risk Alerts & Recommendations")
    
    if COMPONENTS_AVAILABLE:
        insight_engine = PMInsightEngine()
        insights = insight_engine.analyze_portfolio(
            weights=data.weights,
            returns=ret_series,
            diag=data.diag,
            exposure=data.exposure,
        )
        
        render_insights_panel(insights, st)
    else:
        # Fallback simple alerts
        alerts = []
        
        if current_dd < -0.15:
            alerts.append(("🚨", "CRITICAL", f"Drawdown at {current_dd:.1%} - review risk exposure"))
        elif current_dd < -0.10:
            alerts.append(("⚠️", "WARNING", f"Elevated drawdown at {current_dd:.1%}"))
        
        if ann_vol > 0.25:
            alerts.append(("⚠️", "WARNING", f"Volatility at {ann_vol:.1%} exceeds target"))
        
        if sharpe < 0:
            alerts.append(("⚠️", "WARNING", f"Negative Sharpe ratio: {sharpe:.2f}"))
        
        if alerts:
            for icon, severity, msg in alerts:
                if severity == "CRITICAL":
                    st.error(f"{icon} **{severity}**: {msg}")
                else:
                    st.warning(f"{icon} **{severity}**: {msg}")
        else:
            st.success("✅ No critical risk alerts. Portfolio within normal parameters.")
    
    # =========================================================================
    # SECTION 5: Risk-Adjusted Performance Context
    # =========================================================================
    st.divider()
    st.markdown("### 📈 Risk-Adjusted Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Sharpe
    sharpe_status = "🟢" if sharpe >= 1.5 else "🟡" if sharpe >= 0.5 else "🔴"
    col1.metric(f"{sharpe_status} Sharpe", f"{sharpe:.2f}")
    
    # Sortino (downside deviation)
    downside_ret = ret_series[ret_series < 0]
    downside_vol = downside_ret.std() * np.sqrt(252) if len(downside_ret) > 0 else ann_vol
    sortino = ann_ret / (downside_vol + 1e-12)
    sortino_status = "🟢" if sortino >= 2.0 else "🟡" if sortino >= 1.0 else "🔴"
    col2.metric(f"{sortino_status} Sortino", f"{sortino:.2f}")
    
    # Calmar (return / max dd)
    calmar = ann_ret / (abs(max_dd) + 1e-12)
    calmar_status = "🟢" if calmar >= 1.0 else "🟡" if calmar >= 0.5 else "🔴"
    col3.metric(f"{calmar_status} Calmar", f"{calmar:.2f}")
    
    # Information ratio (vs rolling mean)
    if len(ret_series) >= 252:
        tracking_error = ret_series.rolling(252).std().iloc[-1] * np.sqrt(252)
        ir = ann_ret / (tracking_error + 1e-12)
        ir_status = "🟢" if ir >= 1.0 else "🟡" if ir >= 0.5 else "🔴"
        col4.metric(f"{ir_status} Info Ratio", f"{ir:.2f}")
    else:
        col4.metric("Info Ratio", "N/A")
    
    # Win Rate
    win_rate = (ret_series > 0).mean()
    win_status = "🟢" if win_rate >= 0.55 else "🟡" if win_rate >= 0.50 else "🔴"
    col5.metric(f"{win_status} Win Rate", f"{win_rate:.1%}")
    
    # Performance interpretation
    st.markdown("""
    <div style="background: #1e1e1e; padding: 15px; border-radius: 8px; margin-top: 10px;">
    <small>
    <b>Metric Targets:</b> Sharpe > 1.5 | Sortino > 2.0 | Calmar > 1.0 | Info Ratio > 1.0 | Win Rate > 55%
    </small>
    </div>
    """, unsafe_allow_html=True)
