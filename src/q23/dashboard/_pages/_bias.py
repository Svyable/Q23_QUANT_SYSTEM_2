"""
Bias Analysis Page

Comprehensive Long/Short/Net Exposure analysis for Portfolio Managers.
Shows exposure evolution, bias metrics, position breakdown, and actionable insights.

Author: Q23 Quant System
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from q23.dashboard.core import DashboardData
from q23.dashboard.analytics import select_return_series, top_positions

# Try to import Plotly for enhanced visualizations
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# =============================================================================
# Color Schemes
# =============================================================================

BIAS_COLORS = {
    "long": "#2ecc71",       # Green
    "short": "#e74c3c",      # Red
    "net": "#3498db",        # Blue
    "gross": "#9b59b6",      # Purple
    "neutral": "#95a5a6",    # Gray
    "warning": "#f39c12",    # Orange
    "background": "#0E1117",
    "card": "#262730",
    "text": "#FAFAFA",
    "grid": "#3A3A3A",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExposureMetrics:
    """Current exposure snapshot."""
    long_exposure: float
    short_exposure: float
    net_exposure: float
    gross_exposure: float
    long_count: int
    short_count: int
    total_count: int
    long_short_ratio: float  # Long / |Short|
    

@dataclass
class ExposureStats:
    """Historical exposure statistics."""
    avg_long: float
    avg_short: float
    avg_net: float
    avg_gross: float
    std_net: float
    min_net: float
    max_net: float
    net_zscore: float  # Current net vs historical
    regime: str  # "long_biased", "neutral", "short_biased"


# =============================================================================
# Analysis Functions
# =============================================================================

def _compute_exposure_series(weights: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time series of exposure metrics.
    
    Returns DataFrame with columns: long, short, net, gross
    """
    long_exp = weights.clip(lower=0).sum(axis=1)
    short_exp = weights.clip(upper=0).sum(axis=1).abs()
    net_exp = weights.sum(axis=1)
    gross_exp = weights.abs().sum(axis=1)
    
    return pd.DataFrame({
        "long": long_exp,
        "short": short_exp,
        "net": net_exp,
        "gross": gross_exp,
        "long_count": (weights > 1e-6).sum(axis=1),
        "short_count": (weights < -1e-6).sum(axis=1),
    })


def _compute_current_metrics(weights: pd.DataFrame) -> ExposureMetrics:
    """Compute current exposure snapshot."""
    if weights.empty:
        return ExposureMetrics(0, 0, 0, 0, 0, 0, 0, 0)
    
    w_last = weights.iloc[-1]
    longs = w_last[w_last > 1e-6]
    shorts = w_last[w_last < -1e-6]
    
    long_exp = float(longs.sum())
    short_exp = float(shorts.sum())
    net_exp = long_exp + short_exp
    gross_exp = long_exp + abs(short_exp)
    
    return ExposureMetrics(
        long_exposure=long_exp,
        short_exposure=short_exp,
        net_exposure=net_exp,
        gross_exposure=gross_exp,
        long_count=len(longs),
        short_count=len(shorts),
        total_count=len(longs) + len(shorts),
        long_short_ratio=long_exp / abs(short_exp) if abs(short_exp) > 1e-6 else float('inf'),
    )


def _compute_historical_stats(exposure_df: pd.DataFrame, lookback: int = 252) -> ExposureStats:
    """Compute historical exposure statistics."""
    df = exposure_df.tail(lookback)
    
    if df.empty:
        return ExposureStats(0, 0, 0, 0, 0, 0, 0, 0, "unknown")
    
    avg_net = float(df["net"].mean())
    std_net = float(df["net"].std())
    current_net = float(df["net"].iloc[-1])
    
    # Compute z-score
    zscore = (current_net - avg_net) / (std_net + 1e-8)
    
    # Determine regime
    if avg_net > 0.1:
        regime = "long_biased"
    elif avg_net < -0.1:
        regime = "short_biased"
    else:
        regime = "neutral"
    
    return ExposureStats(
        avg_long=float(df["long"].mean()),
        avg_short=float(df["short"].mean()),
        avg_net=avg_net,
        avg_gross=float(df["gross"].mean()),
        std_net=std_net,
        min_net=float(df["net"].min()),
        max_net=float(df["net"].max()),
        net_zscore=zscore,
        regime=regime,
    )


def _compute_position_breakdown(weights: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get detailed position breakdown for longs and shorts."""
    w_last = weights.iloc[-1]
    
    longs = w_last[w_last > 1e-6].sort_values(ascending=False)
    shorts = w_last[w_last < -1e-6].sort_values()
    
    long_df = pd.DataFrame({
        "ticker": longs.index,
        "weight": longs.values,
        "pct_of_long": longs.values / longs.sum() if longs.sum() > 0 else 0,
    }).reset_index(drop=True)
    
    short_df = pd.DataFrame({
        "ticker": shorts.index,
        "weight": shorts.values,
        "pct_of_short": shorts.values / shorts.sum() if shorts.sum() != 0 else 0,
    }).reset_index(drop=True)
    
    return long_df, short_df


def _compute_exposure_changes(exposure_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily changes in exposure."""
    changes = exposure_df.diff()
    changes.columns = [f"{c}_change" for c in changes.columns]
    return changes


def _compute_concentration_by_side(weights: pd.DataFrame) -> Dict[str, float]:
    """Compute concentration metrics for longs and shorts separately."""
    w_last = weights.iloc[-1]
    
    longs = w_last[w_last > 1e-6]
    shorts = w_last[w_last < -1e-6].abs()
    
    def hhi(w):
        if w.sum() < 1e-8:
            return 0
        normalized = w / w.sum()
        return float((normalized ** 2).sum())
    
    def top_n_conc(w, n=5):
        if w.sum() < 1e-8:
            return 0
        return float(w.nlargest(n).sum() / w.sum())
    
    return {
        "long_hhi": hhi(longs),
        "short_hhi": hhi(shorts),
        "long_top5": top_n_conc(longs),
        "short_top5": top_n_conc(shorts),
        "long_effective_n": 1 / hhi(longs) if hhi(longs) > 0 else 0,
        "short_effective_n": 1 / hhi(shorts) if hhi(shorts) > 0 else 0,
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def _create_exposure_time_series_chart(exposure_df: pd.DataFrame, height: Optional[int] = None) -> Optional[object]:
    """Create main exposure time series chart with unified hover."""
    if not PLOTLY_AVAILABLE or exposure_df.empty:
        return None
    
    fig = go.Figure()
    
    # Long exposure (green fill)
    fig.add_trace(go.Scatter(
        x=exposure_df.index,
        y=exposure_df["long"],
        mode="lines",
        name="Long",
        line={"color": BIAS_COLORS["long"], "width": 2},
        fill="tozeroy",
        fillcolor="rgba(46, 204, 113, 0.2)",
        hovertemplate="%{y:.1%}",
    ))
    
    # Short exposure (red, shown as negative for visual clarity)
    fig.add_trace(go.Scatter(
        x=exposure_df.index,
        y=-exposure_df["short"],  # Negative for visual
        mode="lines",
        name="Short",
        line={"color": BIAS_COLORS["short"], "width": 2},
        fill="tozeroy",
        fillcolor="rgba(231, 76, 60, 0.2)",
        hovertemplate="%{y:.1%}",
    ))
    
    # Net exposure (blue line)
    fig.add_trace(go.Scatter(
        x=exposure_df.index,
        y=exposure_df["net"],
        mode="lines",
        name="Net (Bias)",
        line={"color": BIAS_COLORS["net"], "width": 2.5},
        hovertemplate="%{y:.1%}",
    ))
    
    # Gross exposure (purple, secondary axis)
    fig.add_trace(go.Scatter(
        x=exposure_df.index,
        y=exposure_df["gross"],
        mode="lines",
        name="Gross",
        line={"color": BIAS_COLORS["gross"], "width": 1.5, "dash": "dot"},
        yaxis="y2",
        hovertemplate="%{y:.1%}",
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    
    layout = {
        "title": {"text": "Exposure Over Time", "font": {"size": 16, "color": BIAS_COLORS["text"]}},
        "paper_bgcolor": BIAS_COLORS["background"],
        "plot_bgcolor": BIAS_COLORS["card"],
        "font": {"color": BIAS_COLORS["text"], "size": 11},
        "autosize": True,
        "showlegend": False,  # Clean - tooltips show all
        "xaxis": {"gridcolor": BIAS_COLORS["grid"]},
        "yaxis": {
            "title": "Net Exposure",
            "gridcolor": BIAS_COLORS["grid"],
            "tickformat": ".0%",
            "zeroline": True,
            "zerolinecolor": "rgba(255,255,255,0.3)",
        },
        "yaxis2": {
            "title": "Gross Exposure",
            "overlaying": "y",
            "side": "right",
            "tickformat": ".0%",
            "showgrid": False,
        },
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": "#1e1e1e",
            "bordercolor": "#444",
            "font": {"size": 12, "color": BIAS_COLORS["text"]},
        },
        "margin": {"l": 60, "r": 60, "t": 50, "b": 40},
    }
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    
    return fig


def _create_long_short_area_chart(exposure_df: pd.DataFrame, height: Optional[int] = None) -> Optional[object]:
    """Create stacked area chart showing long vs short."""
    if not PLOTLY_AVAILABLE or exposure_df.empty:
        return None
    
    fig = go.Figure()
    
    # Long area
    fig.add_trace(go.Scatter(
        x=exposure_df.index,
        y=exposure_df["long"],
        mode="lines",
        name="Long",
        line={"color": BIAS_COLORS["long"], "width": 0},
        fill="tozeroy",
        fillcolor="rgba(46, 204, 113, 0.6)",
        hovertemplate="Long: %{y:.1%}",
    ))
    
    # Short area (shown as positive for stacking)
    fig.add_trace(go.Scatter(
        x=exposure_df.index,
        y=exposure_df["short"],
        mode="lines",
        name="Short",
        line={"color": BIAS_COLORS["short"], "width": 0},
        fill="tozeroy",
        fillcolor="rgba(231, 76, 60, 0.6)",
        hovertemplate="Short: %{y:.1%}",
    ))
    
    layout = {
        "title": {"text": "Long vs Short Exposure", "font": {"size": 14, "color": BIAS_COLORS["text"]}},
        "paper_bgcolor": BIAS_COLORS["background"],
        "plot_bgcolor": BIAS_COLORS["card"],
        "font": {"color": BIAS_COLORS["text"]},
        "autosize": True,
        "showlegend": False,
        "xaxis": {"gridcolor": BIAS_COLORS["grid"]},
        "yaxis": {"title": "Exposure", "gridcolor": BIAS_COLORS["grid"], "tickformat": ".0%"},
        "hovermode": "x unified",
        "hoverlabel": {"bgcolor": "#1e1e1e", "font": {"color": BIAS_COLORS["text"]}},
        "margin": {"l": 50, "r": 20, "t": 50, "b": 40},
    }
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    
    return fig


def _create_position_count_chart(exposure_df: pd.DataFrame, height: Optional[int] = None) -> Optional[object]:
    """Create position count chart."""
    if not PLOTLY_AVAILABLE or exposure_df.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=exposure_df.index,
        y=exposure_df["long_count"],
        mode="lines",
        name="Long Positions",
        line={"color": BIAS_COLORS["long"], "width": 2},
        fill="tozeroy",
        fillcolor="rgba(46, 204, 113, 0.2)",
        hovertemplate="%{y:.0f}",
    ))
    
    fig.add_trace(go.Scatter(
        x=exposure_df.index,
        y=exposure_df["short_count"],
        mode="lines",
        name="Short Positions",
        line={"color": BIAS_COLORS["short"], "width": 2},
        fill="tozeroy",
        fillcolor="rgba(231, 76, 60, 0.2)",
        hovertemplate="%{y:.0f}",
    ))
    
    fig.update_layout(
        title={"text": "Position Counts", "font": {"size": 14, "color": BIAS_COLORS["text"]}},
        paper_bgcolor=BIAS_COLORS["background"],
        plot_bgcolor=BIAS_COLORS["card"],
        font={"color": BIAS_COLORS["text"]},
        height=height,
        showlegend=False,
        xaxis={"gridcolor": BIAS_COLORS["grid"]},
        yaxis={"title": "# Positions", "gridcolor": BIAS_COLORS["grid"]},
        hovermode="x unified",
        hoverlabel={"bgcolor": "#1e1e1e", "font": {"color": BIAS_COLORS["text"]}},
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
    )
    
    return fig


def _create_net_exposure_histogram(exposure_df: pd.DataFrame, height: Optional[int] = None) -> Optional[object]:
    """Create histogram of net exposure distribution."""
    if not PLOTLY_AVAILABLE or exposure_df.empty:
        return None
    
    net = exposure_df["net"]
    current = net.iloc[-1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=net.values,
        nbinsx=30,
        marker_color=BIAS_COLORS["net"],
        opacity=0.7,
        hovertemplate="Net Exposure: %{x:.1%}<br>Count: %{y}<extra></extra>",
    ))
    
    # Add current value marker
    fig.add_vline(
        x=current, 
        line_dash="dash", 
        line_color=BIAS_COLORS["warning"],
        annotation_text=f"Current: {current:.1%}",
        annotation_position="top",
    )
    
    layout = {
        "title": {"text": "Net Exposure Distribution", "font": {"size": 14, "color": BIAS_COLORS["text"]}},
        "paper_bgcolor": BIAS_COLORS["background"],
        "plot_bgcolor": BIAS_COLORS["card"],
        "font": {"color": BIAS_COLORS["text"]},
        "autosize": True,
        "xaxis": {"title": "Net Exposure", "gridcolor": BIAS_COLORS["grid"], "tickformat": ".0%"},
        "yaxis": {"title": "Frequency", "gridcolor": BIAS_COLORS["grid"]},
        "margin": {"l": 50, "r": 20, "t": 50, "b": 40},
    }
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    
    return fig


def _create_bias_gauge(current_net: float, height: Optional[int] = None) -> Optional[object]:
    """Create a gauge showing current bias level."""
    if not PLOTLY_AVAILABLE:
        return None
    
    # Clamp to reasonable range
    display_val = max(min(current_net, 1.5), -0.5)
    
    # Determine color
    if current_net > 0.8:
        color = BIAS_COLORS["long"]
    elif current_net < 0.2:
        color = BIAS_COLORS["short"]
    else:
        color = BIAS_COLORS["net"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_val * 100,
        number={"suffix": "%", "font": {"size": 28, "color": BIAS_COLORS["text"]}},
        title={"text": "Net Bias", "font": {"size": 14, "color": BIAS_COLORS["text"]}},
        gauge={
            "axis": {"range": [-50, 150], "ticksuffix": "%", "tickcolor": BIAS_COLORS["text"]},
            "bar": {"color": color},
            "bgcolor": BIAS_COLORS["card"],
            "borderwidth": 2,
            "bordercolor": BIAS_COLORS["grid"],
            "steps": [
                {"range": [-50, 20], "color": "rgba(231, 76, 60, 0.2)"},
                {"range": [20, 80], "color": "rgba(52, 152, 219, 0.2)"},
                {"range": [80, 150], "color": "rgba(46, 204, 113, 0.2)"},
            ],
            "threshold": {
                "line": {"color": BIAS_COLORS["warning"], "width": 3},
                "thickness": 0.8,
                "value": 100,  # Target = 100% (fully invested long)
            },
        },
    ))
    
    layout = {
        "paper_bgcolor": BIAS_COLORS["background"],
        "font": {"color": BIAS_COLORS["text"]},
        "autosize": True,
        "margin": {"l": 30, "r": 30, "t": 50, "b": 20},
    }
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    
    return fig


def _generate_bias_csv(
    exposure_df: pd.DataFrame,
    current_metrics: ExposureMetrics,
    historical_stats: ExposureStats,
    concentration: Dict[str, float],
) -> str:
    """
    Generate CSV export of bias analysis data.
    
    Args:
        exposure_df: Time series exposure DataFrame
        current_metrics: Current exposure snapshot
        historical_stats: Historical stats
        concentration: Concentration metrics
        
    Returns:
        CSV string
    """
    # Summary section
    summary_rows = [
        {"Metric": "Current Net Exposure", "Value": f"{current_metrics.net_exposure:.2%}"},
        {"Metric": "Current Long Exposure", "Value": f"{current_metrics.long_exposure:.2%}"},
        {"Metric": "Current Short Exposure", "Value": f"{current_metrics.short_exposure:.2%}"},
        {"Metric": "Current Gross Exposure", "Value": f"{current_metrics.gross_exposure:.2%}"},
        {"Metric": "Long Positions", "Value": str(current_metrics.long_count)},
        {"Metric": "Short Positions", "Value": str(current_metrics.short_count)},
        {"Metric": "L/S Ratio", "Value": f"{current_metrics.long_short_ratio:.2f}"},
        {"Metric": "Avg Net (Historical)", "Value": f"{historical_stats.avg_net:.2%}"},
        {"Metric": "Std Net (Historical)", "Value": f"{historical_stats.std_net:.2%}"},
        {"Metric": "Net Z-Score", "Value": f"{historical_stats.net_zscore:.2f}"},
        {"Metric": "Regime", "Value": historical_stats.regime},
        {"Metric": "Long HHI", "Value": f"{concentration['long_hhi']:.4f}"},
        {"Metric": "Short HHI", "Value": f"{concentration['short_hhi']:.4f}"},
        {"Metric": "Long Top5 Conc", "Value": f"{concentration['long_top5']:.2%}"},
        {"Metric": "Short Top5 Conc", "Value": f"{concentration['short_top5']:.2%}"},
    ]
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Time series data
    ts_df = exposure_df.copy()
    ts_df.index.name = "Date"
    ts_df = ts_df.reset_index()
    
    # Combine into single CSV with sections
    csv_parts = []
    csv_parts.append("# BIAS ANALYSIS SUMMARY")
    csv_parts.append(summary_df.to_csv(index=False))
    csv_parts.append("\n# TIME SERIES DATA")
    csv_parts.append(ts_df.to_csv(index=False))
    
    return "\n".join(csv_parts)


def _create_rolling_bias_chart(exposure_df: pd.DataFrame, window: int = 21, height: Optional[int] = None) -> Optional[object]:
    """Create rolling average bias chart."""
    if not PLOTLY_AVAILABLE or exposure_df.empty or len(exposure_df) < window:
        return None
    
    rolling_net = exposure_df["net"].rolling(window).mean()
    rolling_std = exposure_df["net"].rolling(window).std()
    
    fig = go.Figure()
    
    # Rolling mean
    fig.add_trace(go.Scatter(
        x=rolling_net.index,
        y=rolling_net.values,
        mode="lines",
        name=f"{window}d Avg Bias",
        line={"color": BIAS_COLORS["net"], "width": 2},
        hovertemplate="%{y:.1%}",
    ))
    
    # +/- 1 std bands
    upper = rolling_net + rolling_std
    lower = rolling_net - rolling_std
    
    fig.add_trace(go.Scatter(
        x=upper.index.tolist() + lower.index.tolist()[::-1],
        y=upper.values.tolist() + lower.values.tolist()[::-1],
        fill="toself",
        fillcolor="rgba(52, 152, 219, 0.15)",
        line={"color": "rgba(0,0,0,0)"},
        name="±1 Std",
        hoverinfo="skip",
    ))
    
    fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(46, 204, 113, 0.5)", 
                  annotation_text="Full Long")
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255, 255, 255, 0.3)")
    
    layout = {
        "title": {"text": f"Rolling {window}-Day Average Bias", "font": {"size": 14, "color": BIAS_COLORS["text"]}},
        "paper_bgcolor": BIAS_COLORS["background"],
        "plot_bgcolor": BIAS_COLORS["card"],
        "font": {"color": BIAS_COLORS["text"]},
        "autosize": True,
        "showlegend": False,
        "xaxis": {"gridcolor": BIAS_COLORS["grid"]},
        "yaxis": {"title": "Net Exposure", "gridcolor": BIAS_COLORS["grid"], "tickformat": ".0%"},
        "hovermode": "x unified",
        "hoverlabel": {"bgcolor": "#1e1e1e", "font": {"color": BIAS_COLORS["text"]}},
        "margin": {"l": 50, "r": 20, "t": 50, "b": 40},
    }
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    
    return fig


# =============================================================================
# Main Render Function
# =============================================================================

def render_bias_page(data: DashboardData) -> None:
    """
    Render the Bias Analysis page.
    
    Comprehensive Long/Short/Net exposure analysis for Portfolio Managers.
    
    Args:
        data: Dashboard data bundle
    """
    # Header with download button
    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.subheader("⚖️ Bias Analysis")
    
    st.markdown("""
    Comprehensive Long/Short exposure analysis. Track portfolio bias evolution,
    position breakdown, and exposure concentration.
    """)
    
    if data.weights is None or data.weights.empty:
        st.warning("No weights data available")
        st.stop()
    
    # ==========================================================================
    # Compute All Metrics
    # ==========================================================================
    exposure_df = _compute_exposure_series(data.weights)
    current_metrics = _compute_current_metrics(data.weights)
    historical_stats = _compute_historical_stats(exposure_df)
    concentration = _compute_concentration_by_side(data.weights)
    
    # Add download button after data is computed
    with header_col2:
        csv_data = _generate_bias_csv(exposure_df, current_metrics, historical_stats, concentration)
        st.download_button(
            label="📥 Export",
            data=csv_data,
            file_name="bias_analysis.csv",
            mime="text/csv",
            help="Download bias analysis data as CSV",
        )
    
    # ==========================================================================
    # CURRENT SNAPSHOT
    # ==========================================================================
    st.markdown("### 📊 Current Exposure Snapshot")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        color = "normal" if current_metrics.net_exposure > 0 else "inverse"
        st.metric(
            "Net Exposure",
            f"{current_metrics.net_exposure:.1%}",
            delta=f"z={historical_stats.net_zscore:.1f}" if abs(historical_stats.net_zscore) > 1 else None,
            delta_color=color,
        )
    
    with col2:
        st.metric(
            "Long Exposure",
            f"{current_metrics.long_exposure:.1%}",
            delta=f"{current_metrics.long_count} positions",
        )
    
    with col3:
        st.metric(
            "Short Exposure",
            f"{current_metrics.short_exposure:.1%}",
            delta=f"{current_metrics.short_count} positions",
        )
    
    with col4:
        st.metric(
            "Gross Exposure",
            f"{current_metrics.gross_exposure:.1%}",
            delta=f"{current_metrics.total_count} total",
        )
    
    with col5:
        ratio_str = f"{current_metrics.long_short_ratio:.1f}x" if current_metrics.long_short_ratio < 100 else "Long Only"
        st.metric(
            "L/S Ratio",
            ratio_str,
            delta=historical_stats.regime.replace("_", " ").title(),
        )
    
    st.divider()
    
    # ==========================================================================
    # MAIN EXPOSURE CHART
    # ==========================================================================
    st.markdown("### 📈 Exposure Evolution")
    
    if PLOTLY_AVAILABLE:
        fig = _create_exposure_time_series_chart(exposure_df.tail(500))
        if fig:
            st.plotly_chart(fig)
    else:
        st.line_chart(exposure_df[["long", "short", "net", "gross"]].tail(500), height=350)
    
    st.divider()
    
    # ==========================================================================
    # DETAILED ANALYSIS SECTION
    # ==========================================================================
    col_left, col_right = st.columns([1.5, 1])
    
    with col_left:
        # Long vs Short area chart
        st.markdown("### 📊 Long vs Short Breakdown")
        
        if PLOTLY_AVAILABLE:
            fig = _create_long_short_area_chart(exposure_df.tail(500))
            if fig:
                st.plotly_chart(fig)
        else:
            st.line_chart(exposure_df[["long", "short"]].tail(500), height=250)
        
        # Position count chart
        st.markdown("### 🔢 Position Counts")
        
        if PLOTLY_AVAILABLE:
            fig = _create_position_count_chart(exposure_df.tail(500))
            if fig:
                st.plotly_chart(fig)
        else:
            st.line_chart(exposure_df[["long_count", "short_count"]].tail(500), height=200)
    
    with col_right:
        # Bias gauge
        st.markdown("### 🎯 Current Bias")
        
        if PLOTLY_AVAILABLE:
            fig = _create_bias_gauge(current_metrics.net_exposure)
            if fig:
                st.plotly_chart(fig)
        else:
            bias_pct = current_metrics.net_exposure * 100
            if bias_pct > 80:
                st.success(f"**Net Bias: {bias_pct:.0f}%** (Long Biased)")
            elif bias_pct > 20:
                st.info(f"**Net Bias: {bias_pct:.0f}%** (Balanced)")
            else:
                st.warning(f"**Net Bias: {bias_pct:.0f}%** (Short/Hedged)")
        
        # Net exposure distribution
        st.markdown("### 📉 Bias Distribution")
        
        if PLOTLY_AVAILABLE:
            fig = _create_net_exposure_histogram(exposure_df.tail(252))
            if fig:
                st.plotly_chart(fig)
        else:
            st.caption("Distribution requires Plotly")
        
        # Historical stats
        st.markdown("### 📋 Historical Stats")
        
        stats_df = pd.DataFrame({
            "Metric": ["Avg Net", "Std Net", "Min Net", "Max Net", "Avg Gross"],
            "Value": [
                f"{historical_stats.avg_net:.1%}",
                f"{historical_stats.std_net:.1%}",
                f"{historical_stats.min_net:.1%}",
                f"{historical_stats.max_net:.1%}",
                f"{historical_stats.avg_gross:.1%}",
            ],
        })
        st.dataframe(stats_df, hide_index=True, width="stretch")
    
    st.divider()
    
    # ==========================================================================
    # ROLLING BIAS & CONCENTRATION
    # ==========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Rolling Bias Analysis")
        
        window = st.selectbox(
            "📈 Rolling Window",
            options=[5, 10, 21, 63],
            index=2,
            format_func=lambda x: f"{x} days",
            key="bias_rolling_window",
            help="Time window for rolling bias calculations. Shorter windows show recent trends, longer windows show structural bias.",
        )
        
        if PLOTLY_AVAILABLE:
            fig = _create_rolling_bias_chart(exposure_df.tail(500), window=window)
            if fig:
                st.plotly_chart(fig)
        else:
            rolling = exposure_df["net"].rolling(window).mean()
            st.line_chart(rolling.tail(500), height=250)
    
    with col2:
        st.markdown("### 🎯 Concentration by Side")
        
        conc_col1, conc_col2 = st.columns(2)
        
        with conc_col1:
            st.markdown("**Long Side**")
            st.metric("Top 5 Conc", f"{concentration['long_top5']:.0%}")
            st.metric("HHI", f"{concentration['long_hhi']:.3f}")
            st.metric("Effective N", f"{concentration['long_effective_n']:.1f}")
        
        with conc_col2:
            st.markdown("**Short Side**")
            if current_metrics.short_count > 0:
                st.metric("Top 5 Conc", f"{concentration['short_top5']:.0%}")
                st.metric("HHI", f"{concentration['short_hhi']:.3f}")
                st.metric("Effective N", f"{concentration['short_effective_n']:.1f}")
            else:
                st.caption("No short positions")
    
    st.divider()
    
    # ==========================================================================
    # TOP POSITIONS BY SIDE
    # ==========================================================================
    st.markdown("### 🏆 Top Positions by Side")
    
    long_df, short_df = _compute_position_breakdown(data.weights)
    
    col_long, col_short = st.columns(2)
    
    with col_long:
        st.markdown("**Top Longs**")
        if not long_df.empty:
            display_long = long_df.head(15).copy()
            display_long["weight"] = display_long["weight"].apply(lambda x: f"{x:.2%}")
            display_long["pct_of_long"] = display_long["pct_of_long"].apply(lambda x: f"{x:.1%}")
            st.dataframe(display_long, hide_index=True, height=400, width="stretch")
        else:
            st.caption("No long positions")
    
    with col_short:
        st.markdown("**Top Shorts**")
        if not short_df.empty:
            display_short = short_df.head(15).copy()
            display_short["weight"] = display_short["weight"].apply(lambda x: f"{x:.2%}")
            display_short["pct_of_short"] = display_short["pct_of_short"].apply(lambda x: f"{x:.1%}")
            st.dataframe(display_short, hide_index=True, height=400, width="stretch")
        else:
            st.caption("No short positions")
    
    st.divider()
    
    # ==========================================================================
    # RAW DATA
    # ==========================================================================
    with st.expander("📋 Raw Exposure Data", expanded=False):
        display_df = exposure_df.tail(100).copy()
        display_df["long"] = display_df["long"].apply(lambda x: f"{x:.2%}")
        display_df["short"] = display_df["short"].apply(lambda x: f"{x:.2%}")
        display_df["net"] = display_df["net"].apply(lambda x: f"{x:.2%}")
        display_df["gross"] = display_df["gross"].apply(lambda x: f"{x:.2%}")
        st.dataframe(display_df, width="stretch", height=400)

