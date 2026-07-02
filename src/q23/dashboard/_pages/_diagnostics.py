"""
Enhanced Portfolio Diagnostics Page

Comprehensive portfolio diagnostics for Portfolio Managers including:
- Turnover Analysis with TC estimation
- Concentration & Position Metrics
- Rolling Risk-Adjusted Performance
- Signal Decay & Alpha Analysis
- Trade Quality Assessment
- Regime Detection
- PM Alerts & Action Items

Author: Q23 Quant System
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from q23.dashboard.core import DashboardData
from q23.dashboard.analytics import (
    select_return_series,
    hit_rate,
    winner_loser_sizing,
    rolling_sharpe,
    turnover_series,
)

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

DIAG_COLORS = {
    "primary": "#3498db",      # Blue
    "secondary": "#2ecc71",    # Green
    "warning": "#f39c12",      # Orange
    "danger": "#e74c3c",       # Red
    "neutral": "#95a5a6",      # Gray
    "purple": "#9b59b6",       # Purple
    "teal": "#1abc9c",         # Teal
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TurnoverMetrics:
    """Turnover analysis results."""
    series: pd.Series
    avg_daily: float
    current: float
    max: float
    min: float
    volatility: float
    annualized: float
    estimated_tc_bps: float  # Estimated transaction cost in bps
    tc_drag_annual: float    # Annual TC drag on returns


@dataclass
class ConcentrationMetrics:
    """Concentration analysis results."""
    time_series: pd.DataFrame
    current_n_positions: int
    current_top5_conc: float
    current_top10_conc: float
    current_hhi: float
    effective_n: float  # 1/HHI
    max_position_weight: float
    position_count_trend: str  # "increasing", "stable", "decreasing"


@dataclass 
class SignalDecayMetrics:
    """Signal decay analysis results."""
    decay_rate: float  # Daily decay rate
    half_life_days: float  # Days until signal strength halves
    autocorrelation: pd.Series  # Lag autocorrelations
    is_decaying: bool
    decay_severity: str  # "minimal", "moderate", "severe"


@dataclass
class TradeQualityMetrics:
    """Trade execution quality assessment."""
    avg_trade_size: float
    trade_frequency: float  # Trades per day
    rebalance_efficiency: float  # How much turnover leads to return
    churn_ratio: float  # Unnecessary trading ratio
    timing_score: float  # 0-100, how well timed are trades


# =============================================================================
# Analysis Functions
# =============================================================================

def _compute_turnover_analysis(
    weights: pd.DataFrame,
    tc_bps: float = 10.0,
) -> TurnoverMetrics:
    """
    Compute comprehensive turnover statistics with TC estimation.
    
    Args:
        weights: Portfolio weights (time x asset)
        tc_bps: Estimated transaction cost in basis points (default 10 bps)
    
    Returns:
        TurnoverMetrics with all statistics
    """
    turnover = turnover_series(weights)
    
    if turnover.empty:
        return TurnoverMetrics(
            series=pd.Series(dtype=float),
            avg_daily=0.0,
            current=0.0,
            max=0.0,
            min=0.0,
            volatility=0.0,
            annualized=0.0,
            estimated_tc_bps=0.0,
            tc_drag_annual=0.0,
        )
    
    avg_daily = float(turnover.mean())
    annualized = avg_daily * 252  # Approx trading days
    
    # TC estimation: turnover * tc_bps / 10000
    tc_per_rebalance = avg_daily * (tc_bps / 10000)
    tc_drag_annual = tc_per_rebalance * 252
    
    return TurnoverMetrics(
        series=turnover,
        avg_daily=avg_daily,
        current=float(turnover.iloc[-1]) if len(turnover) > 0 else 0.0,
        max=float(turnover.max()),
        min=float(turnover.min()),
        volatility=float(turnover.std()),
        annualized=annualized,
        estimated_tc_bps=tc_bps,
        tc_drag_annual=tc_drag_annual,
    )


def _compute_concentration_metrics(weights: pd.DataFrame) -> ConcentrationMetrics:
    """Compute comprehensive concentration metrics over time."""
    metrics = []
    
    # Vectorized computation: compute gross exposure for all dates at once
    gross_exposure = weights.abs().sum(axis=1)
    valid_dates = gross_exposure[gross_exposure >= 1e-12].index
    
    # Process each date (still need loop for per-date metrics, but use vectorized operations)
    for dt in valid_dates:
        w_row = weights.loc[dt]
        w_abs = w_row.abs()
        gross = gross_exposure.loc[dt]
        
        w_sorted = w_abs.sort_values(ascending=False)
        w_normalized = w_abs / gross
        
        hhi = (w_normalized ** 2).sum()
        
        metrics.append({
            "date": dt,
            "n_positions": int((w_abs > 1e-6).sum()),
            "top1_concentration": float(w_sorted.iloc[0] / gross) if len(w_sorted) > 0 else 0.0,
            "top5_concentration": float(w_sorted.head(5).sum() / gross) if len(w_sorted) >= 5 else 1.0,
            "top10_concentration": float(w_sorted.head(10).sum() / gross) if len(w_sorted) >= 10 else 1.0,
            "herfindahl": float(hhi),
            "effective_n": float(1 / hhi) if hhi > 0 else 0.0,
            "max_weight": float(w_sorted.iloc[0]) if len(w_sorted) > 0 else 0.0,
        })
    
    if not metrics:
        return ConcentrationMetrics(
            time_series=pd.DataFrame(),
            current_n_positions=0,
            current_top5_conc=0.0,
            current_top10_conc=0.0,
            current_hhi=0.0,
            effective_n=0.0,
            max_position_weight=0.0,
            position_count_trend="unknown",
        )
    
    df = pd.DataFrame(metrics).set_index("date")
    latest = df.iloc[-1]
    
    # Determine position count trend
    if len(df) >= 21:
        recent_avg = df["n_positions"].tail(21).mean()
        older_avg = df["n_positions"].iloc[-42:-21].mean() if len(df) >= 42 else recent_avg
        
        if recent_avg > older_avg * 1.1:
            trend = "increasing"
        elif recent_avg < older_avg * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        trend = "unknown"
    
    return ConcentrationMetrics(
        time_series=df,
        current_n_positions=int(latest["n_positions"]),
        current_top5_conc=float(latest["top5_concentration"]),
        current_top10_conc=float(latest["top10_concentration"]),
        current_hhi=float(latest["herfindahl"]),
        effective_n=float(latest["effective_n"]),
        max_position_weight=float(latest["max_weight"]),
        position_count_trend=trend,
    )


def _compute_signal_decay(
    returns: pd.Series,
    weights: pd.DataFrame,
    max_lag: int = 21,
) -> SignalDecayMetrics:
    """
    Analyze signal decay by measuring autocorrelation of returns.
    
    Strong signal decay indicates alpha is being arbitraged away.
    """
    if returns is None or len(returns) < max_lag * 2:
        return SignalDecayMetrics(
            decay_rate=0.0,
            half_life_days=float('inf'),
            autocorrelation=pd.Series(dtype=float),
            is_decaying=False,
            decay_severity="unknown",
        )
    
    # Compute autocorrelation at various lags
    autocorrs = []
    for lag in range(1, max_lag + 1):
        ac = returns.autocorr(lag=lag)
        autocorrs.append({"lag": lag, "autocorr": ac if not np.isnan(ac) else 0.0})
    
    ac_series = pd.DataFrame(autocorrs).set_index("lag")["autocorr"]
    
    # Estimate decay rate from autocorrelation
    # If AC(1) and AC(5) are both positive and declining, signal is decaying
    ac1 = ac_series.iloc[0] if len(ac_series) > 0 else 0
    ac5 = ac_series.iloc[4] if len(ac_series) > 4 else 0
    
    if ac1 > 0.05:
        # Positive momentum - signal persists
        if ac5 < ac1 * 0.5:
            decay_rate = (ac1 - ac5) / 5
            half_life = np.log(2) / decay_rate if decay_rate > 0 else float('inf')
            is_decaying = True
            severity = "severe" if half_life < 5 else "moderate" if half_life < 10 else "minimal"
        else:
            decay_rate = 0.0
            half_life = float('inf')
            is_decaying = False
            severity = "minimal"
    else:
        # No significant autocorrelation
        decay_rate = 0.0
        half_life = float('inf')
        is_decaying = False
        severity = "minimal"
    
    return SignalDecayMetrics(
        decay_rate=float(decay_rate),
        half_life_days=float(half_life),
        autocorrelation=ac_series,
        is_decaying=is_decaying,
        decay_severity=severity,
    )


def _compute_trade_quality(
    weights: pd.DataFrame,
    returns: Optional[pd.Series],
    turnover: pd.Series,
) -> TradeQualityMetrics:
    """Assess trade execution quality."""
    if turnover.empty or returns is None or len(returns) < 21:
        return TradeQualityMetrics(
            avg_trade_size=0.0,
            trade_frequency=0.0,
            rebalance_efficiency=0.0,
            churn_ratio=0.0,
            timing_score=50.0,
        )
    
    # Average trade size (avg weight change)
    weight_changes = weights.diff().abs()
    avg_trade_size = float(weight_changes.mean().mean())
    
    # Trade frequency (days with significant turnover)
    trade_days = (turnover > 0.01).sum() / len(turnover)
    
    # Rebalance efficiency: return generated per unit turnover
    # Higher is better - getting more return for less trading
    cumulative_return = (1 + returns).cumprod().iloc[-1] - 1
    cumulative_turnover = turnover.sum()
    rebalance_efficiency = cumulative_return / (cumulative_turnover + 1e-8)
    
    # Churn ratio: proportion of round-trip trades within short period
    # Approximated by measuring how often positions reverse
    position_signs = np.sign(weights)
    sign_changes = (position_signs.diff().abs() > 0).sum(axis=1).mean()
    churn_ratio = sign_changes / (weights.shape[1] + 1e-8)
    
    # Timing score: based on whether returns are positive after trades
    # Simplified: correlation between turnover and next-day returns
    if len(returns) > 1 and len(turnover) > 1:
        aligned_turnover = turnover.reindex(returns.index[:-1])
        next_returns = returns.iloc[1:].reindex(aligned_turnover.index)
        
        valid_mask = aligned_turnover.notna() & next_returns.notna()
        if valid_mask.sum() > 10:
            corr = aligned_turnover[valid_mask].corr(next_returns[valid_mask])
            # Convert correlation to 0-100 score
            timing_score = 50 + (corr * 50 if not np.isnan(corr) else 0)
        else:
            timing_score = 50.0
    else:
        timing_score = 50.0
    
    return TradeQualityMetrics(
        avg_trade_size=float(avg_trade_size),
        trade_frequency=float(trade_days),
        rebalance_efficiency=float(rebalance_efficiency),
        churn_ratio=float(churn_ratio),
        timing_score=float(timing_score),
    )


def _compute_rolling_metrics(
    returns: pd.Series,
    window: int = 63,
) -> pd.DataFrame:
    """Compute various rolling risk-adjusted metrics."""
    if returns is None or len(returns) < window:
        return pd.DataFrame()
    
    metrics = pd.DataFrame(index=returns.index)
    
    # Rolling Sharpe
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    metrics["sharpe"] = (rolling_mean / (rolling_std + 1e-8)) * np.sqrt(252)
    
    # Rolling Sortino
    downside = returns.copy()
    downside[downside > 0] = 0
    rolling_downside_std = downside.rolling(window).std()
    metrics["sortino"] = (rolling_mean / (rolling_downside_std + 1e-8)) * np.sqrt(252)
    
    # Rolling Calmar (return / max drawdown)
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.rolling(window, min_periods=1).max()
    rolling_dd = (cumulative / rolling_max) - 1
    max_dd_rolling = rolling_dd.rolling(window).min().abs()
    annual_return = rolling_mean * 252
    metrics["calmar"] = annual_return / (max_dd_rolling + 1e-8)
    
    # Rolling volatility (annualized)
    metrics["volatility"] = rolling_std * np.sqrt(252)
    
    # Rolling win rate
    metrics["win_rate"] = returns.rolling(window).apply(lambda x: (x > 0).mean())
    
    return metrics.dropna()


def _extract_asset_returns(data: DashboardData) -> Optional[pd.DataFrame]:
    """
    Attempt to extract or compute asset-level returns from available data.
    """
    if data.factor_vectors is not None and not data.factor_vectors.empty:
        if "total_pnl_contrib" in data.factor_vectors.columns:
            pass
    
    if data.budget is not None and not data.budget.empty:
        pass
    
    return None


def _generate_diagnostics_csv(
    turnover_metrics: TurnoverMetrics,
    concentration_metrics: ConcentrationMetrics,
    signal_decay: Optional[SignalDecayMetrics],
    trade_quality: TradeQualityMetrics,
    rolling_metrics: pd.DataFrame,
    alerts: List[Dict],
) -> str:
    """
    Generate CSV export of diagnostic analysis data.
    
    Args:
        turnover_metrics: Turnover analysis results
        concentration_metrics: Concentration metrics
        signal_decay: Signal decay metrics (optional)
        trade_quality: Trade quality metrics
        rolling_metrics: Rolling performance metrics
        alerts: List of diagnostic alerts
        
    Returns:
        CSV string
    """
    csv_parts = []
    
    # Summary metrics
    summary_rows = [
        {"Category": "Turnover", "Metric": "Avg Daily Turnover", "Value": f"{turnover_metrics.avg_daily:.4f}"},
        {"Category": "Turnover", "Metric": "Current Turnover", "Value": f"{turnover_metrics.current:.4f}"},
        {"Category": "Turnover", "Metric": "Annualized Turnover", "Value": f"{turnover_metrics.annualized:.2f}x"},
        {"Category": "Turnover", "Metric": "Est. TC Drag (Annual)", "Value": f"{turnover_metrics.tc_drag_annual:.4f}"},
        {"Category": "Concentration", "Metric": "N Positions", "Value": str(concentration_metrics.current_n_positions)},
        {"Category": "Concentration", "Metric": "Top 5 Conc", "Value": f"{concentration_metrics.current_top5_conc:.4f}"},
        {"Category": "Concentration", "Metric": "Top 10 Conc", "Value": f"{concentration_metrics.current_top10_conc:.4f}"},
        {"Category": "Concentration", "Metric": "HHI", "Value": f"{concentration_metrics.current_hhi:.4f}"},
        {"Category": "Concentration", "Metric": "Effective N", "Value": f"{concentration_metrics.effective_n:.1f}"},
        {"Category": "Trade Quality", "Metric": "Avg Trade Size", "Value": f"{trade_quality.avg_trade_size:.4f}"},
        {"Category": "Trade Quality", "Metric": "Trade Frequency", "Value": f"{trade_quality.trade_frequency:.2f}%"},
        {"Category": "Trade Quality", "Metric": "Timing Score", "Value": f"{trade_quality.timing_score:.1f}"},
        {"Category": "Trade Quality", "Metric": "Churn Rate", "Value": f"{trade_quality.churn_rate:.2f}%"},
    ]
    
    if signal_decay:
        summary_rows.extend([
            {"Category": "Signal Decay", "Metric": "Decay Rate", "Value": f"{signal_decay.decay_rate:.4f}"},
            {"Category": "Signal Decay", "Metric": "Half Life (days)", "Value": f"{signal_decay.half_life_days:.1f}"},
            {"Category": "Signal Decay", "Metric": "Severity", "Value": signal_decay.decay_severity},
        ])
    
    csv_parts.append("# DIAGNOSTIC SUMMARY")
    csv_parts.append(pd.DataFrame(summary_rows).to_csv(index=False))
    
    # Alerts
    if alerts:
        alert_rows = [{"Severity": a["severity"], "Category": a["category"], 
                       "Message": a["message"], "Action": a["action"]} for a in alerts]
        csv_parts.append("\n# ALERTS")
        csv_parts.append(pd.DataFrame(alert_rows).to_csv(index=False))
    
    # Rolling metrics time series
    if not rolling_metrics.empty:
        csv_parts.append("\n# ROLLING METRICS")
        rm_df = rolling_metrics.reset_index()
        rm_df.rename(columns={"index": "Date"}, inplace=True)
        csv_parts.append(rm_df.to_csv(index=False))
    
    # Turnover time series
    csv_parts.append("\n# TURNOVER TIME SERIES")
    to_df = turnover_metrics.series.to_frame(name="turnover").reset_index()
    to_df.rename(columns={"index": "Date"}, inplace=True)
    csv_parts.append(to_df.to_csv(index=False))
    
    return "\n".join(csv_parts)


def _generate_diagnostic_alerts(
    turnover: TurnoverMetrics,
    concentration: ConcentrationMetrics,
    returns: Optional[pd.Series],
    rolling_metrics: pd.DataFrame,
) -> List[Dict]:
    """Generate PM alerts based on diagnostic analysis."""
    alerts = []
    
    # Turnover alerts
    if turnover.annualized > 10:
        alerts.append({
            "severity": "warning",
            "category": "Turnover",
            "message": f"High annualized turnover ({turnover.annualized:.1f}x)",
            "action": f"Est. TC drag: {turnover.tc_drag_annual:.2%}/year. Consider reducing rebalance frequency.",
        })
    
    if turnover.current > turnover.avg_daily * 2:
        alerts.append({
            "severity": "info",
            "category": "Turnover",
            "message": f"Recent turnover spike ({turnover.current:.1%} vs avg {turnover.avg_daily:.1%})",
            "action": "Review recent trades for unusual activity.",
        })
    
    # Concentration alerts
    if concentration.current_hhi > 0.15:
        alerts.append({
            "severity": "warning",
            "category": "Concentration",
            "message": f"High concentration (HHI: {concentration.current_hhi:.3f})",
            "action": f"Effective positions: {concentration.effective_n:.0f}. Consider diversifying.",
        })
    
    if concentration.current_top5_conc > 0.6:
        alerts.append({
            "severity": "alert",
            "category": "Concentration",
            "message": f"Top 5 positions are {concentration.current_top5_conc:.0%} of portfolio",
            "action": "Risk is concentrated in few names.",
        })
    
    if concentration.position_count_trend == "decreasing":
        alerts.append({
            "severity": "info",
            "category": "Concentration",
            "message": "Position count declining",
            "action": "Portfolio becoming more concentrated over time.",
        })
    
    # Performance alerts
    if not rolling_metrics.empty:
        current_sharpe = rolling_metrics["sharpe"].iloc[-1] if "sharpe" in rolling_metrics else 0
        current_vol = rolling_metrics["volatility"].iloc[-1] if "volatility" in rolling_metrics else 0
        
        if current_sharpe < 0:
            alerts.append({
                "severity": "warning",
                "category": "Performance",
                "message": f"Negative rolling Sharpe ({current_sharpe:.2f})",
                "action": "Strategy underperforming on risk-adjusted basis.",
            })
        
        if current_vol > 0.25:
            alerts.append({
                "severity": "info",
                "category": "Risk",
                "message": f"Elevated volatility ({current_vol:.1%} annualized)",
                "action": "Consider position sizing reduction.",
            })
    
    return alerts


# =============================================================================
# Visualization Functions
# =============================================================================

def _create_turnover_chart(turnover: pd.Series, avg: float, height: Optional[int] = None) -> Optional["go.Figure"]:
    """Create enhanced turnover chart."""
    if not PLOTLY_AVAILABLE or turnover.empty:
        return None
    
    fig = go.Figure()
    
    # Turnover line
    fig.add_trace(go.Scatter(
        x=turnover.index,
        y=turnover.values * 100,
        mode="lines",
        name="Daily Turnover",
        line=dict(color=DIAG_COLORS["primary"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(52, 152, 219, 0.2)",
        hovertemplate="Date: %{x}<br>Turnover: %{y:.2f}%<extra></extra>",
    ))
    
    # Average line
    fig.add_hline(y=avg * 100, line_dash="dash", line_color=DIAG_COLORS["warning"], 
                  annotation_text=f"Avg: {avg:.2%}")
    
    # High turnover threshold
    fig.add_hline(y=avg * 200, line_dash="dot", line_color=DIAG_COLORS["danger"],
                  annotation_text="2x Avg")
    
    layout = {
        "title": {"text": "Daily Portfolio Turnover", "font": {"size": 14, "color": "#FAFAFA"}},
        "paper_bgcolor": "#0E1117",
        "plot_bgcolor": "#262730",
        "font": {"color": "#FAFAFA", "size": 11},
        "autosize": True,
        "showlegend": False,
        "xaxis": {"gridcolor": "#3A3A3A"},
        "yaxis": {"title": "Turnover (%)", "gridcolor": "#3A3A3A"},
        "margin": {"l": 60, "r": 40, "t": 50, "b": 40},
    }
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    
    return fig


def _create_concentration_chart(conc_df: pd.DataFrame, height: Optional[int] = None) -> Optional["go.Figure"]:
    """Create concentration metrics chart."""
    if not PLOTLY_AVAILABLE or conc_df.empty:
        return None
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Position Count", "Concentration Metrics"))
    
    # Position count
    fig.add_trace(go.Scatter(
        x=conc_df.index,
        y=conc_df["n_positions"],
        mode="lines",
        name="Positions",
        line=dict(color=DIAG_COLORS["primary"], width=2),
        hovertemplate="Date: %{x}<br>Positions: %{y:.0f}<extra></extra>",
    ), row=1, col=1)
    
    # Concentration metrics
    for col, color in [("top5_concentration", DIAG_COLORS["secondary"]), 
                       ("herfindahl", DIAG_COLORS["warning"])]:
        if col in conc_df.columns:
            multiplier = 100 if "concentration" in col else 1
            fig.add_trace(go.Scatter(
                x=conc_df.index,
                y=conc_df[col] * multiplier,
                mode="lines",
                name="Top 5 %" if "top5" in col else "HHI",
                line=dict(color=color, width=1.5),
            ), row=1, col=2)
    
    layout = {
        "paper_bgcolor": "#0E1117",
        "plot_bgcolor": "#262730",
        "font": {"color": "#FAFAFA", "size": 11},
        "autosize": True,
        "showlegend": True,
        "legend": dict(orientation="h", yanchor="bottom", y=1.02),
        "margin": {"l": 60, "r": 40, "t": 80, "b": 40},
    }
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    
    fig.update_xaxes(gridcolor="#3A3A3A")
    fig.update_yaxes(gridcolor="#3A3A3A")
    
    return fig


def _create_rolling_metrics_chart(metrics_df: pd.DataFrame, height: Optional[int] = None) -> Optional["go.Figure"]:
    """Create rolling risk-adjusted metrics chart."""
    if not PLOTLY_AVAILABLE or metrics_df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Rolling Sharpe", "Rolling Sortino", "Rolling Volatility", "Rolling Win Rate"),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    
    plots = [
        ("sharpe", 1, 1, DIAG_COLORS["primary"]),
        ("sortino", 1, 2, DIAG_COLORS["secondary"]),
        ("volatility", 2, 1, DIAG_COLORS["warning"]),
        ("win_rate", 2, 2, DIAG_COLORS["purple"]),
    ]
    
    for col, row, col_num, color in plots:
        if col in metrics_df.columns:
            y_vals = metrics_df[col].values * (100 if col in ["volatility", "win_rate"] else 1)
            fig.add_trace(go.Scatter(
                x=metrics_df.index,
                y=y_vals,
                mode="lines",
                name=col.title(),
                line=dict(color=color, width=1.5),
                showlegend=False,
            ), row=row, col=col_num)
            
            # Add zero line for Sharpe/Sortino
            if col in ["sharpe", "sortino"]:
                fig.add_hline(y=0, line_dash="dash", line_color="#95a5a6", row=row, col=col_num)
    
    layout = {
        "paper_bgcolor": "#0E1117",
        "plot_bgcolor": "#262730",
        "font": {"color": "#FAFAFA", "size": 10},
        "autosize": True,
        "margin": {"l": 50, "r": 30, "t": 60, "b": 40},
    }
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    
    fig.update_xaxes(gridcolor="#3A3A3A")
    fig.update_yaxes(gridcolor="#3A3A3A")
    
    return fig


def _create_autocorrelation_chart(ac_series: pd.Series, height: Optional[int] = None) -> Optional["go.Figure"]:
    """Create autocorrelation (signal decay) chart."""
    if not PLOTLY_AVAILABLE or ac_series.empty:
        return None
    
    colors = [DIAG_COLORS["secondary"] if v > 0 else DIAG_COLORS["danger"] for v in ac_series.values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=ac_series.index.tolist(),
        y=ac_series.values,
        marker_color=colors,
        hovertemplate="Lag %{x}<br>AC: %{y:.3f}<extra></extra>",
    ))
    
    fig.add_hline(y=0, line_color="#95a5a6")
    
    # Significance bands (approx ±2/sqrt(n))
    fig.add_hline(y=0.1, line_dash="dot", line_color="#95a5a6", annotation_text="Sig.")
    fig.add_hline(y=-0.1, line_dash="dot", line_color="#95a5a6")
    
    layout = {
        "title": {"text": "Return Autocorrelation (Signal Persistence)", "font": {"size": 12, "color": "#FAFAFA"}},
        "paper_bgcolor": "#0E1117",
        "plot_bgcolor": "#262730",
        "font": {"color": "#FAFAFA", "size": 11},
        "autosize": True,
        "xaxis": {"title": "Lag (days)", "gridcolor": "#3A3A3A"},
        "yaxis": {"title": "Autocorrelation", "gridcolor": "#3A3A3A"},
        "margin": {"l": 60, "r": 40, "t": 50, "b": 40},
    }
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    
    return fig


def _create_trade_quality_gauge(score: float, title: str, height: Optional[int] = None) -> Optional["go.Figure"]:
    """Create a gauge for trade quality score."""
    if not PLOTLY_AVAILABLE:
        return None
    
    color = DIAG_COLORS["secondary"] if score >= 60 else DIAG_COLORS["warning"] if score >= 40 else DIAG_COLORS["danger"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 24, "color": "#FAFAFA"}},
        title={"text": title, "font": {"size": 12, "color": "#FAFAFA"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#FAFAFA"},
            "bar": {"color": color},
            "bgcolor": "#262730",
            "borderwidth": 2,
            "bordercolor": "#3A3A3A",
            "steps": [
                {"range": [0, 40], "color": "rgba(231, 76, 60, 0.2)"},
                {"range": [40, 60], "color": "rgba(243, 156, 18, 0.2)"},
                {"range": [60, 100], "color": "rgba(46, 204, 113, 0.2)"},
            ],
        },
    ))
    
    layout = {
        "paper_bgcolor": "#0E1117",
        "font": {"color": "#FAFAFA"},
        "autosize": True,
        "margin": {"l": 20, "r": 20, "t": 40, "b": 20},
    }
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    
    return fig


# =============================================================================
# Main Render Function
# =============================================================================

def render_diagnostics_page(data: DashboardData) -> None:
    """
    Render the enhanced Portfolio Diagnostics page.
    
    Args:
        data: Dashboard data bundle
    """
    # Header with download button placeholder
    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.subheader("🔬 Portfolio Diagnostics")
    
    st.markdown("""
    Comprehensive portfolio health analysis with execution quality metrics,
    signal decay detection, and actionable PM insights.
    """)
    
    if data.weights is None or data.weights.empty:
        st.warning("No weights available")
        st.stop()
    
    # Get return series
    ret_series = select_return_series(data.diag)
    
    # ==========================================================================
    # TC Configuration
    # ==========================================================================
    with st.expander("⚙️ Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            tc_bps = st.number_input(
                "Transaction Cost (bps)",
                min_value=1.0,
                max_value=50.0,
                value=10.0,
                step=1.0,
                help="Estimated cost per trade in basis points",
            )
        with col2:
            rolling_window = st.selectbox(
                "Rolling Window",
                options=[21, 63, 126, 252],
                index=1,
                format_func=lambda x: f"{x} days ({x//21}M)" if x >= 21 else f"{x} days",
            )
    
    # ==========================================================================
    # Compute All Metrics
    # ==========================================================================
    turnover_metrics = _compute_turnover_analysis(data.weights, tc_bps=tc_bps)
    concentration_metrics = _compute_concentration_metrics(data.weights)
    signal_decay = _compute_signal_decay(ret_series, data.weights) if ret_series is not None else None
    trade_quality = _compute_trade_quality(data.weights, ret_series, turnover_metrics.series)
    rolling_metrics = _compute_rolling_metrics(ret_series, window=rolling_window) if ret_series is not None else pd.DataFrame()
    
    # Generate alerts
    alerts = _generate_diagnostic_alerts(turnover_metrics, concentration_metrics, ret_series, rolling_metrics)
    
    # Add download button now that data is computed
    with header_col2:
        csv_data = _generate_diagnostics_csv(
            turnover_metrics, concentration_metrics, signal_decay,
            trade_quality, rolling_metrics, alerts
        )
        st.download_button(
            label="📥 Export",
            data=csv_data,
            file_name="diagnostics.csv",
            mime="text/csv",
            help="Download diagnostic data as CSV",
        )
    
    # ==========================================================================
    # ALERTS SECTION (if any)
    # ==========================================================================
    if alerts:
        st.markdown("### ⚠️ Diagnostic Alerts")
        
        for alert in alerts:
            severity = alert["severity"]
            if severity == "alert":
                st.error(f"🚨 **{alert['category']}**: {alert['message']}")
            elif severity == "warning":
                st.warning(f"⚠️ **{alert['category']}**: {alert['message']}")
            else:
                st.info(f"ℹ️ **{alert['category']}**: {alert['message']}")
            st.caption(f"→ {alert['action']}")
        
        st.divider()
    
    # ==========================================================================
    # EXECUTIVE SUMMARY
    # ==========================================================================
    st.markdown("### 📊 Executive Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Positions",
            f"{concentration_metrics.current_n_positions}",
            delta=concentration_metrics.position_count_trend,
        )
    
    with col2:
        tc_drag_pct = turnover_metrics.tc_drag_annual * 100
        st.metric(
            "Est. TC Drag",
            f"{tc_drag_pct:.1f}%/yr",
            delta=f"{turnover_metrics.annualized:.1f}x turnover",
            delta_color="inverse",
        )
    
    with col3:
        if not rolling_metrics.empty and "sharpe" in rolling_metrics.columns:
            current_sharpe = rolling_metrics["sharpe"].iloc[-1]
            st.metric(
                "Rolling Sharpe",
                f"{current_sharpe:.2f}",
                delta="Good" if current_sharpe > 1 else "Weak" if current_sharpe < 0.5 else None,
            )
        else:
            st.metric("Rolling Sharpe", "N/A")
    
    with col4:
        st.metric(
            "Concentration",
            f"{concentration_metrics.current_hhi:.3f}",
            delta=f"Eff. N={concentration_metrics.effective_n:.0f}",
        )
    
    with col5:
        timing_color = "normal" if trade_quality.timing_score >= 50 else "inverse"
        st.metric(
            "Trade Timing",
            f"{trade_quality.timing_score:.0f}/100",
            delta="Good" if trade_quality.timing_score > 55 else "Poor" if trade_quality.timing_score < 45 else None,
            delta_color=timing_color,
        )
    
    st.divider()
    
    # ==========================================================================
    # TURNOVER ANALYSIS
    # ==========================================================================
    st.markdown("### 📈 Turnover Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if PLOTLY_AVAILABLE:
            fig = _create_turnover_chart(turnover_metrics.series.tail(500), turnover_metrics.avg_daily)
            if fig:
                st.plotly_chart(fig)
        else:
            st.line_chart(turnover_metrics.series.tail(500), height=280)
    
    with col2:
        st.markdown("**Turnover Metrics**")
        
        st.metric("Avg Daily", f"{turnover_metrics.avg_daily:.2%}")
        st.metric("Current", f"{turnover_metrics.current:.2%}")
        st.metric("Annualized", f"{turnover_metrics.annualized:.1f}x")
        
        st.divider()
        
        st.markdown("**Cost Estimation**")
        st.metric("TC Rate", f"{turnover_metrics.estimated_tc_bps:.0f} bps")
        st.metric("Annual Drag", f"{turnover_metrics.tc_drag_annual:.2%}")
    
    st.divider()
    
    # ==========================================================================
    # CONCENTRATION ANALYSIS
    # ==========================================================================
    st.markdown("### 🎯 Concentration Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if PLOTLY_AVAILABLE:
            fig = _create_concentration_chart(concentration_metrics.time_series.tail(500))
            if fig:
                st.plotly_chart(fig)
        else:
            conc_cols = ["n_positions", "top5_concentration", "herfindahl"]
            st.line_chart(concentration_metrics.time_series[conc_cols].tail(500), height=280)
    
    with col2:
        st.markdown("**Current Snapshot**")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Top 5", f"{concentration_metrics.current_top5_conc:.0%}")
        with col_b:
            st.metric("Top 10", f"{concentration_metrics.current_top10_conc:.0%}")
        
        st.metric("HHI", f"{concentration_metrics.current_hhi:.4f}")
        st.metric("Effective N", f"{concentration_metrics.effective_n:.1f}")
        st.metric("Max Position", f"{concentration_metrics.max_position_weight:.1%}")
    
    st.divider()
    
    # ==========================================================================
    # ROLLING RISK-ADJUSTED METRICS
    # ==========================================================================
    st.markdown("### 📉 Rolling Risk-Adjusted Performance")
    
    if not rolling_metrics.empty:
        col1, col2 = st.columns([2.5, 1])
        
        with col1:
            if PLOTLY_AVAILABLE:
                fig = _create_rolling_metrics_chart(rolling_metrics.tail(500))
                if fig:
                    st.plotly_chart(fig)
            else:
                st.line_chart(rolling_metrics[["sharpe", "sortino"]].tail(500), height=300)
        
        with col2:
            st.markdown("**Current Values**")
            
            for metric in ["sharpe", "sortino", "calmar"]:
                if metric in rolling_metrics.columns:
                    val = rolling_metrics[metric].iloc[-1]
                    avg = rolling_metrics[metric].mean()
                    delta = f"vs avg {avg:.2f}"
                    st.metric(metric.title(), f"{val:.2f}", delta=delta)
            
            if "volatility" in rolling_metrics.columns:
                vol = rolling_metrics["volatility"].iloc[-1]
                st.metric("Volatility", f"{vol:.1%}")
            
            if "win_rate" in rolling_metrics.columns:
                wr = rolling_metrics["win_rate"].iloc[-1]
                st.metric("Win Rate", f"{wr:.0%}")
    else:
        st.info("Insufficient return data for rolling metrics")
    
    st.divider()
    
    # ==========================================================================
    # SIGNAL DECAY & TRADE QUALITY
    # ==========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔋 Signal Decay Analysis")
        
        if signal_decay is not None and not signal_decay.autocorrelation.empty:
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | Decay Severity | **{signal_decay.decay_severity.title()}** |
            | Half-Life | **{signal_decay.half_life_days:.1f} days** |
            | Is Decaying | **{'Yes' if signal_decay.is_decaying else 'No'}** |
            """)
            
            if PLOTLY_AVAILABLE:
                fig = _create_autocorrelation_chart(signal_decay.autocorrelation)
                if fig:
                    st.plotly_chart(fig)
            else:
                st.bar_chart(signal_decay.autocorrelation, height=200)
        else:
            st.info("Insufficient data for signal decay analysis")
    
    with col2:
        st.markdown("### ⚡ Trade Quality Assessment")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if PLOTLY_AVAILABLE:
                fig = _create_trade_quality_gauge(trade_quality.timing_score, "Trade Timing")
                if fig:
                    st.plotly_chart(fig)
            else:
                st.metric("Timing Score", f"{trade_quality.timing_score:.0f}/100")
        
        with col_b:
            efficiency_score = min(max(trade_quality.rebalance_efficiency * 100 + 50, 0), 100)
            if PLOTLY_AVAILABLE:
                fig = _create_trade_quality_gauge(efficiency_score, "Efficiency")
                if fig:
                    st.plotly_chart(fig)
            else:
                st.metric("Efficiency", f"{efficiency_score:.0f}/100")
        
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Avg Trade Size | {trade_quality.avg_trade_size:.2%} |
        | Trade Frequency | {trade_quality.trade_frequency:.0%} of days |
        | Churn Ratio | {trade_quality.churn_ratio:.2%} |
        """)
    
    st.divider()
    
    # ==========================================================================
    # HIT RATE ANALYSIS (if asset returns available)
    # ==========================================================================
    st.markdown("### 🎯 Hit Rate & Sizing Analysis")
    
    asset_returns = _extract_asset_returns(data)
    
    if asset_returns is not None and not asset_returns.empty:
        weights_df = data.weights
        common_dates = weights_df.index.intersection(asset_returns.index)
        common_assets = weights_df.columns.intersection(asset_returns.columns)
        
        if len(common_dates) > 20 and len(common_assets) > 5:
            w_aligned = weights_df.loc[common_dates, common_assets]
            r_aligned = asset_returns.loc[common_dates, common_assets]
            
            hit_rate_series = hit_rate(w_aligned, r_aligned)
            wl_sizing = winner_loser_sizing(w_aligned, r_aligned)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Hit Rate Over Time**")
                st.line_chart(hit_rate_series.tail(500), height=300)
            with col2:
                st.markdown("**Winner vs Loser Sizing**")
                st.line_chart(
                    wl_sizing[["avg_winner_weight", "avg_loser_weight"]].tail(500),
                    height=300
                )
        else:
            st.info("Insufficient overlapping data for hit rate analysis")
    else:
        st.info("""
        **Hit rate analysis requires asset-level returns data.**
        
        This feature will be available when:
        - Asset returns are stored during strategy execution
        - Or price data is loaded alongside weights
        
        Currently showing turnover, concentration, and signal metrics instead.
        """)
    
    st.divider()
    
    # ==========================================================================
    # DIAGNOSTICS DATA TABLE
    # ==========================================================================
    with st.expander("📋 Raw Diagnostics Data", expanded=False):
        tabs = st.tabs(["Diagnostics", "Turnover", "Concentration"])
        
        with tabs[0]:
            if data.diag is not None and not data.diag.empty:
                st.dataframe(data.diag.tail(100), width='stretch', height=400)
            else:
                st.info("No diagnostics data available")
        
        with tabs[1]:
            if not turnover_metrics.series.empty:
                turnover_df = turnover_metrics.series.to_frame("turnover")
                turnover_df["cumulative"] = turnover_df["turnover"].cumsum()
                st.dataframe(turnover_df.tail(100), width='stretch', height=400)
        
        with tabs[2]:
            if not concentration_metrics.time_series.empty:
                st.dataframe(concentration_metrics.time_series.tail(100), width='stretch', height=400)
