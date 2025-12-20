"""
PM Insights Generator

Provides actionable insights and alerts for portfolio managers:
- Risk warnings and alerts
- Performance anomaly detection
- Factor exposure drift alerts
- Concentration warnings
- Regime change notifications
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class InsightSeverity(Enum):
    """Severity levels for insights."""
    INFO = "info"
    WARNING = "warning"
    ALERT = "alert"
    CRITICAL = "critical"


class InsightCategory(Enum):
    """Categories of insights."""
    RISK = "risk"
    PERFORMANCE = "performance"
    EXPOSURE = "exposure"
    CONCENTRATION = "concentration"
    FACTOR = "factor"
    REGIME = "regime"
    TRADING = "trading"


@dataclass
class PMInsight:
    """A single PM insight/alert."""
    title: str
    message: str
    severity: InsightSeverity
    category: InsightCategory
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    recommendation: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "recommendation": self.recommendation,
        }


class PMInsightEngine:
    """
    Engine for generating PM insights from portfolio data.
    
    Analyzes various aspects of the portfolio and generates
    actionable insights with severity levels.
    """
    
    # Default thresholds (can be customized)
    DEFAULT_THRESHOLDS = {
        "max_drawdown_warning": -0.10,
        "max_drawdown_alert": -0.15,
        "max_drawdown_critical": -0.20,
        "volatility_high": 0.20,
        "volatility_very_high": 0.30,
        "sharpe_low": 0.5,
        "sharpe_very_low": 0.0,
        "concentration_top5": 0.50,
        "concentration_top10": 0.70,
        "turnover_high": 0.10,
        "turnover_very_high": 0.20,
        "factor_drift": 1.5,  # Z-score
        "position_count_low": 20,
        "position_count_very_low": 10,
        "gross_exposure_high": 2.0,
        "net_exposure_drift": 0.20,
        "correlation_warning": 0.7,
        "var_breach_pct": 0.05,
    }
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """Initialize with optional custom thresholds."""
        self.thresholds = {**self.DEFAULT_THRESHOLDS}
        if thresholds:
            self.thresholds.update(thresholds)
        self.insights: List[PMInsight] = []
    
    def clear(self):
        """Clear all insights."""
        self.insights = []
    
    def analyze_portfolio(
        self,
        weights: pd.DataFrame,
        returns: Optional[pd.Series] = None,
        diag: Optional[pd.DataFrame] = None,
        exposure: Optional[pd.DataFrame] = None,
    ) -> List[PMInsight]:
        """
        Run full portfolio analysis and generate insights.
        
        Args:
            weights: Portfolio weights (time x asset)
            returns: Portfolio returns (optional)
            diag: Portfolio diagnostics (optional)
            exposure: Factor exposures (optional)
            
        Returns:
            List of PMInsight objects
        """
        self.clear()
        
        # Risk analysis
        if returns is not None:
            self._analyze_returns(returns)
        
        if diag is not None:
            self._analyze_diagnostics(diag)
        
        # Concentration analysis
        self._analyze_concentration(weights)
        
        # Exposure analysis
        if exposure is not None:
            self._analyze_factor_exposure(exposure)
        
        # Trading analysis
        self._analyze_trading(weights)
        
        return self.insights
    
    def _add_insight(
        self,
        title: str,
        message: str,
        severity: InsightSeverity,
        category: InsightCategory,
        metric_value: Optional[float] = None,
        threshold: Optional[float] = None,
        recommendation: Optional[str] = None,
    ):
        """Add an insight to the list."""
        self.insights.append(PMInsight(
            title=title,
            message=message,
            severity=severity,
            category=category,
            metric_value=metric_value,
            threshold=threshold,
            recommendation=recommendation,
        ))
    
    def _analyze_returns(self, returns: pd.Series):
        """Analyze return series for insights."""
        if returns.empty or len(returns) < 20:
            return
        
        # Drawdown analysis
        equity = (1 + returns).cumprod()
        peak = equity.cummax()
        drawdown = (equity / peak - 1).iloc[-1]
        max_dd = (equity / peak - 1).min()
        
        if drawdown < self.thresholds["max_drawdown_critical"]:
            self._add_insight(
                "Critical Drawdown",
                f"Portfolio is in {drawdown:.1%} drawdown, approaching critical levels.",
                InsightSeverity.CRITICAL,
                InsightCategory.RISK,
                metric_value=drawdown,
                threshold=self.thresholds["max_drawdown_critical"],
                recommendation="Consider reducing risk exposure or hedging tail risk.",
            )
        elif drawdown < self.thresholds["max_drawdown_alert"]:
            self._add_insight(
                "Elevated Drawdown",
                f"Current drawdown of {drawdown:.1%} exceeds alert threshold.",
                InsightSeverity.ALERT,
                InsightCategory.RISK,
                metric_value=drawdown,
                threshold=self.thresholds["max_drawdown_alert"],
                recommendation="Monitor closely and prepare contingency plans.",
            )
        elif drawdown < self.thresholds["max_drawdown_warning"]:
            self._add_insight(
                "Drawdown Warning",
                f"Portfolio drawdown at {drawdown:.1%}.",
                InsightSeverity.WARNING,
                InsightCategory.RISK,
                metric_value=drawdown,
                threshold=self.thresholds["max_drawdown_warning"],
            )
        
        # Volatility analysis
        vol = returns.std() * np.sqrt(252)
        
        if vol > self.thresholds["volatility_very_high"]:
            self._add_insight(
                "Very High Volatility",
                f"Annualized volatility of {vol:.1%} is significantly elevated.",
                InsightSeverity.ALERT,
                InsightCategory.RISK,
                metric_value=vol,
                threshold=self.thresholds["volatility_very_high"],
                recommendation="Consider volatility targeting or position sizing reduction.",
            )
        elif vol > self.thresholds["volatility_high"]:
            self._add_insight(
                "Elevated Volatility",
                f"Annualized volatility at {vol:.1%}.",
                InsightSeverity.WARNING,
                InsightCategory.RISK,
                metric_value=vol,
                threshold=self.thresholds["volatility_high"],
            )
        
        # Sharpe analysis (rolling 252 days)
        if len(returns) >= 252:
            ann_ret = returns.mean() * 252
            sharpe = ann_ret / (vol + 1e-12)
            
            if sharpe < self.thresholds["sharpe_very_low"]:
                self._add_insight(
                    "Negative Risk-Adjusted Return",
                    f"Sharpe ratio of {sharpe:.2f} indicates poor risk-adjusted performance.",
                    InsightSeverity.ALERT,
                    InsightCategory.PERFORMANCE,
                    metric_value=sharpe,
                    threshold=self.thresholds["sharpe_very_low"],
                    recommendation="Review factor exposures and signal quality.",
                )
            elif sharpe < self.thresholds["sharpe_low"]:
                self._add_insight(
                    "Below Target Sharpe",
                    f"Current Sharpe ratio of {sharpe:.2f} is below target.",
                    InsightSeverity.WARNING,
                    InsightCategory.PERFORMANCE,
                    metric_value=sharpe,
                    threshold=self.thresholds["sharpe_low"],
                )
        
        # Recent performance
        if len(returns) >= 5:
            last_5d = returns.tail(5).sum()
            if last_5d < -0.05:
                self._add_insight(
                    "Recent Underperformance",
                    f"Portfolio down {last_5d:.1%} over last 5 days.",
                    InsightSeverity.WARNING,
                    InsightCategory.PERFORMANCE,
                    metric_value=last_5d,
                )
        
        # VaR breach check
        var_95 = np.percentile(returns, 5)
        recent_breaches = (returns.tail(21) < var_95).sum()
        expected_breaches = 21 * 0.05
        
        if recent_breaches > expected_breaches * 2:
            self._add_insight(
                "VaR Breaches Elevated",
                f"{recent_breaches} VaR breaches in last 21 days (expected ~{expected_breaches:.0f}).",
                InsightSeverity.ALERT,
                InsightCategory.RISK,
                metric_value=recent_breaches,
                threshold=expected_breaches,
                recommendation="Tail risk may be underestimated. Review risk model.",
            )
    
    def _analyze_diagnostics(self, diag: pd.DataFrame):
        """Analyze diagnostics DataFrame for insights."""
        if diag.empty:
            return
        
        latest = diag.iloc[-1]
        
        # Gross exposure
        if "gross_exposure" in diag.columns:
            gross = latest["gross_exposure"]
            if gross > self.thresholds["gross_exposure_high"]:
                self._add_insight(
                    "High Gross Exposure",
                    f"Gross exposure at {gross:.1%} exceeds normal levels.",
                    InsightSeverity.WARNING,
                    InsightCategory.EXPOSURE,
                    metric_value=gross,
                    threshold=self.thresholds["gross_exposure_high"],
                )
        
        # Net exposure drift
        if "net_exposure" in diag.columns and len(diag) > 20:
            net = latest["net_exposure"]
            avg_net = diag["net_exposure"].tail(60).mean()
            drift = abs(net - avg_net)
            
            if drift > self.thresholds["net_exposure_drift"]:
                self._add_insight(
                    "Net Exposure Drift",
                    f"Net exposure ({net:.1%}) has drifted from recent average ({avg_net:.1%}).",
                    InsightSeverity.INFO,
                    InsightCategory.EXPOSURE,
                    metric_value=drift,
                    threshold=self.thresholds["net_exposure_drift"],
                )
        
        # Position count
        if "n_positions" in diag.columns:
            n_pos = int(latest["n_positions"])
            if n_pos < self.thresholds["position_count_very_low"]:
                self._add_insight(
                    "Very Concentrated Portfolio",
                    f"Only {n_pos} positions - concentration risk is high.",
                    InsightSeverity.ALERT,
                    InsightCategory.CONCENTRATION,
                    metric_value=n_pos,
                    threshold=self.thresholds["position_count_very_low"],
                    recommendation="Consider adding positions for diversification.",
                )
            elif n_pos < self.thresholds["position_count_low"]:
                self._add_insight(
                    "Low Position Count",
                    f"Portfolio has {n_pos} positions.",
                    InsightSeverity.WARNING,
                    InsightCategory.CONCENTRATION,
                    metric_value=n_pos,
                    threshold=self.thresholds["position_count_low"],
                )
        
        # Turnover
        if "turnover" in diag.columns and len(diag) > 5:
            recent_turnover = diag["turnover"].tail(5).mean()
            if recent_turnover > self.thresholds["turnover_very_high"]:
                self._add_insight(
                    "Very High Turnover",
                    f"Recent turnover averaging {recent_turnover:.1%} daily.",
                    InsightSeverity.ALERT,
                    InsightCategory.TRADING,
                    metric_value=recent_turnover,
                    threshold=self.thresholds["turnover_very_high"],
                    recommendation="High turnover may erode returns through costs.",
                )
            elif recent_turnover > self.thresholds["turnover_high"]:
                self._add_insight(
                    "Elevated Turnover",
                    f"Daily turnover at {recent_turnover:.1%}.",
                    InsightSeverity.WARNING,
                    InsightCategory.TRADING,
                    metric_value=recent_turnover,
                    threshold=self.thresholds["turnover_high"],
                )
    
    def _analyze_concentration(self, weights: pd.DataFrame):
        """Analyze position concentration."""
        if weights.empty:
            return
        
        w_last = weights.iloc[-1]
        w_abs = w_last.abs()
        w_sorted = w_abs.sort_values(ascending=False)
        
        gross = w_abs.sum()
        if gross < 1e-12:
            return
        
        # Top 5 concentration
        top5_conc = w_sorted.head(5).sum() / gross
        if top5_conc > self.thresholds["concentration_top5"]:
            top5_names = ", ".join(w_sorted.head(5).index.tolist())
            self._add_insight(
                "High Top-5 Concentration",
                f"Top 5 positions represent {top5_conc:.1%} of gross exposure.",
                InsightSeverity.WARNING,
                InsightCategory.CONCENTRATION,
                metric_value=top5_conc,
                threshold=self.thresholds["concentration_top5"],
                recommendation=f"Consider reducing: {top5_names}",
            )
        
        # Top 10 concentration
        top10_conc = w_sorted.head(10).sum() / gross
        if top10_conc > self.thresholds["concentration_top10"]:
            self._add_insight(
                "High Top-10 Concentration",
                f"Top 10 positions represent {top10_conc:.1%} of gross exposure.",
                InsightSeverity.INFO,
                InsightCategory.CONCENTRATION,
                metric_value=top10_conc,
                threshold=self.thresholds["concentration_top10"],
            )
        
        # Single position dominance
        if len(w_sorted) > 0:
            largest_pos = w_sorted.iloc[0] / gross
            if largest_pos > 0.15:
                self._add_insight(
                    "Single Position Dominance",
                    f"Largest position ({w_sorted.index[0]}) is {largest_pos:.1%} of portfolio.",
                    InsightSeverity.WARNING,
                    InsightCategory.CONCENTRATION,
                    metric_value=largest_pos,
                    threshold=0.15,
                )
    
    def _analyze_factor_exposure(self, exposure: pd.DataFrame):
        """Analyze factor exposure for drift and extremes."""
        if exposure.empty or len(exposure) < 20:
            return
        
        latest = exposure.iloc[-1]
        
        for factor in exposure.columns:
            series = exposure[factor].dropna()
            if len(series) < 20:
                continue
            
            current = latest[factor]
            mean = series.tail(60).mean()
            std = series.tail(60).std()
            
            if std > 1e-12:
                z_score = (current - mean) / std
                
                if abs(z_score) > self.thresholds["factor_drift"]:
                    direction = "above" if z_score > 0 else "below"
                    self._add_insight(
                        f"Factor Drift: {factor}",
                        f"{factor} exposure ({current:.2f}) is {abs(z_score):.1f} std {direction} recent average.",
                        InsightSeverity.WARNING if abs(z_score) > 2 else InsightSeverity.INFO,
                        InsightCategory.FACTOR,
                        metric_value=z_score,
                        threshold=self.thresholds["factor_drift"],
                    )
    
    def _analyze_trading(self, weights: pd.DataFrame):
        """Analyze trading patterns."""
        if weights.empty or len(weights) < 2:
            return
        
        # Calculate turnover
        turnover = weights.diff().abs().sum(axis=1)
        
        if len(turnover) >= 5:
            recent_turnover = turnover.tail(5).mean()
            historical_turnover = turnover.mean()
            
            if recent_turnover > historical_turnover * 2:
                self._add_insight(
                    "Unusual Trading Activity",
                    f"Recent turnover ({recent_turnover:.1%}) is 2x historical average.",
                    InsightSeverity.INFO,
                    InsightCategory.TRADING,
                    metric_value=recent_turnover,
                    threshold=historical_turnover * 2,
                )
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary counts by severity."""
        summary = {s.value: 0 for s in InsightSeverity}
        for insight in self.insights:
            summary[insight.severity.value] += 1
        return summary
    
    def get_by_category(self, category: InsightCategory) -> List[PMInsight]:
        """Filter insights by category."""
        return [i for i in self.insights if i.category == category]
    
    def get_by_severity(self, severity: InsightSeverity) -> List[PMInsight]:
        """Filter insights by severity."""
        return [i for i in self.insights if i.severity == severity]
    
    def get_critical_alerts(self) -> List[PMInsight]:
        """Get only critical and alert level insights."""
        return [i for i in self.insights 
                if i.severity in (InsightSeverity.CRITICAL, InsightSeverity.ALERT)]


def render_insights_panel(insights: List[PMInsight], streamlit_module):
    """
    Render insights panel in Streamlit.
    
    Args:
        insights: List of PMInsight objects
        streamlit_module: The streamlit module (st)
    """
    st = streamlit_module
    
    if not insights:
        st.success("✅ No alerts or warnings. Portfolio looks healthy.")
        return
    
    # Group by severity
    critical = [i for i in insights if i.severity == InsightSeverity.CRITICAL]
    alerts = [i for i in insights if i.severity == InsightSeverity.ALERT]
    warnings = [i for i in insights if i.severity == InsightSeverity.WARNING]
    infos = [i for i in insights if i.severity == InsightSeverity.INFO]
    
    # Critical alerts
    for insight in critical:
        with st.container():
            st.error(f"🚨 **{insight.title}**")
            st.markdown(insight.message)
            if insight.recommendation:
                st.markdown(f"💡 *{insight.recommendation}*")
    
    # Alerts
    for insight in alerts:
        with st.container():
            st.warning(f"⚠️ **{insight.title}**")
            st.markdown(insight.message)
            if insight.recommendation:
                st.markdown(f"💡 *{insight.recommendation}*")
    
    # Warnings (collapsible)
    if warnings:
        with st.expander(f"📋 Warnings ({len(warnings)})", expanded=False):
            for insight in warnings:
                st.markdown(f"**{insight.title}**: {insight.message}")
                if insight.recommendation:
                    st.caption(f"💡 {insight.recommendation}")
    
    # Info (collapsible)
    if infos:
        with st.expander(f"ℹ️ Information ({len(infos)})", expanded=False):
            for insight in infos:
                st.markdown(f"**{insight.title}**: {insight.message}")


def generate_executive_summary(
    insights: List[PMInsight],
    perf: Dict[str, float],
) -> str:
    """
    Generate an executive summary for PM review.
    
    Args:
        insights: List of insights
        perf: Performance metrics dictionary
        
    Returns:
        Formatted summary string
    """
    summary_parts = []
    
    # Performance summary
    summary_parts.append("## Performance Summary\n")
    summary_parts.append(f"- **Annual Return**: {perf.get('annual_return', 0):.2%}")
    summary_parts.append(f"- **Sharpe Ratio**: {perf.get('sharpe', 0):.2f}")
    summary_parts.append(f"- **Max Drawdown**: {perf.get('max_drawdown', 0):.2%}")
    summary_parts.append(f"- **Current Drawdown**: {perf.get('current_drawdown', 0):.2%}\n")
    
    # Risk summary
    n_critical = len([i for i in insights if i.severity == InsightSeverity.CRITICAL])
    n_alerts = len([i for i in insights if i.severity == InsightSeverity.ALERT])
    
    summary_parts.append("## Risk Status\n")
    
    if n_critical > 0:
        summary_parts.append(f"🚨 **{n_critical} CRITICAL ISSUES** require immediate attention.\n")
    elif n_alerts > 0:
        summary_parts.append(f"⚠️ **{n_alerts} alerts** require review.\n")
    else:
        summary_parts.append("✅ No critical issues detected.\n")
    
    # Key insights
    if insights:
        summary_parts.append("## Key Observations\n")
        for insight in insights[:5]:  # Top 5 insights
            icon = {"critical": "🚨", "alert": "⚠️", "warning": "📋", "info": "ℹ️"}.get(
                insight.severity.value, ""
            )
            summary_parts.append(f"- {icon} **{insight.title}**: {insight.message}")
    
    return "\n".join(summary_parts)
