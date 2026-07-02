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
    PREDICTIVE = "predictive"  # New: ML-based predictions
    OPTIMIZATION = "optimization"  # New: Portfolio optimization opportunities
    MARKET_TIMING = "timing"  # New: Market timing signals
    STRUCTURAL = "structural"  # New: Structural changes in portfolio


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

    def analyze_predictive_signals(
        self,
        returns: pd.Series,
        features: Optional[pd.DataFrame] = None,
        forecast_horizon: int = 21,
    ) -> List[PMInsight]:
        """
        Generate predictive analytics insights using ML forecasting.
        """
        self.clear()

        if len(returns) < 100:
            return self.insights

        try:
            from q23.dashboard.analytics.advanced import PredictiveReturnForecaster

            forecaster = PredictiveReturnForecaster()

            # Prepare features if not provided
            if features is None:
                # Use simple lagged returns as features
                features = pd.DataFrame({
                    f'lag_{i}d': returns.shift(i) for i in [1, 2, 3, 5, 10, 21]
                }).dropna()

            # Align data
            common_idx = features.index.intersection(returns.index)
            features_aligned = features.loc[common_idx]
            returns_aligned = returns.loc[common_idx]

            # Train models
            results = forecaster.train_models(features_aligned, returns_aligned)

            # Analyze model performance
            ensemble_r2 = (results['rf']['test_r2'] + results['gb']['test_r2']) / 2

            if ensemble_r2 > 0.1:  # Decent predictive power
                self._add_insight(
                    "Strong Predictive Signals Detected",
                    f"ML models show {ensemble_r2:.1%} R² in out-of-sample testing, indicating predictive power for {forecast_horizon}-day returns.",
                    InsightSeverity.INFO,
                    InsightCategory.PREDICTIVE,
                    ensemble_r2,
                    recommendation="Consider incorporating these signals into portfolio positioning."
                )
            elif ensemble_r2 > 0.05:
                self._add_insight(
                    "Moderate Predictive Signals",
                    f"ML models show {ensemble_r2:.1%} R², suggesting some predictive information available.",
                    InsightSeverity.INFO,
                    InsightCategory.PREDICTIVE,
                    ensemble_r2,
                )
            else:
                self._add_insight(
                    "Weak Predictive Signals",
                    f"ML models show only {ensemble_r2:.1%} R², indicating limited predictive power.",
                    InsightSeverity.WARNING,
                    InsightCategory.PREDICTIVE,
                    ensemble_r2,
                )

            # Analyze feature importance
            if forecaster.feature_importance:
                top_features = {}
                for model_name, importance_dict in forecaster.feature_importance.items():
                    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                    top_features[model_name] = sorted_features[:3]

                # Check for dominant factors
                rf_top = top_features.get('rf', [])
                if rf_top and rf_top[0][1] > 0.3:
                    self._add_insight(
                        f"Dominant Factor: {rf_top[0][0]}",
                        f"Feature {rf_top[0][0]} explains {rf_top[0][1]:.1%} of predictive power.",
                        InsightSeverity.INFO,
                        InsightCategory.PREDICTIVE,
                        rf_top[0][1],
                    )

        except Exception as e:
            self._add_insight(
                "Predictive Analysis Error",
                f"Could not complete ML forecasting analysis: {str(e)}",
                InsightSeverity.WARNING,
                InsightCategory.PREDICTIVE,
            )

        return self.insights

    def analyze_portfolio_optimization_opportunities(
        self,
        weights: pd.DataFrame,
        returns: pd.Series,
        covariance: Optional[pd.DataFrame] = None,
    ) -> List[PMInsight]:
        """
        Identify portfolio optimization opportunities.
        """
        self.clear()

        try:
            from q23.dashboard.analytics.risk_analytics import compute_beta_analysis

            # Calculate current portfolio characteristics
            w_current = weights.iloc[-1]
            active_positions = w_current[w_current.abs() > 1e-4]

            # Concentration analysis
            top5_concentration = active_positions.abs().nlargest(5).sum()
            herfindahl = (active_positions ** 2).sum()

            if top5_concentration > 0.5:
                self._add_insight(
                    "High Concentration Risk",
                    f"Top 5 positions represent {top5_concentration:.1%} of portfolio, increasing idiosyncratic risk.",
                    InsightSeverity.WARNING,
                    InsightCategory.OPTIMIZATION,
                    top5_concentration,
                    0.5,
                    recommendation="Consider diversifying across more positions or sectors."
                )

            if herfindahl > 0.15:
                self._add_insight(
                    "Extreme Concentration",
                    f"Herfindahl index of {herfindahl:.1%} indicates very concentrated portfolio.",
                    InsightSeverity.ALERT,
                    InsightCategory.OPTIMIZATION,
                    herfindahl,
                    0.15,
                    recommendation="Rebalance to reduce single-position concentration risk."
                )

            # Risk-adjusted return opportunities
            if covariance is not None:
                # Calculate risk contributions
                port_vol = np.sqrt(w_current.T @ covariance @ w_current)
                marginal_contrib = covariance @ w_current
                risk_contrib = w_current * marginal_contrib

                # Find positions with high risk contribution relative to weight
                risk_weight_ratio = risk_contrib / w_current
                high_risk_positions = risk_weight_ratio[risk_weight_ratio > risk_weight_ratio.quantile(0.9)]

                if len(high_risk_positions) > 0:
                    top_risk_pos = high_risk_positions.idxmax()
                    self._add_insight(
                        f"High Risk Contribution: {top_risk_pos}",
                        f"Position contributes disproportionately to portfolio risk ({high_risk_positions.max():.1%} risk per unit weight).",
                        InsightSeverity.WARNING,
                        InsightCategory.OPTIMIZATION,
                        high_risk_positions.max(),
                        recommendation="Consider reducing position size or hedging."
                    )

            # Turnover analysis for transaction costs
            if len(weights) > 1:
                turnover = (weights - weights.shift(1).fillna(0)).abs().sum(axis=1).mean()
                if turnover > 0.05:  # 5% monthly turnover
                    self._add_insight(
                        "High Portfolio Turnover",
                        f"Average monthly turnover of {turnover:.1%} may increase transaction costs significantly.",
                        InsightSeverity.WARNING,
                        InsightCategory.OPTIMIZATION,
                        turnover,
                        0.05,
                        recommendation="Review rebalancing frequency and transaction cost impact."
                    )

        except Exception as e:
            self._add_insight(
                "Optimization Analysis Error",
                f"Could not complete optimization analysis: {str(e)}",
                InsightSeverity.WARNING,
                InsightCategory.OPTIMIZATION,
            )

        return self.insights

    def analyze_market_timing_signals(
        self,
        returns: pd.Series,
        regime_data: Optional[pd.DataFrame] = None,
        macro_indicators: Optional[pd.DataFrame] = None,
    ) -> List[PMInsight]:
        """
        Analyze market timing and regime-based signals.
        """
        self.clear()

        try:
            from q23.dashboard.analytics.advanced import RegimeDetector

            # Detect market regimes if not provided
            if regime_data is None:
                detector = RegimeDetector(returns)
                regime_data = detector.detect_regimes()

            current_regime = regime_data.iloc[-1]['regime']
            regime_confidence = len(regime_data[regime_data['regime'] == current_regime]) / len(regime_data)

            # Regime transition analysis
            regime_changes = regime_data['regime'] != regime_data['regime'].shift(1)
            recent_changes = regime_changes.tail(21).sum()  # Last ~1 month

            if recent_changes > 2:
                self._add_insight(
                    "Regime Instability",
                    f"Detected {recent_changes} regime changes in past 21 days, indicating market uncertainty.",
                    InsightSeverity.WARNING,
                    InsightCategory.TIMING,
                    recent_changes,
                    recommendation="Monitor closely for potential regime shift."
                )

            # Current regime assessment
            regime_characteristics = {
                'crisis': {'volatility': 'high', 'returns': 'negative', 'action': 'defensive'},
                'normal': {'volatility': 'moderate', 'returns': 'neutral', 'action': 'balanced'},
                'trend': {'volatility': 'moderate', 'returns': 'positive', 'action': 'aggressive'},
                'calm': {'volatility': 'low', 'returns': 'neutral', 'action': 'risk-on'},
            }

            if current_regime in regime_characteristics:
                chars = regime_characteristics[current_regime]
                confidence_level = "high" if regime_confidence > 0.7 else "moderate" if regime_confidence > 0.5 else "low"

                self._add_insight(
                    f"Current Market Regime: {current_regime.title()}",
                    f"Market shows {chars['volatility']} volatility with {chars['returns']} return profile. Recommended action: {chars['action']} positioning. Confidence: {confidence_level} ({regime_confidence:.1%}).",
                    InsightSeverity.INFO,
                    InsightCategory.TIMING,
                    regime_confidence,
                )

            # Momentum analysis
            short_momentum = returns.tail(21).mean() / returns.tail(252).std()
            medium_momentum = returns.tail(63).mean() / returns.tail(252).std()

            if short_momentum > 2 and medium_momentum > 1.5:
                self._add_insight(
                    "Strong Positive Momentum",
                    f"Short-term momentum ({short_momentum:.1f}σ) and medium-term ({medium_momentum:.1f}σ) both elevated, suggesting bullish conditions.",
                    InsightSeverity.INFO,
                    InsightCategory.TIMING,
                    short_momentum,
                    recommendation="Consider increasing exposure to momentum factors."
                )
            elif short_momentum < -2 and medium_momentum < -1.5:
                self._add_insight(
                    "Strong Negative Momentum",
                    f"Both short-term ({short_momentum:.1f}σ) and medium-term ({medium_momentum:.1f}σ) momentum negative, suggesting risk-off conditions.",
                    InsightSeverity.ALERT,
                    InsightCategory.TIMING,
                    short_momentum,
                    recommendation="Consider defensive positioning and risk reduction."
                )

            # Macro indicator analysis
            if macro_indicators is not None:
                # Simple macro regime detection based on common indicators
                macro_regime_signals = []

                if 'vix' in macro_indicators.columns:
                    vix_current = macro_indicators['vix'].iloc[-1]
                    vix_ma = macro_indicators['vix'].tail(21).mean()
                    if vix_current > vix_ma * 1.2:
                        macro_regime_signals.append("elevated volatility")

                if 'yield_curve' in macro_indicators.columns:
                    yc_current = macro_indicators['yield_curve'].iloc[-1]
                    if yc_current < 0:
                        macro_regime_signals.append("inverted yield curve")

                if macro_regime_signals:
                    self._add_insight(
                        "Macro Regime Signals",
                        f"Market showing: {', '.join(macro_regime_signals)}. Consider adjusting portfolio positioning accordingly.",
                        InsightSeverity.WARNING,
                        InsightCategory.TIMING,
                        len(macro_regime_signals),
                    )

        except Exception as e:
            self._add_insight(
                "Timing Analysis Error",
                f"Could not complete market timing analysis: {str(e)}",
                InsightSeverity.WARNING,
                InsightCategory.TIMING,
            )

        return self.insights

    def analyze_structural_changes(
        self,
        weights_history: pd.DataFrame,
        returns: pd.Series,
        lookback_periods: int = 252,
    ) -> List[PMInsight]:
        """
        Detect structural changes in portfolio composition and market relationships.
        """
        self.clear()

        try:
            # Structural break analysis using rolling correlations
            if len(weights_history) > lookback_periods:
                # Factor exposure stability
                exposure_stability = weights_history.rolling(lookback_periods).std().mean(axis=1)
                recent_stability = exposure_stability.tail(21).mean()
                long_term_stability = exposure_stability.tail(lookback_periods).mean()

                stability_ratio = recent_stability / (long_term_stability + 1e-12)

                if stability_ratio > 2:
                    self._add_insight(
                        "Increasing Portfolio Instability",
                        f"Recent exposure volatility ({stability_ratio:.1f}x) significantly higher than long-term average, suggesting structural changes.",
                        InsightSeverity.WARNING,
                        InsightCategory.STRUCTURAL,
                        stability_ratio,
                        recommendation="Review factor exposures and rebalancing strategy."
                    )

            # Position size distribution changes
            w_current = weights_history.iloc[-1]
            w_previous = weights_history.iloc[-21] if len(weights_history) > 21 else weights_history.iloc[0]

            # Compare concentration metrics
            current_herfindahl = (w_current ** 2).sum()
            previous_herfindahl = (w_previous ** 2).sum()

            concentration_change = (current_herfindahl - previous_herfindahl) / (previous_herfindahl + 1e-12)

            if abs(concentration_change) > 0.5:
                direction = "increased" if concentration_change > 0 else "decreased"
                severity = InsightSeverity.WARNING if abs(concentration_change) > 1 else InsightSeverity.INFO

                self._add_insight(
                    f"Portfolio Concentration {direction.title()}",
                    f"Portfolio concentration has {direction} by {abs(concentration_change):.1%} over past 21 days.",
                    severity,
                    InsightCategory.STRUCTURAL,
                    concentration_change,
                    recommendation="Monitor for unintended concentration risk changes."
                )

            # New position analysis
            new_positions = set(w_current[w_current.abs() > 1e-4].index) - set(w_previous[w_previous.abs() > 1e-4].index)
            exited_positions = set(w_previous[w_previous.abs() > 1e-4].index) - set(w_current[w_current.abs() > 1e-4].index)

            if len(new_positions) > 3:
                self._add_insight(
                    "Significant Portfolio Changes",
                    f"Added {len(new_positions)} new positions in recent period, indicating major strategy shift.",
                    InsightSeverity.INFO,
                    InsightCategory.STRUCTURAL,
                    len(new_positions),
                )

            if len(exited_positions) > 3:
                self._add_insight(
                    "Positions Exited",
                    f"Removed {len(exited_positions)} positions, potentially signaling factor rotation.",
                    InsightSeverity.INFO,
                    InsightCategory.STRUCTURAL,
                    len(exited_positions),
                )

            # Performance attribution to structural changes
            if len(weights_history) > 21:
                # Simple attribution: compare returns with different weight sets
                recent_returns = returns.tail(21)
                old_weights = weights_history.iloc[-21]
                new_weights = weights_history.iloc[-1]

                # Simplified attribution (would need proper return attribution in production)
                structural_impact = abs((new_weights - old_weights).abs().sum()) * recent_returns.std()

                if structural_impact > recent_returns.std():
                    self._add_insight(
                        "Structural Changes Impact Performance",
                        f"Recent portfolio restructuring may have significantly impacted returns (impact: {structural_impact:.1%} annualized vol equivalent).",
                        InsightSeverity.INFO,
                        InsightCategory.STRUCTURAL,
                        structural_impact,
                    )

        except Exception as e:
            self._add_insight(
                "Structural Analysis Error",
                f"Could not complete structural change analysis: {str(e)}",
                InsightSeverity.WARNING,
                InsightCategory.STRUCTURAL,
            )

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
