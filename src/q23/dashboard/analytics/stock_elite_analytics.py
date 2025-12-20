"""
Stock Elite Analytics

Per-stock institutional-grade quantitative analytics including:
- Regime-conditional performance
- Tail risk metrics (VaR, CVaR, skewness, kurtosis)
- Beta and sensitivity analysis
- Return distribution analysis
- Signal quality metrics
- Risk decomposition

PM Best Practices:
- All metrics annualized where appropriate
- Clear statistical significance indicators
- Consistent with industry standards
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from q23.dashboard.analytics.advanced import RegimeDetector, TailRiskAnalyzer


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EliteStockMetrics:
    """Comprehensive elite metrics for a single stock."""
    symbol: str
    
    # Risk-adjusted returns
    sharpe: float
    sortino: float
    calmar: float
    
    # Tail metrics
    var_95: float
    cvar_95: float
    max_drawdown: float
    worst_day: float
    
    # Distribution
    skewness: float
    kurtosis: float
    
    # Statistical significance
    t_stat: float
    p_value: float
    
    # Sample info
    n_observations: int


@dataclass
class RegimePerformance:
    """Performance metrics for a specific regime."""
    regime: str
    total_return: float
    ann_return: float
    ann_vol: float
    sharpe: float
    win_rate: float
    n_days: int
    avg_daily_return: float


# =============================================================================
# Regime Analysis
# =============================================================================

def compute_stock_regime_performance(
    stock_returns: pd.Series,
    portfolio_returns: pd.Series,
    lookback: int = 30,
) -> Dict[str, Any]:
    """
    Compute stock performance broken down by market regime.
    
    Uses portfolio returns to detect regimes, then analyzes stock performance
    within each regime.
    
    Args:
        stock_returns: Stock daily returns
        portfolio_returns: Portfolio returns (for regime detection)
        lookback: Regime detection lookback period
        
    Returns:
        Dictionary with regime performance metrics
    """
    if stock_returns is None or stock_returns.empty:
        return {}
    
    if portfolio_returns is None or portfolio_returns.empty:
        return {}
    
    # Detect regimes using portfolio returns
    detector = RegimeDetector(portfolio_returns, lookback=lookback)
    regimes = detector.detect_regimes()
    
    # Align stock returns with regime data
    aligned_returns, aligned_regimes = stock_returns.align(
        regimes["regime"], join="inner"
    )
    
    if aligned_returns.empty:
        return {}
    
    # Compute performance by regime
    regime_names = ["crisis", "normal", "trend", "calm"]
    regime_perf = {}
    
    for regime_name in regime_names:
        mask = aligned_regimes == regime_name
        regime_returns = aligned_returns[mask]
        
        if len(regime_returns) < 5:  # Need minimum data
            continue
        
        # Compute metrics
        total_ret = float((1 + regime_returns).prod() - 1)
        n_days = len(regime_returns)
        ann_factor = 252 / n_days if n_days > 0 else 0
        ann_ret = float((1 + total_ret) ** ann_factor - 1) if n_days > 0 else 0.0
        ann_vol = float(regime_returns.std() * np.sqrt(252))
        sharpe = ann_ret / (ann_vol + 1e-12)
        win_rate = float((regime_returns > 0).mean())
        avg_daily = float(regime_returns.mean())
        
        regime_perf[regime_name] = RegimePerformance(
            regime=regime_name,
            total_return=total_ret,
            ann_return=ann_ret,
            ann_vol=ann_vol,
            sharpe=sharpe,
            win_rate=win_rate,
            n_days=n_days,
            avg_daily_return=avg_daily,
        )
    
    # Get regime timeline for visualization
    regime_timeline = regimes[["regime", "vol_z", "trend_corr"]].copy()
    
    # Overall statistics
    regime_distribution = aligned_regimes.value_counts(normalize=True).to_dict()
    
    return {
        "regime_performance": regime_perf,
        "regime_timeline": regime_timeline,
        "regime_distribution": regime_distribution,
        "lookback": lookback,
    }


def get_regime_exposure_summary(
    weights: pd.Series,
    portfolio_returns: pd.Series,
    lookback: int = 30,
) -> pd.DataFrame:
    """
    Get summary of stock exposure during each regime.
    
    Args:
        weights: Stock weight series
        portfolio_returns: Portfolio returns for regime detection
        lookback: Regime detection lookback
        
    Returns:
        DataFrame with regime exposure summary
    """
    if weights is None or weights.empty:
        return pd.DataFrame()
    
    if portfolio_returns is None or portfolio_returns.empty:
        return pd.DataFrame()
    
    detector = RegimeDetector(portfolio_returns, lookback=lookback)
    regimes = detector.detect_regimes()
    
    aligned_weights, aligned_regimes = weights.align(
        regimes["regime"], join="inner"
    )
    
    results = []
    for regime_name in ["crisis", "normal", "trend", "calm"]:
        mask = aligned_regimes == regime_name
        regime_weights = aligned_weights[mask]
        
        if len(regime_weights) < 1:
            continue
        
        results.append({
            "regime": regime_name,
            "avg_weight": float(regime_weights.mean()),
            "max_weight": float(regime_weights.abs().max()),
            "days_held": int((regime_weights.abs() > 1e-12).sum()),
            "total_days": int(mask.sum()),
            "pct_time_held": float((regime_weights.abs() > 1e-12).mean()),
        })
    
    return pd.DataFrame(results)


# =============================================================================
# Tail Risk Analysis
# =============================================================================

def compute_stock_tail_metrics(
    stock_returns: pd.Series,
) -> Dict[str, float]:
    """
    Compute comprehensive tail risk metrics for a stock.
    
    Args:
        stock_returns: Stock daily returns
        
    Returns:
        Dictionary with tail risk metrics
    """
    if stock_returns is None or stock_returns.empty:
        return {}
    
    rets = stock_returns.dropna()
    if len(rets) < 20:
        return {"error": "Insufficient data for tail analysis"}
    
    # Use existing TailRiskAnalyzer
    analyzer = TailRiskAnalyzer(rets)
    tail_metrics = analyzer.compute_tail_metrics()
    
    # Add additional metrics
    
    # Downside deviation (for Sortino)
    negative_rets = rets[rets < 0]
    downside_dev = float(negative_rets.std() * np.sqrt(252)) if len(negative_rets) > 0 else 0.0
    tail_metrics["downside_deviation"] = downside_dev
    
    # Gain/loss ratio
    positive_rets = rets[rets > 0]
    if len(negative_rets) > 0 and len(positive_rets) > 0:
        avg_gain = positive_rets.mean()
        avg_loss = abs(negative_rets.mean())
        tail_metrics["gain_loss_ratio"] = float(avg_gain / (avg_loss + 1e-12))
    else:
        tail_metrics["gain_loss_ratio"] = 0.0
    
    # Profit factor
    total_gains = positive_rets.sum() if len(positive_rets) > 0 else 0
    total_losses = abs(negative_rets.sum()) if len(negative_rets) > 0 else 0
    tail_metrics["profit_factor"] = float(total_gains / (total_losses + 1e-12))
    
    # Maximum consecutive losses
    is_loss = (rets < 0).astype(int)
    consecutive_losses = is_loss.groupby((is_loss != is_loss.shift()).cumsum()).cumsum()
    tail_metrics["max_consecutive_losses"] = int(consecutive_losses.max())
    
    # Recovery time from worst drawdown
    cum = (1 + rets).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    
    # Find worst drawdown trough
    trough_idx = drawdown.idxmin()
    if trough_idx is not None:
        # Find when it recovered (if ever)
        recovery_mask = cum.loc[trough_idx:] >= running_max.loc[trough_idx]
        if recovery_mask.any():
            recovery_idx = recovery_mask.idxmax()
            recovery_days = len(cum.loc[trough_idx:recovery_idx])
            tail_metrics["recovery_days"] = recovery_days
        else:
            tail_metrics["recovery_days"] = -1  # Not recovered
    else:
        tail_metrics["recovery_days"] = 0
    
    return tail_metrics


def compute_rolling_tail_risk(
    stock_returns: pd.Series,
    window: int = 63,
) -> pd.DataFrame:
    """
    Compute rolling tail risk metrics.
    
    Args:
        stock_returns: Stock daily returns
        window: Rolling window in days
        
    Returns:
        DataFrame with rolling tail metrics
    """
    if stock_returns is None or len(stock_returns) < window:
        return pd.DataFrame()
    
    results = []
    
    for i in range(window, len(stock_returns) + 1):
        subset = stock_returns.iloc[i - window:i]
        date = stock_returns.index[i - 1]
        
        sorted_rets = np.sort(subset.values)
        
        # VaR and CVaR
        var_95 = np.percentile(sorted_rets, 5)
        tail_95 = sorted_rets[sorted_rets <= var_95]
        cvar_95 = tail_95.mean() if len(tail_95) > 0 else var_95
        
        # Skewness and kurtosis
        skew = stats.skew(subset)
        kurt = stats.kurtosis(subset)
        
        results.append({
            "date": date,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "skewness": skew,
            "kurtosis": kurt,
            "volatility": subset.std() * np.sqrt(252),
        })
    
    return pd.DataFrame(results).set_index("date")


# =============================================================================
# Beta and Sensitivity Analysis
# =============================================================================

def compute_stock_beta_analysis(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Dict[str, Any]:
    """
    Compute comprehensive beta and sensitivity metrics.
    
    Args:
        stock_returns: Stock daily returns
        benchmark_returns: Benchmark (portfolio) returns
        
    Returns:
        Dictionary with beta metrics
    """
    if stock_returns is None or stock_returns.empty:
        return {}
    
    if benchmark_returns is None or benchmark_returns.empty:
        return {}
    
    # Align returns
    aligned_stock, aligned_bench = stock_returns.align(benchmark_returns, join="inner")
    
    if len(aligned_stock) < 20:
        return {"error": "Insufficient overlapping data"}
    
    # Overall beta
    cov_matrix = np.cov(aligned_stock, aligned_bench)
    beta = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-12)
    
    # R-squared (systematic vs idiosyncratic)
    correlation = aligned_stock.corr(aligned_bench)
    r_squared = correlation ** 2
    
    # Alpha (Jensen's alpha)
    # alpha = stock_return - beta * benchmark_return
    expected_return = beta * aligned_bench.mean() * 252
    actual_return = aligned_stock.mean() * 252
    alpha = actual_return - expected_return
    
    # Upside/downside beta (convexity)
    up_days = aligned_bench > 0
    down_days = aligned_bench < 0
    
    if up_days.sum() > 10:
        up_cov = np.cov(aligned_stock[up_days], aligned_bench[up_days])
        upside_beta = up_cov[0, 1] / (up_cov[1, 1] + 1e-12)
    else:
        upside_beta = beta
    
    if down_days.sum() > 10:
        down_cov = np.cov(aligned_stock[down_days], aligned_bench[down_days])
        downside_beta = down_cov[0, 1] / (down_cov[1, 1] + 1e-12)
    else:
        downside_beta = beta
    
    # Asymmetry (positive = convex payoff)
    asymmetry = upside_beta - downside_beta
    
    # Tracking error
    excess_returns = aligned_stock - aligned_bench
    tracking_error = excess_returns.std() * np.sqrt(252)
    
    # Information ratio
    info_ratio = (excess_returns.mean() * 252) / (tracking_error + 1e-12)
    
    # Correlation stability (rolling correlation std)
    if len(aligned_stock) >= 63:
        rolling_corr = aligned_stock.rolling(21).corr(aligned_bench)
        corr_stability = 1 - rolling_corr.std()  # Higher = more stable
    else:
        corr_stability = None
    
    return {
        "beta": float(beta),
        "r_squared": float(r_squared),
        "alpha": float(alpha),
        "upside_beta": float(upside_beta),
        "downside_beta": float(downside_beta),
        "asymmetry": float(asymmetry),
        "tracking_error": float(tracking_error),
        "information_ratio": float(info_ratio),
        "correlation": float(correlation),
        "correlation_stability": float(corr_stability) if corr_stability is not None else None,
        "n_observations": len(aligned_stock),
    }


def compute_rolling_beta(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 63,
) -> pd.DataFrame:
    """
    Compute rolling beta and related metrics.
    
    Args:
        stock_returns: Stock daily returns
        benchmark_returns: Benchmark returns
        window: Rolling window in days
        
    Returns:
        DataFrame with rolling metrics
    """
    aligned_stock, aligned_bench = stock_returns.align(benchmark_returns, join="inner")
    
    if len(aligned_stock) < window:
        return pd.DataFrame()
    
    results = []
    
    for i in range(window, len(aligned_stock) + 1):
        s_sub = aligned_stock.iloc[i - window:i]
        b_sub = aligned_bench.iloc[i - window:i]
        date = aligned_stock.index[i - 1]
        
        # Beta
        cov_matrix = np.cov(s_sub, b_sub)
        beta = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-12)
        
        # Correlation
        corr = s_sub.corr(b_sub)
        
        # R-squared
        r_sq = corr ** 2
        
        results.append({
            "date": date,
            "rolling_beta": beta,
            "rolling_correlation": corr,
            "rolling_r_squared": r_sq,
        })
    
    return pd.DataFrame(results).set_index("date")


# =============================================================================
# Distribution Analysis
# =============================================================================

def compute_stock_distribution_stats(
    stock_returns: pd.Series,
) -> Dict[str, Any]:
    """
    Compute comprehensive distribution statistics.
    
    Args:
        stock_returns: Stock daily returns
        
    Returns:
        Dictionary with distribution metrics
    """
    if stock_returns is None or stock_returns.empty:
        return {}
    
    rets = stock_returns.dropna()
    if len(rets) < 20:
        return {"error": "Insufficient data"}
    
    # Basic moments
    mean = float(rets.mean())
    std = float(rets.std())
    skewness = float(stats.skew(rets))
    kurtosis = float(stats.kurtosis(rets))  # Excess kurtosis
    
    # Annualized
    ann_return = mean * 252
    ann_vol = std * np.sqrt(252)
    
    # Normality tests
    try:
        jb_stat, jb_pvalue = stats.jarque_bera(rets)
    except Exception:
        jb_stat, jb_pvalue = np.nan, np.nan
    
    try:
        shapiro_stat, shapiro_pvalue = stats.shapiro(rets[:5000])  # Shapiro limited to 5000
    except Exception:
        shapiro_stat, shapiro_pvalue = np.nan, np.nan
    
    # Percentiles
    percentiles = {
        "p1": float(np.percentile(rets, 1)),
        "p5": float(np.percentile(rets, 5)),
        "p10": float(np.percentile(rets, 10)),
        "p25": float(np.percentile(rets, 25)),
        "p50": float(np.percentile(rets, 50)),
        "p75": float(np.percentile(rets, 75)),
        "p90": float(np.percentile(rets, 90)),
        "p95": float(np.percentile(rets, 95)),
        "p99": float(np.percentile(rets, 99)),
    }
    
    # Histogram data
    hist_values, hist_edges = np.histogram(rets, bins=50, density=True)
    
    # Normal distribution overlay
    x_normal = np.linspace(rets.min(), rets.max(), 100)
    y_normal = stats.norm.pdf(x_normal, mean, std)
    
    # T-test for mean != 0
    t_stat, t_pvalue = stats.ttest_1samp(rets, 0)
    
    return {
        # Moments
        "mean": mean,
        "std": std,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        
        # Normality
        "jarque_bera_stat": float(jb_stat) if not np.isnan(jb_stat) else None,
        "jarque_bera_pvalue": float(jb_pvalue) if not np.isnan(jb_pvalue) else None,
        "shapiro_stat": float(shapiro_stat) if not np.isnan(shapiro_stat) else None,
        "shapiro_pvalue": float(shapiro_pvalue) if not np.isnan(shapiro_pvalue) else None,
        "is_normal_jb": bool(jb_pvalue > 0.05) if not np.isnan(jb_pvalue) else None,
        "is_normal_shapiro": bool(shapiro_pvalue > 0.05) if not np.isnan(shapiro_pvalue) else None,
        
        # Percentiles
        "percentiles": percentiles,
        
        # Histogram data (for visualization)
        "hist_values": hist_values.tolist(),
        "hist_edges": hist_edges.tolist(),
        "normal_x": x_normal.tolist(),
        "normal_y": y_normal.tolist(),
        
        # T-test
        "t_stat": float(t_stat),
        "t_pvalue": float(t_pvalue),
        "mean_significant": bool(t_pvalue < 0.05),
        
        # Sample info
        "n_observations": len(rets),
    }


def compute_qq_data(
    stock_returns: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Q-Q plot data against normal distribution.
    
    Args:
        stock_returns: Stock daily returns
        
    Returns:
        Tuple of (theoretical quantiles, sample quantiles)
    """
    rets = stock_returns.dropna()
    if len(rets) < 10:
        return np.array([]), np.array([])
    
    # Standardize returns
    standardized = (rets - rets.mean()) / rets.std()
    
    # Get theoretical quantiles
    n = len(standardized)
    theoretical = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
    sample = np.sort(standardized.values)
    
    return theoretical, sample


# =============================================================================
# Signal Quality Analysis
# =============================================================================

def compute_stock_signal_quality(
    weights_series: pd.Series,
    stock_returns: pd.Series,
    max_horizon: int = 21,
) -> Dict[str, Any]:
    """
    Analyze the quality of the position-sizing signal.
    
    Tests whether position size predicts forward returns.
    
    Args:
        weights_series: Stock weights over time
        stock_returns: Stock returns
        max_horizon: Maximum forward horizon to test
        
    Returns:
        Dictionary with signal quality metrics
    """
    if weights_series is None or weights_series.empty:
        return {}
    
    if stock_returns is None or stock_returns.empty:
        return {}
    
    # Align data
    weights, returns = weights_series.align(stock_returns, join="inner")
    
    if len(weights) < 50:
        return {"error": "Insufficient data for signal analysis"}
    
    # IC at various horizons
    ic_by_horizon = {}
    
    for h in range(1, min(max_horizon + 1, len(weights) // 2)):
        fwd_ret = returns.shift(-h)
        
        # Align and drop NaN
        common_idx = weights.index.intersection(fwd_ret.dropna().index)
        if len(common_idx) < 20:
            continue
        
        w = weights.loc[common_idx]
        r = fwd_ret.loc[common_idx]
        
        # Pearson IC
        ic = w.corr(r)
        
        # Rank IC (Spearman)
        rank_ic = w.rank().corr(r.rank())
        
        ic_by_horizon[h] = {
            "pearson_ic": float(ic),
            "rank_ic": float(rank_ic),
        }
    
    # Find optimal horizon (max average IC)
    if ic_by_horizon:
        avg_ic = {h: (v["pearson_ic"] + v["rank_ic"]) / 2 for h, v in ic_by_horizon.items()}
        optimal_horizon = max(avg_ic, key=avg_ic.get)
    else:
        optimal_horizon = 1
    
    # Hit rate analysis (did position direction predict return direction?)
    fwd_1d = returns.shift(-1)
    common = weights.index.intersection(fwd_1d.dropna().index)
    
    if len(common) > 20:
        w_common = weights.loc[common]
        r_common = fwd_1d.loc[common]
        
        # Only count when we have a position
        has_position = w_common.abs() > 1e-12
        w_pos = w_common[has_position]
        r_pos = r_common[has_position]
        
        if len(w_pos) > 10:
            correct = ((w_pos > 0) & (r_pos > 0)) | ((w_pos < 0) & (r_pos < 0))
            hit_rate = float(correct.mean())
            
            # Average return when correct vs incorrect
            avg_ret_correct = float(r_pos[correct].abs().mean()) if correct.any() else 0.0
            avg_ret_incorrect = float(r_pos[~correct].abs().mean()) if (~correct).any() else 0.0
        else:
            hit_rate = None
            avg_ret_correct = None
            avg_ret_incorrect = None
    else:
        hit_rate = None
        avg_ret_correct = None
        avg_ret_incorrect = None
    
    # Weight-return correlation over time (rolling IC)
    if len(weights) >= 63:
        rolling_ic = weights.rolling(21).corr(returns.shift(-1))
        rolling_ic_mean = float(rolling_ic.mean())
        rolling_ic_std = float(rolling_ic.std())
    else:
        rolling_ic = None
        rolling_ic_mean = None
        rolling_ic_std = None
    
    return {
        "ic_by_horizon": ic_by_horizon,
        "optimal_horizon": optimal_horizon,
        "hit_rate": hit_rate,
        "avg_ret_correct": avg_ret_correct,
        "avg_ret_incorrect": avg_ret_incorrect,
        "rolling_ic_mean": rolling_ic_mean,
        "rolling_ic_std": rolling_ic_std,
        "rolling_ic_series": rolling_ic,
    }


# =============================================================================
# Risk Decomposition
# =============================================================================

def compute_stock_risk_decomposition(
    stock_returns: pd.Series,
    portfolio_returns: pd.Series,
    stock_weight: pd.Series,
) -> Dict[str, Any]:
    """
    Decompose stock risk into systematic and idiosyncratic components.
    
    Args:
        stock_returns: Stock returns
        portfolio_returns: Portfolio returns
        stock_weight: Stock weight series
        
    Returns:
        Dictionary with risk decomposition
    """
    if stock_returns is None or stock_returns.empty:
        return {}
    
    if portfolio_returns is None or portfolio_returns.empty:
        return {}
    
    # Align
    s_ret, p_ret = stock_returns.align(portfolio_returns, join="inner")
    
    if len(s_ret) < 30:
        return {"error": "Insufficient data for risk decomposition"}
    
    # Total variance
    total_var = s_ret.var()
    
    # Systematic component (beta * market variance)
    beta_analysis = compute_stock_beta_analysis(s_ret, p_ret)
    beta = beta_analysis.get("beta", 0)
    r_squared = beta_analysis.get("r_squared", 0)
    
    systematic_var = (beta ** 2) * p_ret.var()
    idiosyncratic_var = total_var - systematic_var
    
    # Ensure non-negative
    idiosyncratic_var = max(0, idiosyncratic_var)
    
    # Proportion
    systematic_pct = systematic_var / (total_var + 1e-12)
    idiosyncratic_pct = idiosyncratic_var / (total_var + 1e-12)
    
    # Annualized
    total_risk = float(np.sqrt(total_var) * np.sqrt(252))
    systematic_risk = float(np.sqrt(systematic_var) * np.sqrt(252))
    idiosyncratic_risk = float(np.sqrt(idiosyncratic_var) * np.sqrt(252))
    
    # Marginal contribution to portfolio risk
    # Align weight with returns
    w, _ = stock_weight.align(p_ret, join="inner")
    avg_weight = float(w.mean())
    
    # Contribution = weight * beta * portfolio_vol
    portfolio_vol = float(p_ret.std() * np.sqrt(252))
    marginal_contribution = avg_weight * beta * portfolio_vol
    
    # Diversification ratio
    # How much diversification benefit does this stock provide?
    if avg_weight > 1e-12:
        stand_alone_vol = total_risk
        diversified_vol = marginal_contribution / (avg_weight + 1e-12)
        diversification_ratio = stand_alone_vol / (diversified_vol + 1e-12)
    else:
        diversification_ratio = 1.0
    
    return {
        # Variance decomposition
        "total_variance": float(total_var),
        "systematic_variance": float(systematic_var),
        "idiosyncratic_variance": float(idiosyncratic_var),
        "systematic_pct": float(systematic_pct * 100),
        "idiosyncratic_pct": float(idiosyncratic_pct * 100),
        
        # Risk (annualized vol)
        "total_risk": total_risk,
        "systematic_risk": systematic_risk,
        "idiosyncratic_risk": idiosyncratic_risk,
        
        # Portfolio contribution
        "beta": float(beta),
        "r_squared": float(r_squared),
        "avg_weight": avg_weight,
        "marginal_var_contribution": float(marginal_contribution),
        
        # Diversification
        "diversification_ratio": float(diversification_ratio),
        
        # Sample
        "n_observations": len(s_ret),
    }


# =============================================================================
# Elite Score Card
# =============================================================================

def compute_elite_scorecard(
    stock_returns: pd.Series,
    portfolio_returns: pd.Series,
    weights: pd.Series,
) -> EliteStockMetrics:
    """
    Compute comprehensive elite metrics scorecard for a stock.
    
    Args:
        stock_returns: Stock returns
        portfolio_returns: Portfolio returns (for beta, etc.)
        weights: Stock weights
        
    Returns:
        EliteStockMetrics dataclass
    """
    symbol = weights.name if hasattr(weights, 'name') and weights.name else "UNKNOWN"
    
    if stock_returns is None or stock_returns.empty:
        return EliteStockMetrics(
            symbol=symbol,
            sharpe=0.0, sortino=0.0, calmar=0.0,
            var_95=0.0, cvar_95=0.0, max_drawdown=0.0, worst_day=0.0,
            skewness=0.0, kurtosis=0.0,
            t_stat=0.0, p_value=1.0,
            n_observations=0,
        )
    
    rets = stock_returns.dropna()
    n = len(rets)
    
    if n < 20:
        return EliteStockMetrics(
            symbol=symbol,
            sharpe=0.0, sortino=0.0, calmar=0.0,
            var_95=0.0, cvar_95=0.0, max_drawdown=0.0, worst_day=0.0,
            skewness=0.0, kurtosis=0.0,
            t_stat=0.0, p_value=1.0,
            n_observations=n,
        )
    
    # Returns
    ann_ret = float(rets.mean() * 252)
    ann_vol = float(rets.std() * np.sqrt(252))
    
    # Sharpe
    sharpe = ann_ret / (ann_vol + 1e-12)
    
    # Sortino
    downside = rets[rets < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
    sortino = ann_ret / (downside_std + 1e-12)
    
    # Max drawdown
    cum = (1 + rets).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_dd = float(drawdown.min())
    
    # Calmar
    calmar = ann_ret / (abs(max_dd) + 1e-12)
    
    # Tail metrics
    tail = compute_stock_tail_metrics(stock_returns)
    var_95 = tail.get("var_95", 0.0)
    cvar_95 = tail.get("cvar_95", 0.0)
    worst_day = tail.get("worst_day", 0.0)
    
    # Distribution
    skewness = float(stats.skew(rets))
    kurtosis = float(stats.kurtosis(rets))
    
    # T-test
    t_stat, p_value = stats.ttest_1samp(rets, 0)
    
    return EliteStockMetrics(
        symbol=symbol,
        sharpe=float(sharpe),
        sortino=float(sortino),
        calmar=float(calmar),
        var_95=float(var_95),
        cvar_95=float(cvar_95),
        max_drawdown=float(max_dd),
        worst_day=float(worst_day),
        skewness=float(skewness),
        kurtosis=float(kurtosis),
        t_stat=float(t_stat),
        p_value=float(p_value),
        n_observations=n,
    )
