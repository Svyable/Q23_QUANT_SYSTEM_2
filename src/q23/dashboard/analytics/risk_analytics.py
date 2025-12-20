"""
Risk Analytics Module for PM Dashboard

Provides comprehensive risk and performance attribution tools:
- Risk Attribution (factor, sector, asset level)
- Brinson Performance Attribution (allocation, selection, interaction)
- Ex-Ante Risk Metrics (predicted tracking error, factor risk)
- Sector/Industry Attribution
- Correlation Analysis

Author: Q23 Quant System
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class RiskAttributionResult:
    """Result container for risk attribution analysis."""
    total_risk: float
    factor_risk: float
    specific_risk: float
    factor_contributions: pd.Series
    marginal_contributions: pd.Series
    percentage_contributions: pd.Series
    correlation_contribution: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "total_risk": self.total_risk,
            "factor_risk": self.factor_risk,
            "specific_risk": self.specific_risk,
            "correlation_contribution": self.correlation_contribution,
        }


@dataclass
class BrinsonResult:
    """Result container for Brinson attribution analysis."""
    allocation_effect: pd.Series
    selection_effect: pd.Series
    interaction_effect: pd.Series
    total_active_return: pd.Series
    
    # Summary metrics
    total_allocation: float
    total_selection: float
    total_interaction: float
    total_excess: float
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "allocation": self.allocation_effect,
            "selection": self.selection_effect,
            "interaction": self.interaction_effect,
            "total_active": self.total_active_return,
        })


@dataclass 
class ExAnteRiskResult:
    """Result container for ex-ante risk analysis."""
    total_risk: float
    tracking_error: float
    factor_risk: float
    specific_risk: float
    factor_contributions: pd.Series
    diversification_ratio: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "total_risk": self.total_risk,
            "tracking_error": self.tracking_error,
            "factor_risk": self.factor_risk,
            "specific_risk": self.specific_risk,
            "diversification_ratio": self.diversification_ratio,
        }


@dataclass
class ForwardRiskProjection:
    """Forward-looking risk projections for T+1 to T+N days.
    
    Used for risk cone visualization showing potential portfolio
    outcomes with confidence bands.
    """
    horizons: List[int]  # [1, 2, 3, 4, 5] days forward
    base_value: float  # Starting portfolio value (normalized to 100)
    expected_values: np.ndarray  # Central path (expected values)
    upper_1sigma: np.ndarray  # Upper 1-sigma bound
    lower_1sigma: np.ndarray  # Lower 1-sigma bound
    upper_2sigma: np.ndarray  # Upper 2-sigma bound (95% CI)
    lower_2sigma: np.ndarray  # Lower 2-sigma bound (95% CI)
    daily_vol: float  # Daily volatility used
    annual_vol: float  # Annualized volatility
    expected_daily_return: float  # Expected daily return (drift)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for easy plotting."""
        return pd.DataFrame({
            "horizon": self.horizons,
            "expected": self.expected_values,
            "upper_1σ": self.upper_1sigma,
            "lower_1σ": self.lower_1sigma,
            "upper_2σ": self.upper_2sigma,
            "lower_2σ": self.lower_2sigma,
        })
    
    def get_range_at_horizon(self, t: int) -> Dict[str, float]:
        """Get the projected range at a specific horizon."""
        if t not in self.horizons:
            return {}
        idx = self.horizons.index(t)
        return {
            "expected": float(self.expected_values[idx]),
            "upper_1σ": float(self.upper_1sigma[idx]),
            "lower_1σ": float(self.lower_1sigma[idx]),
            "upper_2σ": float(self.upper_2sigma[idx]),
            "lower_2σ": float(self.lower_2sigma[idx]),
            "range_1σ": float(self.upper_1sigma[idx] - self.lower_1sigma[idx]),
            "range_2σ": float(self.upper_2sigma[idx] - self.lower_2sigma[idx]),
        }


@dataclass
class ExpectedReturnEstimate:
    """Expected return estimation combining IC-based and historical approaches.
    
    Provides PM with forward-looking return estimates and alerts
    when predictions are significantly below historical levels.
    """
    # Return estimates
    ic_based_return: float  # From factor IC × exposure
    drift_based_return: float  # Historical rolling mean
    combined_estimate: float  # Weighted average of both
    
    # Historical comparison
    z_score_vs_history: float  # How unusual is today's prediction
    percentile: float  # Where does today's estimate rank (0-100)
    historical_mean: float  # Mean of historical predictions
    historical_std: float  # Std of historical predictions
    
    # Signal quality
    signal_freshness: float  # 0-1, how fresh/strong are IC signals
    ic_trend: str  # "improving", "stable", "declining"
    
    # Alerts
    is_significantly_low: bool  # Alert flag (below -1.5 sigma)
    is_significantly_high: bool  # Alert flag (above +1.5 sigma)
    alert_message: Optional[str]  # Human-readable alert
    
    def to_dict(self) -> Dict[str, Union[float, str, bool]]:
        return {
            "ic_based_return": self.ic_based_return,
            "drift_based_return": self.drift_based_return,
            "combined_estimate": self.combined_estimate,
            "z_score": self.z_score_vs_history,
            "percentile": self.percentile,
            "signal_freshness": self.signal_freshness,
            "is_alert": self.is_significantly_low or self.is_significantly_high,
        }


# =============================================================================
# Risk Attribution
# =============================================================================

def compute_risk_attribution(
    weights: pd.Series,
    factor_exposures: pd.DataFrame,
    factor_covariance: pd.DataFrame,
    specific_risk: Optional[pd.Series] = None,
    annualize: bool = True,
) -> RiskAttributionResult:
    """
    Compute risk attribution by factor.
    
    Decomposes portfolio risk into:
    - Factor risk contribution (systematic)
    - Specific/idiosyncratic risk
    - Correlation effects
    
    Args:
        weights: Portfolio weights (asset)
        factor_exposures: Factor exposures (asset x factor)
        factor_covariance: Factor covariance matrix (factor x factor)
        specific_risk: Asset-level specific risk/variance (optional)
        annualize: Whether to annualize (multiply by sqrt(252))
        
    Returns:
        RiskAttributionResult with detailed risk decomposition
    """
    eps = 1e-12
    ann_factor = np.sqrt(252) if annualize else 1.0
    
    # Align weights with exposures
    common_assets = weights.index.intersection(factor_exposures.index)
    if len(common_assets) == 0:
        return RiskAttributionResult(
            total_risk=0.0,
            factor_risk=0.0,
            specific_risk=0.0,
            factor_contributions=pd.Series(dtype=float),
            marginal_contributions=pd.Series(dtype=float),
            percentage_contributions=pd.Series(dtype=float),
            correlation_contribution=0.0,
        )
    
    w = weights.reindex(common_assets).fillna(0.0).values
    F = factor_exposures.reindex(common_assets).fillna(0.0)
    
    # Align factor covariance with exposures
    common_factors = F.columns.intersection(factor_covariance.index)
    if len(common_factors) == 0:
        return RiskAttributionResult(
            total_risk=0.0,
            factor_risk=0.0,
            specific_risk=0.0,
            factor_contributions=pd.Series(dtype=float),
            marginal_contributions=pd.Series(dtype=float),
            percentage_contributions=pd.Series(dtype=float),
            correlation_contribution=0.0,
        )
    
    F = F[common_factors].values
    cov = factor_covariance.reindex(index=common_factors, columns=common_factors).fillna(0.0).values
    
    # Portfolio factor exposures: sum(w * F) for each factor
    port_factor_exp = F.T @ w  # (n_factors,)
    
    # Factor variance: x' * Cov * x
    factor_var = port_factor_exp @ cov @ port_factor_exp
    factor_risk = np.sqrt(max(factor_var, 0.0)) * ann_factor
    
    # Specific risk
    if specific_risk is not None:
        spec_risk_aligned = specific_risk.reindex(common_assets).fillna(0.0).values
        specific_var = np.sum((w ** 2) * (spec_risk_aligned ** 2))
    else:
        # Estimate specific risk as 20% of total weight variance
        specific_var = 0.04 * np.sum(w ** 2)
    
    spec_risk = np.sqrt(max(specific_var, 0.0)) * ann_factor
    
    # Total risk
    total_var = factor_var + specific_var
    total_risk = np.sqrt(max(total_var, 0.0)) * ann_factor
    
    # Factor contributions (marginal contribution to risk)
    # MCR_i = (Cov @ x)_i / sigma
    if factor_var > eps:
        mcr = (cov @ port_factor_exp) * port_factor_exp / (np.sqrt(factor_var) + eps)
        mcr = mcr * ann_factor
    else:
        mcr = np.zeros(len(common_factors))
    
    # Percentage contributions
    pct_contrib = mcr / (total_risk + eps)
    
    # Create result series
    factor_names = list(common_factors)
    factor_contributions = pd.Series(mcr, index=factor_names, name="contribution")
    marginal_contributions = pd.Series(
        (cov @ port_factor_exp) / (np.sqrt(factor_var) + eps) * ann_factor,
        index=factor_names,
        name="marginal_contribution"
    )
    percentage_contributions = pd.Series(pct_contrib, index=factor_names, name="pct_contribution")
    
    # Correlation contribution (diversification benefit)
    sum_individual = np.sum(np.abs(mcr))
    correlation_contribution = sum_individual - factor_risk
    
    return RiskAttributionResult(
        total_risk=float(total_risk),
        factor_risk=float(factor_risk),
        specific_risk=float(spec_risk),
        factor_contributions=factor_contributions,
        marginal_contributions=marginal_contributions,
        percentage_contributions=percentage_contributions,
        correlation_contribution=float(correlation_contribution),
    )


def compute_rolling_risk_attribution(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    factor_exposures: pd.DataFrame,
    window: int = 63,
    min_periods: int = 21,
) -> pd.DataFrame:
    """
    Compute rolling risk attribution over time.
    
    Uses realized covariance for factor risk estimation.
    
    Args:
        weights: Portfolio weights (time x asset)
        returns: Asset returns (time x asset)
        factor_exposures: Factor exposures (time x factor or asset x factor)
        window: Rolling window for covariance estimation
        min_periods: Minimum periods required
        
    Returns:
        DataFrame with rolling risk metrics (time x metric)
    """
    results = []
    
    # Align data
    common_dates = weights.index.intersection(returns.index)
    if len(common_dates) < min_periods:
        return pd.DataFrame()
    
    weights = weights.reindex(common_dates)
    returns = returns.reindex(common_dates)
    
    for i in range(window, len(common_dates)):
        date = common_dates[i]
        start_idx = i - window
        
        # Get current weights
        w = weights.iloc[i]
        
        # Compute rolling covariance
        ret_window = returns.iloc[start_idx:i]
        cov_matrix = ret_window.cov() * 252  # Annualized
        
        # Portfolio variance
        common_assets = w.index.intersection(cov_matrix.index)
        if len(common_assets) < 2:
            continue
            
        w_aligned = w.reindex(common_assets).fillna(0.0).values
        cov_aligned = cov_matrix.reindex(index=common_assets, columns=common_assets).fillna(0.0).values
        
        port_var = w_aligned @ cov_aligned @ w_aligned
        port_vol = np.sqrt(max(port_var, 0.0))
        
        # Asset contributions
        mcr = cov_aligned @ w_aligned
        asset_contrib = w_aligned * mcr
        
        # Concentration metrics
        top5_contrib = np.sort(np.abs(asset_contrib))[-5:].sum() / (port_var + 1e-12)
        
        results.append({
            "date": date,
            "portfolio_vol": port_vol,
            "portfolio_var": port_var,
            "top5_risk_contrib": top5_contrib,
            "n_assets": np.sum(np.abs(w_aligned) > 1e-12),
        })
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results).set_index("date")


# =============================================================================
# Brinson Performance Attribution
# =============================================================================

def brinson_attribution(
    portfolio_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    sector_map: Optional[Dict[str, str]] = None,
) -> BrinsonResult:
    """
    Compute Brinson-Fachler performance attribution.
    
    Decomposes active return into:
    - Allocation effect: over/underweight in better/worse performing sectors
    - Selection effect: stock picking within sectors
    - Interaction effect: combined allocation and selection
    
    Args:
        portfolio_weights: Portfolio weights (time x asset)
        benchmark_weights: Benchmark weights (time x asset)
        asset_returns: Asset returns (time x asset)
        sector_map: Optional mapping of asset -> sector (for sector-level attribution)
        
    Returns:
        BrinsonResult with attribution effects
    """
    # Align all data
    common_dates = (
        portfolio_weights.index
        .intersection(benchmark_weights.index)
        .intersection(asset_returns.index)
    )
    
    if len(common_dates) == 0:
        return BrinsonResult(
            allocation_effect=pd.Series(dtype=float),
            selection_effect=pd.Series(dtype=float),
            interaction_effect=pd.Series(dtype=float),
            total_active_return=pd.Series(dtype=float),
            total_allocation=0.0,
            total_selection=0.0,
            total_interaction=0.0,
            total_excess=0.0,
        )
    
    port_w = portfolio_weights.reindex(common_dates).fillna(0.0)
    bench_w = benchmark_weights.reindex(common_dates).fillna(0.0)
    returns = asset_returns.reindex(common_dates).fillna(0.0)
    
    # Lag weights for proper attribution
    port_w_lag = port_w.shift(1).fillna(0.0)
    bench_w_lag = bench_w.shift(1).fillna(0.0)
    
    if sector_map is None:
        # Asset-level attribution
        return _brinson_asset_level(port_w_lag, bench_w_lag, returns)
    else:
        # Sector-level attribution
        return _brinson_sector_level(port_w_lag, bench_w_lag, returns, sector_map)


def _brinson_asset_level(
    port_w: pd.DataFrame,
    bench_w: pd.DataFrame,
    returns: pd.DataFrame,
) -> BrinsonResult:
    """Asset-level Brinson attribution."""
    
    # Portfolio and benchmark returns
    port_ret = (port_w * returns).sum(axis=1)
    bench_ret = (bench_w * returns).sum(axis=1)
    active_ret = port_ret - bench_ret
    
    # Weight differences
    w_diff = port_w - bench_w
    
    # Benchmark average return (for allocation)
    bench_avg_ret = (bench_w * returns).sum(axis=1)
    
    # Allocation effect: overweight in outperforming assets
    # Sum of (w_p - w_b) * (r_b - R_b) where R_b is benchmark total return
    allocation = pd.Series(index=returns.index, dtype=float)
    selection = pd.Series(index=returns.index, dtype=float)
    interaction = pd.Series(index=returns.index, dtype=float)
    
    for date in returns.index:
        if date not in port_w.index:
            continue
            
        wp = port_w.loc[date]
        wb = bench_w.loc[date]
        r = returns.loc[date]
        
        # Benchmark return for this period
        rb = (wb * r).sum()
        
        # Asset-level contributions
        alloc_contrib = (wp - wb) * rb
        select_contrib = wb * (r - rb)
        inter_contrib = (wp - wb) * (r - rb)
        
        allocation.loc[date] = alloc_contrib.sum()
        selection.loc[date] = select_contrib.sum()
        interaction.loc[date] = inter_contrib.sum()
    
    return BrinsonResult(
        allocation_effect=allocation.dropna(),
        selection_effect=selection.dropna(),
        interaction_effect=interaction.dropna(),
        total_active_return=active_ret.dropna(),
        total_allocation=float(allocation.sum()),
        total_selection=float(selection.sum()),
        total_interaction=float(interaction.sum()),
        total_excess=float(active_ret.sum()),
    )


def _brinson_sector_level(
    port_w: pd.DataFrame,
    bench_w: pd.DataFrame,
    returns: pd.DataFrame,
    sector_map: Dict[str, str],
) -> BrinsonResult:
    """Sector-level Brinson attribution."""
    
    # Get unique sectors
    sectors = list(set(sector_map.values()))
    
    allocation = pd.Series(index=returns.index, dtype=float)
    selection = pd.Series(index=returns.index, dtype=float)
    interaction = pd.Series(index=returns.index, dtype=float)
    
    for date in returns.index:
        if date not in port_w.index:
            continue
            
        wp = port_w.loc[date]
        wb = bench_w.loc[date]
        r = returns.loc[date]
        
        # Aggregate to sector level
        port_sector_w = {}
        bench_sector_w = {}
        sector_ret_port = {}
        sector_ret_bench = {}
        
        for sector in sectors:
            # Assets in this sector
            sector_assets = [a for a, s in sector_map.items() if s == sector]
            sector_assets = [a for a in sector_assets if a in wp.index]
            
            if not sector_assets:
                continue
            
            # Sector weights
            port_sector_w[sector] = wp[sector_assets].sum()
            bench_sector_w[sector] = wb[sector_assets].sum()
            
            # Sector returns (weighted by within-sector weights)
            if port_sector_w[sector] > 1e-12:
                within_w = wp[sector_assets] / port_sector_w[sector]
                sector_ret_port[sector] = (within_w * r[sector_assets]).sum()
            else:
                sector_ret_port[sector] = 0.0
                
            if bench_sector_w[sector] > 1e-12:
                within_w = wb[sector_assets] / bench_sector_w[sector]
                sector_ret_bench[sector] = (within_w * r[sector_assets]).sum()
            else:
                sector_ret_bench[sector] = 0.0
        
        # Benchmark total return
        rb_total = sum(bench_sector_w.get(s, 0) * sector_ret_bench.get(s, 0) for s in sectors)
        
        # Brinson effects by sector
        alloc_sum = 0.0
        select_sum = 0.0
        inter_sum = 0.0
        
        for sector in sectors:
            wp_s = port_sector_w.get(sector, 0.0)
            wb_s = bench_sector_w.get(sector, 0.0)
            rp_s = sector_ret_port.get(sector, 0.0)
            rb_s = sector_ret_bench.get(sector, 0.0)
            
            # Allocation: sector weight difference * (sector bench return - total bench return)
            alloc_sum += (wp_s - wb_s) * (rb_s - rb_total)
            
            # Selection: bench weight * (port sector return - bench sector return)
            select_sum += wb_s * (rp_s - rb_s)
            
            # Interaction
            inter_sum += (wp_s - wb_s) * (rp_s - rb_s)
        
        allocation.loc[date] = alloc_sum
        selection.loc[date] = select_sum
        interaction.loc[date] = inter_sum
    
    # Total active return
    port_ret = (port_w * returns).sum(axis=1)
    bench_ret = (bench_w * returns).sum(axis=1)
    active_ret = port_ret - bench_ret
    
    return BrinsonResult(
        allocation_effect=allocation.dropna(),
        selection_effect=selection.dropna(),
        interaction_effect=interaction.dropna(),
        total_active_return=active_ret.dropna(),
        total_allocation=float(allocation.sum()),
        total_selection=float(selection.sum()),
        total_interaction=float(interaction.sum()),
        total_excess=float(active_ret.sum()),
    )


# =============================================================================
# Ex-Ante Risk Metrics
# =============================================================================

def compute_ex_ante_risk(
    weights: pd.Series,
    factor_exposures: pd.DataFrame,
    factor_covariance: pd.DataFrame,
    specific_risk: Optional[pd.Series] = None,
    benchmark_weights: Optional[pd.Series] = None,
) -> ExAnteRiskResult:
    """
    Compute ex-ante (predicted) risk metrics.
    
    Uses factor model:
        r = B * f + e
        Var(r) = B * Cov(f) * B' + Var(e)
    
    Args:
        weights: Portfolio weights (asset)
        factor_exposures: Factor loadings (asset x factor)
        factor_covariance: Factor covariance matrix (factor x factor)
        specific_risk: Asset-level specific risk (optional)
        benchmark_weights: Benchmark weights for tracking error (optional)
        
    Returns:
        ExAnteRiskResult with predicted risk metrics
    """
    eps = 1e-12
    
    # Get risk attribution first
    risk_attr = compute_risk_attribution(
        weights=weights,
        factor_exposures=factor_exposures,
        factor_covariance=factor_covariance,
        specific_risk=specific_risk,
        annualize=True,
    )
    
    # Compute tracking error if benchmark provided
    if benchmark_weights is not None:
        active_weights = weights.sub(benchmark_weights, fill_value=0.0)
        te_attr = compute_risk_attribution(
            weights=active_weights,
            factor_exposures=factor_exposures,
            factor_covariance=factor_covariance,
            specific_risk=specific_risk,
            annualize=True,
        )
        tracking_error = te_attr.total_risk
    else:
        tracking_error = 0.0
    
    # Diversification ratio
    # DR = sum(|w| * asset_vol) / portfolio_vol
    if specific_risk is not None and risk_attr.total_risk > eps:
        common = weights.index.intersection(specific_risk.index)
        w = weights.reindex(common).fillna(0.0)
        sv = specific_risk.reindex(common).fillna(0.0)
        undiversified = (w.abs() * sv).sum()
        diversification_ratio = undiversified / risk_attr.total_risk
    else:
        diversification_ratio = 1.0
    
    return ExAnteRiskResult(
        total_risk=risk_attr.total_risk,
        tracking_error=float(tracking_error),
        factor_risk=risk_attr.factor_risk,
        specific_risk=risk_attr.specific_risk,
        factor_contributions=risk_attr.factor_contributions,
        diversification_ratio=float(diversification_ratio),
    )


def compute_factor_covariance(
    factor_returns: pd.DataFrame,
    window: int = 252,
    half_life: Optional[int] = None,
) -> pd.DataFrame:
    """
    Estimate factor covariance matrix from returns.
    
    Args:
        factor_returns: Factor returns (time x factor)
        window: Lookback window
        half_life: Optional half-life for exponential weighting
        
    Returns:
        Factor covariance matrix (factor x factor)
    """
    if half_life is not None:
        # Exponentially weighted covariance
        return factor_returns.tail(window).ewm(halflife=half_life).cov().iloc[-len(factor_returns.columns):]
    else:
        # Simple covariance
        return factor_returns.tail(window).cov()


# =============================================================================
# Forward Risk Projections (Risk Cones)
# =============================================================================

def compute_forward_risk_cone(
    returns: pd.Series,
    horizons: List[int] = [1, 2, 3, 4, 5],
    base_value: float = 100.0,
    expected_return: Optional[float] = None,
    vol_window: int = 63,
) -> ForwardRiskProjection:
    """
    Compute forward-looking risk cone projections for T+1 to T+N days.
    
    Uses current volatility estimate to project portfolio value ranges
    with 1-sigma and 2-sigma confidence bands.
    
    Args:
        returns: Historical portfolio returns (daily)
        horizons: List of forward horizons in days (default [1,2,3,4,5])
        base_value: Starting portfolio value (default 100 for percentage)
        expected_return: Expected daily return (drift). If None, uses historical mean.
        vol_window: Window for volatility estimation (default 63 days / 3 months)
        
    Returns:
        ForwardRiskProjection with confidence bands
        
    Formula:
        daily_vol = ann_vol / sqrt(252)
        T+n expected = base * (1 + drift * n)
        T+n upper_kσ = expected * exp(k * daily_vol * sqrt(n))
        T+n lower_kσ = expected * exp(-k * daily_vol * sqrt(n))
    """
    if returns is None or len(returns) < vol_window:
        # Not enough data - return flat projection
        n_horizons = len(horizons)
        return ForwardRiskProjection(
            horizons=horizons,
            base_value=base_value,
            expected_values=np.full(n_horizons, base_value),
            upper_1sigma=np.full(n_horizons, base_value),
            lower_1sigma=np.full(n_horizons, base_value),
            upper_2sigma=np.full(n_horizons, base_value),
            lower_2sigma=np.full(n_horizons, base_value),
            daily_vol=0.0,
            annual_vol=0.0,
            expected_daily_return=0.0,
        )
    
    # Calculate volatility from recent returns
    recent_returns = returns.tail(vol_window)
    daily_vol = float(recent_returns.std())
    annual_vol = daily_vol * np.sqrt(252)
    
    # Expected daily return (drift)
    if expected_return is None:
        expected_daily_return = float(recent_returns.mean())
    else:
        expected_daily_return = expected_return
    
    # Project forward for each horizon
    n_horizons = len(horizons)
    expected_values = np.zeros(n_horizons)
    upper_1sigma = np.zeros(n_horizons)
    lower_1sigma = np.zeros(n_horizons)
    upper_2sigma = np.zeros(n_horizons)
    lower_2sigma = np.zeros(n_horizons)
    
    for i, t in enumerate(horizons):
        # Expected value at horizon t
        expected_val = base_value * (1 + expected_daily_return * t)
        expected_values[i] = expected_val
        
        # Volatility scales with sqrt(time)
        vol_at_t = daily_vol * np.sqrt(t)
        
        # 1-sigma bounds (68% confidence)
        upper_1sigma[i] = expected_val * np.exp(1.0 * vol_at_t)
        lower_1sigma[i] = expected_val * np.exp(-1.0 * vol_at_t)
        
        # 2-sigma bounds (95% confidence)
        upper_2sigma[i] = expected_val * np.exp(2.0 * vol_at_t)
        lower_2sigma[i] = expected_val * np.exp(-2.0 * vol_at_t)
    
    return ForwardRiskProjection(
        horizons=horizons,
        base_value=base_value,
        expected_values=expected_values,
        upper_1sigma=upper_1sigma,
        lower_1sigma=lower_1sigma,
        upper_2sigma=upper_2sigma,
        lower_2sigma=lower_2sigma,
        daily_vol=daily_vol,
        annual_vol=annual_vol,
        expected_daily_return=expected_daily_return,
    )


# =============================================================================
# Expected Return Estimation
# =============================================================================

def compute_expected_return_ic(
    factor_exposures: pd.Series,
    ic_values: pd.Series,
    factor_weights: Optional[pd.Series] = None,
    scale_factor: float = 1.0,
) -> float:
    """
    Compute IC-based expected return from factor exposures.
    
    Expected return = Σ (factor_exposure_i × IC_i × weight_i)
    
    Args:
        factor_exposures: Current factor exposures (factor -> exposure)
        ic_values: Smoothed IC values for each factor (factor -> IC)
        factor_weights: Optional factor weights (default: equal weight)
        scale_factor: Scaling factor for return estimate
        
    Returns:
        Expected daily return based on IC model
    """
    # Align factors
    common_factors = factor_exposures.index.intersection(ic_values.index)
    if len(common_factors) == 0:
        return 0.0
    
    exp = factor_exposures.reindex(common_factors).fillna(0.0)
    ic = ic_values.reindex(common_factors).fillna(0.0)
    
    if factor_weights is not None:
        weights = factor_weights.reindex(common_factors).fillna(0.0)
        # Normalize weights
        weight_sum = weights.abs().sum()
        if weight_sum > 1e-12:
            weights = weights / weight_sum
        else:
            weights = pd.Series(1.0 / len(common_factors), index=common_factors)
    else:
        # Equal weight
        weights = pd.Series(1.0 / len(common_factors), index=common_factors)
    
    # Expected return = Σ (exposure × IC × weight)
    expected_ret = (exp * ic * weights).sum() * scale_factor
    
    return float(expected_ret)


def compute_expected_return_drift(
    returns: pd.Series,
    window: int = 21,
) -> Tuple[float, float, float]:
    """
    Compute drift-based expected return from historical returns.
    
    Args:
        returns: Historical portfolio returns
        window: Lookback window for drift estimation (default 21 days)
        
    Returns:
        Tuple of (expected_return, mean, std)
    """
    if returns is None or len(returns) < window:
        return 0.0, 0.0, 0.0
    
    recent = returns.tail(window)
    drift = float(recent.mean())
    mean = float(returns.mean())
    std = float(returns.std())
    
    return drift, mean, std


def compute_signal_freshness(
    ic_series: pd.DataFrame,
    lookback: int = 21,
) -> Tuple[float, str]:
    """
    Compute signal freshness indicator based on IC trend.
    
    Freshness measures how well the IC signals are performing recently
    compared to their historical levels.
    
    Args:
        ic_series: IC values over time (time x factor)
        lookback: Lookback window for trend
        
    Returns:
        Tuple of (freshness_score 0-1, trend description)
    """
    if ic_series is None or ic_series.empty or len(ic_series) < lookback * 2:
        return 0.5, "unknown"
    
    # Compute average IC across factors
    avg_ic = ic_series.abs().mean(axis=1)
    
    if len(avg_ic) < lookback * 2:
        return 0.5, "unknown"
    
    # Recent IC vs historical
    recent_ic = avg_ic.tail(lookback).mean()
    historical_ic = avg_ic.iloc[:-lookback].mean()
    
    if historical_ic < 1e-12:
        return 0.5, "unknown"
    
    # Freshness ratio (capped at 0-1)
    ratio = recent_ic / historical_ic
    freshness = min(max(ratio, 0.0), 1.0)
    
    # Trend determination
    if ratio > 1.1:
        trend = "improving"
    elif ratio < 0.9:
        trend = "declining"
    else:
        trend = "stable"
    
    return float(freshness), trend


def compute_expected_return_estimate(
    returns: pd.Series,
    factor_exposures: Optional[pd.Series] = None,
    ic_values: Optional[pd.Series] = None,
    factor_weights: Optional[pd.Series] = None,
    ic_series: Optional[pd.DataFrame] = None,
    ic_weight: float = 0.6,
    drift_window: int = 21,
    history_window: int = 252,
) -> ExpectedReturnEstimate:
    """
    Compute comprehensive expected return estimate for PM dashboard.
    
    Combines IC-based and drift-based estimates, computes z-score vs history,
    and generates alerts when predictions are significantly abnormal.
    
    Args:
        returns: Historical portfolio returns
        factor_exposures: Current factor exposures (optional)
        ic_values: Current smoothed IC values (optional)
        factor_weights: Factor weights (optional)
        ic_series: Historical IC values for freshness calc (optional)
        ic_weight: Weight for IC-based estimate (0-1)
        drift_window: Window for drift estimate
        history_window: Window for historical comparison
        
    Returns:
        ExpectedReturnEstimate with all metrics and alerts
    """
    # Compute IC-based return
    if factor_exposures is not None and ic_values is not None:
        ic_return = compute_expected_return_ic(factor_exposures, ic_values, factor_weights)
    else:
        ic_return = 0.0
        ic_weight = 0.0  # Can't use IC if not available
    
    # Compute drift-based return
    drift_return, hist_mean, hist_std = compute_expected_return_drift(returns, drift_window)
    
    # Combined estimate
    if ic_weight > 0 and ic_return != 0:
        combined = ic_weight * ic_return + (1 - ic_weight) * drift_return
    else:
        combined = drift_return
    
    # Historical comparison
    if returns is not None and len(returns) >= history_window:
        # Compute historical expected returns (rolling drift)
        historical_drifts = returns.rolling(drift_window).mean().dropna()
        if len(historical_drifts) > 0:
            historical_mean = float(historical_drifts.mean())
            historical_std = float(historical_drifts.std())
            
            if historical_std > 1e-12:
                z_score = (combined - historical_mean) / historical_std
                # Percentile
                percentile = float(stats.percentileofscore(historical_drifts.values, combined))
            else:
                z_score = 0.0
                percentile = 50.0
        else:
            historical_mean = 0.0
            historical_std = 0.0
            z_score = 0.0
            percentile = 50.0
    else:
        historical_mean = hist_mean
        historical_std = hist_std
        z_score = (combined - hist_mean) / (hist_std + 1e-12) if hist_std > 1e-12 else 0.0
        percentile = 50.0
    
    # Signal freshness
    if ic_series is not None:
        freshness, trend = compute_signal_freshness(ic_series)
    else:
        freshness = 0.5
        trend = "unknown"
    
    # Alert logic
    is_low = z_score < -1.5
    is_high = z_score > 1.5
    
    alert_msg = None
    if is_low:
        alert_msg = f"⚠️ Expected return ({combined:.3%}) is {abs(z_score):.1f}σ below historical average. Consider reducing exposure."
    elif is_high:
        alert_msg = f"📈 Expected return ({combined:.3%}) is {z_score:.1f}σ above average. Favorable conditions."
    elif freshness < 0.7 and trend == "declining":
        alert_msg = f"📉 Signal freshness at {freshness:.0%}. IC signals may be decaying."
    
    return ExpectedReturnEstimate(
        ic_based_return=float(ic_return),
        drift_based_return=float(drift_return),
        combined_estimate=float(combined),
        z_score_vs_history=float(z_score),
        percentile=float(percentile),
        historical_mean=float(historical_mean),
        historical_std=float(historical_std),
        signal_freshness=float(freshness),
        ic_trend=trend,
        is_significantly_low=is_low,
        is_significantly_high=is_high,
        alert_message=alert_msg,
    )


# =============================================================================
# Sector/Industry Attribution
# =============================================================================

def compute_sector_attribution(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    sector_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Compute performance attribution by sector.
    
    Args:
        weights: Portfolio weights (time x asset)
        returns: Asset returns (time x asset)
        sector_map: Mapping of asset -> sector
        
    Returns:
        DataFrame with sector-level P&L contribution (time x sector)
    """
    # Get sectors
    sectors = list(set(sector_map.values()))
    
    # Align data
    common_dates = weights.index.intersection(returns.index)
    weights = weights.reindex(common_dates)
    returns = returns.reindex(common_dates)
    
    # Lag weights
    w_lag = weights.shift(1).fillna(0.0)
    
    # Compute contribution by sector
    sector_contrib = pd.DataFrame(index=common_dates, columns=sectors, dtype=float)
    sector_weights = pd.DataFrame(index=common_dates, columns=sectors, dtype=float)
    
    for sector in sectors:
        sector_assets = [a for a, s in sector_map.items() if s == sector]
        sector_assets = [a for a in sector_assets if a in weights.columns and a in returns.columns]
        
        if sector_assets:
            # Contribution = lagged weight * return
            contrib = (w_lag[sector_assets] * returns[sector_assets]).sum(axis=1)
            sector_contrib[sector] = contrib
            sector_weights[sector] = weights[sector_assets].sum(axis=1)
        else:
            sector_contrib[sector] = 0.0
            sector_weights[sector] = 0.0
    
    return sector_contrib.fillna(0.0)


def compute_sector_exposure_timeseries(
    weights: pd.DataFrame,
    sector_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Compute sector exposure over time.
    
    Args:
        weights: Portfolio weights (time x asset)
        sector_map: Mapping of asset -> sector
        
    Returns:
        DataFrame with sector weights over time (time x sector)
    """
    sectors = list(set(sector_map.values()))
    
    sector_weights = pd.DataFrame(index=weights.index, columns=sectors, dtype=float)
    
    for sector in sectors:
        sector_assets = [a for a, s in sector_map.items() if s == sector]
        sector_assets = [a for a in sector_assets if a in weights.columns]
        
        if sector_assets:
            sector_weights[sector] = weights[sector_assets].sum(axis=1)
        else:
            sector_weights[sector] = 0.0
    
    return sector_weights.fillna(0.0)


def compute_sector_risk_contribution(
    weights: pd.Series,
    returns: pd.DataFrame,
    sector_map: Dict[str, str],
    window: int = 252,
) -> pd.DataFrame:
    """
    Compute risk contribution by sector.
    
    Args:
        weights: Current portfolio weights (asset)
        returns: Historical asset returns (time x asset)
        sector_map: Mapping of asset -> sector
        window: Lookback window for covariance
        
    Returns:
        DataFrame with sector risk metrics
    """
    sectors = list(set(sector_map.values()))
    
    # Compute asset covariance
    common_assets = weights.index.intersection(returns.columns)
    w = weights.reindex(common_assets).fillna(0.0)
    ret = returns[list(common_assets)].tail(window)
    
    cov = ret.cov() * 252  # Annualized
    
    # Portfolio variance
    port_var = w.values @ cov.values @ w.values
    port_vol = np.sqrt(max(port_var, 0.0))
    
    results = []
    
    for sector in sectors:
        sector_assets = [a for a, s in sector_map.items() if s == sector]
        sector_assets = [a for a in sector_assets if a in common_assets]
        
        if not sector_assets:
            results.append({
                "sector": sector,
                "weight": 0.0,
                "risk_contribution": 0.0,
                "pct_risk": 0.0,
            })
            continue
        
        # Sector weight
        sector_w = w[sector_assets].sum()
        
        # Marginal contribution to risk
        mcr = cov.values @ w.values
        sector_mcr = sum(w[a] * mcr[list(common_assets).index(a)] for a in sector_assets)
        
        results.append({
            "sector": sector,
            "weight": float(sector_w),
            "risk_contribution": float(np.sqrt(max(sector_mcr, 0.0))),
            "pct_risk": float(sector_mcr / (port_var + 1e-12)),
        })
    
    return pd.DataFrame(results)


def compute_industry_concentration(
    weights: pd.Series,
    sector_map: Dict[str, str],
) -> Dict[str, float]:
    """
    Compute industry/sector concentration metrics.
    
    Args:
        weights: Portfolio weights (asset)
        sector_map: Mapping of asset -> sector
        
    Returns:
        Dictionary with concentration metrics
    """
    sectors = list(set(sector_map.values()))
    
    sector_weights = {}
    for sector in sectors:
        sector_assets = [a for a, s in sector_map.items() if s == sector]
        sector_assets = [a for a in sector_assets if a in weights.index]
        sector_weights[sector] = weights[sector_assets].sum() if sector_assets else 0.0
    
    sector_w_series = pd.Series(sector_weights)
    
    # Herfindahl index
    hhi = (sector_w_series ** 2).sum()
    
    # Effective number of sectors
    eff_sectors = 1.0 / (hhi + 1e-12) if hhi > 0 else len(sectors)
    
    # Top sector concentration
    top3_conc = sector_w_series.abs().nlargest(3).sum()
    
    return {
        "herfindahl_index": float(hhi),
        "effective_sectors": float(eff_sectors),
        "top3_concentration": float(top3_conc),
        "n_sectors": len([s for s in sectors if abs(sector_weights.get(s, 0)) > 1e-12]),
        "largest_sector": sector_w_series.abs().idxmax() if len(sector_w_series) > 0 else None,
        "largest_sector_weight": float(sector_w_series.abs().max()) if len(sector_w_series) > 0 else 0.0,
    }


# =============================================================================
# Correlation Analysis
# =============================================================================

def compute_correlation_analysis(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    window: int = 63,
) -> pd.DataFrame:
    """
    Compute rolling correlations between portfolio and factors/benchmark.
    
    Args:
        portfolio_returns: Portfolio return series (time)
        factor_returns: Factor returns (time x factor)
        benchmark_returns: Optional benchmark returns (time)
        window: Rolling window for correlation
        
    Returns:
        DataFrame with rolling correlations (time x factor/benchmark)
    """
    # Align data
    common_idx = portfolio_returns.index.intersection(factor_returns.index)
    if benchmark_returns is not None:
        common_idx = common_idx.intersection(benchmark_returns.index)
    
    port_ret = portfolio_returns.reindex(common_idx)
    fact_ret = factor_returns.reindex(common_idx)
    
    # Rolling correlations with factors
    corr_results = pd.DataFrame(index=common_idx)
    
    for factor in fact_ret.columns:
        corr_results[f"corr_{factor}"] = port_ret.rolling(window).corr(fact_ret[factor])
    
    # Benchmark correlation
    if benchmark_returns is not None:
        bench_ret = benchmark_returns.reindex(common_idx)
        corr_results["corr_benchmark"] = port_ret.rolling(window).corr(bench_ret)
    
    return corr_results.dropna()


def compute_correlation_matrix(
    returns: pd.DataFrame,
    window: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute correlation matrix for assets or factors.
    
    Args:
        returns: Returns (time x asset/factor)
        window: Optional window to use (uses all data if None)
        
    Returns:
        Correlation matrix
    """
    if window is not None:
        returns = returns.tail(window)
    
    return returns.corr()


def compute_beta_analysis(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """
    Compute rolling betas to factors.
    
    Args:
        portfolio_returns: Portfolio returns (time)
        factor_returns: Factor returns (time x factor)
        window: Rolling window
        
    Returns:
        DataFrame with rolling betas (time x factor)
    """
    common_idx = portfolio_returns.index.intersection(factor_returns.index)
    port_ret = portfolio_returns.reindex(common_idx)
    fact_ret = factor_returns.reindex(common_idx)
    
    betas = pd.DataFrame(index=common_idx, columns=fact_ret.columns, dtype=float)
    
    for i in range(window, len(common_idx)):
        date = common_idx[i]
        start = i - window
        
        y = port_ret.iloc[start:i].values
        
        for factor in fact_ret.columns:
            x = fact_ret[factor].iloc[start:i].values
            
            # Simple OLS beta
            cov_xy = np.cov(x, y)[0, 1]
            var_x = np.var(x)
            
            betas.loc[date, factor] = cov_xy / (var_x + 1e-12)
    
    return betas.dropna()


def compute_up_down_capture(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    threshold: float = 0.0,
) -> Dict[str, float]:
    """
    Compute up/down capture ratios.
    
    Args:
        portfolio_returns: Portfolio returns (time)
        benchmark_returns: Benchmark returns (time)
        threshold: Threshold for up/down classification (default 0)
        
    Returns:
        Dictionary with capture ratios
    """
    # Align
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
    port = portfolio_returns.reindex(common_idx)
    bench = benchmark_returns.reindex(common_idx)
    
    # Up days
    up_mask = bench > threshold
    down_mask = bench < threshold
    
    # Capture ratios
    if up_mask.sum() > 0:
        up_capture = port[up_mask].mean() / (bench[up_mask].mean() + 1e-12)
    else:
        up_capture = 1.0
    
    if down_mask.sum() > 0:
        down_capture = port[down_mask].mean() / (bench[down_mask].mean() + 1e-12)
    else:
        down_capture = 1.0
    
    # Batting average (how often beat benchmark)
    batting_avg = ((port > bench) | ((port == bench) & (port > 0))).mean()
    
    return {
        "up_capture": float(up_capture),
        "down_capture": float(down_capture),
        "capture_ratio": float(up_capture / (down_capture + 1e-12)),
        "batting_average": float(batting_avg),
        "up_days": int(up_mask.sum()),
        "down_days": int(down_mask.sum()),
    }


# =============================================================================
# Utility Functions
# =============================================================================

def estimate_specific_risk(
    returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    factor_exposures: pd.DataFrame,
    window: int = 252,
) -> pd.Series:
    """
    Estimate asset-level specific (idiosyncratic) risk.
    
    Uses residuals from factor model regression.
    
    Args:
        returns: Asset returns (time x asset)
        factor_returns: Factor returns (time x factor)
        factor_exposures: Factor exposures (asset x factor)
        window: Lookback window
        
    Returns:
        Series of specific risk by asset
    """
    specific_risk = {}
    
    for asset in returns.columns:
        if asset not in factor_exposures.index:
            specific_risk[asset] = returns[asset].std() * np.sqrt(252)
            continue
        
        # Get factor exposures for this asset
        B = factor_exposures.loc[asset]
        common_factors = B.index.intersection(factor_returns.columns)
        
        if len(common_factors) == 0:
            specific_risk[asset] = returns[asset].std() * np.sqrt(252)
            continue
        
        # Compute residual returns
        factor_contrib = (factor_returns[common_factors] * B[common_factors]).sum(axis=1)
        common_dates = returns[asset].index.intersection(factor_contrib.index)
        
        residuals = returns[asset].reindex(common_dates) - factor_contrib.reindex(common_dates)
        specific_risk[asset] = residuals.tail(window).std() * np.sqrt(252)
    
    return pd.Series(specific_risk)
