"""OUv1 Factor Library - Ornstein-Uhlenbeck Based Factors.

Implements novel factors derived from OU process estimation:

The OU process: dX = θ(μ - X)dt + σdW

Estimation approach:
1. Fit AR(1) model: X[t] = a + b*X[t-1] + ε
2. Extract: θ = -ln(b), μ = a/(1-b), σ = std(ε)/sqrt(1-b²)
3. Half-life = ln(2)/θ

Cross-sectional application:
- Estimate OU parameters for each stock's price deviation from trend
- Use half-life to determine signal strength
- Z-score measures deviation from equilibrium

SOLID Principles Applied:
- Single Responsibility: Only handles OU-specific factor computation
- Open/Closed: Extends FactorLibrary without modifying it
- Liskov Substitution: Can be used wherever FactorLibrary is expected
- Interface Segregation: Factor methods follow _factor_* naming convention
- Dependency Inversion: Depends on abstract FactorLibrary, not concrete implementation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, TYPE_CHECKING, Union, Any

import numpy as np
import pandas as pd

try:
    import xarray as xr
except ImportError:
    xr = None

from q23.strategy.factors import FactorLibrary, FactorParams, _robust_zscore

if TYPE_CHECKING:
    from q23.strategy.data_loader import MarketDataBundle


def z_score_cs(x: "xr.DataArray", eps: float = 1e-12) -> "xr.DataArray":
    """Cross-sectional z-score (robust)."""
    return _robust_zscore(x, dim="asset", eps=eps, clip=5.0)


class OUFactorLibrary(FactorLibrary):
    """Extended factor library with Ornstein-Uhlenbeck factors.
    
    Adds 8 novel OU-based factors to the standard factor library.
    Follows Open/Closed principle - extends without modifying base.
    """
    
    def __init__(
        self,
        market_data: Union["xr.Dataset", Any],
        *,
        params: Optional[FactorParams] = None,
        factors: Optional[Sequence[str]] = None,
        asset_dim: str = "asset",
    ):
        # Store bundle reference if it's a MarketDataBundle
        self.bundle = market_data
        
        # Initialize parent with factor list
        super().__init__(market_data, params=params, factors=factors, asset_dim=asset_dim)
        
        # OU-specific config from params (use getattr for flexibility)
        p = self.params
        self.ou_short_win = getattr(p, 'OU_SHORT_WIN', 21)
        self.ou_med_win = getattr(p, 'OU_MED_WIN', 63)
        self.ou_long_win = getattr(p, 'OU_LONG_WIN', 126)
        self.ou_halflife_min = getattr(p, 'OU_HALFLIFE_MIN', 2)
        self.ou_halflife_max = getattr(p, 'OU_HALFLIFE_MAX', 42)
        self.ou_zscore_clip = getattr(p, 'OU_ZSCORE_CLIP', 3.0)
        self.eps = p.eps
    
    def _estimate_ou_params(
        self,
        series: "xr.DataArray",
        window: int,
    ) -> Dict[str, "xr.DataArray"]:
        """Estimate OU parameters using rolling AR(1) regression.
        
        For X[t] = a + b*X[t-1] + ε:
        - θ (kappa) = -ln(b) / dt
        - μ = a / (1 - b)
        - σ = std(ε) / sqrt(1 - b²)
        - half_life = ln(2) / θ
        
        Returns dict with: theta, mu, sigma, half_life, zscore
        """
        # Use self.close from parent class (already loaded)
        close = self.close
        
        log_price = np.log(close.where(close > 0))
        
        # Rolling AR(1) estimation via vectorized operations
        # Using: b = cov(X[t], X[t-1]) / var(X[t-1])
        #        a = mean(X[t]) - b * mean(X[t-1])
        
        X = log_price
        X_lag = X.shift(time=1)
        
        # Rolling mean
        X_mean = X.rolling(time=window, min_periods=window//2).mean()
        X_lag_mean = X_lag.rolling(time=window, min_periods=window//2).mean()
        
        # Rolling variance and covariance
        X_var = X_lag.rolling(time=window, min_periods=window//2).var()
        
        # Covariance: E[XY] - E[X]E[Y]
        XY = X * X_lag
        XY_mean = XY.rolling(time=window, min_periods=window//2).mean()
        cov_XY = XY_mean - X_mean * X_lag_mean
        
        # AR(1) coefficient b
        b = cov_XY / (X_var + self.eps)
        b = xr.where(np.abs(b) < 0.9999, b, 0.9999 * np.sign(b))  # Clip for stability
        
        # Intercept a
        a = X_mean - b * X_lag_mean
        
        # OU parameters (dt = 1 day)
        theta = -np.log(np.abs(b) + self.eps)  # Mean-reversion speed
        theta = xr.where(theta > 0, theta, self.eps)
        
        mu = a / (1 - b + self.eps)  # Long-term mean
        
        # Residual volatility
        predicted = a + b * X_lag
        residuals = X - predicted
        sigma = residuals.rolling(time=window, min_periods=window//2).std()
        
        # Half-life in days
        half_life = np.log(2) / (theta + self.eps)
        half_life = xr.where(half_life > 0, half_life, self.ou_halflife_max)
        half_life = half_life.clip(self.ou_halflife_min, self.ou_halflife_max)
        
        # Z-score: (current - equilibrium) / sigma
        deviation = X - mu
        zscore = deviation / (sigma + self.eps)
        zscore = zscore.clip(-self.ou_zscore_clip, self.ou_zscore_clip)
        
        return {
            "theta": theta,
            "mu": mu,
            "sigma": sigma,
            "half_life": half_life,
            "zscore": zscore,
            "b": b,
        }
    
    # =========================================================================
    # OU NOVEL FACTORS
    # =========================================================================
    
    def _factor_ou_zscore_short(self) -> "xr.DataArray":
        """Short-term OU z-score (21d window).
        
        Measures deviation from short-term equilibrium.
        Negative z-score = undervalued → expect positive return.
        """
        ou_params = self._estimate_ou_params(self.ret, self.ou_short_win)
        
        # Invert sign: low z-score (undervalued) = high expected return
        factor = -ou_params["zscore"]
        return z_score_cs(factor, self.eps)
    
    def _factor_ou_zscore_med(self) -> "xr.DataArray":
        """Medium-term OU z-score (63d window).
        
        More stable equilibrium estimate over 3 months.
        """
        ou_params = self._estimate_ou_params(self.ret, self.ou_med_win)
        
        factor = -ou_params["zscore"]
        return z_score_cs(factor, self.eps)
    
    def _factor_ou_halflife_signal(self) -> "xr.DataArray":
        """Half-life based signal strength.
        
        Shorter half-life = faster mean reversion = stronger signal.
        Inverted so high value = strong reversion.
        """
        ou_params = self._estimate_ou_params(self.ret, self.ou_med_win)
        
        half_life = ou_params["half_life"]
        
        # Inverse half-life: shorter = better
        # Normalize: 1/halflife, then z-score
        inv_halflife = 1.0 / (half_life + 1)
        
        return z_score_cs(inv_halflife, self.eps)
    
    def _factor_ou_reversion_strength(self) -> "xr.DataArray":
        """Mean-reversion speed (θ) as factor.
        
        Higher θ = faster reversion = stronger mean-reversion signal validity.
        Cross-sectionally ranked.
        """
        ou_params = self._estimate_ou_params(self.ret, self.ou_med_win)
        
        theta = ou_params["theta"]
        return z_score_cs(theta, self.eps)
    
    def _factor_ou_predicted_return(self) -> "xr.DataArray":
        """Expected return from OU dynamics.
        
        E[X[t+1] - X[t]] = θ(μ - X[t])
        Stocks far below equilibrium with high θ have high expected returns.
        """
        log_price = np.log(self.close.where(self.close > 0))
        
        ou_params = self._estimate_ou_params(self.ret, self.ou_med_win)
        
        theta = ou_params["theta"]
        mu = ou_params["mu"]
        
        # Expected return: θ(μ - X)
        predicted_return = theta * (mu - log_price)
        
        return z_score_cs(predicted_return, self.eps)
    
    def _factor_ou_regime_indicator(self) -> "xr.DataArray":
        """Cross-sectional OU parameter dispersion as regime indicator.
        
        High dispersion in half-lives = diverse market conditions.
        Stocks with median half-life = most reliable signals.
        """
        ou_params = self._estimate_ou_params(self.ret, self.ou_med_win)
        
        half_life = ou_params["half_life"]
        
        # Compute cross-sectional median half-life
        hl_median = half_life.median(dim="asset")
        
        # Distance from median (closer = more reliable)
        dist_from_median = np.abs(half_life - hl_median)
        
        # Invert: closer to median = higher score
        factor = -dist_from_median
        
        return z_score_cs(factor, self.eps)
    
    def _factor_ou_equilibrium_dist(self) -> "xr.DataArray":
        """Distance from equilibrium (normalized).
        
        How far price is from OU-estimated equilibrium.
        Combined with reversion strength for actionable signal.
        """
        log_price = np.log(self.close.where(self.close > 0))
        
        ou_short = self._estimate_ou_params(self.ret, self.ou_short_win)
        ou_med = self._estimate_ou_params(self.ret, self.ou_med_win)
        
        # Blend short and medium equilibrium estimates
        mu_blend = 0.3 * ou_short["mu"] + 0.7 * ou_med["mu"]
        sigma_blend = 0.3 * ou_short["sigma"] + 0.7 * ou_med["sigma"]
        
        # Normalized distance
        dist = (log_price - mu_blend) / (sigma_blend + self.eps)
        
        # Invert: below equilibrium = positive expected return
        factor = -dist.clip(-self.ou_zscore_clip, self.ou_zscore_clip)
        
        return z_score_cs(factor, self.eps)
    
    def _factor_ou_momentum_blend(self) -> "xr.DataArray":
        """OU-weighted momentum blend.
        
        Short half-life stocks: favor reversal signals
        Long half-life stocks: favor momentum signals
        
        This creates an adaptive momentum/reversal factor.
        """
        ou_params = self._estimate_ou_params(self.ret, self.ou_med_win)
        
        half_life = ou_params["half_life"]
        
        # Compute momentum (positive = recent gains)
        mom_21 = self.ret.rolling(time=21, min_periods=15).sum()
        mom_63 = self.ret.rolling(time=63, min_periods=42).sum()
        
        # Compute reversal signal (negative recent returns = expect reversal up)
        rev_5 = -self.ret.rolling(time=5, min_periods=3).sum()
        
        # Normalize half-life to [0, 1] weight
        # 0 = very short half-life (favor reversal)
        # 1 = very long half-life (favor momentum)
        hl_norm = (half_life - self.ou_halflife_min) / (self.ou_halflife_max - self.ou_halflife_min)
        hl_norm = hl_norm.clip(0, 1)
        
        # Blend: low hl_norm = more reversal, high hl_norm = more momentum
        momentum_component = 0.6 * z_score_cs(mom_21, self.eps) + 0.4 * z_score_cs(mom_63, self.eps)
        reversal_component = z_score_cs(rev_5, self.eps)
        
        # Weighted blend
        factor = hl_norm * momentum_component + (1 - hl_norm) * reversal_component
        
        return z_score_cs(factor, self.eps)
