"""Q23 Neural Alpha - Novel Factor Implementations.

This module extends the base FactorLibrary with 15 novel research-backed factors:

Behavioral Finance Factors:
    - attention_momentum: Volume-price attention signal
    - disposition_alpha: Unrealized P&L vs anchoring price  
    - anchoring_bias: Distance from psychological price levels

Market Microstructure Factors:
    - efficiency_ratio: Kaufman Efficiency Ratio
    - information_flow: Volume-weighted price impact asymmetry
    - mean_reversion_speed: Half-life of price deviations

Momentum Quality Factors:
    - momentum_quality_ratio: Signal-to-noise of momentum
    - momentum_persistence: Autocorrelation-based momentum strength
    - momentum_divergence: Price vs residual momentum divergence

Risk/Regime Factors:
    - vol_of_vol: Volatility of volatility
    - skewness_factor: Rolling return skewness
    - kurtosis_factor: Rolling return kurtosis
    - regime_momentum: Volatility-regime-adaptive momentum

Cross-Sectional Factors:
    - cross_sectional_dispersion: Return dispersion opportunity
    - liquidity_momentum: Liquidity-adjusted momentum
"""

from __future__ import annotations

from typing import Optional, Sequence, Union, Any, Dict

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import xarray as xr
except ImportError:
    xr = None

from q23.strategy.factors import (
    FactorLibrary,
    FactorParams,
    _sma,
    _std,
    _ema,
    _rolling_max,
    _rolling_min,
    _atr,
)


def _require_xr() -> None:
    if xr is None:
        raise ImportError("xarray is required for neural_alpha.factors")


class NeuralAlphaFactorLibrary(FactorLibrary):
    """Extended FactorLibrary with 15 novel Neural Alpha factors.
    
    Inherits all base factors and adds:
    - Behavioral finance factors (attention, disposition, anchoring)
    - Market microstructure factors (efficiency, information flow, mean reversion)
    - Momentum quality factors (quality ratio, persistence, divergence)
    - Risk/regime factors (vol-of-vol, skewness, kurtosis, regime momentum)
    - Cross-sectional factors (dispersion, liquidity momentum)
    """
    
    def __init__(
        self,
        market_data: Union["xr.Dataset", Any],
        *,
        params: Optional[FactorParams] = None,
        factors: Sequence[str] = None,
        asset_dim: str = "asset",
    ):
        """Initialize with market data and factor configuration.
        
        Args:
            market_data: xr.Dataset or bundle-like with .data containing OHLCV
            params: FactorParams for window configuration
            factors: List of factor names to compute
            asset_dim: Name of asset dimension (default: "asset")
        """
        from q23.strategies.q23_neural_alpha.config import NEURAL_ALPHA_FACTORS
        
        if factors is None:
            factors = NEURAL_ALPHA_FACTORS
            
        super().__init__(
            market_data,
            params=params,
            factors=factors,
            asset_dim=asset_dim,
        )
    
    # =========================================================================
    # BEHAVIORAL FINANCE FACTORS
    # =========================================================================
    
    def _factor_attention_momentum(self) -> "xr.DataArray":
        """Attention-weighted momentum signal.
        
        High volume combined with positive returns indicates attention buying.
        This factor captures investor attention flows based on volume-price dynamics.
        
        Formula: sign(ret) * (vol_ratio - 1) * |ret_zscore|
        Where vol_ratio = current_vol / avg_vol
        
        Research basis: Barber & Odean (2008), investor attention and trading
        """
        p = self.params
        
        # Volume ratio (current vs average)
        vol_avg = _sma(self.vol, p.ADV_WIN_LONG)
        vol_ratio = self.vol / (vol_avg + p.eps)
        vol_ratio = vol_ratio.fillna(1.0).clip(0.1, 10.0)
        
        # Attention signal: volume spike * return direction * return magnitude
        ret_std = _std(self.ret, p.MOM_SHORT)
        ret_zscore = self.ret / (ret_std + p.eps)
        ret_zscore = ret_zscore.fillna(0.0).clip(-3, 3)
        
        # Combine: attention buying = high volume + positive returns
        attention = np.sign(self.ret) * (vol_ratio - 1.0) * np.abs(ret_zscore)
        
        # Smooth over attention window
        attention_win = getattr(p, 'ATTENTION_WIN', 10)
        return _sma(attention, attention_win).fillna(0.0)
    
    def _factor_disposition_alpha(self) -> "xr.DataArray":
        """Disposition effect alpha signal.
        
        Measures unrealized P&L relative to anchoring price (52-week midpoint).
        Stocks with unrealized gains tend to be sold (negative future returns),
        while stocks with unrealized losses are held (continuation).
        
        We invert this: prefer stocks in "paper loss" territory that are recovering.
        
        Research basis: Shefrin & Statman (1985), Grinblatt & Han (2005)
        """
        p = self.params
        
        # Anchoring price: 52-week high-low midpoint
        high_52w = _rolling_max(self.high, 252)
        low_52w = _rolling_min(self.low, 252)
        anchor_price = (high_52w + low_52w) / 2.0
        
        # Unrealized gain/loss relative to anchor
        unrealized_pnl = (self.close - anchor_price) / (anchor_price + p.eps)
        
        # Disposition signal: favor stocks in paper loss territory
        # (contrarian to disposition effect selling pressure)
        disposition = -unrealized_pnl  # Negative unrealized = paper loss = buy signal
        
        # Add momentum confirmation: only if recent momentum is positive
        mom_short = _sma(self.ret, p.MOM_SHORT)
        recovery_signal = (mom_short > 0).astype(float)
        
        # Combined: paper loss + recovering = strong buy
        disp_win = getattr(p, 'DISPOSITION_WIN', 63)
        return _sma(disposition * (0.5 + 0.5 * recovery_signal), disp_win).fillna(0.0)
    
    def _factor_anchoring_bias(self) -> "xr.DataArray":
        """Anchoring bias factor.
        
        Measures proximity to psychologically significant price levels:
        - Distance from round numbers (psychological support/resistance)
        - Distance from 52-week high (ceiling effect)
        - Distance from 52-week low (floor effect)
        
        Stocks near round numbers or key levels exhibit predictable behavior.
        
        Research basis: Bhattacharya et al. (2012), round number anchoring
        """
        p = self.params
        
        # 52-week high proximity (inverted - near high = momentum)
        high_52w = _rolling_max(self.high, 252)
        high_prox = self.close / (high_52w + p.eps)
        high_prox = high_prox.fillna(0.5).clip(0, 1)
        
        # 52-week low proximity (above low = strength)
        low_52w = _rolling_min(self.low, 252)
        low_prox = (self.close - low_52w) / (high_52w - low_52w + p.eps)
        low_prox = low_prox.fillna(0.5).clip(0, 1)
        
        # Combined anchoring score
        # Favor stocks that are:
        # 1. Near 52w high (momentum breakout potential)
        # 2. Well above 52w low (strength)
        anchoring = 0.6 * high_prox + 0.4 * low_prox
        
        return anchoring.fillna(0.5)
    
    # =========================================================================
    # MARKET MICROSTRUCTURE FACTORS
    # =========================================================================
    
    def _factor_efficiency_ratio(self) -> "xr.DataArray":
        """Kaufman Efficiency Ratio.
        
        Measures price path efficiency: directional movement / total movement.
        Higher ratio = more efficient/trending price movement.
        Lower ratio = more noise/chop.
        
        ER = |P(t) - P(t-n)| / sum(|P(t-i) - P(t-i-1)|) for i in 0..n-1
        
        Research basis: Kaufman (1995), technical analysis efficiency
        """
        p = self.params
        eff_win = getattr(p, 'EFFICIENCY_WIN', 21)
        
        # Directional movement over window
        direction = np.abs(self.close - self.close.shift(time=eff_win))
        
        # Total movement (volatility path)
        abs_changes = np.abs(self.close - self.close.shift(time=1))
        volatility = abs_changes.rolling(time=eff_win, min_periods=eff_win).sum()
        
        # Efficiency ratio
        er = direction / (volatility + p.eps)
        
        # Combine with momentum direction for signal
        mom_dir = np.sign(self.close - self.close.shift(time=eff_win))
        
        return (er * mom_dir).fillna(0.0).clip(-1, 1)
    
    def _factor_information_flow(self) -> "xr.DataArray":
        """Information flow asymmetry factor.
        
        Measures volume-weighted price impact:
        - High volume on up days vs down days
        - Asymmetric flow indicates informed trading
        
        Formula: (up_vol_impact - down_vol_impact) / total_vol_impact
        
        Research basis: Kyle (1985), Easley & O'Hara (1987) - informed trading
        """
        p = self.params
        
        # Classify up/down days
        is_up = (self.ret > 0).astype(float)
        is_down = (self.ret < 0).astype(float)
        
        # Volume-weighted price impact
        dollar_vol = self.vol * self.close
        up_impact = _sma(dollar_vol * is_up * np.abs(self.ret), p.ADV_WIN_LONG)
        down_impact = _sma(dollar_vol * is_down * np.abs(self.ret), p.ADV_WIN_LONG)
        total_impact = up_impact + down_impact + p.eps
        
        # Asymmetry: more up-day impact = positive signal
        info_flow = (up_impact - down_impact) / total_impact
        
        return info_flow.fillna(0.0).clip(-1, 1)
    
    def _factor_mean_reversion_speed(self) -> "xr.DataArray":
        """Mean reversion speed factor.
        
        Estimates the half-life of price deviations from trend.
        Faster mean reversion = more tradeable.
        
        Uses deviation from EMA and measures decay rate.
        
        Research basis: Ornstein-Uhlenbeck process in finance
        """
        p = self.params
        
        # Deviation from trend (EMA)
        trend = _ema(self.close, p.EMA_SLOW)
        deviation = (self.close - trend) / (trend + p.eps)
        
        # Lag-1 autocorrelation of deviations (proxy for mean reversion speed)
        # Higher negative autocorr = faster reversion
        dev_df = deviation.transpose("time", "asset").to_pandas()
        dev_lag = dev_df.shift(1)
        
        # Rolling correlation
        autocorr = dev_df.rolling(63, min_periods=21).corr(dev_lag).fillna(0.0)
        
        # Convert to speed: negative autocorr = fast reversion = high score
        mr_speed = xr.DataArray(
            -autocorr.values,
            coords=deviation.coords,
            dims=deviation.dims
        )
        
        return mr_speed.fillna(0.0).clip(-1, 1)
    
    # =========================================================================
    # MOMENTUM QUALITY FACTORS
    # =========================================================================
    
    def _factor_momentum_quality_ratio(self) -> "xr.DataArray":
        """Momentum quality ratio (signal-to-noise).
        
        Measures the Sharpe ratio of rolling returns.
        High quality = consistent positive returns (high mean, low vol).
        
        MQR = mean(ret) / std(ret) over rolling window
        
        Research basis: Israel et al. (2020), quality momentum
        """
        p = self.params
        
        # Rolling mean and std of returns
        ret_mean = _sma(self.ret, p.MOM_WIN)
        ret_std = _std(self.ret, p.MOM_WIN)
        
        # Quality ratio (Sharpe-like)
        mqr = ret_mean / (ret_std + p.eps)
        
        # Annualize for interpretability
        mqr_ann = mqr * np.sqrt(252)
        
        return mqr_ann.fillna(0.0).clip(-5, 5)
    
    def _factor_momentum_persistence(self) -> "xr.DataArray":
        """Momentum persistence factor.
        
        Measures autocorrelation of returns (momentum continuation signal).
        High persistence = momentum likely to continue.
        
        Research basis: Moskowitz et al. (2012), time series momentum
        """
        p = self.params
        persist_win = getattr(p, 'PERSISTENCE_WIN', 42)
        
        # Convert to pandas for rolling correlation
        ret_df = self.ret.transpose("time", "asset").to_pandas()
        ret_lag = ret_df.shift(1)
        
        # Rolling autocorrelation
        persistence = ret_df.rolling(persist_win, min_periods=21).corr(ret_lag).fillna(0.0)
        
        # Positive persistence = momentum continuation
        persist_da = xr.DataArray(
            persistence.values,
            coords=self.ret.coords,
            dims=self.ret.dims
        )
        
        return persist_da.fillna(0.0).clip(-1, 1)
    
    def _factor_momentum_divergence(self) -> "xr.DataArray":
        """Momentum divergence factor.
        
        Measures divergence between price momentum and residual (alpha) momentum.
        Large divergence indicates beta-driven vs alpha-driven moves.
        
        Prefer stocks where residual momentum leads price momentum (alpha generation).
        
        Research basis: Separating alpha from beta in momentum
        """
        p = self.params
        
        # Price momentum
        price_mom = _sma(self.ret, p.MOM_WIN)
        
        # Residual momentum (alpha component)
        resid_mom = _sma(self.resid, p.MOM_WIN)
        
        # Divergence: residual leading vs lagging price
        # Positive = residual mom > price mom = alpha-driven
        divergence = resid_mom - price_mom
        
        # Normalize by volatility
        div_std = _std(divergence, p.IDIO_WIN)
        divergence_z = divergence / (div_std + p.eps)
        
        return divergence_z.fillna(0.0).clip(-3, 3)
    
    # =========================================================================
    # RISK/REGIME FACTORS
    # =========================================================================
    
    def _factor_vol_of_vol(self) -> "xr.DataArray":
        """Volatility of volatility factor.
        
        Measures uncertainty about volatility itself.
        High vol-of-vol indicates unstable risk regime.
        
        Inverted: prefer low vol-of-vol (stable risk).
        
        Research basis: Uncertainty premium in options markets
        """
        p = self.params
        vov_win = getattr(p, 'VOV_WIN', 21)
        
        # Rolling volatility
        vol = _std(self.ret, vov_win)
        
        # Vol-of-vol (std of volatility)
        vol_of_vol = _std(vol, p.IDIO_WIN)
        
        # Inverse: low vol-of-vol = stable = good
        inv_vov = 1.0 / (vol_of_vol + p.eps)
        
        return inv_vov.fillna(0.0)
    
    def _factor_skewness_factor(self) -> "xr.DataArray":
        """Return skewness factor.
        
        Measures asymmetry of return distribution.
        Positive skew = more upside potential, negative skew = more downside risk.
        
        Prefer positive skewness (lottery-like upside).
        
        Research basis: Bali et al. (2011), skewness preference
        """
        p = self.params
        skew_win = getattr(p, 'SKEW_WIN', 63)
        
        # Rolling skewness using pandas
        ret_df = self.ret.transpose("time", "asset").to_pandas()
        
        skewness = ret_df.rolling(skew_win, min_periods=21).skew().fillna(0.0)
        
        skew_da = xr.DataArray(
            skewness.values,
            coords=self.ret.coords,
            dims=self.ret.dims
        )
        
        # Prefer positive skew
        return skew_da.fillna(0.0).clip(-3, 3)
    
    def _factor_kurtosis_factor(self) -> "xr.DataArray":
        """Return kurtosis factor.
        
        Measures tail thickness of return distribution.
        High kurtosis = fat tails = more extreme events.
        
        Inverted: prefer lower kurtosis (fewer extreme moves).
        
        Research basis: Tail risk in equity markets
        """
        p = self.params
        kurt_win = getattr(p, 'KURT_WIN', 63)
        
        # Rolling kurtosis using pandas
        ret_df = self.ret.transpose("time", "asset").to_pandas()
        
        kurtosis = ret_df.rolling(kurt_win, min_periods=21).kurt().fillna(0.0)
        
        kurt_da = xr.DataArray(
            kurtosis.values,
            coords=self.ret.coords,
            dims=self.ret.dims
        )
        
        # Inverse: lower kurtosis = safer = good
        # Clip extreme values first
        kurt_clipped = kurt_da.clip(-10, 20)
        inv_kurt = -kurt_clipped  # Negate: low kurt becomes high score
        
        return inv_kurt.fillna(0.0)
    
    def _factor_regime_momentum(self) -> "xr.DataArray":
        """Regime-adaptive momentum factor.
        
        Dynamically blends short and long momentum based on volatility regime.
        - High vol regime: use shorter momentum (faster reaction)
        - Low vol regime: use longer momentum (more stable signal)
        
        Research basis: Daniel & Moskowitz (2016), momentum crashes
        """
        p = self.params
        
        # Volatility regime indicator
        vol_short = _std(self.ret, p.MOM_SHORT)
        vol_long = _std(self.ret, p.IDIO_WIN)
        vol_ratio = vol_short / (vol_long + p.eps)
        
        # Regime weight: high vol ratio = recent vol spike = use short mom
        # Sigmoid-like transformation
        regime_weight = 1.0 / (1.0 + np.exp(-2 * (vol_ratio - 1.0)))
        regime_weight = regime_weight.clip(0.2, 0.8)
        
        # Short and long momentum
        mom_short = _sma(self.resid, p.MOM_SHORT)
        mom_long = _sma(self.resid, p.MOM_LONG)
        
        # Blend based on regime
        regime_mom = regime_weight * mom_short + (1 - regime_weight) * mom_long
        
        return regime_mom.fillna(0.0)
    
    # =========================================================================
    # CROSS-SECTIONAL FACTORS
    # =========================================================================
    
    def _factor_cross_sectional_dispersion(self) -> "xr.DataArray":
        """Cross-sectional return dispersion factor.
        
        Measures each stock's position relative to cross-sectional dispersion.
        High dispersion periods offer more alpha opportunities.
        
        Signal: stock outperformance relative to cross-sectional mean,
        weighted by overall dispersion level.
        
        Research basis: Stivers (2010), cross-sectional dispersion
        """
        p = self.params
        
        # Cross-sectional dispersion (std across assets)
        cs_std = self.ret.std("asset")
        cs_mean = self.ret.mean("asset")
        
        # Smooth dispersion
        disp_smooth = _sma(cs_std, p.MOM_SHORT)
        
        # Stock's relative position
        rel_ret = self.ret - cs_mean
        
        # Weight by dispersion: high dispersion = opportunities
        # Normalize dispersion
        disp_normalized = disp_smooth / (disp_smooth.mean("time") + p.eps)
        
        # Signal: outperformance * dispersion level
        dispersion_signal = rel_ret * disp_normalized.broadcast_like(rel_ret)
        
        # Smooth
        return _sma(dispersion_signal, p.MOM_SHORT).fillna(0.0)
    
    def _factor_liquidity_momentum(self) -> "xr.DataArray":
        """Liquidity-adjusted momentum factor.
        
        Weights momentum signal by liquidity quality.
        More liquid stocks provide more reliable momentum signals.
        
        LiqMom = resid_mom * liquidity_score
        
        Research basis: Avramov et al. (2006), liquidity and momentum
        """
        p = self.params
        
        # Residual momentum
        resid_mom = _sma(self.resid, p.MOM_WIN)
        
        # Liquidity score (relative ADV)
        adv = _sma(self.vol * self.close, p.ADV_WIN_LONG)
        adv_mean = adv.mean("asset")
        liq_score = adv / (adv_mean + p.eps)
        
        # Normalize liquidity score to [0, 2] range
        liq_normalized = liq_score.clip(0, 5) / 2.5
        
        # Weight momentum by liquidity
        liq_mom = resid_mom * liq_normalized
        
        return liq_mom.fillna(0.0)
