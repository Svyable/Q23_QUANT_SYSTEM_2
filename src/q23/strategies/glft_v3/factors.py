"""GLFTv3 Factor Library - Advanced Market Microstructure Factors.

V3 adds cutting-edge microstructure signals:
1. MTF-OFI Alignment - multi-timeframe order flow consensus
2. Sector-Relative Flow - flow vs sector average
3. Regime-Adaptive Toxicity - toxicity relative to regime
4. Market Flow Momentum - market-wide flow direction
5. Execution Quality - trade implementation quality
6. Adverse Selection Decomposition - informed vs noise
7. Optimal Spread Signal - from GLFT optimal spread formula
8. Flow Persistence - autocorrelation of order flow
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


def ewma(x: "xr.DataArray", span: int) -> "xr.DataArray":
    """Exponentially weighted moving average."""
    df = x.transpose("time", "asset").to_pandas()
    out = df.ewm(span=span, adjust=False, min_periods=span).mean()
    return xr.DataArray(
        out.values, 
        coords={"time": x.time, "asset": x.asset}, 
        dims=["time", "asset"]
    ).fillna(0.0)


class GLFTv3FactorLibrary(FactorLibrary):
    """Advanced GLFTv3 factor library with cutting-edge microstructure signals."""
    
    def __init__(
        self,
        market_data: Union["xr.Dataset", Any],
        *,
        params: Optional[FactorParams] = None,
        factors: Optional[Sequence[str]] = None,
        asset_dim: str = "asset",
    ):
        self.bundle = market_data
        super().__init__(market_data, params=params, factors=factors, asset_dim=asset_dim)
        
        p = self.params
        # V1 params
        self.ofi_short_win = getattr(p, 'OFI_SHORT_WIN', 5)
        self.ofi_med_win = getattr(p, 'OFI_MED_WIN', 21)
        self.ofi_long_win = getattr(p, 'OFI_LONG_WIN', 63)
        self.vpin_win = getattr(p, 'VPIN_WIN', 42)
        self.spread_est_win = getattr(p, 'SPREAD_EST_WIN', 21)
        self.impact_win = getattr(p, 'IMPACT_WIN', 42)
        self.inventory_decay = getattr(p, 'INVENTORY_DECAY', 0.94)
        
        # V2 params
        self.kyle_lambda_win = getattr(p, 'KYLE_LAMBDA_WIN', 42)
        self.volume_clock_win = getattr(p, 'VOLUME_CLOCK_WIN', 21)
        self.bvc_win = getattr(p, 'BVC_WIN', 10)
        self.flow_disp_win = getattr(p, 'FLOW_DISP_WIN', 21)
        self.reservation_win = getattr(p, 'RESERVATION_WIN', 14)
        
        # V3 new params
        self.mtf_ofi_short = getattr(p, 'MTF_OFI_SHORT', 5)
        self.mtf_ofi_med = getattr(p, 'MTF_OFI_MED', 21)
        self.mtf_ofi_long = getattr(p, 'MTF_OFI_LONG', 63)
        self.regime_win = getattr(p, 'REGIME_WIN', 42)
        self.flow_persist_win = getattr(p, 'FLOW_PERSIST_WIN', 10)
        self.exec_quality_win = getattr(p, 'EXEC_QUALITY_WIN', 21)
        self.adverse_win = getattr(p, 'ADVERSE_WIN', 14)
        self.optimal_spread_win = getattr(p, 'OPTIMAL_SPREAD_WIN', 21)
        
        self.eps = p.eps
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _compute_signed_volume(self) -> "xr.DataArray":
        close = self.close
        vol = self.vol
        price_change = close.diff(dim="time")
        sign = xr.where(price_change > 0, 1.0, xr.where(price_change < 0, -1.0, 0.0))
        return sign * vol
    
    def _compute_dollar_volume(self) -> "xr.DataArray":
        return self.close * self.vol
    
    def _compute_ofi(self, window: int) -> "xr.DataArray":
        signed_vol = self._compute_signed_volume()
        vol = self.vol
        signed_sum = signed_vol.rolling(time=window, min_periods=window//2).sum()
        vol_sum = vol.rolling(time=window, min_periods=window//2).sum()
        ofi = signed_sum / (vol_sum + self.eps)
        return ofi.clip(-1, 1)
    
    def _compute_vpin_proxy(self) -> "xr.DataArray":
        signed_vol = self._compute_signed_volume()
        buy_vol = xr.where(signed_vol > 0, signed_vol, 0)
        sell_vol = xr.where(signed_vol < 0, -signed_vol, 0)
        buy_sum = buy_vol.rolling(time=self.vpin_win, min_periods=self.vpin_win//2).sum()
        sell_sum = sell_vol.rolling(time=self.vpin_win, min_periods=self.vpin_win//2).sum()
        total_vol = buy_sum + sell_sum + self.eps
        return np.abs(buy_sum - sell_sum) / total_vol
    
    def _compute_inventory_proxy(self) -> "xr.DataArray":
        signed_vol = self._compute_signed_volume()
        return ewma(signed_vol, span=int(1/(1-self.inventory_decay)))
    
    def _compute_bvc(self) -> "xr.DataArray":
        """Bulk Volume Classification."""
        high, low, close = self.high, self.low, self.close
        bar_range = high - low + self.eps
        return (close - low) / bar_range
    
    # =========================================================================
    # V1 FACTORS (10 factors)
    # =========================================================================
    
    def _factor_glft_ofi_short(self) -> "xr.DataArray":
        return z_score_cs(self._compute_ofi(self.ofi_short_win), self.eps)
    
    def _factor_glft_ofi_med(self) -> "xr.DataArray":
        return z_score_cs(self._compute_ofi(self.ofi_med_win), self.eps)
    
    def _factor_glft_ofi_momentum(self) -> "xr.DataArray":
        ofi_short = self._compute_ofi(self.ofi_short_win)
        ofi_long = self._compute_ofi(self.ofi_long_win)
        return z_score_cs(ofi_short - ofi_long, self.eps)
    
    def _factor_glft_flow_toxicity(self) -> "xr.DataArray":
        return z_score_cs(-self._compute_vpin_proxy(), self.eps)
    
    def _factor_glft_toxic_momentum(self) -> "xr.DataArray":
        vpin = self._compute_vpin_proxy()
        vpin_short = vpin.rolling(time=5, min_periods=3).mean()
        vpin_long = vpin.rolling(time=21, min_periods=14).mean()
        return z_score_cs(-(vpin_short - vpin_long), self.eps)
    
    def _factor_glft_impact_asymmetry(self) -> "xr.DataArray":
        returns, vol = self.ret, self.vol
        up_ret = xr.where(returns > 0, returns, 0)
        down_ret = xr.where(returns < 0, returns, 0)
        up_vol = xr.where(returns > 0, vol, 0)
        down_vol = xr.where(returns < 0, vol, 0)
        
        up_impact = (up_ret * up_vol).rolling(time=self.impact_win, min_periods=self.impact_win//2).sum()
        up_vol_sum = up_vol.rolling(time=self.impact_win, min_periods=self.impact_win//2).sum()
        down_impact = (down_ret * down_vol).rolling(time=self.impact_win, min_periods=self.impact_win//2).sum()
        down_vol_sum = down_vol.rolling(time=self.impact_win, min_periods=self.impact_win//2).sum()
        
        avg_up = up_impact / (up_vol_sum + self.eps)
        avg_down = np.abs(down_impact) / (down_vol_sum + self.eps)
        return z_score_cs(avg_up - avg_down, self.eps)
    
    def _factor_glft_inventory_signal(self) -> "xr.DataArray":
        return z_score_cs(-self._compute_inventory_proxy(), self.eps)
    
    def _factor_glft_inventory_risk_prem(self) -> "xr.DataArray":
        inventory = self._compute_inventory_proxy()
        inv_vol = inventory.rolling(time=self.impact_win, min_periods=self.impact_win//2).std()
        return z_score_cs(inv_vol, self.eps)
    
    def _factor_glft_spread_adjusted_mom(self) -> "xr.DataArray":
        returns, high, low = self.ret, self.high, self.low
        hl_ratio = np.log(high / (low + self.eps) + self.eps)
        spread_est = hl_ratio.rolling(time=self.spread_est_win, min_periods=self.spread_est_win//2).mean()
        mom_21 = returns.rolling(time=21, min_periods=15).sum()
        return z_score_cs(mom_21 - 2 * spread_est, self.eps)
    
    def _factor_glft_mm_edge(self) -> "xr.DataArray":
        vpin = self._compute_vpin_proxy()
        returns = self.ret
        low_tox = z_score_cs(-vpin, self.eps)
        vol_21 = returns.rolling(time=21, min_periods=15).std()
        vol_z = z_score_cs(vol_21, self.eps)
        ret_5 = returns.rolling(time=5, min_periods=3).sum()
        mr_z = z_score_cs(-ret_5, self.eps)
        return z_score_cs(0.4 * low_tox + 0.3 * vol_z + 0.3 * mr_z, self.eps)
    
    # =========================================================================
    # V2 FACTORS (6 factors)
    # =========================================================================
    
    def _factor_glft_kyle_lambda(self) -> "xr.DataArray":
        returns = self.ret
        signed_vol = self._compute_signed_volume()
        ret_mean = returns.rolling(time=self.kyle_lambda_win, min_periods=self.kyle_lambda_win//2).mean()
        sv_mean = signed_vol.rolling(time=self.kyle_lambda_win, min_periods=self.kyle_lambda_win//2).mean()
        cov = ((returns - ret_mean) * (signed_vol - sv_mean)).rolling(
            time=self.kyle_lambda_win, min_periods=self.kyle_lambda_win//2
        ).mean()
        var = ((signed_vol - sv_mean) ** 2).rolling(
            time=self.kyle_lambda_win, min_periods=self.kyle_lambda_win//2
        ).mean()
        kyle_lambda = cov / (var + self.eps)
        return z_score_cs(-kyle_lambda, self.eps)
    
    def _factor_glft_volume_clock_ofi(self) -> "xr.DataArray":
        signed_vol = self._compute_signed_volume()
        dollar_vol = self._compute_dollar_volume()
        dv_signed = signed_vol * self.close
        dv_signed_sum = dv_signed.rolling(time=self.volume_clock_win, min_periods=self.volume_clock_win//2).sum()
        dv_sum = dollar_vol.rolling(time=self.volume_clock_win, min_periods=self.volume_clock_win//2).sum()
        return z_score_cs(dv_signed_sum / (dv_sum + self.eps), self.eps)
    
    def _factor_glft_bvc_imbalance(self) -> "xr.DataArray":
        bvc = self._compute_bvc()
        vol = self.vol
        buy_vol = bvc * vol
        sell_vol = (1 - bvc) * vol
        buy_sum = buy_vol.rolling(time=self.bvc_win, min_periods=self.bvc_win//2).sum()
        sell_sum = sell_vol.rolling(time=self.bvc_win, min_periods=self.bvc_win//2).sum()
        total = buy_sum + sell_sum + self.eps
        return z_score_cs((buy_sum - sell_sum) / total, self.eps)
    
    def _factor_glft_flow_dispersion(self) -> "xr.DataArray":
        ofi = self._compute_ofi(self.ofi_med_win)
        ofi_cs_std = ofi.std(dim="asset")
        ofi_median = ofi.median(dim="asset")
        dist_from_median = np.abs(ofi - ofi_median)
        dispersion_broadcast = ofi_cs_std.broadcast_like(ofi)
        reliability = dispersion_broadcast / (dist_from_median + self.eps)
        return z_score_cs(reliability, self.eps)
    
    def _factor_glft_reservation_signal(self) -> "xr.DataArray":
        close = self.close
        returns = self.ret
        inventory = self._compute_inventory_proxy()
        vol = returns.rolling(time=self.reservation_win, min_periods=self.reservation_win//2).std()
        inventory_effect = inventory * (vol ** 2)
        reservation = close - inventory_effect
        signal = (reservation - close) / (close + self.eps)
        return z_score_cs(signal, self.eps)
    
    def _factor_glft_amihud_hybrid(self) -> "xr.DataArray":
        returns = self.ret
        dollar_vol = self._compute_dollar_volume()
        high, low = self.high, self.low
        amihud = np.abs(returns) / (dollar_vol + self.eps)
        amihud_roll = amihud.rolling(time=self.impact_win, min_periods=self.impact_win//2).mean()
        hl_ratio = np.log(high / (low + self.eps) + self.eps)
        spread = hl_ratio.rolling(time=self.spread_est_win, min_periods=self.spread_est_win//2).mean()
        vol_norm = self.vol / (self.vol.rolling(time=63, min_periods=42).mean() + self.eps)
        amihud_z = z_score_cs(-amihud_roll, self.eps)
        spread_z = z_score_cs(-spread, self.eps)
        vol_z = z_score_cs(vol_norm, self.eps)
        return z_score_cs(0.4 * amihud_z + 0.3 * spread_z + 0.3 * vol_z, self.eps)
    
    # =========================================================================
    # V3 NEW FACTORS (8 factors)
    # =========================================================================
    
    def _factor_glft_mtf_ofi_alignment(self) -> "xr.DataArray":
        """Multi-timeframe OFI alignment.
        
        Consensus across short/medium/long OFI timeframes.
        All positive or all negative = strong signal.
        """
        ofi_s = self._compute_ofi(self.mtf_ofi_short)
        ofi_m = self._compute_ofi(self.mtf_ofi_med)
        ofi_l = self._compute_ofi(self.mtf_ofi_long)
        
        # Sign alignment: +1 if all same sign, else weighted average
        sign_s = xr.where(ofi_s > 0, 1.0, xr.where(ofi_s < 0, -1.0, 0.0))
        sign_m = xr.where(ofi_m > 0, 1.0, xr.where(ofi_m < 0, -1.0, 0.0))
        sign_l = xr.where(ofi_l > 0, 1.0, xr.where(ofi_l < 0, -1.0, 0.0))
        
        # Alignment score: sum of signs (3 = all positive, -3 = all negative)
        alignment = sign_s + sign_m + sign_l
        
        # Weight by magnitude
        magnitude = (np.abs(ofi_s) + np.abs(ofi_m) + np.abs(ofi_l)) / 3
        
        aligned_signal = alignment * magnitude
        
        return z_score_cs(aligned_signal, self.eps)
    
    def _factor_glft_sector_rel_flow(self) -> "xr.DataArray":
        """Sector-relative flow signal.
        
        OFI relative to cross-sectional average (sector proxy).
        Positive = stronger buying than peers.
        """
        ofi = self._compute_ofi(self.ofi_med_win)
        
        # Cross-sectional mean as sector proxy
        ofi_market = ofi.mean(dim="asset")
        
        # Relative OFI
        relative_ofi = ofi - ofi_market.broadcast_like(ofi)
        
        return z_score_cs(relative_ofi, self.eps)
    
    def _factor_glft_regime_toxicity(self) -> "xr.DataArray":
        """Regime-adaptive toxicity.
        
        Toxicity normalized by market regime.
        Low toxicity in high-toxicity regime = very good.
        """
        vpin = self._compute_vpin_proxy()
        
        # Market-wide toxicity regime
        market_vpin = vpin.mean(dim="asset")
        market_vpin_roll = market_vpin.rolling(time=self.regime_win, min_periods=self.regime_win//2).mean()
        
        # Relative toxicity: stock vs market regime
        relative_tox = vpin / (market_vpin_roll.broadcast_like(vpin) + self.eps)
        
        # Invert: low relative toxicity = good
        return z_score_cs(-relative_tox, self.eps)
    
    def _factor_glft_market_flow_mom(self) -> "xr.DataArray":
        """Market-wide flow momentum.
        
        When market flow is accelerating, favor stocks with aligned flow.
        """
        ofi = self._compute_ofi(self.ofi_med_win)
        
        # Market-wide OFI
        market_ofi = ofi.mean(dim="asset")
        market_ofi_short = market_ofi.rolling(time=5, min_periods=3).mean()
        market_ofi_long = market_ofi.rolling(time=21, min_periods=14).mean()
        
        # Market flow momentum
        market_flow_mom = market_ofi_short - market_ofi_long
        
        # Stock alignment with market flow momentum
        # Favor stocks with OFI in same direction as market momentum
        alignment = ofi * market_flow_mom.broadcast_like(ofi)
        
        return z_score_cs(alignment, self.eps)
    
    def _factor_glft_exec_quality(self) -> "xr.DataArray":
        """Execution quality score.
        
        Combines:
        - Low spread
        - Low impact
        - High volume
        - Low toxicity
        """
        high, low = self.high, self.low
        dollar_vol = self._compute_dollar_volume()
        returns = self.ret
        vpin = self._compute_vpin_proxy()
        
        # Spread estimate
        spread = np.log(high / (low + self.eps) + self.eps)
        spread_roll = spread.rolling(time=self.exec_quality_win, min_periods=self.exec_quality_win//2).mean()
        
        # Impact estimate (Amihud)
        impact = np.abs(returns) / (dollar_vol + self.eps)
        impact_roll = impact.rolling(time=self.exec_quality_win, min_periods=self.exec_quality_win//2).mean()
        
        # Volume relative
        vol_rel = dollar_vol / (dollar_vol.rolling(time=63, min_periods=42).mean() + self.eps)
        
        # Combine (all inverted where needed)
        spread_z = z_score_cs(-spread_roll, self.eps)  # Low spread = good
        impact_z = z_score_cs(-impact_roll, self.eps)  # Low impact = good
        vol_z = z_score_cs(vol_rel, self.eps)          # High volume = good
        tox_z = z_score_cs(-vpin, self.eps)            # Low toxicity = good
        
        exec_quality = 0.25 * spread_z + 0.25 * impact_z + 0.25 * vol_z + 0.25 * tox_z
        
        return z_score_cs(exec_quality, self.eps)
    
    def _factor_glft_adverse_decomp(self) -> "xr.DataArray":
        """Adverse selection decomposition.
        
        Decomposes price movement into:
        - Informed component (permanent)
        - Noise component (temporary)
        
        High noise ratio = less adverse selection = safer.
        """
        returns = self.ret
        signed_vol = self._compute_signed_volume()
        
        # Total variance
        total_var = returns.rolling(time=self.adverse_win, min_periods=self.adverse_win//2).var()
        
        # Estimate informed variance (correlated with signed volume)
        ret_sv_cov = (returns * signed_vol).rolling(
            time=self.adverse_win, min_periods=self.adverse_win//2
        ).mean() - (
            returns.rolling(time=self.adverse_win, min_periods=self.adverse_win//2).mean() *
            signed_vol.rolling(time=self.adverse_win, min_periods=self.adverse_win//2).mean()
        )
        
        sv_var = signed_vol.rolling(time=self.adverse_win, min_periods=self.adverse_win//2).var()
        
        # Informed variance proxy
        informed_var = (ret_sv_cov ** 2) / (sv_var + self.eps)
        
        # Noise ratio: 1 - informed/total
        noise_ratio = 1 - informed_var / (total_var + self.eps)
        noise_ratio = noise_ratio.clip(0, 1)
        
        # High noise ratio = less adverse selection = good
        return z_score_cs(noise_ratio, self.eps)
    
    def _factor_glft_optimal_spread(self) -> "xr.DataArray":
        """Optimal spread signal from GLFT formula.
        
        δ* = γσ²τ + (2/γ)ln(1 + γ/k)
        
        When actual spread < optimal → undervalued spread → good to hold
        """
        returns = self.ret
        high, low = self.high, self.low
        vol = self.vol
        
        # Estimate volatility (σ)
        sigma = returns.rolling(time=self.optimal_spread_win, min_periods=self.optimal_spread_win//2).std()
        
        # Estimate order intensity (k) - normalized volume
        k = vol / (vol.rolling(time=63, min_periods=42).mean() + self.eps)
        
        # Simplified optimal spread (γ=1, τ=1)
        # δ* ≈ σ² + 2*ln(1 + 1/k)
        optimal_spread = (sigma ** 2) + 2 * np.log(1 + 1 / (k + self.eps))
        
        # Actual spread estimate
        actual_spread = np.log(high / (low + self.eps) + self.eps)
        actual_spread_roll = actual_spread.rolling(
            time=self.optimal_spread_win, min_periods=self.optimal_spread_win//2
        ).mean()
        
        # Signal: optimal - actual (positive = undervalued spread)
        spread_signal = optimal_spread - actual_spread_roll
        
        return z_score_cs(spread_signal, self.eps)
    
    def _factor_glft_flow_persistence(self) -> "xr.DataArray":
        """Flow persistence factor.
        
        Autocorrelation of order flow.
        High persistence = trend-following OFI
        Low persistence = mean-reverting OFI
        
        We use the persistence level to weight signals.
        """
        ofi = self._compute_ofi(self.ofi_short_win)
        
        # Lagged OFI
        ofi_lag = ofi.shift(time=1)
        
        # Rolling correlation (autocorrelation)
        ofi_df = ofi.transpose("time", "asset").to_pandas()
        ofi_lag_df = ofi_lag.transpose("time", "asset").to_pandas()
        
        # Compute rolling correlation
        autocorr = ofi_df.rolling(self.flow_persist_win).corr(ofi_lag_df)
        
        autocorr_da = xr.DataArray(
            autocorr.fillna(0.0).values,
            coords=ofi.coords,
            dims=ofi.dims
        ).fillna(0.0)
        
        # High persistence + high OFI = strong trend signal
        persistence_signal = autocorr_da * ofi
        
        return z_score_cs(persistence_signal, self.eps)
