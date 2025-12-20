"""GLFTv2 Factor Library - Enhanced Market Microstructure Factors.

V2 adds advanced microstructure signals:
1. Kyle's Lambda - permanent price impact estimation
2. Volume Clock OFI - time-invariant order flow
3. BVC Imbalance - Bulk Volume Classification
4. Flow Dispersion - cross-sectional regime indicator
5. Reservation Signal - Avellaneda-Stoikov inspired
6. Amihud-GLFT Hybrid - combined liquidity measure
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


class GLFTv2FactorLibrary(FactorLibrary):
    """Enhanced GLFTv2 factor library with advanced microstructure signals."""
    
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
        
        # V2 new params
        self.kyle_lambda_win = getattr(p, 'KYLE_LAMBDA_WIN', 42)
        self.volume_clock_win = getattr(p, 'VOLUME_CLOCK_WIN', 21)
        self.bvc_win = getattr(p, 'BVC_WIN', 10)
        self.flow_disp_win = getattr(p, 'FLOW_DISP_WIN', 21)
        self.reservation_win = getattr(p, 'RESERVATION_WIN', 14)
        
        self.eps = p.eps
    
    # =========================================================================
    # Helper Methods (inherited and extended from v1)
    # =========================================================================
    
    def _compute_signed_volume(self) -> "xr.DataArray":
        """Compute signed volume using price change classification."""
        close = self.close
        vol = self.vol
        price_change = close.diff(dim="time")
        sign = xr.where(price_change > 0, 1.0, xr.where(price_change < 0, -1.0, 0.0))
        return sign * vol
    
    def _compute_dollar_volume(self) -> "xr.DataArray":
        """Compute dollar volume."""
        return self.close * self.vol
    
    def _compute_ofi(self, window: int) -> "xr.DataArray":
        """Compute Order Flow Imbalance."""
        signed_vol = self._compute_signed_volume()
        vol = self.vol
        signed_sum = signed_vol.rolling(time=window, min_periods=window//2).sum()
        vol_sum = vol.rolling(time=window, min_periods=window//2).sum()
        ofi = signed_sum / (vol_sum + self.eps)
        return ofi.clip(-1, 1)
    
    def _compute_vpin_proxy(self) -> "xr.DataArray":
        """Compute VPIN-style flow toxicity."""
        signed_vol = self._compute_signed_volume()
        buy_vol = xr.where(signed_vol > 0, signed_vol, 0)
        sell_vol = xr.where(signed_vol < 0, -signed_vol, 0)
        buy_sum = buy_vol.rolling(time=self.vpin_win, min_periods=self.vpin_win//2).sum()
        sell_sum = sell_vol.rolling(time=self.vpin_win, min_periods=self.vpin_win//2).sum()
        total_vol = buy_sum + sell_sum + self.eps
        return np.abs(buy_sum - sell_sum) / total_vol
    
    def _compute_inventory_proxy(self) -> "xr.DataArray":
        """Compute cumulative inventory proxy."""
        signed_vol = self._compute_signed_volume()
        return ewma(signed_vol, span=int(1/(1-self.inventory_decay)))
    
    # =========================================================================
    # V1 FACTORS (10 factors)
    # =========================================================================
    
    def _factor_glft_ofi_short(self) -> "xr.DataArray":
        ofi = self._compute_ofi(self.ofi_short_win)
        return z_score_cs(ofi, self.eps)
    
    def _factor_glft_ofi_med(self) -> "xr.DataArray":
        ofi = self._compute_ofi(self.ofi_med_win)
        return z_score_cs(ofi, self.eps)
    
    def _factor_glft_ofi_momentum(self) -> "xr.DataArray":
        ofi_short = self._compute_ofi(self.ofi_short_win)
        ofi_long = self._compute_ofi(self.ofi_long_win)
        return z_score_cs(ofi_short - ofi_long, self.eps)
    
    def _factor_glft_flow_toxicity(self) -> "xr.DataArray":
        vpin = self._compute_vpin_proxy()
        return z_score_cs(-vpin, self.eps)
    
    def _factor_glft_toxic_momentum(self) -> "xr.DataArray":
        vpin = self._compute_vpin_proxy()
        vpin_short = vpin.rolling(time=5, min_periods=3).mean()
        vpin_long = vpin.rolling(time=21, min_periods=14).mean()
        return z_score_cs(-(vpin_short - vpin_long), self.eps)
    
    def _factor_glft_impact_asymmetry(self) -> "xr.DataArray":
        returns = self.ret
        vol = self.vol
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
        inventory = self._compute_inventory_proxy()
        return z_score_cs(-inventory, self.eps)
    
    def _factor_glft_inventory_risk_prem(self) -> "xr.DataArray":
        inventory = self._compute_inventory_proxy()
        inv_vol = inventory.rolling(time=self.impact_win, min_periods=self.impact_win//2).std()
        return z_score_cs(inv_vol, self.eps)
    
    def _factor_glft_spread_adjusted_mom(self) -> "xr.DataArray":
        returns = self.ret
        high, low = self.high, self.low
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
        mm_edge = 0.4 * low_tox + 0.3 * vol_z + 0.3 * mr_z
        return z_score_cs(mm_edge, self.eps)
    
    # =========================================================================
    # V2 NEW FACTORS (6 factors)
    # =========================================================================
    
    def _factor_glft_kyle_lambda(self) -> "xr.DataArray":
        """Kyle's Lambda - permanent price impact coefficient.
        
        λ = Cov(ΔP, SignedVolume) / Var(SignedVolume)
        Higher lambda = higher price impact = less liquid
        We invert: lower lambda = more liquid = higher score
        """
        returns = self.ret
        signed_vol = self._compute_signed_volume()
        
        # Rolling covariance and variance
        ret_mean = returns.rolling(time=self.kyle_lambda_win, min_periods=self.kyle_lambda_win//2).mean()
        sv_mean = signed_vol.rolling(time=self.kyle_lambda_win, min_periods=self.kyle_lambda_win//2).mean()
        
        cov = ((returns - ret_mean) * (signed_vol - sv_mean)).rolling(
            time=self.kyle_lambda_win, min_periods=self.kyle_lambda_win//2
        ).mean()
        
        var = ((signed_vol - sv_mean) ** 2).rolling(
            time=self.kyle_lambda_win, min_periods=self.kyle_lambda_win//2
        ).mean()
        
        kyle_lambda = cov / (var + self.eps)
        
        # Invert: lower impact = better liquidity = higher score
        return z_score_cs(-kyle_lambda, self.eps)
    
    def _factor_glft_volume_clock_ofi(self) -> "xr.DataArray":
        """Volume-clock adjusted OFI.
        
        Normalizes OFI by volume clock (cumulative volume buckets)
        to create time-invariant order flow measure.
        """
        signed_vol = self._compute_signed_volume()
        dollar_vol = self._compute_dollar_volume()
        
        # Volume-weighted signed volume
        dv_signed = signed_vol * self.close  # Dollar-signed volume
        
        # Rolling sums
        dv_signed_sum = dv_signed.rolling(time=self.volume_clock_win, min_periods=self.volume_clock_win//2).sum()
        dv_sum = dollar_vol.rolling(time=self.volume_clock_win, min_periods=self.volume_clock_win//2).sum()
        
        # Volume-clock OFI
        vc_ofi = dv_signed_sum / (dv_sum + self.eps)
        
        return z_score_cs(vc_ofi, self.eps)
    
    def _factor_glft_bvc_imbalance(self) -> "xr.DataArray":
        """Bulk Volume Classification (BVC) imbalance.
        
        Classifies volume into buy/sell using price position within bar.
        BVC = (Close - Low) / (High - Low) approximates buy fraction.
        """
        high, low, close = self.high, self.low, self.close
        vol = self.vol
        
        # BVC: close position within bar
        bar_range = high - low + self.eps
        bvc = (close - low) / bar_range  # 0 = at low, 1 = at high
        
        # Buy volume estimate
        buy_vol = bvc * vol
        sell_vol = (1 - bvc) * vol
        
        # Rolling imbalance
        buy_sum = buy_vol.rolling(time=self.bvc_win, min_periods=self.bvc_win//2).sum()
        sell_sum = sell_vol.rolling(time=self.bvc_win, min_periods=self.bvc_win//2).sum()
        total = buy_sum + sell_sum + self.eps
        
        bvc_imbalance = (buy_sum - sell_sum) / total
        
        return z_score_cs(bvc_imbalance, self.eps)
    
    def _factor_glft_flow_dispersion(self) -> "xr.DataArray":
        """Cross-sectional flow dispersion - regime indicator.
        
        High dispersion = heterogeneous market (stock-picking opportunity)
        Low dispersion = herding behavior (momentum/reversal regime)
        
        Stocks with median flow have most reliable signals.
        """
        ofi = self._compute_ofi(self.ofi_med_win)
        
        # Cross-sectional standard deviation
        ofi_cs_std = ofi.std(dim="asset")
        
        # Distance from cross-sectional median
        ofi_median = ofi.median(dim="asset")
        dist_from_median = np.abs(ofi - ofi_median)
        
        # Combine: favor stocks near median when dispersion is high
        # (more reliable signals in heterogeneous market)
        dispersion_broadcast = ofi_cs_std.broadcast_like(ofi)
        
        # Near-median in high-dispersion = reliable = high score
        reliability = dispersion_broadcast / (dist_from_median + self.eps)
        
        return z_score_cs(reliability, self.eps)
    
    def _factor_glft_reservation_signal(self) -> "xr.DataArray":
        """Avellaneda-Stoikov reservation price signal.
        
        Reservation price: r = s - q·γ·σ²·τ
        
        When price < reservation → undervalued → buy signal
        We estimate reservation as: price - inventory_effect
        """
        close = self.close
        returns = self.ret
        inventory = self._compute_inventory_proxy()
        
        # Estimate volatility
        vol = returns.rolling(time=self.reservation_win, min_periods=self.reservation_win//2).std()
        
        # Inventory effect (q·γ·σ²·τ proxy)
        # γ (risk aversion) and τ (horizon) absorbed into scaling
        inventory_effect = inventory * (vol ** 2)
        
        # Reservation price proxy
        reservation = close - inventory_effect
        
        # Signal: actual price vs reservation
        # price < reservation → undervalued → positive signal
        signal = (reservation - close) / (close + self.eps)
        
        return z_score_cs(signal, self.eps)
    
    def _factor_glft_amihud_hybrid(self) -> "xr.DataArray":
        """Amihud-GLFT hybrid liquidity measure.
        
        Combines:
        - Amihud illiquidity ratio
        - Kyle's lambda (price impact)
        - Spread estimate
        
        Creates comprehensive liquidity score.
        """
        returns = self.ret
        dollar_vol = self._compute_dollar_volume()
        high, low = self.high, self.low
        
        # Amihud: |return| / dollar volume
        amihud = np.abs(returns) / (dollar_vol + self.eps)
        amihud_roll = amihud.rolling(time=self.impact_win, min_periods=self.impact_win//2).mean()
        
        # Spread estimate
        hl_ratio = np.log(high / (low + self.eps) + self.eps)
        spread = hl_ratio.rolling(time=self.spread_est_win, min_periods=self.spread_est_win//2).mean()
        
        # Volume normalized
        vol_norm = self.vol / (self.vol.rolling(time=63, min_periods=42).mean() + self.eps)
        
        # Hybrid: low Amihud + low spread + high volume = liquid = good
        amihud_z = z_score_cs(-amihud_roll, self.eps)  # Invert: low Amihud = good
        spread_z = z_score_cs(-spread, self.eps)       # Invert: low spread = good
        vol_z = z_score_cs(vol_norm, self.eps)         # High volume = good
        
        hybrid = 0.4 * amihud_z + 0.3 * spread_z + 0.3 * vol_z
        
        return z_score_cs(hybrid, self.eps)
