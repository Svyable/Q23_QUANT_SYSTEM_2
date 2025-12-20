"""GLFTv1 Factor Library - Market Microstructure Based Factors.

Implements factors derived from the GLFT market-making framework and
modern microstructure research.

Key concepts:
1. Order Flow Imbalance (OFI): Signed volume analysis
2. VPIN: Volume-synchronized probability of informed trading
3. Price Impact: How trades move prices (Kyle's lambda)
4. Inventory Risk: Premium for holding positions

Cross-sectional application:
- Compute microstructure metrics for each stock
- Rank stocks by toxicity, OFI, inventory signals
- Generate alpha by exploiting microstructure inefficiencies

SOLID Principles Applied:
- Single Responsibility: Only handles GLFT-specific factor computation
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


def ewma(x: "xr.DataArray", span: int) -> "xr.DataArray":
    """Exponentially weighted moving average."""
    df = x.transpose("time", "asset").to_pandas()
    out = df.ewm(span=span, adjust=False, min_periods=span).mean()
    return xr.DataArray(
        out.values, 
        coords={"time": x.time, "asset": x.asset}, 
        dims=["time", "asset"]
    ).fillna(0.0)


class GLFTFactorLibrary(FactorLibrary):
    """Extended factor library with GLFT microstructure factors.
    
    Adds 10 novel microstructure-based factors to the standard factor library.
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
        
        # GLFT-specific config from params (use getattr for flexibility)
        p = self.params
        self.ofi_short_win = getattr(p, 'OFI_SHORT_WIN', 5)
        self.ofi_med_win = getattr(p, 'OFI_MED_WIN', 21)
        self.ofi_long_win = getattr(p, 'OFI_LONG_WIN', 63)
        self.vpin_bucket_size = getattr(p, 'VPIN_BUCKET_SIZE', 50)
        self.vpin_win = getattr(p, 'VPIN_WIN', 50)
        self.spread_est_win = getattr(p, 'SPREAD_EST_WIN', 21)
        self.impact_win = getattr(p, 'IMPACT_WIN', 42)
        self.inventory_decay = getattr(p, 'INVENTORY_DECAY', 0.95)
        self.eps = p.eps
    
    def _compute_signed_volume(self) -> "xr.DataArray":
        """Compute signed volume using Lee-Ready style classification.
        
        Without tick data, we use price change to classify trades:
        - Price up: volume is buy-initiated
        - Price down: volume is sell-initiated
        """
        # Use parent class properties
        close = self.close
        vol = self.vol
        
        # Price change sign determines trade direction
        price_change = close.diff(dim="time")
        sign = xr.where(price_change > 0, 1.0, xr.where(price_change < 0, -1.0, 0.0))
        
        # Signed volume
        signed_vol = sign * vol
        
        return signed_vol
    
    def _compute_ofi(self, window: int) -> "xr.DataArray":
        """Compute Order Flow Imbalance over a window.
        
        OFI = sum(signed_volume) / sum(abs_volume)
        
        Positive OFI = net buying pressure
        Negative OFI = net selling pressure
        """
        signed_vol = self._compute_signed_volume()
        vol = self.vol
        
        # Rolling sums
        signed_sum = signed_vol.rolling(time=window, min_periods=window//2).sum()
        vol_sum = vol.rolling(time=window, min_periods=window//2).sum()
        
        # OFI normalized by total volume
        ofi = signed_sum / (vol_sum + self.eps)
        
        return ofi.clip(-1, 1)
    
    def _compute_vpin_proxy(self) -> "xr.DataArray":
        """Compute VPIN-style flow toxicity proxy.
        
        VPIN (Volume-synchronized Probability of INformed trading) measures
        the probability that a trade is from an informed trader.
        
        Without tick data, we approximate using:
        - Volume-weighted absolute price changes
        - Rolling standard deviation of signed volume
        """
        signed_vol = self._compute_signed_volume()
        vol = self.vol
        
        # Compute buy/sell volume split (using sign)
        buy_vol = xr.where(signed_vol > 0, signed_vol, 0)
        sell_vol = xr.where(signed_vol < 0, -signed_vol, 0)
        
        # Rolling buy and sell volumes
        buy_sum = buy_vol.rolling(time=self.vpin_win, min_periods=self.vpin_win//2).sum()
        sell_sum = sell_vol.rolling(time=self.vpin_win, min_periods=self.vpin_win//2).sum()
        total_vol = buy_sum + sell_sum + self.eps
        
        # VPIN proxy: absolute imbalance / total volume
        vpin = np.abs(buy_sum - sell_sum) / total_vol
        
        return vpin
    
    def _compute_inventory_proxy(self) -> "xr.DataArray":
        """Compute cumulative inventory proxy.
        
        Simulates market maker inventory using exponentially-weighted
        cumulative signed volume.
        """
        signed_vol = self._compute_signed_volume()
        
        # EWMA of signed volume as inventory proxy
        # High positive = accumulated longs, high negative = accumulated shorts
        inventory = ewma(signed_vol, span=int(1/(1-self.inventory_decay)))
        
        return inventory
    
    # =========================================================================
    # GLFT NOVEL FACTORS
    # =========================================================================
    
    def _factor_glft_ofi_short(self) -> "xr.DataArray":
        """Short-term Order Flow Imbalance (5d).
        
        Captures recent buying/selling pressure.
        Positive OFI = buying pressure = expect continuation.
        """
        ofi = self._compute_ofi(self.ofi_short_win)
        return z_score_cs(ofi, self.eps)
    
    def _factor_glft_ofi_med(self) -> "xr.DataArray":
        """Medium-term Order Flow Imbalance (21d).
        
        More stable measure of order flow direction.
        """
        ofi = self._compute_ofi(self.ofi_med_win)
        return z_score_cs(ofi, self.eps)
    
    def _factor_glft_ofi_momentum(self) -> "xr.DataArray":
        """OFI Momentum - Change in order flow imbalance.
        
        Acceleration in buying/selling pressure.
        Rising OFI = increasing demand = positive signal.
        """
        ofi_short = self._compute_ofi(self.ofi_short_win)
        ofi_long = self._compute_ofi(self.ofi_long_win)
        
        # OFI momentum: short-term vs long-term
        ofi_mom = ofi_short - ofi_long
        
        return z_score_cs(ofi_mom, self.eps)
    
    def _factor_glft_flow_toxicity(self) -> "xr.DataArray":
        """VPIN-style flow toxicity measure.
        
        High toxicity = informed trading = avoid providing liquidity.
        Low toxicity = safer to hold = positive signal for fundamental plays.
        
        We invert: low toxicity = high score (safer stocks).
        """
        vpin = self._compute_vpin_proxy()
        
        # Invert: lower toxicity = higher score
        factor = -vpin
        
        return z_score_cs(factor, self.eps)
    
    def _factor_glft_toxic_momentum(self) -> "xr.DataArray":
        """Change in toxicity - informed flow acceleration.
        
        Rising toxicity = increasing informed trading = caution.
        Falling toxicity = normalizing conditions = opportunity.
        """
        vpin = self._compute_vpin_proxy()
        
        vpin_ma_short = vpin.rolling(time=5, min_periods=3).mean()
        vpin_ma_long = vpin.rolling(time=21, min_periods=14).mean()
        
        # Falling toxicity = positive signal
        toxic_mom = -(vpin_ma_short - vpin_ma_long)
        
        return z_score_cs(toxic_mom, self.eps)
    
    def _factor_glft_impact_asymmetry(self) -> "xr.DataArray":
        """Asymmetric price impact: buy vs sell pressure.
        
        Measures whether buys move price more than sells (or vice versa).
        Asymmetry suggests informed flow on one side.
        """
        returns = self.ret
        vol = self.vol
        
        # Separate up and down returns
        up_ret = xr.where(returns > 0, returns, 0)
        down_ret = xr.where(returns < 0, returns, 0)
        
        # Volume on up vs down days
        up_vol = xr.where(returns > 0, vol, 0)
        down_vol = xr.where(returns < 0, vol, 0)
        
        # Rolling impact
        up_impact = (up_ret * up_vol).rolling(time=self.impact_win, min_periods=self.impact_win//2).sum()
        up_vol_sum = up_vol.rolling(time=self.impact_win, min_periods=self.impact_win//2).sum()
        
        down_impact = (down_ret * down_vol).rolling(time=self.impact_win, min_periods=self.impact_win//2).sum()
        down_vol_sum = down_vol.rolling(time=self.impact_win, min_periods=self.impact_win//2).sum()
        
        avg_up_impact = up_impact / (up_vol_sum + self.eps)
        avg_down_impact = np.abs(down_impact) / (down_vol_sum + self.eps)
        
        # Asymmetry: positive if buys have more impact (bullish)
        asymmetry = avg_up_impact - avg_down_impact
        
        return z_score_cs(asymmetry, self.eps)
    
    def _factor_glft_inventory_signal(self) -> "xr.DataArray":
        """Inventory-based contrarian signal.
        
        Based on GLFT: large inventory = pressure to unwind = mean reversion.
        High accumulated buys = expect selling = negative signal.
        """
        inventory = self._compute_inventory_proxy()
        
        # Contrarian: high inventory = expect reversal
        # Negative inventory relationship to future returns
        factor = -inventory
        
        return z_score_cs(factor, self.eps)
    
    def _factor_glft_inventory_risk_prem(self) -> "xr.DataArray":
        """Premium for holding inventory risk.
        
        Stocks with high inventory volatility should have higher expected returns
        as compensation for the risk.
        """
        inventory = self._compute_inventory_proxy()
        
        # Inventory volatility
        inv_vol = inventory.rolling(time=self.impact_win, min_periods=self.impact_win//2).std()
        
        # Cross-sectional volatility as risk measure
        # Higher inventory volatility = higher risk premium expected
        return z_score_cs(inv_vol, self.eps)
    
    def _factor_glft_spread_adjusted_mom(self) -> "xr.DataArray":
        """Momentum adjusted for spread/transaction costs.
        
        Raw momentum minus estimated trading costs.
        Only count momentum that survives transaction costs.
        """
        returns = self.ret
        high = self.high
        low = self.low
        
        # High-Low spread estimator (Corwin-Schultz style)
        hl_ratio = np.log(high / (low + self.eps) + self.eps)
        spread_est = hl_ratio.rolling(time=self.spread_est_win, min_periods=self.spread_est_win//2).mean()
        
        # Raw momentum
        mom_21 = returns.rolling(time=21, min_periods=15).sum()
        
        # Spread-adjusted: momentum minus round-trip cost estimate
        spread_adj_mom = mom_21 - 2 * spread_est  # Round-trip cost
        
        return z_score_cs(spread_adj_mom, self.eps)
    
    def _factor_glft_mm_edge(self) -> "xr.DataArray":
        """Market maker edge - when to provide vs take liquidity.
        
        Combines:
        - Low toxicity (safe to provide)
        - High spread (profitable to provide)
        - Mean-reverting behavior (inventory will normalize)
        
        High score = good conditions for patient, liquidity-providing strategies.
        """
        vpin = self._compute_vpin_proxy()
        returns = self.ret
        
        # Components
        low_toxicity = -vpin  # Inverted: low toxicity = high score
        
        # Volatility (higher = wider spreads = more MM profit)
        vol_21 = returns.rolling(time=21, min_periods=15).std()
        
        # Mean reversion indicator (from short-term reversal)
        ret_5 = returns.rolling(time=5, min_periods=3).sum()
        mean_reversion = -ret_5  # Recent losers tend to revert
        
        # Combine into MM edge score
        # Normalize each component
        low_tox_z = z_score_cs(low_toxicity, self.eps)
        vol_z = z_score_cs(vol_21, self.eps)
        mr_z = z_score_cs(mean_reversion, self.eps)
        
        # Weighted combination
        mm_edge = 0.4 * low_tox_z + 0.3 * vol_z + 0.3 * mr_z
        
        return z_score_cs(mm_edge, self.eps)
