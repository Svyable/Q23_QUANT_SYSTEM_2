"""
q23.strategy.factors
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Any

import numpy as np
import pandas as pd

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from q23.shared.math_utils import safe_zscore, safe_corrcoef


def _require_xr() -> None:
    if xr is None:
        raise ImportError("xarray is required for q23.strategy.factors")


# ==============================================================================
# Factor Categories & Metadata
# ==============================================================================

class FactorCategory(str, Enum):
    """Factor categories for organization and filtering."""
    DEFENSIVE = "defensive"      # Low-risk / volatility focused
    MOMENTUM = "momentum"        # Price momentum and trend
    REVERSAL = "reversal"        # Mean-reversion signals
    TECHNICAL = "technical"      # Technical analysis based
    QUALITY = "quality"          # Quality metrics
    VALUE = "value"              # Value indicators
    LIQUIDITY = "liquidity"      # Liquidity metrics
    MICROSTRUCTURE = "microstructure"  # Market microstructure
    INTERACTION = "interaction"  # Factor combinations
    CROSS_SECTIONAL = "cross_sectional"  # Relative signals
    MTF_MOMENTUM = "mtf_momentum"  # Multi-timeframe momentum (W/M/Q/Y)
    MTF_ALIGNMENT = "mtf_alignment"  # Multi-timeframe alignment signals
    MTF_HLOC = "mtf_hloc"        # Multi-timeframe HLOC-enhanced
    OU_MEAN_REVERSION = "ou_mean_reversion"  # Ornstein-Uhlenbeck mean-reversion
    GLFT_MICROSTRUCTURE = "glft_microstructure"  # GLFT market-making signals


@dataclass(frozen=True)
class FactorDefinition:
    """Metadata for a single factor - used for registry and dashboard display."""
    name: str
    category: FactorCategory
    description: str
    window_params: Tuple[str, ...]  # Names of window params this factor uses
    sign: int = 1  # +1 = higher is better, -1 = lower is better
    tradeable: bool = True  # Can be used in strategies

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "window_params": list(self.window_params),
            "sign": self.sign,
            "tradeable": self.tradeable,
        }


# ==============================================================================
# Factor Registry - Single source of truth for all factors
# ==============================================================================

FACTOR_REGISTRY: Dict[str, FactorDefinition] = {
    # Defensive / Low Risk
    "inv_vol": FactorDefinition(
        name="inv_vol",
        category=FactorCategory.DEFENSIVE,
        description="Inverse ATR volatility - favor low-volatility assets",
        window_params=("ATR_WIN",),
        sign=1,
    ),
    "inv_idio": FactorDefinition(
        name="inv_idio",
        category=FactorCategory.DEFENSIVE,
        description="Inverse idiosyncratic volatility after beta adjustment",
        window_params=("IDIO_WIN", "BETA_WIN"),
        sign=1,
    ),
    "inv_down": FactorDefinition(
        name="inv_down",
        category=FactorCategory.DEFENSIVE,
        description="Inverse downside volatility - protect against left-tail",
        window_params=("DOWN_WIN",),
        sign=1,
    ),
    "low_corr": FactorDefinition(
        name="low_corr",
        category=FactorCategory.DEFENSIVE,
        description="Low market correlation - diversification benefit",
        window_params=("CORR_WIN",),
        sign=1,
    ),
    "low_beta": FactorDefinition(
        name="low_beta",
        category=FactorCategory.DEFENSIVE,
        description="Low market beta - reduced systematic risk",
        window_params=("BETA_WIN",),
        sign=1,
    ),
    
    # Liquidity
    "liquidity": FactorDefinition(
        name="liquidity",
        category=FactorCategory.LIQUIDITY,
        description="Average dollar volume relative to universe",
        window_params=("ADV_WIN_LONG",),
        sign=1,
    ),
    "amihud_inv": FactorDefinition(
        name="amihud_inv",
        category=FactorCategory.LIQUIDITY,
        description="Inverse Amihud illiquidity - favor liquid names",
        window_params=(),
        sign=1,
    ),
    
    # Momentum
    "resid_mom": FactorDefinition(
        name="resid_mom",
        category=FactorCategory.MOMENTUM,
        description="Residual momentum (beta-adjusted) over medium term",
        window_params=("MOM_WIN", "BETA_WIN"),
        sign=1,
    ),
    "resid_mom_short": FactorDefinition(
        name="resid_mom_short",
        category=FactorCategory.MOMENTUM,
        description="Short-term residual momentum (21d)",
        window_params=("MOM_SHORT", "BETA_WIN"),
        sign=1,
    ),
    "resid_mom_long": FactorDefinition(
        name="resid_mom_long",
        category=FactorCategory.MOMENTUM,
        description="Long-term residual momentum (126d)",
        window_params=("MOM_LONG", "BETA_WIN"),
        sign=1,
    ),
    "resid_mom_mix": FactorDefinition(
        name="resid_mom_mix",
        category=FactorCategory.MOMENTUM,
        description="Blended momentum: 50% 63d + 30% 126d + 20% 21d",
        window_params=("BETA_WIN",),
        sign=1,
    ),
    "cross_momentum": FactorDefinition(
        name="cross_momentum",
        category=FactorCategory.CROSS_SECTIONAL,
        description="Cross-sectional momentum rank (demeaned)",
        window_params=("MOM_WIN",),
        sign=1,
    ),
    
    # Reversal
    "srev": FactorDefinition(
        name="srev",
        category=FactorCategory.REVERSAL,
        description="Short-term reversal (5d) - mean reversion signal",
        window_params=("REV_WIN",),
        sign=1,
    ),
    "lrev": FactorDefinition(
        name="lrev",
        category=FactorCategory.REVERSAL,
        description="Long-term reversal (21d) - contrarian signal",
        window_params=("REV_LONG",),
        sign=1,
    ),
    
    # Technical
    "breakout": FactorDefinition(
        name="breakout",
        category=FactorCategory.TECHNICAL,
        description="Donchian channel breakout position (0-1)",
        window_params=("DON_WIN",),
        sign=1,
    ),
    "slope": FactorDefinition(
        name="slope",
        category=FactorCategory.TECHNICAL,
        description="EMA trend slope normalized by ATR",
        window_params=("EMA_FAST", "EMA_SLOW", "ATR_WIN"),
        sign=1,
    ),
    "calm_flow": FactorDefinition(
        name="calm_flow",
        category=FactorCategory.TECHNICAL,
        description="Volume normalization - favor declining volume",
        window_params=("ADV_WIN_SHORT", "ADV_WIN_LONG"),
        sign=1,
    ),
    "prox_52w_high": FactorDefinition(
        name="prox_52w_high",
        category=FactorCategory.TECHNICAL,
        description="Proximity to 52-week high (inverted)",
        window_params=(),
        sign=1,
    ),
    "vol_surprise": FactorDefinition(
        name="vol_surprise",
        category=FactorCategory.TECHNICAL,
        description="Volume surprise vs 20d average",
        window_params=(),
        sign=1,
    ),
    "vol_breakout": FactorDefinition(
        name="vol_breakout",
        category=FactorCategory.TECHNICAL,
        description="ATR expansion with positive momentum confirmation",
        window_params=("ATR_WIN",),
        sign=1,
    ),
    "ma_cloud": FactorDefinition(
        name="ma_cloud",
        category=FactorCategory.TECHNICAL,
        description="Moving average cloud: distance from MA50/MA200",
        window_params=(),
        sign=1,
    ),
    
    # Microstructure / Quality
    "idio_tail_risk": FactorDefinition(
        name="idio_tail_risk",
        category=FactorCategory.MICROSTRUCTURE,
        description="Idiosyncratic tail risk (5th percentile/vol)",
        window_params=("BETA_WIN",),
        sign=1,
    ),
    "idio_jump_freq": FactorDefinition(
        name="idio_jump_freq",
        category=FactorCategory.MICROSTRUCTURE,
        description="Frequency of large idiosyncratic moves (>2.5σ)",
        window_params=("BETA_WIN",),
        sign=1,
    ),
    "beta_stability": FactorDefinition(
        name="beta_stability",
        category=FactorCategory.MICROSTRUCTURE,
        description="Beta stability over rolling 21d window",
        window_params=("BETA_WIN",),
        sign=1,
    ),
    "micro_noise": FactorDefinition(
        name="micro_noise",
        category=FactorCategory.MICROSTRUCTURE,
        description="Microstructure noise ratio (close-close vs Parkinson)",
        window_params=(),
        sign=1,
    ),
    
    # Quality / Value
    "quality_score": FactorDefinition(
        name="quality_score",
        category=FactorCategory.QUALITY,
        description="Composite quality: low vol + high liquidity + stable returns",
        window_params=("ATR_WIN", "ADV_WIN_LONG", "IDIO_WIN"),
        sign=1,
    ),
    "value_score": FactorDefinition(
        name="value_score",
        category=FactorCategory.VALUE,
        description="Value proxy: price relative to long-term MA",
        window_params=("VALUE_WIN",),
        sign=1,
    ),
    
    # Interactions
    "value_mom": FactorDefinition(
        name="value_mom",
        category=FactorCategory.INTERACTION,
        description="Value-momentum interaction: resid_mom × srev",
        window_params=("MOM_WIN", "REV_WIN", "BETA_WIN"),
        sign=1,
    ),
    "quality_defensive": FactorDefinition(
        name="quality_defensive",
        category=FactorCategory.INTERACTION,
        description="Quality-defensive interaction: liquidity × inv_vol",
        window_params=("ADV_WIN_LONG", "ATR_WIN"),
        sign=1,
    ),
    
    # Cross-sectional
    "rel_sector_mom": FactorDefinition(
        name="rel_sector_mom",
        category=FactorCategory.CROSS_SECTIONAL,
        description="Sector-relative momentum (vs universe mean)",
        window_params=("SECTOR_MOM_WIN",),
        sign=1,
    ),
    "cross_volatility": FactorDefinition(
        name="cross_volatility",
        category=FactorCategory.CROSS_SECTIONAL,
        description="Cross-sectional volatility rank (demeaned)",
        window_params=("IDIO_WIN",),
        sign=1,
    ),
    "cross_liquidity": FactorDefinition(
        name="cross_liquidity",
        category=FactorCategory.CROSS_SECTIONAL,
        description="Cross-sectional liquidity rank (demeaned)",
        window_params=("ADV_WIN_LONG",),
        sign=1,
    ),
    
    # ==========================================================================
    # Multi-Timeframe Momentum (MTF) Factors - NEW
    # Weekly/Monthly/Quarterly/Yearly with IC-adaptive weighting
    # ==========================================================================
    
    # To-Date Returns
    "ret_wtd": FactorDefinition(
        name="ret_wtd",
        category=FactorCategory.MTF_MOMENTUM,
        description="Week-to-date return (from last Monday)",
        window_params=("WEEK",),
        sign=1,
    ),
    "ret_mtd": FactorDefinition(
        name="ret_mtd",
        category=FactorCategory.MTF_MOMENTUM,
        description="Month-to-date return (from 1st of month)",
        window_params=("MONTH",),
        sign=1,
    ),
    "ret_qtd": FactorDefinition(
        name="ret_qtd",
        category=FactorCategory.MTF_MOMENTUM,
        description="Quarter-to-date return",
        window_params=("QUARTER",),
        sign=1,
    ),
    "ret_ytd": FactorDefinition(
        name="ret_ytd",
        category=FactorCategory.MTF_MOMENTUM,
        description="Year-to-date return",
        window_params=("YEAR",),
        sign=1,
    ),
    
    # Completed Period Returns
    "ret_1w": FactorDefinition(
        name="ret_1w",
        category=FactorCategory.MTF_MOMENTUM,
        description="Last complete week return (5 trading days)",
        window_params=("WEEK",),
        sign=1,
    ),
    "ret_1m": FactorDefinition(
        name="ret_1m",
        category=FactorCategory.MTF_MOMENTUM,
        description="Last 21 trading days return (~1 month)",
        window_params=("MONTH",),
        sign=1,
    ),
    "ret_1q": FactorDefinition(
        name="ret_1q",
        category=FactorCategory.MTF_MOMENTUM,
        description="Last 63 trading days return (~1 quarter)",
        window_params=("QUARTER",),
        sign=1,
    ),
    "ret_1y": FactorDefinition(
        name="ret_1y",
        category=FactorCategory.MTF_MOMENTUM,
        description="Last 252 trading days return (~1 year)",
        window_params=("YEAR",),
        sign=1,
    ),
    "ret_2_12m": FactorDefinition(
        name="ret_2_12m",
        category=FactorCategory.MTF_MOMENTUM,
        description="Months 2-12 return (classic momentum, skip recent)",
        window_params=("MONTH", "YEAR"),
        sign=1,
    ),
    
    # Alignment / Confluence
    "mtf_alignment": FactorDefinition(
        name="mtf_alignment",
        category=FactorCategory.MTF_ALIGNMENT,
        description="Momentum direction alignment across W/M/Q/Y (0-4 score)",
        window_params=("WEEK", "MONTH", "QUARTER", "YEAR"),
        sign=1,
    ),
    "mtf_alignment_strength": FactorDefinition(
        name="mtf_alignment_strength",
        category=FactorCategory.MTF_ALIGNMENT,
        description="Alignment weighted by return magnitude",
        window_params=("WEEK", "MONTH", "QUARTER"),
        sign=1,
    ),
    "trend_consistency": FactorDefinition(
        name="trend_consistency",
        category=FactorCategory.MTF_ALIGNMENT,
        description="Consistency of trend across timeframes",
        window_params=("WEEK", "MONTH", "QUARTER"),
        sign=1,
    ),
    
    # IC-Weighted Adaptive
    "mtf_ic_momentum": FactorDefinition(
        name="mtf_ic_momentum",
        category=FactorCategory.MTF_MOMENTUM,
        description="IC-weighted momentum composite (adapts to best timeframe)",
        window_params=("WEEK", "MONTH", "QUARTER", "IC_WINDOW"),
        sign=1,
    ),
    "mtf_ic_regime": FactorDefinition(
        name="mtf_ic_regime",
        category=FactorCategory.MTF_MOMENTUM,
        description="Which timeframe has highest IC (1=W, 2=M, 3=Q)",
        window_params=("IC_WINDOW",),
        sign=1,
    ),
    
    # Acceleration
    "momentum_acceleration": FactorDefinition(
        name="momentum_acceleration",
        category=FactorCategory.MTF_MOMENTUM,
        description="Short-term vs long-term momentum (acceleration signal)",
        window_params=("WEEK", "MONTH", "QUARTER"),
        sign=1,
    ),
    "mtf_slope": FactorDefinition(
        name="mtf_slope",
        category=FactorCategory.MTF_MOMENTUM,
        description="Slope of returns across timeframes (improving vs deteriorating)",
        window_params=("WEEK", "MONTH", "QUARTER"),
        sign=1,
    ),
    
    # HLOC-Enhanced
    "hloc_range_expansion": FactorDefinition(
        name="hloc_range_expansion",
        category=FactorCategory.MTF_HLOC,
        description="Weekly range vs monthly range (breakout signal)",
        window_params=("WEEK", "MONTH"),
        sign=1,
    ),
    "hloc_close_position": FactorDefinition(
        name="hloc_close_position",
        category=FactorCategory.MTF_HLOC,
        description="Close position within monthly HLOC range",
        window_params=("MONTH",),
        sign=1,
    ),
    "hloc_momentum_quality": FactorDefinition(
        name="hloc_momentum_quality",
        category=FactorCategory.MTF_HLOC,
        description="Returns supported by HLOC confirmation (close near high)",
        window_params=("WEEK", "MONTH"),
        sign=1,
    ),
    
    # Relative MTF
    "relative_mtf_momentum": FactorDefinition(
        name="relative_mtf_momentum",
        category=FactorCategory.MTF_MOMENTUM,
        description="MTF momentum relative to market (alpha component)",
        window_params=("WEEK", "MONTH", "QUARTER"),
        sign=1,
    ),
    "sector_relative_mtf": FactorDefinition(
        name="sector_relative_mtf",
        category=FactorCategory.MTF_MOMENTUM,
        description="MTF momentum vs universe mean (cross-sectional)",
        window_params=("MONTH", "QUARTER"),
        sign=1,
    ),
    
    # ==========================================================================
    # Neural Alpha Novel Factors - Behavioral Finance, Microstructure, Quality
    # ==========================================================================
    
    # Behavioral Finance Factors
    "attention_momentum": FactorDefinition(
        name="attention_momentum",
        category=FactorCategory.MOMENTUM,
        description="Volume-price attention signal (high volume + return = attention)",
        window_params=("ADV_WIN_LONG", "MOM_SHORT"),
        sign=1,
    ),
    "disposition_alpha": FactorDefinition(
        name="disposition_alpha",
        category=FactorCategory.REVERSAL,
        description="Disposition effect alpha - unrealized P&L vs anchoring price",
        window_params=("MOM_SHORT",),
        sign=1,
    ),
    "anchoring_bias": FactorDefinition(
        name="anchoring_bias",
        category=FactorCategory.TECHNICAL,
        description="Distance from psychological price levels (52w high/low)",
        window_params=(),
        sign=1,
    ),
    
    # Market Microstructure Factors
    "efficiency_ratio": FactorDefinition(
        name="efficiency_ratio",
        category=FactorCategory.TECHNICAL,
        description="Kaufman Efficiency Ratio - directional vs total movement",
        window_params=("MOM_WIN",),
        sign=1,
    ),
    "information_flow": FactorDefinition(
        name="information_flow",
        category=FactorCategory.MICROSTRUCTURE,
        description="Volume-weighted price impact asymmetry (informed trading)",
        window_params=("ADV_WIN_LONG",),
        sign=1,
    ),
    "mean_reversion_speed": FactorDefinition(
        name="mean_reversion_speed",
        category=FactorCategory.REVERSAL,
        description="Half-life of price deviations from trend",
        window_params=("EMA_SLOW",),
        sign=1,
    ),
    
    # Momentum Quality Factors
    "momentum_quality_ratio": FactorDefinition(
        name="momentum_quality_ratio",
        category=FactorCategory.QUALITY,
        description="Signal-to-noise ratio of momentum (rolling Sharpe)",
        window_params=("MOM_WIN",),
        sign=1,
    ),
    "momentum_persistence": FactorDefinition(
        name="momentum_persistence",
        category=FactorCategory.MOMENTUM,
        description="Autocorrelation-based momentum strength",
        window_params=("MOM_WIN",),
        sign=1,
    ),
    "momentum_divergence": FactorDefinition(
        name="momentum_divergence",
        category=FactorCategory.MOMENTUM,
        description="Price vs residual momentum divergence (alpha vs beta)",
        window_params=("MOM_WIN", "IDIO_WIN"),
        sign=1,
    ),
    
    # Risk/Regime Factors
    "vol_of_vol": FactorDefinition(
        name="vol_of_vol",
        category=FactorCategory.DEFENSIVE,
        description="Volatility of volatility - uncertainty premium (inverted)",
        window_params=("IDIO_WIN",),
        sign=1,
    ),
    "skewness_factor": FactorDefinition(
        name="skewness_factor",
        category=FactorCategory.QUALITY,
        description="Rolling return skewness - positive skew preference",
        window_params=("IDIO_WIN",),
        sign=1,
    ),
    "kurtosis_factor": FactorDefinition(
        name="kurtosis_factor",
        category=FactorCategory.QUALITY,
        description="Rolling return kurtosis - fat tail avoidance (inverted)",
        window_params=("IDIO_WIN",),
        sign=1,
    ),
    "regime_momentum": FactorDefinition(
        name="regime_momentum",
        category=FactorCategory.MOMENTUM,
        description="Volatility-regime-adaptive momentum blend",
        window_params=("MOM_SHORT", "MOM_LONG", "IDIO_WIN"),
        sign=1,
    ),
    
    # Cross-Sectional Factors
    "cross_sectional_dispersion": FactorDefinition(
        name="cross_sectional_dispersion",
        category=FactorCategory.CROSS_SECTIONAL,
        description="Return dispersion opportunity signal",
        window_params=("MOM_SHORT",),
        sign=1,
    ),
    "liquidity_momentum": FactorDefinition(
        name="liquidity_momentum",
        category=FactorCategory.LIQUIDITY,
        description="Liquidity-adjusted momentum (liquid momentum premium)",
        window_params=("MOM_WIN", "ADV_WIN_LONG"),
        sign=1,
    ),
    
    # ==========================================================================
    # Ornstein-Uhlenbeck (OU) Mean-Reversion Factors
    # Based on canonical OU process: dX = θ(μ - X)dt + σdW
    # ==========================================================================
    
    "ou_zscore_short": FactorDefinition(
        name="ou_zscore_short",
        category=FactorCategory.OU_MEAN_REVERSION,
        description="Short-term OU z-score (21d) - deviation from equilibrium",
        window_params=("OU_SHORT_WIN",),
        sign=1,
    ),
    "ou_zscore_med": FactorDefinition(
        name="ou_zscore_med",
        category=FactorCategory.OU_MEAN_REVERSION,
        description="Medium-term OU z-score (63d) - stable equilibrium estimate",
        window_params=("OU_MED_WIN",),
        sign=1,
    ),
    "ou_halflife_signal": FactorDefinition(
        name="ou_halflife_signal",
        category=FactorCategory.OU_MEAN_REVERSION,
        description="Inverse half-life signal - faster reversion = stronger",
        window_params=("OU_MED_WIN",),
        sign=1,
    ),
    "ou_reversion_strength": FactorDefinition(
        name="ou_reversion_strength",
        category=FactorCategory.OU_MEAN_REVERSION,
        description="Mean-reversion speed (θ) cross-sectionally ranked",
        window_params=("OU_MED_WIN",),
        sign=1,
    ),
    "ou_predicted_return": FactorDefinition(
        name="ou_predicted_return",
        category=FactorCategory.OU_MEAN_REVERSION,
        description="Expected return from OU dynamics: E[return] = θ(μ - X)",
        window_params=("OU_MED_WIN",),
        sign=1,
    ),
    "ou_regime_indicator": FactorDefinition(
        name="ou_regime_indicator",
        category=FactorCategory.OU_MEAN_REVERSION,
        description="OU parameter dispersion regime - median half-life proximity",
        window_params=("OU_MED_WIN",),
        sign=1,
    ),
    "ou_equilibrium_dist": FactorDefinition(
        name="ou_equilibrium_dist",
        category=FactorCategory.OU_MEAN_REVERSION,
        description="Blended distance from OU equilibrium (short/med)",
        window_params=("OU_SHORT_WIN", "OU_MED_WIN"),
        sign=1,
    ),
    "ou_momentum_blend": FactorDefinition(
        name="ou_momentum_blend",
        category=FactorCategory.OU_MEAN_REVERSION,
        description="OU-weighted momentum/reversal blend based on half-life",
        window_params=("OU_MED_WIN",),
        sign=1,
    ),
    
    # ==========================================================================
    # GLFT (Guéant–Lehalle–Fernandez-Tapia) Market Microstructure Factors
    # Derived from optimal market-making framework adapted for alpha
    # ==========================================================================
    
    "glft_ofi_short": FactorDefinition(
        name="glft_ofi_short",
        category=FactorCategory.GLFT_MICROSTRUCTURE,
        description="Short-term Order Flow Imbalance (5d) - buying/selling pressure",
        window_params=("OFI_SHORT_WIN",),
        sign=1,
    ),
    "glft_ofi_med": FactorDefinition(
        name="glft_ofi_med",
        category=FactorCategory.GLFT_MICROSTRUCTURE,
        description="Medium-term Order Flow Imbalance (21d)",
        window_params=("OFI_MED_WIN",),
        sign=1,
    ),
    "glft_ofi_momentum": FactorDefinition(
        name="glft_ofi_momentum",
        category=FactorCategory.GLFT_MICROSTRUCTURE,
        description="OFI Momentum - acceleration in order flow",
        window_params=("OFI_SHORT_WIN", "OFI_LONG_WIN"),
        sign=1,
    ),
    "glft_flow_toxicity": FactorDefinition(
        name="glft_flow_toxicity",
        category=FactorCategory.GLFT_MICROSTRUCTURE,
        description="VPIN-style flow toxicity (informed trading probability)",
        window_params=("VPIN_WIN",),
        sign=1,
    ),
    "glft_toxic_momentum": FactorDefinition(
        name="glft_toxic_momentum",
        category=FactorCategory.GLFT_MICROSTRUCTURE,
        description="Change in toxicity - informed flow acceleration",
        window_params=("VPIN_WIN",),
        sign=1,
    ),
    "glft_impact_asymmetry": FactorDefinition(
        name="glft_impact_asymmetry",
        category=FactorCategory.GLFT_MICROSTRUCTURE,
        description="Asymmetric price impact - buy vs sell pressure",
        window_params=("IMPACT_WIN",),
        sign=1,
    ),
    "glft_inventory_signal": FactorDefinition(
        name="glft_inventory_signal",
        category=FactorCategory.GLFT_MICROSTRUCTURE,
        description="Inventory-based contrarian signal from GLFT model",
        window_params=("INVENTORY_DECAY",),
        sign=1,
    ),
    "glft_inventory_risk_prem": FactorDefinition(
        name="glft_inventory_risk_prem",
        category=FactorCategory.GLFT_MICROSTRUCTURE,
        description="Premium for holding inventory risk",
        window_params=("IMPACT_WIN",),
        sign=1,
    ),
    "glft_spread_adjusted_mom": FactorDefinition(
        name="glft_spread_adjusted_mom",
        category=FactorCategory.GLFT_MICROSTRUCTURE,
        description="Momentum adjusted for spread/transaction costs",
        window_params=("SPREAD_EST_WIN",),
        sign=1,
    ),
    "glft_mm_edge": FactorDefinition(
        name="glft_mm_edge",
        category=FactorCategory.GLFT_MICROSTRUCTURE,
        description="Market maker edge score - optimal liquidity provision",
        window_params=("VPIN_WIN",),
        sign=1,
    ),
}


# ==============================================================================
# Window Configuration - PM-configurable parameters
# ==============================================================================

@dataclass
class FactorWindowConfig:
    """
    Configurable window parameters for factor computation.
    
    PMs can adjust these via dashboard to tune factor responsiveness.
    All windows are in trading days.
    
    Usage:
        # Default config
        config = FactorWindowConfig()
        
        # Faster-reacting config (scale all windows by 0.5)
        fast_config = config.scale(0.5)
        
        # Custom overrides
        custom_config = config.with_overrides(MOM_WIN=63, BETA_WIN=126)
    """
    # Risk/Beta estimation
    BETA_WIN: int = 162       # Rolling beta window (~ 6 months)
    IDIO_WIN: int = 63        # Idiosyncratic vol window (~ 3 months)
    DOWN_WIN: int = 63        # Downside vol window
    CORR_WIN: int = 63        # Correlation window
    
    # Volume/Liquidity
    ADV_WIN_LONG: int = 63    # Long-term ADV
    ADV_WIN_SHORT: int = 5    # Short-term ADV
    
    # Momentum
    MOM_WIN: int = 84         # Primary momentum lookback
    MOM_SHORT: int = 21       # Short-term momentum
    MOM_LONG: int = 126       # Long-term momentum
    
    # Reversal
    REV_WIN: int = 5          # Short-term reversal
    REV_LONG: int = 21        # Long-term reversal
    
    # Technical
    DON_WIN: int = 63         # Donchian channel
    EMA_FAST: int = 20        # Fast EMA
    EMA_SLOW: int = 42        # Slow EMA
    ATR_WIN: int = 42         # ATR window
    
    # Other
    SECTOR_MOM_WIN: int = 63  # Sector momentum
    VALUE_WIN: int = 200      # Value MA window
    
    # Multi-Timeframe (MTF) Windows - NEW
    WEEK: int = 5             # Weekly period (trading days)
    MONTH: int = 21           # Monthly period
    QUARTER: int = 63         # Quarterly period
    YEAR: int = 252           # Yearly period
    IC_WINDOW: int = 63       # Rolling IC computation window
    IC_EWMA_LAMBDA: float = 0.94  # IC smoothing parameter
    
    # Ornstein-Uhlenbeck (OU) Windows
    OU_SHORT_WIN: int = 21    # Short-term OU estimation (1 month)
    OU_MED_WIN: int = 63      # Medium-term OU estimation (3 months)
    OU_LONG_WIN: int = 126    # Long-term OU estimation (6 months)
    OU_HALFLIFE_MIN: int = 2  # Min half-life for valid signal (days)
    OU_HALFLIFE_MAX: int = 42 # Max half-life (longer = weaker signal)
    OU_ZSCORE_CLIP: float = 3.0  # Clip extreme z-scores
    
    # GLFT (Market Microstructure) Windows
    OFI_SHORT_WIN: int = 5    # Order flow imbalance short window
    OFI_MED_WIN: int = 21     # OFI medium window
    OFI_LONG_WIN: int = 63    # OFI long window
    VPIN_WIN: int = 50        # Number of buckets for VPIN
    SPREAD_EST_WIN: int = 21  # Window for spread estimation
    IMPACT_WIN: int = 42      # Price impact estimation window
    INVENTORY_DECAY: float = 0.95  # Decay for inventory proxy

    # Neural Alpha / behavioral-microstructure windows
    EFFICIENCY_WIN: int = 21      # Kaufman ER window
    ATTENTION_WIN: int = 10       # Attention smoothing window
    DISPOSITION_WIN: int = 63     # Disposition effect smoothing
    SKEW_WIN: int = 63            # Skewness window
    KURT_WIN: int = 63            # Kurtosis window
    VOV_WIN: int = 21             # Vol-of-vol base window
    PERSISTENCE_WIN: int = 42     # Momentum persistence (autocorr) window

    def scale(self, factor: float, min_window: int = 5) -> "FactorWindowConfig":
        """
        Scale all windows by a factor.
        
        Useful for regime-based adjustment:
        - factor < 1: Faster reacting (volatile regime)
        - factor > 1: Slower, more stable (calm regime)
        """
        return FactorWindowConfig(
            BETA_WIN=max(min_window, int(self.BETA_WIN * factor)),
            IDIO_WIN=max(min_window, int(self.IDIO_WIN * factor)),
            DOWN_WIN=max(min_window, int(self.DOWN_WIN * factor)),
            CORR_WIN=max(min_window, int(self.CORR_WIN * factor)),
            ADV_WIN_LONG=max(min_window, int(self.ADV_WIN_LONG * factor)),
            ADV_WIN_SHORT=max(min_window, int(self.ADV_WIN_SHORT * factor)),
            MOM_WIN=max(min_window, int(self.MOM_WIN * factor)),
            MOM_SHORT=max(min_window, int(self.MOM_SHORT * factor)),
            MOM_LONG=max(min_window, int(self.MOM_LONG * factor)),
            REV_WIN=max(min_window, int(self.REV_WIN * factor)),
            REV_LONG=max(min_window, int(self.REV_LONG * factor)),
            DON_WIN=max(min_window, int(self.DON_WIN * factor)),
            EMA_FAST=max(min_window, int(self.EMA_FAST * factor)),
            EMA_SLOW=max(min_window, int(self.EMA_SLOW * factor)),
            ATR_WIN=max(min_window, int(self.ATR_WIN * factor)),
            SECTOR_MOM_WIN=max(min_window, int(self.SECTOR_MOM_WIN * factor)),
            VALUE_WIN=max(min_window, int(self.VALUE_WIN * factor)),
            # MTF windows - scale proportionally
            WEEK=max(3, int(self.WEEK * factor)),
            MONTH=max(min_window, int(self.MONTH * factor)),
            QUARTER=max(min_window, int(self.QUARTER * factor)),
            YEAR=max(63, int(self.YEAR * factor)),  # Don't go below quarter
            IC_WINDOW=max(21, int(self.IC_WINDOW * factor)),
            IC_EWMA_LAMBDA=self.IC_EWMA_LAMBDA,  # Don't scale
            # OU windows - scale proportionally
            OU_SHORT_WIN=max(min_window, int(self.OU_SHORT_WIN * factor)),
            OU_MED_WIN=max(min_window, int(self.OU_MED_WIN * factor)),
            OU_LONG_WIN=max(21, int(self.OU_LONG_WIN * factor)),
            OU_HALFLIFE_MIN=self.OU_HALFLIFE_MIN,  # Don't scale
            OU_HALFLIFE_MAX=max(10, int(self.OU_HALFLIFE_MAX * factor)),
            OU_ZSCORE_CLIP=self.OU_ZSCORE_CLIP,  # Don't scale
            # GLFT windows - scale proportionally
            OFI_SHORT_WIN=max(3, int(self.OFI_SHORT_WIN * factor)),
            OFI_MED_WIN=max(min_window, int(self.OFI_MED_WIN * factor)),
            OFI_LONG_WIN=max(min_window, int(self.OFI_LONG_WIN * factor)),
            VPIN_WIN=max(10, int(self.VPIN_WIN * factor)),
            SPREAD_EST_WIN=max(min_window, int(self.SPREAD_EST_WIN * factor)),
            IMPACT_WIN=max(min_window, int(self.IMPACT_WIN * factor)),
            INVENTORY_DECAY=self.INVENTORY_DECAY,  # Don't scale
            # Neural windows - scale proportionally
            EFFICIENCY_WIN=max(min_window, int(self.EFFICIENCY_WIN * factor)),
            ATTENTION_WIN=max(3, int(self.ATTENTION_WIN * factor)),
            DISPOSITION_WIN=max(21, int(self.DISPOSITION_WIN * factor)),
            SKEW_WIN=max(21, int(self.SKEW_WIN * factor)),
            KURT_WIN=max(21, int(self.KURT_WIN * factor)),
            VOV_WIN=max(min_window, int(self.VOV_WIN * factor)),
            PERSISTENCE_WIN=max(21, int(self.PERSISTENCE_WIN * factor)),
        )

    def with_overrides(self, **kwargs) -> "FactorWindowConfig":
        """Create new config with specific window overrides."""
        return replace(self, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization/dashboard."""
        return {
            "BETA_WIN": self.BETA_WIN,
            "IDIO_WIN": self.IDIO_WIN,
            "DOWN_WIN": self.DOWN_WIN,
            "CORR_WIN": self.CORR_WIN,
            "ADV_WIN_LONG": self.ADV_WIN_LONG,
            "ADV_WIN_SHORT": self.ADV_WIN_SHORT,
            "MOM_WIN": self.MOM_WIN,
            "MOM_SHORT": self.MOM_SHORT,
            "MOM_LONG": self.MOM_LONG,
            "REV_WIN": self.REV_WIN,
            "REV_LONG": self.REV_LONG,
            "DON_WIN": self.DON_WIN,
            "EMA_FAST": self.EMA_FAST,
            "EMA_SLOW": self.EMA_SLOW,
            "ATR_WIN": self.ATR_WIN,
            "SECTOR_MOM_WIN": self.SECTOR_MOM_WIN,
            "VALUE_WIN": self.VALUE_WIN,
            # MTF parameters
            "WEEK": self.WEEK,
            "MONTH": self.MONTH,
            "QUARTER": self.QUARTER,
            "YEAR": self.YEAR,
            "IC_WINDOW": self.IC_WINDOW,
            "IC_EWMA_LAMBDA": self.IC_EWMA_LAMBDA,
            # OU parameters
            "OU_SHORT_WIN": self.OU_SHORT_WIN,
            "OU_MED_WIN": self.OU_MED_WIN,
            "OU_LONG_WIN": self.OU_LONG_WIN,
            "OU_HALFLIFE_MIN": self.OU_HALFLIFE_MIN,
            "OU_HALFLIFE_MAX": self.OU_HALFLIFE_MAX,
            "OU_ZSCORE_CLIP": self.OU_ZSCORE_CLIP,
            # GLFT parameters
            "OFI_SHORT_WIN": self.OFI_SHORT_WIN,
            "OFI_MED_WIN": self.OFI_MED_WIN,
            "OFI_LONG_WIN": self.OFI_LONG_WIN,
            "VPIN_WIN": self.VPIN_WIN,
            "SPREAD_EST_WIN": self.SPREAD_EST_WIN,
            "IMPACT_WIN": self.IMPACT_WIN,
            "INVENTORY_DECAY": self.INVENTORY_DECAY,
            # Neural windows
            "EFFICIENCY_WIN": self.EFFICIENCY_WIN,
            "ATTENTION_WIN": self.ATTENTION_WIN,
            "DISPOSITION_WIN": self.DISPOSITION_WIN,
            "SKEW_WIN": self.SKEW_WIN,
            "KURT_WIN": self.KURT_WIN,
            "VOV_WIN": self.VOV_WIN,
            "PERSISTENCE_WIN": self.PERSISTENCE_WIN,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "FactorWindowConfig":
        """Create from dict (dashboard form submission)."""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})

    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get parameter metadata for dashboard UI generation.
        
        Returns dict with min/max/step/description for each param.
        """
        return {
            "BETA_WIN": {"min": 21, "max": 252, "step": 21, "default": 162, "desc": "Beta estimation window"},
            "IDIO_WIN": {"min": 10, "max": 126, "step": 7, "default": 63, "desc": "Idiosyncratic vol window"},
            "DOWN_WIN": {"min": 10, "max": 126, "step": 7, "default": 63, "desc": "Downside volatility window"},
            "CORR_WIN": {"min": 10, "max": 126, "step": 7, "default": 63, "desc": "Correlation window"},
            "ADV_WIN_LONG": {"min": 10, "max": 126, "step": 7, "default": 63, "desc": "Long-term ADV window"},
            "ADV_WIN_SHORT": {"min": 3, "max": 21, "step": 1, "default": 5, "desc": "Short-term ADV window"},
            "MOM_WIN": {"min": 21, "max": 252, "step": 21, "default": 84, "desc": "Primary momentum window"},
            "MOM_SHORT": {"min": 5, "max": 63, "step": 7, "default": 21, "desc": "Short momentum window"},
            "MOM_LONG": {"min": 63, "max": 252, "step": 21, "default": 126, "desc": "Long momentum window"},
            "REV_WIN": {"min": 1, "max": 21, "step": 1, "default": 5, "desc": "Short-term reversal window"},
            "REV_LONG": {"min": 5, "max": 63, "step": 7, "default": 21, "desc": "Long-term reversal window"},
            "DON_WIN": {"min": 10, "max": 126, "step": 7, "default": 63, "desc": "Donchian channel window"},
            "EMA_FAST": {"min": 5, "max": 50, "step": 5, "default": 20, "desc": "Fast EMA span"},
            "EMA_SLOW": {"min": 20, "max": 100, "step": 7, "default": 42, "desc": "Slow EMA span"},
            "ATR_WIN": {"min": 10, "max": 63, "step": 7, "default": 42, "desc": "ATR window"},
            "SECTOR_MOM_WIN": {"min": 21, "max": 126, "step": 21, "default": 63, "desc": "Sector momentum window"},
            "VALUE_WIN": {"min": 100, "max": 504, "step": 21, "default": 200, "desc": "Value MA window"},
            # MTF Parameters
            "WEEK": {"min": 3, "max": 10, "step": 1, "default": 5, "desc": "Weekly period (trading days)"},
            "MONTH": {"min": 15, "max": 25, "step": 1, "default": 21, "desc": "Monthly period (trading days)"},
            "QUARTER": {"min": 42, "max": 84, "step": 7, "default": 63, "desc": "Quarterly period (trading days)"},
            "YEAR": {"min": 200, "max": 260, "step": 10, "default": 252, "desc": "Yearly period (trading days)"},
            "IC_WINDOW": {"min": 21, "max": 126, "step": 7, "default": 63, "desc": "Rolling IC computation window"},
            "IC_EWMA_LAMBDA": {"min": 0.85, "max": 0.99, "step": 0.01, "default": 0.94, "desc": "IC EWMA smoothing"},
            # OU Parameters
            "OU_SHORT_WIN": {"min": 10, "max": 42, "step": 7, "default": 21, "desc": "OU short estimation window"},
            "OU_MED_WIN": {"min": 21, "max": 126, "step": 7, "default": 63, "desc": "OU medium estimation window"},
            "OU_LONG_WIN": {"min": 63, "max": 252, "step": 21, "default": 126, "desc": "OU long estimation window"},
            "OU_HALFLIFE_MIN": {"min": 1, "max": 10, "step": 1, "default": 2, "desc": "Min valid half-life (days)"},
            "OU_HALFLIFE_MAX": {"min": 21, "max": 126, "step": 7, "default": 42, "desc": "Max valid half-life (days)"},
            "OU_ZSCORE_CLIP": {"min": 2.0, "max": 5.0, "step": 0.5, "default": 3.0, "desc": "OU z-score clipping"},
            # GLFT Parameters
            "OFI_SHORT_WIN": {"min": 3, "max": 10, "step": 1, "default": 5, "desc": "OFI short window"},
            "OFI_MED_WIN": {"min": 10, "max": 42, "step": 7, "default": 21, "desc": "OFI medium window"},
            "OFI_LONG_WIN": {"min": 21, "max": 126, "step": 7, "default": 63, "desc": "OFI long window"},
            "VPIN_WIN": {"min": 20, "max": 100, "step": 10, "default": 50, "desc": "VPIN bucket window"},
            "SPREAD_EST_WIN": {"min": 10, "max": 42, "step": 7, "default": 21, "desc": "Spread estimation window"},
            "IMPACT_WIN": {"min": 21, "max": 84, "step": 7, "default": 42, "desc": "Price impact window"},
            "INVENTORY_DECAY": {"min": 0.90, "max": 0.99, "step": 0.01, "default": 0.95, "desc": "Inventory decay factor"},
            # Neural / behavioral-microstructure params
            "EFFICIENCY_WIN": {"min": 10, "max": 84, "step": 7, "default": 21, "desc": "Efficiency ratio window"},
            "ATTENTION_WIN": {"min": 5, "max": 42, "step": 1, "default": 10, "desc": "Attention smoothing window"},
            "DISPOSITION_WIN": {"min": 21, "max": 252, "step": 21, "default": 63, "desc": "Disposition effect window"},
            "SKEW_WIN": {"min": 21, "max": 252, "step": 21, "default": 63, "desc": "Skewness window"},
            "KURT_WIN": {"min": 21, "max": 252, "step": 21, "default": 63, "desc": "Kurtosis window"},
            "VOV_WIN": {"min": 10, "max": 84, "step": 7, "default": 21, "desc": "Vol-of-vol base window"},
            "PERSISTENCE_WIN": {"min": 21, "max": 126, "step": 7, "default": 42, "desc": "Momentum persistence window"},
        }


# ==============================================================================
# Factor Params - Main configuration dataclass (backward compatible)
# ==============================================================================

@dataclass(frozen=True)
class FactorParams:
    """
    Complete factor computation parameters.
    
    Combines window config with normalization settings.
    Kept frozen for safety but can create modified versions via replace().
    """
    eps: float = 1e-12
    zclip: float = 5.0
    robust_cs_z: bool = True  # Use robust (median/MAD) cross-sectional z-score

    # Window configuration - can pass FactorWindowConfig or use defaults
    BETA_WIN: int = 162
    IDIO_WIN: int = 63
    DOWN_WIN: int = 63
    CORR_WIN: int = 63
    ADV_WIN_LONG: int = 63
    ADV_WIN_SHORT: int = 5
    MOM_WIN: int = 84
    MOM_SHORT: int = 21
    MOM_LONG: int = 126
    REV_WIN: int = 5
    REV_LONG: int = 21
    DON_WIN: int = 63
    EMA_FAST: int = 20
    EMA_SLOW: int = 42
    ATR_WIN: int = 42
    SECTOR_MOM_WIN: int = 63
    VALUE_WIN: int = 200
    
    # Multi-Timeframe (MTF) Parameters
    WEEK: int = 5
    MONTH: int = 21
    QUARTER: int = 63
    YEAR: int = 252
    IC_WINDOW: int = 63
    IC_EWMA_LAMBDA: float = 0.94
    
    # Ornstein-Uhlenbeck (OU) Parameters
    OU_SHORT_WIN: int = 21
    OU_MED_WIN: int = 63
    OU_LONG_WIN: int = 126
    OU_HALFLIFE_MIN: int = 2
    OU_HALFLIFE_MAX: int = 42
    OU_ZSCORE_CLIP: float = 3.0
    
    # GLFT (Market Microstructure) Parameters
    OFI_SHORT_WIN: int = 5
    OFI_MED_WIN: int = 21
    OFI_LONG_WIN: int = 63
    VPIN_WIN: int = 50
    SPREAD_EST_WIN: int = 21
    IMPACT_WIN: int = 42
    INVENTORY_DECAY: float = 0.95

    # Neural / behavioral-microstructure parameters
    EFFICIENCY_WIN: int = 21
    ATTENTION_WIN: int = 10
    DISPOSITION_WIN: int = 63
    SKEW_WIN: int = 63
    KURT_WIN: int = 63
    VOV_WIN: int = 21
    PERSISTENCE_WIN: int = 42

    @classmethod
    def from_window_config(
        cls,
        window_config: FactorWindowConfig,
        eps: float = 1e-12,
        zclip: float = 5.0,
        robust_cs_z: bool = True,
    ) -> "FactorParams":
        """Create FactorParams from a FactorWindowConfig."""
        return cls(
            eps=eps,
            zclip=zclip,
            robust_cs_z=robust_cs_z,
            **window_config.to_dict(),
        )

    def get_window_config(self) -> FactorWindowConfig:
        """Extract FactorWindowConfig from params."""
        return FactorWindowConfig(
            BETA_WIN=self.BETA_WIN,
            IDIO_WIN=self.IDIO_WIN,
            DOWN_WIN=self.DOWN_WIN,
            CORR_WIN=self.CORR_WIN,
            ADV_WIN_LONG=self.ADV_WIN_LONG,
            ADV_WIN_SHORT=self.ADV_WIN_SHORT,
            MOM_WIN=self.MOM_WIN,
            MOM_SHORT=self.MOM_SHORT,
            MOM_LONG=self.MOM_LONG,
            REV_WIN=self.REV_WIN,
            REV_LONG=self.REV_LONG,
            DON_WIN=self.DON_WIN,
            EMA_FAST=self.EMA_FAST,
            EMA_SLOW=self.EMA_SLOW,
            ATR_WIN=self.ATR_WIN,
            SECTOR_MOM_WIN=self.SECTOR_MOM_WIN,
            VALUE_WIN=self.VALUE_WIN,
            # MTF parameters
            WEEK=self.WEEK,
            MONTH=self.MONTH,
            QUARTER=self.QUARTER,
            YEAR=self.YEAR,
            IC_WINDOW=self.IC_WINDOW,
            IC_EWMA_LAMBDA=self.IC_EWMA_LAMBDA,
            # OU parameters
            OU_SHORT_WIN=self.OU_SHORT_WIN,
            OU_MED_WIN=self.OU_MED_WIN,
            OU_LONG_WIN=self.OU_LONG_WIN,
            OU_HALFLIFE_MIN=self.OU_HALFLIFE_MIN,
            OU_HALFLIFE_MAX=self.OU_HALFLIFE_MAX,
            OU_ZSCORE_CLIP=self.OU_ZSCORE_CLIP,
            # GLFT parameters
            OFI_SHORT_WIN=self.OFI_SHORT_WIN,
            OFI_MED_WIN=self.OFI_MED_WIN,
            OFI_LONG_WIN=self.OFI_LONG_WIN,
            VPIN_WIN=self.VPIN_WIN,
            SPREAD_EST_WIN=self.SPREAD_EST_WIN,
            IMPACT_WIN=self.IMPACT_WIN,
            INVENTORY_DECAY=self.INVENTORY_DECAY,
            # Neural windows
            EFFICIENCY_WIN=self.EFFICIENCY_WIN,
            ATTENTION_WIN=self.ATTENTION_WIN,
            DISPOSITION_WIN=self.DISPOSITION_WIN,
            SKEW_WIN=self.SKEW_WIN,
            KURT_WIN=self.KURT_WIN,
            VOV_WIN=self.VOV_WIN,
            PERSISTENCE_WIN=self.PERSISTENCE_WIN,
        )

    def with_scaled_windows(self, factor: float) -> "FactorParams":
        """Create new params with scaled windows."""
        scaled_config = self.get_window_config().scale(factor)
        return FactorParams.from_window_config(
            scaled_config,
            eps=self.eps,
            zclip=self.zclip,
            robust_cs_z=self.robust_cs_z,
        )

    def with_window_overrides(self, **kwargs) -> "FactorParams":
        """Create new params with specific window overrides."""
        return replace(self, **kwargs)


# ==============================================================================
# Default Factor Sets - Pre-configured factor groups
# ==============================================================================

# V4 24-factor set (original production set)
V4_24_FACTORS = (
    # Defensive / low risk
    "inv_vol",
    "inv_idio",
    "inv_down",
    "low_corr",
    "low_beta",
    "liquidity",
    "amihud_inv",

    # Alpha
    "resid_mom",
    "srev",
    "breakout",
    "slope",
    "calm_flow",
    "prox_52w_high",
    "resid_mom_mix",
    "vol_surprise",
    "vol_breakout",
    "ma_cloud",

    # Idio-quality / microstructure
    "idio_tail_risk",
    "idio_jump_freq",
    "beta_stability",
    "micro_noise",

    # Interactions
    "value_mom",
    "quality_defensive",

    # Sector-relative proxy
    "rel_sector_mom",
)

# Defensive-only subset
DEFENSIVE_FACTORS = (
    "inv_vol",
    "inv_idio",
    "inv_down",
    "low_corr",
    "low_beta",
    "liquidity",
    "amihud_inv",
)

# Alpha-focused subset
ALPHA_FACTORS = (
    "resid_mom",
    "srev",
    "breakout",
    "slope",
    "calm_flow",
    "prox_52w_high",
    "resid_mom_mix",
    "vol_surprise",
    "vol_breakout",
    "ma_cloud",
)

# Quality subset
QUALITY_FACTORS = (
    "idio_tail_risk",
    "idio_jump_freq",
    "beta_stability",
    "micro_noise",
    "quality_score",
)

# Multi-Timeframe Momentum subset (NEW)
MTF_MOMENTUM_FACTORS = (
    # Core returns at each timeframe
    "ret_wtd",
    "ret_mtd",
    "ret_qtd",
    "ret_ytd",
    "ret_1w",
    "ret_1m",
    "ret_1q",
    "ret_2_12m",
    # Alignment signals
    "mtf_alignment",
    "mtf_alignment_strength",
    "trend_consistency",
    # IC-weighted composite
    "mtf_ic_momentum",
    # Acceleration
    "momentum_acceleration",
    "mtf_slope",
    # HLOC-enhanced
    "hloc_close_position",
    "hloc_momentum_quality",
)

# V4 + MTF combined (32+ traditional + 17 MTF = 49 factors)
V4_PLUS_MTF_FACTORS = V4_24_FACTORS + MTF_MOMENTUM_FACTORS

# Ornstein-Uhlenbeck (OU) Mean-Reversion Factors (8 factors)
OU_FACTORS = (
    "ou_zscore_short",
    "ou_zscore_med",
    "ou_halflife_signal",
    "ou_reversion_strength",
    "ou_predicted_return",
    "ou_regime_indicator",
    "ou_equilibrium_dist",
    "ou_momentum_blend",
)

# GLFT Market Microstructure Factors (10 factors)
GLFT_FACTORS = (
    "glft_ofi_short",
    "glft_ofi_med",
    "glft_ofi_momentum",
    "glft_flow_toxicity",
    "glft_toxic_momentum",
    "glft_impact_asymmetry",
    "glft_inventory_signal",
    "glft_inventory_risk_prem",
    "glft_spread_adjusted_mom",
    "glft_mm_edge",
)

# All available factors
ALL_FACTORS = tuple(FACTOR_REGISTRY.keys())


# ==============================================================================
# Helper functions for dashboard integration
# ==============================================================================

def get_factor_info(factor_name: str) -> Optional[FactorDefinition]:
    """Get metadata for a factor."""
    return FACTOR_REGISTRY.get(factor_name)


def get_factors_by_category(category: FactorCategory) -> List[str]:
    """Get all factors in a category."""
    return [
        name for name, defn in FACTOR_REGISTRY.items()
        if defn.category == category
    ]


def get_all_categories() -> List[FactorCategory]:
    """Get all available factor categories."""
    return list(FactorCategory)


def get_dashboard_factor_config() -> Dict[str, Any]:
    """
    Get complete factor configuration for dashboard display.
    
    Returns structure suitable for Streamlit UI generation:
    {
        "categories": [...],
        "factors": {...},
        "window_params": {...},
        "factor_sets": {...}
    }
    """
    return {
        "categories": [
            {"id": cat.value, "name": cat.value.replace("_", " ").title()}
            for cat in FactorCategory
        ],
        "factors": {
            name: defn.to_dict()
            for name, defn in FACTOR_REGISTRY.items()
        },
        "window_params": FactorWindowConfig.get_param_info(),
        "factor_sets": {
            "v4_24": list(V4_24_FACTORS),
            "defensive": list(DEFENSIVE_FACTORS),
            "alpha": list(ALPHA_FACTORS),
            "quality": list(QUALITY_FACTORS),
            "mtf_momentum": list(MTF_MOMENTUM_FACTORS),
            "v4_plus_mtf": list(V4_PLUS_MTF_FACTORS),
            "ou_mean_reversion": list(OU_FACTORS),
            "glft_microstructure": list(GLFT_FACTORS),
            "all": list(ALL_FACTORS),
        },
    }


# ==============================================================================
# Helper functions (pure xarray) - Internal use
# ==============================================================================

def _clip_ret(r: "xr.DataArray") -> "xr.DataArray":
    return r.fillna(0.0).clip(min=-0.999999)


def _log1p_safe(x: "xr.DataArray") -> "xr.DataArray":
    x = x.fillna(0.0).clip(min=-0.999999)
    return xr.apply_ufunc(np.log1p, x)


def _sma(x: "xr.DataArray", win: int) -> "xr.DataArray":
    return x.rolling(time=win, min_periods=win).mean()


def _std(x: "xr.DataArray", win: int) -> "xr.DataArray":
    return x.rolling(time=win, min_periods=win).std()


def _ema(x: "xr.DataArray", span: int) -> "xr.DataArray":
    df = x.transpose("time", "asset").to_pandas()
    out = df.ewm(span=span, adjust=False, min_periods=span).mean()
    return xr.DataArray(
        out.values, 
        coords={"time": x.time, "asset": x.asset}, 
        dims=["time", "asset"]
    ).fillna(0.0)


def _rolling_max(x: "xr.DataArray", win: int) -> "xr.DataArray":
    return x.rolling(time=win, min_periods=win).max()


def _rolling_min(x: "xr.DataArray", win: int) -> "xr.DataArray":
    return x.rolling(time=win, min_periods=win).min()


def _atr(
    high: "xr.DataArray", 
    low: "xr.DataArray", 
    close: "xr.DataArray", 
    win: int
) -> "xr.DataArray":
    prev = close.shift(time=1)
    hl = high - low
    hc = np.abs(high - prev)
    lc = np.abs(low - prev)
    tr = xr.where(hl > hc, hl, hc)
    tr = xr.where(tr > lc, tr, lc)
    return _sma(tr.fillna(0.0), win).fillna(0.0)


def _market_return(r: "xr.DataArray") -> "xr.DataArray":
    return r.fillna(0.0).mean("asset")


def _rolling_beta(
    returns: "xr.DataArray", 
    mkt: "xr.DataArray", 
    win: int, 
    eps: float = 1e-12
) -> "xr.DataArray":
    r, m = xr.align(returns, mkt, join="inner")
    cov = _sma(r * m, win) - _sma(r, win) * _sma(m, win)
    var = _sma(m * m, win) - _sma(m, win) ** 2
    return (cov / (var + eps)).fillna(0.0)


def _residual_returns(
    returns: "xr.DataArray", 
    beta: "xr.DataArray", 
    mkt: "xr.DataArray"
) -> "xr.DataArray":
    r, b, m = xr.align(returns, beta, mkt, join="inner")
    return (r - b * m).fillna(0.0)


def _robust_zscore(
    x: "xr.DataArray", 
    dim: str = "asset", 
    eps: float = 1e-12, 
    clip: float = 5.0
) -> "xr.DataArray":
    med = x.median(dim=dim, skipna=True)
    mad = np.abs(x - med).median(dim=dim, skipna=True)
    z = (x - med) / (mad * 1.4826 + eps)
    return z.fillna(0.0).clip(-clip, clip)


# ==============================================================================
# FactorLibrary - Main computation class
# ==============================================================================

class FactorLibrary:
    """
    Factor computation library with PM-configurable windows.
    
    Usage:
        # Default computation
        lib = FactorLibrary(market_data)
        F, artifacts = lib.compute()
        
        # Custom windows
        params = FactorParams(MOM_WIN=63, BETA_WIN=126)
        lib = FactorLibrary(market_data, params=params)
        
        # Scaled windows (e.g., volatile regime)
        params = FactorParams().with_scaled_windows(0.7)
        lib = FactorLibrary(market_data, params=params)
    """
    
    def __init__(
        self,
        market_data: Union["xr.Dataset", Any],
        *,
        params: Optional[FactorParams] = None,
        factors: Sequence[str] = V4_24_FACTORS,
        asset_dim: str = "asset",
    ):
        _require_xr()
        self.params = params or FactorParams()
        self.factors = list(factors)
        self.asset_dim = asset_dim

        if isinstance(market_data, xr.Dataset):
            self.data = market_data
        else:
            if hasattr(market_data, "data") and isinstance(getattr(market_data, "data"), xr.Dataset):
                self.data = getattr(market_data, "data")
            else:
                raise TypeError("market_data must be xr.Dataset or bundle-like with .data")

        # Validate required variables
        for v in ("close", "high", "low", "vol"):
            if v not in self.data:
                raise ValueError(f"Dataset missing required variable '{v}'")

        self.data = self.data.transpose("time", asset_dim, missing_dims="ignore")
        self._artifacts: Dict[str, "xr.DataArray"] = {}

        # Precompute common series
        self.close = self.data["close"].fillna(0.0)
        self.high = self.data["high"].fillna(0.0)
        self.low = self.data["low"].fillna(0.0)
        self.vol = self.data["vol"].fillna(0.0)

        self.ret = _clip_ret(self.close / self.close.shift(time=1) - 1.0)
        self.mkt = _market_return(self.ret)

        self.beta = _rolling_beta(self.ret, self.mkt, self.params.BETA_WIN, eps=self.params.eps)
        self.resid = _residual_returns(self.ret, self.beta, self.mkt)

        self._artifacts["beta"] = self.beta
        self._artifacts["resid"] = self.resid

    def _cs_z(self, da: "xr.DataArray") -> "xr.DataArray":
        """Cross-sectional z-score normalization."""
        if self.params.robust_cs_z:
            return _robust_zscore(da, dim=self.asset_dim, eps=self.params.eps, clip=self.params.zclip)
        return safe_zscore(da, dim=self.asset_dim, eps=self.params.eps, clip=self.params.zclip)

    def compute(self) -> Tuple["xr.DataArray", Dict[str, "xr.DataArray"]]:
        """
        Compute all requested factors.
        
        Returns:
            F: DataArray with dims (factor, time, asset)
            artifacts: Dict of intermediate computations (beta, resid, etc.)
        """
        Fs: List["xr.DataArray"] = []
        names: List[str] = []

        for name in self.factors:
            fn = getattr(self, f"_factor_{name}", None)
            if fn is None:
                raise ValueError(f"Unknown factor '{name}'. Implement _factor_{name}.")
            raw = fn().transpose("time", self.asset_dim).fillna(0.0)
            z = self._cs_z(raw).fillna(0.0)
            z.name = name
            Fs.append(z)
            names.append(name)

        F = xr.concat(Fs, dim="factor", join="outer").assign_coords(factor=names).transpose("factor", "time", self.asset_dim)
        F.name = "F"
        return F, dict(self._artifacts)

    def compute_single(self, factor_name: str) -> "xr.DataArray":
        """Compute a single factor (useful for debugging/display)."""
        fn = getattr(self, f"_factor_{factor_name}", None)
        if fn is None:
            raise ValueError(f"Unknown factor '{factor_name}'")
        raw = fn().transpose("time", self.asset_dim).fillna(0.0)
        return self._cs_z(raw).fillna(0.0)

    def get_factor_metadata(self, factor_name: str) -> Optional[FactorDefinition]:
        """Get metadata for a computed factor."""
        return get_factor_info(factor_name)

    # ==========================================================================
    # DEFENSIVE / LOW RISK FACTORS
    # ==========================================================================

    def _factor_inv_vol(self) -> "xr.DataArray":
        """Inverse ATR volatility."""
        atr = _atr(self.high, self.low, self.close, self.params.ATR_WIN)
        return (1.0 / (atr + self.params.eps)).fillna(0.0)

    def _factor_inv_idio(self) -> "xr.DataArray":
        """Inverse idiosyncratic volatility."""
        idio_vol = _std(self.resid, self.params.IDIO_WIN).fillna(0.0)
        return (1.0 / (idio_vol + self.params.eps)).fillna(0.0)

    def _factor_inv_down(self) -> "xr.DataArray":
        """Inverse downside volatility."""
        down = self.ret.where(self.ret < 0.0, 0.0)
        down_vol = _std(down, self.params.DOWN_WIN).fillna(0.0)
        return (1.0 / (down_vol + self.params.eps)).fillna(0.0)

    def _factor_low_corr(self) -> "xr.DataArray":
        """Negative market correlation (low corr is good)."""
        r_df = self.ret.transpose("time", "asset").to_pandas()
        m = self.mkt.to_pandas()
        corr = pd.DataFrame(index=r_df.index, columns=r_df.columns, dtype=float)
        for c in r_df.columns:
            corr[c] = r_df[c].rolling(self.params.CORR_WIN).corr(m)
        da = xr.DataArray(corr.fillna(0.0).values, coords=self.ret.coords, dims=self.ret.dims)
        return (-da).fillna(0.0)

    def _factor_low_beta(self) -> "xr.DataArray":
        """Negative beta (low beta is good)."""
        return (-self.beta).fillna(0.0)

    # ==========================================================================
    # LIQUIDITY FACTORS
    # ==========================================================================

    def _factor_liquidity(self) -> "xr.DataArray":
        """Average dollar volume relative to universe mean."""
        adv = _sma(self.vol * self.close, self.params.ADV_WIN_LONG).fillna(0.0)
        mu = adv.mean("asset")
        return (adv / (mu + self.params.eps)).fillna(0.0)

    def _factor_amihud_inv(self) -> "xr.DataArray":
        """Inverse Amihud illiquidity."""
        dv = (self.vol * self.close).fillna(0.0)
        amihud = (np.abs(self.ret) / (dv + self.params.eps)).fillna(0.0)
        return (1.0 / (amihud + self.params.eps)).fillna(0.0)

    # ==========================================================================
    # MOMENTUM FACTORS
    # ==========================================================================

    def _factor_resid_mom(self) -> "xr.DataArray":
        """Residual momentum over MOM_WIN."""
        return _sma(self.resid, self.params.MOM_WIN).fillna(0.0)

    def _factor_resid_mom_short(self) -> "xr.DataArray":
        """Short-term residual momentum."""
        return _sma(self.resid, self.params.MOM_SHORT).fillna(0.0)

    def _factor_resid_mom_long(self) -> "xr.DataArray":
        """Long-term residual momentum."""
        return _sma(self.resid, self.params.MOM_LONG).fillna(0.0)

    def _factor_resid_mom_mix(self) -> "xr.DataArray":
        """Blended momentum: 50% medium + 30% long + 20% short."""
        mom_short = _sma(self.resid, self.params.MOM_SHORT)
        mom_med = _sma(self.resid, self.params.MOM_WIN)
        mom_long = _sma(self.resid, self.params.MOM_LONG)
        return (0.50 * mom_med + 0.30 * mom_long + 0.20 * mom_short).fillna(0.0)

    def _factor_cross_momentum(self) -> "xr.DataArray":
        """Cross-sectional momentum (demeaned)."""
        cross_mom = _sma(self.ret, self.params.MOM_WIN)
        cross_mom = cross_mom - cross_mom.mean("asset")
        return cross_mom.fillna(0.0)

    # ==========================================================================
    # REVERSAL FACTORS
    # ==========================================================================

    def _factor_srev(self) -> "xr.DataArray":
        """Short-term reversal."""
        return (-(self.close / self.close.shift(time=self.params.REV_WIN) - 1.0)).fillna(0.0)

    def _factor_lrev(self) -> "xr.DataArray":
        """Long-term reversal."""
        return (-(self.close / self.close.shift(time=self.params.REV_LONG) - 1.0)).fillna(0.0)

    # ==========================================================================
    # TECHNICAL FACTORS
    # ==========================================================================

    def _factor_breakout(self) -> "xr.DataArray":
        """Donchian channel breakout position (0-1)."""
        mx = _rolling_max(self.close, self.params.DON_WIN)
        mn = _rolling_min(self.close, self.params.DON_WIN)
        return ((self.close - mn) / (mx - mn + self.params.eps)).fillna(0.0)

    def _factor_slope(self) -> "xr.DataArray":
        """EMA trend slope normalized by ATR."""
        atr = _atr(self.high, self.low, self.close, self.params.ATR_WIN)
        ema_f = _ema(self.close, self.params.EMA_FAST)
        ema_s = _ema(self.close, self.params.EMA_SLOW)
        return ((ema_f - ema_s) / (atr + self.params.eps)).fillna(0.0)

    def _factor_calm_flow(self) -> "xr.DataArray":
        """Volume normalization - favor declining volume."""
        vs = _sma(self.vol, self.params.ADV_WIN_SHORT)
        vl = _sma(self.vol, self.params.ADV_WIN_LONG)
        return (-(vs / (vl + self.params.eps) - 1.0)).fillna(0.0)

    def _factor_prox_52w_high(self) -> "xr.DataArray":
        """Proximity to 52-week high (inverted)."""
        roll_max = _rolling_max(self.close, 252)
        prox = (self.close / (roll_max + self.params.eps) - 1.0)
        return (-prox).fillna(0.0)

    def _factor_vol_surprise(self) -> "xr.DataArray":
        """Volume surprise vs 20d average."""
        dv = (self.vol * self.close).fillna(0.0)
        base = _sma(dv, 20)
        return (dv / (base + self.params.eps) - 1.0).fillna(0.0)

    def _factor_vol_breakout(self) -> "xr.DataArray":
        """ATR expansion with positive momentum confirmation."""
        atr = _atr(self.high, self.low, self.close, self.params.ATR_WIN)
        atr_exp = (atr / (_sma(atr, 20) + self.params.eps) - 1.0).fillna(0.0)
        mom_med = _sma(self.resid, self.params.MOM_WIN)
        return (atr_exp * (mom_med > 0).astype(float)).fillna(0.0)

    def _factor_ma_cloud(self) -> "xr.DataArray":
        """Moving average cloud: distance from MA50/MA200."""
        ma50 = _sma(self.close, 50)
        ma200 = _sma(self.close, 200)
        atr20 = _atr(self.high, self.low, self.close, 20)
        z50 = (self.close - ma50) / (atr20 + self.params.eps)
        z200 = (self.close - ma200) / (atr20 + self.params.eps)
        return (0.6 * z50 + 0.4 * z200).fillna(0.0)

    # ==========================================================================
    # MICROSTRUCTURE / QUALITY FACTORS
    # ==========================================================================

    def _factor_idio_tail_risk(self) -> "xr.DataArray":
        """Idiosyncratic tail risk (5th percentile / vol)."""
        resid = self.resid.fillna(0.0)
        sigma20 = _std(resid, 20).fillna(0.0)
        q05 = resid.rolling(time=20, min_periods=10).reduce(np.nanquantile, q=0.05)
        return (-(q05 / (sigma20 + self.params.eps))).fillna(0.0)

    def _factor_idio_jump_freq(self) -> "xr.DataArray":
        """Frequency of large idiosyncratic moves (>2.5σ)."""
        resid = self.resid.fillna(0.0)
        sigma63 = _std(resid, self.params.IDIO_WIN).fillna(0.0)
        thr = 2.5 * (sigma63 + self.params.eps)
        jumps = (np.abs(resid) > thr).astype(float)
        return (-(jumps.rolling(time=20, min_periods=20).mean())).fillna(0.0)

    def _factor_beta_stability(self) -> "xr.DataArray":
        """Beta stability over rolling 21d window."""
        beta21 = _sma(self.beta.fillna(0.0), 21).fillna(0.0)
        return (-np.abs(beta21 - beta21.shift(time=21))).fillna(0.0)

    def _factor_micro_noise(self) -> "xr.DataArray":
        """Microstructure noise ratio (close-close vs Parkinson)."""
        cc = _clip_ret(self.close / self.close.shift(time=1) - 1.0)
        var_cc = _sma(cc * cc, 20)

        hl = (self.high / (self.low + self.params.eps)).clip(min=1.0)
        ln_hl2 = _sma((xr.apply_ufunc(np.log, hl) ** 2), 20)
        var_pk = (1.0 / (4.0 * np.log(2.0))) * ln_hl2
        return (-(var_cc / (var_pk + self.params.eps))).fillna(0.0)

    def _factor_quality_score(self) -> "xr.DataArray":
        """Composite quality: low vol + high liquidity + stable returns."""
        atr = _atr(self.high, self.low, self.close, self.params.ATR_WIN)
        adv = _sma(self.vol * self.close, self.params.ADV_WIN_LONG)
        idio_vol = _std(self.resid, self.params.IDIO_WIN).fillna(0.0)
        
        quality = (
            (1.0 / (atr + self.params.eps)) * 0.4 +
            (adv / (adv.mean("asset") + self.params.eps)) * 0.3 +
            (1.0 / (idio_vol + self.params.eps)) * 0.3
        )
        return quality.fillna(0.0)

    # ==========================================================================
    # VALUE FACTORS
    # ==========================================================================

    def _factor_value_score(self) -> "xr.DataArray":
        """Value proxy: price relative to long-term MA."""
        ma_long = _sma(self.close, self.params.VALUE_WIN)
        return (-(self.close / ma_long - 1.0)).fillna(0.0)

    # ==========================================================================
    # INTERACTION FACTORS
    # ==========================================================================

    def _factor_value_mom(self) -> "xr.DataArray":
        """Value-momentum interaction: resid_mom × srev."""
        return (self._factor_resid_mom() * self._factor_srev()).fillna(0.0)

    def _factor_quality_defensive(self) -> "xr.DataArray":
        """Quality-defensive interaction: liquidity × inv_vol."""
        return (self._factor_liquidity() * self._factor_inv_vol()).fillna(0.0)

    # ==========================================================================
    # CROSS-SECTIONAL FACTORS
    # ==========================================================================

    def _factor_rel_sector_mom(self) -> "xr.DataArray":
        """Sector-relative momentum (vs universe mean)."""
        if self.ret.sizes.get("time", 0) < self.params.SECTOR_MOM_WIN + 2:
            return xr.zeros_like(self.ret)

        asset_mom = _sma(self.ret, self.params.SECTOR_MOM_WIN)
        proxy = asset_mom.mean("asset")
        return (asset_mom - proxy.broadcast_like(asset_mom)).fillna(0.0)

    def _factor_cross_volatility(self) -> "xr.DataArray":
        """Cross-sectional volatility rank (demeaned)."""
        cross_vol = _std(self.ret, self.params.IDIO_WIN)
        cross_vol = cross_vol - cross_vol.mean("asset")
        return cross_vol.fillna(0.0)

    def _factor_cross_liquidity(self) -> "xr.DataArray":
        """Cross-sectional liquidity rank (demeaned)."""
        adv = _sma(self.vol * self.close, self.params.ADV_WIN_LONG)
        cross_liq = adv - adv.mean("asset")
        return cross_liq.fillna(0.0)

    # ==========================================================================
    # MULTI-TIMEFRAME (MTF) MOMENTUM FACTORS
    # Weekly/Monthly/Quarterly/Yearly with IC-adaptive weighting
    # ==========================================================================

    def _compute_period_return(self, lookback: int) -> "xr.DataArray":
        """Helper: compute return over lookback period."""
        return (self.close / self.close.shift(time=lookback) - 1.0).fillna(0.0)

    # --- To-Date Returns ---

    def _factor_ret_wtd(self) -> "xr.DataArray":
        """Week-to-date return."""
        return self._compute_period_return(self.params.WEEK)

    def _factor_ret_mtd(self) -> "xr.DataArray":
        """Month-to-date return."""
        return self._compute_period_return(self.params.MONTH)

    def _factor_ret_qtd(self) -> "xr.DataArray":
        """Quarter-to-date return."""
        return self._compute_period_return(self.params.QUARTER)

    def _factor_ret_ytd(self) -> "xr.DataArray":
        """Year-to-date return."""
        return self._compute_period_return(self.params.YEAR)

    # --- Completed Period Returns ---

    def _factor_ret_1w(self) -> "xr.DataArray":
        """Last complete week return (5 trading days)."""
        return self._compute_period_return(self.params.WEEK)

    def _factor_ret_1m(self) -> "xr.DataArray":
        """Last 21 trading days return (~1 month)."""
        return self._compute_period_return(self.params.MONTH)

    def _factor_ret_1q(self) -> "xr.DataArray":
        """Last 63 trading days return (~1 quarter)."""
        return self._compute_period_return(self.params.QUARTER)

    def _factor_ret_1y(self) -> "xr.DataArray":
        """Last 252 trading days return (~1 year)."""
        return self._compute_period_return(self.params.YEAR)

    def _factor_ret_2_12m(self) -> "xr.DataArray":
        """Months 2-12 return (classic momentum, skip recent month)."""
        p = self.params
        ret_12m = self.close / self.close.shift(time=p.YEAR) - 1.0
        ret_1m = self.close / self.close.shift(time=p.MONTH) - 1.0
        return (ret_12m - ret_1m).fillna(0.0)

    # --- Momentum Alignment / Confluence ---

    def _factor_mtf_alignment(self) -> "xr.DataArray":
        """
        Momentum direction alignment across W/M/Q/Y.
        Score: count of timeframes with positive returns (0-4).
        """
        ret_w = self._factor_ret_1w()
        ret_m = self._factor_ret_1m()
        ret_q = self._factor_ret_1q()
        ret_y = self._factor_ret_1y()

        w_pos = (ret_w > 0).astype(float)
        m_pos = (ret_m > 0).astype(float)
        q_pos = (ret_q > 0).astype(float)
        y_pos = (ret_y > 0).astype(float)

        return (w_pos + m_pos + q_pos + y_pos).fillna(0.0)

    def _factor_mtf_alignment_strength(self) -> "xr.DataArray":
        """Alignment weighted by return magnitude."""
        p = self.params
        ret_w = self._factor_ret_1w()
        ret_m = self._factor_ret_1m()
        ret_q = self._factor_ret_1q()

        # Normalize each return by its typical magnitude
        # Note: xarray rolling has max window of 250, so cap YEAR at 250
        w_z = ret_w / (_std(ret_w, p.QUARTER) + p.eps)
        m_z = ret_m / (_std(ret_m, p.QUARTER) + p.eps)
        q_z = ret_q / (_std(ret_q, min(p.YEAR, 250)) + p.eps)

        return (w_z + m_z + q_z).fillna(0.0).clip(-10, 10)

    def _factor_trend_consistency(self) -> "xr.DataArray":
        """Consistency of trend across timeframes (inverted std)."""
        p = self.params
        ret_w = self._factor_ret_1w()
        ret_m = self._factor_ret_1m()
        ret_q = self._factor_ret_1q()

        # Stack and compute std
        ret_stack = xr.concat([ret_w, ret_m, ret_q], dim="tf")
        tf_std = ret_stack.std("tf")

        # Invert: low std = consistent = high score
        return (1.0 / (tf_std + p.eps)).fillna(0.0)

    # --- IC-Weighted Adaptive Momentum ---

    def _factor_mtf_ic_momentum(self) -> "xr.DataArray":
        """
        IC-weighted momentum composite.
        Dynamically weights W/M/Q returns by their rolling autocorrelation (IC proxy).
        """
        p = self.params
        ret_w = self._factor_ret_1w()
        ret_m = self._factor_ret_1m()
        ret_q = self._factor_ret_1q()

        # Convert to pandas for rolling correlation
        ret_w_df = ret_w.transpose("time", "asset").to_pandas()
        ret_m_df = ret_m.transpose("time", "asset").to_pandas()
        ret_q_df = ret_q.transpose("time", "asset").to_pandas()

        # IC proxy: autocorrelation (lagged self-correlation)
        w_ic = ret_w_df.shift(p.WEEK).rolling(p.IC_WINDOW).corr(ret_w_df).fillna(0.0).clip(lower=0)
        m_ic = ret_m_df.shift(p.MONTH).rolling(p.IC_WINDOW).corr(ret_m_df).fillna(0.0).clip(lower=0)
        q_ic = ret_q_df.shift(p.QUARTER).rolling(p.IC_WINDOW).corr(ret_q_df).fillna(0.0).clip(lower=0)

        # Normalize weights
        total_ic = w_ic + m_ic + q_ic + p.eps
        w_w = w_ic / total_ic
        w_m = m_ic / total_ic
        w_q = q_ic / total_ic

        # Weighted combination
        result = w_w * ret_w_df + w_m * ret_m_df + w_q * ret_q_df
        return xr.DataArray(
            result.fillna(0.0).values,
            coords=ret_w.coords,
            dims=ret_w.dims
        ).fillna(0.0)

    def _factor_mtf_ic_regime(self) -> "xr.DataArray":
        """Dominant timeframe regime (1=W, 2=M, 3=Q)."""
        p = self.params
        ret_w = self._factor_ret_1w()
        ret_m = self._factor_ret_1m()
        ret_q = self._factor_ret_1q()

        ret_w_df = ret_w.transpose("time", "asset").to_pandas()
        ret_m_df = ret_m.transpose("time", "asset").to_pandas()
        ret_q_df = ret_q.transpose("time", "asset").to_pandas()

        w_ic = np.abs(ret_w_df.shift(p.WEEK).rolling(p.IC_WINDOW).corr(ret_w_df).fillna(0.0))
        m_ic = np.abs(ret_m_df.shift(p.MONTH).rolling(p.IC_WINDOW).corr(ret_m_df).fillna(0.0))
        q_ic = np.abs(ret_q_df.shift(p.QUARTER).rolling(p.IC_WINDOW).corr(ret_q_df).fillna(0.0))

        # Find dominant
        regime = pd.DataFrame(1.0, index=ret_w_df.index, columns=ret_w_df.columns)
        regime = regime.where(w_ic >= m_ic, 0.0) + regime
        regime = regime.where((m_ic > w_ic) & (m_ic >= q_ic), 2.0).where(
            ~((m_ic > w_ic) & (m_ic >= q_ic)), regime
        )
        regime = regime.where((q_ic > w_ic) & (q_ic > m_ic), 3.0).where(
            ~((q_ic > w_ic) & (q_ic > m_ic)), regime
        )

        return xr.DataArray(
            regime.fillna(1.0).values,
            coords=ret_w.coords,
            dims=ret_w.dims
        ).fillna(1.0)

    # --- Acceleration / Rate of Change ---

    def _factor_momentum_acceleration(self) -> "xr.DataArray":
        """Short-term vs long-term momentum (acceleration signal)."""
        p = self.params
        ret_w = self._factor_ret_1w()
        ret_m = self._factor_ret_1m()

        w_vol = _std(ret_w, p.QUARTER) + p.eps
        m_vol = _std(ret_m, p.QUARTER) + p.eps

        w_z = ret_w / w_vol
        m_z = ret_m / m_vol

        return (w_z - m_z).fillna(0.0).clip(-5, 5)

    def _factor_mtf_slope(self) -> "xr.DataArray":
        """Slope of returns across timeframes."""
        p = self.params
        ret_w = self._factor_ret_1w()
        ret_m = self._factor_ret_1m()
        ret_q = self._factor_ret_1q()

        # Normalize returns
        w_norm = ret_w / (np.abs(ret_w).rolling(time=63, min_periods=10).mean() + p.eps)
        m_norm = ret_m / (np.abs(ret_m).rolling(time=63, min_periods=10).mean() + p.eps)
        q_norm = ret_q / (np.abs(ret_q).rolling(time=63, min_periods=10).mean() + p.eps)

        # X values: log(days)
        x = np.array([np.log(p.WEEK), np.log(p.MONTH), np.log(p.QUARTER)])
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()

        # Stack and compute mean
        y_stack = xr.concat([w_norm, m_norm, q_norm], dim="tf")
        y_mean = y_stack.mean("tf")

        # Slope calculation
        cov = (
            (x[0] - x_mean) * (w_norm - y_mean) +
            (x[1] - x_mean) * (m_norm - y_mean) +
            (x[2] - x_mean) * (q_norm - y_mean)
        )

        return (cov / (x_var + p.eps)).fillna(0.0).clip(-5, 5)

    # --- HLOC-Enhanced Factors ---

    def _factor_hloc_range_expansion(self) -> "xr.DataArray":
        """Weekly range vs monthly range (breakout signal)."""
        p = self.params

        # Weekly range
        w_high = _rolling_max(self.high, p.WEEK)
        w_low = _rolling_min(self.low, p.WEEK)
        w_range = w_high - w_low

        # Monthly range
        m_high = _rolling_max(self.high, p.MONTH)
        m_low = _rolling_min(self.low, p.MONTH)
        m_range = m_high - m_low

        return (w_range / (m_range + p.eps)).fillna(0.0).clip(0, 3)

    def _factor_hloc_close_position(self) -> "xr.DataArray":
        """Close position within monthly HLOC range (0=low, 1=high)."""
        p = self.params

        m_high = _rolling_max(self.high, p.MONTH)
        m_low = _rolling_min(self.low, p.MONTH)

        return ((self.close - m_low) / (m_high - m_low + p.eps)).fillna(0.5).clip(0, 1)

    def _factor_hloc_momentum_quality(self) -> "xr.DataArray":
        """Momentum quality: returns supported by HLOC confirmation."""
        p = self.params
        ret = self._factor_ret_1m()
        close_pos = self._factor_hloc_close_position()

        return (ret * (close_pos - 0.5) * 2).fillna(0.0)

    # --- Relative MTF Factors ---

    def _factor_relative_mtf_momentum(self) -> "xr.DataArray":
        """MTF momentum relative to market."""
        p = self.params
        asset_mom = self._factor_mtf_ic_momentum()
        mkt_mom = _sma(self.mkt, p.MONTH)
        return (asset_mom - mkt_mom.broadcast_like(asset_mom)).fillna(0.0)

    def _factor_sector_relative_mtf(self) -> "xr.DataArray":
        """MTF momentum vs universe mean."""
        mom = self._factor_mtf_ic_momentum()
        universe_mean = mom.mean("asset")
        return (mom - universe_mean.broadcast_like(mom)).fillna(0.0)

    # ==========================================================================
    # Neural Alpha Novel Factors (ported into base library)
    # ==========================================================================

    def _factor_attention_momentum(self) -> "xr.DataArray":
        """Attention-weighted momentum: volume spikes + return direction/magnitude."""
        p = self.params

        vol_avg = _sma(self.vol, p.ADV_WIN_LONG)
        vol_ratio = self.vol / (vol_avg + p.eps)
        vol_ratio = vol_ratio.fillna(1.0).clip(0.1, 10.0)

        ret_std = _std(self.ret, p.MOM_SHORT)
        ret_z = (self.ret / (ret_std + p.eps)).fillna(0.0).clip(-3, 3)

        attention = np.sign(self.ret) * (vol_ratio - 1.0) * np.abs(ret_z)

        attention_win = getattr(p, "ATTENTION_WIN", 10)
        return _sma(attention, int(attention_win)).fillna(0.0)

    def _factor_disposition_alpha(self) -> "xr.DataArray":
        """Disposition effect proxy: prefer paper-loss names that are recovering."""
        p = self.params

        high_52w = _rolling_max(self.high, 252)
        low_52w = _rolling_min(self.low, 252)
        anchor_price = (high_52w + low_52w) / 2.0

        unrealized_pnl = (self.close - anchor_price) / (anchor_price + p.eps)
        disposition = -unrealized_pnl

        mom_short = _sma(self.ret, p.MOM_SHORT)
        recovery = (mom_short > 0).astype(float)

        disp_win = getattr(p, "DISPOSITION_WIN", 63)
        return _sma(disposition * (0.5 + 0.5 * recovery), int(disp_win)).fillna(0.0)

    def _factor_anchoring_bias(self) -> "xr.DataArray":
        """Anchoring proxy: proximity to 52w high + position in 52w range."""
        p = self.params

        high_52w = _rolling_max(self.high, 252)
        low_52w = _rolling_min(self.low, 252)

        high_prox = (self.close / (high_52w + p.eps)).fillna(0.5).clip(0, 1)
        low_prox = ((self.close - low_52w) / (high_52w - low_52w + p.eps)).fillna(0.5).clip(0, 1)

        return (0.6 * high_prox + 0.4 * low_prox).fillna(0.5)

    def _factor_efficiency_ratio(self) -> "xr.DataArray":
        """Kaufman Efficiency Ratio with momentum direction."""
        p = self.params
        eff_win = int(getattr(p, "EFFICIENCY_WIN", 21))

        direction = np.abs(self.close - self.close.shift(time=eff_win))
        abs_changes = np.abs(self.close - self.close.shift(time=1))
        volatility = abs_changes.rolling(time=eff_win, min_periods=eff_win).sum()

        er = direction / (volatility + p.eps)
        mom_dir = np.sign(self.close - self.close.shift(time=eff_win))
        return (er * mom_dir).fillna(0.0).clip(-1, 1)

    def _factor_information_flow(self) -> "xr.DataArray":
        """Volume-weighted price impact asymmetry."""
        p = self.params

        is_up = (self.ret > 0).astype(float)
        is_down = (self.ret < 0).astype(float)

        dollar_vol = self.vol * self.close
        up_impact = _sma(dollar_vol * is_up * np.abs(self.ret), p.ADV_WIN_LONG)
        down_impact = _sma(dollar_vol * is_down * np.abs(self.ret), p.ADV_WIN_LONG)
        total = up_impact + down_impact + p.eps

        info_flow = (up_impact - down_impact) / total
        return info_flow.fillna(0.0).clip(-1, 1)

    def _factor_mean_reversion_speed(self) -> "xr.DataArray":
        """Mean reversion speed proxy: negative autocorr of trend deviations."""
        p = self.params

        trend = _ema(self.close, p.EMA_SLOW)
        dev = ((self.close - trend) / (trend + p.eps)).fillna(0.0)

        dev_df = dev.transpose("time", "asset").to_pandas()
        dev_lag = dev_df.shift(1)
        autocorr = dev_df.rolling(63, min_periods=21).corr(dev_lag).fillna(0.0)

        mr_speed = xr.DataArray(-autocorr.values, coords=dev.coords, dims=dev.dims)
        return mr_speed.fillna(0.0).clip(-1, 1)

    def _factor_momentum_quality_ratio(self) -> "xr.DataArray":
        """Rolling Sharpe-like momentum quality ratio (annualized)."""
        p = self.params
        ret_mean = _sma(self.ret, p.MOM_WIN)
        ret_std = _std(self.ret, p.MOM_WIN)
        mqr = ret_mean / (ret_std + p.eps)
        return (mqr * np.sqrt(252)).fillna(0.0).clip(-5, 5)

    def _factor_momentum_persistence(self) -> "xr.DataArray":
        """Autocorrelation of returns (momentum continuation)."""
        p = self.params
        win = int(getattr(p, "PERSISTENCE_WIN", 42))

        ret_df = self.ret.transpose("time", "asset").to_pandas()
        ret_lag = ret_df.shift(1)
        pers = ret_df.rolling(win, min_periods=21).corr(ret_lag).fillna(0.0)
        pers_da = xr.DataArray(pers.values, coords=self.ret.coords, dims=self.ret.dims)
        return pers_da.fillna(0.0).clip(-1, 1)

    def _factor_momentum_divergence(self) -> "xr.DataArray":
        """Residual momentum minus price momentum (normalized)."""
        p = self.params

        price_mom = _sma(self.ret, p.MOM_WIN)
        resid_mom = _sma(self.resid, p.MOM_WIN)
        div = resid_mom - price_mom
        div_std = _std(div, p.IDIO_WIN)
        return (div / (div_std + p.eps)).fillna(0.0).clip(-3, 3)

    def _factor_vol_of_vol(self) -> "xr.DataArray":
        """Inverse vol-of-vol (stability premium)."""
        p = self.params
        vov_win = int(getattr(p, "VOV_WIN", 21))
        vol = _std(self.ret, vov_win)
        vov = _std(vol, p.IDIO_WIN)
        return (1.0 / (vov + p.eps)).fillna(0.0)

    def _factor_skewness_factor(self) -> "xr.DataArray":
        """Rolling return skewness."""
        p = self.params
        win = int(getattr(p, "SKEW_WIN", 63))
        ret_df = self.ret.transpose("time", "asset").to_pandas()
        skew = ret_df.rolling(win, min_periods=21).skew().fillna(0.0)
        skew_da = xr.DataArray(skew.values, coords=self.ret.coords, dims=self.ret.dims)
        return skew_da.fillna(0.0).clip(-3, 3)

    def _factor_kurtosis_factor(self) -> "xr.DataArray":
        """Negative rolling return kurtosis (fat-tail avoidance)."""
        p = self.params
        win = int(getattr(p, "KURT_WIN", 63))
        ret_df = self.ret.transpose("time", "asset").to_pandas()
        kurt = ret_df.rolling(win, min_periods=21).kurt().fillna(0.0)
        kurt_da = xr.DataArray(kurt.values, coords=self.ret.coords, dims=self.ret.dims)
        return (-kurt_da.clip(-10, 20)).fillna(0.0)

    def _factor_regime_momentum(self) -> "xr.DataArray":
        """Volatility-regime-adaptive momentum blend (short vs long)."""
        p = self.params

        vol_short = _std(self.ret, p.MOM_SHORT)
        vol_long = _std(self.ret, p.IDIO_WIN)
        vol_ratio = vol_short / (vol_long + p.eps)

        w = (1.0 / (1.0 + np.exp(-2.0 * (vol_ratio - 1.0)))).clip(0.2, 0.8)
        mom_short = _sma(self.resid, p.MOM_SHORT)
        mom_long = _sma(self.resid, p.MOM_LONG)
        return (w * mom_short + (1.0 - w) * mom_long).fillna(0.0)

    def _factor_cross_sectional_dispersion(self) -> "xr.DataArray":
        """Stock relative return weighted by cross-sectional dispersion level."""
        p = self.params
        cs_std = self.ret.std("asset")
        cs_mean = self.ret.mean("asset")

        disp = _sma(cs_std, p.MOM_SHORT)
        disp_norm = disp / (disp.mean("time") + p.eps)

        rel_ret = self.ret - cs_mean
        sig = rel_ret * disp_norm.broadcast_like(rel_ret)
        return _sma(sig, p.MOM_SHORT).fillna(0.0)

    def _factor_liquidity_momentum(self) -> "xr.DataArray":
        """Liquidity-adjusted residual momentum."""
        p = self.params
        resid_mom = _sma(self.resid, p.MOM_WIN)
        adv = _sma(self.vol * self.close, p.ADV_WIN_LONG)
        liq = adv / (adv.mean("asset") + p.eps)
        liq_norm = liq.clip(0, 5) / 2.5
        return (resid_mom * liq_norm).fillna(0.0)

    # ==========================================================================
    # OU Mean-Reversion Factors (ported into base library)
    # ==========================================================================

    def _ou_estimate_params(self, window: int) -> Dict[str, "xr.DataArray"]:
        """Estimate OU parameters via rolling AR(1) on log-price."""
        p = self.params

        close = self.close
        logp = xr.apply_ufunc(np.log, close.where(close > 0)).fillna(0.0)
        X = logp
        X_lag = X.shift(time=1)

        minp = max(3, int(window // 2))
        X_mean = X.rolling(time=window, min_periods=minp).mean()
        X_lag_mean = X_lag.rolling(time=window, min_periods=minp).mean()

        X_var = X_lag.rolling(time=window, min_periods=minp).var()
        XY_mean = (X * X_lag).rolling(time=window, min_periods=minp).mean()
        cov = XY_mean - X_mean * X_lag_mean

        b = cov / (X_var + p.eps)
        b = xr.where(np.abs(b) < 0.9999, b, 0.9999 * np.sign(b))
        a = X_mean - b * X_lag_mean

        theta = -xr.apply_ufunc(np.log, np.abs(b) + p.eps)
        theta = xr.where(theta > 0, theta, p.eps)
        mu = a / (1.0 - b + p.eps)

        predicted = a + b * X_lag
        resid = X - predicted
        sigma = resid.rolling(time=window, min_periods=minp).std()

        hl_min = float(getattr(p, "OU_HALFLIFE_MIN", 2.0))
        hl_max = float(getattr(p, "OU_HALFLIFE_MAX", 42.0))
        half_life = (np.log(2.0) / (theta + p.eps)).clip(hl_min, hl_max)

        zclip = float(getattr(p, "OU_ZSCORE_CLIP", 3.0))
        z = ((X - mu) / (sigma + p.eps)).clip(-zclip, zclip)

        return {
            "theta": theta.fillna(0.0),
            "mu": mu.fillna(0.0),
            "sigma": sigma.fillna(0.0),
            "half_life": half_life.fillna(hl_max),
            "zscore": z.fillna(0.0),
        }

    def _factor_ou_zscore_short(self) -> "xr.DataArray":
        """Short-term OU z-score (inverted: below equilibrium is positive)."""
        p = self.params
        win = int(getattr(p, "OU_SHORT_WIN", 21))
        ou = self._ou_estimate_params(win)
        return (-ou["zscore"]).fillna(0.0)

    def _factor_ou_zscore_med(self) -> "xr.DataArray":
        """Medium-term OU z-score (inverted)."""
        p = self.params
        win = int(getattr(p, "OU_MED_WIN", 63))
        ou = self._ou_estimate_params(win)
        return (-ou["zscore"]).fillna(0.0)

    def _factor_ou_halflife_signal(self) -> "xr.DataArray":
        """Inverse OU half-life (shorter is stronger)."""
        p = self.params
        win = int(getattr(p, "OU_MED_WIN", 63))
        ou = self._ou_estimate_params(win)
        return (1.0 / (ou["half_life"] + 1.0)).fillna(0.0)

    def _factor_ou_reversion_strength(self) -> "xr.DataArray":
        """OU theta (mean reversion speed)."""
        p = self.params
        win = int(getattr(p, "OU_MED_WIN", 63))
        ou = self._ou_estimate_params(win)
        return ou["theta"].fillna(0.0)

    def _factor_ou_predicted_return(self) -> "xr.DataArray":
        """OU expected return proxy: theta*(mu - log_price)."""
        p = self.params
        win = int(getattr(p, "OU_MED_WIN", 63))
        ou = self._ou_estimate_params(win)

        logp = xr.apply_ufunc(np.log, self.close.where(self.close > 0)).fillna(0.0)
        return (ou["theta"] * (ou["mu"] - logp)).fillna(0.0)

    def _factor_ou_regime_indicator(self) -> "xr.DataArray":
        """Proximity to cross-sectional median half-life (closer is better)."""
        p = self.params
        win = int(getattr(p, "OU_MED_WIN", 63))
        ou = self._ou_estimate_params(win)
        hl = ou["half_life"]
        hl_med = hl.median(dim="asset")
        return (-np.abs(hl - hl_med)).fillna(0.0)

    def _factor_ou_equilibrium_dist(self) -> "xr.DataArray":
        """Blended short/med distance from OU equilibrium (inverted)."""
        p = self.params
        w_s = int(getattr(p, "OU_SHORT_WIN", 21))
        w_m = int(getattr(p, "OU_MED_WIN", 63))

        ou_s = self._ou_estimate_params(w_s)
        ou_m = self._ou_estimate_params(w_m)

        mu_blend = 0.3 * ou_s["mu"] + 0.7 * ou_m["mu"]
        sig_blend = 0.3 * ou_s["sigma"] + 0.7 * ou_m["sigma"]

        zclip = float(getattr(p, "OU_ZSCORE_CLIP", 3.0))
        logp = xr.apply_ufunc(np.log, self.close.where(self.close > 0)).fillna(0.0)
        dist = ((logp - mu_blend) / (sig_blend + p.eps)).clip(-zclip, zclip)
        return (-dist).fillna(0.0)

    def _factor_ou_momentum_blend(self) -> "xr.DataArray":
        """Half-life adaptive blend: reversal for short HL, momentum for long HL."""
        p = self.params
        win = int(getattr(p, "OU_MED_WIN", 63))
        ou = self._ou_estimate_params(win)

        hl_min = float(getattr(p, "OU_HALFLIFE_MIN", 2.0))
        hl_max = float(getattr(p, "OU_HALFLIFE_MAX", 42.0))
        hl = ou["half_life"]
        hl_norm = ((hl - hl_min) / (hl_max - hl_min + p.eps)).clip(0, 1)

        mom_21 = self.ret.rolling(time=21, min_periods=15).sum()
        mom_63 = self.ret.rolling(time=63, min_periods=42).sum()
        rev_5 = -self.ret.rolling(time=5, min_periods=3).sum()

        mom_comp = 0.6 * _robust_zscore(mom_21, dim="asset", eps=p.eps) + 0.4 * _robust_zscore(mom_63, dim="asset", eps=p.eps)
        rev_comp = _robust_zscore(rev_5, dim="asset", eps=p.eps)
        return (hl_norm * mom_comp + (1.0 - hl_norm) * rev_comp).fillna(0.0)

    # ==========================================================================
    # GLFT Microstructure Factors (ported into base library)
    # ==========================================================================

    def _glft_signed_volume(self) -> "xr.DataArray":
        close = self.close
        vol = self.vol
        price_change = close.diff(dim="time")
        sign = xr.where(price_change > 0, 1.0, xr.where(price_change < 0, -1.0, 0.0))
        return (sign * vol).fillna(0.0)

    def _glft_ofi(self, window: int) -> "xr.DataArray":
        p = self.params
        signed_vol = self._glft_signed_volume()
        signed_sum = signed_vol.rolling(time=window, min_periods=max(2, window // 2)).sum()
        vol_sum = self.vol.rolling(time=window, min_periods=max(2, window // 2)).sum()
        return (signed_sum / (vol_sum + p.eps)).clip(-1, 1).fillna(0.0)

    def _glft_vpin_proxy(self) -> "xr.DataArray":
        p = self.params
        win = int(getattr(p, "VPIN_WIN", 50))
        signed = self._glft_signed_volume()
        buy = xr.where(signed > 0, signed, 0.0)
        sell = xr.where(signed < 0, -signed, 0.0)
        buy_sum = buy.rolling(time=win, min_periods=max(2, win // 2)).sum()
        sell_sum = sell.rolling(time=win, min_periods=max(2, win // 2)).sum()
        total = buy_sum + sell_sum + p.eps
        return (np.abs(buy_sum - sell_sum) / total).fillna(0.0).clip(0, 1)

    def _glft_inventory_proxy(self) -> "xr.DataArray":
        p = self.params
        decay = float(getattr(p, "INVENTORY_DECAY", 0.95))
        decay = float(np.clip(decay, 0.50, 0.999))
        span = int(max(3, round(1.0 / max(1e-6, (1.0 - decay)))))

        signed = self._glft_signed_volume()
        df = signed.transpose("time", "asset").to_pandas()
        inv = df.ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()
        return xr.DataArray(inv.values, coords=signed.coords, dims=signed.dims).fillna(0.0)

    def _factor_glft_ofi_short(self) -> "xr.DataArray":
        p = self.params
        return self._glft_ofi(int(getattr(p, "OFI_SHORT_WIN", 5)))

    def _factor_glft_ofi_med(self) -> "xr.DataArray":
        p = self.params
        return self._glft_ofi(int(getattr(p, "OFI_MED_WIN", 21)))

    def _factor_glft_ofi_momentum(self) -> "xr.DataArray":
        p = self.params
        ofi_s = self._glft_ofi(int(getattr(p, "OFI_SHORT_WIN", 5)))
        ofi_l = self._glft_ofi(int(getattr(p, "OFI_LONG_WIN", 63)))
        return (ofi_s - ofi_l).fillna(0.0)

    def _factor_glft_flow_toxicity(self) -> "xr.DataArray":
        """Inverted VPIN proxy: low toxicity is good."""
        return (-self._glft_vpin_proxy()).fillna(0.0)

    def _factor_glft_toxic_momentum(self) -> "xr.DataArray":
        """Falling toxicity is good."""
        p = self.params
        vpin = self._glft_vpin_proxy()
        short = vpin.rolling(time=5, min_periods=3).mean()
        long = vpin.rolling(time=21, min_periods=14).mean()
        return (-(short - long)).fillna(0.0)

    def _factor_glft_impact_asymmetry(self) -> "xr.DataArray":
        p = self.params
        win = int(getattr(p, "IMPACT_WIN", 42))

        up_ret = xr.where(self.ret > 0, self.ret, 0.0)
        down_ret = xr.where(self.ret < 0, self.ret, 0.0)
        up_vol = xr.where(self.ret > 0, self.vol, 0.0)
        down_vol = xr.where(self.ret < 0, self.vol, 0.0)

        minp = max(2, win // 2)
        up_impact = (up_ret * up_vol).rolling(time=win, min_periods=minp).sum()
        up_vol_sum = up_vol.rolling(time=win, min_periods=minp).sum()
        down_impact = (down_ret * down_vol).rolling(time=win, min_periods=minp).sum()
        down_vol_sum = down_vol.rolling(time=win, min_periods=minp).sum()

        avg_up = up_impact / (up_vol_sum + p.eps)
        avg_down = np.abs(down_impact) / (down_vol_sum + p.eps)
        return (avg_up - avg_down).fillna(0.0)

    def _factor_glft_inventory_signal(self) -> "xr.DataArray":
        """Contrarian to inventory: high inventory -> negative future returns."""
        return (-self._glft_inventory_proxy()).fillna(0.0)

    def _factor_glft_inventory_risk_prem(self) -> "xr.DataArray":
        """Inventory volatility as risk premium proxy."""
        p = self.params
        win = int(getattr(p, "IMPACT_WIN", 42))
        inv = self._glft_inventory_proxy()
        return inv.rolling(time=win, min_periods=max(2, win // 2)).std().fillna(0.0)

    def _factor_glft_spread_adjusted_mom(self) -> "xr.DataArray":
        p = self.params
        win = int(getattr(p, "SPREAD_EST_WIN", 21))
        hl_ratio = xr.apply_ufunc(np.log, (self.high / (self.low + p.eps) + p.eps).clip(min=p.eps))
        spread_est = hl_ratio.rolling(time=win, min_periods=max(2, win // 2)).mean()
        mom_21 = self.ret.rolling(time=21, min_periods=15).sum()
        return (mom_21 - 2.0 * spread_est).fillna(0.0)

    def _factor_glft_mm_edge(self) -> "xr.DataArray":
        """Composite MM edge: low toxicity + vol + mean reversion."""
        p = self.params
        vpin = self._glft_vpin_proxy()
        low_tox = -vpin

        vol_21 = self.ret.rolling(time=21, min_periods=15).std()
        ret_5 = self.ret.rolling(time=5, min_periods=3).sum()
        mr = -ret_5

        low_tox_z = _robust_zscore(low_tox, dim="asset", eps=p.eps)
        vol_z = _robust_zscore(vol_21, dim="asset", eps=p.eps)
        mr_z = _robust_zscore(mr, dim="asset", eps=p.eps)

        return (0.4 * low_tox_z + 0.3 * vol_z + 0.3 * mr_z).fillna(0.0)
