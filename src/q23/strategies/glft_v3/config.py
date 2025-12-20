"""GLFTv3 - Advanced GLFT Microstructure Strategy Configuration.

V3 Enhancements over V2:
1. Multi-timeframe OFI alignment (MTF-OFI)
2. Sector-relative flow signals
3. Regime-adaptive toxicity (RAT)
4. Market-wide flow momentum
5. Execution quality score
6. Adverse selection decomposition
7. Optimal spread signal
8. Flow persistence factor

Risk Management Enhancements:
- Lower drawdown target (Sharpe-optimized)
- Adaptive leverage based on flow regime
- Enhanced defensive factor weighting
- Tighter position limits

Target Profile:
- Sharpe > 3.5
- MaxDD < -5%
- CAGR > 40%
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class GLFTv3Config:
    """Configuration for GLFTv3 Advanced Microstructure Strategy."""
    
    # Data configuration
    MIN_DATE: str = "2020-01-01"
    EXCHANGES: List[str] = ("NAS", "NYS")
    
    # Portfolio construction - Sharpe-optimized
    LONG_SEATS: int = 12
    SHORT_SEATS: int = 8
    LONG_ONLY: bool = False
    
    # Position sizing - conservative for lower DD
    TOPN_BASE: int = 20
    TOPN_VOLATILE: int = 14
    MAX_POS: float = 0.075
    MIN_POS: float = 0.004
    
    # Risk/Leverage - Sharpe-optimized, lower DD
    TARGET_VOL_BASE: float = 0.13
    TARGET_VOL_VOLATILE: float = 0.09
    LEV_CAP: float = 1.45
    LEV_MIN: float = 0.40
    
    # Transaction costs
    TC_BPS: float = 5.0
    
    # Smoothing - balanced
    WEIGHT_SMOOTH_ALPHA: float = 0.32
    SCORE_SMOOTH_WIN: int = 3
    SOFTMAX_TILT_ALPHA: float = 0.90
    
    # Risk throttle - aggressive DD protection
    RISK_OFF_STRETCH: float = 0.75
    RISK_OFF_FLOOR: float = 0.50
    DD_WIN: int = 126
    
    # IC weighting
    IC_LAMBDA: float = 0.96
    
    # GLFT base parameters
    OFI_SHORT_WIN: int = 5
    OFI_MED_WIN: int = 21
    OFI_LONG_WIN: int = 63
    VPIN_WIN: int = 42
    SPREAD_EST_WIN: int = 21
    IMPACT_WIN: int = 42
    INVENTORY_DECAY: float = 0.94
    
    # V2 parameters
    KYLE_LAMBDA_WIN: int = 42
    VOLUME_CLOCK_WIN: int = 21
    BVC_WIN: int = 10
    FLOW_DISP_WIN: int = 21
    RESERVATION_WIN: int = 14
    
    # V3 NEW parameters
    MTF_OFI_SHORT: int = 5         # Multi-timeframe OFI short
    MTF_OFI_MED: int = 21          # Multi-timeframe OFI medium
    MTF_OFI_LONG: int = 63         # Multi-timeframe OFI long
    REGIME_WIN: int = 42           # Regime detection window
    FLOW_PERSIST_WIN: int = 10     # Flow persistence window
    EXEC_QUALITY_WIN: int = 21     # Execution quality window
    ADVERSE_WIN: int = 14          # Adverse selection window
    OPTIMAL_SPREAD_WIN: int = 21   # Optimal spread window
    
    # Standard factor windows
    BETA_WIN: int = 126
    IDIO_WIN: int = 42
    DOWN_WIN: int = 42
    CORR_WIN: int = 42
    ADV_WIN_LONG: int = 63
    ADV_WIN_SHORT: int = 5
    MOM_WIN: int = 84
    MOM_SHORT: int = 21
    MOM_LONG: int = 126
    REV_WIN: int = 5
    REV_LONG: int = 21
    DON_WIN: int = 63
    EMA_FAST: int = 12
    EMA_SLOW: int = 26
    ATR_WIN: int = 14
    
    EPS: float = 1e-12
    
    def __post_init__(self):
        object.__setattr__(self, "EXCHANGES", list(self.EXCHANGES))


glft_v3_config = GLFTv3Config()


# =============================================================================
# GLFTv3 Factors (24 novel + 16 complementary = 40 total)
# =============================================================================

GLFT_V3_NOVEL_FACTORS: List[str] = [
    # V1 Core (10 factors)
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
    
    # V2 Factors (6 factors)
    "glft_kyle_lambda",
    "glft_volume_clock_ofi",
    "glft_bvc_imbalance",
    "glft_flow_dispersion",
    "glft_reservation_signal",
    "glft_amihud_hybrid",
    
    # V3 NEW Factors (8 factors)
    "glft_mtf_ofi_alignment",     # Multi-timeframe OFI alignment
    "glft_sector_rel_flow",       # Sector-relative flow signal
    "glft_regime_toxicity",       # Regime-adaptive toxicity
    "glft_market_flow_mom",       # Market-wide flow momentum
    "glft_exec_quality",          # Execution quality score
    "glft_adverse_decomp",        # Adverse selection decomposition
    "glft_optimal_spread",        # Optimal spread signal
    "glft_flow_persistence",      # Flow persistence factor
]

GLFT_V3_COMPLEMENTARY_FACTORS: List[str] = [
    # Liquidity
    "liquidity",
    "amihud_inv",
    
    # Defensive (enhanced weight in v3)
    "inv_vol",
    "inv_idio",
    "inv_down",
    "low_beta",
    "low_corr",
    
    # Momentum/Reversal
    "resid_mom",
    "resid_mom_mix",
    "srev",
    "lrev",
    
    # Technical
    "breakout",
    "slope",
    "calm_flow",
    
    # Quality
    "beta_stability",
    "quality_defensive",
]

GLFT_V3_FACTORS: List[str] = GLFT_V3_NOVEL_FACTORS + GLFT_V3_COMPLEMENTARY_FACTORS
