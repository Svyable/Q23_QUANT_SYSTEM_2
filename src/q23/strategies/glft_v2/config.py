"""GLFTv2 - Enhanced GLFT Microstructure Strategy Configuration.

V2 Enhancements over V1:
1. Kyle's Lambda estimation - price impact coefficient
2. Amihud-GLFT hybrid liquidity factor
3. Volume clock signals - time vs volume scaling
4. Avellaneda-Stoikov reservation price signals
5. Cross-sectional flow dispersion (market regime)
6. Enhanced OFI with dollar-volume weighting
7. Improved toxic flow decomposition

Mathematical Additions:
- λ (Kyle's lambda): Price impact per unit volume
- φ (phi): Volume clock adjustment factor
- ρ (rho): Cross-sectional flow correlation
- τ_eff: Effective time horizon under volume clock

Key Innovations:
- Dollar-volume weighted OFI for size-adjusted signals
- Kyle's lambda estimation for permanent impact
- Volume clock transformation for time-invariant signals
- Enhanced toxicity with Bulk Volume Classification (BVC)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class GLFTv2Config:
    """Configuration for GLFTv2 Enhanced Microstructure Strategy."""
    
    # Data configuration
    MIN_DATE: str = "2020-01-01"
    EXCHANGES: List[str] = ("NAS", "NYS")
    
    # Portfolio construction - more aggressive for microstructure alpha
    LONG_SEATS: int = 14
    SHORT_SEATS: int = 10
    LONG_ONLY: bool = False
    
    # Position sizing - tighter for microstructure
    TOPN_BASE: int = 24
    TOPN_VOLATILE: int = 16
    MAX_POS: float = 0.085
    MIN_POS: float = 0.003
    
    # Risk/Leverage - enhanced for microstructure edge
    TARGET_VOL_BASE: float = 0.16
    TARGET_VOL_VOLATILE: float = 0.12
    LEV_CAP: float = 1.6
    LEV_MIN: float = 0.35
    
    # Transaction costs
    TC_BPS: float = 5.5
    
    # Smoothing
    WEIGHT_SMOOTH_ALPHA: float = 0.30
    SCORE_SMOOTH_WIN: int = 3
    SOFTMAX_TILT_ALPHA: float = 0.88
    
    # Risk throttle
    RISK_OFF_STRETCH: float = 0.70
    RISK_OFF_FLOOR: float = 0.45
    DD_WIN: int = 126
    
    # IC weighting
    IC_LAMBDA: float = 0.95
    
    # GLFT-specific parameters (enhanced from v1)
    OFI_SHORT_WIN: int = 5
    OFI_MED_WIN: int = 21
    OFI_LONG_WIN: int = 63
    VPIN_WIN: int = 42
    SPREAD_EST_WIN: int = 21
    IMPACT_WIN: int = 42
    INVENTORY_DECAY: float = 0.94
    
    # V2 New Parameters
    KYLE_LAMBDA_WIN: int = 42     # Window for Kyle's lambda estimation
    VOLUME_CLOCK_WIN: int = 21    # Volume clock normalization window
    BVC_WIN: int = 10             # Bulk Volume Classification window
    FLOW_DISP_WIN: int = 21       # Cross-sectional flow dispersion
    RESERVATION_WIN: int = 14     # Reservation price computation
    
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


glft_v2_config = GLFTv2Config()


# =============================================================================
# GLFTv2 Factors (16 novel + 14 complementary = 30 total)
# =============================================================================

GLFT_V2_NOVEL_FACTORS: List[str] = [
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
    
    # V2 New Factors (6 factors)
    "glft_kyle_lambda",           # Kyle's price impact coefficient
    "glft_volume_clock_ofi",      # Volume-clock adjusted OFI
    "glft_bvc_imbalance",         # Bulk Volume Classification imbalance
    "glft_flow_dispersion",       # Cross-sectional flow dispersion
    "glft_reservation_signal",    # Avellaneda-Stoikov reservation price
    "glft_amihud_hybrid",         # Amihud-GLFT hybrid liquidity
]

GLFT_V2_COMPLEMENTARY_FACTORS: List[str] = [
    # Liquidity
    "liquidity",
    "amihud_inv",
    
    # Defensive
    "inv_vol",
    "inv_idio",
    "inv_down",
    "low_beta",
    
    # Momentum/Reversal
    "resid_mom",
    "resid_mom_mix",
    "srev",
    "lrev",
    
    # Technical
    "breakout",
    "slope",
    "calm_flow",
    "vol_breakout",
]

GLFT_V2_FACTORS: List[str] = GLFT_V2_NOVEL_FACTORS + GLFT_V2_COMPLEMENTARY_FACTORS
