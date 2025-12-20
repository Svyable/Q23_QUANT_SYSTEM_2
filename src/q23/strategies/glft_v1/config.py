"""GLFTv1 - Guéant–Lehalle–Fernandez-Tapia Microstructure Strategy Configuration.

Based on the GLFT optimal market-making framework, adapted for alpha generation.

The GLFT model optimizes market-making under inventory risk:
- Optimal spread: δ = γσ²τ + (2/γ)ln(1 + γ/k)
- Reservation price: r = s - q·γ·σ²·τ (penalizes inventory)
- Price impact: Λ(q) models adverse selection

For alpha generation, we extract signals from:
1. Order flow imbalance (OFI) - signed volume analysis
2. Price impact asymmetry - bid vs ask pressure
3. Inventory risk premium - excess return for holding risk
4. Spread-normalized returns - risk-adjusted momentum
5. Adverse selection indicators - detecting informed flow

Mathematical Foundation:
- γ (gamma): Inventory risk aversion coefficient
- σ: Volatility of mid-price
- k: Order arrival intensity
- q: Current inventory
- τ: Time horizon

Key Innovations:
- Cross-sectional order flow toxicity (VPIN-inspired)
- Inventory imbalance as contrarian signal
- Spread-adjusted alpha signals
- Market maker edge detection (when to provide vs take liquidity)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class GLFTv1Config:
    """Configuration for GLFT Microstructure Strategy.
    
    Leverages market-making insights for systematic alpha generation.
    """
    
    # Data configuration
    MIN_DATE: str = "2020-01-01"
    EXCHANGES: List[str] = ("NAS", "NYS")
    
    # Portfolio construction - microstructure-optimized
    LONG_SEATS: int = 16
    SHORT_SEATS: int = 8
    LONG_ONLY: bool = False
    
    # Position sizing
    TOPN_BASE: int = 24
    TOPN_VOLATILE: int = 18
    MAX_POS: float = 0.08
    MIN_POS: float = 0.002
    
    # Risk/Leverage
    TARGET_VOL_BASE: float = 0.15
    TARGET_VOL_VOLATILE: float = 0.11
    LEV_CAP: float = 1.55
    LEV_MIN: float = 0.40
    
    # Transaction costs - critical for microstructure strategy
    TC_BPS: float = 6.0  # Lower as we're trying to be smart about execution
    
    # Smoothing - microstructure signals need less smoothing
    WEIGHT_SMOOTH_ALPHA: float = 0.28
    SCORE_SMOOTH_WIN: int = 3
    SOFTMAX_TILT_ALPHA: float = 0.85
    
    # Risk throttle
    RISK_OFF_STRETCH: float = 0.68
    RISK_OFF_FLOOR: float = 0.48
    DD_WIN: int = 126
    
    # IC weighting
    IC_LAMBDA: float = 0.94
    
    # GLFT-specific parameters
    OFI_SHORT_WIN: int = 5       # Order flow imbalance short window
    OFI_MED_WIN: int = 21        # OFI medium window
    OFI_LONG_WIN: int = 63       # OFI long window
    VPIN_BUCKET_SIZE: int = 50   # Volume buckets for VPIN-style calc
    VPIN_WIN: int = 50           # Number of buckets for VPIN
    SPREAD_EST_WIN: int = 21     # Window for spread estimation
    IMPACT_WIN: int = 42         # Price impact estimation window
    INVENTORY_DECAY: float = 0.95  # Decay for inventory proxy
    
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


glft_v1_config = GLFTv1Config()


# =============================================================================
# GLFT-Specific Factors (10 novel factors)
# =============================================================================

GLFT_NOVEL_FACTORS: List[str] = [
    # Order Flow Analysis
    "glft_ofi_short",           # Order flow imbalance (5d)
    "glft_ofi_med",             # Order flow imbalance (21d)
    "glft_ofi_momentum",        # OFI momentum (change in OFI)
    
    # VPIN-style Toxicity
    "glft_flow_toxicity",       # Volume-synchronized probability of informed trading
    "glft_toxic_momentum",      # Change in toxicity (informed flow acceleration)
    
    # Price Impact / Inventory
    "glft_impact_asymmetry",    # Asymmetric price impact (buy vs sell)
    "glft_inventory_signal",    # Inventory-based contrarian signal
    "glft_inventory_risk_prem", # Premium for holding inventory risk
    
    # Spread/Execution Quality
    "glft_spread_adjusted_mom", # Momentum adjusted for spread costs
    "glft_mm_edge",             # Market maker edge (when to provide liquidity)
]

# Complementary factors
GLFT_COMPLEMENTARY_FACTORS: List[str] = [
    # Liquidity (natural complement)
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
    
    # Technical
    "breakout",
    "slope",
    "calm_flow",
    "vol_breakout",
    
    # Quality
    "beta_stability",
    "micro_noise",
]

# Combined factor list (25 total: 10 GLFT-novel + 15 complementary)
GLFT_V1_FACTORS: List[str] = GLFT_NOVEL_FACTORS + GLFT_COMPLEMENTARY_FACTORS
