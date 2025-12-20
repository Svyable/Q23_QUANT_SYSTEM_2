"""OUv1 - Ornstein-Uhlenbeck Mean-Reversion Strategy Configuration.

Based on the canonical OU process: dX = θ(μ - X)dt + σdW

This strategy captures alpha through:
1. Cross-sectional mean-reversion using OU-estimated parameters
2. Half-life based position sizing (faster reversion = higher confidence)
3. Z-score deviation from estimated equilibrium
4. Regime-adaptive parameters based on market volatility

Mathematical Foundation:
- θ (kappa): Mean-reversion speed - estimated via AR(1) regression
- μ (mu): Long-term equilibrium level - rolling mean
- σ (sigma): Volatility of innovations
- Half-life = ln(2)/θ - time to revert halfway to mean

Key Innovations:
- Dynamic half-life estimation for optimal holding period
- Cross-sectional OU parameter dispersion as regime indicator
- Multi-timeframe OU signals (short/medium/long half-lives)
- Volatility-regime adaptive mean-reversion strength
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class OUv1Config:
    """Configuration for OU Mean-Reversion Strategy.
    
    Leverages Ornstein-Uhlenbeck dynamics for systematic mean-reversion alpha.
    """
    
    # Data configuration
    MIN_DATE: str = "2020-01-01"
    EXCHANGES: List[str] = ("NAS", "NYS")
    
    # Portfolio construction - mean-reversion optimized
    LONG_SEATS: int = 15  # More positions for diversification
    SHORT_SEATS: int = 10  # Balanced short book for mean-reversion
    LONG_ONLY: bool = False  # L/S critical for mean-reversion
    
    # Position sizing - conservative for mean-reversion
    TOPN_BASE: int = 25
    TOPN_VOLATILE: int = 20
    MAX_POS: float = 0.08  # Lower max position
    MIN_POS: float = 0.002
    
    # Risk/Leverage - mean-reversion tends to have lower vol
    TARGET_VOL_BASE: float = 0.14
    TARGET_VOL_VOLATILE: float = 0.10
    LEV_CAP: float = 1.5
    LEV_MIN: float = 0.4
    
    # Transaction costs
    TC_BPS: float = 8.0
    
    # Smoothing - faster for mean-reversion signals
    WEIGHT_SMOOTH_ALPHA: float = 0.25  # Less smoothing for faster signals
    SCORE_SMOOTH_WIN: int = 3
    SOFTMAX_TILT_ALPHA: float = 0.80
    
    # Risk throttle
    RISK_OFF_STRETCH: float = 0.65
    RISK_OFF_FLOOR: float = 0.50
    DD_WIN: int = 126
    
    # IC weighting
    IC_LAMBDA: float = 0.94
    
    # OU-specific parameters
    OU_SHORT_WIN: int = 21      # Short-term OU estimation (1 month)
    OU_MED_WIN: int = 63        # Medium-term OU estimation (3 months)
    OU_LONG_WIN: int = 126      # Long-term OU estimation (6 months)
    OU_HALFLIFE_MIN: int = 2    # Min half-life for valid signal (days)
    OU_HALFLIFE_MAX: int = 42   # Max half-life (longer = weaker signal)
    OU_ZSCORE_CLIP: float = 3.0 # Clip extreme z-scores
    
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


ou_v1_config = OUv1Config()


# =============================================================================
# OU-Specific Factors (8 novel factors)
# =============================================================================

OU_NOVEL_FACTORS: List[str] = [
    # Core OU factors
    "ou_zscore_short",      # Short-term OU z-score (21d)
    "ou_zscore_med",        # Medium-term OU z-score (63d)
    "ou_halflife_signal",   # Inverse half-life (faster reversion = stronger)
    "ou_reversion_strength", # θ parameter normalized cross-sectionally
    
    # OU-derived signals
    "ou_predicted_return",   # E[return] from OU dynamics
    "ou_regime_indicator",   # Cross-sectional OU param dispersion
    "ou_equilibrium_dist",   # Distance from equilibrium (normalized)
    "ou_momentum_blend",     # OU-weighted momentum (short halflife = reversal)
]

# Proven complementary factors
OU_COMPLEMENTARY_FACTORS: List[str] = [
    # Reversal (natural complement to OU)
    "srev",                 # Short-term reversal
    "lrev",                 # Long-term reversal
    
    # Defensive (risk management)
    "inv_vol",
    "inv_idio",
    "inv_down",
    "low_beta",
    "liquidity",
    "amihud_inv",
    
    # Momentum (regime balance)
    "resid_mom",
    "resid_mom_mix",
    
    # Technical
    "breakout",
    "slope",
    "ma_cloud",
    "calm_flow",
    
    # Quality
    "beta_stability",
    "quality_defensive",
]

# Combined factor list (24 total: 8 OU-novel + 16 complementary)
OU_V1_FACTORS: List[str] = OU_NOVEL_FACTORS + OU_COMPLEMENTARY_FACTORS
