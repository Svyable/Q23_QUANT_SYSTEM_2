"""Q23 Neural Alpha Strategy Configuration.

A state-of-the-art Pure Alpha Long-Short strategy incorporating 15 novel factors
derived from behavioral finance, market microstructure, and machine learning research.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class NeuralAlphaConfig:
    """Configuration for Q23 Neural Alpha Strategy.
    
    This is an aggressive pure-alpha long-short strategy designed to maximize
    risk-adjusted returns using novel research-backed factors.
    """
    
    # Data configuration
    MIN_DATE: str = "2020-01-01"
    EXCHANGES: List[str] = ("NAS", "NYS")
    
    # Long-Short configuration
    LONG_SEATS: int = 12
    SHORT_SEATS: int = 8
    LONG_ONLY: bool = False
    
    # Position sizing - aggressive alpha parameters
    TOPN_BASE: int = 20
    TOPN_VOLATILE: int = 15
    MAX_POS: float = 0.12  # Higher concentration for conviction
    MIN_POS: float = 0.003
    
    # Risk/Leverage parameters
    TARGET_VOL_BASE: float = 0.18  # Higher target volatility
    TARGET_VOL_VOLATILE: float = 0.12
    LEV_CAP: float = 1.8
    LEV_MIN: float = 0.4
    
    # Transaction costs
    TC_BPS: float = 10.0
    
    # Smoothing parameters
    WEIGHT_SMOOTH_ALPHA: float = 0.35
    SCORE_SMOOTH_WIN: int = 3
    SOFTMAX_TILT_ALPHA: float = 0.90  # Higher tilt for more conviction
    
    # Risk throttle
    RISK_OFF_STRETCH: float = 0.65
    RISK_OFF_FLOOR: float = 0.45
    DD_WIN: int = 252
    
    # IC weighting
    IC_LAMBDA: float = 0.94
    
    # Factor window parameters (can be overridden)
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
    
    # Neural Alpha specific windows
    EFFICIENCY_WIN: int = 21  # Efficiency ratio lookback
    ATTENTION_WIN: int = 10   # Attention signal window
    DISPOSITION_WIN: int = 63 # Disposition effect lookback
    SKEW_WIN: int = 63        # Skewness calculation window
    KURT_WIN: int = 63        # Kurtosis calculation window
    VOV_WIN: int = 21         # Vol-of-vol window
    PERSISTENCE_WIN: int = 42 # Momentum persistence window
    
    # Numeric precision
    EPS: float = 1e-12
    
    def __post_init__(self):
        object.__setattr__(self, "EXCHANGES", list(self.EXCHANGES))


# Default configuration instance
neural_alpha_config = NeuralAlphaConfig()


# =============================================================================
# Factor Lists
# =============================================================================

# 15 Novel factors implemented in this strategy
NEURAL_ALPHA_NOVEL_FACTORS: List[str] = [
    # Behavioral Finance Factors (3)
    "attention_momentum",      # Volume-price attention signal
    "disposition_alpha",       # Unrealized P&L vs anchoring price
    "anchoring_bias",          # Distance from psychological price levels
    
    # Market Microstructure Factors (3)
    "efficiency_ratio",        # Kaufman Efficiency Ratio
    "information_flow",        # Volume-weighted price impact asymmetry
    "mean_reversion_speed",    # Half-life of price deviations
    
    # Momentum Quality Factors (3)
    "momentum_quality_ratio",  # Signal-to-noise of momentum
    "momentum_persistence",    # Autocorrelation-based momentum strength
    "momentum_divergence",     # Price vs residual momentum divergence
    
    # Risk/Regime Factors (4)
    "vol_of_vol",              # Volatility of volatility
    "skewness_factor",         # Rolling return skewness
    "kurtosis_factor",         # Rolling return kurtosis
    "regime_momentum",         # Volatility-regime-adaptive momentum
    
    # Cross-Sectional Factors (2)
    "cross_sectional_dispersion",  # Return dispersion opportunity
    "liquidity_momentum",          # Liquidity-adjusted momentum
]

# Proven existing factors to include
NEURAL_ALPHA_EXISTING_FACTORS: List[str] = [
    # Defensive (3)
    "inv_vol",
    "inv_idio",
    "low_beta",
    
    # Momentum (3)
    "resid_mom",
    "resid_mom_mix",
    "mtf_ic_momentum",
    
    # Reversal (2)
    "srev",
    "lrev",
    
    # Technical (3)
    "breakout",
    "slope",
    "ma_cloud",
    
    # Liquidity (2)
    "liquidity",
    "amihud_inv",
    
    # Quality (2)
    "idio_tail_risk",
    "beta_stability",
    
    # MTF (2)
    "mtf_alignment",
    "hloc_close_position",
]

# Combined factor list (32 total: 15 novel + 17 existing)
NEURAL_ALPHA_FACTORS: List[str] = (
    NEURAL_ALPHA_NOVEL_FACTORS + NEURAL_ALPHA_EXISTING_FACTORS
)
