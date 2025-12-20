"""Q23 Neural Alpha v2 Strategy Configuration.

An enhanced version of Neural Alpha combining:
- All 15 novel behavioral/microstructure factors from v1
- Superior risk controls inspired by GTP51MAX (lowest DD at -3.75%)
- Best momentum factors from NASNYS V4 (highest CAGR at 61.91%)
- Quality/defensive factors from COMPOSER v1 (Sharpe 3.106)

Target Improvements over v1:
- MaxDD: -6.13% → < -5% (tighter risk controls)
- Sharpe: 3.245 → > 3.5 (better risk-adjusted returns)
- Maintain CAGR > 55%

Key Design Decisions:
1. Asymmetric L/S (14L/6S): More longs for alpha, focused shorts for hedge
2. Enhanced risk throttle with faster DD response (126 vs 252 days)
3. Correlation penalty to reduce factor overlap
4. Vol overlay for drawdown protection
5. 40 factors (15 novel + 25 proven) for robust alpha capture
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class NeuralAlphaV2Config:
    """Configuration for Q23 Neural Alpha v2 Strategy.
    
    An evolved pure-alpha long-short strategy combining the best aspects of:
    - Neural Alpha v1 (novel factors, best Sharpe/Calmar)
    - GTP51MAX (lowest drawdown, risk controls)
    - NASNYS V4 (highest CAGR, proven factors)
    - COMPOSER v1 (balanced Sharpe/Sortino)
    """
    
    # Data configuration
    MIN_DATE: str = "2020-01-01"  # Longer history for better IC estimation
    EXCHANGES: List[str] = ("NAS", "NYS")
    
    # Long-Short configuration - asymmetric for alpha capture
    LONG_SEATS: int = 14  # More longs for upside capture
    SHORT_SEATS: int = 6   # Focused shorts for downside hedge
    LONG_ONLY: bool = False
    
    # Position sizing - balanced between concentration and diversification
    TOPN_BASE: int = 22  # Slightly more positions for risk distribution
    TOPN_VOLATILE: int = 18  # Reduce in volatile markets
    MAX_POS: float = 0.09  # Between v1's 0.12 and GTP51MAX's 0.07
    MIN_POS: float = 0.002
    
    # Risk/Leverage parameters - tighter controls inspired by GTP51MAX
    TARGET_VOL_BASE: float = 0.16  # Lower than v1's 0.18 for risk control
    TARGET_VOL_VOLATILE: float = 0.12  # More conservative in vol regime
    LEV_CAP: float = 1.65  # Lower than v1's 1.8
    LEV_MIN: float = 0.45
    
    # Transaction costs
    TC_BPS: float = 8.0  # Slightly lower with optimized turnover
    
    # Smoothing parameters - optimized for signal quality
    WEIGHT_SMOOTH_ALPHA: float = 0.38  # Slightly faster adaptation
    SCORE_SMOOTH_WIN: int = 4  # Between 3 and 5
    SOFTMAX_TILT_ALPHA: float = 0.88  # Balanced conviction
    
    # Risk throttle - faster response inspired by GTP51MAX
    RISK_OFF_STRETCH: float = 0.70  # More aggressive risk reduction
    RISK_OFF_FLOOR: float = 0.50
    DD_WIN: int = 126  # Faster response (v1=252, GTP51MAX=126)
    
    # IC weighting
    IC_LAMBDA: float = 0.95  # Slightly higher for stability
    
    # Factor window parameters - optimized blend
    BETA_WIN: int = 126  # Shorter for adaptation (v1=162)
    IDIO_WIN: int = 52  # Between 42 and 63
    DOWN_WIN: int = 52
    CORR_WIN: int = 52
    ADV_WIN_LONG: int = 63
    ADV_WIN_SHORT: int = 5
    MOM_WIN: int = 84
    MOM_SHORT: int = 21
    MOM_LONG: int = 126
    REV_WIN: int = 5
    REV_LONG: int = 21
    DON_WIN: int = 63
    EMA_FAST: int = 16  # Slightly faster (GTP51MAX style)
    EMA_SLOW: int = 42
    ATR_WIN: int = 28  # Shorter for responsiveness
    
    # Neural Alpha specific windows (from v1)
    EFFICIENCY_WIN: int = 21
    ATTENTION_WIN: int = 10
    DISPOSITION_WIN: int = 63
    SKEW_WIN: int = 63
    KURT_WIN: int = 63
    VOV_WIN: int = 21
    PERSISTENCE_WIN: int = 42
    
    # Additional windows for v2 factors
    SECTOR_MOM_WIN: int = 63
    VALUE_WIN: int = 240
    QUALITY_WIN: int = 252
    
    # Numeric precision
    EPS: float = 1e-12
    
    def __post_init__(self):
        object.__setattr__(self, "EXCHANGES", list(self.EXCHANGES))


# Default configuration instance
neural_alpha_v2_config = NeuralAlphaV2Config()


# =============================================================================
# Factor Lists - Curated 40-factor ensemble
# =============================================================================

# All 15 Novel factors from Neural Alpha v1 (the differentiators)
NEURAL_ALPHA_V2_NOVEL_FACTORS: List[str] = [
    # Behavioral Finance Factors (3) - unique alpha source
    "attention_momentum",      # Volume-price attention signal
    "disposition_alpha",       # Unrealized P&L vs anchoring price
    "anchoring_bias",          # Distance from psychological price levels
    
    # Market Microstructure Factors (3) - information edge
    "efficiency_ratio",        # Kaufman Efficiency Ratio
    "information_flow",        # Volume-weighted price impact asymmetry
    "mean_reversion_speed",    # Half-life of price deviations
    
    # Momentum Quality Factors (3) - signal refinement
    "momentum_quality_ratio",  # Signal-to-noise of momentum
    "momentum_persistence",    # Autocorrelation-based momentum strength
    "momentum_divergence",     # Price vs residual momentum divergence
    
    # Risk/Regime Factors (4) - regime awareness
    "vol_of_vol",              # Volatility of volatility
    "skewness_factor",         # Rolling return skewness
    "kurtosis_factor",         # Rolling return kurtosis
    "regime_momentum",         # Volatility-regime-adaptive momentum
    
    # Cross-Sectional Factors (2) - relative value
    "cross_sectional_dispersion",  # Return dispersion opportunity
    "liquidity_momentum",          # Liquidity-adjusted momentum
]

# Core Defensive Factors (8) - risk control inspired by GTP51MAX
NEURAL_ALPHA_V2_DEFENSIVE_FACTORS: List[str] = [
    "inv_vol",           # Inverse volatility - core defensive
    "inv_idio",          # Inverse idiosyncratic volatility
    "inv_down",          # Inverse downside volatility (Sortino-focused)
    "low_corr",          # Low correlation to market
    "low_beta",          # Low beta
    "liquidity",         # High liquidity
    "amihud_inv",        # Inverse Amihud illiquidity
    "beta_stability",    # Stable beta (lower risk)
]

# Momentum Factors (8) - proven alpha from all strategies
NEURAL_ALPHA_V2_MOMENTUM_FACTORS: List[str] = [
    "resid_mom",         # Residual momentum (core)
    "resid_mom_short",   # Short-term residual momentum
    "resid_mom_long",    # Long-term residual momentum
    "resid_mom_mix",     # Mixed momentum
    "srev",              # Short-term reversal
    "lrev",              # Long-term reversal
    "mtf_ic_momentum",   # Multi-timeframe IC momentum
    "trend_consistency", # Trend consistency (from GTP51MAX)
]

# Technical Factors (6) - price action signals
NEURAL_ALPHA_V2_TECHNICAL_FACTORS: List[str] = [
    "breakout",          # Price breakout
    "slope",             # EMA slope
    "ma_cloud",          # Moving average cloud
    "prox_52w_high",     # Proximity to 52-week high
    "hloc_close_position",  # HLOC position (MTF)
    "calm_flow",         # Calm flow (low vol/high return)
]

# Quality/Value Factors (3) - fundamental anchoring
NEURAL_ALPHA_V2_QUALITY_FACTORS: List[str] = [
    "value_mom",         # Value momentum
    "quality_defensive", # Quality defensive
    "quality_score",     # Composite quality score
]

# Combined factor list (40 total)
NEURAL_ALPHA_V2_FACTORS: List[str] = (
    NEURAL_ALPHA_V2_NOVEL_FACTORS +      # 15 novel
    NEURAL_ALPHA_V2_DEFENSIVE_FACTORS +   # 8 defensive
    NEURAL_ALPHA_V2_MOMENTUM_FACTORS +    # 8 momentum
    NEURAL_ALPHA_V2_TECHNICAL_FACTORS +   # 6 technical
    NEURAL_ALPHA_V2_QUALITY_FACTORS       # 3 quality
)

# Verify count
assert len(NEURAL_ALPHA_V2_FACTORS) == 40, f"Expected 40 factors, got {len(NEURAL_ALPHA_V2_FACTORS)}"
