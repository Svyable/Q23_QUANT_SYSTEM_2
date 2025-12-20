from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Q23ComposerV2Config:
    """
    Q23_COMPOSERv2 Configuration
    
    Advanced strategy optimized for maximum long-side alpha capture with
    defensive short overlay for downside protection.
    
    Key Innovations:
    - Asymmetric long-short positioning (18L/6S) focused on alpha capture
    - Enhanced factor set (35+ factors) combining best from all strategies
    - Regime-aware IC weighting with factor decay
    - Dynamic short sizing based on market stress
    - Superior IC health through factor selection and weighting
    """
    MIN_DATE: str = "2020-01-01"  # Longer history for better IC estimation
    EXCHANGES: List[str] = ("NAS", "NYS")  # Both exchanges for diversification
    
    # Portfolio construction - asymmetric long-focused
    TOPN_BASE: int = 24  # More positions for diversification
    TOPN_VOLATILE: int = 20
    MAX_POS: float = 0.10  # Balanced max position
    MIN_POS: float = 0.001
    
    # Volatility targeting - optimized for alpha capture
    TARGET_VOL_BASE: float = 0.20  # Higher vol target for alpha
    TARGET_VOL_VOLATILE: float = 0.18
    LEV_CAP: float = 1.7  # Higher leverage for alpha capture
    LEV_MIN: float = 0.5
    
    # Transaction costs - realistic for Quantiacs
    TC_BPS: float = 8.0  # Moderate transaction cost
    
    # Long-Short configuration - asymmetric
    LONG_SEATS: int = 18  # More longs for alpha capture
    SHORT_SEATS: int = 6  # Fewer shorts, focused on downside protection
    LONG_ONLY: bool = False
    
    # Smoothing parameters - optimized for signal quality
    WEIGHT_SMOOTH_ALPHA: float = 0.40  # Faster adaptation for regime changes
    SCORE_SMOOTH_WIN: int = 5  # Longer smoothing for stability
    SOFTMAX_TILT_ALPHA: float = 0.92  # Strong tilt toward top signals
    
    # Risk management - enhanced downside protection
    RISK_OFF_STRETCH: float = 0.75  # More aggressive risk reduction
    RISK_OFF_FLOOR: float = 0.50
    DD_WIN: int = 126  # Shorter window for faster response
    
    # IC weighting - enhanced for better IC health
    IC_LAMBDA: float = 0.96  # Higher persistence for stability
    IC_MIN_WEIGHT: float = 0.015  # Higher minimum for factor diversity
    IC_DECAY_HALFLIFE: int = 84  # Decay for stale factors
    IC_CLIP: float = 0.30  # Higher clip for better signal capture
    
    # Factor computation windows - optimized
    BETA_WIN: int = 126  # Standard beta window
    IDIO_WIN: int = 42  # Shorter for recent idiosyncratic risk
    DOWN_WIN: int = 42  # Focus on recent downside volatility
    CORR_WIN: int = 42
    ADV_WIN_LONG: int = 63
    ADV_WIN_SHORT: int = 5
    
    # Momentum windows - multi-timeframe
    MOM_SHORT: int = 21  # Short-term momentum
    MOM_MED: int = 63   # Medium-term momentum
    MOM_LONG: int = 126  # Long-term momentum
    REV_SHORT: int = 5   # Short-term reversal
    REV_LONG: int = 21   # Long-term reversal
    
    # Technical indicators
    DON_WIN: int = 63
    EMA_FAST: int = 12   # Faster EMA for responsiveness
    EMA_SLOW: int = 26
    EMA_TREND: int = 50  # Trend EMA
    ATR_WIN: int = 14    # Shorter ATR for volatility
    
    # Sector/relative momentum
    SECTOR_MOM_WIN: int = 42
    
    # Quality/value factors
    QUALITY_WIN: int = 252
    VALUE_WIN: int = 252
    
    # Volatility factors
    VOL_SURPRISE_WIN: int = 21
    VOL_BREAKOUT_WIN: int = 42
    
    # Neural Alpha specific windows (for novel factors)
    EFFICIENCY_WIN: int = 21  # Efficiency ratio lookback
    ATTENTION_WIN: int = 10   # Attention signal window
    DISPOSITION_WIN: int = 63 # Disposition effect lookback
    SKEW_WIN: int = 63        # Skewness calculation window
    KURT_WIN: int = 63        # Kurtosis calculation window
    VOV_WIN: int = 21         # Vol-of-vol window
    PERSISTENCE_WIN: int = 42 # Momentum persistence window
    
    EPS: float = 1e-12
    
    def __post_init__(self):
        object.__setattr__(self, "EXCHANGES", list(self.EXCHANGES))


composer_v2_config = Q23ComposerV2Config()


# Enhanced factor set - 36 factors targeting maximum alpha with downside protection
COMPOSER_V2_FACTORS: List[str] = [
    # === Risk/Defensive Factors (9) ===
    "inv_vol",           # Inverse volatility - core defensive
    "inv_idio",          # Inverse idiosyncratic volatility
    "inv_down",          # Inverse downside volatility (Sortino-focused)
    "low_corr",          # Low correlation to market
    "low_beta",          # Low beta
    "liquidity",         # High liquidity
    "amihud_inv",        # Inverse Amihud illiquidity
    "beta_stability",    # Stable beta (lower risk)
    "idio_tail_risk",    # Idiosyncratic tail risk (inverse)
    
    # === Momentum Factors (9) ===
    "resid_mom",         # Residual momentum
    "resid_mom_short",   # Short-term residual momentum
    "resid_mom_long",    # Long-term residual momentum
    "resid_mom_mix",     # Mixed momentum
    "srev",              # Short-term reversal
    "lrev",              # Long-term reversal
    "rel_sector_mom",    # Relative sector momentum
    "cross_momentum",    # Cross-sectional momentum
    "mtf_ic_momentum",   # Multi-timeframe IC momentum
    
    # === Technical/Price Factors (7) ===
    "breakout",          # Price breakout
    "prox_52w_high",     # Proximity to 52-week high
    "slope",             # EMA slope
    "ma_cloud",          # Moving average cloud
    "vol_breakout",      # Volatility breakout
    "calm_flow",         # Calm flow (low vol/high return)
    "hloc_close_position",  # Multi-timeframe HLOC position
    
    # === Volatility Factors (4) ===
    "vol_surprise",      # Volatility surprise
    "idio_jump_freq",    # Idiosyncratic jump frequency (inverse)
    "micro_noise",       # Microstructure noise (inverse)
    "vol_of_vol",        # Volatility of volatility (novel)
    
    # === Quality/Value Factors (4) ===
    "value_mom",         # Value momentum
    "quality_defensive", # Quality defensive
    "quality_score",     # Composite quality score
    "value_score",       # Composite value score
    
    # === Novel Factors from Neural Alpha (3) ===
    "efficiency_ratio",  # Kaufman Efficiency Ratio
    "attention_momentum",  # Volume-price attention signal
    "momentum_quality_ratio",  # Signal-to-noise of momentum
]
