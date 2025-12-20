from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Q23ComposerV1Config:
    """
    Q23_COMPOSERv1 Configuration
    
    Optimized for maximum Sharpe and Sortino ratios through:
    - Enhanced factor selection (30+ factors)
    - Sophisticated IC weighting with regime awareness
    - Advanced risk management focused on downside protection
    - Dynamic portfolio construction with adaptive position sizing
    """
    MIN_DATE: str = "2020-01-01"
    EXCHANGES: List[str] = ("NAS", "NYS")  # Both exchanges for diversification
    
    # Portfolio construction - optimized for risk-adjusted returns
    TOPN_BASE: int = 25  # More positions for diversification
    TOPN_VOLATILE: int = 20
    MAX_POS: float = 0.08  # Slightly lower max position for risk control
    MIN_POS: float = 0.001
    
    # Volatility targeting - balanced for Sharpe optimization
    TARGET_VOL_BASE: float = 0.18  # Moderate volatility target
    TARGET_VOL_VOLATILE: float = 0.22
    LEV_CAP: float = 1.6  # Higher leverage cap for alpha capture
    LEV_MIN: float = 0.4
    
    # Transaction costs - realistic for Quantiacs
    TC_BPS: float = 5.0  # Moderate transaction cost assumption
    
    # Long-only configuration
    LONG_SEATS: int = 25
    SHORT_SEATS: int = 0
    LONG_ONLY: bool = True
    
    # Smoothing parameters - optimized for signal quality
    WEIGHT_SMOOTH_ALPHA: float = 0.35  # Faster adaptation
    SCORE_SMOOTH_WIN: int = 5  # Longer smoothing for stability
    SOFTMAX_TILT_ALPHA: float = 0.90  # Stronger tilt toward top signals
    
    # Risk management - Sortino-focused (downside protection)
    RISK_OFF_STRETCH: float = 0.70  # More aggressive risk reduction
    RISK_OFF_FLOOR: float = 0.45
    DD_WIN: int = 126  # Shorter window for faster response
    
    # IC weighting - enhanced for regime awareness
    IC_LAMBDA: float = 0.94  # Higher persistence
    IC_MIN_WEIGHT: float = 0.01  # Minimum factor weight
    IC_DECAY_HALFLIFE: int = 63  # Decay for stale factors
    
    # Factor computation windows - optimized
    BETA_WIN: int = 126  # Shorter for faster adaptation
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
    
    EPS: float = 1e-12
    
    def __post_init__(self):
        object.__setattr__(self, "EXCHANGES", list(self.EXCHANGES))


composer_v1_config = Q23ComposerV1Config()


# Enhanced factor set - 32 factors targeting multiple alpha sources
COMPOSER_V1_FACTORS: List[str] = [
    # === Risk/Defensive Factors (8) ===
    "inv_vol",           # Inverse volatility - core defensive
    "inv_idio",          # Inverse idiosyncratic volatility
    "inv_down",          # Inverse downside volatility (Sortino-focused)
    "low_corr",          # Low correlation to market
    "low_beta",          # Low beta
    "liquidity",         # High liquidity
    "amihud_inv",        # Inverse Amihud illiquidity
    "beta_stability",    # Stable beta (lower risk)
    
    # === Momentum Factors (7) ===
    "resid_mom",         # Residual momentum
    "resid_mom_short",   # Short-term residual momentum
    "resid_mom_long",    # Long-term residual momentum
    "resid_mom_mix",     # Mixed momentum
    "srev",              # Short-term reversal
    "lrev",              # Long-term reversal
    "rel_sector_mom",    # Relative sector momentum
    
    # === Technical/Price Factors (6) ===
    "breakout",          # Price breakout
    "prox_52w_high",     # Proximity to 52-week high
    "slope",             # EMA slope
    "ma_cloud",          # Moving average cloud
    "vol_breakout",      # Volatility breakout
    "calm_flow",         # Calm flow (low vol/high return)
    
    # === Volatility Factors (4) ===
    "vol_surprise",      # Volatility surprise
    "idio_tail_risk",    # Idiosyncratic tail risk
    "idio_jump_freq",    # Idiosyncratic jump frequency
    "micro_noise",       # Microstructure noise
    
    # === Quality/Value Factors (4) ===
    "value_mom",         # Value momentum
    "quality_defensive", # Quality defensive
    "quality_score",     # Composite quality score
    "value_score",       # Composite value score
    
    # === Cross-Sectional Factors (3) ===
    "cross_momentum",    # Cross-sectional momentum
    "cross_volatility",  # Cross-sectional volatility rank
    "cross_liquidity",   # Cross-sectional liquidity rank
]
