from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class GTP51MAXConfig:
    """Configuration tuned for max Sharpe under vol/DD guardrails."""

    MIN_DATE: str = "2020-01-01"
    EXCHANGES: Tuple[str, ...] = ("NAS", "NYS")

    # Portfolio construction
    TOPN_BASE: int = 28
    TOPN_VOLATILE: int = 22
    MAX_POS: float = 0.07
    MIN_POS: float = 0.0015

    # Vol targeting
    TARGET_VOL_BASE: float = 0.16
    TARGET_VOL_VOLATILE: float = 0.18
    LEV_CAP: float = 1.60
    LEV_MIN: float = 0.40

    # Transaction costs
    TC_BPS: float = 6.0

    # Book structure
    LONG_SEATS: int = 28
    SHORT_SEATS: int = 0
    LONG_ONLY: bool = True

    # Smoothing / conviction
    WEIGHT_SMOOTH_ALPHA: float = 0.32
    SCORE_SMOOTH_WIN: int = 5
    SOFTMAX_TILT_ALPHA: float = 0.92

    # Risk throttle
    RISK_OFF_STRETCH: float = 0.65
    RISK_OFF_FLOOR: float = 0.55
    DD_WIN: int = 126

    # IC weighting
    IC_LAMBDA: float = 0.94
    CORR_PENALTY: float = 0.35
    MIN_DIVERSIFICATION: float = 0.25

    # Factor windows
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
    EMA_FAST: int = 16
    EMA_SLOW: int = 48
    ATR_WIN: int = 28
    SECTOR_MOM_WIN: int = 63
    VALUE_WIN: int = 240

    # Risk overlay
    VOL_OVERLAY_WIN: int = 63
    VOL_OVERLAY_MAX: float = 0.22

    EPS: float = 1e-12

    def __post_init__(self) -> None:
        object.__setattr__(self, "EXCHANGES", tuple(self.EXCHANGES))


gtp51max_config = GTP51MAXConfig()

GTP51MAX_FACTORS: List[str] = [
    # Defensive / Liquidity
    "inv_vol",
    "inv_idio",
    "inv_down",
    "low_corr",
    "low_beta",
    "liquidity",
    "amihud_inv",
    "beta_stability",
    "micro_noise",
    "idio_tail_risk",
    # Momentum / Multi-timeframe
    "resid_mom",
    "resid_mom_short",
    "resid_mom_long",
    "resid_mom_mix",
    "mtf_alignment",
    "mtf_alignment_strength",
    "trend_consistency",
    "momentum_acceleration",
    "mtf_ic_momentum",
    "ret_1m",
    "ret_1q",
    "ret_2_12m",
    # Mean reversion
    "srev",
    # Technical / Breakout
    "breakout",
    "slope",
    "ma_cloud",
    "vol_breakout",
    "calm_flow",
    "prox_52w_high",
    "hloc_range_expansion",
    "hloc_close_position",
    # Quality / Value / Cross-sectional
    "value_mom",
    "quality_defensive",
    "quality_score",
    "value_score",
    "rel_sector_mom",
    "cross_liquidity",
    "cross_volatility",
]
