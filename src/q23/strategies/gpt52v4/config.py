from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from q23.strategy.factors import (
    ALPHA_FACTORS,
    DEFENSIVE_FACTORS,
    QUALITY_FACTORS,
    OU_FACTORS,
    GLFT_FACTORS,
)


@dataclass(frozen=True)
class GPT52V4Config:
    """GPT5.2 v4: adaptive sleeve ensemble (Sharpe + DD control)."""

    MIN_DATE: str = "2020-01-01"
    EXCHANGES: Tuple[str, ...] = ("NAS", "NYS")

    # Portfolio construction
    LONG_SEATS: int = 16
    SHORT_SEATS: int = 6
    LONG_ONLY: bool = False
    TOPN_BASE: int = 22
    TOPN_VOLATILE: int = 18
    MAX_POS: float = 0.08
    MIN_POS: float = 0.0015

    # Vol / leverage caps
    TARGET_VOL_BASE: float = 0.15
    TARGET_VOL_VOLATILE: float = 0.11
    LEV_CAP: float = 1.55
    LEV_MIN: float = 0.40

    TC_BPS: float = 8.0

    # Smoothing / conviction
    WEIGHT_SMOOTH_ALPHA: float = 0.34
    SCORE_SMOOTH_WIN: int = 4
    SOFTMAX_TILT_ALPHA: float = 0.88

    # Drawdown throttle + overlay
    DD_WIN: int = 126
    RISK_OFF_STRETCH: float = 0.70
    RISK_OFF_FLOOR: float = 0.50
    VOL_OVERLAY_WIN: int = 63
    VOL_OVERLAY_MAX: float = 0.22

    # Factor IC weighting (within sleeve)
    IC_LAMBDA: float = 0.95
    FACTOR_CORR_PENALTY: float = 0.30
    FACTOR_MIN_DIVERSIFICATION: float = 0.25
    FACTOR_MIN_WEIGHT: float = 0.012

    # Sleeve weighting (across sleeves)
    SLEEVE_IC_LAMBDA: float = 0.94
    SLEEVE_CORR_PENALTY: float = 0.35
    SLEEVE_MIN_WEIGHT: float = 0.12

    # Stress prior (boost defensive/MR during drawdowns)
    STRESS_DD_CAP: float = 0.10  # dd=10% -> stress=1
    STRESS_MOM_MULT: float = 0.60
    STRESS_MR_MULT: float = 0.40
    STRESS_MS_MULT: float = 0.20
    STRESS_DEF_MULT: float = 0.80

    # Horizons per sleeve
    MOM_HORIZONS: Tuple[int, ...] = (21, 63)
    MR_HORIZONS: Tuple[int, ...] = (5, 21)
    MS_HORIZONS: Tuple[int, ...] = (5, 21)
    DEF_HORIZONS: Tuple[int, ...] = (21,)

    # Windows (FactorParams)
    BETA_WIN: int = 126
    IDIO_WIN: int = 52
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
    EMA_FAST: int = 16
    EMA_SLOW: int = 42
    ATR_WIN: int = 28
    SECTOR_MOM_WIN: int = 63
    VALUE_WIN: int = 240

    # Neural windows
    EFFICIENCY_WIN: int = 21
    ATTENTION_WIN: int = 10
    DISPOSITION_WIN: int = 63
    SKEW_WIN: int = 63
    KURT_WIN: int = 63
    VOV_WIN: int = 21
    PERSISTENCE_WIN: int = 42

    # OU windows
    OU_SHORT_WIN: int = 21
    OU_MED_WIN: int = 63
    OU_HALFLIFE_MIN: int = 2
    OU_HALFLIFE_MAX: int = 42
    OU_ZSCORE_CLIP: float = 3.0

    # GLFT windows
    OFI_SHORT_WIN: int = 5
    OFI_MED_WIN: int = 21
    OFI_LONG_WIN: int = 63
    VPIN_WIN: int = 50
    SPREAD_EST_WIN: int = 21
    IMPACT_WIN: int = 42
    INVENTORY_DECAY: float = 0.95

    EPS: float = 1e-12

    def __post_init__(self) -> None:
        object.__setattr__(self, "EXCHANGES", tuple(self.EXCHANGES))
        object.__setattr__(self, "MOM_HORIZONS", tuple(self.MOM_HORIZONS))
        object.__setattr__(self, "MR_HORIZONS", tuple(self.MR_HORIZONS))
        object.__setattr__(self, "MS_HORIZONS", tuple(self.MS_HORIZONS))
        object.__setattr__(self, "DEF_HORIZONS", tuple(self.DEF_HORIZONS))


gpt52v4_config = GPT52V4Config()


MOM_SLEEVE: List[str] = list(set(ALPHA_FACTORS) | {
    "resid_mom_short",
    "resid_mom_long",
    "lrev",
    "rel_sector_mom",
    "cross_momentum",
    "vol_breakout",
    "ma_cloud",
    "mtf_ic_momentum",
    "trend_consistency",
    "momentum_acceleration",
    "hloc_close_position",
})

MR_SLEEVE: List[str] = list(set(OU_FACTORS) | {
    "srev",
    "lrev",
    "mean_reversion_speed",
    "disposition_alpha",
})

MS_SLEEVE: List[str] = list(set(GLFT_FACTORS) | {
    "information_flow",
    "efficiency_ratio",
    "liquidity_momentum",
    "micro_noise",
    "amihud_inv",
})

DEF_SLEEVE: List[str] = list(set(DEFENSIVE_FACTORS) | set(QUALITY_FACTORS) | {
    "quality_score",
    "value_score",
    "value_mom",
    "quality_defensive",
    "vol_of_vol",
    "skewness_factor",
    "kurtosis_factor",
    "regime_momentum",
    "mtf_alignment",
})

GPT52V4_FACTORS: List[str] = sorted(set(MOM_SLEEVE) | set(MR_SLEEVE) | set(MS_SLEEVE) | set(DEF_SLEEVE))

