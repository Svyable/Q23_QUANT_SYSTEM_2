from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class NASNYSV4Config:
    MIN_DATE: str = "2020-01-01"
    EXCHANGES: List[str] = ("NAS", "NYS")

    TOPN_BASE: int = 20
    TOPN_VOLATILE: int = 15
    MAX_POS: float = 0.10
    MIN_POS: float = 0.002

    TARGET_VOL_BASE: float = 0.15
    TARGET_VOL_VOLATILE: float = 0.20
    LEV_CAP: float = 1.5
    LEV_MIN: float = 0.3

    TC_BPS: float = 10.0

    LONG_SEATS: int = 20
    SHORT_SEATS: int = 0
    LONG_ONLY: bool = True

    WEIGHT_SMOOTH_ALPHA: float = 0.30
    SCORE_SMOOTH_WIN: int = 3
    SOFTMAX_TILT_ALPHA: float = 0.85

    RISK_OFF_STRETCH: float = 0.60
    RISK_OFF_FLOOR: float = 0.50
    DD_WIN: int = 252

    EPS: float = 1e-12

    def __post_init__(self):
        object.__setattr__(self, "EXCHANGES", list(self.EXCHANGES))


v4_config = NASNYSV4Config()


V4_24_FACTORS: List[str] = [
    "inv_vol",
    "inv_idio",
    "inv_down",
    "low_corr",
    "low_beta",
    "liquidity",
    "amihud_inv",
    "resid_mom",
    "srev",
    "breakout",
    "slope",
    "calm_flow",
    "prox_52w_high",
    "resid_mom_mix",
    "vol_surprise",
    "vol_breakout",
    "ma_cloud",
    "idio_tail_risk",
    "idio_jump_freq",
    "beta_stability",
    "micro_noise",
    "value_mom",
    "quality_defensive",
    "rel_sector_mom",
]
