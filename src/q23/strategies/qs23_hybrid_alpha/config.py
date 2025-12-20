from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class QS23HybridAlphaConfig:
    MIN_DATE: str = "2020-01-01"
    SUBMISSION_START_DATE: str = "2024-01-01"
    EXCHANGE: str = "NAS"
    
    TOPN: int = 11
    MAX_POS: float = 0.10
    MIN_POS: float = 0.002
    TARGET_VOL: float = 0.25
    LEV_CAP: float = 1.5
    LEV_MIN: float = 0.3
    
    BETA_WIN: int = 162
    IDIO_WIN: int = 63
    DOWN_WIN: int = 63
    CORR_WIN: int = 63
    ADV_WIN_LONG: int = 63
    ADV_WIN_SHORT: int = 5
    IC_LAMBDA: float = 0.93
    
    MOM_WIN: int = 84
    REV_WIN: int = 5
    DON_WIN: int = 63
    EMA_FAST: int = 20
    EMA_SLOW: int = 42
    ATR_WIN: int = 42
    
    SCORE_SMOOTH_WIN: int = 3
    WEIGHT_SMOOTH_ALPHA: float = 0.30
    DD_WIN: int = 252
    RISK_OFF_STRETCH: float = 0.60
    RISK_OFF_FLOOR: float = 0.50
    
    TC_BPS: float = 2.0
    EPS: float = 1e-12


QS23_FACTORS: List[str] = [
    "inv_vol",
    "inv_idio",
    "inv_down",
    "low_corr",
    "low_beta",
    "liquidity",
    "resid_mom",
    "srev",
    "breakout",
    "slope",
    "calm_flow",
]


qs23_config = QS23HybridAlphaConfig()
