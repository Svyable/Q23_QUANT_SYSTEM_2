from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

try:
    import xarray as xr
except ImportError:
    xr = None


@dataclass
class StrategyConfig:
    name: str
    display_name: str
    version: str
    min_date: str
    exchanges: List[str]
    factors: List[str]
    topn: int
    max_pos: float
    min_pos: float
    target_vol: float
    lev_cap: float
    lev_min: float
    tc_bps: float
    description: str = ""
    long_only: bool = True
    long_seats: int = 20
    short_seats: int = 0
    weight_smooth_alpha: float = 0.30
    score_smooth_win: int = 3
    softmax_tilt_alpha: float = 0.85
    risk_off_stretch: float = 0.60
    risk_off_floor: float = 0.50
    dd_win: int = 252
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyArtifacts:
    weights: "xr.DataArray"
    budget: "xr.DataArray"
    factor_weights: "xr.DataArray"
    ic_raw: "xr.DataArray"
    ic_smooth: "xr.DataArray"
    composite_score: "xr.DataArray"
    F: "xr.DataArray"
    meta: Dict[str, Any]
    tag: str
    strategy_id: str


class StrategyBase(ABC):
    
    @classmethod
    @abstractmethod
    def strategy_id(cls) -> str:
        pass
    
    @property
    @abstractmethod
    def config(self) -> StrategyConfig:
        pass
    
    @abstractmethod
    def run(
        self,
        *,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        tag: Optional[str] = None,
        write_outputs: bool = True,
        force_live_data: bool = False,
    ) -> StrategyArtifacts:
        pass
    
    @abstractmethod
    def get_factors(self) -> Dict[str, "xr.DataArray"]:
        pass
    
    def get_output_dir(self) -> str:
        from q23.shared.config import cfg
        return f"{cfg.paths.OUTPUT_ROOT}/{self.strategy_id()}"
    
    def get_base_name(self) -> str:
        return self.strategy_id()


@dataclass
class StrategyInfo:
    strategy_id: str
    display_name: str
    version: str
    description: str
    min_date: str
    factors_count: int
    long_only: bool
