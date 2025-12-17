"""q23.shared.config

Central configuration definitions used by both Strategy and Dashboard.

Principles:
- Split backtest-affecting parameters (StrategyConfig) from UI-only parameters (DashboardConfig).
- Allow environment overrides for paths so the same code runs on multiple machines/CI.
- Keep validation close to the config to fail fast.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List


# -----------------------------------------------------------------------------
# Enums (shared definitions)
# -----------------------------------------------------------------------------

class PortfolioMode(Enum):
    """Whether the strategy can take short positions."""
    LONG_ONLY = "long_only"
    LONG_SHORT = "long_short"


class BudgetMode(Enum):
    """How the strategy targets gross exposure."""
    UNIT = "unit"                 # Fixed 100% gross
    USE_LEVERAGE = "use_leverage" # Dynamic vol-targeting leverage
    FIXED = "fixed"               # Fixed gross target (e.g. 1.5x)


class SideSplitMode(Enum):
    """How capital is divided between long and short books."""
    FIXED = "fixed"          # e.g., 60/40 split
    PROP_MASS = "prop_mass"  # proportional to signal mass
    IC_WEIGHTED = "ic_weighted"  # reserved (treat like PROP_MASS for now)


# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------

@dataclass
class PathConfig:
    """File/dir settings (safe to override via env vars)."""
    BASE_NAME: str = field(default_factory=lambda: os.getenv("Q23_BASE_NAME", "NASNYS_V4"))
    OUTPUT_ROOT: str = field(default_factory=lambda: os.getenv("Q23_OUTPUT_DIR", "outputs/NASNYS_V4"))

    @property
    def wide_weights_pattern(self) -> str:
        return f"{self.BASE_NAME}_wide_*.csv"

    @property
    def factor_vectors_pattern(self) -> str:
        return f"{self.BASE_NAME}_factor_vectors_*.csv"


@dataclass
class StrategyConfig:
    """Backtest/strategy parameters (changing these changes results)."""

    # Data / Universe
    MIN_DATE: str = "2004-01-01"
    SUBMISSION_START_DATE: str = "2005-01-01"
    EXCHANGES: List[str] = field(default_factory=lambda: ["NAS", "NYS"])
    EPS: float = 1e-12

    # Constraints
    PORTFOLIO_MODE: PortfolioMode = PortfolioMode.LONG_ONLY
    MAX_POS: float = 0.10
    MIN_POS: float = 0.002

    # Budgeting
    PM_TARGET_MODE: BudgetMode = BudgetMode.UNIT
    PM_GROSS_TARGET: float = 1.00

    # Long/short specifics
    SIDE_SPLIT_MODE: SideSplitMode = SideSplitMode.FIXED
    FIXED_LONG_FRAC: float = 0.60
    LONG_SEATS: int = 12
    SHORT_SEATS: int = 8

    # Conviction & smoothing
    USE_EQUAL_TOPK: bool = False
    TOPN_BASE: int = 20
    TOPN_VOLATILE: int = 15
    SCORE_SMOOTH_WIN: int = 3
    WEIGHT_SMOOTH_ALPHA: float = 0.30
    SOFTMAX_TILT_ALPHA: float = 0.85

    # Risk / leverage
    TARGET_VOL_BASE: float = 0.15
    TARGET_VOL_VOLATILE: float = 0.10
    LEV_CAP: float = 1.5
    LEV_MIN: float = 0.3

    # Transaction costs / rebalance control
    TC_BASIS_POINTS: float = 10.0
    MIN_REBALANCE_THRESHOLD: float = 0.02

    def validate(self) -> None:
        if self.MIN_POS >= self.MAX_POS:
            raise ValueError(f"MIN_POS ({self.MIN_POS}) must be < MAX_POS ({self.MAX_POS})")
        if self.PM_GROSS_TARGET <= 0:
            raise ValueError("PM_GROSS_TARGET must be positive")
        if self.PORTFOLIO_MODE == PortfolioMode.LONG_SHORT:
            if self.LONG_SEATS + self.SHORT_SEATS <= 0:
                raise ValueError("Must have at least one seat in Long/Short mode")
            if not (0.0 <= self.FIXED_LONG_FRAC <= 1.0):
                raise ValueError("FIXED_LONG_FRAC must be within [0, 1]")
        if self.TOPN_BASE <= 0:
            raise ValueError("TOPN_BASE must be > 0")


@dataclass
class DashboardConfig:
    """UI defaults (do not change backtest math)."""
    DEFAULT_LOOKBACK_YEARS: int = 1
    TRADING_DAYS_1M: int = 21
    TRADING_DAYS_12M: int = 252
    COLOR_UP: str = "#2ecc71"
    COLOR_DOWN: str = "#e74c3c"


@dataclass
class GlobalConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


# Singleton instance
cfg = GlobalConfig()
cfg.strategy.validate()
