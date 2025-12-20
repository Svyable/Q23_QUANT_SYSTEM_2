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
from typing import Any, Dict, List, Optional


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


class TransactionCostScheme(Enum):
    """Available transaction cost models."""
    QUANTIACS_ATR = "quantiacs_atr"  # 5% of ATR(14) per position change (default)
    FLAT_BPS = "flat_bps"            # Fixed basis points on turnover
    PERCENTAGE = "percentage"         # Fixed percentage of trade value
    TIERED = "tiered"                # Tiered based on trade size
    CUSTOM = "custom"                # User-defined function


# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------

@dataclass
class PathConfig:
    """File/dir settings (safe to override via env vars)."""
    BASE_NAME: str = field(default_factory=lambda: os.getenv("Q23_BASE_NAME", "NASNYS_V4"))
    OUTPUT_ROOT: str = field(default_factory=lambda: os.getenv("Q23_OUTPUT_DIR", "outputs"))

    @property
    def wide_weights_pattern(self) -> str:
        return f"{self.BASE_NAME}_wide_*.csv"

    @property
    def factor_vectors_pattern(self) -> str:
        return f"{self.BASE_NAME}_factor_vectors_*.csv"


@dataclass
class TransactionCostConfig:
    """Transaction cost configuration.
    
    Default is Quantiacs contest model:
        TC = atr_multiplier × ATR(atr_window) × |ΔPosition|
        TC = 0.05 × ATR(14) × |ΔPosition|
    
    Attributes:
        scheme: Which TC model to use (default: QUANTIACS_ATR)
        atr_window: ATR lookback period for Quantiacs model (default: 14)
        atr_multiplier: Multiplier for ATR model (default: 0.05 = 5%)
        flat_bps: Basis points for FLAT_BPS scheme (default: 10.0)
        percentage: Percentage for PERCENTAGE scheme (default: 0.001 = 0.1%)
        min_cost_bps: Floor cost in bps (default: 0.0)
        max_cost_bps: Ceiling cost in bps (default: 100.0)
    """
    scheme: TransactionCostScheme = TransactionCostScheme.QUANTIACS_ATR
    atr_window: int = 14
    atr_multiplier: float = 0.05
    flat_bps: float = 10.0
    percentage: float = 0.001
    min_cost_bps: float = 0.0
    max_cost_bps: float = 100.0
    tiered_thresholds: list = field(default_factory=lambda: [
        (0.01, 5.0),   # < 1% of portfolio: 5 bps
        (0.05, 10.0),  # < 5% of portfolio: 10 bps
        (0.10, 15.0),  # < 10% of portfolio: 15 bps
        (1.0, 25.0),   # >= 10% of portfolio: 25 bps
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary for persistence."""
        return {
            "scheme": self.scheme.value,
            "atr_window": self.atr_window,
            "atr_multiplier": self.atr_multiplier,
            "flat_bps": self.flat_bps,
            "percentage": self.percentage,
            "min_cost_bps": self.min_cost_bps,
            "max_cost_bps": self.max_cost_bps,
            "tiered_thresholds": self.tiered_thresholds,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransactionCostConfig":
        """Deserialize config from dictionary."""
        scheme_val = data.get("scheme", "quantiacs_atr")
        try:
            scheme = TransactionCostScheme(scheme_val)
        except ValueError:
            scheme = TransactionCostScheme.QUANTIACS_ATR
        
        return cls(
            scheme=scheme,
            atr_window=int(data.get("atr_window", 14)),
            atr_multiplier=float(data.get("atr_multiplier", 0.05)),
            flat_bps=float(data.get("flat_bps", 10.0)),
            percentage=float(data.get("percentage", 0.001)),
            min_cost_bps=float(data.get("min_cost_bps", 0.0)),
            max_cost_bps=float(data.get("max_cost_bps", 100.0)),
            tiered_thresholds=data.get("tiered_thresholds", [
                (0.01, 5.0), (0.05, 10.0), (0.10, 15.0), (1.0, 25.0)
            ]),
        )
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.atr_window < 1:
            raise ValueError(f"atr_window must be >= 1, got {self.atr_window}")
        if self.atr_multiplier < 0:
            raise ValueError(f"atr_multiplier must be >= 0, got {self.atr_multiplier}")
        if self.flat_bps < 0:
            raise ValueError(f"flat_bps must be >= 0, got {self.flat_bps}")
        if self.min_cost_bps > self.max_cost_bps:
            raise ValueError(f"min_cost_bps ({self.min_cost_bps}) > max_cost_bps ({self.max_cost_bps})")
    
    @property
    def effective_bps(self) -> float:
        """Return effective BPS for backward compatibility.
        
        For ATR scheme, returns a rough estimate based on typical ATR values.
        For other schemes, returns the configured BPS.
        """
        if self.scheme == TransactionCostScheme.FLAT_BPS:
            return self.flat_bps
        elif self.scheme == TransactionCostScheme.PERCENTAGE:
            return self.percentage * 10000
        elif self.scheme == TransactionCostScheme.QUANTIACS_ATR:
            # Rough estimate: typical ATR ~1-2% of price, so 5% of 1.5% ≈ 7.5 bps
            # This is just for backward-compat display; actual TC uses ATR
            return self.atr_multiplier * 150  # ~7.5 bps for default 0.05
        else:
            return self.flat_bps


@dataclass
class StrategyConfig:
    """Backtest/strategy parameters (changing these changes results)."""

    # Data / Universe
    MIN_DATE: str = "2024-01-01"  # Updated for faster validation; can be overridden per strategy
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
    # DEPRECATED: Use GlobalConfig.tc instead. Kept for backward compatibility.
    TC_BASIS_POINTS: float = 10.0
    MIN_REBALANCE_THRESHOLD: float = 0.02
    
    # Legacy TC_BPS alias (used by engine.py)
    @property
    def TC_BPS(self) -> float:
        """Alias for TC_BASIS_POINTS for backward compatibility."""
        return self.TC_BASIS_POINTS

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
class BenchmarkConfig:
    """Benchmark strategy configuration."""
    ENABLED: bool = True
    AUTO_INCLUDE: bool = False  # Auto-include in comparisons
    MARKET_CAP_PROXY_METHOD: str = "close_volume"  # or "close_squared_volume"
    REBALANCE_FREQUENCY: str = "daily"
    MIN_DATE: str = "2005-01-01"  # Default benchmark start date


@dataclass
class GlobalConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    tc: TransactionCostConfig = field(default_factory=TransactionCostConfig)


# Singleton instance
cfg = GlobalConfig()
cfg.strategy.validate()
cfg.tc.validate()
