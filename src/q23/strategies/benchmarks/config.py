"""q23.strategies.benchmarks.config

Benchmark strategy configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark strategy configuration."""
    ENABLED: bool = True
    AUTO_INCLUDE: bool = False  # Auto-include in comparisons
    MARKET_CAP_PROXY_METHOD: str = "close_volume"  # or "close_squared_volume"
    REBALANCE_FREQUENCY: str = "daily"
    MIN_DATE: str = "2005-01-01"  # Default benchmark start date
    
    # Index-specific settings
    INDEX_MIN_DATE: str = "2006-01-01"  # Per Quantiacs template example
    SUPPORTED_INDICES: Tuple[str, ...] = ("SP500", "NAS100")


# Index display name mapping
INDEX_DISPLAY_NAMES = {
    "SP500": "S&P 500",
    "NAS100": "NASDAQ 100",
}


benchmark_config = BenchmarkConfig()
