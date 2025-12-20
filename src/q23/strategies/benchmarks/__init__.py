"""q23.strategies.benchmarks

Benchmark strategies for NYSE and NASDAQ indices, as well as
index-based benchmarks (S&P 500, NASDAQ 100).

Auto-registers benchmark strategies when imported.
"""

from q23.strategies.benchmarks.strategies import (
    # Exchange-based benchmarks
    NYSEEqualWeightBenchmark,
    NYSEMarketCapBenchmark,
    NASDAQEqualWeightBenchmark,
    NASDAQMarketCapBenchmark,
    # Index-based benchmarks (SP500, NAS100)
    SP500EqualWeightBenchmark,
    SP500MarketCapBenchmark,
    NAS100EqualWeightBenchmark,
    NAS100MarketCapBenchmark,
)

# Importing the strategies registers them with StrategyRegistry
__all__ = [
    # Exchange-based benchmarks
    "NYSEEqualWeightBenchmark",
    "NYSEMarketCapBenchmark",
    "NASDAQEqualWeightBenchmark",
    "NASDAQMarketCapBenchmark",
    # Index-based benchmarks
    "SP500EqualWeightBenchmark",
    "SP500MarketCapBenchmark",
    "NAS100EqualWeightBenchmark",
    "NAS100MarketCapBenchmark",
]
