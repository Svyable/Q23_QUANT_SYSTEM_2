"""q23.strategies.benchmarks.engine

Benchmark computation engine for equal-weighted and market-cap weighted indices.

Supports both exchange-based benchmarks (NYSE, NASDAQ) and index-based benchmarks
(SP500, NAS100).
"""

from __future__ import annotations

from typing import Optional, Tuple

try:
    import xarray as xr  # type: ignore
except ImportError:
    xr = None

from q23.strategy.benchmark_data import (
    load_market_cap_data,
    compute_equal_weighted_returns,
    compute_market_cap_weighted_returns,
    # Index-specific functions
    load_index_data,
    load_index_market_cap_data,
    compute_index_equal_weighted_returns,
    compute_index_market_cap_weighted_returns,
)
from q23.strategy.data_loader import MarketDataBundle
from q23.strategies.benchmarks.config import benchmark_config


class BenchmarkEngine:
    """Engine for computing benchmark portfolio weights and returns."""
    
    def __init__(self, proxy_method: Optional[str] = None):
        """Initialize benchmark engine.
        
        Args:
            proxy_method: Market cap proxy method (overrides config if provided)
        """
        self.proxy_method = proxy_method or benchmark_config.MARKET_CAP_PROXY_METHOD
    
    def compute_equal_weighted_benchmark(
        self,
        bundle: MarketDataBundle,
        exchange: str,
    ) -> Tuple["xr.DataArray", "xr.DataArray", "xr.DataArray"]:
        """Compute equal-weighted benchmark portfolio.
        
        Args:
            bundle: MarketDataBundle containing market data
            exchange: Exchange code ("NYS" for NYSE, "NAS" for NASDAQ)
        
        Returns:
            Tuple of (weights, portfolio_returns, budget) DataArrays
        """
        if xr is None:
            raise ImportError("xarray is required for benchmark computation")
        
        weights, portfolio_returns = compute_equal_weighted_returns(
            bundle=bundle,
            exchange=exchange,
        )
        
        # Budget is always 1.0 for benchmarks (no leverage)
        budget = xr.ones_like(portfolio_returns)
        
        return weights, portfolio_returns, budget
    
    def compute_market_cap_weighted_benchmark(
        self,
        bundle: MarketDataBundle,
        exchange: str,
    ) -> Tuple["xr.DataArray", "xr.DataArray", "xr.DataArray"]:
        """Compute market-cap weighted benchmark portfolio.
        
        Args:
            bundle: MarketDataBundle containing market data
            exchange: Exchange code ("NYS" for NYSE, "NAS" for NASDAQ)
        
        Returns:
            Tuple of (weights, portfolio_returns, budget) DataArrays
            Returns zero arrays if market cap data unavailable
        """
        if xr is None:
            raise ImportError("xarray is required for benchmark computation")
        
        # Load market cap data (with proxy fallback)
        market_cap = load_market_cap_data(
            bundle=bundle,
            proxy_method=self.proxy_method,
        )
        
        if market_cap is None:
            # Return zero arrays if market cap unavailable
            empty_weights = xr.zeros_like(bundle.returns)
            empty_returns = xr.zeros_like(bundle.returns.isel(asset=0))
            empty_budget = xr.zeros_like(empty_returns)
            return empty_weights, empty_returns, empty_budget
        
        weights, portfolio_returns = compute_market_cap_weighted_returns(
            bundle=bundle,
            market_cap=market_cap,
            exchange=exchange,
        )
        
        # Budget is always 1.0 for benchmarks (no leverage)
        budget = xr.ones_like(portfolio_returns)
        
        return weights, portfolio_returns, budget
    
    # -------------------------------------------------------------------------
    # Index-based benchmark computation (SP500, NAS100)
    # -------------------------------------------------------------------------
    
    def compute_index_equal_weighted_benchmark(
        self,
        index_type: str,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
    ) -> Tuple[MarketDataBundle, "xr.DataArray", "xr.DataArray", "xr.DataArray"]:
        """Compute equal-weighted benchmark for an index (SP500 or NAS100).
        
        Args:
            index_type: Index type ("SP500" or "NAS100")
            min_date: Optional minimum date (defaults to config INDEX_MIN_DATE)
            max_date: Optional maximum date
        
        Returns:
            Tuple of (bundle, weights, portfolio_returns, budget)
            - bundle: MarketDataBundle containing index constituent data
            - weights: Asset weights DataArray (time, asset)
            - portfolio_returns: Portfolio returns DataArray (time,)
            - budget: Budget DataArray (time,) - always 1.0 for benchmarks
        """
        if xr is None:
            raise ImportError("xarray is required for benchmark computation")
        
        use_min_date = min_date or benchmark_config.INDEX_MIN_DATE
        
        # Load index-specific data
        bundle = load_index_data(
            index_type=index_type,
            min_date=use_min_date,
            max_date=max_date,
        )
        
        # Compute equal-weighted returns for all index constituents
        weights, portfolio_returns = compute_index_equal_weighted_returns(bundle)
        
        # Budget is always 1.0 for benchmarks (no leverage)
        budget = xr.ones_like(portfolio_returns)
        
        return bundle, weights, portfolio_returns, budget
    
    def compute_index_market_cap_weighted_benchmark(
        self,
        index_type: str,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
    ) -> Tuple[MarketDataBundle, "xr.DataArray", "xr.DataArray", "xr.DataArray"]:
        """Compute market-cap weighted benchmark for an index (SP500 or NAS100).
        
        Args:
            index_type: Index type ("SP500" or "NAS100")
            min_date: Optional minimum date (defaults to config INDEX_MIN_DATE)
            max_date: Optional maximum date
        
        Returns:
            Tuple of (bundle, weights, portfolio_returns, budget)
            - bundle: MarketDataBundle containing index constituent data
            - weights: Asset weights DataArray (time, asset)
            - portfolio_returns: Portfolio returns DataArray (time,)
            - budget: Budget DataArray (time,) - always 1.0 for benchmarks
        """
        if xr is None:
            raise ImportError("xarray is required for benchmark computation")
        
        use_min_date = min_date or benchmark_config.INDEX_MIN_DATE
        
        # Load index-specific data
        bundle = load_index_data(
            index_type=index_type,
            min_date=use_min_date,
            max_date=max_date,
        )
        
        # Load market cap data (with proxy fallback)
        market_cap = load_index_market_cap_data(
            bundle=bundle,
            proxy_method=self.proxy_method,
        )
        
        # Compute market-cap weighted returns for all index constituents
        weights, portfolio_returns = compute_index_market_cap_weighted_returns(
            bundle=bundle,
            market_cap=market_cap,
        )
        
        # Budget is always 1.0 for benchmarks (no leverage)
        budget = xr.ones_like(portfolio_returns)
        
        return bundle, weights, portfolio_returns, budget
