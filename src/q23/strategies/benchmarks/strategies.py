"""q23.strategies.benchmarks.strategies

Benchmark strategy implementations for NYSE and NASDAQ indices,
as well as index-based benchmarks (SP500, NAS100).
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

try:
    import xarray as xr  # type: ignore
except ImportError:
    xr = None

import numpy as np

from q23.strategies.base import StrategyBase, StrategyConfig, StrategyArtifacts
from q23.strategies.registry import StrategyRegistry
from q23.strategies.benchmarks.engine import BenchmarkEngine
from q23.strategies.benchmarks.config import benchmark_config, INDEX_DISPLAY_NAMES
from q23.strategy.data_loader import load_market_data
from q23.strategy.engine import _compute_portfolio_diag
from q23.strategy.benchmark_data import get_exchange_filtered_assets


def _create_empty_factor_arrays(
    time_coords: "xr.DataArray",
    asset_coords: "xr.DataArray",
    n_factors: int = 1,
) -> Dict[str, "xr.DataArray"]:
    """Create empty factor arrays for benchmarks (benchmarks don't have factors)."""
    if xr is None:
        raise ImportError("xarray required")
    
    factor_coords = [f"benchmark_factor_{i}" for i in range(n_factors)]
    
    return {
        "factor_weights": xr.DataArray(
            np.zeros((len(time_coords), n_factors)),
            dims=["time", "factor"],
            coords={"time": time_coords, "factor": factor_coords}
        ),
        "ic_raw": xr.DataArray(
            np.zeros((len(time_coords), n_factors)),
            dims=["time", "factor"],
            coords={"time": time_coords, "factor": factor_coords}
        ),
        "ic_smooth": xr.DataArray(
            np.zeros((len(time_coords), n_factors)),
            dims=["time", "factor"],
            coords={"time": time_coords, "factor": factor_coords}
        ),
        "composite_score": xr.DataArray(
            np.zeros((len(time_coords), len(asset_coords))),
            dims=["time", "asset"],
            coords={"time": time_coords, "asset": asset_coords}
        ),
        "F": xr.DataArray(
            np.zeros((n_factors, len(time_coords), len(asset_coords))),
            dims=["factor", "time", "asset"],
            coords={"factor": factor_coords, "time": time_coords, "asset": asset_coords}
        ),
    }


class BaseBenchmarkStrategy(StrategyBase):
    """Base class for benchmark strategies."""
    
    def __init__(self, exchange: str, weighting: str):
        """Initialize benchmark strategy.
        
        Args:
            exchange: Exchange code ("NYS" for NYSE, "NAS" for NASDAQ)
            weighting: Weighting method ("equal_weight" or "market_cap")
        """
        self.exchange = exchange.upper()
        self.weighting = weighting
        self._engine = BenchmarkEngine()
        self._factors: Dict[str, "xr.DataArray"] = {}
    
    @property
    def config(self) -> StrategyConfig:
        exchange_name = "NYSE" if self.exchange == "NYS" else "NASDAQ"
        weighting_name = "Equal-Weighted" if self.weighting == "equal_weight" else "Market-Cap Weighted"
        
        return StrategyConfig(
            name=self.strategy_id(),
            display_name=f"{exchange_name} {weighting_name} Benchmark",
            version="1.0",
            min_date=benchmark_config.MIN_DATE,
            exchanges=[self.exchange],
            factors=[],  # Benchmarks don't have factors
            topn=0,
            max_pos=1.0,
            min_pos=0.0,
            target_vol=0.0,
            lev_cap=1.0,
            lev_min=1.0,
            tc_bps=0.0,  # Benchmarks assume no transaction costs
            description=f"{exchange_name} {weighting_name} benchmark index",
            long_only=True,
            long_seats=0,
            short_seats=0,
        )
    
    def get_factors(self) -> Dict[str, "xr.DataArray"]:
        return self._factors
    
    def run(
        self,
        *,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        tag: Optional[str] = None,
        write_outputs: bool = False,  # Benchmarks typically don't write outputs
    ) -> StrategyArtifacts:
        if xr is None:
            raise ImportError("xarray required")
        
        use_min_date = min_date or benchmark_config.MIN_DATE
        
        # Load market data for the exchange
        bundle = load_market_data(
            min_date=use_min_date,
            max_date=max_date,
            exchanges=[self.exchange],
            pinned=None,
        )
        
        # Compute benchmark weights and returns
        if self.weighting == "equal_weight":
            weights, portfolio_returns, budget = self._engine.compute_equal_weighted_benchmark(
                bundle=bundle,
                exchange=self.exchange,
            )
        elif self.weighting == "market_cap":
            weights, portfolio_returns, budget = self._engine.compute_market_cap_weighted_benchmark(
                bundle=bundle,
                exchange=self.exchange,
            )
        else:
            raise ValueError(f"Unknown weighting method: {self.weighting}")
        
        # Create empty factor arrays (benchmarks don't have factors)
        time_coords = weights.time
        asset_coords = weights.asset
        factor_arrays = _create_empty_factor_arrays(time_coords, asset_coords, n_factors=1)
        
        # Compute portfolio diagnostics
        # For benchmarks, portfolio returns are already computed in the engine
        # We need to align weights with returns and compute diag
        # Filter returns by exchange if needed
        exchange_assets = get_exchange_filtered_assets(bundle, self.exchange)
        
        if exchange_assets:
            asset_mask = xr.DataArray(
                [aid in exchange_assets for aid in bundle.asset_ids],
                dims=["asset"],
                coords={"asset": bundle.returns.asset}
            )
            returns_filtered = bundle.returns.where(asset_mask, 0.0)
        else:
            # If we can't filter by exchange, use all returns
            # (bundle was already filtered by exchange in load_market_data)
            returns_filtered = bundle.returns
        
        # Align weights with returns
        weights_aligned = weights.reindex(
            time=returns_filtered.time,
            asset=returns_filtered.asset,
            method="ffill"
        ).fillna(0.0)
        
        diag = _compute_portfolio_diag(
            final_weights=weights_aligned,
            returns=returns_filtered,
            tc_bps=0.0,  # Benchmarks assume no transaction costs
        )
        
        use_tag = tag or datetime.now().strftime("%Y-%m-%d")
        
        meta = {
            "strategy_id": self.strategy_id(),
            "strategy_version": self.config.version,
            "display_name": self.config.display_name,
            "tag": use_tag,
            "run_timestamp": datetime.now().isoformat(),
            "date_range": [
                str(weights.time.values[0])[:10] if len(weights.time) > 0 else "",
                str(weights.time.values[-1])[:10] if len(weights.time) > 0 else "",
            ],
            "exchange": self.exchange,
            "weighting": self.weighting,
            "is_benchmark": True,
        }
        
        # Write outputs if requested
        if write_outputs:
            from q23.strategy.outputs import OutputWriter
            
            output_dir = self.get_output_dir()
            writer = OutputWriter(
                base_name=self.strategy_id(),
                output_dir=output_dir,
            )
            
            writer.write_wide_weights(weights, tag=use_tag)
            writer.write_budget(budget, tag=use_tag)
            writer.write_factor_weights(factor_arrays["factor_weights"], tag=use_tag)
            writer.write_ic(factor_arrays["ic_raw"], factor_arrays["ic_smooth"], tag=use_tag)
            
            if diag is not None and not diag.empty:
                writer.write_portfolio_diag(diag, tag=use_tag)
            
            writer.write_meta(meta, tag=use_tag)
        else:
            # Only store _diag in meta for in-memory use (not when persisting)
            meta["_diag"] = diag
        
        return StrategyArtifacts(
            weights=weights,
            budget=budget,
            factor_weights=factor_arrays["factor_weights"],
            ic_raw=factor_arrays["ic_raw"],
            ic_smooth=factor_arrays["ic_smooth"],
            composite_score=factor_arrays["composite_score"],
            F=factor_arrays["F"],
            meta=meta,
            tag=use_tag,
            strategy_id=self.strategy_id(),
        )


@StrategyRegistry.register
class NYSEEqualWeightBenchmark(BaseBenchmarkStrategy):
    """NYSE Equal-Weighted Benchmark."""
    
    def __init__(self):
        super().__init__(exchange="NYS", weighting="equal_weight")
    
    @classmethod
    def strategy_id(cls) -> str:
        return "benchmark_nys_ew"


@StrategyRegistry.register
class NYSEMarketCapBenchmark(BaseBenchmarkStrategy):
    """NYSE Market-Cap Weighted Benchmark."""
    
    def __init__(self):
        super().__init__(exchange="NYS", weighting="market_cap")
    
    @classmethod
    def strategy_id(cls) -> str:
        return "benchmark_nys_mc"


@StrategyRegistry.register
class NASDAQEqualWeightBenchmark(BaseBenchmarkStrategy):
    """NASDAQ Equal-Weighted Benchmark."""
    
    def __init__(self):
        super().__init__(exchange="NAS", weighting="equal_weight")
    
    @classmethod
    def strategy_id(cls) -> str:
        return "benchmark_nas_ew"


@StrategyRegistry.register
class NASDAQMarketCapBenchmark(BaseBenchmarkStrategy):
    """NASDAQ Market-Cap Weighted Benchmark."""
    
    def __init__(self):
        super().__init__(exchange="NAS", weighting="market_cap")
    
    @classmethod
    def strategy_id(cls) -> str:
        return "benchmark_nas_mc"


# =============================================================================
# Index-based Benchmark Strategies (SP500, NAS100)
# =============================================================================

class BaseIndexBenchmarkStrategy(StrategyBase):
    """Base class for index-based benchmark strategies (SP500, NAS100).
    
    Unlike exchange-based benchmarks, index benchmarks load data using
    dedicated Quantiacs index loaders (load_spx_data, load_ndx_data).
    """
    
    def __init__(self, index_type: str, weighting: str):
        """Initialize index benchmark strategy.
        
        Args:
            index_type: Index type ("SP500" or "NAS100")
            weighting: Weighting method ("equal_weight" or "market_cap")
        """
        if index_type not in benchmark_config.SUPPORTED_INDICES:
            raise ValueError(
                f"Unsupported index type: {index_type}. "
                f"Supported: {benchmark_config.SUPPORTED_INDICES}"
            )
        
        self.index_type = index_type
        self.weighting = weighting
        self._engine = BenchmarkEngine()
        self._factors: Dict[str, "xr.DataArray"] = {}
    
    @property
    def config(self) -> StrategyConfig:
        index_display = INDEX_DISPLAY_NAMES.get(self.index_type, self.index_type)
        weighting_name = "Equal-Weighted" if self.weighting == "equal_weight" else "Market-Cap Weighted"
        
        return StrategyConfig(
            name=self.strategy_id(),
            display_name=f"{index_display} {weighting_name} Benchmark",
            version="1.0",
            min_date=benchmark_config.INDEX_MIN_DATE,
            exchanges=[],  # Index benchmarks are not exchange-specific
            factors=[],  # Benchmarks don't have factors
            topn=0,
            max_pos=1.0,
            min_pos=0.0,
            target_vol=0.0,
            lev_cap=1.0,
            lev_min=1.0,
            tc_bps=0.0,  # Benchmarks assume no transaction costs
            description=f"{index_display} {weighting_name} benchmark index",
            long_only=True,
            long_seats=0,
            short_seats=0,
        )
    
    def get_factors(self) -> Dict[str, "xr.DataArray"]:
        return self._factors
    
    def run(
        self,
        *,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        tag: Optional[str] = None,
        write_outputs: bool = False,  # Benchmarks typically don't write outputs
    ) -> StrategyArtifacts:
        if xr is None:
            raise ImportError("xarray required")
        
        use_min_date = min_date or benchmark_config.INDEX_MIN_DATE
        
        # Compute benchmark weights and returns using index-specific loaders
        if self.weighting == "equal_weight":
            bundle, weights, portfolio_returns, budget = self._engine.compute_index_equal_weighted_benchmark(
                index_type=self.index_type,
                min_date=use_min_date,
                max_date=max_date,
            )
        elif self.weighting == "market_cap":
            bundle, weights, portfolio_returns, budget = self._engine.compute_index_market_cap_weighted_benchmark(
                index_type=self.index_type,
                min_date=use_min_date,
                max_date=max_date,
            )
        else:
            raise ValueError(f"Unknown weighting method: {self.weighting}")
        
        # Create empty factor arrays (benchmarks don't have factors)
        time_coords = weights.time
        asset_coords = weights.asset
        factor_arrays = _create_empty_factor_arrays(time_coords, asset_coords, n_factors=1)
        
        # Align weights with returns for diagnostics
        weights_aligned = weights.reindex(
            time=bundle.returns.time,
            asset=bundle.returns.asset,
            method="ffill"
        ).fillna(0.0)
        
        # Compute portfolio diagnostics
        diag = _compute_portfolio_diag(
            final_weights=weights_aligned,
            returns=bundle.returns,
            tc_bps=0.0,  # Benchmarks assume no transaction costs
        )
        
        use_tag = tag or datetime.now().strftime("%Y-%m-%d")
        
        index_display = INDEX_DISPLAY_NAMES.get(self.index_type, self.index_type)
        
        meta = {
            "strategy_id": self.strategy_id(),
            "strategy_version": self.config.version,
            "display_name": self.config.display_name,
            "tag": use_tag,
            "run_timestamp": datetime.now().isoformat(),
            "date_range": [
                str(weights.time.values[0])[:10] if len(weights.time) > 0 else "",
                str(weights.time.values[-1])[:10] if len(weights.time) > 0 else "",
            ],
            "index_type": self.index_type,
            "index_display_name": index_display,
            "weighting": self.weighting,
            "is_benchmark": True,
            "is_index_benchmark": True,
            "n_constituents": len(bundle.asset_ids),
        }
        
        # Write outputs if requested
        if write_outputs:
            from q23.strategy.outputs import OutputWriter
            
            output_dir = self.get_output_dir()
            writer = OutputWriter(
                base_name=self.strategy_id(),
                output_dir=output_dir,
            )
            
            writer.write_wide_weights(weights, tag=use_tag)
            writer.write_budget(budget, tag=use_tag)
            writer.write_factor_weights(factor_arrays["factor_weights"], tag=use_tag)
            writer.write_ic(factor_arrays["ic_raw"], factor_arrays["ic_smooth"], tag=use_tag)
            
            if diag is not None and not diag.empty:
                writer.write_portfolio_diag(diag, tag=use_tag)
            
            writer.write_meta(meta, tag=use_tag)
        else:
            # Only store _diag in meta for in-memory use (not when persisting)
            meta["_diag"] = diag
        
        return StrategyArtifacts(
            weights=weights,
            budget=budget,
            factor_weights=factor_arrays["factor_weights"],
            ic_raw=factor_arrays["ic_raw"],
            ic_smooth=factor_arrays["ic_smooth"],
            composite_score=factor_arrays["composite_score"],
            F=factor_arrays["F"],
            meta=meta,
            tag=use_tag,
            strategy_id=self.strategy_id(),
        )


# -----------------------------------------------------------------------------
# S&P 500 Index Benchmarks
# -----------------------------------------------------------------------------

@StrategyRegistry.register
class SP500EqualWeightBenchmark(BaseIndexBenchmarkStrategy):
    """S&P 500 Equal-Weighted Benchmark.
    
    Equal-weighted portfolio of all S&P 500 constituents loaded via
    qndata.stocks.load_spx_data().
    """
    
    def __init__(self):
        super().__init__(index_type="SP500", weighting="equal_weight")
    
    @classmethod
    def strategy_id(cls) -> str:
        return "benchmark_sp500_ew"


@StrategyRegistry.register
class SP500MarketCapBenchmark(BaseIndexBenchmarkStrategy):
    """S&P 500 Market-Cap Weighted Benchmark.
    
    Market-cap weighted portfolio of all S&P 500 constituents loaded via
    qndata.stocks.load_spx_data().
    """
    
    def __init__(self):
        super().__init__(index_type="SP500", weighting="market_cap")
    
    @classmethod
    def strategy_id(cls) -> str:
        return "benchmark_sp500_mc"


# -----------------------------------------------------------------------------
# NASDAQ 100 Index Benchmarks
# -----------------------------------------------------------------------------

@StrategyRegistry.register
class NAS100EqualWeightBenchmark(BaseIndexBenchmarkStrategy):
    """NASDAQ 100 Equal-Weighted Benchmark.
    
    Equal-weighted portfolio of all NASDAQ 100 constituents loaded via
    qndata.stocks.load_ndx_data().
    """
    
    def __init__(self):
        super().__init__(index_type="NAS100", weighting="equal_weight")
    
    @classmethod
    def strategy_id(cls) -> str:
        return "benchmark_nas100_ew"


@StrategyRegistry.register
class NAS100MarketCapBenchmark(BaseIndexBenchmarkStrategy):
    """NASDAQ 100 Market-Cap Weighted Benchmark.
    
    Market-cap weighted portfolio of all NASDAQ 100 constituents loaded via
    qndata.stocks.load_ndx_data().
    """
    
    def __init__(self):
        super().__init__(index_type="NAS100", weighting="market_cap")
    
    @classmethod
    def strategy_id(cls) -> str:
        return "benchmark_nas100_mc"
