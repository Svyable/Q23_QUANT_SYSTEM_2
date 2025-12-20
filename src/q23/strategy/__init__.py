"""q23.strategy

Strategy package (Quantiacs implementation).

Modules are organized to keep boundaries clean:
- data_loader: data access / universe selection (qnt.data)
- factors: factor library construction (32+ factors)
- mtf_momentum: Multi-Timeframe Momentum factors (W/M/Q/Y with IC-weighting)
- ic_weighting: dynamic IC weighting & composite score
- portfolio: portfolio construction / constraints / transaction costs
- outputs: dashboard-ready CSV exports
- engine: orchestration (ties everything together)

Factor Sets Available:
- V4_24_FACTORS: Original production 24-factor set
- MTF_MOMENTUM_FACTORS: Multi-timeframe momentum factors
- V4_PLUS_MTF_FACTORS: Combined 41+ factor set
- ALL_FACTORS: All available factors (50+)
"""

# Convenience imports
from q23.strategy.factors import (
    FactorLibrary,
    FactorParams,
    FactorWindowConfig,
    FactorCategory,
    FACTOR_REGISTRY,
    V4_24_FACTORS,
    MTF_MOMENTUM_FACTORS,
    V4_PLUS_MTF_FACTORS,
    ALL_FACTORS,
    get_dashboard_factor_config,
)

from q23.strategy.mtf_momentum import (
    MTFMomentumLibrary,
    MTFMomentumParams,
    MTF_FACTORS,
    MTF_CORE_FACTORS,
    get_mtf_dashboard_config,
)

from q23.strategy.ic_weighting import (
    DynamicICWeighting,
    ICWeightingParams,
    compute_rank_ic,
)

from q23.strategy.data_loader import (
    load_market_data,
    MarketDataBundle,
)

from q23.strategy.transaction_costs import (
    TransactionCostModel,
    compute_atr_pandas,
    compute_atr_xarray,
    compute_quantiacs_tc,
    compute_flat_tc,
)

__all__ = [
    # Factor Library
    "FactorLibrary",
    "FactorParams",
    "FactorWindowConfig",
    "FactorCategory",
    "FACTOR_REGISTRY",
    "V4_24_FACTORS",
    "MTF_MOMENTUM_FACTORS",
    "V4_PLUS_MTF_FACTORS",
    "ALL_FACTORS",
    "get_dashboard_factor_config",
    # MTF Momentum
    "MTFMomentumLibrary",
    "MTFMomentumParams",
    "MTF_FACTORS",
    "MTF_CORE_FACTORS",
    "get_mtf_dashboard_config",
    # IC Weighting
    "DynamicICWeighting",
    "ICWeightingParams",
    "compute_rank_ic",
    # Data Loading
    "load_market_data",
    "MarketDataBundle",
    # Transaction Costs
    "TransactionCostModel",
    "compute_atr_pandas",
    "compute_atr_xarray",
    "compute_quantiacs_tc",
    "compute_flat_tc",
]
