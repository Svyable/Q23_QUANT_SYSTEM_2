"""q23.strategy.benchmark_data

Benchmark-specific data loading and computation utilities.

This module provides:
- Market cap data loading from Quantiacs with proxy fallback
- Exchange-filtered asset identification
- Benchmark return computation utilities
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from q23.strategy.data_loader import MarketDataBundle, _require_xr, compute_returns, pinned_indices
from q23.shared.config import cfg
from q23.strategy.data_cache import (
    load_cached_data,
    save_to_cache,
    is_cache_valid,
    cleanup_old_cache,
)


# Index type mapping: maps index name to (list_loader_name, data_loader_name)
INDEX_LOADERS = {
    "SP500": ("load_spx_list", "load_spx_data"),
    "NAS100": ("load_ndx_list", "load_ndx_data"),
}

# Cache for index lists (keyed by (index_type, min_date))
_index_list_cache: Dict[Tuple[str, str], List[Dict]] = {}

# Cache for index data bundles (keyed by (index_type, min_date, max_date))
_index_data_cache: Dict[Tuple[str, str, Optional[str]], "MarketDataBundle"] = {}


def _require_pd() -> None:
    if pd is None:
        raise ImportError("pandas is required for q23.strategy.benchmark_data")


def load_market_cap_data(
    *,
    bundle: MarketDataBundle,
    proxy_method: str = "close_volume",
) -> Optional["xr.DataArray"]:
    """Load market cap data from Quantiacs with proxy fallback.
    
    Args:
        bundle: MarketDataBundle containing market data
        proxy_method: Method to use for proxy if market cap unavailable
            - "close_volume": close * volume (default)
            - "close_squared_volume": close^2 * volume
    
    Returns:
        Market cap DataArray (time, asset) or None if unavailable
    """
    _require_xr()
    
    ds = bundle.data
    
    # Try to load market cap from Quantiacs
    try:
        import qnt.data as qndata  # type: ignore
    except Exception:
        # Quantiacs not available, use proxy
        return _compute_market_cap_proxy(bundle, proxy_method)
    
    # OPTIMIZED: Try loading all field names at once first (single API call)
    # Then fall back to individual attempts if batch fails
    market_cap_fields = ["marketcap", "market_cap", "mktcap", "mcap"]
    
    if hasattr(qndata, "stocks") and hasattr(qndata.stocks, "load_data"):
        # Try batch load first (more efficient - single API call)
        try:
            raw_batch = qndata.stocks.load_data(
                min_date=str(bundle.meta["min_date"]),
                max_date=bundle.meta.get("max_date"),
                assets=bundle.asset_ids,
                fields=market_cap_fields,  # Try all at once
            )
            
            # Check if we got any market cap data
            if raw_batch is not None:
                # Normalize to Dataset if needed
                if isinstance(raw_batch, xr.DataArray) and "field" in raw_batch.dims:
                    raw_batch = raw_batch.to_dataset(dim="field")
                
                if isinstance(raw_batch, xr.Dataset):
                    # Try each field name in order of preference
                    for field_name in market_cap_fields:
                        if field_name in raw_batch.variables:
                            mcap = raw_batch[field_name]
                            # Align with bundle data
                            mcap = mcap.reindex(
                                time=ds.time,
                                asset=ds.asset,
                                method="ffill"
                            )
                            # Ensure positive values
                            mcap = xr.where(mcap > 0, mcap, np.nan)
                            return mcap
        except Exception:
            # Batch load failed, fall through to individual attempts
            pass
    
    # Fallback: Try individual field names (original approach)
    for field_name in market_cap_fields:
        try:
            # Try loading with the field using stocks.load_data
            raw = None
            
            if hasattr(qndata, "stocks") and hasattr(qndata.stocks, "load_data"):
                try:
                    raw = qndata.stocks.load_data(
                        min_date=str(bundle.meta["min_date"]),
                        max_date=bundle.meta.get("max_date"),
                        assets=bundle.asset_ids,
                        fields=[field_name],
                    )
                except Exception:
                    continue
            
            if raw is None:
                continue
            
            # Normalize to DataArray
            if isinstance(raw, xr.Dataset):
                if field_name in raw.variables:
                    mcap = raw[field_name]
                    # Align with bundle data
                    mcap = mcap.reindex(
                        time=ds.time,
                        asset=ds.asset,
                        method="ffill"
                    )
                    # Ensure positive values
                    mcap = xr.where(mcap > 0, mcap, np.nan)
                    return mcap
            elif isinstance(raw, xr.DataArray):
                if "field" in raw.dims:
                    ds_field = raw.to_dataset(dim="field")
                    if field_name in ds_field.variables:
                        mcap = ds_field[field_name]
                        mcap = mcap.reindex(
                            time=ds.time,
                            asset=ds.asset,
                            method="ffill"
                        )
                        mcap = xr.where(mcap > 0, mcap, np.nan)
                        return mcap
        except Exception:
            continue
    
    # Fallback to proxy
    return _compute_market_cap_proxy(bundle, proxy_method)


def _compute_market_cap_proxy(
    bundle: MarketDataBundle,
    method: str = "close_volume",
) -> "xr.DataArray":
    """Compute market cap proxy from available data.
    
    Args:
        bundle: MarketDataBundle containing market data
        method: Proxy method ("close_volume" or "close_squared_volume")
    
    Returns:
        Market cap proxy DataArray (time, asset)
    """
    _require_xr()
    
    ds = bundle.data
    
    if "close" not in ds.variables:
        raise ValueError("Cannot compute market cap proxy: 'close' not available")
    
    close = ds["close"]
    
    if method == "close_volume":
        if "vol" not in ds.variables:
            raise ValueError("Cannot compute market cap proxy: 'vol' not available")
        volume = ds["vol"]
        # Proxy: close * volume (daily trading value as proxy for market cap)
        proxy = close * volume
    elif method == "close_squared_volume":
        if "vol" not in ds.variables:
            raise ValueError("Cannot compute market cap proxy: 'vol' not available")
        volume = ds["vol"]
        # Proxy: close^2 * volume (alternative proxy)
        proxy = close ** 2 * volume
    else:
        raise ValueError(f"Unknown proxy method: {method}")
    
    # Ensure positive values
    proxy = xr.where(proxy > 0, proxy, np.nan)
    
    return proxy


# -----------------------------------------------------------------------------
# Index-specific data loading (SP500, NAS100)
# -----------------------------------------------------------------------------

def load_index_list(
    index_type: str,
    min_date: str,
) -> List[Dict]:
    """Load index constituent list from Quantiacs.
    
    Uses memoization to cache index list loads by (index_type, min_date),
    avoiding redundant Quantiacs API calls.
    
    Args:
        index_type: Index type ("SP500" or "NAS100")
        min_date: Minimum date for the index list
    
    Returns:
        List of asset dictionaries with 'id' and other metadata
    
    Raises:
        ValueError: If index_type is not supported
        ImportError: If qnt.data is not available
        RuntimeError: If the index list loader is not available
    """
    if index_type not in INDEX_LOADERS:
        raise ValueError(
            f"Unsupported index type: {index_type}. "
            f"Supported types: {list(INDEX_LOADERS.keys())}"
        )
    
    cache_key = (index_type, min_date)
    
    # Check cache first
    if cache_key in _index_list_cache:
        return _index_list_cache[cache_key]
    
    try:
        import qnt.data as qndata  # type: ignore
    except Exception as e:
        raise ImportError(
            "Quantiacs qnt package not found. Activate your qntdev conda env."
        ) from e
    
    list_loader_name, _ = INDEX_LOADERS[index_type]
    
    # Get the loader function
    if not (hasattr(qndata, "stocks") and hasattr(qndata.stocks, list_loader_name)):
        raise RuntimeError(
            f"Index list loader not available: qnt.data.stocks.{list_loader_name}. "
            f"Update your qnt package or check index type: {index_type}"
        )
    
    loader = getattr(qndata.stocks, list_loader_name)
    
    try:
        stocks_list = loader(min_date=min_date)
        # Ensure it's a list of dicts
        if not isinstance(stocks_list, list):
            stocks_list = list(stocks_list)
        
        # Cache the result
        _index_list_cache[cache_key] = stocks_list
        
        return stocks_list
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {index_type} list from Quantiacs: {e}"
        ) from e


def _bundle_from_dataset(
    ds: "xr.Dataset",
    index_type: str,
    min_date: str,
    max_date: Optional[str] = None,
) -> Optional[MarketDataBundle]:
    """Convert a cached xr.Dataset to MarketDataBundle.
    
    Args:
        ds: Dataset loaded from cache
        index_type: Index type for metadata
        min_date: Min date for filtering
        max_date: Max date for filtering (optional)
    
    Returns:
        MarketDataBundle or None if dataset is invalid
    """
    _require_xr()
    
    # Validate dataset has required dimensions
    if "time" not in ds.dims or "asset" not in ds.dims:
        return None
    
    if ds.sizes.get("time", 0) == 0 or ds.sizes.get("asset", 0) == 0:
        return None
    
    if "close" not in ds.variables:
        return None
    
    # Apply date filters safely
    try:
        if min_date and "time" in ds.dims:
            ds = ds.sel(time=slice(min_date, None))
        if max_date and "time" in ds.dims:
            ds = ds.sel(time=slice(None, max_date))
    except Exception:
        # If filtering fails, return None
        return None
    
    # Check we still have data after filtering
    if ds.sizes.get("time", 0) == 0 or ds.sizes.get("asset", 0) == 0:
        return None
    
    # Compute returns
    rets = compute_returns(ds["close"])
    
    # Extract asset IDs
    asset_ids = list(map(str, ds.asset.values))
    
    # No pinned assets for index benchmarks
    pin_idx = np.array([], dtype=int)
    
    meta = {
        "min_date": min_date,
        "max_date": max_date,
        "n_assets": int(ds.sizes.get("asset", 0)),
        "n_days": int(ds.sizes.get("time", 0)),
        "index_type": index_type,
        "assets_mode": "index_constituents",
        "from_cache": True,
    }
    
    return MarketDataBundle(
        data=ds,
        returns=rets,
        asset_ids=asset_ids,
        pin_idx=pin_idx,
        meta=meta,
    )


def load_index_data(
    index_type: str,
    min_date: str,
    max_date: Optional[str] = None,
    use_cache: bool = True,
    use_file_cache: bool = True,
) -> MarketDataBundle:
    """Load index constituent OHLCV data from Quantiacs.
    
    Uses two-level caching:
    1. File cache: Daily .nc files in .cache/data/YYYYMMDD/ (persists across sessions)
    2. Memory cache: In-memory dict (faster, within session only)
    
    Args:
        index_type: Index type ("SP500" or "NAS100")
        min_date: Minimum date for the data
        max_date: Optional maximum date for the data
        use_cache: Whether to use in-memory cached data (default True)
        use_file_cache: Whether to use file-based daily cache (default True)
    
    Returns:
        MarketDataBundle containing index constituent data
    
    Raises:
        ValueError: If index_type is not supported
        ImportError: If qnt.data or xarray is not available
        RuntimeError: If data loading fails
    """
    _require_xr()
    
    if index_type not in INDEX_LOADERS:
        raise ValueError(
            f"Unsupported index type: {index_type}. "
            f"Supported types: {list(INDEX_LOADERS.keys())}"
        )
    
    cache_key = (index_type, min_date, max_date)
    
    # Check in-memory cache first (fastest)
    if use_cache and cache_key in _index_data_cache:
        return _index_data_cache[cache_key]
    
    # Check file cache second (avoids API call)
    if use_file_cache and is_cache_valid(index_type):
        cached_ds = load_cached_data(index_type)
        if cached_ds is not None:
            bundle = _bundle_from_dataset(cached_ds, index_type, min_date, max_date)
            if bundle is not None:
                # Store in memory cache too
                if use_cache:
                    _index_data_cache[cache_key] = bundle
                return bundle
            # If bundle is None, cache was invalid - continue to API load
    
    # Load from Quantiacs API
    try:
        import qnt.data as qndata  # type: ignore
    except Exception as e:
        raise ImportError(
            "Quantiacs qnt package not found. Activate your qntdev conda env."
        ) from e
    
    list_loader_name, data_loader_name = INDEX_LOADERS[index_type]
    
    # First, load the index constituent list
    stocks_list = load_index_list(index_type, min_date)
    
    if not stocks_list:
        raise RuntimeError(
            f"No constituents found for {index_type} index with min_date={min_date}"
        )
    
    # Get the data loader function
    if not (hasattr(qndata, "stocks") and hasattr(qndata.stocks, data_loader_name)):
        raise RuntimeError(
            f"Index data loader not available: qnt.data.stocks.{data_loader_name}. "
            f"Update your qnt package or check index type: {index_type}"
        )
    
    data_loader = getattr(qndata.stocks, data_loader_name)
    
    try:
        # Load data using the index-specific loader
        raw = data_loader(min_date=min_date, assets=stocks_list)
        
        # Normalize to Dataset
        if isinstance(raw, xr.DataArray):
            if "field" in raw.dims:
                ds = raw.to_dataset(dim="field")
            else:
                raise ValueError(
                    f"Unexpected xarray.DataArray without 'field' dimension "
                    f"from {data_loader_name}"
                )
        elif isinstance(raw, xr.Dataset):
            ds = raw
        else:
            raise TypeError(
                f"Expected xarray Dataset or DataArray from {data_loader_name}, "
                f"got {type(raw)}"
            )
        
        # Apply max_date filter if specified
        if max_date is not None:
            ds = ds.sel(time=slice(None, max_date))
        
        # Ensure canonical dims order
        ds = ds.transpose("time", "asset", missing_dims="ignore")
        
        # Ensure required vars exist
        need = {"open", "high", "low", "close", "vol"}
        missing = [v for v in need if v not in ds.variables]
        if missing:
            raise ValueError(
                f"Loaded {index_type} dataset missing required vars: {missing}. "
                f"Available vars={list(ds.variables)}"
            )
        
        # Ensure is_liquid exists (infer if needed)
        if "is_liquid" not in ds.variables:
            liq = xr.where(np.isfinite(ds["vol"]) & (ds["vol"] > 0), 1.0, 0.0)
            ds["is_liquid"] = liq
        
        # Save to file cache (full dataset without date filtering)
        # This allows future requests with different date ranges to use the same cache
        if use_file_cache:
            save_to_cache(index_type, ds)
            # Clean up old cache files (keep last 3 days)
            cleanup_old_cache(keep_days=3)
        
        # Compute returns
        rets = compute_returns(ds["close"])
        
        # Extract asset IDs
        asset_ids = list(map(str, ds.asset.values))
        
        # No pinned assets for index benchmarks
        pin_idx = np.array([], dtype=int)
        
        meta = {
            "min_date": min_date,
            "max_date": max_date,
            "n_assets": int(ds.sizes.get("asset", 0)),
            "n_days": int(ds.sizes.get("time", 0)),
            "index_type": index_type,
            "assets_mode": "index_constituents",
            "from_cache": False,
        }
        
        bundle = MarketDataBundle(
            data=ds,
            returns=rets,
            asset_ids=asset_ids,
            pin_idx=pin_idx,
            meta=meta,
        )
        
        # Cache in memory
        if use_cache:
            _index_data_cache[cache_key] = bundle
        
        return bundle
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {index_type} data from Quantiacs: {e}"
        ) from e


def get_index_asset_ids(
    index_type: str,
    min_date: str,
) -> List[str]:
    """Get list of asset IDs for an index.
    
    Args:
        index_type: Index type ("SP500" or "NAS100")
        min_date: Minimum date for the index list
    
    Returns:
        List of asset ID strings
    """
    stocks_list = load_index_list(index_type, min_date)
    return [str(x.get("id", "")) for x in stocks_list if x.get("id")]


# Cache for exchange-filtered SPX lists (keyed by (min_date, exchange))
_exchange_assets_cache: Dict[Tuple[str, str], List[str]] = {}


def get_exchange_filtered_assets(
    bundle: MarketDataBundle,
    exchange: str,
) -> List[str]:
    """Get asset IDs filtered by exchange.
    
    OPTIMIZED: Uses memoization to cache exchange-filtered asset lists,
    avoiding redundant Quantiacs API calls for the same date/exchange combination.
    
    Args:
        bundle: MarketDataBundle containing market data
        exchange: Exchange code ("NYS" for NYSE, "NAS" for NASDAQ)
    
    Returns:
        List of asset IDs for the specified exchange
    """
    exchange_upper = exchange.upper()
    min_date_str = str(bundle.meta["min_date"])
    cache_key = (min_date_str, exchange_upper)
    
    # Check cache first
    if cache_key in _exchange_assets_cache:
        cached_ids = _exchange_assets_cache[cache_key]
        # Still need to intersect with bundle assets (bundle may have different assets)
        bundle_assets = set(bundle.asset_ids)
        return [aid for aid in cached_ids if aid in bundle_assets]
    
    # Try to get exchange info from Quantiacs
    try:
        import qnt.data as qndata  # type: ignore
        _require_pd()
        
        if hasattr(qndata, "stocks") and hasattr(qndata.stocks, "load_spx_list"):
            try:
                df = pd.DataFrame(
                    qndata.stocks.load_spx_list(min_date=min_date_str)
                )
                if not df.empty and "id" in df.columns and "exchange" in df.columns:
                    # Filter by exchange
                    ex_mask = df["exchange"].astype(str).str.upper() == exchange_upper
                    filtered_ids = df.loc[ex_mask, "id"].dropna().astype(str).unique().tolist()
                    
                    # Cache the result
                    _exchange_assets_cache[cache_key] = filtered_ids
                    
                    # Intersect with bundle assets
                    bundle_assets = set(bundle.asset_ids)
                    return [aid for aid in filtered_ids if aid in bundle_assets]
            except Exception:
                pass
    except Exception:
        pass
    
    # Fallback: if bundle was loaded with exchange filter, use all assets
    # This is a best-effort approach when exchange metadata isn't available
    bundle_exchanges = bundle.meta.get("exchanges", [])
    if exchange_upper in [ex.upper() for ex in bundle_exchanges]:
        # If the exchange matches bundle exchanges, return all assets
        # (assuming bundle was already filtered)
        return bundle.asset_ids
    
    # If we can't determine exchange, return empty list
    # This will cause benchmarks to fail gracefully
    return []


def _validate_returns(returns: "xr.DataArray") -> bool:
    """Check if returns DataArray has valid structure."""
    _require_xr()
    
    if returns is None:
        return False
    
    # Must have time and asset dimensions
    if "time" not in returns.dims or "asset" not in returns.dims:
        return False
    
    # Must have at least some data
    if returns.sizes.get("time", 0) == 0 or returns.sizes.get("asset", 0) == 0:
        return False
    
    return True


def compute_equal_weighted_returns(
    bundle: MarketDataBundle,
    exchange: Optional[str] = None,
) -> Tuple["xr.DataArray", "xr.DataArray"]:
    """Compute equal-weighted portfolio weights and returns.
    
    Args:
        bundle: MarketDataBundle containing market data
        exchange: Optional exchange filter ("NYS" or "NAS")
    
    Returns:
        Tuple of (weights, portfolio_returns) DataArrays
    
    Raises:
        ValueError: If bundle contains invalid data
    """
    _require_xr()
    
    returns = bundle.returns
    asset_ids = bundle.asset_ids
    
    # Validate returns data
    if not _validate_returns(returns):
        raise ValueError(
            "Invalid market data: returns DataArray is missing required "
            "dimensions (time, asset) or is empty. Ensure data was loaded correctly."
        )
    
    # Filter by exchange if specified
    if exchange:
        exchange_assets = get_exchange_filtered_assets(bundle, exchange)
        if not exchange_assets:
            # No assets for this exchange - create properly structured empty arrays
            # instead of using zeros_like on potentially invalid data
            empty_weights = xr.DataArray(
                np.zeros((returns.sizes["time"], returns.sizes["asset"])),
                dims=["time", "asset"],
                coords={"time": returns.time, "asset": returns.asset}
            )
            empty_returns = xr.DataArray(
                np.zeros(returns.sizes["time"]),
                dims=["time"],
                coords={"time": returns.time}
            )
            return empty_weights, empty_returns
        
        # Create mask for exchange assets
        asset_mask = xr.DataArray(
            [aid in exchange_assets for aid in asset_ids],
            dims=["asset"],
            coords={"asset": returns.asset}
        )
        returns_filtered = returns.where(asset_mask, 0.0)
    else:
        returns_filtered = returns
    
    # Compute equal weights: 1/n_assets for each date
    # Handle missing assets (NaN) by counting valid assets per date
    is_valid = xr.where(
        np.isfinite(returns_filtered) & (returns_filtered != 0.0),
        1.0,
        0.0
    )
    n_assets_per_date = is_valid.sum(dim="asset")
    
    # Avoid division by zero
    n_assets_safe = xr.where(n_assets_per_date > 0, n_assets_per_date, 1.0)
    
    # Equal weights: 1/n for valid assets, 0 for invalid
    weights = xr.where(
        is_valid > 0,
        1.0 / n_assets_safe,
        0.0
    )
    
    # Compute portfolio returns: sum(weights * returns)
    portfolio_returns = (weights * returns_filtered).sum(dim="asset")
    
    return weights, portfolio_returns


def compute_market_cap_weighted_returns(
    bundle: MarketDataBundle,
    market_cap: "xr.DataArray",
    exchange: Optional[str] = None,
) -> Tuple["xr.DataArray", "xr.DataArray"]:
    """Compute market-cap weighted portfolio weights and returns.
    
    Args:
        bundle: MarketDataBundle containing market data
        market_cap: Market cap DataArray (time, asset)
        exchange: Optional exchange filter ("NYS" or "NAS")
    
    Returns:
        Tuple of (weights, portfolio_returns) DataArrays
    
    Raises:
        ValueError: If bundle contains invalid data
    """
    _require_xr()
    
    returns = bundle.returns
    asset_ids = bundle.asset_ids
    
    # Validate returns data
    if not _validate_returns(returns):
        raise ValueError(
            "Invalid market data: returns DataArray is missing required "
            "dimensions (time, asset) or is empty. Ensure data was loaded correctly."
        )
    
    # Filter by exchange if specified
    if exchange:
        exchange_assets = get_exchange_filtered_assets(bundle, exchange)
        if not exchange_assets:
            # No assets for this exchange - create properly structured empty arrays
            empty_weights = xr.DataArray(
                np.zeros((returns.sizes["time"], returns.sizes["asset"])),
                dims=["time", "asset"],
                coords={"time": returns.time, "asset": returns.asset}
            )
            empty_returns = xr.DataArray(
                np.zeros(returns.sizes["time"]),
                dims=["time"],
                coords={"time": returns.time}
            )
            return empty_weights, empty_returns
        
        # Create mask for exchange assets
        asset_mask = xr.DataArray(
            [aid in exchange_assets for aid in asset_ids],
            dims=["asset"],
            coords={"asset": returns.asset}
        )
        returns_filtered = returns.where(asset_mask, 0.0)
        market_cap_filtered = market_cap.where(asset_mask, 0.0)
    else:
        returns_filtered = returns
        market_cap_filtered = market_cap
    
    # Align market cap with returns
    market_cap_filtered = market_cap_filtered.reindex(
        time=returns_filtered.time,
        asset=returns_filtered.asset,
        method="ffill"
    )
    
    # Compute weights: market_cap / sum(market_cap) per date
    market_cap_sum = market_cap_filtered.sum(dim="asset")
    
    # Avoid division by zero
    weights = xr.where(
        market_cap_sum > 0,
        market_cap_filtered / market_cap_sum,
        0.0
    )
    
    # Handle NaN values
    weights = xr.where(np.isfinite(weights), weights, 0.0)
    
    # Normalize weights to sum to 1 (in case of rounding errors)
    weight_sum = weights.sum(dim="asset")
    weights = xr.where(weight_sum > 0, weights / weight_sum, 0.0)
    
    # Compute portfolio returns: sum(weights * returns)
    portfolio_returns = (weights * returns_filtered).sum(dim="asset")
    
    return weights, portfolio_returns


# -----------------------------------------------------------------------------
# Index-specific return computation (SP500, NAS100)
# -----------------------------------------------------------------------------

def compute_index_equal_weighted_returns(
    bundle: MarketDataBundle,
) -> Tuple["xr.DataArray", "xr.DataArray"]:
    """Compute equal-weighted portfolio weights and returns for index constituents.
    
    This is specifically designed for index benchmarks (SP500, NAS100) where
    all assets in the bundle are index constituents (no exchange filtering needed).
    
    Args:
        bundle: MarketDataBundle containing index constituent data
    
    Returns:
        Tuple of (weights, portfolio_returns) DataArrays
    """
    _require_xr()
    
    returns = bundle.returns
    
    # Compute equal weights: 1/n_assets for each date
    # Handle missing assets (NaN) by counting valid assets per date
    is_valid = xr.where(
        np.isfinite(returns) & (returns != 0.0),
        1.0,
        0.0
    )
    n_assets_per_date = is_valid.sum(dim="asset")
    
    # Avoid division by zero
    n_assets_safe = xr.where(n_assets_per_date > 0, n_assets_per_date, 1.0)
    
    # Equal weights: 1/n for valid assets, 0 for invalid
    weights = xr.where(
        is_valid > 0,
        1.0 / n_assets_safe,
        0.0
    )
    
    # Compute portfolio returns: sum(weights * returns)
    portfolio_returns = (weights * returns).sum(dim="asset")
    
    return weights, portfolio_returns


def compute_index_market_cap_weighted_returns(
    bundle: MarketDataBundle,
    market_cap: "xr.DataArray",
) -> Tuple["xr.DataArray", "xr.DataArray"]:
    """Compute market-cap weighted portfolio weights and returns for index constituents.
    
    This is specifically designed for index benchmarks (SP500, NAS100) where
    all assets in the bundle are index constituents (no exchange filtering needed).
    
    Args:
        bundle: MarketDataBundle containing index constituent data
        market_cap: Market cap DataArray (time, asset)
    
    Returns:
        Tuple of (weights, portfolio_returns) DataArrays
    """
    _require_xr()
    
    returns = bundle.returns
    
    # Align market cap with returns
    market_cap_aligned = market_cap.reindex(
        time=returns.time,
        asset=returns.asset,
        method="ffill"
    ).fillna(0.0)
    
    # Compute weights: market_cap / sum(market_cap) per date
    market_cap_sum = market_cap_aligned.sum(dim="asset")
    
    # Avoid division by zero
    weights = xr.where(
        market_cap_sum > 0,
        market_cap_aligned / market_cap_sum,
        0.0
    )
    
    # Handle NaN values
    weights = xr.where(np.isfinite(weights), weights, 0.0)
    
    # Normalize weights to sum to 1 (in case of rounding errors)
    weight_sum = weights.sum(dim="asset")
    weights = xr.where(weight_sum > 0, weights / weight_sum, 0.0)
    
    # Compute portfolio returns: sum(weights * returns)
    portfolio_returns = (weights * returns).sum(dim="asset")
    
    return weights, portfolio_returns


def load_index_market_cap_data(
    bundle: MarketDataBundle,
    proxy_method: str = "close_volume",
) -> "xr.DataArray":
    """Load or compute market cap data for index constituents.
    
    Attempts to load actual market cap data from Quantiacs, falling back
    to proxy computation if unavailable.
    
    Args:
        bundle: MarketDataBundle containing index constituent data
        proxy_method: Proxy method if market cap unavailable
    
    Returns:
        Market cap DataArray (time, asset)
    """
    # Try to load actual market cap first
    mcap = load_market_cap_data(bundle=bundle, proxy_method=proxy_method)
    
    if mcap is not None:
        return mcap
    
    # Fall back to proxy
    return _compute_market_cap_proxy(bundle, proxy_method)
