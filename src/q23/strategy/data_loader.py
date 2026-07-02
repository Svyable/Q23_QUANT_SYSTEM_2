"""q23.strategy.data_loader

Quantiacs data loading + universe selection.

Goals
-----
- Hide qnt.data API churn behind a stable internal interface (MarketDataBundle).
- Load only the fields we need, but be robust if some optional fields are unavailable.
- Keep the strategy + dashboard aligned with a consistent schema:
    xr.Dataset vars: open, high, low, close, vol, (is_liquid optional)
    dims: time, asset

Common failure fixed
--------------------
Some qnt versions error with:
    KeyError: "not all values found in index 'field'"
when you request a field that doesn't exist (e.g., 'is_liquid').
We now retry with a smaller field set automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from q23.shared.config import cfg
from q23.strategy.data_cache import (
    load_cached_data,
    save_to_cache,
    is_cache_valid,
    cleanup_old_cache,
)
from q23.strategy.marketstack_client import MarketstackClient
from q23.strategy.symbol_mapper import batch_convert_assets
from q23.strategy.data_gap_detector import (
    should_fetch_marketstack,
    get_missing_date_range,
    get_latest_quantiacs_date,
)
from q23.strategy.data_merger import (
    marketstack_to_xarray,
    merge_datasets,
)


def _require_xr() -> None:
    if xr is None:
        raise ImportError("xarray is required for q23.strategy.data_loader")


@dataclass(frozen=True)
class MarketDataBundle:
    """Normalized market data container."""

    data: "xr.Dataset"          # open/high/low/close/vol/(is_liquid)
    returns: "xr.DataArray"     # daily returns (time, asset)
    asset_ids: List[str]
    pin_idx: np.ndarray           # indices of pinned assets (for seat allocation)
    meta: Dict[str, Any]


# -----------------------------------------------------------------------------
# Internal: robust qnt loader invocation
# -----------------------------------------------------------------------------

def _try_call(fn, **kwargs):
    """Try calling `fn` with a subset of kwargs (only for signature mismatch).

    This is used to survive minor parameter-name differences across qnt versions.
    It only retries on TypeError (unexpected kwarg).
    """
    try:
        return fn(**kwargs)
    except TypeError:
        pass

    keys = list(kwargs.keys())
    for k in keys:
        kk = dict(kwargs)
        kk.pop(k, None)
        try:
            return fn(**kk)
        except TypeError:
            continue

    # last resort: positional min_date
    try:
        if "min_date" in kwargs:
            return fn(kwargs["min_date"])
    except Exception:
        pass

    # re-raise the original error context is lost here; better to fail loudly
    return fn(**kwargs)


def _looks_like_missing_field_error(e: Exception) -> bool:
    msg = str(e)
    return ("index 'field'" in msg) and ("not all values found" in msg)


def _normalize_to_dataset(obj: Any) -> "xr.Dataset":
    """Normalize qnt return types into an xr.Dataset."""
    _require_xr()
    if isinstance(obj, xr.Dataset):
        return obj
    if isinstance(obj, xr.DataArray):
        if "field" in obj.dims:
            return obj.to_dataset(dim="field")
        # Some qnt loaders return Dataset already; DataArray w/out field is unexpected
        raise ValueError("Unexpected xarray.DataArray without 'field' dimension.")
    raise TypeError(f"Expected xarray Dataset or DataArray, got {type(obj)}")


# -----------------------------------------------------------------------------
# Quantiacs OHLCV loader
# -----------------------------------------------------------------------------

def _is_valid_dataset(ds: "xr.Dataset") -> bool:
    """Check if a dataset has the required structure for market data."""
    _require_xr()
    
    # Must have time and asset dimensions
    if "time" not in ds.dims or "asset" not in ds.dims:
        return False
    
    # Must have at least some data
    if ds.sizes.get("time", 0) == 0 or ds.sizes.get("asset", 0) == 0:
        return False
    
    # Must have close price at minimum
    if "close" not in ds.variables:
        return False
    
    return True


def _filter_cached_dataset(
    ds: "xr.Dataset",
    min_date: Optional[str],
    max_date: Optional[str],
    assets: Optional[Sequence[str]],
) -> Optional["xr.Dataset"]:
    """Filter a cached dataset by date range and assets.
    
    Returns None if the dataset is invalid or empty after filtering.
    """
    _require_xr()
    
    # Validate input dataset
    if not _is_valid_dataset(ds):
        return None
    
    try:
        # Apply date filters
        if min_date and "time" in ds.dims:
            ds = ds.sel(time=slice(min_date, None))
        if max_date and "time" in ds.dims:
            ds = ds.sel(time=slice(None, max_date))
        
        # Apply asset filter
        if assets is not None and "asset" in ds.dims:
            valid_assets = [a for a in assets if a in set(ds.asset.values)]
            if valid_assets:
                ds = ds.sel(asset=valid_assets)
        
        # Validate result
        if not _is_valid_dataset(ds):
            return None
        
        return ds
    except Exception:
        return None


def load_quantiacs_stocks(
    *,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    assets: Optional[Sequence[str]] = None,
    fields: Optional[Sequence[str]] = None,
    forward_order: bool = True,
    use_file_cache: bool = True,
    fill_recent_days: Optional[int] = None,
    use_marketstack: bool = True,
    force_live_data: bool = False,
    strategy_id: Optional[str] = None,
) -> "xr.Dataset":
    """Load equity OHLCV (and is_liquid if available) from Quantiacs.

    Uses two-level caching:
    1. File cache: Daily .nc files in .cache/data/YYYYMMDD/ (persists across sessions)
    2. In-memory: Via caller (load_market_data uses MarketDataBundle caching)

    Robustness:
    - Tries multiple qnt loader entry points (load_spx_data, stocks.load_data, qnt.data.load_data)
    - Retries with reduced field sets if qnt rejects a requested field (common with 'is_liquid')
    - Optionally fetches recent missing days from Marketstack API
    
    Args:
        min_date: Minimum date to load
        max_date: Maximum date to load (optional)
        assets: Specific asset IDs to load (optional)
        fields: Specific fields to load (optional)
        forward_order: Whether to sort by time ascending
        use_file_cache: Whether to use file-based daily cache (default True)
        fill_recent_days: Number of recent days to fill from Marketstack (defaults to cfg.marketstack.DEFAULT_LOOKBACK_DAYS if use_marketstack=True)
        use_marketstack: Enable Marketstack integration to fill missing recent days (default True)
        force_live_data: If True, always fetch latest Marketstack data even if no gap detected (default False)
    
    Returns:
        xr.Dataset with OHLCV data (potentially merged with Marketstack data)
    """
    _require_xr()

    min_date = min_date or cfg.strategy.MIN_DATE
    
    # Check file cache first (SPX is the main data source)
    # We cache the full dataset without asset filtering for maximum reuse
    cache_key = "SPX"
    
    if use_file_cache and is_cache_valid(cache_key):
        cached_ds = load_cached_data(cache_key)
        if cached_ds is not None:
            # #region agent log
            try:
                import json
                with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
                    latest_cached = get_latest_quantiacs_date(cached_ds) if hasattr(cached_ds, 'time') else None
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "F",
                        "location": "data_loader.py:230",
                        "message": "Using cached data",
                        "data": {
                            "latest_cached_date": str(latest_cached) if latest_cached else None,
                            "force_live_data": force_live_data
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + "\n")
            except: pass
            # #endregion
            # Apply filters to cached data
            ds = _filter_cached_dataset(cached_ds, min_date, max_date, assets)
            if ds is not None:
                # Ensure canonical dims order
                ds = ds.transpose("time", "asset", missing_dims="ignore")
                # If force_live_data, skip cache and fetch fresh Marketstack data
                if force_live_data:
                    # #region agent log
                    try:
                        import json
                        with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "F",
                                "location": "data_loader.py:238",
                                "message": "Skipping cache due to force_live_data",
                                "data": {"strategy_id": strategy_id},
                                "timestamp": int(__import__('time').time() * 1000)
                            }) + "\n")
                    except: pass
                    # #endregion
                    # Don't return cached data - continue to API load to get fresh Marketstack data
                    pass
                else:
                    # #region agent log
                    try:
                        import json
                        with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
                            latest_cached = get_latest_quantiacs_date(ds) if hasattr(ds, 'time') else None
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "F",
                                "location": "data_loader.py:245",
                                "message": "Returning cached data (no force_live_data)",
                                "data": {
                                    "latest_cached_date": str(latest_cached) if latest_cached else None,
                                    "force_live_data": force_live_data,
                                    "strategy_id": strategy_id
                                },
                                "timestamp": int(__import__('time').time() * 1000)
                            }) + "\n")
                    except: pass
                    # #endregion
                    return ds
            # If ds is None, cache was invalid - continue to API load

    try:
        import qnt.data as qndata  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Quantiacs qnt package not found. Activate your qntdev conda env."
        ) from e

    # Field candidates: try richest first, then fall back.
    base_fields = list(fields) if fields is not None else [
        "open", "high", "low", "close", "vol", "is_liquid"
    ]
    field_candidates = [
        base_fields,
        [f for f in base_fields if f != "is_liquid"],
        ["open", "high", "low", "close", "vol"],
        ["close"],  # absolute last resort (will error later if missing others)
    ]

    # Loader candidates: prefer load_spx_data (stable) when present.
    loaders = []
    if hasattr(qndata, "stocks") and hasattr(qndata.stocks, "load_spx_data"):
        loaders.append(qndata.stocks.load_spx_data)
    if hasattr(qndata, "stocks") and hasattr(qndata.stocks, "load_data"):
        loaders.append(qndata.stocks.load_data)
    if hasattr(qndata, "load_data"):
        loaders.append(qndata.load_data)

    if not loaders:  # pragma: no cover
        raise AttributeError("Could not find qnt data loader (stocks.load_spx_data/load_data or load_data)")

    last_err: Optional[Exception] = None

    for loader in loaders:
        for fset in field_candidates:
            try:
                # Many qnt loaders accept (min_date, max_date, assets, fields, forward_order) but not all.
                raw = _try_call(
                    loader,
                    min_date=min_date,
                    max_date=max_date,
                    assets=assets,
                    fields=fset,
                    forward_order=forward_order,
                )
                ds = _normalize_to_dataset(raw)

                # If loader ignores assets kw, filter after the fact.
                if assets is not None and "asset" in ds.dims:
                    ds = ds.sel(asset=[a for a in assets if a in set(ds.asset.values)])

                # Canonical dims order
                ds = ds.transpose("time", "asset", missing_dims="ignore")

                # Always fetch latest available data from Marketstack when enabled
                if use_marketstack and cfg.marketstack.ENABLED and "close" in ds.variables:
                    # #region agent log
                    try:
                        import json
                        with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
                            latest_date = get_latest_quantiacs_date(ds) if hasattr(ds, 'time') else None
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "A",
                                "location": "data_loader.py:297",
                                "message": "Marketstack fetch check",
                                "data": {
                                    "use_marketstack": use_marketstack,
                                    "enabled": cfg.marketstack.ENABLED,
                                    "force_live_data": force_live_data,
                                    "latest_date": str(latest_date) if latest_date else None,
                                    "has_close": "close" in ds.variables
                                },
                                "timestamp": int(__import__('time').time() * 1000)
                            }) + "\n")
                    except: pass
                    # #endregion
                    try:
                        # Get asset IDs from dataset
                        dataset_assets = list(ds.asset.values)
                        
                        # Convert asset IDs to tickers
                        asset_ticker_map = batch_convert_assets(dataset_assets)
                        tickers = [ticker for ticker in asset_ticker_map.values() if ticker is not None]
                        
                        if tickers:
                            # Check for missing days and fetch them
                            lookback_days = fill_recent_days if fill_recent_days is not None else cfg.marketstack.DEFAULT_LOOKBACK_DAYS
                            
                            # Always check for latest data (force_live_data=True for neural_alpha)
                            date_range = get_missing_date_range(ds, lookback_days=lookback_days, force_live_data=force_live_data)
                            # #region agent log
                            try:
                                import json
                                with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
                                    f.write(json.dumps({
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "B",
                                        "location": "data_loader.py:311",
                                        "message": "Gap detection result",
                                        "data": {
                                            "date_range": str(date_range),
                                            "lookback_days": lookback_days,
                                            "force_live_data": force_live_data,
                                            "ticker_count": len(tickers)
                                        },
                                        "timestamp": int(__import__('time').time() * 1000)
                                    }) + "\n")
                            except: pass
                            # #endregion
                            
                            if date_range:
                                date_from, date_to = date_range
                                # #region agent log
                                try:
                                    import json
                                    with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
                                        f.write(json.dumps({
                                            "sessionId": "debug-session",
                                            "runId": "run1",
                                            "hypothesisId": "C",
                                            "location": "data_loader.py:314",
                                            "message": "Fetching Marketstack data",
                                            "data": {
                                                "date_from": date_from,
                                                "date_to": date_to,
                                                "is_single_date": date_from == date_to
                                            },
                                            "timestamp": int(__import__('time').time() * 1000)
                                        }) + "\n")
                                except: pass
                                # #endregion
                                client = MarketstackClient()
                                
                                # Fetch the date range (could be single day or multiple days)
                                if date_from == date_to:
                                    # Single date - use most_recent method for better handling
                                    marketstack_df, fetched_date = client.fetch_most_recent_eod(
                                        symbols=tickers,
                                        target_date=date_from,
                                    )
                                    # #region agent log
                                    try:
                                        import json
                                        with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
                                            f.write(json.dumps({
                                                "sessionId": "debug-session",
                                                "runId": "run1",
                                                "hypothesisId": "D",
                                                "location": "data_loader.py:320",
                                                "message": "Marketstack fetch result (single)",
                                                "data": {
                                                    "fetched_date": fetched_date,
                                                    "row_count": len(marketstack_df),
                                                    "dates_in_df": list(marketstack_df['date'].unique())[:10] if not marketstack_df.empty and 'date' in marketstack_df.columns else []
                                                },
                                                "timestamp": int(__import__('time').time() * 1000)
                                            }) + "\n")
                                    except: pass
                                    # #endregion
                                else:
                                    # Date range - fetch all missing days
                                    marketstack_df = client.fetch_eod_for_date_range(
                                        symbols=tickers,
                                        start_date=date_from,
                                        end_date=date_to,
                                    )
                                    # #region agent log
                                    try:
                                        import json
                                        with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
                                            f.write(json.dumps({
                                                "sessionId": "debug-session",
                                                "runId": "run1",
                                                "hypothesisId": "D",
                                                "location": "data_loader.py:326",
                                                "message": "Marketstack fetch result (range)",
                                                "data": {
                                                    "row_count": len(marketstack_df),
                                                    "dates_in_df": list(marketstack_df['date'].unique())[:10] if not marketstack_df.empty and 'date' in marketstack_df.columns else []
                                                },
                                                "timestamp": int(__import__('time').time() * 1000)
                                            }) + "\n")
                                    except: pass
                                    # #endregion
                                
                                if not marketstack_df.empty:
                                    # Convert to xarray and merge
                                    marketstack_ds = marketstack_to_xarray(
                                        marketstack_df,
                                        dataset_assets,
                                    )
                                    
                                    # Merge datasets
                                    ds_before_merge = ds
                                    ds = merge_datasets(ds, marketstack_ds)
                                    
                                    # Record successful fetch with strategy context
                                    try:
                                        from q23.strategy.marketstack_telemetry import record_marketstack_fetch, FetchResult
                                        record_marketstack_fetch(
                                            strategy_id=strategy_id,
                                            symbols_count=len(tickers),
                                            date_from=date_from,
                                            date_to=date_to,
                                            fetched_date=fetched_date if date_from == date_to else None,
                                            result=FetchResult.SUCCESS,
                                            rows_fetched=len(marketstack_df),
                                            duration_ms=0.0,  # Already recorded in client
                                        )
                                    except Exception:
                                        pass  # Don't fail on telemetry
                                    
                                    # Sort by time
                                    ds = ds.sortby('time')
                                    
                                    # #region agent log
                                    try:
                                        import json
                                        with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
                                            latest_before = get_latest_quantiacs_date(ds_before_merge) if hasattr(ds_before_merge, 'time') else None
                                            latest_after = get_latest_quantiacs_date(ds) if hasattr(ds, 'time') else None
                                            f.write(json.dumps({
                                                "sessionId": "debug-session",
                                                "runId": "run1",
                                                "hypothesisId": "E",
                                                "location": "data_loader.py:340",
                                                "message": "After merge",
                                                "data": {
                                                    "latest_before": str(latest_before) if latest_before else None,
                                                    "latest_after": str(latest_after) if latest_after else None,
                                                    "time_dim_size": ds.sizes.get('time', 0) if hasattr(ds, 'sizes') else 0,
                                                    "strategy_id": strategy_id,
                                                    "marketstack_fetched": True,
                                                    "fetched_date": fetched_date if date_from == date_to else f"{date_from} to {date_to}"
                                                },
                                                "timestamp": int(__import__('time').time() * 1000)
                                            }) + "\n")
                                    except: pass
                                    # #endregion
                                    
                                    import warnings
                                    if date_from == date_to:
                                        print(f"  ✅ Marketstack: Fetched {fetched_date} ({len(marketstack_df)} rows) for {strategy_id or 'strategy'}")
                                        warnings.warn(
                                            f"Fetched Marketstack data for {fetched_date}",
                                            UserWarning,
                                        )
                                    else:
                                        print(f"  ✅ Marketstack: Fetched {date_from} to {date_to} ({len(marketstack_df)} rows) for {strategy_id or 'strategy'}")
                                        warnings.warn(
                                            f"Fetched Marketstack data for {date_from} to {date_to}",
                                            UserWarning,
                                        )
                            else:
                                # No date range returned and force_live_data is False
                                # #region agent log
                                try:
                                    import json
                                    with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
                                        f.write(json.dumps({
                                            "sessionId": "debug-session",
                                            "runId": "run1",
                                            "hypothesisId": "B",
                                            "location": "data_loader.py:559",
                                            "message": "No Marketstack fetch (no gap, force_live_data=False)",
                                            "data": {
                                                "force_live_data": force_live_data,
                                                "date_range_was_none": True
                                            },
                                            "timestamp": int(__import__('time').time() * 1000)
                                        }) + "\n")
                                except: pass
                                # #endregion
                                # Force live data but no gap detected - fetch most recent anyway
                                # #region agent log
                                try:
                                    import json
                                    with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
                                        f.write(json.dumps({
                                            "sessionId": "debug-session",
                                            "runId": "run1",
                                            "hypothesisId": "B",
                                            "location": "data_loader.py:356",
                                            "message": "Force live data path (no gap detected)",
                                            "data": {
                                                "force_live_data": force_live_data,
                                                "date_range_was_none": True
                                            },
                                            "timestamp": int(__import__('time').time() * 1000)
                                        }) + "\n")
                                except: pass
                                # #endregion
                                client = MarketstackClient()
                                marketstack_df, fetched_date = client.fetch_most_recent_eod(
                                    symbols=tickers,
                                )
                                # #region agent log
                                try:
                                    import json
                                    with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
                                        f.write(json.dumps({
                                            "sessionId": "debug-session",
                                            "runId": "run1",
                                            "hypothesisId": "D",
                                            "location": "data_loader.py:359",
                                            "message": "Force fetch result",
                                            "data": {
                                                "fetched_date": fetched_date,
                                                "row_count": len(marketstack_df),
                                                "empty": marketstack_df.empty
                                            },
                                            "timestamp": int(__import__('time').time() * 1000)
                                        }) + "\n")
                                except: pass
                                # #endregion
                                
                                if not marketstack_df.empty:
                                    marketstack_ds = marketstack_to_xarray(
                                        marketstack_df,
                                        dataset_assets,
                                    )
                                    ds_before = ds
                                    ds_before_force = ds
                                    ds = merge_datasets(ds, marketstack_ds)
                                    ds = ds.sortby('time')
                                    
                                    # #region agent log
                                    try:
                                        import json
                                        with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
                                            latest_before = get_latest_quantiacs_date(ds_before_force) if hasattr(ds_before_force, 'time') else None
                                            latest_after = get_latest_quantiacs_date(ds) if hasattr(ds, 'time') else None
                                            f.write(json.dumps({
                                                "sessionId": "debug-session",
                                                "runId": "run1",
                                                "hypothesisId": "E",
                                                "location": "data_loader.py:368",
                                                "message": "After force merge",
                                                "data": {
                                                    "latest_before": str(latest_before) if latest_before else None,
                                                    "latest_after": str(latest_after) if latest_after else None,
                                                    "strategy_id": strategy_id,
                                                    "marketstack_fetched": True,
                                                    "fetched_date": fetched_date
                                                },
                                                "timestamp": int(__import__('time').time() * 1000)
                                            }) + "\n")
                                    except: pass
                                    # #endregion
                                    
                                    print(f"  ✅ Marketstack: Fetched latest data {fetched_date} ({len(marketstack_df)} rows) for {strategy_id or 'strategy'}")
                                    import warnings
                                    warnings.warn(
                                        f"Fetched latest Marketstack data for {fetched_date}",
                                        UserWarning,
                                    )
                    except Exception as e:
                        # If Marketstack fetch fails, log warning and continue with Quantiacs data only
                        # #region agent log
                        try:
                            import json
                            import traceback
                            with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "C",
                                    "location": "data_loader.py:376",
                                    "message": "Marketstack fetch exception",
                                    "data": {
                                        "error": str(e),
                                        "error_type": type(e).__name__,
                                        "traceback": traceback.format_exc()[:500]
                                    },
                                    "timestamp": int(__import__('time').time() * 1000)
                                }) + "\n")
                        except: pass
                        # #endregion
                        import warnings
                        warnings.warn(
                            f"Failed to fetch Marketstack data: {e}. "
                            "Continuing with Quantiacs data only.",
                            UserWarning,
                        )

                # Save to file cache (full dataset for maximum reuse)
                # Only cache if we have valid data
                if use_file_cache and "close" in ds.variables:
                    save_to_cache(cache_key, ds)
                    # Clean up old cache files (keep last 3 days)
                    cleanup_old_cache(keep_days=3)

                # Require at least close/vol; other fields validated later.
                return ds
            except KeyError as e:
                last_err = e
                # Classic case: requested field doesn't exist in qnt's 'field' coordinate.
                if _looks_like_missing_field_error(e):
                    continue
                # Other KeyErrors likely real (e.g., missing asset). Re-raise.
                raise
            except Exception as e:
                last_err = e
                # If it's a missing-field error wrapped differently, keep trying.
                if _looks_like_missing_field_error(e):
                    continue
                # Some loaders won't accept fields/assets at all; _try_call handles TypeError,
                # but other exceptions should stop this loader attempt.
                continue

    # If we got here, none worked.
    if last_err is not None:
        raise RuntimeError(f"Failed to load Quantiacs stocks data. Last error: {last_err}") from last_err
    raise RuntimeError("Failed to load Quantiacs stocks data (unknown error).")


# -----------------------------------------------------------------------------
# Universe selection (SPX list + exchange filter)
# -----------------------------------------------------------------------------

# Simple memoization cache for SPX list loads (keyed by min_date)
# This avoids redundant Quantiacs API calls when loading the same universe multiple times
_spx_list_cache: Dict[str, Optional[pd.DataFrame]] = {}


def _load_spx_universe_ids(
    *,
    min_date: str,
    exchanges: Optional[Sequence[str]],
    pinned: Optional[Sequence[str]],
) -> Optional[List[str]]:
    """Best-effort universe selection using qnt.data.stocks.load_spx_list.
    
    OPTIMIZED: Uses memoization to cache SPX list loads by min_date,
    avoiding redundant Quantiacs API calls when the same date is requested.

    Returns list of asset ids or None if unavailable.
    """
    if pd is None:
        return None
    try:
        import qnt.data as qndata  # type: ignore
    except Exception:
        return None

    if not (hasattr(qndata, "stocks") and hasattr(qndata.stocks, "load_spx_list")):
        return None

    # Check cache first
    if min_date in _spx_list_cache:
        df = _spx_list_cache[min_date]
    else:
        # Load from Quantiacs
        try:
            df = pd.DataFrame(qndata.stocks.load_spx_list(min_date=min_date))
            # Cache the result (even if empty, to avoid retrying)
            _spx_list_cache[min_date] = df
        except Exception:
            _spx_list_cache[min_date] = None
            return None

    if df is None or df.empty or "id" not in df.columns:
        return None

    ids = df["id"].dropna().astype(str).unique().tolist()

    # Exchange filter if the column exists.
    if exchanges and "exchange" in df.columns:
        ex = set(str(e).upper() for e in exchanges)
        m = df["exchange"].astype(str).str.upper().isin(ex)
        ids = df.loc[m, "id"].dropna().astype(str).unique().tolist()

    # Ensure pinned assets are included.
    if pinned:
        pin = [str(x) for x in pinned]
        s = set(ids)
        for p in pin:
            if p not in s:
                ids.append(p)

    return ids


def pinned_indices(asset_ids: Sequence[str], pinned: Optional[Sequence[str]] = None) -> np.ndarray:
    """Return indices for pinned tickers to always include (if present)."""
    if not pinned:
        return np.array([], dtype=int)
    aset = {str(a): i for i, a in enumerate(asset_ids)}
    idx = [aset[str(a)] for a in pinned if str(a) in aset]
    return np.array(idx, dtype=int)


def compute_returns(close: "xr.DataArray") -> "xr.DataArray":
    """Compute daily returns from close prices."""
    _require_xr()
    r = close / close.shift(time=1) - 1.0
    return r.fillna(0.0)


def load_market_data(
    *,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    exchanges: Optional[Sequence[str]] = None,
    pinned: Optional[Sequence[str]] = None,
    assets: Optional[Sequence[str]] = None,
    fill_recent_days: Optional[int] = None,
    use_marketstack: bool = True,
    force_live_data: bool = False,
    strategy_id: Optional[str] = None,
) -> MarketDataBundle:
    """High-level loader returning a normalized MarketDataBundle.

    Behavior:
    - If `assets` is provided: load exactly those (best-effort).
    - Else: use SPX list filtered by `exchanges` when possible (matches your v4 script).
    - If `force_live_data` is True, always fetches latest Marketstack data for pre-market runs.
    
    Args:
        min_date: Minimum date to load
        max_date: Maximum date to load (optional)
        exchanges: Exchange filters (optional)
        pinned: Pinned asset IDs (optional)
        assets: Specific asset IDs to load (optional)
        fill_recent_days: Number of recent days to fill from Marketstack
        use_marketstack: Enable Marketstack integration (default True)
        force_live_data: Force fetching latest data even if no gap detected (default False)
    """
    _require_xr()

    min_date_eff = min_date or cfg.strategy.MIN_DATE
    exchanges_eff = list(exchanges) if exchanges is not None else list(cfg.strategy.EXCHANGES)

    if assets is None:
        assets = _load_spx_universe_ids(min_date=min_date_eff, exchanges=exchanges_eff, pinned=pinned)

    # #region agent log
    try:
        import json
        import time
        with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "A",
                "location": "data_loader.py:802",
                "message": "load_market_data entry",
                "data": {
                    "strategy_id": strategy_id,
                    "use_marketstack": use_marketstack,
                    "force_live_data": force_live_data,
                    "min_date": min_date_eff,
                    "max_date": max_date
                },
                "timestamp": int(time.time() * 1000)
            }) + "\n")
    except: pass
    # #endregion
    
    ds = load_quantiacs_stocks(
        min_date=min_date_eff,
        max_date=max_date,
        assets=assets,
        fill_recent_days=fill_recent_days,
        use_marketstack=use_marketstack,
        force_live_data=force_live_data,
        strategy_id=strategy_id,
    )
    
    # #region agent log
    try:
        import json
        import time
        latest_date = get_latest_quantiacs_date(ds) if hasattr(ds, 'time') else None
        with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "A",
                "location": "data_loader.py:820",
                "message": "load_quantiacs_stocks returned",
                "data": {
                    "latest_date": str(latest_date) if latest_date else None,
                    "time_dim_size": ds.sizes.get('time', 0) if hasattr(ds, 'sizes') else 0
                },
                "timestamp": int(time.time() * 1000)
            }) + "\n")
    except: pass
    # #endregion

    # Ensure required vars exist (some loaders may return only a subset)
    need = {"open", "high", "low", "close", "vol"}
    missing = [v for v in need if v not in ds.variables]
    if missing:
        raise ValueError(
            f"Loaded dataset missing required vars: {missing}. " 
            f"Available vars={list(ds.variables)}"
        )

    # Ensure is_liquid exists (infer if needed)
    if "is_liquid" not in ds.variables:
        liq = xr.where(np.isfinite(ds["vol"]) & (ds["vol"] > 0), 1.0, 0.0)
        ds["is_liquid"] = liq

    # Returns
    rets = compute_returns(ds["close"])

    # Pins
    asset_ids = list(map(str, ds.asset.values))
    pin_idx = pinned_indices(asset_ids, pinned=pinned)

    # Get latest date from dataset
    latest_date_str = None
    if ds.sizes.get("time", 0) > 0:
        try:
            latest_date = get_latest_quantiacs_date(ds)
            if latest_date:
                latest_date_str = latest_date.strftime("%Y-%m-%d")
        except Exception:
            pass
    
    # Check if Marketstack was used
    marketstack_used = False
    marketstack_fetched_date = None
    if use_marketstack and cfg.marketstack.ENABLED:
        try:
            from q23.strategy.marketstack_telemetry import get_telemetry_manager
            tel = get_telemetry_manager().get_telemetry()
            if tel.last_fetch and tel.last_fetch.strategy_id == strategy_id:
                marketstack_used = tel.last_fetch.result.value == "success"
                marketstack_fetched_date = tel.last_fetch.fetched_date
        except Exception:
            pass
    
    meta = {
        "min_date": min_date_eff,
        "max_date": max_date,
        "latest_date": latest_date_str,  # Actual latest date in dataset
        "n_assets": int(ds.sizes.get("asset", 0)),
        "n_days": int(ds.sizes.get("time", 0)),
        "exchanges": exchanges_eff,
        "assets_mode": "explicit" if assets is not None else "spx_list",
        "marketstack": {
            "enabled": use_marketstack and cfg.marketstack.ENABLED,
            "used": marketstack_used,
            "fetched_date": marketstack_fetched_date,
            "force_live_data": force_live_data,
        },
    }

    return MarketDataBundle(
        data=ds,
        returns=rets,
        asset_ids=asset_ids,
        pin_idx=pin_idx,
        meta=meta,
    )
