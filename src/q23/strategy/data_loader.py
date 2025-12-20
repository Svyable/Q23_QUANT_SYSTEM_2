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
) -> "xr.Dataset":
    """Load equity OHLCV (and is_liquid if available) from Quantiacs.

    Uses two-level caching:
    1. File cache: Daily .nc files in .cache/data/YYYYMMDD/ (persists across sessions)
    2. In-memory: Via caller (load_market_data uses MarketDataBundle caching)

    Robustness:
    - Tries multiple qnt loader entry points (load_spx_data, stocks.load_data, qnt.data.load_data)
    - Retries with reduced field sets if qnt rejects a requested field (common with 'is_liquid')
    
    Args:
        min_date: Minimum date to load
        max_date: Maximum date to load (optional)
        assets: Specific asset IDs to load (optional)
        fields: Specific fields to load (optional)
        forward_order: Whether to sort by time ascending
        use_file_cache: Whether to use file-based daily cache (default True)
    
    Returns:
        xr.Dataset with OHLCV data
    """
    _require_xr()

    min_date = min_date or cfg.strategy.MIN_DATE
    
    # Check file cache first (SPX is the main data source)
    # We cache the full dataset without asset filtering for maximum reuse
    cache_key = "SPX"
    
    if use_file_cache and is_cache_valid(cache_key):
        cached_ds = load_cached_data(cache_key)
        if cached_ds is not None:
            # Apply filters to cached data
            ds = _filter_cached_dataset(cached_ds, min_date, max_date, assets)
            if ds is not None:
                # Ensure canonical dims order
                ds = ds.transpose("time", "asset", missing_dims="ignore")
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
) -> MarketDataBundle:
    """High-level loader returning a normalized MarketDataBundle.

    Behavior:
    - If `assets` is provided: load exactly those (best-effort).
    - Else: use SPX list filtered by `exchanges` when possible (matches your v4 script).
    """
    _require_xr()

    min_date_eff = min_date or cfg.strategy.MIN_DATE
    exchanges_eff = list(exchanges) if exchanges is not None else list(cfg.strategy.EXCHANGES)

    if assets is None:
        assets = _load_spx_universe_ids(min_date=min_date_eff, exchanges=exchanges_eff, pinned=pinned)

    ds = load_quantiacs_stocks(min_date=min_date_eff, max_date=max_date, assets=assets)

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

    meta = {
        "min_date": min_date_eff,
        "max_date": max_date,
        "n_assets": int(ds.sizes.get("asset", 0)),
        "n_days": int(ds.sizes.get("time", 0)),
        "exchanges": exchanges_eff,
        "assets_mode": "explicit" if assets is not None else "spx_list",
    }

    return MarketDataBundle(
        data=ds,
        returns=rets,
        asset_ids=asset_ids,
        pin_idx=pin_idx,
        meta=meta,
    )
