"""q23.strategy.data_loader

Vendor-agnostic market data loading + universe selection.

Goals
-----
- Hide vendor API churn behind a stable internal interface (MarketDataBundle),
  itself built on top of q23.data.provider.DataProvider adapters.
- Load only the fields we need, but be robust if some optional fields are unavailable.
- Keep the strategy + dashboard aligned with a consistent schema:
    xr.Dataset vars: open, high, low, close, vol, (is_liquid optional)
    dims: time, asset

This module owns the *orchestration* (file caching, gap detection, merging
a primary price provider with a gap-fill provider) — it no longer talks to
qnt.data or the Marketstack API directly. That vendor-specific logic lives
in q23.data.providers.quantiacs.QuantiacsProvider and
q23.data.providers.marketstack.MarketstackProvider, respectively.
"""

from __future__ import annotations

import warnings
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
from q23.strategy.data_gap_detector import (
    get_missing_date_range,
    get_latest_quantiacs_date,
)
from q23.strategy.data_merger import merge_datasets
from q23.data.providers.quantiacs import QuantiacsProvider
from q23.data.providers.marketstack import MarketstackProvider


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


# Module-level provider singletons. QuantiacsProvider caches SPX list loads
# per min_date internally; sharing one instance across calls in a process
# preserves that memoization exactly as it worked before this refactor.
_quantiacs_provider = QuantiacsProvider()


def _is_valid_dataset(ds: "xr.Dataset") -> bool:
    """Check if a dataset has the required structure for market data."""
    _require_xr()

    if "time" not in ds.dims or "asset" not in ds.dims:
        return False
    if ds.sizes.get("time", 0) == 0 or ds.sizes.get("asset", 0) == 0:
        return False
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

    if not _is_valid_dataset(ds):
        return None

    try:
        if min_date and "time" in ds.dims:
            ds = ds.sel(time=slice(min_date, None))
        if max_date and "time" in ds.dims:
            ds = ds.sel(time=slice(None, max_date))

        if assets is not None and "asset" in ds.dims:
            valid_assets = [a for a in assets if a in set(ds.asset.values)]
            if valid_assets:
                ds = ds.sel(asset=valid_assets)

        if not _is_valid_dataset(ds):
            return None

        return ds
    except Exception:
        return None


def _fetch_marketstack_gap_fill(
    ds: "xr.Dataset",
    *,
    dataset_assets: List[str],
    fill_recent_days: Optional[int],
    force_live_data: bool,
    strategy_id: Optional[str],
) -> "xr.Dataset":
    """Fill in missing recent days using Marketstack, matching `ds`'s assets.

    Mirrors the two Quantiacs-contest-era fetch shapes: a single missing day
    (fetch_most_recent_eod) vs. a genuine range (fetch_eod_for_date_range).
    Any failure here is non-fatal — we fall back to Quantiacs-only data and
    warn, since Marketstack is a supplement, not the source of record.
    """
    provider = MarketstackProvider()

    # If none of these assets have a ticker mapping (id-translation.csv
    # doesn't cover this universe), there is nothing Marketstack can look
    # up. Bail out silently rather than calling into the client with an
    # empty symbol list, which raises deep inside MarketstackClient.
    if not provider.tickers_for(dataset_assets):
        return ds

    try:
        lookback_days = fill_recent_days if fill_recent_days is not None else cfg.marketstack.DEFAULT_LOOKBACK_DAYS
        date_range = get_missing_date_range(ds, lookback_days=lookback_days, force_live_data=force_live_data)

        if date_range:
            date_from, date_to = date_range
            if date_from == date_to:
                marketstack_ds, marketstack_df, fetched_date = provider.get_latest_prices(
                    assets=dataset_assets,
                    target_date=date_from,
                )
            else:
                marketstack_ds, marketstack_df = provider.get_prices_with_raw(
                    min_date=date_from,
                    max_date=date_to,
                    assets=dataset_assets,
                )
                fetched_date = None

            fetched_rows = len(marketstack_df)
            has_data = marketstack_ds.sizes.get("time", 0) > 0

            if has_data:
                ds = merge_datasets(ds, marketstack_ds)
                ds = ds.sortby("time")

                try:
                    from q23.strategy.marketstack_telemetry import record_marketstack_fetch, FetchResult
                    record_marketstack_fetch(
                        strategy_id=strategy_id,
                        symbols_count=len(dataset_assets),
                        date_from=date_from,
                        date_to=date_to,
                        fetched_date=fetched_date if date_from == date_to else None,
                        result=FetchResult.SUCCESS,
                        rows_fetched=fetched_rows,
                        duration_ms=0.0,  # Already recorded in client
                    )
                except Exception:
                    pass  # Don't fail on telemetry

                if date_from == date_to:
                    print(f"  ✅ Marketstack: Fetched {fetched_date} ({fetched_rows} rows) for {strategy_id or 'strategy'}")
                    warnings.warn(f"Fetched Marketstack data for {fetched_date}", UserWarning)
                else:
                    print(f"  ✅ Marketstack: Fetched {date_from} to {date_to} ({fetched_rows} rows) for {strategy_id or 'strategy'}")
                    warnings.warn(f"Fetched Marketstack data for {date_from} to {date_to}", UserWarning)
        else:
            # No gap detected and force_live_data requests the freshest snapshot anyway.
            marketstack_ds, marketstack_df, fetched_date = provider.get_latest_prices(assets=dataset_assets)

            if marketstack_ds.sizes.get("time", 0) > 0:
                ds = merge_datasets(ds, marketstack_ds)
                ds = ds.sortby("time")
                print(f"  ✅ Marketstack: Fetched latest data {fetched_date} ({len(marketstack_df)} rows) for {strategy_id or 'strategy'}")
                warnings.warn(f"Fetched latest Marketstack data for {fetched_date}", UserWarning)
    except Exception as e:
        warnings.warn(
            f"Failed to fetch Marketstack data: {e}. Continuing with Quantiacs data only.",
            UserWarning,
        )

    return ds


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
    """Load equity OHLCV (and is_liquid if available) via QuantiacsProvider.

    Uses two-level caching:
    1. File cache: Daily .nc files in .cache/data/YYYYMMDD/ (persists across sessions)
    2. In-memory: Via caller (load_market_data uses MarketDataBundle caching)

    Optionally fetches recent missing days from Marketstack via
    MarketstackProvider to patch the most recent day(s) Quantiacs hasn't
    published yet.

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

    # Check file cache first (SPX is the main data source).
    # We cache the full dataset without asset filtering for maximum reuse.
    cache_key = "SPX"

    if use_file_cache and is_cache_valid(cache_key):
        cached_ds = load_cached_data(cache_key)
        if cached_ds is not None:
            ds = _filter_cached_dataset(cached_ds, min_date, max_date, assets)
            if ds is not None:
                ds = ds.transpose("time", "asset", missing_dims="ignore")
                if not force_live_data:
                    return ds
                # force_live_data: don't return cached data, continue to fetch fresh Marketstack data
            # If ds is None, cache was invalid - continue to API load

    ds = _quantiacs_provider.get_prices(
        min_date=min_date,
        max_date=max_date,
        assets=assets,
        fields=fields,
        forward_order=forward_order,
    )

    if use_marketstack and cfg.marketstack.ENABLED and "close" in ds.variables:
        dataset_assets = list(ds.asset.values)
        ds = _fetch_marketstack_gap_fill(
            ds,
            dataset_assets=dataset_assets,
            fill_recent_days=fill_recent_days,
            force_live_data=force_live_data,
            strategy_id=strategy_id,
        )

    if use_file_cache and "close" in ds.variables:
        save_to_cache(cache_key, ds)
        cleanup_old_cache(keep_days=3)

    return ds


def _load_spx_universe_ids(
    *,
    min_date: str,
    exchanges: Optional[Sequence[str]],
    pinned: Optional[Sequence[str]],
) -> Optional[List[str]]:
    """Best-effort universe selection via QuantiacsProvider.get_universe.

    Returns list of asset ids or None if unavailable.
    """
    return _quantiacs_provider.get_universe(min_date=min_date, exchanges=exchanges, pinned=pinned)


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

    ds = load_quantiacs_stocks(
        min_date=min_date_eff,
        max_date=max_date,
        assets=assets,
        fill_recent_days=fill_recent_days,
        use_marketstack=use_marketstack,
        force_live_data=force_live_data,
        strategy_id=strategy_id,
    )

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
