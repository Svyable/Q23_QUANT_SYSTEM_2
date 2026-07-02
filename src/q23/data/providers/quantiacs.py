"""q23.data.providers.quantiacs

DataProvider adapter over Quantiacs' qnt.data package.

Isolates all qnt.data API churn (loader entry points, field-name quirks,
kwarg signature differences across qnt versions) behind QuantiacsProvider.
This is a direct extraction of logic that previously lived inline in
q23.strategy.data_loader — no behavior change, just a stable seam.
"""

from __future__ import annotations

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

from q23.data.provider import BaseDataProvider


def _require_xr() -> None:
    if xr is None:
        raise ImportError("xarray is required for q23.data.providers.quantiacs")


def _try_call(fn, **kwargs):
    """Try calling `fn` with a subset of kwargs (only for signature mismatch).

    Survives minor parameter-name differences across qnt versions. Only
    retries on TypeError (unexpected kwarg).
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

    try:
        if "min_date" in kwargs:
            return fn(kwargs["min_date"])
    except Exception:
        pass

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
        raise ValueError("Unexpected xarray.DataArray without 'field' dimension.")
    raise TypeError(f"Expected xarray Dataset or DataArray, got {type(obj)}")


class QuantiacsProvider(BaseDataProvider):
    """DataProvider adapter for Quantiacs (qnt.data).

    Point-in-time, survivorship-bias-free equity universe + daily OHLCV.
    This is the harness's price-of-record source; other providers only
    ever supplement it (see MarketstackProvider).
    """

    name = "quantiacs"

    def __init__(self) -> None:
        self._spx_list_cache: Dict[str, Optional["pd.DataFrame"]] = {}

    def get_prices(
        self,
        *,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        assets: Optional[Sequence[str]] = None,
        fields: Optional[Sequence[str]] = None,
        forward_order: bool = True,
    ) -> "xr.Dataset":
        """Fetch OHLCV(+is_liquid) from qnt.data, tolerating loader/field drift.

        Tries multiple qnt loader entry points (load_spx_data, stocks.load_data,
        qnt.data.load_data) and retries with reduced field sets if qnt rejects a
        requested field (common with 'is_liquid').
        """
        _require_xr()

        try:
            import qnt.data as qndata  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Quantiacs qnt package not found. Activate your qntdev conda env."
            ) from e

        base_fields = list(fields) if fields is not None else [
            "open", "high", "low", "close", "vol", "is_liquid"
        ]
        field_candidates = [
            base_fields,
            [f for f in base_fields if f != "is_liquid"],
            ["open", "high", "low", "close", "vol"],
            ["close"],  # absolute last resort (will error later if missing others)
        ]

        loaders = []
        if hasattr(qndata, "stocks") and hasattr(qndata.stocks, "load_spx_data"):
            loaders.append(qndata.stocks.load_spx_data)
        if hasattr(qndata, "stocks") and hasattr(qndata.stocks, "load_data"):
            loaders.append(qndata.stocks.load_data)
        if hasattr(qndata, "load_data"):
            loaders.append(qndata.load_data)

        if not loaders:  # pragma: no cover
            raise AttributeError(
                "Could not find qnt data loader (stocks.load_spx_data/load_data or load_data)"
            )

        last_err: Optional[Exception] = None

        for loader in loaders:
            for fset in field_candidates:
                try:
                    raw = _try_call(
                        loader,
                        min_date=min_date,
                        max_date=max_date,
                        assets=assets,
                        fields=fset,
                        forward_order=forward_order,
                    )
                    ds = _normalize_to_dataset(raw)

                    if assets is not None and "asset" in ds.dims:
                        ds = ds.sel(asset=[a for a in assets if a in set(ds.asset.values)])

                    ds = ds.transpose("time", "asset", missing_dims="ignore")
                    return ds
                except KeyError as e:
                    last_err = e
                    if _looks_like_missing_field_error(e):
                        continue
                    raise
                except Exception as e:
                    last_err = e
                    if _looks_like_missing_field_error(e):
                        continue
                    continue

        if last_err is not None:
            raise RuntimeError(f"Failed to load Quantiacs stocks data. Last error: {last_err}") from last_err
        raise RuntimeError("Failed to load Quantiacs stocks data (unknown error).")

    def get_universe(
        self,
        *,
        min_date: str,
        exchanges: Optional[Sequence[str]] = None,
        pinned: Optional[Sequence[str]] = None,
    ) -> Optional[List[str]]:
        """Best-effort universe selection via qnt.data.stocks.load_spx_list.

        Memoizes SPX list loads by min_date to avoid redundant API calls
        when the same date is requested repeatedly in one process.
        Returns None if the universe can't be determined (caller should treat
        that as "fall back to explicit assets", distinct from a known-empty list).
        """
        if pd is None:
            return None
        try:
            import qnt.data as qndata  # type: ignore
        except Exception:
            return None

        if not (hasattr(qndata, "stocks") and hasattr(qndata.stocks, "load_spx_list")):
            return None

        if min_date in self._spx_list_cache:
            df = self._spx_list_cache[min_date]
        else:
            try:
                df = pd.DataFrame(qndata.stocks.load_spx_list(min_date=min_date))
                self._spx_list_cache[min_date] = df
            except Exception:
                self._spx_list_cache[min_date] = None
                return None

        if df is None or df.empty or "id" not in df.columns:
            return None

        ids = df["id"].dropna().astype(str).unique().tolist()

        if exchanges and "exchange" in df.columns:
            ex = set(str(e).upper() for e in exchanges)
            m = df["exchange"].astype(str).str.upper().isin(ex)
            ids = df.loc[m, "id"].dropna().astype(str).unique().tolist()

        if pinned:
            pin = [str(x) for x in pinned]
            s = set(ids)
            for p in pin:
                if p not in s:
                    ids.append(p)

        return ids
