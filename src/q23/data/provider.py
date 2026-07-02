"""q23.data.provider

Vendor-agnostic data access contract.

Every data vendor (Quantiacs, Marketstack, and future additions like Daloopa
or Quiver Quant) implements this contract instead of being called directly
from strategy code. q23.strategy.engine, q23.strategy.factors, and the
dashboard should only ever see MarketDataBundle / DataFrames coming out of a
DataProvider — never a vendor SDK.

Not every provider supports every capability: a pure fundamentals vendor has
no reason to implement get_prices, and a pure price vendor has no reason to
implement get_fundamentals. BaseDataProvider raises NotImplementedError for
anything a subclass doesn't override, so callers get a clear error instead
of a silent no-op if they ask a provider for something it doesn't do.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, runtime_checkable

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


@runtime_checkable
class DataProvider(Protocol):
    """Structural type describing the shape of a data vendor adapter."""

    def get_prices(
        self,
        *,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        assets: Optional[Sequence[str]] = None,
        fields: Optional[Sequence[str]] = None,
    ) -> "xr.Dataset":
        """OHLCV(+) data as an xr.Dataset with (time, asset) dims."""
        ...

    def get_universe(
        self,
        *,
        min_date: str,
        exchanges: Optional[Sequence[str]] = None,
        pinned: Optional[Sequence[str]] = None,
    ) -> Optional[List[str]]:
        """Point-in-time asset ids for a universe (e.g. SPX/NDX constituents).

        Returns None (not []) if the universe can't be determined — callers
        distinguish "unavailable, fall back to explicit assets" from "known
        empty universe".
        """
        ...

    def get_fundamentals(
        self,
        *,
        assets: Sequence[str],
        as_of: Optional[str] = None,
    ) -> "pd.DataFrame":
        """Fundamentals as a long DataFrame: asset, metric, value, filing_date."""
        ...

    def get_alt_signals(
        self,
        *,
        assets: Sequence[str],
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
    ) -> "pd.DataFrame":
        """Alternative-data signals as a long DataFrame: asset, date, signal, value."""
        ...


class BaseDataProvider:
    """Convenience base class for DataProvider adapters.

    Subclass and override only the capabilities the vendor actually offers;
    the rest raise NotImplementedError with the vendor's name for a clear
    error message instead of an AttributeError deep in strategy code.
    """

    name: str = "base"

    def get_prices(
        self,
        *,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        assets: Optional[Sequence[str]] = None,
        fields: Optional[Sequence[str]] = None,
    ) -> "xr.Dataset":
        raise NotImplementedError(f"{self.name} does not implement get_prices")

    def get_universe(
        self,
        *,
        min_date: str,
        exchanges: Optional[Sequence[str]] = None,
        pinned: Optional[Sequence[str]] = None,
    ) -> Optional[List[str]]:
        raise NotImplementedError(f"{self.name} does not implement get_universe")

    def get_fundamentals(
        self,
        *,
        assets: Sequence[str],
        as_of: Optional[str] = None,
    ) -> "pd.DataFrame":
        raise NotImplementedError(f"{self.name} does not implement get_fundamentals")

    def get_alt_signals(
        self,
        *,
        assets: Sequence[str],
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
    ) -> "pd.DataFrame":
        raise NotImplementedError(f"{self.name} does not implement get_alt_signals")
