"""q23.data.providers.marketstack

DataProvider adapter over Marketstack, used to gap-fill the most recent
day(s) that Quantiacs hasn't published yet. Thin wrapper around the existing
MarketstackClient — this is an adapter, not a reimplementation.

Marketstack speaks tickers; the rest of the harness speaks Quantiacs asset
ids (e.g. "NAS:AAPL"). This provider accepts Quantiacs asset ids (matching
the DataProvider.get_prices `assets` contract used elsewhere) and handles
the ticker translation + reverse-mapping internally via symbol_mapper, so
callers never have to think about the identity bridge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from q23.data.provider import BaseDataProvider

if TYPE_CHECKING:
    from q23.strategy.marketstack_client import MarketstackClient

# NOTE: q23.strategy.marketstack_client / symbol_mapper / data_merger are
# imported lazily (inside methods, not at module scope) rather than at the
# top of this file. q23.strategy's package __init__ eagerly imports
# q23.strategy.data_loader, which imports this module — a module-level
# import here of anything under q23.strategy would close a circular import
# loop whenever q23.data.providers.marketstack is the *first* thing touched
# in a process (e.g. a standalone script or test importing the provider
# directly, before q23.strategy has been loaded). Deferring the import to
# call time sidesteps the cycle entirely; this mirrors the lazy-import
# pattern q23.strategy.data_loader itself already uses for
# marketstack_telemetry.


class MarketstackProvider(BaseDataProvider):
    """DataProvider adapter for Marketstack EOD data."""

    name = "marketstack"

    def __init__(self, client: Optional["MarketstackClient"] = None) -> None:
        if client is None:
            from q23.strategy.marketstack_client import MarketstackClient as _MarketstackClient
            client = _MarketstackClient()
        self._client = client

    def tickers_for(self, assets: Sequence[str]) -> List[str]:
        from q23.strategy.symbol_mapper import batch_convert_assets
        asset_ticker_map = batch_convert_assets(list(assets))
        return [ticker for ticker in asset_ticker_map.values() if ticker is not None]

    def get_prices(
        self,
        *,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        assets: Optional[Sequence[str]] = None,
        fields: Optional[Sequence[str]] = None,
    ) -> "xr.Dataset":
        """Fetch EOD data for a date range, keyed back to Quantiacs asset ids.

        `assets` must be provided (Quantiacs asset ids) — Marketstack has no
        independent universe concept in this harness, it only supplements a
        Quantiacs-defined universe.
        """
        ds, _df = self.get_prices_with_raw(min_date=min_date, max_date=max_date, assets=assets)
        return ds

    def get_prices_with_raw(
        self,
        *,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        assets: Optional[Sequence[str]] = None,
    ) -> Tuple["xr.Dataset", "pd.DataFrame"]:
        """Same as get_prices, but also returns the raw Marketstack DataFrame
        (e.g. for row-count logging/telemetry at the call site)."""
        if not assets:
            raise ValueError("MarketstackProvider.get_prices requires `assets`")
        if not min_date or not max_date:
            raise ValueError("MarketstackProvider.get_prices requires min_date and max_date")

        from q23.strategy.data_merger import marketstack_to_xarray

        tickers = self.tickers_for(assets)
        df = self._client.fetch_eod_for_date_range(
            symbols=tickers,
            start_date=min_date,
            end_date=max_date,
        )
        return marketstack_to_xarray(df, list(assets)), df

    def get_latest_prices(
        self,
        *,
        assets: Sequence[str],
        target_date: Optional[str] = None,
    ) -> Tuple["xr.Dataset", "pd.DataFrame", Optional[str]]:
        """Fetch the most recent available EOD snapshot.

        Mirrors MarketstackClient.fetch_most_recent_eod's single-date fetch
        path (used when the gap detector finds exactly one missing day, or
        when force_live_data requests "whatever is freshest" with no gap).

        Returns (dataset keyed by Quantiacs asset ids, raw DataFrame, fetched_date)
        so callers can still record telemetry / row counts the way
        q23.strategy.data_loader does today.
        """
        from q23.strategy.data_merger import marketstack_to_xarray

        tickers = self.tickers_for(assets)
        df, fetched_date = self._client.fetch_most_recent_eod(
            symbols=tickers,
            target_date=target_date,
        )
        ds = marketstack_to_xarray(df, list(assets))
        return ds, df, fetched_date
