"""Unit tests for q23.data.providers.marketstack.MarketstackProvider.

MarketstackClient is injected directly (constructor arg) rather than
mocked via monkeypatch, since MarketstackProvider supports dependency
injection for exactly this purpose. batch_convert_assets and
marketstack_to_xarray are patched at their source module (they're imported
lazily inside MarketstackProvider methods to avoid a circular import with
q23.strategy's package __init__ — see the module docstring in
q23/data/providers/marketstack.py).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest
import xarray as xr

from q23.data.providers.marketstack import MarketstackProvider


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def patch_symbol_mapper(monkeypatch):
    import q23.strategy.symbol_mapper as symbol_mapper

    def fake_batch_convert(assets):
        # NAS:AAPL -> AAPL, drop anything without a colon-prefixed exchange
        return {a: a.split(":")[-1] if ":" in a else None for a in assets}

    monkeypatch.setattr(symbol_mapper, "batch_convert_assets", fake_batch_convert)
    return fake_batch_convert


@pytest.fixture
def patch_data_merger(monkeypatch):
    import q23.strategy.data_merger as data_merger

    def fake_marketstack_to_xarray(df, asset_ids):
        time = pd.date_range("2026-06-29", periods=1)
        return xr.Dataset(
            {"close": (("time", "asset"), [[1.0] * len(asset_ids)])},
            coords={"time": time, "asset": asset_ids},
        )

    monkeypatch.setattr(data_merger, "marketstack_to_xarray", fake_marketstack_to_xarray)
    return fake_marketstack_to_xarray


class TestTickersFor:
    def test_drops_unmapped_assets(self, mock_client, patch_symbol_mapper):
        provider = MarketstackProvider(client=mock_client)
        tickers = provider.tickers_for(["NAS:AAPL", "UNMAPPABLE", "NAS:MSFT"])
        assert tickers == ["AAPL", "MSFT"]


class TestGetPricesWithRaw:
    def test_requires_assets(self, mock_client):
        provider = MarketstackProvider(client=mock_client)
        with pytest.raises(ValueError, match="requires `assets`"):
            provider.get_prices_with_raw(min_date="2026-06-01", max_date="2026-06-02", assets=[])

    def test_requires_date_range(self, mock_client):
        provider = MarketstackProvider(client=mock_client)
        with pytest.raises(ValueError, match="requires min_date and max_date"):
            provider.get_prices_with_raw(assets=["NAS:AAPL"])

    def test_delegates_to_client_with_translated_tickers(
        self, mock_client, patch_symbol_mapper, patch_data_merger
    ):
        mock_client.fetch_eod_for_date_range.return_value = pd.DataFrame(
            {"symbol": ["AAPL"], "date": ["2026-06-29"], "close": [200.0]}
        )
        provider = MarketstackProvider(client=mock_client)

        ds, df = provider.get_prices_with_raw(
            min_date="2026-06-01", max_date="2026-06-29", assets=["NAS:AAPL", "NAS:MSFT"]
        )

        mock_client.fetch_eod_for_date_range.assert_called_once_with(
            symbols=["AAPL", "MSFT"], start_date="2026-06-01", end_date="2026-06-29"
        )
        assert "close" in ds.variables
        assert list(ds.asset.values) == ["NAS:AAPL", "NAS:MSFT"]
        assert len(df) == 1

    def test_get_prices_returns_dataset_only(
        self, mock_client, patch_symbol_mapper, patch_data_merger
    ):
        mock_client.fetch_eod_for_date_range.return_value = pd.DataFrame()
        provider = MarketstackProvider(client=mock_client)
        ds = provider.get_prices(min_date="2026-06-01", max_date="2026-06-29", assets=["NAS:AAPL"])
        assert isinstance(ds, xr.Dataset)


class TestGetLatestPrices:
    def test_passes_target_date_through(self, mock_client, patch_symbol_mapper, patch_data_merger):
        mock_client.fetch_most_recent_eod.return_value = (
            pd.DataFrame({"symbol": ["AAPL"], "date": ["2026-06-29"], "close": [200.0]}),
            "2026-06-29",
        )
        provider = MarketstackProvider(client=mock_client)

        ds, df, fetched_date = provider.get_latest_prices(assets=["NAS:AAPL"], target_date="2026-06-29")

        mock_client.fetch_most_recent_eod.assert_called_once_with(
            symbols=["AAPL"], target_date="2026-06-29"
        )
        assert fetched_date == "2026-06-29"
        assert len(df) == 1

    def test_no_target_date_defaults_to_none(self, mock_client, patch_symbol_mapper, patch_data_merger):
        mock_client.fetch_most_recent_eod.return_value = (pd.DataFrame(), None)
        provider = MarketstackProvider(client=mock_client)

        provider.get_latest_prices(assets=["NAS:AAPL"])

        mock_client.fetch_most_recent_eod.assert_called_once_with(symbols=["AAPL"], target_date=None)
