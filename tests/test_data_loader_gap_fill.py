"""Regression coverage for q23.strategy.data_loader._fetch_marketstack_gap_fill.

This locks in a real bug caught during the Phase 0 provider extraction: the
original inline code guarded the entire Marketstack block with
`if tickers:` and did nothing (no warning, no call) when no asset in the
universe had a ticker mapping. The first cut of the extraction dropped that
guard, so an unmapped universe (e.g. id-translation.csv not covering a given
index) fell through to MarketstackClient with an empty symbol list, which
raises "Symbols list cannot be empty" — turning a silent no-op into a
spurious UserWarning on every run. Caught by diffing a live end-to-end
strategy run against the pre-refactor tree; this test makes sure it can't
come back silently.

q23.strategy.data_loader.MarketstackProvider and get_missing_date_range are
patched at the data_loader module's own namespace (that's what the function
under test actually calls), not at their defining module.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import q23.strategy.data_loader as data_loader
from q23.strategy.data_loader import _fetch_marketstack_gap_fill


def _quantiacs_ds(n_time=5, assets=("NAS:AAPL", "NAS:MSFT")):
    time = pd.date_range("2026-06-01", periods=n_time)
    close = np.arange(n_time * len(assets)).reshape(n_time, len(assets)).astype(float)
    return xr.Dataset({"close": (("time", "asset"), close)}, coords={"time": time, "asset": list(assets)})


def _empty_marketstack_ds(assets):
    return xr.Dataset(
        {"close": (("time", "asset"), np.empty((0, len(assets))))},
        coords={"time": pd.DatetimeIndex([]), "asset": list(assets)},
    )


def _one_day_marketstack_ds(assets, date="2026-06-06"):
    return xr.Dataset(
        {"close": (("time", "asset"), [[123.0] * len(assets)])},
        coords={"time": pd.to_datetime([date]), "asset": list(assets)},
    )


@pytest.fixture
def fake_provider_factory(monkeypatch):
    """Install a fake MarketstackProvider class in data_loader's namespace
    and return the single instance it will hand back, so tests can set
    return values / assert calls on it directly."""
    instance = MagicMock()
    monkeypatch.setattr(data_loader, "MarketstackProvider", MagicMock(return_value=instance))
    return instance


class TestNoTickerMapping:
    def test_returns_dataset_unchanged_and_silent_when_no_tickers_mapped(
        self, fake_provider_factory, recwarn
    ):
        fake_provider_factory.tickers_for.return_value = []
        ds = _quantiacs_ds()

        result = _fetch_marketstack_gap_fill(
            ds,
            dataset_assets=["NAS:AAPL", "NAS:MSFT"],
            fill_recent_days=None,
            force_live_data=False,
            strategy_id="test",
        )

        assert result is ds
        fake_provider_factory.get_latest_prices.assert_not_called()
        fake_provider_factory.get_prices_with_raw.assert_not_called()
        assert len(recwarn) == 0, f"expected no warnings, got: {[str(w.message) for w in recwarn]}"


class TestSingleDayGap:
    def test_merges_single_day_via_get_latest_prices(self, fake_provider_factory, monkeypatch):
        assets = ["NAS:AAPL", "NAS:MSFT"]
        fake_provider_factory.tickers_for.return_value = ["AAPL", "MSFT"]
        monkeypatch.setattr(data_loader, "get_missing_date_range", lambda ds, **kw: ("2026-06-06", "2026-06-06"))
        fake_provider_factory.get_latest_prices.return_value = (
            _one_day_marketstack_ds(assets),
            pd.DataFrame({"symbol": ["AAPL", "MSFT"], "date": ["2026-06-06"] * 2}),
            "2026-06-06",
        )

        ds = _quantiacs_ds()
        result = _fetch_marketstack_gap_fill(
            ds, dataset_assets=assets, fill_recent_days=None, force_live_data=False, strategy_id="test"
        )

        fake_provider_factory.get_latest_prices.assert_called_once_with(
            assets=assets, target_date="2026-06-06"
        )
        assert result.sizes["time"] == ds.sizes["time"] + 1


class TestDateRangeGap:
    def test_merges_range_via_get_prices_with_raw(self, fake_provider_factory, monkeypatch):
        assets = ["NAS:AAPL", "NAS:MSFT"]
        fake_provider_factory.tickers_for.return_value = ["AAPL", "MSFT"]
        monkeypatch.setattr(data_loader, "get_missing_date_range", lambda ds, **kw: ("2026-06-06", "2026-06-08"))

        range_ds = xr.Dataset(
            {"close": (("time", "asset"), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
            coords={"time": pd.date_range("2026-06-06", periods=3), "asset": assets},
        )
        fake_provider_factory.get_prices_with_raw.return_value = (
            range_ds,
            pd.DataFrame({"symbol": ["AAPL"] * 3, "date": ["2026-06-06", "2026-06-07", "2026-06-08"]}),
        )

        ds = _quantiacs_ds()
        result = _fetch_marketstack_gap_fill(
            ds, dataset_assets=assets, fill_recent_days=None, force_live_data=False, strategy_id="test"
        )

        fake_provider_factory.get_prices_with_raw.assert_called_once_with(
            min_date="2026-06-06", max_date="2026-06-08", assets=assets
        )
        assert result.sizes["time"] == ds.sizes["time"] + 3


class TestNoGapButForceLiveData:
    def test_fetches_latest_unconditionally_when_no_gap_detected(self, fake_provider_factory, monkeypatch):
        """Matches the original's (slightly misleadingly commented) behavior:
        this branch runs whenever no gap is detected, not only when
        force_live_data=True."""
        assets = ["NAS:AAPL", "NAS:MSFT"]
        fake_provider_factory.tickers_for.return_value = ["AAPL", "MSFT"]
        monkeypatch.setattr(data_loader, "get_missing_date_range", lambda ds, **kw: None)
        fake_provider_factory.get_latest_prices.return_value = (
            _one_day_marketstack_ds(assets, date="2026-06-06"),
            pd.DataFrame({"symbol": ["AAPL", "MSFT"]}),
            "2026-06-06",
        )

        ds = _quantiacs_ds()
        _fetch_marketstack_gap_fill(
            ds, dataset_assets=assets, fill_recent_days=None, force_live_data=False, strategy_id="test"
        )

        fake_provider_factory.get_latest_prices.assert_called_once_with(assets=assets)

    def test_empty_result_leaves_dataset_untouched(self, fake_provider_factory, monkeypatch):
        assets = ["NAS:AAPL", "NAS:MSFT"]
        fake_provider_factory.tickers_for.return_value = ["AAPL", "MSFT"]
        monkeypatch.setattr(data_loader, "get_missing_date_range", lambda ds, **kw: None)
        fake_provider_factory.get_latest_prices.return_value = (
            _empty_marketstack_ds(assets), pd.DataFrame(), None,
        )

        ds = _quantiacs_ds()
        result = _fetch_marketstack_gap_fill(
            ds, dataset_assets=assets, fill_recent_days=None, force_live_data=False, strategy_id="test"
        )
        assert result.sizes["time"] == ds.sizes["time"]


class TestFailureIsNonFatal:
    def test_exception_from_provider_is_caught_and_warns(self, fake_provider_factory, monkeypatch):
        assets = ["NAS:AAPL", "NAS:MSFT"]
        fake_provider_factory.tickers_for.return_value = ["AAPL", "MSFT"]
        monkeypatch.setattr(data_loader, "get_missing_date_range", lambda ds, **kw: ("2026-06-06", "2026-06-06"))
        fake_provider_factory.get_latest_prices.side_effect = RuntimeError("Symbols list cannot be empty")

        ds = _quantiacs_ds()
        with pytest.warns(UserWarning, match="Failed to fetch Marketstack data"):
            result = _fetch_marketstack_gap_fill(
                ds, dataset_assets=assets, fill_recent_days=None, force_live_data=False, strategy_id="test"
            )
        assert result is ds
