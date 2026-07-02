"""Unit tests for q23.data.providers.quantiacs.QuantiacsProvider.

These mock qnt.data entirely (no network / API key required) so they can
run in CI. The live end-to-end behavior of QuantiacsProvider was verified
separately via a before/after diff against the pre-refactor data_loader.py
using real Quantiacs API calls (see Phase 0 migration notes) — these tests
lock in that behavior going forward (loader/field fallback ordering, in
particular, has bitten this codebase before per the module's own docstring
history).
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from q23.data.providers.quantiacs import (
    QuantiacsProvider,
    _looks_like_missing_field_error,
    _normalize_to_dataset,
    _try_call,
)


def _make_dataset(fields=("open", "high", "low", "close", "vol", "is_liquid")):
    time = pd.date_range("2026-06-01", periods=3)
    assets = ["NAS:AAPL", "NAS:MSFT"]
    data = {
        f: (("time", "asset"), np.ones((3, 2)))
        for f in fields
    }
    return xr.Dataset(data, coords={"time": time, "asset": assets})


@pytest.fixture
def fake_qnt(monkeypatch):
    """Install a fake `qnt.data` module in sys.modules for the duration of a test."""
    fake_qndata = types.ModuleType("qnt.data")
    fake_stocks = types.ModuleType("qnt.data.stocks")
    fake_qndata.stocks = fake_stocks
    monkeypatch.setitem(sys.modules, "qnt.data", fake_qndata)
    monkeypatch.setitem(sys.modules, "qnt.data.stocks", fake_stocks)
    return fake_qndata


class TestTryCall:
    def test_calls_with_full_kwargs_when_accepted(self):
        fn = MagicMock(return_value="ok")
        result = _try_call(fn, min_date="2026-01-01", assets=["A"])
        assert result == "ok"
        fn.assert_called_once_with(min_date="2026-01-01", assets=["A"])

    def test_drops_unexpected_kwarg_on_type_error(self):
        def fn(min_date):
            return f"got {min_date}"

        result = _try_call(fn, min_date="2026-01-01", assets=["A"], fields=["close"])
        assert result == "got 2026-01-01"

    def test_falls_back_to_positional_min_date(self):
        def fn(min_date):
            return f"positional {min_date}"

        # No kwarg subset matches fn's signature except min_date alone,
        # which the kwarg-dropping loop already covers — this exercises
        # the final positional fallback for a fn accepting *only* a single
        # positional arg.
        result = _try_call(fn, min_date="2026-01-01")
        assert result == "positional 2026-01-01"


class TestLooksLikeMissingFieldError:
    def test_matches_known_qnt_message(self):
        e = KeyError("not all values found in index 'field'")
        assert _looks_like_missing_field_error(e) is True

    def test_does_not_match_unrelated_error(self):
        e = KeyError("asset 'NAS:ZZZZ' not found")
        assert _looks_like_missing_field_error(e) is False


class TestNormalizeToDataset:
    def test_dataset_passthrough(self):
        ds = _make_dataset()
        assert _normalize_to_dataset(ds) is ds

    def test_dataarray_with_field_dim_converts(self):
        time = pd.date_range("2026-06-01", periods=2)
        assets = ["NAS:AAPL"]
        fields = ["open", "close"]
        da = xr.DataArray(
            np.ones((2, 1, 2)),
            dims=["time", "asset", "field"],
            coords={"time": time, "asset": assets, "field": fields},
        )
        ds = _normalize_to_dataset(da)
        assert isinstance(ds, xr.Dataset)
        assert set(ds.data_vars) == {"open", "close"}

    def test_dataarray_without_field_dim_raises(self):
        da = xr.DataArray(np.ones((2, 1)), dims=["time", "asset"])
        with pytest.raises(ValueError):
            _normalize_to_dataset(da)

    def test_unexpected_type_raises(self):
        with pytest.raises(TypeError):
            _normalize_to_dataset({"not": "xarray"})


class TestQuantiacsProviderGetPrices:
    def test_uses_load_spx_data_when_available(self, fake_qnt):
        ds = _make_dataset()
        fake_qnt.stocks.load_spx_data = MagicMock(return_value=ds)

        provider = QuantiacsProvider()
        result = provider.get_prices(min_date="2026-06-01", assets=["NAS:AAPL", "NAS:MSFT"])

        assert "close" in result.variables
        assert list(result.dims) == ["time", "asset"] or set(result.dims) == {"time", "asset"}
        fake_qnt.stocks.load_spx_data.assert_called()

    def test_retries_with_reduced_fields_on_missing_field_keyerror(self, fake_qnt):
        """Reproduces the exact failure documented in the original module
        docstring: some qnt versions reject 'is_liquid' with a KeyError on
        the 'field' index. The provider should silently retry with a
        smaller field set rather than propagating the error."""
        good_ds = _make_dataset(fields=("open", "high", "low", "close", "vol"))

        call_log = []

        def flaky_loader(**kwargs):
            call_log.append(kwargs.get("fields"))
            if kwargs.get("fields") and "is_liquid" in kwargs["fields"]:
                raise KeyError("not all values found in index 'field'")
            return good_ds

        fake_qnt.stocks.load_spx_data = flaky_loader

        provider = QuantiacsProvider()
        result = provider.get_prices(min_date="2026-06-01", assets=["NAS:AAPL", "NAS:MSFT"])

        assert "close" in result.variables
        assert "is_liquid" not in result.variables
        # First attempt included is_liquid and failed; second attempt dropped it.
        assert call_log[0] is not None and "is_liquid" in call_log[0]
        assert "is_liquid" not in call_log[1]

    def test_raises_when_no_loader_available(self, fake_qnt):
        provider = QuantiacsProvider()
        with pytest.raises(AttributeError):
            provider.get_prices(min_date="2026-06-01")

    def test_raises_runtime_error_when_all_loaders_fail(self, fake_qnt):
        fake_qnt.stocks.load_spx_data = MagicMock(side_effect=RuntimeError("boom"))
        provider = QuantiacsProvider()
        with pytest.raises(RuntimeError, match="Failed to load Quantiacs stocks data"):
            provider.get_prices(min_date="2026-06-01")


class TestQuantiacsProviderGetUniverse:
    def test_returns_none_when_qnt_unavailable(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "qnt.data", raising=False)
        monkeypatch.delitem(sys.modules, "qnt", raising=False)
        provider = QuantiacsProvider()
        # Simulate qnt truly missing by making the import fail.
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "qnt.data" or name.startswith("qnt."):
                raise ImportError("no qnt here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert provider.get_universe(min_date="2026-06-01") is None

    def test_filters_by_exchange_and_includes_pinned(self, fake_qnt):
        fake_qnt.stocks.load_spx_list = MagicMock(
            return_value=[
                {"id": "NAS:AAPL", "exchange": "NAS"},
                {"id": "NYS:JPM", "exchange": "NYS"},
                {"id": "NAS:MSFT", "exchange": "NAS"},
            ]
        )
        provider = QuantiacsProvider()
        ids = provider.get_universe(min_date="2026-06-01", exchanges=["NAS"], pinned=["NYS:JPM"])
        assert ids is not None
        assert set(ids) == {"NAS:AAPL", "NAS:MSFT", "NYS:JPM"}

    def test_memoizes_by_min_date(self, fake_qnt):
        fake_qnt.stocks.load_spx_list = MagicMock(
            return_value=[{"id": "NAS:AAPL", "exchange": "NAS"}]
        )
        provider = QuantiacsProvider()
        provider.get_universe(min_date="2026-06-01")
        provider.get_universe(min_date="2026-06-01")
        assert fake_qnt.stocks.load_spx_list.call_count == 1
