"""Unit tests for the orchestration helpers that remain in
q23.strategy.data_loader after the Phase 0 provider extraction:
cache filtering/validation, pin index resolution, and returns computation.
These are pure functions independent of any vendor and were not touched by
the refactor, but had no regression coverage before this migration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from q23.strategy.data_loader import (
    _filter_cached_dataset,
    _is_valid_dataset,
    compute_returns,
    pinned_indices,
)


def _ds(n_time=5, assets=("NAS:AAPL", "NAS:MSFT"), with_close=True):
    time = pd.date_range("2026-06-01", periods=n_time)
    data_vars = {}
    if with_close:
        data_vars["close"] = (("time", "asset"), np.arange(n_time * len(assets)).reshape(n_time, len(assets)).astype(float))
    return xr.Dataset(data_vars, coords={"time": time, "asset": list(assets)})


class TestIsValidDataset:
    def test_valid_dataset_passes(self):
        assert _is_valid_dataset(_ds()) is True

    def test_missing_close_fails(self):
        assert _is_valid_dataset(_ds(with_close=False)) is False

    def test_empty_time_dim_fails(self):
        assert _is_valid_dataset(_ds(n_time=0)) is False

    def test_missing_dims_fails(self):
        ds = xr.Dataset({"close": ("x", [1.0, 2.0])}, coords={"x": [0, 1]})
        assert _is_valid_dataset(ds) is False


class TestFilterCachedDataset:
    def test_filters_by_date_range(self):
        ds = _ds(n_time=10)
        filtered = _filter_cached_dataset(ds, min_date="2026-06-05", max_date="2026-06-07", assets=None)
        assert filtered is not None
        assert filtered.sizes["time"] == 3

    def test_filters_by_assets(self):
        ds = _ds()
        filtered = _filter_cached_dataset(ds, min_date=None, max_date=None, assets=["NAS:AAPL"])
        assert filtered is not None
        assert list(filtered.asset.values) == ["NAS:AAPL"]

    def test_returns_none_for_invalid_input(self):
        ds = _ds(with_close=False)
        assert _filter_cached_dataset(ds, min_date=None, max_date=None, assets=None) is None

    def test_returns_none_when_filter_empties_result(self):
        ds = _ds()
        # Date range entirely outside the dataset's time coverage.
        filtered = _filter_cached_dataset(ds, min_date="2099-01-01", max_date="2099-01-02", assets=None)
        assert filtered is None


class TestPinnedIndices:
    def test_returns_indices_for_present_pins(self):
        idx = pinned_indices(["NAS:AAPL", "NAS:MSFT", "NAS:GOOGL"], pinned=["NAS:GOOGL", "NAS:AAPL"])
        assert sorted(idx.tolist()) == [0, 2]

    def test_ignores_absent_pins(self):
        idx = pinned_indices(["NAS:AAPL"], pinned=["NAS:ZZZZ"])
        assert idx.tolist() == []

    def test_no_pins_returns_empty(self):
        idx = pinned_indices(["NAS:AAPL"], pinned=None)
        assert idx.tolist() == []


class TestComputeReturns:
    def test_first_day_is_zero(self):
        close = xr.DataArray(
            [[100.0], [110.0], [99.0]],
            dims=["time", "asset"],
            coords={"time": pd.date_range("2026-06-01", periods=3), "asset": ["NAS:AAPL"]},
        )
        rets = compute_returns(close)
        assert rets.isel(time=0).item() == 0.0
        assert rets.isel(time=1).item() == pytest.approx(0.10)
        assert rets.isel(time=2).item() == pytest.approx(99.0 / 110.0 - 1.0)
