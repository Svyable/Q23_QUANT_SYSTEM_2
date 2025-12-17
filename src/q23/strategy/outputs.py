"""
q23.strategy.outputs

Stable, dashboard-ready output writers (CSV now; optional NetCDF/Zarr later).

Responsibilities:
- Deterministic schemas and filenames
- Atomic writes to avoid half-written files
- Convert xarray to pandas safely
- Provide both "new v4" tag-based artifacts and "v6 legacy-like" rich artifacts

This module is deliberately I/O-only (no qnt, no streamlit).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore


def _require_xr() -> None:
    if xr is None:
        raise ImportError("xarray is required for q23.strategy.outputs")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _atomic_write_text(text: str, path: Path) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_write_csv(df: pd.DataFrame, path: Path, **to_csv_kwargs) -> None:
    """Atomic CSV write: write temp then replace."""
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, **to_csv_kwargs)
    os.replace(tmp, path)


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _da_time_asset_to_df(da: "xr.DataArray") -> pd.DataFrame:
    _require_xr()
    if "time" not in da.dims:
        raise ValueError(f"Expected 'time' in dims, got {da.dims}")
    if "asset" not in da.dims:
        raise ValueError(f"Expected 'asset' in dims, got {da.dims}")
    da = da.transpose("time", "asset")
    df = da.to_pandas()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df.index.name = "time"
    return _sanitize_df(df)


def _da_time_factor_to_df(da: "xr.DataArray") -> pd.DataFrame:
    _require_xr()
    if "time" not in da.dims:
        raise ValueError(f"Expected 'time' in dims, got {da.dims}")
    if "factor" not in da.dims:
        raise ValueError(f"Expected 'factor' in dims, got {da.dims}")
    da = da.transpose("time", "factor")
    df = da.to_pandas()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df.index.name = "time"
    return _sanitize_df(df)


def _da_factor_time_to_df(da: "xr.DataArray") -> pd.DataFrame:
    _require_xr()
    if "factor" not in da.dims or "time" not in da.dims:
        raise ValueError(f"Expected dims ('factor','time'), got {da.dims}")
    da = da.transpose("time", "factor")
    df = da.to_pandas()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df.index.name = "time"
    return _sanitize_df(df)


@dataclass
class OutputWriter:
    """
    Output conventions (tag-based):

      {base}_wide_weights_{tag}.csv            time x asset
      {base}_budget_{tag}.csv                  time x 1
      {base}_factor_weights_{tag}.csv          time x factor
      {base}_ic_{tag}.csv                      time x (factor, ic_raw/ic_smooth) via MultiIndex cols
      {base}_meta_{tag}.json

    "Ultimate enhancements" artifacts (still tag-based):

      {base}_portfolio_diag_{tag}.csv          time x diagnostics
      {base}_factor_exposure_{tag}.csv         time x factor (portfolio exposure)
      {base}_factor_vectors_{tag}.csv          asset x (factor + summary columns)
    """

    base_name: str
    output_dir: str

    def __post_init__(self) -> None:
        self.root = Path(self.output_dir)
        _ensure_dir(self.root)

    # ---- filenames (v4 tag-based) ----

    def wide_weights_path(self, tag: str) -> Path:
        return self.root / f"{self.base_name}_wide_weights_{tag}.csv"

    def budget_path(self, tag: str) -> Path:
        return self.root / f"{self.base_name}_budget_{tag}.csv"

    def factor_weights_path(self, tag: str) -> Path:
        return self.root / f"{self.base_name}_factor_weights_{tag}.csv"

    def ic_path(self, tag: str) -> Path:
        return self.root / f"{self.base_name}_ic_{tag}.csv"

    def meta_path(self, tag: str) -> Path:
        return self.root / f"{self.base_name}_meta_{tag}.json"

    # ---- filenames (enhanced) ----

    def portfolio_diag_path(self, tag: str) -> Path:
        return self.root / f"{self.base_name}_portfolio_diag_{tag}.csv"

    def factor_exposure_path(self, tag: str) -> Path:
        return self.root / f"{self.base_name}_factor_exposure_{tag}.csv"

    def factor_vectors_path(self, tag: str) -> Path:
        return self.root / f"{self.base_name}_factor_vectors_{tag}.csv"

    # ---- writers (v4) ----

    def write_wide_weights(self, weights: "xr.DataArray", tag: str) -> Path:
        df = _da_time_asset_to_df(weights)
        path = self.wide_weights_path(tag)
        _atomic_write_csv(df, path, index=True)
        return path

    def write_budget(self, budget: "xr.DataArray", tag: str) -> Path:
        _require_xr()
        if "time" not in budget.dims:
            raise ValueError("budget must have dim 'time'")
        s = budget.to_pandas()
        if not isinstance(s, pd.Series):
            s = pd.Series(s)
        df = _sanitize_df(s.to_frame("budget"))
        df.index.name = "time"
        path = self.budget_path(tag)
        _atomic_write_csv(df, path, index=True)
        return path

    def write_factor_weights(self, w: "xr.DataArray", tag: str) -> Path:
        df = _da_time_factor_to_df(w)
        path = self.factor_weights_path(tag)
        _atomic_write_csv(df, path, index=True)
        return path

    def write_ic(self, ic_raw: "xr.DataArray", ic_smooth: "xr.DataArray", tag: str) -> Path:
        _require_xr()
        raw_df = _da_factor_time_to_df(ic_raw)
        sm_df = _da_factor_time_to_df(ic_smooth)

        # multiindex columns: (factor, kind)
        raw_df.columns = pd.MultiIndex.from_product([raw_df.columns, ["ic_raw"]])
        sm_df.columns = pd.MultiIndex.from_product([sm_df.columns, ["ic_smooth"]])

        df = pd.concat([raw_df, sm_df], axis=1).sort_index(axis=1)
        df = _sanitize_df(df)
        path = self.ic_path(tag)
        _atomic_write_csv(df, path, index=True)
        return path

    def write_meta(self, meta: Dict[str, Any], tag: str) -> Path:
        path = self.meta_path(tag)
        _atomic_write_text(json.dumps(meta, indent=2, default=str), path)
        return path

    # ---- writers (enhanced) ----

    def write_portfolio_diag(self, diag: pd.DataFrame, tag: str) -> Path:
        df = diag.copy()
        if df.index.name != "time":
            df.index.name = "time"
        df = _sanitize_df(df)
        path = self.portfolio_diag_path(tag)
        _atomic_write_csv(df, path, index=True)
        return path

    def write_factor_exposure(self, exposure: pd.DataFrame, tag: str) -> Path:
        df = exposure.copy()
        if df.index.name != "time":
            df.index.name = "time"
        df = _sanitize_df(df)
        path = self.factor_exposure_path(tag)
        _atomic_write_csv(df, path, index=True)
        return path

    def write_factor_vectors(self, vectors: pd.DataFrame, tag: str) -> Path:
        df = vectors.copy()
        if df.index.name not in ("asset", "symbol"):
            df.index.name = "asset"
        df = _sanitize_df(df)
        path = self.factor_vectors_path(tag)
        _atomic_write_csv(df, path, index=True)
        return path
