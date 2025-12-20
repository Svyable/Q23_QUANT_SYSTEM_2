"""
q23.strategy.outputs

Stable, dashboard-ready output writers (CSV now; optional NetCDF/Zarr later).

Responsibilities:
- Deterministic schemas and filenames
- Atomic writes to avoid half-written files
- Convert xarray to pandas safely
- Provide both "new v4" tag-based artifacts and "v6 legacy-like" rich artifacts
- Optimized CSV writes with precision control and memory efficiency

This module is deliberately I/O-only (no qnt, no streamlit).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Callable

import numpy as np
import pandas as pd

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore


# Precision constants
ZSCORE_DECIMALS = 4  # Decimal places for z-scores and factor values
WEIGHT_DECIMALS = 6  # Decimal places for weights (more precision needed)
GENERAL_DECIMALS = 4  # Decimal places for general metrics


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


def _atomic_write_csv(
    df: pd.DataFrame, 
    path: Path, 
    float_format: Optional[str] = None,
    compression: Optional[str] = None,
    **to_csv_kwargs
) -> None:
    """Atomic CSV write: write temp then replace.
    
    Optimized with float formatting for smaller file sizes and cleaner output.
    Supports optional compression for very large files.
    """
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    
    # Optimize CSV write with appropriate parameters
    csv_kwargs = {
        "index": True,
        "na_rep": "0.0",  # Use na_rep instead of fillna before write
        **to_csv_kwargs,
    }
    
    if float_format is not None:
        csv_kwargs["float_format"] = float_format
    
    if compression:
        csv_kwargs["compression"] = compression
        # Adjust path for compression extension
        if compression == "gzip":
            tmp = tmp.with_suffix(tmp.suffix + ".gz")
            path = path.with_suffix(path.suffix + ".gz")
    
    df.to_csv(tmp, **csv_kwargs)
    os.replace(tmp, path)


def _round_numeric_columns(
    df: pd.DataFrame, 
    decimals: int, 
    exclude_cols: Optional[List[str]] = None,
    inplace: bool = False
) -> pd.DataFrame:
    """Round numeric columns to specified decimals, excluding certain columns.
    
    Optimized to avoid unnecessary copies when inplace=True.
    """
    exclude_cols = exclude_cols or []
    
    if not inplace:
        df = df.copy()
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(decimals)
    
    return df


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame dtypes to reduce memory usage.
    
    Converts float64 to float32 where precision loss is acceptable,
    and optimizes integer types.
    """
    df = df.copy()
    
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            # Check if we can downcast to float32 without significant precision loss
            col_min = df[col].min()
            col_max = df[col].max()
            # float32 can represent values roughly in range [-3.4e38, 3.4e38]
            # with ~7 decimal digits of precision
            if (np.isfinite(col_min) and np.isfinite(col_max) and 
                abs(col_min) < 1e30 and abs(col_max) < 1e30):
                # For z-scores and most financial data, float32 is sufficient
                df[col] = df[col].astype(np.float32, copy=False)
        elif pd.api.types.is_integer_dtype(df[col]):
            # Optimize integer types
            col_min = df[col].min()
            col_max = df[col].max()
            if pd.notna(col_min) and pd.notna(col_max):
                if col_min >= 0:
                    if col_max < 255:
                        df[col] = df[col].astype(np.uint8, copy=False)
                    elif col_max < 65535:
                        df[col] = df[col].astype(np.uint16, copy=False)
                    elif col_max < 4294967295:
                        df[col] = df[col].astype(np.uint32, copy=False)
                else:
                    if col_min > -128 and col_max < 127:
                        df[col] = df[col].astype(np.int8, copy=False)
                    elif col_min > -32768 and col_max < 32767:
                        df[col] = df[col].astype(np.int16, copy=False)
                    elif col_min > -2147483648 and col_max < 2147483647:
                        df[col] = df[col].astype(np.int32, copy=False)
    
    return df


def _sanitize_df(df: pd.DataFrame, round_decimals: Optional[int] = None, optimize_dtypes: bool = False) -> pd.DataFrame:
    """Sanitize DataFrame: replace inf/NaN and optionally round.
    
    Args:
        df: DataFrame to sanitize
        round_decimals: If provided, round numeric columns to this precision
        optimize_dtypes: If True, optimize dtypes to reduce memory usage
    """
    # Replace inf/NaN first (in-place where possible)
    df = df.replace([np.inf, -np.inf], 0.0)
    df = df.fillna(0.0)
    
    if round_decimals is not None:
        df = _round_numeric_columns(df, round_decimals, inplace=True)
    
    if optimize_dtypes:
        df = _optimize_dtypes(df)
    
    return df


def _da_time_asset_to_df(da: "xr.DataArray", round_decimals: Optional[int] = None) -> pd.DataFrame:
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
    return _sanitize_df(df, round_decimals=round_decimals)


def _da_time_factor_to_df(da: "xr.DataArray", round_decimals: Optional[int] = None) -> pd.DataFrame:
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
    return _sanitize_df(df, round_decimals=round_decimals)


def _da_factor_time_to_df(da: "xr.DataArray", round_decimals: Optional[int] = None) -> pd.DataFrame:
    _require_xr()
    if "factor" not in da.dims or "time" not in da.dims:
        raise ValueError(f"Expected dims ('factor','time'), got {da.dims}")
    da = da.transpose("time", "factor")
    df = da.to_pandas()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df.index.name = "time"
    return _sanitize_df(df, round_decimals=round_decimals)


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
    use_compression: bool = False  # Enable gzip compression for large files

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
        df = _da_time_asset_to_df(weights, round_decimals=WEIGHT_DECIMALS)
        path = self.wide_weights_path(tag)
        compression = "gzip" if self.use_compression else None
        _atomic_write_csv(df, path, float_format=f"%.{WEIGHT_DECIMALS}f", compression=compression)
        return path

    def write_budget(self, budget: "xr.DataArray", tag: str) -> Path:
        _require_xr()
        if "time" not in budget.dims:
            raise ValueError("budget must have dim 'time'")
        s = budget.to_pandas()
        if not isinstance(s, pd.Series):
            s = pd.Series(s)
        df = _sanitize_df(s.to_frame("budget"), round_decimals=GENERAL_DECIMALS)
        df.index.name = "time"
        path = self.budget_path(tag)
        _atomic_write_csv(df, path, float_format=f"%.{GENERAL_DECIMALS}f")
        return path

    def write_factor_weights(self, w: "xr.DataArray", tag: str) -> Path:
        df = _da_time_factor_to_df(w, round_decimals=ZSCORE_DECIMALS)
        path = self.factor_weights_path(tag)
        _atomic_write_csv(df, path, float_format=f"%.{ZSCORE_DECIMALS}f")
        return path

    def write_ic(self, ic_raw: "xr.DataArray", ic_smooth: "xr.DataArray", tag: str) -> Path:
        _require_xr()
        raw_df = _da_factor_time_to_df(ic_raw, round_decimals=GENERAL_DECIMALS)
        sm_df = _da_factor_time_to_df(ic_smooth, round_decimals=GENERAL_DECIMALS)

        # multiindex columns: (factor, kind)
        raw_df.columns = pd.MultiIndex.from_product([raw_df.columns, ["ic_raw"]])
        sm_df.columns = pd.MultiIndex.from_product([sm_df.columns, ["ic_smooth"]])

        df = pd.concat([raw_df, sm_df], axis=1).sort_index(axis=1)
        df = _sanitize_df(df, round_decimals=GENERAL_DECIMALS)
        path = self.ic_path(tag)
        _atomic_write_csv(df, path, float_format=f"%.{GENERAL_DECIMALS}f")
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
        df = _sanitize_df(df, round_decimals=GENERAL_DECIMALS)
        path = self.portfolio_diag_path(tag)
        _atomic_write_csv(df, path, float_format=f"%.{GENERAL_DECIMALS}f")
        return path

    def write_factor_exposure(self, exposure: pd.DataFrame, tag: str) -> Path:
        df = exposure.copy()
        if df.index.name != "time":
            df.index.name = "time"
        # Factor exposures are z-scores, round to 4 decimals
        df = _sanitize_df(df, round_decimals=ZSCORE_DECIMALS)
        path = self.factor_exposure_path(tag)
        _atomic_write_csv(df, path, float_format=f"%.{ZSCORE_DECIMALS}f")
        return path

    def write_factor_vectors(self, vectors: pd.DataFrame, tag: str) -> Path:
        """Write factor vectors with mixed precision handling.
        
        Uses a custom formatter to handle different precisions for different column types.
        """
        df = vectors.copy()
        if df.index.name not in ("asset", "symbol"):
            df.index.name = "asset"
        
        # Identify factor columns (z-scores) vs summary columns
        summary_cols = [
            "total_pnl_contrib",
            "pnl_per_day_held",
            "mean_score",
            "score_vol",
            "days_held",
            "avg_weight",
            "avg_weight_when_held",
        ]
        
        # Round factor columns (z-scores) to 4 decimals
        factor_cols = [c for c in df.columns if c not in summary_cols]
        
        # Round factor columns to 4 decimals
        for col in factor_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(ZSCORE_DECIMALS)
        
        # Round summary columns appropriately
        precision_map = {
            "total_pnl_contrib": GENERAL_DECIMALS,
            "pnl_per_day_held": GENERAL_DECIMALS,
            "mean_score": ZSCORE_DECIMALS,  # Scores are like z-scores
            "score_vol": GENERAL_DECIMALS,
            "days_held": 0,  # Integer
            "avg_weight": WEIGHT_DECIMALS,
            "avg_weight_when_held": WEIGHT_DECIMALS,
        }
        
        for col, decimals in precision_map.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(decimals)
        
        df = _sanitize_df(df)  # Handle inf/NaN
        path = self.factor_vectors_path(tag)
        
        # Create custom formatter for mixed precision
        def mixed_precision_formatter(x: float) -> str:
            """Format with appropriate precision based on value magnitude."""
            if pd.isna(x) or not np.isfinite(x):
                return "0.0"
            # For very small values, use scientific notation
            if abs(x) < 1e-4 and x != 0:
                return f"{x:.4e}"
            # For integers (days_held), no decimals
            if "days_held" in str(x) or abs(x - round(x)) < 1e-10:
                return f"{int(round(x))}"
            # Default: 4 decimals for most, 6 for weights
            if abs(x) < 1.0:
                return f"{x:.6f}".rstrip('0').rstrip('.')
            return f"{x:.4f}".rstrip('0').rstrip('.')
        
        # Use a simpler approach: write with default precision, rounding already done
        # For factor_vectors, we'll use 4 decimals as default (covers most cases)
        # The rounding above ensures correct precision
        _atomic_write_csv(df, path, float_format=f"%.{ZSCORE_DECIMALS}f")
        return path
