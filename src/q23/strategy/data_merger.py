"""q23.strategy.data_merger

Merge Marketstack data with Quantiacs data.

Converts Marketstack DataFrame to xarray Dataset format and merges it
with existing Quantiacs datasets, handling duplicates and missing data.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

try:
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
    import xarray as xr  # type: ignore
except ImportError:
    pd = None
    np = None
    xr = None

from q23.strategy.symbol_mapper import asset_id_to_ticker


def _require_xr() -> None:
    """Check that xarray is available."""
    if xr is None:
        raise ImportError("xarray is required for data_merger")


def _require_pd() -> None:
    """Check that pandas is available."""
    if pd is None:
        raise ImportError("pandas is required for data_merger")


def marketstack_to_xarray(
    marketstack_df: pd.DataFrame,
    asset_ids: List[str],
) -> "xr.Dataset":
    """Convert Marketstack DataFrame to xarray Dataset format.
    
    Args:
        marketstack_df: DataFrame with columns: symbol, date, open, high, low, close, volume
        asset_ids: List of Quantiacs asset IDs to include (in order)
    
    Returns:
        xarray Dataset with dimensions (time, asset) and variables (open, high, low, close, vol)
    
    Note:
        Assets without Marketstack data will have NaN values.
        Only assets in asset_ids will be included in the output.
    """
    _require_xr()
    _require_pd()
    
    if marketstack_df.empty:
        # Return empty dataset with correct structure
        return _create_empty_dataset(asset_ids)
    
    # Create mapping from ticker to asset_id
    ticker_to_asset = {}
    for asset_id in asset_ids:
        ticker = asset_id_to_ticker(asset_id)
        if ticker:
            ticker_to_asset[ticker] = asset_id
    
    # Filter to only symbols we have mappings for
    available_tickers = [t for t in marketstack_df['symbol'].unique() if t in ticker_to_asset]
    
    if not available_tickers:
        warnings.warn("No Marketstack symbols mapped to asset IDs, returning empty dataset")
        return _create_empty_dataset(asset_ids)
    
    # Filter DataFrame to available tickers
    filtered_df = marketstack_df[marketstack_df['symbol'].isin(available_tickers)].copy()
    
    # Map tickers to asset IDs
    filtered_df['asset_id'] = filtered_df['symbol'].map(ticker_to_asset)
    
    # Pivot to wide format: date x asset_id
    # Create separate DataFrames for each variable
    data_vars = {}
    
    for var_name, marketstack_col in [
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('vol', 'volume'),
    ]:
        pivot_df = filtered_df.pivot_table(
            index='date',
            columns='asset_id',
            values=marketstack_col,
            aggfunc='first',  # In case of duplicates
        )
        
        # Ensure all asset_ids are present (fill with NaN)
        for asset_id in asset_ids:
            if asset_id not in pivot_df.columns:
                pivot_df[asset_id] = np.nan
        
        # Reorder columns to match asset_ids order
        pivot_df = pivot_df[[aid for aid in asset_ids if aid in pivot_df.columns]]
        
        # Convert index to datetime if needed
        if not isinstance(pivot_df.index, pd.DatetimeIndex):
            pivot_df.index = pd.to_datetime(pivot_df.index)
        
        # Convert to DataArray
        data_vars[var_name] = xr.DataArray(
            pivot_df.values,
            dims=['time', 'asset'],
            coords={
                'time': pivot_df.index,
                'asset': asset_ids,
            },
        )
    
    # Create Dataset
    ds = xr.Dataset(data_vars)
    
    # Ensure canonical dimension order
    ds = ds.transpose('time', 'asset')
    
    # Compute is_liquid (volume > 0 and finite)
    if 'vol' in ds.variables:
        ds['is_liquid'] = xr.where(
            np.isfinite(ds['vol']) & (ds['vol'] > 0),
            1.0,
            0.0
        )
    
    return ds


def _create_empty_dataset(asset_ids: List[str]) -> "xr.Dataset":
    """Create an empty xarray Dataset with correct structure.
    
    Args:
        asset_ids: List of asset IDs
    
    Returns:
        Empty Dataset with dimensions (time, asset) and required variables
    """
    _require_xr()
    
    # Create empty time dimension
    time_coords = pd.DatetimeIndex([])
    
    # Create empty DataArrays for each variable
    data_vars = {}
    for var_name in ['open', 'high', 'low', 'close', 'vol']:
        data_vars[var_name] = xr.DataArray(
            np.empty((0, len(asset_ids))),
            dims=['time', 'asset'],
            coords={
                'time': time_coords,
                'asset': asset_ids,
            },
        )
    
    ds = xr.Dataset(data_vars)
    ds['is_liquid'] = xr.zeros_like(ds['vol'])
    
    return ds


def merge_datasets(
    quantiacs_ds: "xr.Dataset",
    marketstack_ds: "xr.Dataset",
) -> "xr.Dataset":
    """Merge Quantiacs and Marketstack datasets.
    
    Args:
        quantiacs_ds: Quantiacs xarray Dataset
        marketstack_ds: Marketstack xarray Dataset (from marketstack_to_xarray)
    
    Returns:
        Merged Dataset with data from both sources
    
    Strategy:
        - Concatenate along time dimension
        - Remove duplicate dates (prefer Marketstack for overlapping dates)
        - Sort by time
        - Ensure consistent asset ordering
    """
    _require_xr()
    
    if marketstack_ds is None or marketstack_ds.sizes.get('time', 0) == 0:
        return quantiacs_ds
    
    if quantiacs_ds is None or quantiacs_ds.sizes.get('time', 0) == 0:
        return marketstack_ds
    
    # Ensure both datasets have the same asset dimension
    quantiacs_assets = list(quantiacs_ds.asset.values)
    marketstack_assets = list(marketstack_ds.asset.values)
    
    # Find common assets
    common_assets = [a for a in quantiacs_assets if a in marketstack_assets]
    
    if not common_assets:
        warnings.warn("No common assets between Quantiacs and Marketstack datasets")
        return quantiacs_ds
    
    # Select common assets from both datasets
    quantiacs_subset = quantiacs_ds.sel(asset=common_assets)
    marketstack_subset = marketstack_ds.sel(asset=common_assets)
    
    # Concatenate along time dimension
    try:
        merged = xr.concat([quantiacs_subset, marketstack_subset], dim='time')
    except Exception as e:
        warnings.warn(f"Failed to concatenate datasets: {e}, returning Quantiacs data only")
        return quantiacs_ds
    
    # Sort by time
    merged = merged.sortby('time')
    
    # Remove duplicate dates (keep last occurrence, which should be Marketstack)
    # Convert to DataFrame for easier duplicate handling
    try:
        merged_df = merged.to_dataframe()
        
        # Reset index to get time as column
        merged_df = merged_df.reset_index()
        
        # Convert time to date for grouping
        merged_df['date'] = pd.to_datetime(merged_df['time']).dt.date
        
        # Group by date and take last (prefer Marketstack data which comes later)
        merged_df = merged_df.groupby('date').last().reset_index(drop=True)
        
        # Set time and asset as index
        merged_df = merged_df.set_index(['time', 'asset'])
        
        # Reconstruct Dataset from DataFrame
        data_vars = {}
        for var_name in ['open', 'high', 'low', 'close', 'vol', 'is_liquid']:
            if var_name in merged_df.columns:
                pivot = merged_df[var_name].unstack('asset')
                data_vars[var_name] = xr.DataArray(
                    pivot.values,
                    dims=['time', 'asset'],
                    coords={
                        'time': pd.to_datetime(pivot.index),
                        'asset': common_assets,
                    },
                )
        
        merged = xr.Dataset(data_vars)
        
    except Exception as e:
        # If DataFrame conversion fails, try simpler approach
        warnings.warn(f"Failed to remove duplicates via DataFrame: {e}, keeping all data")
        # Just return merged without duplicate removal
        pass
    
    # Ensure canonical dimension order
    merged = merged.transpose('time', 'asset')
    
    return merged


def validate_merged_data(merged_ds: "xr.Dataset") -> bool:
    """Validate that merged dataset has correct structure.
    
    Args:
        merged_ds: Merged Dataset to validate
    
    Returns:
        True if valid, False otherwise
    """
    _require_xr()
    
    if merged_ds is None:
        return False
    
    # Check dimensions
    if 'time' not in merged_ds.dims or 'asset' not in merged_ds.dims:
        return False
    
    # Check required variables
    required_vars = ['open', 'high', 'low', 'close', 'vol']
    missing_vars = [v for v in required_vars if v not in merged_ds.variables]
    if missing_vars:
        return False
    
    # Check that time dimension is sorted
    if merged_ds.sizes.get('time', 0) > 1:
        time_values = pd.to_datetime(merged_ds.time.values)
        if not time_values.is_monotonic_increasing:
            return False
    
    return True
