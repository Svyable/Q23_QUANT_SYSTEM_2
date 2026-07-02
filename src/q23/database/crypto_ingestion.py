"""q23.database.crypto_ingestion

Cryptocurrency data ingestion pipeline from Quantiacs to PostgreSQL database.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import warnings

try:
    import pandas as pd
    import numpy as np
    import xarray as xr
    import qnt.data as qndata
except ImportError:
    pd = None
    np = None
    xr = None
    qndata = None

from q23.database.client import DatabaseClient
from q23.database.crypto_schema import crypto_schema_exists

logger = logging.getLogger(__name__)


def _require_dependencies() -> None:
    """Check that required dependencies are available."""
    if pd is None:
        raise ImportError("pandas is required for data ingestion")
    if np is None:
        raise ImportError("numpy is required for data ingestion")
    if xr is None:
        raise ImportError("xarray is required for data ingestion")
    if qndata is None:
        raise ImportError("qnt.data is required for data ingestion. Install Quantiacs package.")


@dataclass
class CryptoIngestionResult:
    """Result of a crypto data ingestion run."""
    start_date: date
    end_date: date
    assets_count: int
    rows_inserted: int
    status: str  # 'success', 'partial', 'failed'
    error_message: Optional[str] = None
    duration_seconds: float = 0.0


def _chunk_dates(start: date, end: date, chunk_days: int = 365 * 2) -> List[Tuple[date, date]]:
    """Split date range into chunks.
    
    Args:
        start: Start date
        end: End date
        chunk_days: Number of calendar days per chunk (~2 years)
        
    Returns:
        List of (chunk_start, chunk_end) tuples
    """
    chunks = []
    current = start
    
    while current <= end:
        chunk_end = current + timedelta(days=chunk_days)
        if chunk_end > end:
            chunk_end = end
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    
    return chunks


def _fetch_crypto_data(
    assets: Optional[List[str]],
    start_date: date,
    end_date: date,
) -> xr.DataArray:
    """Fetch cryptocurrency OHLCV data from Quantiacs.
    
    Args:
        assets: List of asset symbols (e.g., ["BTC", "ETH"]) or None for all
        start_date: Start date
        end_date: End date
        
    Returns:
        xarray DataArray with OHLCV data
    """
    _require_dependencies()
    
    # #region agent log
    try:
        import json
        log_path = '/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log'
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"crypto_ingestion.py:_fetch_crypto_data:ENTRY","message":"Function entry","data":{"assets":assets,"start_date":str(start_date),"end_date":str(end_date)},"timestamp":int(time.time()*1000)})+'\n')
    except Exception as log_err:
        logger.debug(f"Debug log write failed: {log_err}")
    # #endregion
    
    # Expand single-day ranges to avoid NetCDF issues
    original_start = start_date
    original_end = end_date
    if start_date == end_date:
        from datetime import timedelta
        start_date = start_date - timedelta(days=1)
        end_date = end_date + timedelta(days=1)
        logger.info(f"Expanded single-day range to avoid NetCDF issues: {start_date} to {end_date}")
    
    # #region agent log
    try:
        import json
        log_path = '/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log'
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"crypto_ingestion.py:_fetch_crypto_data:AFTER_EXPAND","message":"After date expansion","data":{"original_start":str(original_start),"original_end":str(original_end),"expanded_start":str(start_date),"expanded_end":str(end_date)},"timestamp":int(time.time()*1000)})+'\n')
    except Exception as log_err:
        logger.debug(f"Debug log write failed: {log_err}")
    # #endregion
    
    max_retries = 3
    retry_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            # Calculate tail (calendar days from end_date to start_date)
            tail = (end_date - start_date).days + 365  # Add buffer
            
            # #region agent log
            try:
                import json
                log_path = '/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log'
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"crypto_ingestion.py:_fetch_crypto_data:BEFORE_API_CALL","message":"Before API call","data":{"attempt":attempt+1,"start_date":str(start_date),"end_date":str(end_date),"tail":tail,"assets":assets,"has_dims_param":False},"timestamp":int(time.time()*1000)})+'\n')
            except Exception as log_err:
                pass  # Don't break on logging errors
            # #endregion
            
            # Try without dims parameter first (dims parameter can cause NetCDF errors)
            # Based on Quantiacs docs, cryptodaily.load_data doesn't always need explicit dims
            try:
                # #region agent log
                try:
                    import json
                    log_path = '/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log'
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"crypto_ingestion.py:_fetch_crypto_data:BEFORE_API_NO_DIMS","message":"About to call API without dims","data":{"tail":tail},"timestamp":int(time.time()*1000)})+'\n')
                except: pass
                # #endregion
                
                data = qndata.cryptodaily.load_data(
                    assets=assets,
                    min_date=start_date.strftime("%Y-%m-%d"),
                    max_date=end_date.strftime("%Y-%m-%d"),
                    forward_order=True,
                    tail=tail,
                )
                
                # #region agent log
                try:
                    import json
                    log_path = '/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log'
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"crypto_ingestion.py:_fetch_crypto_data:API_SUCCESS_NO_DIMS","message":"API call succeeded without dims","data":{"dims":list(data.dims) if hasattr(data,'dims') else None,"coords":list(data.coords.keys()) if hasattr(data,'coords') else None,"sizes":dict(data.sizes) if hasattr(data,'sizes') else None},"timestamp":int(time.time()*1000)})+'\n')
                except: pass
                # #endregion
                
            except Exception as dims_error:
                error_str_dims = str(dims_error)
                
                # #region agent log
                try:
                    import json
                    log_path = '/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log'
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"crypto_ingestion.py:_fetch_crypto_data:API_ERROR_NO_DIMS","message":"API call failed without dims","data":{"error":error_str_dims,"error_type":type(dims_error).__name__,"is_netcdf":("NetCDF" in error_str_dims or "NC_UNLIMITED" in error_str_dims)},"timestamp":int(time.time()*1000)})+'\n')
                except: pass
                # #endregion
                
                # If NetCDF error, don't try with dims - it will likely fail the same way
                # Instead, try with a different tail calculation or skip dims entirely
                if "NetCDF" in error_str_dims or "NC_UNLIMITED" in error_str_dims:
                    logger.debug(f"NetCDF error without dims, will try expanded range instead: {dims_error}")
                    raise  # Re-raise to trigger expanded range logic
                
                # For non-NetCDF errors, try with explicit dims
                logger.debug(f"Trying with explicit dims parameter: {dims_error}")
                
                # #region agent log
                try:
                    import json
                    log_path = '/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log'
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"crypto_ingestion.py:_fetch_crypto_data:BEFORE_API_CALL_DIMS","message":"Before API call with dims","data":{"attempt":attempt+1,"has_dims_param":True,"dims":("field","time","asset")},"timestamp":int(time.time()*1000)})+'\n')
                except: pass
                # #endregion
                
                data = qndata.cryptodaily.load_data(
                    assets=assets,
                    min_date=start_date.strftime("%Y-%m-%d"),
                    max_date=end_date.strftime("%Y-%m-%d"),
                    dims=("field", "time", "asset"),
                    forward_order=True,
                    tail=tail,
                )
                
                # #region agent log
                try:
                    import json
                    log_path = '/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log'
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"crypto_ingestion.py:_fetch_crypto_data:API_SUCCESS_WITH_DIMS","message":"API call succeeded with dims","data":{"dims":list(data.dims) if hasattr(data,'dims') else None},"timestamp":int(time.time()*1000)})+'\n')
                except: pass
                # #endregion
            
            logger.info(f"Fetched crypto data: {len(data.coords['asset'].values) if 'asset' in data.coords else 'all'} assets, "
                       f"{data.sizes.get('time', 0)} time points")
            return data
        except Exception as e:
            error_str = str(e)
            
            # #region agent log
            try:
                import json
                log_path = '/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log'
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"crypto_ingestion.py:_fetch_crypto_data:EXCEPTION","message":"Exception caught","data":{"attempt":attempt+1,"error":error_str,"error_type":type(e).__name__,"is_netcdf":("NetCDF" in error_str or "NC_UNLIMITED" in error_str)},"timestamp":int(time.time()*1000)})+'\n')
            except: pass
            # #endregion
            
            # Check if it's a NetCDF error
            if "NetCDF" in error_str or "NC_UNLIMITED" in error_str:
                if attempt < max_retries - 1:
                    logger.warning(f"NetCDF error (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info("This may be a Quantiacs API issue. Trying with expanded date range...")
                    # Expand date range more aggressively
                    from datetime import timedelta
                    expanded_start = start_date - timedelta(days=5)
                    expanded_end = end_date + timedelta(days=5)
                    
                    # #region agent log
                    try:
                        import json
                        log_path = '/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log'
                        with open(log_path, 'a') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"crypto_ingestion.py:_fetch_crypto_data:BEFORE_EXPANDED_RANGE","message":"Before expanded range attempt","data":{"expanded_start":str(expanded_start),"expanded_end":str(expanded_end),"original_start":str(original_start),"original_end":str(original_end)},"timestamp":int(time.time()*1000)})+'\n')
                    except: pass
                    # #endregion
                    
                    try:
                        expanded_tail = (expanded_end - expanded_start).days + 365
                        
                        # #region agent log
                        try:
                            import json
                            log_path = '/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log'
                            with open(log_path, 'a') as f:
                                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"crypto_ingestion.py:_fetch_crypto_data:BEFORE_EXPANDED_API","message":"Before expanded range API call","data":{"expanded_tail":expanded_tail,"no_dims":True},"timestamp":int(time.time()*1000)})+'\n')
                        except: pass
                        # #endregion
                        
                        # Try without dims first for expanded range (dims parameter causes NetCDF errors)
                        data = qndata.cryptodaily.load_data(
                            assets=assets,
                            min_date=expanded_start.strftime("%Y-%m-%d"),
                            max_date=expanded_end.strftime("%Y-%m-%d"),
                            forward_order=True,
                            tail=expanded_tail,
                        )
                        
                        # #region agent log
                        try:
                            import json
                            log_path = '/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log'
                            with open(log_path, 'a') as f:
                                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"crypto_ingestion.py:_fetch_crypto_data:EXPANDED_SUCCESS","message":"Expanded range API succeeded","data":{"time_points":data.sizes.get('time',0) if hasattr(data,'sizes') else 0},"timestamp":int(time.time()*1000)})+'\n')
                        except: pass
                        # #endregion
                        
                        # Filter to original date range after fetching
                        if 'time' in data.coords and len(data.coords['time']) > 0:
                            time_coords = pd.to_datetime(data.coords['time'].values)
                            start_ts = pd.Timestamp(original_start)
                            end_ts = pd.Timestamp(original_end) + pd.Timedelta(days=1)
                            time_mask = (time_coords >= start_ts) & (time_coords < end_ts)
                            
                            # #region agent log
                            try:
                                import json
                                log_path = '/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log'
                                with open(log_path, 'a') as f:
                                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"crypto_ingestion.py:_fetch_crypto_data:AFTER_FILTER","message":"After time filtering","data":{"total_time_points":len(time_coords),"filtered_count":time_mask.sum() if hasattr(time_mask,'sum') else 0,"requested_start":str(original_start),"requested_end":str(original_end)},"timestamp":int(time.time()*1000)})+'\n')
                            except: pass
                            # #endregion
                            
                            if time_mask.sum() > 0:
                                time_indices = np.where(time_mask)[0]
                                data = data.isel(time=time_indices)
                                logger.info(f"Successfully fetched with expanded range, filtered to {len(time_indices)} time points in original date range")
                            else:
                                logger.warning(f"No data in exact range, using all fetched data")
                        else:
                            logger.info(f"Successfully fetched with expanded range (no time coordinate to filter)")
                        return data
                    except Exception as expanded_error:
                        # #region agent log
                        try:
                            import json
                            log_path = '/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log'
                            with open(log_path, 'a') as f:
                                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"crypto_ingestion.py:_fetch_crypto_data:EXPANDED_ERROR","message":"Expanded range also failed","data":{"error":str(expanded_error),"error_type":type(expanded_error).__name__},"timestamp":int(time.time()*1000)})+'\n')
                        except: pass
                        # #endregion
                        logger.warning(f"Expanded range also failed: {expanded_error}")
            
            if attempt < max_retries - 1:
                logger.warning(f"Quantiacs crypto API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"Failed to fetch crypto data after {max_retries} attempts: {e}")
                raise


def _dataarray_to_dataframe(da: xr.DataArray) -> pd.DataFrame:
    """Convert xarray DataArray to pandas DataFrame.
    
    Args:
        da: DataArray with dimensions (field, time, asset)
        
    Returns:
        DataFrame with columns: date, asset, and all fields
    """
    _require_dependencies()
    
    # Select available fields - aligned with Quantiacs cryptodaily fields
    # Quantiacs provides: open, high, low, close, is_liquid (volume may be available)
    wanted_fields = ["open", "high", "low", "close", "is_liquid"]
    # Check for volume (may or may not be available)
    if "vol" in da.coords["field"].values:
        wanted_fields.append("vol")
    elif "volume" in da.coords["field"].values:
        wanted_fields.append("volume")
    available_fields = [f for f in wanted_fields if f in da.coords["field"].values]
    
    if not available_fields:
        logger.warning("No expected fields found in dataset")
        return pd.DataFrame()
    
    # Convert to DataFrame - DataArray with (field, time, asset) dims
    # Results in multi-index DataFrame with field, time, asset as index
    df = da.to_dataframe().reset_index()
    
    # The DataFrame should have: field, time, asset, and a value column
    # Find the value column (it's the one that's not field, time, or asset)
    value_col = None
    for col in df.columns:
        if col not in ["field", "time", "asset"]:
            value_col = col
            break
    
    if value_col is None:
        # Try to find by data type (should be numeric)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                value_col = col
                break
    
    if value_col is None:
        logger.warning("Could not find value column in DataFrame")
        return pd.DataFrame()
    
    # Pivot to wide format: date | asset | open | high | low | close | is_liquid
    df_pivot = df.pivot_table(
        index=["time", "asset"],
        columns="field",
        values=value_col,
        aggfunc="first"
    ).reset_index()
    
    # Rename time to date
    df_pivot = df_pivot.rename(columns={"time": "date"})
    
    # Flatten column names if needed (pivot creates MultiIndex)
    if isinstance(df_pivot.columns, pd.MultiIndex):
        df_pivot.columns = [col[1] if col[1] else col[0] for col in df_pivot.columns]
    
    # Ensure we have the expected columns
    keep_cols = ["date", "asset"] + available_fields
    df_pivot = df_pivot[[c for c in keep_cols if c in df_pivot.columns]]
    
    # Convert date to date object
    if "date" in df_pivot.columns:
        df_pivot["date"] = pd.to_datetime(df_pivot["date"]).dt.date
    
    # Rename vol to volume if present
    if "vol" in df_pivot.columns:
        df_pivot = df_pivot.rename(columns={"vol": "volume"})
    
    return df_pivot.sort_values(["asset", "date"])


def _upsert_crypto_universe(
    client: DatabaseClient,
    asset_ids: List[str],
    start_date: date,
) -> int:
    """Upsert crypto universe data into database.
    
    Args:
        client: Database client
        asset_ids: List of asset IDs (symbols like "BTC", "ETH")
        start_date: First date of data
        
    Returns:
        Number of records inserted/updated
    """
    if not asset_ids:
        return 0
    
    records = []
    for asset_id in asset_ids:
        # For crypto, asset_id is typically just the symbol
        symbol = asset_id
        name = asset_id  # Default name to symbol if not available
        
        records.append((
            asset_id, symbol, name,
            None, None, True,  # first_seen_date, last_seen_date, is_active (derived from data)
        ))
    
    if not records:
        return 0
    
    with client.get_connection() as conn:
        with conn.cursor() as cur:
            # Use INSERT ... ON CONFLICT UPDATE for upsert
            cur.executemany("""
                INSERT INTO crypto_universe 
                    (asset_id, symbol, name, first_seen_date, last_seen_date, is_active, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (asset_id) 
                DO UPDATE SET
                    symbol = EXCLUDED.symbol,
                    name = EXCLUDED.name,
                    first_seen_date = COALESCE(crypto_universe.first_seen_date, EXCLUDED.first_seen_date),
                    last_seen_date = GREATEST(
                        COALESCE(crypto_universe.last_seen_date, '1900-01-01'::date),
                        COALESCE(EXCLUDED.last_seen_date, '1900-01-01'::date)
                    ),
                    is_active = EXCLUDED.is_active,
                    updated_at = CURRENT_TIMESTAMP
            """, records)
            conn.commit()
            return len(records)


def _upsert_crypto_ohlcv(
    client: DatabaseClient,
    df: pd.DataFrame,
    batch_size: int = 1000,
) -> int:
    """Upsert crypto OHLCV data into database.
    
    Args:
        client: Database client
        df: DataFrame with OHLCV data
        batch_size: Batch size for inserts
        
    Returns:
        Number of rows inserted/updated
    """
    _require_dependencies()
    
    if df.empty:
        return 0
    
    # Prepare data for insertion - only fields provided by Quantiacs
    records = []
    for _, row in df.iterrows():
        # Handle volume - may or may not be present
        volume = None
        if "volume" in df.columns and pd.notna(row.get("volume")):
            volume = float(row.get("volume", 0))
        
        record = (
            row.get("date"),
            row.get("asset"),
            float(row.get("open", 0)) if pd.notna(row.get("open")) else None,
            float(row.get("high", 0)) if pd.notna(row.get("high")) else None,
            float(row.get("low", 0)) if pd.notna(row.get("low")) else None,
            float(row.get("close", 0)) if pd.notna(row.get("close")) else None,
            volume,
            bool(row.get("is_liquid")) if pd.notna(row.get("is_liquid")) else None,
        )
        records.append(record)
    
    if not records:
        return 0
    
    rows_inserted = 0
    with client.get_connection() as conn:
        with conn.cursor() as cur:
            # Use INSERT ... ON CONFLICT DO NOTHING for idempotent inserts
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                cur.executemany("""
                    INSERT INTO crypto_ohlcv 
                        (date, asset_id, open, high, low, close, volume, is_liquid)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date, asset_id) DO NOTHING
                """, batch)
                rows_inserted += cur.rowcount
            
            conn.commit()
    
    return rows_inserted


def _log_crypto_ingestion(
    client: DatabaseClient,
    result: CryptoIngestionResult,
) -> None:
    """Log crypto ingestion result to database.
    
    Args:
        client: Database client
        result: Ingestion result
    """
    with client.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO crypto_ingestion_log
                    (start_date, end_date, assets_count, rows_inserted, 
                     status, error_message, duration_seconds)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                result.start_date,
                result.end_date,
                result.assets_count,
                result.rows_inserted,
                result.status,
                result.error_message,
                result.duration_seconds,
            ))
            conn.commit()


def ingest_crypto_data(
    client: DatabaseClient,
    start_date: date,
    end_date: date,
    assets: Optional[List[str]] = None,
    chunk_days: int = 365 * 2,
    force_refresh: bool = False,
) -> CryptoIngestionResult:
    """Ingest cryptocurrency data from Quantiacs into database.
    
    Args:
        client: Database client
        start_date: Start date for ingestion
        end_date: End date for ingestion
        assets: Optional list of asset symbols (e.g., ["BTC", "ETH"]). None loads all.
        chunk_days: Number of calendar days per chunk
        force_refresh: If True, re-ingest existing data
        
    Returns:
        CryptoIngestionResult with status and statistics
    """
    _require_dependencies()
    
    start_time = time.time()
    result = CryptoIngestionResult(
        start_date=start_date,
        end_date=end_date,
        assets_count=0,
        rows_inserted=0,
        status="failed",
    )
    
    try:
        # Check schema exists
        if not crypto_schema_exists(client):
            logger.error("Crypto database schema does not exist. Run init_crypto_db.py first.")
            result.error_message = "Schema does not exist"
            return result
        
        # Fetch data (Quantiacs loads all at once, but we'll process in chunks for memory)
        logger.info(f"Fetching crypto data: {len(assets) if assets else 'all'} assets, "
                   f"{start_date} to {end_date}")
        
        # Process in date chunks to manage memory
        date_chunks = _chunk_dates(start_date, end_date, chunk_days)
        logger.info(f"Processing {len(date_chunks)} date chunks")
        
        total_rows = 0
        errors = []
        all_asset_ids = set()
        
        for chunk_idx, (chunk_start, chunk_end) in enumerate(date_chunks, 1):
            try:
                logger.info(f"Processing chunk {chunk_idx}/{len(date_chunks)}: "
                          f"{chunk_start} to {chunk_end}")
                
                # Fetch data for this chunk
                # #region agent log
                try:
                    import json
                    log_path = '/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log'
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ALL","location":"crypto_ingestion.py:ingest_crypto_data:BEFORE_FETCH","message":"About to call _fetch_crypto_data","data":{"chunk_start":str(chunk_start),"chunk_end":str(chunk_end),"assets":assets},"timestamp":int(time.time()*1000)})+'\n')
                except: pass
                # #endregion
                da = _fetch_crypto_data(assets, chunk_start, chunk_end)
                # #region agent log
                try:
                    import json
                    log_path = '/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log'
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ALL","location":"crypto_ingestion.py:ingest_crypto_data:AFTER_FETCH","message":"After _fetch_crypto_data call","data":{"has_data":da is not None,"dims":list(da.dims) if hasattr(da,'dims') else None},"timestamp":int(time.time()*1000)})+'\n')
                except: pass
                # #endregion
                
                # Extract asset IDs
                if "asset" in da.coords:
                    asset_ids = list(da.coords["asset"].values)
                    all_asset_ids.update(asset_ids)
                else:
                    logger.warning("No asset dimension found in data")
                    continue
                
                # Convert to DataFrame
                df = _dataarray_to_dataframe(da)
                
                if df.empty:
                    logger.warning(f"No data in chunk {chunk_idx}")
                    continue
                
                # Upsert universe
                universe_count = _upsert_crypto_universe(client, asset_ids, chunk_start)
                logger.info(f"  Upserted {universe_count} universe records")
                
                # Upsert OHLCV
                rows = _upsert_crypto_ohlcv(client, df)
                total_rows += rows
                
                logger.info(f"  Inserted {rows} OHLCV rows")
                
            except Exception as e:
                error_msg = f"Error in chunk {chunk_idx} ({chunk_start} to {chunk_end}): {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                if not force_refresh:
                    continue
                else:
                    raise
        
        result.assets_count = len(all_asset_ids)
        result.rows_inserted = total_rows
        result.duration_seconds = time.time() - start_time
        
        # Update first_seen_date and last_seen_date in universe from actual data
        if total_rows > 0:
            logger.info("Updating crypto universe date ranges from actual data...")
            with client.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE crypto_universe u
                        SET 
                            first_seen_date = COALESCE(
                                u.first_seen_date,
                                (SELECT MIN(date) FROM crypto_ohlcv WHERE asset_id = u.asset_id)
                            ),
                            last_seen_date = GREATEST(
                                COALESCE(u.last_seen_date, '1900-01-01'::date),
                                (SELECT MAX(date) FROM crypto_ohlcv WHERE asset_id = u.asset_id)
                            ),
                            updated_at = CURRENT_TIMESTAMP
                        WHERE EXISTS (SELECT 1 FROM crypto_ohlcv WHERE asset_id = u.asset_id)
                    """)
                    conn.commit()
                    updated_count = cur.rowcount
                    logger.info(f"Updated date ranges for {updated_count} crypto assets")
        
        if errors:
            result.status = "partial"
            result.error_message = "; ".join(errors[:3])
        else:
            result.status = "success"
        
        # Log ingestion result
        _log_crypto_ingestion(client, result)
        
        logger.info(f"Crypto ingestion completed: {result.status}, "
                   f"{result.rows_inserted} rows in {result.duration_seconds:.1f}s")
        
        return result
        
    except Exception as e:
        result.duration_seconds = time.time() - start_time
        result.error_message = str(e)
        result.status = "failed"
        
        # Log failed ingestion
        try:
            _log_crypto_ingestion(client, result)
        except Exception:
            pass
        
        logger.error(f"Crypto ingestion failed: {e}")
        raise
