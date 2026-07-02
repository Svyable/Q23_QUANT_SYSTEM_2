"""q23.database.ingestion

Data ingestion pipeline from Quantiacs to PostgreSQL database.
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
from q23.database.schema import schema_exists

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
class IngestionResult:
    """Result of a data ingestion run."""
    start_date: date
    end_date: date
    assets_count: int
    rows_inserted: int
    status: str  # 'success', 'partial', 'failed'
    error_message: Optional[str] = None
    duration_seconds: float = 0.0


def _chunk_dates(start: date, end: date, chunk_days: int = 252 * 3) -> List[Tuple[date, date]]:
    """Split date range into chunks.
    
    Args:
        start: Start date
        end: End date
        chunk_days: Number of trading days per chunk (~3 years)
        
    Returns:
        List of (chunk_start, chunk_end) tuples
    """
    chunks = []
    current = start
    
    while current <= end:
        # Estimate chunk end (add chunk_days trading days)
        # Approximate: 252 trading days per year, so chunk_days ≈ years * 252
        chunk_end = current + timedelta(days=int(chunk_days * 365 / 252))
        if chunk_end > end:
            chunk_end = end
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    
    return chunks


def _fetch_ndx_universe(min_date: date) -> List[Dict[str, Any]]:
    """Fetch NDX universe from Quantiacs with retry logic.
    
    Args:
        min_date: Minimum date for universe membership
        
    Returns:
        List of stock dictionaries with id, name, sector, etc.
    """
    _require_dependencies()
    
    max_retries = 3
    retry_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            stocks_list = qndata.stocks.load_ndx_list(min_date=min_date.strftime("%Y-%m-%d"))
            logger.info(f"Fetched {len(stocks_list)} NDX stocks from Quantiacs")
            return stocks_list
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Quantiacs universe fetch failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"Failed to fetch NDX universe after {max_retries} attempts: {e}")
                raise


def _fetch_ndx_data(
    asset_ids: List[str],
    start_date: date,
    end_date: date,
) -> xr.Dataset:
    """Fetch NDX OHLCV data from Quantiacs.
    
    Args:
        asset_ids: List of asset IDs to fetch
        start_date: Start date
        end_date: End date
        
    Returns:
        xarray Dataset with OHLCV data
    """
    _require_dependencies()
    
    # Expand single-day ranges to avoid NetCDF issues
    # Quantiacs sometimes has issues with single-day fetches
    if start_date == end_date:
        # Expand to at least 3 days to avoid NetCDF errors
        from datetime import timedelta
        start_date = start_date - timedelta(days=1)
        end_date = end_date + timedelta(days=1)
        logger.info(f"Expanded single-day range to avoid NetCDF issues: {start_date} to {end_date}")
    
    max_retries = 3
    retry_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            # Use max_date to limit data fetch (more efficient)
            # Try without dims parameter first (let Quantiacs use defaults)
            try:
                data = qndata.stocks.load_ndx_data(
                    assets=asset_ids,
                    min_date=start_date.strftime("%Y-%m-%d"),
                    max_date=end_date.strftime("%Y-%m-%d"),
                    forward_order=True,
                )
            except Exception as dims_error:
                # If that fails, try with explicit dims
                logger.debug(f"Trying with explicit dims parameter: {dims_error}")
                data = qndata.stocks.load_ndx_data(
                    assets=asset_ids,
                    min_date=start_date.strftime("%Y-%m-%d"),
                    max_date=end_date.strftime("%Y-%m-%d"),
                    dims=("time", "field", "asset"),
                    forward_order=True,
                )
            
            logger.info(f"Fetched NDX data: {len(asset_ids)} assets, "
                       f"{data.sizes.get('time', 0)} time points")
            return data
        except Exception as e:
            error_str = str(e)
            # Check if it's a NetCDF error
            if "NetCDF" in error_str or "NC_UNLIMITED" in error_str:
                if attempt < max_retries - 1:
                    logger.warning(f"NetCDF error (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info("This may be a Quantiacs API issue. Trying with expanded date range...")
                    # Expand date range more aggressively
                    from datetime import timedelta
                    expanded_start = start_date - timedelta(days=5)
                    expanded_end = end_date + timedelta(days=5)
                    try:
                        data = qndata.stocks.load_ndx_data(
                            assets=asset_ids,
                            min_date=expanded_start.strftime("%Y-%m-%d"),
                            max_date=expanded_end.strftime("%Y-%m-%d"),
                            forward_order=True,
                        )
                        # Filter to original date range after fetching
                        if 'time' in data.coords and len(data.coords['time']) > 0:
                            time_coords = pd.to_datetime(data.coords['time'].values)
                            start_ts = pd.Timestamp(start_date)
                            end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # Include end date
                            time_mask = (time_coords >= start_ts) & (time_coords < end_ts)
                            
                            if time_mask.sum() > 0:
                                # Use isel with boolean mask indices
                                time_indices = np.where(time_mask)[0]
                                data = data.isel(time=time_indices)
                                logger.info(f"Successfully fetched with expanded range, filtered to {len(time_indices)} time points in original date range")
                            else:
                                # No data in exact range - this is common for:
                                # 1. Weekends/holidays (no trading)
                                # 2. Very recent dates (data not published yet)
                                # 3. Date range that spans non-trading days
                                
                                logger.warning(f"No data points found in exact range {start_date} to {end_date} after filtering")
                                if len(time_coords) > 0:
                                    actual_start = time_coords.min().date()
                                    actual_end = time_coords.max().date()
                                    logger.info(f"Fetched data contains: {actual_start} to {actual_end} ({len(time_coords)} time points)")
                                    
                                    # For recent dates (within 5 days), use what we got
                                    # For older dates, the gap might be real (weekends/holidays)
                                    days_from_today = (date.today() - end_date).days
                                    if days_from_today <= 5:
                                        logger.info(f"Recent date range - using all fetched data (data may not be available yet for exact dates)")
                                        # Don't filter - use all data we got, let upsert handle duplicates
                                    else:
                                        logger.info(f"Older date range with no matching data - likely weekend/holiday, using all fetched data")
                                        # Use all data - let the upsert logic handle duplicates via ON CONFLICT DO NOTHING
                                else:
                                    logger.warning(f"Fetched data has no time points at all")
                                    return data.isel(time=[])
                                # Return all fetched data (don't filter) - upsert will handle what's new
                                return data
                        else:
                            logger.info(f"Successfully fetched with expanded range (no time coordinate to filter)")
                        return data
                    except Exception as expanded_error:
                        logger.warning(f"Expanded range also failed: {expanded_error}")
            
            if attempt < max_retries - 1:
                logger.warning(f"Quantiacs API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"Failed to fetch NDX data after {max_retries} attempts: {e}")
                raise


def _dataset_to_dataframe(ds: xr.Dataset, chunk: Optional[xr.Dataset] = None) -> pd.DataFrame:
    """Convert xarray Dataset to pandas DataFrame (chunked if provided).
    
    Args:
        ds: Full dataset
        chunk: Optional chunk to process (if None, processes full dataset)
        
    Returns:
        DataFrame with columns: date, asset, and all fields
    """
    _require_dependencies()
    
    data_to_process = chunk if chunk is not None else ds
    
    # Select available fields - aligned with Quantiacs actual fields
    # Quantiacs provides: open, high, low, close, vol, divs, is_liquid, is_stock, is_spx, is_ndx
    wanted_fields = [
        "open", "high", "low", "close", "vol",
        "divs",
        "is_liquid", "is_stock", "is_spx", "is_ndx",
    ]
    available_fields = [f for f in wanted_fields if f in data_to_process.coords["field"].values]
    
    if not available_fields:
        logger.warning("No expected fields found in dataset")
        return pd.DataFrame()
    
    # xarray Dataset structure: data variable(s) with dims (time, field, asset)
    # Fields are in the "field" coordinate, not as separate data variables
    # We need to select by field coordinate and convert to DataFrame
    
    # Get the first (and usually only) data variable
    data_vars = list(data_to_process.data_vars.keys())
    
    if not data_vars:
        # If no data variables, try converting the whole dataset
        # This happens when Quantiacs returns a Dataset with all data in coordinates
        logger.debug("No data variables found, converting entire dataset...")
        try:
            df = data_to_process.to_dataframe().reset_index()
            df = df.rename(columns={"time": "date"})
            
            # Check if we have a 'field' column that needs pivoting
            if "field" in df.columns:
                # Find the value column (should be numeric and not field/time/asset/date)
                value_col = None
                for col in df.columns:
                    if col not in ["date", "asset", "field", "time"] and pd.api.types.is_numeric_dtype(df[col]):
                        value_col = col
                        break
                
                if value_col:
                    df = df.pivot_table(
                        index=["date", "asset"],
                        columns="field",
                        values=value_col,
                        aggfunc="first"
                    ).reset_index()
                    
                    # Flatten column names if MultiIndex
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[1] if col[1] else col[0] for col in df.columns.values]
                else:
                    logger.warning(f"Could not find value column. Available columns: {df.columns.tolist()}")
                    return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to convert dataset to DataFrame: {e}")
            logger.debug(f"Dataset structure: {data_to_process}")
            return pd.DataFrame()
    else:
        # Use the first data variable (usually there's only one)
        main_var = data_to_process[data_vars[0]]
        
        logger.debug(f"Using data variable: {data_vars[0]}, dims: {main_var.dims}")
        
        # Convert to DataFrame - this creates a multi-index with time, field, asset
        try:
            df = main_var.to_dataframe(name="value").reset_index()
            df = df.rename(columns={"time": "date"})
            
            # Pivot so each field becomes a column
            if "field" in df.columns and "value" in df.columns:
                df = df.pivot_table(
                    index=["date", "asset"],
                    columns="field",
                    values="value",
                    aggfunc="first"
                ).reset_index()
                
                # Flatten column names if MultiIndex
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[1] if col[1] else col[0] for col in df.columns.values]
            else:
                logger.warning(f"Unexpected DataFrame structure after conversion. Columns: {df.columns.tolist()}")
                logger.debug(f"DataFrame head:\n{df.head()}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to convert data variable to DataFrame: {e}")
            logger.debug(f"Data variable structure: {main_var}")
            return pd.DataFrame()
    
    # Select and order columns - ensure we have date, asset, and available fields
    keep_cols = ["date", "asset"] + available_fields
    df = df[[c for c in keep_cols if c in df.columns]]
    
    # Convert date to date object (remove time component)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    
    # Rename Quantiacs field names to database column names
    df = df.rename(columns={"vol": "volume", "divs": "dividend"})
    
    # Debug: Log if we have empty or all-NaN data
    if not df.empty:
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                total_count = len(df)
                if non_null_count == 0:
                    logger.warning(f"Column {col} has all NaN values ({total_count} rows)")
                elif non_null_count < total_count * 0.9:
                    logger.warning(f"Column {col} has {non_null_count}/{total_count} non-null values ({100*non_null_count/total_count:.1f}%)")
    
    return df.sort_values(["asset", "date"])


def _upsert_universe(
    client: DatabaseClient,
    stocks_list: List[Dict[str, Any]],
) -> int:
    """Upsert universe data into database.
    
    Args:
        client: Database client
        stocks_list: List of stock dictionaries from Quantiacs
        
    Returns:
        Number of records inserted/updated
    """
    if not stocks_list:
        return 0
    
    records = []
    for stock in stocks_list:
        asset_id = stock.get("id", "")
        if not asset_id:
            continue
        
        # Extract metadata - aligned with Quantiacs load_ndx_list() structure
        symbol = asset_id.split(":")[-1] if ":" in asset_id else asset_id
        name = stock.get("name", "")
        sector = stock.get("sector", "")
        exchange = stock.get("exchange", "")
        figi = stock.get("FIGI", stock.get("figi", ""))  # Handle both cases
        
        # Note: Quantiacs doesn't provide first_date/last_date in list
        # We'll derive these from actual data during ingestion
        # For now, set to NULL and update from data
        
        records.append((
            asset_id, symbol, name, sector, exchange, figi,
            None, None, True,  # first_seen_date, last_seen_date, is_active
        ))
    
    if not records:
        return 0
    
    with client.get_connection() as conn:
        with conn.cursor() as cur:
            # Use INSERT ... ON CONFLICT UPDATE for upsert
            cur.executemany("""
                INSERT INTO ndx_universe 
                    (asset_id, symbol, name, sector, exchange, figi, first_seen_date, last_seen_date, is_active, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (asset_id) 
                DO UPDATE SET
                    symbol = EXCLUDED.symbol,
                    name = EXCLUDED.name,
                    sector = EXCLUDED.sector,
                    exchange = EXCLUDED.exchange,
                    figi = EXCLUDED.figi,
                    first_seen_date = COALESCE(ndx_universe.first_seen_date, EXCLUDED.first_seen_date),
                    last_seen_date = GREATEST(ndx_universe.last_seen_date, EXCLUDED.last_seen_date),
                    is_active = EXCLUDED.is_active,
                    updated_at = CURRENT_TIMESTAMP
            """, records)
            conn.commit()
            return len(records)


def _upsert_ohlcv(
    client: DatabaseClient,
    df: pd.DataFrame,
    batch_size: int = 1000,
) -> int:
    """Upsert OHLCV data into database.
    
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
        # Handle NaN values properly - convert to None for database
        def safe_float(val):
            """Convert value to float, or None if NaN/None."""
            if pd.isna(val) or val is None:
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None
        
        def safe_int(val):
            """Convert value to int, or None if NaN/None."""
            if pd.isna(val) or val is None:
                return None
            try:
                return int(val)
            except (ValueError, TypeError):
                return None
        
        record = (
            row.get("date"),
            row.get("asset"),
            safe_float(row.get("open")),
            safe_float(row.get("high")),
            safe_float(row.get("low")),
            safe_float(row.get("close")),
            safe_int(row.get("volume")),
            safe_float(row.get("dividend")),
            bool(row.get("is_liquid")) if pd.notna(row.get("is_liquid")) else None,
            bool(row.get("is_stock")) if pd.notna(row.get("is_stock")) else None,
            bool(row.get("is_spx")) if pd.notna(row.get("is_spx")) else None,
            bool(row.get("is_ndx")) if pd.notna(row.get("is_ndx")) else None,
        )
        records.append(record)
    
    if not records:
        return 0
    
    rows_inserted = 0
    with client.get_connection() as conn:
        with conn.cursor() as cur:
            # Use INSERT ... ON CONFLICT DO NOTHING for idempotent inserts
            # (or DO UPDATE if we want to refresh existing data)
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                cur.executemany("""
                    INSERT INTO ndx_ohlcv 
                        (date, asset_id, open, high, low, close, volume,
                         dividend, is_liquid, is_stock, is_spx, is_ndx)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date, asset_id) DO NOTHING
                """, batch)
                rows_inserted += cur.rowcount
            
            conn.commit()
    
    return rows_inserted


def _log_ingestion(
    client: DatabaseClient,
    result: IngestionResult,
) -> None:
    """Log ingestion result to database.
    
    Args:
        client: Database client
        result: Ingestion result
    """
    with client.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO ndx_ingestion_log
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


def ingest_ndx_data(
    client: DatabaseClient,
    start_date: date,
    end_date: date,
    chunk_days: int = 252 * 3,
    force_refresh: bool = False,
) -> IngestionResult:
    """Ingest NDX data from Quantiacs into database.
    
    Args:
        client: Database client
        start_date: Start date for ingestion
        end_date: End date for ingestion
        chunk_days: Number of trading days per chunk
        force_refresh: If True, re-ingest existing data
        
    Returns:
        IngestionResult with status and statistics
    """
    _require_dependencies()
    
    start_time = time.time()
    result = IngestionResult(
        start_date=start_date,
        end_date=end_date,
        assets_count=0,
        rows_inserted=0,
        status="failed",
    )
    
    try:
        # Check schema exists
        if not schema_exists(client):
            logger.error("Database schema does not exist. Run init_ndx_db.py first.")
            result.error_message = "Schema does not exist"
            return result
        
        # Fetch universe
        logger.info(f"Fetching NDX universe (min_date={start_date})...")
        stocks_list = _fetch_ndx_universe(start_date)
        asset_ids = [s["id"] for s in stocks_list if s.get("id")]
        
        if not asset_ids:
            result.error_message = "No assets found in universe"
            return result
        
        result.assets_count = len(asset_ids)
        logger.info(f"Processing {len(asset_ids)} assets")
        
        # Upsert universe metadata
        universe_count = _upsert_universe(client, stocks_list)
        logger.info(f"Upserted {universe_count} universe records")
        
        # Update first_seen_date and last_seen_date from actual data after ingestion
        # This will be done after all data is loaded
        
        # Process data in chunks
        date_chunks = _chunk_dates(start_date, end_date, chunk_days)
        logger.info(f"Processing {len(date_chunks)} date chunks")
        
        total_rows = 0
        errors = []
        
        for chunk_idx, (chunk_start, chunk_end) in enumerate(date_chunks, 1):
            try:
                logger.info(f"Processing chunk {chunk_idx}/{len(date_chunks)}: "
                          f"{chunk_start} to {chunk_end}")
                
                # Fetch data for this chunk
                ds = _fetch_ndx_data(asset_ids, chunk_start, chunk_end)
                
                # Process in time chunks to manage memory
                times = ds.coords["time"].values
                if len(times) == 0:
                    logger.warning(f"No time points in chunk {chunk_idx}")
                    continue
                
                # Process in sub-chunks of ~1 year
                time_chunk_size = min(252, len(times))
                for t_start_idx in range(0, len(times), time_chunk_size):
                    t_end_idx = min(t_start_idx + time_chunk_size, len(times))
                    time_slice = slice(times[t_start_idx], times[t_end_idx - 1])
                    ds_chunk = ds.sel(time=time_slice)
                    
                    # Convert to DataFrame
                    df = _dataset_to_dataframe(ds, chunk=ds_chunk)
                    
                    if df.empty:
                        continue
                    
                    # Upsert to database
                    rows = _upsert_ohlcv(client, df)
                    total_rows += rows
                    
                    logger.info(f"  Inserted {rows} rows from time slice "
                              f"{pd.Timestamp(times[t_start_idx]).date()} to "
                              f"{pd.Timestamp(times[t_end_idx - 1]).date()}")
                
            except Exception as e:
                error_msg = f"Error in chunk {chunk_idx} ({chunk_start} to {chunk_end}): {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                if not force_refresh:
                    # Continue with next chunk on error
                    continue
                else:
                    raise
        
        result.rows_inserted = total_rows
        result.duration_seconds = time.time() - start_time
        
        # Update first_seen_date and last_seen_date in universe from actual data
        if total_rows > 0:
            logger.info("Updating universe date ranges from actual data...")
            with client.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE ndx_universe u
                        SET 
                            first_seen_date = COALESCE(
                                u.first_seen_date,
                                (SELECT MIN(date) FROM ndx_ohlcv WHERE asset_id = u.asset_id)
                            ),
                            last_seen_date = GREATEST(
                                COALESCE(u.last_seen_date, '1900-01-01'::date),
                                (SELECT MAX(date) FROM ndx_ohlcv WHERE asset_id = u.asset_id)
                            ),
                            updated_at = CURRENT_TIMESTAMP
                        WHERE EXISTS (SELECT 1 FROM ndx_ohlcv WHERE asset_id = u.asset_id)
                    """)
                    conn.commit()
                    updated_count = cur.rowcount
                    logger.info(f"Updated date ranges for {updated_count} assets")
        
        if errors:
            result.status = "partial"
            result.error_message = "; ".join(errors[:3])  # Limit error message length
        else:
            result.status = "success"
        
        # Log ingestion result
        _log_ingestion(client, result)
        
        logger.info(f"Ingestion completed: {result.status}, "
                   f"{result.rows_inserted} rows in {result.duration_seconds:.1f}s")
        
        return result
        
    except Exception as e:
        result.duration_seconds = time.time() - start_time
        result.error_message = str(e)
        result.status = "failed"
        
        # Log failed ingestion
        try:
            _log_ingestion(client, result)
        except Exception:
            pass  # Don't fail on logging error
        
        logger.error(f"Ingestion failed: {e}")
        raise
