"""q23.database.blockchain_ingestion

Blockchain metrics data ingestion pipeline from Quantiacs to PostgreSQL database.
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
from q23.database.crypto_schema import blockchain_schema_exists

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
class BlockchainIngestionResult:
    """Result of a blockchain data ingestion run."""
    metric_id: str
    start_date: date
    end_date: date
    rows_inserted: int
    status: str  # 'success', 'partial', 'failed'
    error_message: Optional[str] = None
    duration_seconds: float = 0.0


def fetch_blockchain_metrics_list() -> pd.DataFrame:
    """Fetch list of available blockchain metrics from Quantiacs.
    
    Returns:
        DataFrame with columns: id, name, description, etc.
    """
    _require_dependencies()
    
    try:
        metrics = qndata.blockchaincom_load_list()
        
        # Convert to DataFrame
        if isinstance(metrics, pd.DataFrame):
            df = metrics
        elif hasattr(metrics, 'to_pandas'):
            df = metrics.to_pandas()
        else:
            # Try xarray conversion
            if isinstance(metrics, xr.DataArray) or isinstance(metrics, xr.Dataset):
                df = metrics.to_pandas()
            else:
                # Try as list/dict
                df = pd.DataFrame(metrics)
        
        logger.info(f"Fetched {len(df)} blockchain metrics from Quantiacs")
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch blockchain metrics list: {e}")
        raise


def _upsert_blockchain_metrics_metadata(
    client: DatabaseClient,
    metrics_df: pd.DataFrame,
) -> int:
    """Upsert blockchain metrics metadata into database.
    
    Args:
        client: Database client
        metrics_df: DataFrame with metric metadata (must have 'id' column)
        
    Returns:
        Number of records inserted/updated
    """
    if metrics_df.empty:
        return 0
    
    records = []
    for _, row in metrics_df.iterrows():
        metric_id = str(row.get('id', ''))
        if not metric_id:
            continue
        
        name = str(row.get('name', '')) if pd.notna(row.get('name')) else None
        description = str(row.get('description', '')) if pd.notna(row.get('description')) else None
        unit = str(row.get('unit', '')) if pd.notna(row.get('unit')) else None
        
        records.append((
            metric_id,
            name,
            description,
            unit,
            'blockchain.com',  # source
            None,  # first_seen_date (derived from data)
            None,  # last_seen_date (derived from data)
            True,  # is_active
        ))
    
    if not records:
        return 0
    
    with client.get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO crypto_blockchain_metrics 
                    (metric_id, name, description, unit, source, first_seen_date, last_seen_date, is_active, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (metric_id) 
                DO UPDATE SET
                    name = COALESCE(EXCLUDED.name, crypto_blockchain_metrics.name),
                    description = COALESCE(EXCLUDED.description, crypto_blockchain_metrics.description),
                    unit = COALESCE(EXCLUDED.unit, crypto_blockchain_metrics.unit),
                    updated_at = CURRENT_TIMESTAMP
            """, records)
            conn.commit()
            return len(records)


def _fetch_blockchain_data(
    metric_id: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> xr.DataArray:
    """Fetch blockchain metric data from Quantiacs.
    
    Args:
        metric_id: Metric ID (e.g., 'miners-revenue')
        start_date: Optional start date
        end_date: Optional end date
        
    Returns:
        xarray DataArray with time-series data
    """
    _require_dependencies()
    
    max_retries = 3
    retry_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            kwargs = {'id': metric_id}
            
            if start_date:
                kwargs['min_date'] = start_date.strftime("%Y-%m-%d")
            if end_date:
                kwargs['max_date'] = end_date.strftime("%Y-%m-%d")
            
            data = qndata.blockchaincom_load_data(**kwargs)
            
            logger.info(f"Fetched blockchain data for {metric_id}: {data.sizes.get('time', 0)} time points")
            return data
            
        except Exception as e:
            error_str = str(e)
            if attempt < max_retries - 1:
                logger.warning(f"Blockchain API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                time.sleep(retry_delay * (attempt + 1))
            else:
                logger.error(f"Failed to fetch blockchain data after {max_retries} attempts: {e}")
                raise
    
    raise RuntimeError("Should not reach here")


def _dataarray_to_dataframe(da: xr.DataArray) -> pd.DataFrame:
    """Convert xarray DataArray to pandas DataFrame.
    
    Args:
        da: DataArray with time dimension
        
    Returns:
        DataFrame with columns: date, value
    """
    _require_dependencies()
    
    # Convert to DataFrame
    df = da.to_dataframe().reset_index()
    
    # Find the time column
    time_col = None
    for col in ['time', 'date']:
        if col in df.columns:
            time_col = col
            break
    
    if time_col is None:
        logger.warning("No time column found in blockchain data")
        return pd.DataFrame()
    
    # Find the value column (should be the numeric one that's not time)
    value_col = None
    for col in df.columns:
        if col != time_col and pd.api.types.is_numeric_dtype(df[col]):
            value_col = col
            break
    
    if value_col is None:
        logger.warning("No value column found in blockchain data")
        return pd.DataFrame()
    
    # Create clean DataFrame
    result_df = pd.DataFrame({
        'date': pd.to_datetime(df[time_col]).dt.date,
        'value': df[value_col]
    })
    
    # Remove NaN values
    result_df = result_df.dropna(subset=['value'])
    
    return result_df.sort_values('date')


def _upsert_blockchain_data(
    client: DatabaseClient,
    metric_id: str,
    df: pd.DataFrame,
    batch_size: int = 1000,
) -> int:
    """Upsert blockchain metric data into database.
    
    Args:
        client: Database client
        metric_id: Metric ID
        df: DataFrame with columns: date, value
        batch_size: Batch size for inserts
        
    Returns:
        Number of rows inserted/updated
    """
    _require_dependencies()
    
    if df.empty:
        return 0
    
    records = []
    for _, row in df.iterrows():
        record = (
            row.get("date"),
            metric_id,
            float(row.get("value", 0)) if pd.notna(row.get("value")) else None,
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
                    INSERT INTO crypto_blockchain_data 
                        (date, metric_id, value)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (date, metric_id) DO NOTHING
                """, batch)
                rows_inserted += cur.rowcount
            
            conn.commit()
    
    return rows_inserted


def _log_blockchain_ingestion(
    client: DatabaseClient,
    result: BlockchainIngestionResult,
) -> None:
    """Log blockchain ingestion result to database.
    
    Args:
        client: Database client
        result: Ingestion result
    """
    with client.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO crypto_blockchain_ingestion_log
                    (metric_id, start_date, end_date, rows_inserted, 
                     status, error_message, duration_seconds)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                result.metric_id,
                result.start_date,
                result.end_date,
                result.rows_inserted,
                result.status,
                result.error_message,
                result.duration_seconds,
            ))
            conn.commit()


def ingest_blockchain_metrics_list(client: DatabaseClient) -> int:
    """Fetch and store list of available blockchain metrics.
    
    Args:
        client: Database client
        
    Returns:
        Number of metrics stored
    """
    _require_dependencies()
    
    try:
        metrics_df = fetch_blockchain_metrics_list()
        count = _upsert_blockchain_metrics_metadata(client, metrics_df)
        logger.info(f"Stored {count} blockchain metrics metadata records")
        return count
    except Exception as e:
        logger.error(f"Failed to ingest blockchain metrics list: {e}")
        raise


def ingest_blockchain_data(
    client: DatabaseClient,
    metric_id: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    force_refresh: bool = False,
) -> BlockchainIngestionResult:
    """Ingest blockchain metric data from Quantiacs into database.
    
    Args:
        client: Database client
        metric_id: Metric ID (e.g., 'miners-revenue')
        start_date: Optional start date for ingestion
        end_date: Optional end date for ingestion (defaults to today)
        force_refresh: If True, re-ingest existing data
        
    Returns:
        BlockchainIngestionResult with status and statistics
    """
    _require_dependencies()
    
    start_time = time.time()
    
    # Default end_date to today
    if end_date is None:
        end_date = date.today()
    
    result = BlockchainIngestionResult(
        metric_id=metric_id,
        start_date=start_date or date(2010, 1, 1),  # Default to early date
        end_date=end_date,
        rows_inserted=0,
        status="failed",
    )
    
    try:
        # Check schema exists
        if not blockchain_schema_exists(client):
            logger.error("Blockchain database schema does not exist. Run init_crypto_db.py first.")
            result.error_message = "Schema does not exist"
            return result
        
        # Fetch data
        logger.info(f"Fetching blockchain data for metric: {metric_id}, "
                   f"{start_date or 'all'} to {end_date}")
        
        da = _fetch_blockchain_data(metric_id, start_date, end_date)
        
        # Convert to DataFrame
        df = _dataarray_to_dataframe(da)
        
        if df.empty:
            logger.warning(f"No data found for metric {metric_id}")
            result.status = "partial"
            result.error_message = "No data found"
            result.duration_seconds = time.time() - start_time
            _log_blockchain_ingestion(client, result)
            return result
        
        # Upsert data
        rows = _upsert_blockchain_data(client, metric_id, df)
        
        # Update first_seen_date and last_seen_date in metrics metadata
        if rows > 0:
            with client.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE crypto_blockchain_metrics
                        SET 
                            first_seen_date = COALESCE(
                                first_seen_date,
                                (SELECT MIN(date) FROM crypto_blockchain_data WHERE metric_id = %s)
                            ),
                            last_seen_date = GREATEST(
                                COALESCE(last_seen_date, '1900-01-01'::date),
                                (SELECT MAX(date) FROM crypto_blockchain_data WHERE metric_id = %s)
                            ),
                            updated_at = CURRENT_TIMESTAMP
                        WHERE metric_id = %s
                    """, (metric_id, metric_id, metric_id))
                    conn.commit()
        
        result.rows_inserted = rows
        result.duration_seconds = time.time() - start_time
        result.status = "success"
        
        # Log ingestion result
        _log_blockchain_ingestion(client, result)
        
        logger.info(f"Blockchain ingestion completed: {result.status}, "
                   f"{result.rows_inserted} rows in {result.duration_seconds:.1f}s")
        
        return result
        
    except Exception as e:
        result.duration_seconds = time.time() - start_time
        result.error_message = str(e)
        result.status = "failed"
        
        # Log failed ingestion
        try:
            _log_blockchain_ingestion(client, result)
        except Exception:
            pass
        
        logger.error(f"Blockchain ingestion failed: {e}")
        raise
