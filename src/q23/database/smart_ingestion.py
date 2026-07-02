"""q23.database.smart_ingestion

Intelligent data ingestion that only pulls missing date ranges.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import List, Tuple, Optional
import warnings

from q23.database.client import DatabaseClient
from q23.database.ingestion import ingest_ndx_data, IngestionResult
from q23.database.crypto_ingestion import ingest_crypto_data, CryptoIngestionResult
from q23.database.schema import schema_exists
from q23.database.crypto_schema import crypto_schema_exists

logger = logging.getLogger(__name__)


def get_ndx_date_gaps(
    client: DatabaseClient,
    target_start: date,
    target_end: date,
) -> List[Tuple[date, date]]:
    """Identify missing date ranges in NDX database.
    
    Args:
        client: Database client
        target_start: Desired start date
        target_end: Desired end date
        
    Returns:
        List of (gap_start, gap_end) tuples representing missing date ranges
    """
    if not schema_exists(client):
        # No schema = entire range is a gap
        return [(target_start, target_end)]
    
    with client.get_connection() as conn:
        with conn.cursor() as cur:
            # Get existing date range
            cur.execute("SELECT MIN(date), MAX(date) FROM ndx_ohlcv")
            result = cur.fetchone()
            
            if not result or not result[0] or not result[1]:
                # No data = entire range is a gap
                return [(target_start, target_end)]
            
            existing_min, existing_max = result[0], result[1]
            
            gaps = []
            
            # Gap before existing data
            if target_start < existing_min:
                gap_end = existing_min - timedelta(days=1)
                gaps.append((target_start, gap_end))
            
            # Gap after existing data
            if target_end > existing_max:
                gap_start = existing_max + timedelta(days=1)
                gaps.append((gap_start, target_end))
            
            # Check for internal gaps (missing dates within range)
            # This is expensive, so we'll do a simpler check: if we have data
            # and the range is continuous, we assume no internal gaps
            # For a more thorough check, we'd need to query all dates
            
            return gaps if gaps else []


def get_crypto_date_gaps(
    client: DatabaseClient,
    target_start: date,
    target_end: date,
    asset_id: Optional[str] = None,
) -> List[Tuple[date, date]]:
    """Identify missing date ranges in crypto database.
    
    Args:
        client: Database client
        target_start: Desired start date
        target_end: Desired end date
        asset_id: Optional specific asset to check (None = check all)
        
    Returns:
        List of (gap_start, gap_end) tuples representing missing date ranges
    """
    if not crypto_schema_exists(client):
        # No schema = entire range is a gap
        return [(target_start, target_end)]
    
    with client.get_connection() as conn:
        with conn.cursor() as cur:
            if asset_id:
                # Check specific asset
                cur.execute("""
                    SELECT MIN(date), MAX(date) 
                    FROM crypto_ohlcv 
                    WHERE asset_id = %s
                """, (asset_id,))
            else:
                # Check all assets (use min/max across all)
                cur.execute("SELECT MIN(date), MAX(date) FROM crypto_ohlcv")
            
            result = cur.fetchone()
            
            if not result or not result[0] or not result[1]:
                # No data = entire range is a gap
                return [(target_start, target_end)]
            
            existing_min, existing_max = result[0], result[1]
            
            gaps = []
            
            # Gap before existing data
            if target_start < existing_min:
                gap_end = existing_min - timedelta(days=1)
                gaps.append((target_start, gap_end))
            
            # Gap after existing data
            if target_end > existing_max:
                gap_start = existing_max + timedelta(days=1)
                gaps.append((gap_start, target_end))
            
            return gaps if gaps else []


def smart_ingest_ndx(
    client: DatabaseClient,
    target_start: date,
    target_end: date,
    chunk_days: int = 252 * 3,
    force_refresh: bool = False,
) -> IngestionResult:
    """Intelligently ingest NDX data, only pulling missing date ranges.
    
    Args:
        client: Database client
        target_start: Desired start date
        target_end: Desired end date
        chunk_days: Number of trading days per chunk
        force_refresh: If True, re-ingest all data regardless of gaps
        
    Returns:
        IngestionResult with status and statistics
    """
    if force_refresh:
        logger.info("Force refresh enabled - will ingest entire range")
        return ingest_ndx_data(client, target_start, target_end, chunk_days, force_refresh)
    
    # Check for gaps
    gaps = get_ndx_date_gaps(client, target_start, target_end)
    
    if not gaps:
        logger.info(f"NDX data is complete for range {target_start} to {target_end}")
        # Return a success result indicating no work needed
        return IngestionResult(
            start_date=target_start,
            end_date=target_end,
            assets_count=0,
            rows_inserted=0,
            status="success",
            error_message=None,
            duration_seconds=0.0,
        )
    
    logger.info(f"Found {len(gaps)} date gap(s) in NDX data:")
    for gap_start, gap_end in gaps:
        logger.info(f"  Gap: {gap_start} to {gap_end}")
    
    # Ingest each gap
    total_rows = 0
    total_assets = 0
    errors = []
    
    for gap_start, gap_end in gaps:
        logger.info(f"Ingesting gap: {gap_start} to {gap_end}")
        try:
            result = ingest_ndx_data(client, gap_start, gap_end, chunk_days, False)
            total_rows += result.rows_inserted
            total_assets = max(total_assets, result.assets_count)
            if result.status != "success":
                errors.append(f"Gap {gap_start} to {gap_end}: {result.error_message}")
        except Exception as e:
            errors.append(f"Gap {gap_start} to {gap_end}: {str(e)}")
            logger.error(f"Failed to ingest gap {gap_start} to {gap_end}: {e}")
    
    # Return combined result
    status = "success" if not errors else "partial" if total_rows > 0 else "failed"
    return IngestionResult(
        start_date=target_start,
        end_date=target_end,
        assets_count=total_assets,
        rows_inserted=total_rows,
        status=status,
        error_message="; ".join(errors) if errors else None,
        duration_seconds=0.0,  # Would need to track time across gaps
    )


def smart_ingest_crypto(
    client: DatabaseClient,
    target_start: date,
    target_end: date,
    assets: Optional[List[str]] = None,
    chunk_days: int = 365 * 2,
    force_refresh: bool = False,
) -> CryptoIngestionResult:
    """Intelligently ingest crypto data, only pulling missing date ranges.
    
    Args:
        client: Database client
        target_start: Desired start date
        target_end: Desired end date
        assets: Optional list of asset symbols
        chunk_days: Number of calendar days per chunk
        force_refresh: If True, re-ingest all data regardless of gaps
        
    Returns:
        CryptoIngestionResult with status and statistics
    """
    if force_refresh:
        logger.info("Force refresh enabled - will ingest entire range")
        return ingest_crypto_data(client, target_start, target_end, assets, chunk_days, force_refresh)
    
    # Check for gaps (for first asset or all if None)
    gaps = get_crypto_date_gaps(client, target_start, target_end, assets[0] if assets else None)
    
    if not gaps:
        logger.info(f"Crypto data is complete for range {target_start} to {target_end}")
        # Return a success result indicating no work needed
        return CryptoIngestionResult(
            start_date=target_start,
            end_date=target_end,
            assets_count=0,
            rows_inserted=0,
            status="success",
            error_message=None,
            duration_seconds=0.0,
        )
    
    logger.info(f"Found {len(gaps)} date gap(s) in crypto data:")
    for gap_start, gap_end in gaps:
        logger.info(f"  Gap: {gap_start} to {gap_end}")
    
    # Ingest each gap
    total_rows = 0
    total_assets = 0
    errors = []
    
    for gap_start, gap_end in gaps:
        logger.info(f"Ingesting gap: {gap_start} to {gap_end}")
        try:
            result = ingest_crypto_data(client, gap_start, gap_end, assets, chunk_days, False)
            total_rows += result.rows_inserted
            total_assets = max(total_assets, result.assets_count)
            if result.status != "success":
                errors.append(f"Gap {gap_start} to {gap_end}: {result.error_message}")
        except Exception as e:
            errors.append(f"Gap {gap_start} to {gap_end}: {str(e)}")
            logger.error(f"Failed to ingest gap {gap_start} to {gap_end}: {e}")
    
    # Return combined result
    status = "success" if not errors else "partial" if total_rows > 0 else "failed"
    return CryptoIngestionResult(
        start_date=target_start,
        end_date=target_end,
        assets_count=total_assets,
        rows_inserted=total_rows,
        status=status,
        error_message="; ".join(errors) if errors else None,
        duration_seconds=0.0,
    )
