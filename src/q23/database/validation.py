"""q23.database.validation

Data validation and quality checks for NDX database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Dict, Any, Optional, Tuple

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

from q23.database.client import DatabaseClient

logger = logging.getLogger(__name__)


def _require_dependencies() -> None:
    """Check that required dependencies are available."""
    if pd is None:
        raise ImportError("pandas is required for data validation")
    if np is None:
        raise ImportError("numpy is required for data validation")


@dataclass
class ValidationResult:
    """Result of data validation checks."""
    check_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None


def validate_date_continuity(
    client: DatabaseClient,
    asset_id: str,
    start_date: date,
    end_date: date,
) -> ValidationResult:
    """Check for missing trading days in date range.
    
    Args:
        client: Database client
        asset_id: Asset ID to check
        start_date: Start date
        end_date: End date
        
    Returns:
        ValidationResult with continuity check results
    """
    _require_dependencies()
    
    try:
        # Get all dates for this asset
        ohlcv_data = client.get_ohlcv(asset_id, start_date, end_date)
        if not ohlcv_data:
            return ValidationResult(
                check_name="date_continuity",
                passed=False,
                message=f"No data found for {asset_id}",
            )
        
        dates = sorted([pd.Timestamp(row["date"]).date() for row in ohlcv_data])
        
        # Check for gaps (weekends/holidays are expected, but large gaps are suspicious)
        gaps = []
        for i in range(len(dates) - 1):
            gap_days = (dates[i + 1] - dates[i]).days
            if gap_days > 5:  # More than 5 days suggests missing data
                gaps.append((dates[i], dates[i + 1], gap_days))
        
        if gaps:
            return ValidationResult(
                check_name="date_continuity",
                passed=False,
                message=f"Found {len(gaps)} suspicious gaps in date range",
                details={"gaps": gaps, "total_dates": len(dates)},
            )
        
        return ValidationResult(
            check_name="date_continuity",
            passed=True,
            message=f"Date continuity OK: {len(dates)} dates",
            details={"total_dates": len(dates)},
        )
    except Exception as e:
        return ValidationResult(
            check_name="date_continuity",
            passed=False,
            message=f"Error checking continuity: {e}",
        )


def validate_price_relationships(
    client: DatabaseClient,
    asset_id: str,
    start_date: date,
    end_date: date,
) -> ValidationResult:
    """Validate price relationships (high >= low, etc.).
    
    Args:
        client: Database client
        asset_id: Asset ID to check
        start_date: Start date
        end_date: End date
        
    Returns:
        ValidationResult with price validation results
    """
    _require_dependencies()
    
    try:
        ohlcv_data = client.get_ohlcv(asset_id, start_date, end_date)
        if not ohlcv_data:
            return ValidationResult(
                check_name="price_relationships",
                passed=False,
                message=f"No data found for {asset_id}",
            )
        
        df = pd.DataFrame(ohlcv_data)
        
        errors = []
        
        # Check high >= low
        invalid_hl = df[df["high"] < df["low"]]
        if not invalid_hl.empty:
            errors.append(f"high < low: {len(invalid_hl)} rows")
        
        # Check high >= open
        invalid_ho = df[df["high"] < df["open"]]
        if not invalid_ho.empty:
            errors.append(f"high < open: {len(invalid_ho)} rows")
        
        # Check high >= close
        invalid_hc = df[df["high"] < df["close"]]
        if not invalid_hc.empty:
            errors.append(f"high < close: {len(invalid_hc)} rows")
        
        # Check low <= open
        invalid_lo = df[df["low"] > df["open"]]
        if not invalid_lo.empty:
            errors.append(f"low > open: {len(invalid_lo)} rows")
        
        # Check low <= close
        invalid_lc = df[df["low"] > df["close"]]
        if not invalid_lc.empty:
            errors.append(f"low > close: {len(invalid_lc)} rows")
        
        # Check for negative prices
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            negative = df[df[col] < 0]
            if not negative.empty:
                errors.append(f"negative {col}: {len(negative)} rows")
        
        if errors:
            return ValidationResult(
                check_name="price_relationships",
                passed=False,
                message=f"Found {len(errors)} price relationship errors",
                details={"errors": errors, "total_rows": len(df)},
            )
        
        return ValidationResult(
            check_name="price_relationships",
            passed=True,
            message=f"Price relationships OK: {len(df)} rows",
            details={"total_rows": len(df)},
        )
    except Exception as e:
        return ValidationResult(
            check_name="price_relationships",
            passed=False,
            message=f"Error validating prices: {e}",
        )


def validate_volume(
    client: DatabaseClient,
    asset_id: str,
    start_date: date,
    end_date: date,
) -> ValidationResult:
    """Validate volume data (non-negative, reasonable values).
    
    Args:
        client: Database client
        asset_id: Asset ID to check
        start_date: Start date
        end_date: End date
        
    Returns:
        ValidationResult with volume validation results
    """
    _require_dependencies()
    
    try:
        ohlcv_data = client.get_ohlcv(asset_id, start_date, end_date)
        if not ohlcv_data:
            return ValidationResult(
                check_name="volume",
                passed=False,
                message=f"No data found for {asset_id}",
            )
        
        df = pd.DataFrame(ohlcv_data)
        
        errors = []
        
        # Check for negative volume
        negative_vol = df[df["volume"] < 0]
        if not negative_vol.empty:
            errors.append(f"negative volume: {len(negative_vol)} rows")
        
        # Check for zero volume (might be valid for some days, but flag if excessive)
        zero_vol = df[df["volume"] == 0]
        zero_vol_pct = len(zero_vol) / len(df) * 100
        if zero_vol_pct > 10:  # More than 10% zero volume is suspicious
            errors.append(f"excessive zero volume: {len(zero_vol)} rows ({zero_vol_pct:.1f}%)")
        
        if errors:
            return ValidationResult(
                check_name="volume",
                passed=False,
                message=f"Found {len(errors)} volume errors",
                details={"errors": errors, "total_rows": len(df)},
            )
        
        return ValidationResult(
            check_name="volume",
            passed=True,
            message=f"Volume validation OK: {len(df)} rows",
            details={"total_rows": len(df), "zero_volume_pct": zero_vol_pct},
        )
    except Exception as e:
        return ValidationResult(
            check_name="volume",
            passed=False,
            message=f"Error validating volume: {e}",
        )


def validate_duplicates(
    client: DatabaseClient,
    start_date: date,
    end_date: date,
) -> ValidationResult:
    """Check for duplicate records (same date + asset_id).
    
    Args:
        client: Database client
        start_date: Start date
        end_date: End date
        
    Returns:
        ValidationResult with duplicate check results
    """
    _require_dependencies()
    
    try:
        with client.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT date, asset_id, COUNT(*) as cnt
                    FROM ndx_ohlcv
                    WHERE date >= %s AND date <= %s
                    GROUP BY date, asset_id
                    HAVING COUNT(*) > 1
                """, (start_date, end_date))
                
                duplicates = cur.fetchall()
                
                if duplicates:
                    return ValidationResult(
                        check_name="duplicates",
                        passed=False,
                        message=f"Found {len(duplicates)} duplicate date+asset_id combinations",
                        details={"duplicates": duplicates[:10]},  # Limit details
                    )
                
                return ValidationResult(
                    check_name="duplicates",
                    passed=True,
                    message="No duplicate records found",
                )
    except Exception as e:
        return ValidationResult(
            check_name="duplicates",
            passed=False,
            message=f"Error checking duplicates: {e}",
        )


def validate_all(
    client: DatabaseClient,
    sample_assets: Optional[List[str]] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[ValidationResult]:
    """Run all validation checks.
    
    Args:
        client: Database client
        sample_assets: Optional list of asset IDs to validate (defaults to all active)
        start_date: Optional start date (defaults to earliest available)
        end_date: Optional end date (defaults to latest available)
        
    Returns:
        List of ValidationResult objects
    """
    _require_dependencies()
    
    results = []
    
    # Get date range if not provided
    if start_date is None or end_date is None:
        date_range = client.get_date_range()
        if date_range:
            if start_date is None:
                start_date = date_range[0]
            if end_date is None:
                end_date = date_range[1]
        else:
            results.append(ValidationResult(
                check_name="data_availability",
                passed=False,
                message="No data available in database",
            ))
            return results
    
    # Get assets to validate
    if sample_assets is None:
        universe = client.get_universe()
        sample_assets = [u["asset_id"] for u in universe[:10]]  # Sample first 10
    
    logger.info(f"Running validation on {len(sample_assets)} assets, "
               f"{start_date} to {end_date}")
    
    # Check duplicates (global check)
    logger.info("Checking for duplicate records...")
    results.append(validate_duplicates(client, start_date, end_date))
    
    # Per-asset checks
    for asset_id in sample_assets:
        logger.info(f"Validating {asset_id}...")
        
        # Date continuity
        results.append(validate_date_continuity(
            client, asset_id, start_date, end_date
        ))
        
        # Price relationships
        results.append(validate_price_relationships(
            client, asset_id, start_date, end_date
        ))
        
        # Volume
        results.append(validate_volume(
            client, asset_id, start_date, end_date
        ))
    
    return results
