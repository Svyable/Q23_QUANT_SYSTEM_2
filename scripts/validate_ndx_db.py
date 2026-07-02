#!/usr/bin/env python
"""Validate NDX database data quality.

This script runs data quality checks on the NDX database:
- Date continuity (no missing trading days)
- Price relationships (high >= low, etc.)
- Volume validation
- Duplicate detection

Usage:
    python scripts/validate_ndx_db.py
    python scripts/validate_ndx_db.py --start-date 2015-01-01 --end-date 2025-01-24
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

# Load .env file BEFORE any imports
def _load_env():
    """Load .env file into environment."""
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k not in os.environ:
                    os.environ[k] = v
    return project_root

# Set up paths
project_root = _load_env()
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from q23.database.client import DatabaseClient, get_client
from q23.database.validation import validate_all
from q23.shared.config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> date:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")


def main():
    parser = argparse.ArgumentParser(
        description="Validate NDX database data quality"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for validation (YYYY-MM-DD, defaults to earliest available)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for validation (YYYY-MM-DD, defaults to latest available)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of assets to validate (default: 10)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help=f"Database host (default: {cfg.database.HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Database port (default: {cfg.database.PORT})"
    )
    parser.add_argument(
        "--dbname",
        type=str,
        default=None,
        help=f"Database name (default: {cfg.database.NAME})"
    )
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help=f"Database user (default: {cfg.database.USER})"
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="Database password (default: from env or config)"
    )
    
    args = parser.parse_args()
    
    # Parse dates if provided
    start_date = None
    end_date = None
    if args.start_date:
        try:
            start_date = parse_date(args.start_date)
        except ValueError as e:
            logger.error(f"Date parsing error: {e}")
            return 1
    if args.end_date:
        try:
            end_date = parse_date(args.end_date)
        except ValueError as e:
            logger.error(f"Date parsing error: {e}")
            return 1
    
    # Create client
    client = DatabaseClient(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )
    
    try:
        # Test connection
        logger.info(f"Connecting to database: {cfg.database.NAME}@{cfg.database.HOST}:{cfg.database.PORT}")
        if not client.test_connection():
            logger.error("Failed to connect to database")
            return 1
        
        # Get sample assets
        universe = client.get_universe()
        sample_assets = [u["asset_id"] for u in universe[:args.sample_size]]
        
        logger.info(f"Validating {len(sample_assets)} assets")
        if start_date and end_date:
            logger.info(f"Date range: {start_date} to {end_date}")
        
        # Run validation
        results = validate_all(
            client=client,
            sample_assets=sample_assets,
            start_date=start_date,
            end_date=end_date,
        )
        
        # Report results
        logger.info("=" * 60)
        logger.info("Validation Results:")
        logger.info("=" * 60)
        
        passed = 0
        failed = 0
        
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            logger.info(f"[{status}] {result.check_name}: {result.message}")
            
            if result.details:
                for key, value in result.details.items():
                    logger.info(f"  {key}: {value}")
            
            if result.passed:
                passed += 1
            else:
                failed += 1
        
        logger.info("=" * 60)
        logger.info(f"Summary: {passed} passed, {failed} failed")
        logger.info("=" * 60)
        
        return 0 if failed == 0 else 1
        
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
