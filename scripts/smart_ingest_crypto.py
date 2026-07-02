#!/usr/bin/env python
"""Smart cryptocurrency data ingestion - only pulls missing date ranges.

This script intelligently checks what data already exists and only
ingests missing date ranges, making it safe to run repeatedly.

Usage:
    python scripts/smart_ingest_crypto.py
    python scripts/smart_ingest_crypto.py --assets BTC ETH
    python scripts/smart_ingest_crypto.py --force-refresh
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

# Ensure API_KEY is set
if not os.environ.get("API_KEY", "").strip():
    print("\nERROR: API_KEY is not set in .env file", file=sys.stderr)
    print(f"Add to {project_root}/.env:\n  API_KEY=your-key-here\n", file=sys.stderr)
    sys.exit(2)

from q23.database.client import DatabaseClient, get_client
from q23.database.smart_ingestion import smart_ingest_crypto, get_crypto_date_gaps
from q23.database.crypto_schema import crypto_schema_exists
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
        description="Smart cryptocurrency data ingestion - only pulls missing dates"
    )
    parser.add_argument(
        "--target-start",
        type=str,
        default="2014-01-01",
        help="Target start date (YYYY-MM-DD, default: 2014-01-01)"
    )
    parser.add_argument(
        "--target-end",
        type=str,
        default=None,
        help="Target end date (YYYY-MM-DD, default: today)"
    )
    parser.add_argument(
        "--assets",
        type=str,
        nargs="*",
        default=None,
        help="List of asset symbols (e.g., BTC ETH). Default: all available"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh of all data (ignore existing data)"
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=365 * 2,
        help="Number of calendar days per chunk (default: 730)"
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
    
    # Parse dates
    try:
        target_start = parse_date(args.target_start)
        if args.target_end:
            target_end = parse_date(args.target_end)
        else:
            # Default to today
            target_end = date.today()
    except ValueError as e:
        logger.error(f"Date parsing error: {e}")
        return 1
    
    if target_start > target_end:
        logger.error("Start date must be before end date")
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
        
        # Check schema exists
        if not crypto_schema_exists(client):
            logger.warning("Crypto schema does not exist. Initializing...")
            logger.info("Please run: python scripts/init_crypto_db.py")
            return 1
        
        # Check for gaps (unless force refresh)
        if not args.force_refresh:
            gaps = get_crypto_date_gaps(client, target_start, target_end, args.assets[0] if args.assets else None)
            if gaps:
                logger.info(f"Found {len(gaps)} date gap(s) to fill:")
                for gap_start, gap_end in gaps:
                    logger.info(f"  {gap_start} to {gap_end}")
            else:
                logger.info(f"Crypto data is already complete for {target_start} to {target_end}")
                logger.info("Use --force-refresh to re-ingest existing data")
                return 0
        
        assets_str = ", ".join(args.assets) if args.assets else "all available"
        logger.info("=" * 60)
        logger.info("Smart Cryptocurrency Data Ingestion")
        logger.info(f"Target range: {target_start} to {target_end}")
        logger.info(f"Assets: {assets_str}")
        logger.info("=" * 60)
        
        # Run smart ingestion
        result = smart_ingest_crypto(
            client=client,
            target_start=target_start,
            target_end=target_end,
            assets=args.assets,
            chunk_days=args.chunk_days,
            force_refresh=args.force_refresh,
        )
        
        # Report results
        logger.info("=" * 60)
        logger.info("Ingestion Summary:")
        logger.info(f"  Status: {result.status}")
        logger.info(f"  Assets: {result.assets_count}")
        logger.info(f"  Rows inserted: {result.rows_inserted:,}")
        if result.error_message:
            logger.warning(f"  Errors: {result.error_message}")
        logger.info("=" * 60)
        
        if result.status == "failed":
            return 1
        elif result.status == "partial":
            logger.warning("Ingestion completed with some errors")
            return 0
        else:
            logger.info("Ingestion completed successfully!")
            return 0
            
    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
