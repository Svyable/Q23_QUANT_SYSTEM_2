#!/usr/bin/env python
"""Ingest cryptocurrency market data from Quantiacs into PostgreSQL database.

This script fetches cryptocurrency historical market data from Quantiacs
and stores it in a local PostgreSQL database.

Usage:
    python scripts/ingest_crypto.py --start-date 2014-01-01 --end-date 2025-01-24
    python scripts/ingest_crypto.py --start-date 2014-01-01 --end-date 2025-01-24 --assets BTC ETH
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
from q23.database.crypto_ingestion import ingest_crypto_data
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
        description="Ingest cryptocurrency market data from Quantiacs into PostgreSQL"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)"
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
        help="Force refresh of existing data (re-ingest)"
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=365 * 2,  # ~2 years
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
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
    except ValueError as e:
        logger.error(f"Date parsing error: {e}")
        return 1
    
    if start_date > end_date:
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
            logger.error("Crypto database schema does not exist. Run init_crypto_db.py first.")
            return 1
        
        assets_str = ", ".join(args.assets) if args.assets else "all available"
        logger.info(f"Starting crypto ingestion: {start_date} to {end_date}")
        logger.info(f"Assets: {assets_str}")
        if args.force_refresh:
            logger.info("Force refresh enabled - will re-ingest existing data")
        
        # Run ingestion
        result = ingest_crypto_data(
            client=client,
            start_date=start_date,
            end_date=end_date,
            assets=args.assets,
            chunk_days=args.chunk_days,
            force_refresh=args.force_refresh,
        )
        
        # Report results
        logger.info("=" * 60)
        logger.info("Crypto Ingestion Summary:")
        logger.info(f"  Status: {result.status}")
        logger.info(f"  Assets: {result.assets_count}")
        logger.info(f"  Rows inserted: {result.rows_inserted:,}")
        logger.info(f"  Duration: {result.duration_seconds:.1f} seconds")
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
