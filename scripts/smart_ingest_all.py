#!/usr/bin/env python
"""Smart data ingestion for both NDX and Cryptocurrency data.

This master script intelligently checks what data already exists and only
ingests missing date ranges for both NDX stocks and cryptocurrencies.
Safe to run repeatedly - it will only pull new data.

Usage:
    python scripts/smart_ingest_all.py
    python scripts/smart_ingest_all.py --skip-ndx
    python scripts/smart_ingest_all.py --skip-crypto
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
from q23.database.smart_ingestion import (
    smart_ingest_ndx,
    smart_ingest_crypto,
    get_ndx_date_gaps,
    get_crypto_date_gaps,
)
from q23.database.schema import schema_exists
from q23.database.crypto_schema import crypto_schema_exists, blockchain_schema_exists, create_blockchain_schema
from q23.database.blockchain_ingestion import ingest_blockchain_metrics_list, fetch_blockchain_metrics_list, ingest_blockchain_data
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
        description="Smart data ingestion for NDX and Cryptocurrency - only pulls missing dates"
    )
    parser.add_argument(
        "--ndx-start",
        type=str,
        default="2015-01-01",
        help="NDX target start date (YYYY-MM-DD, default: 2015-01-01)"
    )
    parser.add_argument(
        "--ndx-end",
        type=str,
        default=None,
        help="NDX target end date (YYYY-MM-DD, default: today)"
    )
    parser.add_argument(
        "--crypto-start",
        type=str,
        default="2014-01-01",
        help="Crypto target start date (YYYY-MM-DD, default: 2014-01-01)"
    )
    parser.add_argument(
        "--crypto-end",
        type=str,
        default=None,
        help="Crypto target end date (YYYY-MM-DD, default: today)"
    )
    parser.add_argument(
        "--crypto-assets",
        type=str,
        nargs="*",
        default=None,
        help="List of crypto asset symbols (e.g., BTC ETH). Default: all available"
    )
    parser.add_argument(
        "--skip-ndx",
        action="store_true",
        help="Skip NDX data ingestion"
    )
    parser.add_argument(
        "--skip-crypto",
        action="store_true",
        help="Skip cryptocurrency data ingestion"
    )
    parser.add_argument(
        "--skip-blockchain",
        action="store_true",
        help="Skip blockchain metrics data ingestion"
    )
    parser.add_argument(
        "--blockchain-metrics",
        type=str,
        nargs="*",
        default=None,
        help="List of blockchain metric IDs to ingest (e.g., miners-revenue). Default: all available"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh of all data (ignore existing data)"
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
        ndx_start = parse_date(args.ndx_start) if not args.skip_ndx else None
        if args.ndx_end:
            ndx_end = parse_date(args.ndx_end)
        else:
            ndx_end = date.today() if not args.skip_ndx else None
        
        crypto_start = parse_date(args.crypto_start) if not args.skip_crypto else None
        if args.crypto_end:
            crypto_end = parse_date(args.crypto_end)
        else:
            crypto_end = date.today() if not args.skip_crypto else None
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
        
        overall_success = True
        
        # ========== NDX Ingestion ==========
        if not args.skip_ndx:
            logger.info("")
            logger.info("=" * 60)
            logger.info("NDX DATA INGESTION")
            logger.info("=" * 60)
            
            # Check schema
            if not schema_exists(client):
                logger.warning("NDX schema does not exist. Skipping NDX ingestion.")
                logger.info("Run: python scripts/init_ndx_db.py")
                overall_success = False
            else:
                # Check for gaps
                if not args.force_refresh:
                    gaps = get_ndx_date_gaps(client, ndx_start, ndx_end)
                    if gaps:
                        logger.info(f"Found {len(gaps)} date gap(s) in NDX data:")
                        for gap_start, gap_end in gaps:
                            logger.info(f"  {gap_start} to {gap_end}")
                    else:
                        logger.info(f"NDX data is already complete for {ndx_start} to {ndx_end}")
                
                # Run ingestion
                try:
                    result = smart_ingest_ndx(
                        client=client,
                        target_start=ndx_start,
                        target_end=ndx_end,
                        force_refresh=args.force_refresh,
                    )
                    
                    logger.info(f"NDX Status: {result.status}")
                    logger.info(f"NDX Rows inserted: {result.rows_inserted:,}")
                    if result.status != "success":
                        overall_success = False
                except Exception as e:
                    logger.error(f"NDX ingestion failed: {e}")
                    overall_success = False
        else:
            logger.info("Skipping NDX ingestion (--skip-ndx)")
        
        # ========== Crypto Ingestion ==========
        if not args.skip_crypto:
            logger.info("")
            logger.info("=" * 60)
            logger.info("CRYPTOCURRENCY DATA INGESTION")
            logger.info("=" * 60)
            
            # Check schema
            if not crypto_schema_exists(client):
                logger.warning("Crypto schema does not exist. Skipping crypto ingestion.")
                logger.info("Run: python scripts/init_crypto_db.py")
                overall_success = False
            else:
                # Check for gaps
                if not args.force_refresh:
                    gaps = get_crypto_date_gaps(
                        client, crypto_start, crypto_end,
                        args.crypto_assets[0] if args.crypto_assets else None
                    )
                    if gaps:
                        logger.info(f"Found {len(gaps)} date gap(s) in crypto data:")
                        for gap_start, gap_end in gaps:
                            logger.info(f"  {gap_start} to {gap_end}")
                    else:
                        logger.info(f"Crypto data is already complete for {crypto_start} to {crypto_end}")
                
                # Run ingestion
                try:
                    assets_str = ", ".join(args.crypto_assets) if args.crypto_assets else "all available"
                    logger.info(f"Assets: {assets_str}")
                    
                    result = smart_ingest_crypto(
                        client=client,
                        target_start=crypto_start,
                        target_end=crypto_end,
                        assets=args.crypto_assets,
                        force_refresh=args.force_refresh,
                    )
                    
                    logger.info(f"Crypto Status: {result.status}")
                    logger.info(f"Crypto Rows inserted: {result.rows_inserted:,}")
                    if result.status != "success":
                        overall_success = False
                except Exception as e:
                    logger.error(f"Crypto ingestion failed: {e}")
                    overall_success = False
        else:
            logger.info("Skipping crypto ingestion (--skip-crypto)")
        
        # ========== Blockchain Metrics Ingestion ==========
        if not args.skip_blockchain:
            logger.info("")
            logger.info("=" * 60)
            logger.info("BLOCKCHAIN METRICS DATA INGESTION")
            logger.info("=" * 60)
            
            # Check schema
            if not blockchain_schema_exists(client):
                logger.warning("Blockchain schema does not exist. Creating...")
                if not create_blockchain_schema(client):
                    logger.error("Failed to create blockchain schema. Skipping blockchain ingestion.")
                    overall_success = False
                else:
                    logger.info("Blockchain schema created successfully")
            
            if blockchain_schema_exists(client):
                try:
                    # Update metrics list first
                    logger.info("Updating blockchain metrics list...")
                    try:
                        count = ingest_blockchain_metrics_list(client)
                        logger.info(f"✓ Updated {count} blockchain metrics")
                    except Exception as e:
                        logger.warning(f"Failed to update metrics list: {e}")
                    
                    # Determine which metrics to ingest
                    if args.blockchain_metrics:
                        metric_ids = args.blockchain_metrics
                        logger.info(f"Ingesting {len(metric_ids)} specified metrics")
                    else:
                        # Fetch all available metrics
                        logger.info("Fetching list of available metrics...")
                        metrics_df = fetch_blockchain_metrics_list()
                        metric_ids = metrics_df['id'].astype(str).tolist()
                        logger.info(f"Found {len(metric_ids)} metrics to ingest")
                    
                    # Ingest each metric
                    total_rows = 0
                    success_count = 0
                    failed_count = 0
                    
                    for metric_id in metric_ids:
                        try:
                            logger.info(f"\nIngesting metric: {metric_id}")
                            
                            result = ingest_blockchain_data(
                                client,
                                metric_id,
                                start_date=None,  # Let it fetch all available
                                end_date=date.today(),
                                force_refresh=args.force_refresh,
                            )
                            
                            if result.status == "success":
                                success_count += 1
                                total_rows += result.rows_inserted
                                logger.info(f"  ✓ {result.rows_inserted} rows inserted")
                            else:
                                failed_count += 1
                                logger.warning(f"  ✗ Failed: {result.error_message}")
                                
                        except Exception as e:
                            failed_count += 1
                            logger.error(f"  ✗ Error: {e}")
                    
                    logger.info(f"\nBlockchain Summary:")
                    logger.info(f"  Total metrics: {len(metric_ids)}")
                    logger.info(f"  Successful: {success_count}")
                    logger.info(f"  Failed: {failed_count}")
                    logger.info(f"  Total rows inserted: {total_rows:,}")
                    
                    if failed_count > 0:
                        overall_success = False
                        
                except Exception as e:
                    logger.error(f"Blockchain ingestion failed: {e}")
                    overall_success = False
        else:
            logger.info("Skipping blockchain ingestion (--skip-blockchain)")
        
        # ========== Summary ==========
        logger.info("")
        logger.info("=" * 60)
        logger.info("OVERALL SUMMARY")
        logger.info("=" * 60)
        if overall_success:
            logger.info("All ingestion tasks completed successfully!")
            return 0
        else:
            logger.warning("Some ingestion tasks had errors")
            return 1
            
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
