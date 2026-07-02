#!/usr/bin/env python3
"""Ingest blockchain metrics data from Quantiacs into database.

This script fetches and stores blockchain fundamental data (e.g., miners revenue,
hash rate, transaction volume, etc.) from Quantiacs.

Usage:
    # Ingest all available metrics
    python scripts/ingest_blockchain.py --all
    
    # Ingest specific metric
    python scripts/ingest_blockchain.py --metric miners-revenue
    
    # Ingest with date range
    python scripts/ingest_blockchain.py --metric miners-revenue --start-date 2015-01-01 --end-date 2025-01-24
    
    # Update metrics list first
    python scripts/ingest_blockchain.py --update-metrics-list
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import date, datetime

# Load .env file BEFORE any imports
def _load_env():
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
from q23.database.blockchain_ingestion import (
    ingest_blockchain_metrics_list,
    ingest_blockchain_data,
    fetch_blockchain_metrics_list,
)
from q23.database.crypto_schema import blockchain_schema_exists, create_blockchain_schema
from q23.shared.config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest blockchain metrics data from Quantiacs"
    )
    parser.add_argument(
        "--update-metrics-list",
        action="store_true",
        help="Update the list of available blockchain metrics"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all available blockchain metrics"
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="Specific metric ID to ingest (e.g., 'miners-revenue')"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Default: earliest available"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD). Default: today"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-ingest existing data"
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
        
        logger.info("Database connection successful")
        
        # Ensure blockchain schema exists
        if not blockchain_schema_exists(client):
            logger.info("Blockchain schema does not exist. Creating...")
            if not create_blockchain_schema(client):
                logger.error("Failed to create blockchain schema")
                return 1
        
        # Parse dates
        start_date = None
        if args.start_date:
            try:
                start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
            except ValueError:
                logger.error(f"Invalid start date format: {args.start_date}. Use YYYY-MM-DD")
                return 1
        
        end_date = None
        if args.end_date:
            try:
                end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
            except ValueError:
                logger.error(f"Invalid end date format: {args.end_date}. Use YYYY-MM-DD")
                return 1
        
        # Update metrics list if requested
        if args.update_metrics_list:
            logger.info("Updating blockchain metrics list...")
            try:
                count = ingest_blockchain_metrics_list(client)
                logger.info(f"✓ Updated {count} blockchain metrics")
            except Exception as e:
                logger.error(f"Failed to update metrics list: {e}")
                return 1
        
        # Ingest data
        if args.all:
            # Fetch metrics list first if not already in DB
            logger.info("Fetching list of available metrics...")
            try:
                metrics_df = fetch_blockchain_metrics_list()
                metric_ids = metrics_df['id'].astype(str).tolist()
                logger.info(f"Found {len(metric_ids)} metrics to ingest")
            except Exception as e:
                logger.error(f"Failed to fetch metrics list: {e}")
                return 1
            
            # Ingest each metric
            total_rows = 0
            success_count = 0
            failed_count = 0
            
            for metric_id in metric_ids:
                try:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Ingesting metric: {metric_id}")
                    logger.info(f"{'='*60}")
                    
                    result = ingest_blockchain_data(
                        client,
                        metric_id,
                        start_date=start_date,
                        end_date=end_date,
                        force_refresh=args.force_refresh,
                    )
                    
                    if result.status == "success":
                        success_count += 1
                        total_rows += result.rows_inserted
                        logger.info(f"✓ Success: {result.rows_inserted} rows inserted")
                    else:
                        failed_count += 1
                        logger.warning(f"✗ Failed: {result.error_message}")
                        
                except Exception as e:
                    failed_count += 1
                    logger.error(f"✗ Error ingesting {metric_id}: {e}")
            
            logger.info(f"\n{'='*60}")
            logger.info("SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Total metrics: {len(metric_ids)}")
            logger.info(f"Successful: {success_count}")
            logger.info(f"Failed: {failed_count}")
            logger.info(f"Total rows inserted: {total_rows}")
            
        elif args.metric:
            logger.info(f"Ingesting metric: {args.metric}")
            try:
                result = ingest_blockchain_data(
                    client,
                    args.metric,
                    start_date=start_date,
                    end_date=end_date,
                    force_refresh=args.force_refresh,
                )
                
                if result.status == "success":
                    logger.info(f"✓ Success: {result.rows_inserted} rows inserted in {result.duration_seconds:.1f}s")
                    return 0
                else:
                    logger.error(f"✗ Failed: {result.error_message}")
                    return 1
                    
            except Exception as e:
                logger.error(f"Error ingesting {args.metric}: {e}")
                import traceback
                traceback.print_exc()
                return 1
        else:
            logger.error("Must specify --all, --metric, or --update-metrics-list")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
