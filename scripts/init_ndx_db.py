#!/usr/bin/env python
"""Initialize NDX database schema.

This script creates the database schema (tables and indexes) for NDX market data storage.

Usage:
    python scripts/init_ndx_db.py [--drop-existing]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
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
from q23.database.schema import create_schema, drop_schema, schema_exists
from q23.shared.config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Initialize NDX database schema"
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing schema before creating (destructive!)"
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
        
        # Check if schema exists
        if schema_exists(client):
            if args.drop_existing:
                logger.warning("Dropping existing schema...")
                if not drop_schema(client):
                    logger.error("Failed to drop schema")
                    return 1
                logger.info("Schema dropped")
            else:
                logger.warning("Schema already exists. Use --drop-existing to recreate.")
                return 0
        
        # Create schema
        logger.info("Creating database schema...")
        if create_schema(client):
            logger.info("Schema created successfully!")
            logger.info("Tables created: ndx_universe, ndx_ohlcv, ndx_ingestion_log")
            return 0
        else:
            logger.error("Failed to create schema")
            return 1
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
