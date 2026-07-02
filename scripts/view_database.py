#!/usr/bin/env python
"""Quick database viewer - view your NDX and Crypto data.

Usage:
    python scripts/view_database.py
    python scripts/view_database.py --table ndx_universe
    python scripts/view_database.py --table ndx_ohlcv --limit 100
"""

from __future__ import annotations

import argparse
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

from q23.database.client import get_client
from q23.shared.config import cfg
import pandas as pd
from sqlalchemy import create_engine

def get_sqlalchemy_engine():
    """Create SQLAlchemy engine for pandas."""
    # SQLAlchemy uses 'postgresql+psycopg2://' prefix
    connection_string = cfg.database.connection_string
    if connection_string.startswith("postgresql://"):
        connection_string = connection_string.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(connection_string)

def view_table(client, table_name: str, limit: int = 50):
    """View a database table."""
    engine = get_sqlalchemy_engine()
    query = f"SELECT * FROM {table_name} LIMIT {limit}"
    df = pd.read_sql_query(query, engine)
        print(f"\n{'='*80}")
        print(f"Table: {table_name}")
        print(f"Rows: {len(df)} (showing first {limit})")
        print(f"{'='*80}\n")
        print(df.to_string())
        print(f"\n{'='*80}\n")

def view_summary(client):
    """View database summary statistics."""
    engine = get_sqlalchemy_engine()
    print("\n" + "="*80)
    print("DATABASE SUMMARY")
    print("="*80 + "\n")
    
    # NDX Universe
    df = pd.read_sql_query("SELECT COUNT(*) as count FROM ndx_universe", engine)
        print(f"NDX Universe: {df.iloc[0]['count']} assets")
        
    # NDX OHLCV
    df = pd.read_sql_query("""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT asset_id) as assets,
            MIN(date) as earliest_date,
            MAX(date) as latest_date
        FROM ndx_ohlcv
    """, engine)
        if df.iloc[0]['total_rows'] > 0:
            print(f"NDX OHLCV: {df.iloc[0]['total_rows']:,} rows")
            print(f"  Assets: {df.iloc[0]['assets']}")
            print(f"  Date range: {df.iloc[0]['earliest_date']} to {df.iloc[0]['latest_date']}")
        else:
            print("NDX OHLCV: No data yet")
        
    # Crypto Universe
    try:
        df = pd.read_sql_query("SELECT COUNT(*) as count FROM crypto_universe", engine)
        print(f"\nCrypto Universe: {df.iloc[0]['count']} assets")
    except:
        print("\nCrypto Universe: Schema not initialized")
    
    # Crypto OHLCV
    try:
        df = pd.read_sql_query("""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT asset_id) as assets,
                MIN(date) as earliest_date,
                MAX(date) as latest_date
            FROM crypto_ohlcv
        """, engine)
            if df.iloc[0]['total_rows'] > 0:
                print(f"Crypto OHLCV: {df.iloc[0]['total_rows']:,} rows")
                print(f"  Assets: {df.iloc[0]['assets']}")
                print(f"  Date range: {df.iloc[0]['earliest_date']} to {df.iloc[0]['latest_date']}")
            else:
                print("Crypto OHLCV: No data yet")
        except:
            print("Crypto OHLCV: Schema not initialized")
        
    print("\n" + "="*80 + "\n")

def view_sample_data(client, table_name: str = "ndx_ohlcv", limit: int = 10):
    """View sample data from a table."""
    engine = get_sqlalchemy_engine()
    if table_name == "ndx_ohlcv":
        query = """
            SELECT o.date, o.asset_id, u.symbol, o.open, o.high, o.low, o.close, o.volume
            FROM ndx_ohlcv o
            JOIN ndx_universe u ON o.asset_id = u.asset_id
            ORDER BY o.date DESC, u.symbol
            LIMIT :limit
        """
    elif table_name == "ndx_universe":
        query = f"SELECT * FROM {table_name} ORDER BY symbol LIMIT :limit"
    elif table_name == "crypto_ohlcv":
        query = f"SELECT * FROM {table_name} ORDER BY date DESC LIMIT :limit"
    elif table_name == "crypto_universe":
        query = f"SELECT * FROM {table_name} ORDER BY symbol LIMIT :limit"
    else:
        query = f"SELECT * FROM {table_name} LIMIT :limit"
    
    df = pd.read_sql_query(query, engine, params={"limit": limit})
    print(f"\n{'='*80}")
    print(f"Sample Data: {table_name}")
    print(f"{'='*80}\n")
    print(df.to_string())
    print(f"\n{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(description="View database data")
    parser.add_argument("--table", help="Table name to view (ndx_universe, ndx_ohlcv, crypto_universe, crypto_ohlcv)")
    parser.add_argument("--limit", type=int, default=50, help="Number of rows to show (default: 50)")
    parser.add_argument("--summary", action="store_true", help="Show database summary")
    args = parser.parse_args()
    
    client = get_client()
    
    if args.summary:
        view_summary(client)
    elif args.table:
        view_sample_data(client, args.table, args.limit)
    else:
        # Default: show summary and sample data
        view_summary(client)
        print("\nShowing sample NDX OHLCV data...")
        view_sample_data(client, "ndx_ohlcv", 10)

if __name__ == "__main__":
    main()
