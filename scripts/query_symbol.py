#!/usr/bin/env python
"""Quick query for a single symbol - prints OHLCV data.

Usage:
    python scripts/query_symbol.py AAPL
    python scripts/query_symbol.py BTC --crypto
    python scripts/query_symbol.py AAPL --limit 30
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
from sqlalchemy import create_engine, text
import pandas as pd

def get_sqlalchemy_engine():
    """Create SQLAlchemy engine for pandas."""
    connection_string = cfg.database.connection_string
    if connection_string.startswith("postgresql://"):
        connection_string = connection_string.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(connection_string)

def query_ndx_symbol(symbol: str, limit: int = 50):
    """Query NDX data for a symbol."""
    engine = get_sqlalchemy_engine()
    
    query = text("""
        SELECT 
            o.date,
            u.symbol,
            u.name,
            o.open,
            o.high,
            o.low,
            o.close,
            o.volume,
            o.dividend,
            o.is_liquid
        FROM ndx_ohlcv o
        JOIN ndx_universe u ON o.asset_id = u.asset_id
        WHERE UPPER(u.symbol) = UPPER(:symbol)
        ORDER BY o.date DESC
        LIMIT :limit
    """)
    
    df = pd.read_sql_query(query, engine, params={"symbol": symbol, "limit": limit})
    
    if df.empty:
        print(f"\n❌ No data found for symbol: {symbol}")
        print("\nAvailable symbols (sample):")
        check_query = text("SELECT DISTINCT symbol FROM ndx_universe ORDER BY symbol LIMIT 20")
        available = pd.read_sql_query(check_query, engine)
        print(available['symbol'].tolist())
        return
    
    print(f"\n{'='*100}")
    print(f"📊 {symbol.upper()} - {df.iloc[0]['name']}")
    print(f"   Showing {len(df)} most recent rows (out of {limit} requested)")
    print(f"{'='*100}\n")
    
    # Format for display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    
    # Reorder columns for better display
    display_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'dividend', 'is_liquid']
    display_df = df[display_cols].copy()
    
    # Format numbers
    display_df['open'] = display_df['open'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    display_df['high'] = display_df['high'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    display_df['low'] = display_df['low'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    display_df['close'] = display_df['close'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    display_df['volume'] = display_df['volume'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
    display_df['dividend'] = display_df['dividend'].apply(lambda x: f"{x:.4f}" if pd.notna(x) and x > 0 else "0.0000")
    
    print(display_df.to_string(index=False))
    print(f"\n{'='*100}\n")

def query_crypto_symbol(symbol: str, limit: int = 50):
    """Query crypto data for a symbol."""
    engine = get_sqlalchemy_engine()
    
    query = text("""
        SELECT 
            o.date,
            u.symbol,
            u.name,
            o.open,
            o.high,
            o.low,
            o.close,
            o.volume,
            o.is_liquid
        FROM crypto_ohlcv o
        JOIN crypto_universe u ON o.asset_id = u.asset_id
        WHERE UPPER(u.symbol) = UPPER(:symbol)
        ORDER BY o.date DESC
        LIMIT :limit
    """)
    
    df = pd.read_sql_query(query, engine, params={"symbol": symbol, "limit": limit})
    
    if df.empty:
        print(f"\n❌ No data found for crypto symbol: {symbol}")
        print("\nAvailable crypto symbols (sample):")
        check_query = text("SELECT DISTINCT symbol FROM crypto_universe ORDER BY symbol LIMIT 20")
        available = pd.read_sql_query(check_query, engine)
        print(available['symbol'].tolist())
        return
    
    print(f"\n{'='*100}")
    print(f"₿ {symbol.upper()} - {df.iloc[0]['name']}")
    print(f"   Showing {len(df)} most recent rows (out of {limit} requested)")
    print(f"{'='*100}\n")
    
    # Format for display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    
    # Reorder columns for better display
    display_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'is_liquid']
    display_df = df[display_cols].copy()
    
    # Format numbers (crypto prices are typically higher precision)
    display_df['open'] = display_df['open'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    display_df['high'] = display_df['high'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    display_df['low'] = display_df['low'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    display_df['close'] = display_df['close'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    display_df['volume'] = display_df['volume'].apply(lambda x: f"{float(x):,.2f}" if pd.notna(x) else "N/A")
    
    print(display_df.to_string(index=False))
    print(f"\n{'='*100}\n")

def main():
    parser = argparse.ArgumentParser(description="Query single symbol data")
    parser.add_argument("symbol", help="Symbol to query (e.g., AAPL, BTC)")
    parser.add_argument("--crypto", action="store_true", help="Query cryptocurrency data")
    parser.add_argument("--limit", type=int, default=50, help="Number of rows to show (default: 50)")
    args = parser.parse_args()
    
    if args.crypto:
        query_crypto_symbol(args.symbol, args.limit)
    else:
        query_ndx_symbol(args.symbol, args.limit)

if __name__ == "__main__":
    main()
