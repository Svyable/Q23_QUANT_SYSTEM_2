"""q23.database.schema

Database schema definitions for NDX market data storage.
"""

from __future__ import annotations

from typing import Optional
import logging

logger = logging.getLogger(__name__)


# SQL DDL for all tables
SCHEMA_DDL = """
-- NDX Universe table: Stock metadata and membership tracking
-- Aligned with Quantiacs load_ndx_list() output: name, sector, symbol, exchange, id, FIGI
CREATE TABLE IF NOT EXISTS ndx_universe (
    asset_id VARCHAR(50) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    name VARCHAR(200),
    sector VARCHAR(100),
    exchange VARCHAR(10),
    figi VARCHAR(20),
    first_seen_date DATE,
    last_seen_date DATE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- NDX OHLCV table: Daily price and volume data
-- Aligned with Quantiacs load_ndx_data() fields: open, high, low, close, vol, divs, is_liquid, is_stock, is_spx, is_ndx
CREATE TABLE IF NOT EXISTS ndx_ohlcv (
    date DATE NOT NULL,
    asset_id VARCHAR(50) NOT NULL,
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    volume BIGINT,
    dividend DECIMAL(10,4),
    is_liquid BOOLEAN,
    is_stock BOOLEAN,
    is_spx BOOLEAN,
    is_ndx BOOLEAN,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, asset_id),
    FOREIGN KEY (asset_id) REFERENCES ndx_universe(asset_id) ON DELETE CASCADE
);

-- NDX Ingestion Log table: Track data ingestion runs
CREATE TABLE IF NOT EXISTS ndx_ingestion_log (
    id SERIAL PRIMARY KEY,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    assets_count INTEGER NOT NULL,
    rows_inserted BIGINT NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('success', 'partial', 'failed')),
    error_message TEXT,
    duration_seconds DECIMAL(10,2),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_ndx_ohlcv_date ON ndx_ohlcv(date);
CREATE INDEX IF NOT EXISTS idx_ndx_ohlcv_asset_date ON ndx_ohlcv(asset_id, date);
CREATE INDEX IF NOT EXISTS idx_ndx_ohlcv_date_asset ON ndx_ohlcv(date, asset_id);
CREATE INDEX IF NOT EXISTS idx_ndx_universe_symbol ON ndx_universe(symbol);
CREATE INDEX IF NOT EXISTS idx_ndx_universe_active ON ndx_universe(is_active);
CREATE INDEX IF NOT EXISTS idx_ndx_ingestion_log_dates ON ndx_ingestion_log(start_date, end_date);
"""

DROP_SCHEMA_DDL = """
DROP TABLE IF EXISTS ndx_ohlcv CASCADE;
DROP TABLE IF EXISTS ndx_universe CASCADE;
DROP TABLE IF EXISTS ndx_ingestion_log CASCADE;
"""


def create_schema(client: "DatabaseClient") -> bool:
    """Create all database tables and indexes.
    
    Args:
        client: Database client instance
        
    Returns:
        True if schema was created successfully, False otherwise
    """
    try:
        with client.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_DDL)
                conn.commit()
        logger.info("Database schema created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create database schema: {e}")
        return False


def drop_schema(client: "DatabaseClient") -> bool:
    """Drop all database tables (destructive operation).
    
    Args:
        client: Database client instance
        
    Returns:
        True if schema was dropped successfully, False otherwise
    """
    try:
        with client.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(DROP_SCHEMA_DDL)
                conn.commit()
        logger.info("Database schema dropped successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to drop database schema: {e}")
        return False


def schema_exists(client: "DatabaseClient") -> bool:
    """Check if database schema exists.
    
    Args:
        client: Database client instance
        
    Returns:
        True if schema exists, False otherwise
    """
    try:
        with client.get_connection() as conn:
            with conn.cursor() as cur:
                # Check if main table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'ndx_universe'
                    );
                """)
                result = cur.fetchone()
                return result[0] if result else False
    except Exception as e:
        logger.error(f"Failed to check schema existence: {e}")
        return False
