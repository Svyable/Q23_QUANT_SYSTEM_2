"""q23.database.crypto_schema

Database schema definitions for cryptocurrency market data storage.
"""

from __future__ import annotations

from typing import Optional
import logging

logger = logging.getLogger(__name__)


# SQL DDL for crypto tables
CRYPTO_SCHEMA_DDL = """
-- Crypto Universe table: Cryptocurrency metadata
-- Note: Quantiacs cryptodaily doesn't provide universe list, so we derive from data
CREATE TABLE IF NOT EXISTS crypto_universe (
    asset_id VARCHAR(20) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    name VARCHAR(200),
    first_seen_date DATE,
    last_seen_date DATE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Crypto OHLCV table: Daily price and volume data
CREATE TABLE IF NOT EXISTS crypto_ohlcv (
    date DATE NOT NULL,
    asset_id VARCHAR(20) NOT NULL,
    open DECIMAL(18,8),
    high DECIMAL(18,8),
    low DECIMAL(18,8),
    close DECIMAL(18,8),
    volume DECIMAL(20,8),
    is_liquid BOOLEAN,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, asset_id),
    FOREIGN KEY (asset_id) REFERENCES crypto_universe(asset_id) ON DELETE CASCADE
);

-- Crypto Ingestion Log table: Track data ingestion runs
CREATE TABLE IF NOT EXISTS crypto_ingestion_log (
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

-- Blockchain Metrics Metadata table: Available blockchain metrics
CREATE TABLE IF NOT EXISTS crypto_blockchain_metrics (
    metric_id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(200),
    description TEXT,
    unit VARCHAR(50),
    source VARCHAR(50) DEFAULT 'blockchain.com',
    first_seen_date DATE,
    last_seen_date DATE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Blockchain Metrics Data table: Time-series blockchain fundamental data
CREATE TABLE IF NOT EXISTS crypto_blockchain_data (
    date DATE NOT NULL,
    metric_id VARCHAR(100) NOT NULL,
    value DECIMAL(20,8),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, metric_id),
    FOREIGN KEY (metric_id) REFERENCES crypto_blockchain_metrics(metric_id) ON DELETE CASCADE
);

-- Blockchain Ingestion Log table: Track blockchain data ingestion runs
CREATE TABLE IF NOT EXISTS crypto_blockchain_ingestion_log (
    id SERIAL PRIMARY KEY,
    metric_id VARCHAR(100),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    rows_inserted BIGINT NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('success', 'partial', 'failed')),
    error_message TEXT,
    duration_seconds DECIMAL(10,2),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_crypto_ohlcv_date ON crypto_ohlcv(date);
CREATE INDEX IF NOT EXISTS idx_crypto_ohlcv_asset_date ON crypto_ohlcv(asset_id, date);
CREATE INDEX IF NOT EXISTS idx_crypto_ohlcv_date_asset ON crypto_ohlcv(date, asset_id);
CREATE INDEX IF NOT EXISTS idx_crypto_universe_symbol ON crypto_universe(symbol);
CREATE INDEX IF NOT EXISTS idx_crypto_universe_active ON crypto_universe(is_active);
CREATE INDEX IF NOT EXISTS idx_crypto_ingestion_log_dates ON crypto_ingestion_log(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_crypto_blockchain_data_date ON crypto_blockchain_data(date);
CREATE INDEX IF NOT EXISTS idx_crypto_blockchain_data_metric_date ON crypto_blockchain_data(metric_id, date);
CREATE INDEX IF NOT EXISTS idx_crypto_blockchain_data_date_metric ON crypto_blockchain_data(date, metric_id);
CREATE INDEX IF NOT EXISTS idx_crypto_blockchain_metrics_active ON crypto_blockchain_metrics(is_active);
"""

DROP_CRYPTO_SCHEMA_DDL = """
DROP TABLE IF EXISTS crypto_blockchain_data CASCADE;
DROP TABLE IF EXISTS crypto_blockchain_ingestion_log CASCADE;
DROP TABLE IF EXISTS crypto_blockchain_metrics CASCADE;
DROP TABLE IF EXISTS crypto_ohlcv CASCADE;
DROP TABLE IF EXISTS crypto_universe CASCADE;
DROP TABLE IF EXISTS crypto_ingestion_log CASCADE;
"""


def create_crypto_schema(client: "DatabaseClient") -> bool:
    """Create all crypto database tables and indexes.
    
    Args:
        client: Database client instance
        
    Returns:
        True if schema was created successfully, False otherwise
    """
    try:
        with client.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CRYPTO_SCHEMA_DDL)
                conn.commit()
        logger.info("Crypto database schema created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create crypto database schema: {e}")
        return False


def drop_crypto_schema(client: "DatabaseClient") -> bool:
    """Drop all crypto database tables (destructive operation).
    
    Args:
        client: Database client instance
        
    Returns:
        True if schema was dropped successfully, False otherwise
    """
    try:
        with client.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(DROP_CRYPTO_SCHEMA_DDL)
                conn.commit()
        logger.info("Crypto database schema dropped successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to drop crypto database schema: {e}")
        return False


def crypto_schema_exists(client: "DatabaseClient") -> bool:
    """Check if crypto database schema exists.
    
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
                        AND table_name = 'crypto_universe'
                    );
                """)
                result = cur.fetchone()
                return result[0] if result else False
    except Exception as e:
        logger.error(f"Failed to check crypto schema existence: {e}")
        return False


def blockchain_schema_exists(client: "DatabaseClient") -> bool:
    """Check if blockchain metrics schema exists.
    
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
                        AND table_name = 'crypto_blockchain_metrics'
                    );
                """)
                result = cur.fetchone()
                return result[0] if result else False
    except Exception as e:
        logger.error(f"Failed to check blockchain schema existence: {e}")
        return False


def create_blockchain_schema(client: "DatabaseClient") -> bool:
    """Create blockchain metrics tables and indexes.
    
    Args:
        client: Database client instance
        
    Returns:
        True if schema was created successfully, False otherwise
    """
    blockchain_ddl = """
-- Blockchain Metrics Metadata table: Available blockchain metrics
CREATE TABLE IF NOT EXISTS crypto_blockchain_metrics (
    metric_id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(200),
    description TEXT,
    unit VARCHAR(50),
    source VARCHAR(50) DEFAULT 'blockchain.com',
    first_seen_date DATE,
    last_seen_date DATE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Blockchain Metrics Data table: Time-series blockchain fundamental data
CREATE TABLE IF NOT EXISTS crypto_blockchain_data (
    date DATE NOT NULL,
    metric_id VARCHAR(100) NOT NULL,
    value DECIMAL(20,8),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, metric_id),
    FOREIGN KEY (metric_id) REFERENCES crypto_blockchain_metrics(metric_id) ON DELETE CASCADE
);

-- Blockchain Ingestion Log table: Track blockchain data ingestion runs
CREATE TABLE IF NOT EXISTS crypto_blockchain_ingestion_log (
    id SERIAL PRIMARY KEY,
    metric_id VARCHAR(100),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    rows_inserted BIGINT NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('success', 'partial', 'failed')),
    error_message TEXT,
    duration_seconds DECIMAL(10,2),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_crypto_blockchain_data_date ON crypto_blockchain_data(date);
CREATE INDEX IF NOT EXISTS idx_crypto_blockchain_data_metric_date ON crypto_blockchain_data(metric_id, date);
CREATE INDEX IF NOT EXISTS idx_crypto_blockchain_data_date_metric ON crypto_blockchain_data(date, metric_id);
CREATE INDEX IF NOT EXISTS idx_crypto_blockchain_metrics_active ON crypto_blockchain_metrics(is_active);
"""
    try:
        with client.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(blockchain_ddl)
                conn.commit()
        logger.info("Blockchain database schema created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create blockchain database schema: {e}")
        return False
