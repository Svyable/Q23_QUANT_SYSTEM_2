"""q23.database.client

Database client with connection pooling and query utilities.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Tuple
from datetime import date, datetime

try:
    import psycopg2
    from psycopg2 import pool, sql
    from psycopg2.extras import RealDictCursor, execute_batch
    import psycopg2.pool
    HAS_PSYCOPG2 = True
except ImportError:
    psycopg2 = None
    pool = None
    sql = None
    RealDictCursor = None
    execute_batch = None
    HAS_PSYCOPG2 = False

from q23.shared.config import cfg

logger = logging.getLogger(__name__)


def _require_psycopg2() -> None:
    """Check that psycopg2 is available."""
    if psycopg2 is None:
        raise ImportError(
            "psycopg2 is required for database operations. "
            "Install with: pip install psycopg2-binary"
        )


class DatabaseClient:
    """PostgreSQL database client with connection pooling."""
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        dbname: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        pool_size: int = 5,
        max_overflow: int = 10,
    ):
        """Initialize database client.
        
        Args:
            host: Database host (defaults to cfg.database.HOST)
            port: Database port (defaults to cfg.database.PORT)
            dbname: Database name (defaults to cfg.database.NAME)
            user: Database user (defaults to cfg.database.USER)
            password: Database password (defaults to cfg.database.PASSWORD)
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
        """
        _require_psycopg2()
        
        self.host = host or cfg.database.HOST
        self.port = port or cfg.database.PORT
        self.dbname = dbname or cfg.database.NAME
        self.user = user or cfg.database.USER
        self.password = password or cfg.database.PASSWORD
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        
        self._connection_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
    
    def _create_pool(self) -> None:
        """Create connection pool with retry logic."""
        if self._connection_pool is None:
            max_retries = 3
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    self._connection_pool = psycopg2.pool.ThreadedConnectionPool(
                        minconn=1,
                        maxconn=self.pool_size + self.max_overflow,
                        host=self.host,
                        port=self.port,
                        dbname=self.dbname,
                        user=self.user,
                        password=self.password,
                        connect_timeout=10,  # 10 second timeout
                    )
                    logger.info(f"Created connection pool for {self.dbname}@{self.host}:{self.port}")
                    return
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Connection pool creation failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                        import time
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    else:
                        error_msg = str(e)
                        if "Connection refused" in error_msg or "could not connect" in error_msg.lower():
                            logger.error(
                                f"Failed to connect to PostgreSQL after {max_retries} attempts.\n"
                                f"Error: {error_msg}\n"
                                f"Troubleshooting:\n"
                                f"  1. Is PostgreSQL installed? Install with: brew install postgresql@16\n"
                                f"  2. Is PostgreSQL running? Start with: brew services start postgresql@16\n"
                                f"  3. Or use Postgres.app: https://postgresapp.com/\n"
                                f"  4. Check connection settings: host={self.host}, port={self.port}, dbname={self.dbname}\n"
                                f"  5. See scripts/POSTGRESQL_SETUP.md for detailed setup instructions"
                            )
                        else:
                            logger.error(f"Failed to create connection pool after {max_retries} attempts: {e}")
                        raise
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool (context manager).
        
        Yields:
            psycopg2 connection object
        """
        self._create_pool()
        
        conn = None
        try:
            conn = self._connection_pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                self._connection_pool.putconn(conn)
    
    def close(self) -> None:
        """Close all connections in the pool."""
        if self._connection_pool:
            self._connection_pool.closeall()
            self._connection_pool = None
            logger.info("Connection pool closed")
    
    def test_connection(self) -> bool:
        """Test database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_universe(self, as_of_date: Optional[date] = None) -> List[Dict[str, Any]]:
        """Get NDX universe members.
        
        Args:
            as_of_date: Date to get universe for (defaults to latest)
            
        Returns:
            List of universe records with asset_id, symbol, name, etc.
        """
        cursor_factory = RealDictCursor if RealDictCursor else None
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=cursor_factory) as cur:
                if as_of_date:
                    cur.execute("""
                        SELECT asset_id, symbol, name, sector, exchange, figi, 
                               first_seen_date, last_seen_date, is_active
                        FROM ndx_universe
                        WHERE (first_seen_date IS NULL OR first_seen_date <= %s) 
                        AND (last_seen_date IS NULL OR last_seen_date >= %s)
                        ORDER BY symbol
                    """, (as_of_date, as_of_date))
                else:
                    cur.execute("""
                        SELECT asset_id, symbol, name, sector, exchange, figi,
                               first_seen_date, last_seen_date, is_active
                        FROM ndx_universe
                        WHERE is_active = TRUE
                        ORDER BY symbol
                    """)
                if RealDictCursor:
                    return [dict(row) for row in cur.fetchall()]
                else:
                    # Fallback: convert tuple results to dicts
                    columns = [desc[0] for desc in cur.description]
                    return [dict(zip(columns, row)) for row in cur.fetchall()]
    
    def get_ohlcv(
        self,
        asset_id: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        """Get OHLCV data for a single asset.
        
        Args:
            asset_id: Asset ID (e.g., "NAS:AAPL")
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of OHLCV records ordered by date
        """
        cursor_factory = RealDictCursor if RealDictCursor else None
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=cursor_factory) as cur:
                cur.execute("""
                    SELECT date, asset_id, open, high, low, close, volume,
                           dividend, is_liquid, is_stock, is_spx, is_ndx
                    FROM ndx_ohlcv
                    WHERE asset_id = %s AND date >= %s AND date <= %s
                    ORDER BY date ASC
                """, (asset_id, start_date, end_date))
                if RealDictCursor:
                    return [dict(row) for row in cur.fetchall()]
                else:
                    # Fallback: convert tuple results to dicts
                    columns = [desc[0] for desc in cur.description]
                    return [dict(zip(columns, row)) for row in cur.fetchall()]
    
    def get_cross_sectional(
        self,
        as_of_date: date,
        fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get cross-sectional data for all assets on a date.
        
        Args:
            as_of_date: Date to get data for
            fields: Optional list of fields to return (defaults to all)
            
        Returns:
            List of records with date, asset_id, and requested fields
        """
        if fields is None:
            field_list = "*"
        else:
            # Validate fields
            valid_fields = {
                "date", "asset_id", "open", "high", "low", "close", "volume",
                "dividend", "is_liquid", "is_stock", "is_spx", "is_ndx"
            }
            requested = set(fields) | {"date", "asset_id"}  # Always include these
            invalid = requested - valid_fields
            if invalid:
                raise ValueError(f"Invalid fields: {invalid}")
            field_list = ", ".join(requested)
        
        cursor_factory = RealDictCursor if RealDictCursor else None
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=cursor_factory) as cur:
                cur.execute(f"""
                    SELECT {field_list}
                    FROM ndx_ohlcv
                    WHERE date = %s
                    ORDER BY asset_id
                """, (as_of_date,))
                if RealDictCursor:
                    return [dict(row) for row in cur.fetchall()]
                else:
                    # Fallback: convert tuple results to dicts
                    columns = [desc[0] for desc in cur.description]
                    return [dict(zip(columns, row)) for row in cur.fetchall()]
    
    def get_latest_date(self) -> Optional[date]:
        """Get the most recent date with data.
        
        Returns:
            Latest date or None if no data exists
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(date) FROM ndx_ohlcv")
                result = cur.fetchone()
                return result[0] if result and result[0] else None
    
    def get_date_range(self) -> Optional[Tuple[date, date]]:
        """Get the date range of available NDX data.
        
        Returns:
            Tuple of (min_date, max_date) or None if no data exists
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MIN(date), MAX(date) FROM ndx_ohlcv")
                result = cur.fetchone()
                if result and result[0] and result[1]:
                    return (result[0], result[1])
                return None
    
    def get_crypto_universe(self, as_of_date: Optional[date] = None) -> List[Dict[str, Any]]:
        """Get cryptocurrency universe members.
        
        Args:
            as_of_date: Date to get universe for (defaults to latest)
            
        Returns:
            List of universe records with asset_id, symbol, name, etc.
        """
        cursor_factory = RealDictCursor if RealDictCursor else None
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=cursor_factory) as cur:
                if as_of_date:
                    cur.execute("""
                        SELECT asset_id, symbol, name, first_seen_date, last_seen_date, is_active
                        FROM crypto_universe
                        WHERE (first_seen_date IS NULL OR first_seen_date <= %s) 
                        AND (last_seen_date IS NULL OR last_seen_date >= %s)
                        ORDER BY symbol
                    """, (as_of_date, as_of_date))
                else:
                    cur.execute("""
                        SELECT asset_id, symbol, name, first_seen_date, last_seen_date, is_active
                        FROM crypto_universe
                        WHERE is_active = TRUE
                        ORDER BY symbol
                    """)
                if RealDictCursor:
                    return [dict(row) for row in cur.fetchall()]
                else:
                    columns = [desc[0] for desc in cur.description]
                    return [dict(zip(columns, row)) for row in cur.fetchall()]
    
    def get_crypto_ohlcv(
        self,
        asset_id: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        """Get OHLCV data for a single cryptocurrency.
        
        Args:
            asset_id: Asset ID (e.g., "BTC", "ETH")
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of OHLCV records ordered by date
        """
        cursor_factory = RealDictCursor if RealDictCursor else None
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=cursor_factory) as cur:
                cur.execute("""
                    SELECT date, asset_id, open, high, low, close, volume, is_liquid
                    FROM crypto_ohlcv
                    WHERE asset_id = %s AND date >= %s AND date <= %s
                    ORDER BY date ASC
                """, (asset_id, start_date, end_date))
                if RealDictCursor:
                    return [dict(row) for row in cur.fetchall()]
                else:
                    columns = [desc[0] for desc in cur.description]
                    return [dict(zip(columns, row)) for row in cur.fetchall()]
    
    def get_crypto_latest_date(self) -> Optional[date]:
        """Get the most recent date with crypto data.
        
        Returns:
            Latest date or None if no data exists
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(date) FROM crypto_ohlcv")
                result = cur.fetchone()
                return result[0] if result and result[0] else None
    
    def get_crypto_date_range(self) -> Optional[Tuple[date, date]]:
        """Get the date range of available crypto data.
        
        Returns:
            Tuple of (min_date, max_date) or None if no data exists
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MIN(date), MAX(date) FROM crypto_ohlcv")
                result = cur.fetchone()
                if result and result[0] and result[1]:
                    return (result[0], result[1])
                return None


# Singleton client instance
_client: Optional[DatabaseClient] = None


def get_client() -> DatabaseClient:
    """Get or create singleton database client.
    
    Returns:
        DatabaseClient instance
    """
    global _client
    if _client is None:
        _client = DatabaseClient()
    return _client
