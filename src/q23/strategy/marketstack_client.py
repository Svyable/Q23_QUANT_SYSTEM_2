"""q23.strategy.marketstack_client

Marketstack API client for fetching EOD (End-of-Day) market data.

Provides functionality to fetch recent market data from Marketstack API
to supplement Quantiacs data when recent days are missing.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
import warnings

try:
    import pandas as pd  # type: ignore
    import requests  # type: ignore
    import pytz  # type: ignore
except ImportError:
    pd = None
    requests = None
    pytz = None

from q23.shared.config import cfg
from q23.strategy.marketstack_telemetry import (
    record_marketstack_fetch,
    FetchResult,
    MarketstackStatus,
    get_telemetry_manager,
)


def _require_dependencies() -> None:
    """Check that required dependencies are available."""
    if pd is None:
        raise ImportError("pandas is required for marketstack_client")
    if requests is None:
        raise ImportError("requests is required for marketstack_client")


class MarketstackClient:
    """Client for Marketstack API EOD data fetching."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        rate_limit_delay: Optional[float] = None,
    ):
        """Initialize Marketstack client.
        
        Args:
            api_key: Marketstack API key (defaults to cfg.marketstack.API_KEY)
            base_url: Base URL for API (defaults to cfg.marketstack.BASE_URL)
            rate_limit_delay: Delay between requests in seconds (defaults to cfg.marketstack.RATE_LIMIT_DELAY)
        """
        _require_dependencies()
        
        self.api_key = api_key or cfg.marketstack.API_KEY
        self.base_url = (base_url or cfg.marketstack.BASE_URL).rstrip('/')
        self.rate_limit_delay = rate_limit_delay or cfg.marketstack.RATE_LIMIT_DELAY
        self.last_request_time: Optional[float] = None
        
        if not self.api_key:
            raise ValueError("Marketstack API key is required")
    
    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Dict[str, Any]:
        """Make API request with retry logic.
        
        Args:
            endpoint: API endpoint (e.g., 'eod')
            params: Request parameters
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
        
        Returns:
            JSON response as dictionary
        
        Raises:
            requests.RequestException: If all retries fail
        """
        _require_dependencies()
        
        url = f"{self.base_url}/{endpoint}"
        params['access_key'] = self.api_key
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                self._enforce_rate_limit()
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                # Check for API errors in response
                if 'error' in data:
                    error_msg = data.get('error', {}).get('message', 'Unknown API error')
                    raise ValueError(f"Marketstack API error: {error_msg}")
                
                return data
                
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Exponential backoff
                    delay = retry_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    raise RuntimeError(
                        f"Marketstack API request failed after {max_retries} attempts: {e}"
                    ) from e
        
        raise RuntimeError(f"Marketstack API request failed: {last_error}") from last_error
    
    def fetch_eod_data(
        self,
        symbols: List[str],
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        sort: str = "DESC",
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch EOD (End-of-Day) OHLCV data from Marketstack.
        
        Args:
            symbols: List of ticker symbols (e.g., ['AAPL', 'MSFT'])
            date_from: Start date in YYYY-MM-DD format (optional)
            date_to: End date in YYYY-MM-DD format (optional)
            sort: Sort order ('ASC' or 'DESC', default 'DESC')
            limit: Maximum number of records per symbol (optional)
        
        Returns:
            DataFrame with columns: symbol, date, open, high, low, close, volume
            Additional columns may include: adj_open, adj_high, adj_low, adj_close, adj_volume
        
        Raises:
            ValueError: If symbols list is empty or invalid parameters
            RuntimeError: If API request fails
        """
        _require_dependencies()
        
        if not symbols:
            raise ValueError("Symbols list cannot be empty")
        
        # Marketstack API accepts comma-separated symbols
        symbols_str = ','.join(symbols)
        
        params: Dict[str, Any] = {
            'symbols': symbols_str,
            'sort': sort,
        }
        
        if date_from:
            params['date_from'] = date_from
        if date_to:
            params['date_to'] = date_to
        if limit:
            params['limit'] = limit
        
        start_time = time.time()
        try:
            response_data = self._make_request('eod', params)
            
            # Parse response
            data_list = response_data.get('data', [])
            
            if not data_list:
                duration_ms = (time.time() - start_time) * 1000
                record_marketstack_fetch(
                    strategy_id=None,
                    symbols_count=len(symbols),
                    date_from=date_from,
                    date_to=date_to,
                    fetched_date=None,
                    result=FetchResult.EMPTY,
                    rows_fetched=0,
                    duration_ms=duration_ms,
                )
                # Return empty DataFrame with expected columns
                return pd.DataFrame(columns=['symbol', 'date', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert to DataFrame
            df = pd.DataFrame(data_list)
            
            # Rename columns to match expected format
            column_mapping = {
                'symbol': 'symbol',
                'date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
            }
            
            # Ensure required columns exist
            required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Marketstack response missing required columns: {missing_cols}")
            
            # Select and rename columns
            df = df[required_cols].copy()
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Ensure numeric columns are numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by symbol and date
            df = df.sort_values(['symbol', 'date'], ascending=[True, sort == 'ASC'])
            
            duration_ms = (time.time() - start_time) * 1000
            record_marketstack_fetch(
                strategy_id=None,
                symbols_count=len(symbols),
                date_from=date_from,
                date_to=date_to,
                fetched_date=None,
                result=FetchResult.SUCCESS,
                rows_fetched=len(df),
                duration_ms=duration_ms,
            )
            
            return df.reset_index(drop=True)
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            record_marketstack_fetch(
                strategy_id=None,
                symbols_count=len(symbols),
                date_from=date_from,
                date_to=date_to,
                fetched_date=None,
                result=FetchResult.FAILED,
                rows_fetched=0,
                error_message=error_msg,
                duration_ms=duration_ms,
            )
            get_telemetry_manager().update_status(MarketstackStatus.ERROR)
            warnings.warn(f"Failed to fetch Marketstack EOD data: {e}")
            # Return empty DataFrame with expected structure
            return pd.DataFrame(columns=['symbol', 'date', 'open', 'high', 'low', 'close', 'volume'])
    
    def fetch_latest_eod(
        self,
        symbols: List[str],
        limit: int = 5,
    ) -> pd.DataFrame:
        """Fetch latest N days of EOD data for symbols.
        
        Args:
            symbols: List of ticker symbols
            limit: Number of recent days to fetch (default: 5)
        
        Returns:
            DataFrame with EOD data (same format as fetch_eod_data)
        """
        return self.fetch_eod_data(
            symbols=symbols,
            sort='DESC',
            limit=limit,
        )
    
    def fetch_most_recent_eod(
        self,
        symbols: List[str],
        target_date: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, str]:
        """Fetch the most recent available EOD data.
        
        Tries to fetch today's data first. If not available (pre-market, API delay),
        falls back to yesterday's data. Handles weekends/holidays by using the
        most recent trading day.
        
        Args:
            symbols: List of ticker symbols
            target_date: Target date in YYYY-MM-DD format (defaults to today in ET)
        
        Returns:
            Tuple of (DataFrame with EOD data, date_string that was fetched)
            DataFrame format matches fetch_eod_data()
            date_string indicates which date's data was actually fetched
        """
        _require_dependencies()
        
        if pytz is None:
            raise ImportError("pytz is required for fetch_most_recent_eod")
        
        if not symbols:
            empty_df = pd.DataFrame(columns=['symbol', 'date', 'open', 'high', 'low', 'close', 'volume'])
            return empty_df, target_date or datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
        
        # Determine target date (most recent trading day)
        if target_date is None:
            # Import here to avoid circular dependency
            from q23.strategy.data_gap_detector import get_most_recent_trading_day
            target_ts = get_most_recent_trading_day()
            target_date = target_ts.strftime('%Y-%m-%d')
        
        start_time = time.time()
        # Try to fetch target date first
        try:
            df = self.fetch_eod_for_date_range(
                symbols=symbols,
                start_date=target_date,
                end_date=target_date,
            )
            
            # Check if we got data for the target date
            if not df.empty:
                # Filter to only the target date (in case API returns extra dates)
                df['date'] = pd.to_datetime(df['date']).dt.date
                target_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
                df_filtered = df[df['date'] == target_date_obj]
                
                if not df_filtered.empty:
                    duration_ms = (time.time() - start_time) * 1000
                    record_marketstack_fetch(
                        strategy_id=None,
                        symbols_count=len(symbols),
                        date_from=target_date,
                        date_to=target_date,
                        fetched_date=target_date,
                        result=FetchResult.SUCCESS,
                        rows_fetched=len(df_filtered),
                        duration_ms=duration_ms,
                    )
                    return df_filtered.reset_index(drop=True), target_date
            
            # No data for target date, try yesterday
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            yesterday_dt = target_dt - timedelta(days=1)
            yesterday_str = yesterday_dt.strftime('%Y-%m-%d')
            
            df_yesterday = self.fetch_eod_for_date_range(
                symbols=symbols,
                start_date=yesterday_str,
                end_date=yesterday_str,
            )
            
            if not df_yesterday.empty:
                df_yesterday['date'] = pd.to_datetime(df_yesterday['date']).dt.date
                yesterday_date_obj = datetime.strptime(yesterday_str, '%Y-%m-%d').date()
                df_yesterday_filtered = df_yesterday[df_yesterday['date'] == yesterday_date_obj]
                
                if not df_yesterday_filtered.empty:
                    duration_ms = (time.time() - start_time) * 1000
                    record_marketstack_fetch(
                        strategy_id=None,
                        symbols_count=len(symbols),
                        date_from=yesterday_str,
                        date_to=yesterday_str,
                        fetched_date=yesterday_str,
                        result=FetchResult.SUCCESS,
                        rows_fetched=len(df_yesterday_filtered),
                        duration_ms=duration_ms,
                    )
                    warnings.warn(
                        f"Target date {target_date} data not available, using {yesterday_str} instead",
                        UserWarning,
                    )
                    return df_yesterday_filtered.reset_index(drop=True), yesterday_str
            
            # Neither date has data, return empty with target date
            duration_ms = (time.time() - start_time) * 1000
            record_marketstack_fetch(
                strategy_id=None,
                symbols_count=len(symbols),
                date_from=target_date,
                date_to=yesterday_str,
                fetched_date=None,
                result=FetchResult.EMPTY,
                rows_fetched=0,
                duration_ms=duration_ms,
            )
            warnings.warn(
                f"No Marketstack data available for {target_date} or {yesterday_str}",
                UserWarning,
            )
            empty_df = pd.DataFrame(columns=['symbol', 'date', 'open', 'high', 'low', 'close', 'volume'])
            return empty_df, target_date
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            record_marketstack_fetch(
                strategy_id=None,
                symbols_count=len(symbols),
                date_from=target_date,
                date_to=None,
                fetched_date=None,
                result=FetchResult.FAILED,
                rows_fetched=0,
                error_message=error_msg,
                duration_ms=duration_ms,
            )
            get_telemetry_manager().update_status(MarketstackStatus.ERROR)
            warnings.warn(
                f"Failed to fetch most recent Marketstack EOD data: {e}. "
                f"Returning empty DataFrame.",
                UserWarning,
            )
            empty_df = pd.DataFrame(columns=['symbol', 'date', 'open', 'high', 'low', 'close', 'volume'])
            return empty_df, target_date
    
    def fetch_eod_for_date_range(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch EOD data for a specific date range.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            DataFrame with EOD data for the date range
        """
        return self.fetch_eod_data(
            symbols=symbols,
            date_from=start_date,
            date_to=end_date,
            sort='ASC',
        )
