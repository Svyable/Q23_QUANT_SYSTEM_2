"""q23.strategy.data_gap_detector

Detect missing recent trading days in Quantiacs data.

Identifies gaps between the latest available date in a dataset and today,
helping determine when Marketstack data should be fetched to fill gaps.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, List

try:
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
    import xarray as xr  # type: ignore
    import pytz  # type: ignore
except ImportError:
    pd = None
    np = None
    xr = None
    pytz = None

from q23.shared.config import cfg


def _require_xr() -> None:
    """Check that xarray is available."""
    if xr is None:
        raise ImportError("xarray is required for data_gap_detector")


def _require_pd() -> None:
    """Check that pandas is available."""
    if pd is None:
        raise ImportError("pandas is required for data_gap_detector")


def get_latest_quantiacs_date(dataset: "xr.Dataset") -> Optional[pd.Timestamp]:
    """Get the latest date from a Quantiacs xarray Dataset.
    
    Args:
        dataset: xarray Dataset with 'time' dimension
    
    Returns:
        Latest date as pandas Timestamp, or None if dataset is empty
    """
    _require_xr()
    _require_pd()
    
    if dataset is None or 'time' not in dataset.dims:
        return None
    
    if dataset.sizes.get('time', 0) == 0:
        return None
    
    try:
        # Get the last time value
        time_values = dataset.time.values
        if len(time_values) == 0:
            return None
        
        latest_time = time_values[-1]
        
        # Convert to pandas Timestamp if needed
        if isinstance(latest_time, pd.Timestamp):
            return latest_time
        elif isinstance(latest_time, np.datetime64):
            return pd.Timestamp(latest_time)
        else:
            # Try to parse as date string
            return pd.Timestamp(latest_time)
            
    except Exception as e:
        # If we can't determine the date, return None
        return None


def is_trading_day(date: pd.Timestamp) -> bool:
    """Check if a date is a trading day (Monday-Friday).
    
    Args:
        date: Date to check
    
    Returns:
        True if Monday-Friday, False otherwise
    """
    _require_pd()
    
    # 0 = Monday, 6 = Sunday
    return date.weekday() < 5


def get_most_recent_trading_day(
    reference_time: Optional[datetime] = None,
    pre_market_cutoff_hour: Optional[int] = None,
) -> pd.Timestamp:
    """Get the most recent trading day for live data fetching.
    
    For pre-market runs (before market close), returns yesterday.
    For post-market runs (after market close), returns today if it's a trading day.
    Handles weekends and holidays by walking back to the last trading day.
    
    Args:
        reference_time: Reference datetime (defaults to now, in ET timezone)
        pre_market_cutoff_hour: Hour (ET) before which we consider it pre-market (defaults to cfg.marketstack.PRE_MARKET_CUTOFF_HOUR)
    
    Returns:
        Most recent trading day as pandas Timestamp (normalized to midnight ET)
    """
    _require_pd()
    
    if pytz is None:
        raise ImportError("pytz is required for get_most_recent_trading_day")
    
    # Get reference time in ET
    if reference_time is None:
        reference_time = datetime.now(pytz.timezone('US/Eastern'))
    elif reference_time.tzinfo is None:
        # Assume UTC if no timezone, convert to ET
        reference_time = pytz.UTC.localize(reference_time).astimezone(pytz.timezone('US/Eastern'))
    else:
        reference_time = reference_time.astimezone(pytz.timezone('US/Eastern'))
    
    if pre_market_cutoff_hour is None:
        pre_market_cutoff_hour = getattr(cfg.marketstack, 'PRE_MARKET_CUTOFF_HOUR', 9)
    
    # Get today's date in ET
    today_et = pd.Timestamp(reference_time.date())
    
    # Check if today is a trading day
    if not is_trading_day(today_et):
        # Today is weekend/holiday, walk back to last trading day
        candidate = today_et - timedelta(days=1)
        while not is_trading_day(candidate):
            candidate -= timedelta(days=1)
        result = candidate.normalize()
        # #region agent log
        try:
            import json
            import time
            with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "data_gap_detector.py:145",
                    "message": "get_most_recent_trading_day (weekend/holiday)",
                    "data": {
                        "today_et": str(today_et),
                        "result": str(result),
                        "current_hour": reference_time.hour
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except: pass
        # #endregion
        return result
    
    # Today is a trading day
    # Check if we're before market close (pre-market)
    current_hour = reference_time.hour
    
    if current_hour < pre_market_cutoff_hour:
        # Pre-market: use yesterday
        candidate = today_et - timedelta(days=1)
        while not is_trading_day(candidate):
            candidate -= timedelta(days=1)
        result = candidate.normalize()
        # #region agent log
        try:
            import json
            import time
            with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "data_gap_detector.py:165",
                    "message": "get_most_recent_trading_day (pre-market)",
                    "data": {
                        "today_et": str(today_et),
                        "result": str(result),
                        "current_hour": current_hour,
                        "cutoff_hour": pre_market_cutoff_hour
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except: pass
        # #endregion
        return result
    else:
        # Post-market or during market hours: use today
        result = today_et.normalize()
        # #region agent log
        try:
            import json
            import time
            with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "data_gap_detector.py:180",
                    "message": "get_most_recent_trading_day (post-market)",
                    "data": {
                        "today_et": str(today_et),
                        "result": str(result),
                        "current_hour": current_hour
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except: pass
        # #endregion
        return result


def get_trading_days_between(start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[pd.Timestamp]:
    """Get list of trading days between two dates (inclusive).
    
    Args:
        start_date: Start date
        end_date: End date
    
    Returns:
        List of trading days (Monday-Friday) between dates
    """
    _require_pd()
    
    trading_days = []
    current = start_date
    
    while current <= end_date:
        if is_trading_day(current):
            trading_days.append(current)
        current += timedelta(days=1)
    
    return trading_days


def get_missing_dates(
    latest_date: pd.Timestamp,
    target_date: Optional[pd.Timestamp] = None,
    lookback_days: Optional[int] = None,
) -> List[pd.Timestamp]:
    """Calculate missing trading days between latest_date and target_date.
    
    Args:
        latest_date: Latest available date in dataset
        target_date: Target date (defaults to today)
        lookback_days: Maximum number of days to look back (defaults to cfg.marketstack.DEFAULT_LOOKBACK_DAYS)
    
    Returns:
        List of missing trading days (empty if no gap or gap too large)
    """
    _require_pd()
    
    if target_date is None:
        target_date = pd.Timestamp.now().normalize()
    
    if lookback_days is None:
        lookback_days = cfg.marketstack.DEFAULT_LOOKBACK_DAYS
    
    # Normalize dates to midnight
    latest_date = latest_date.normalize()
    target_date = target_date.normalize()
    
    # If latest_date is >= target_date, no gap
    if latest_date >= target_date:
        return []
    
    # Get all trading days between latest_date and target_date
    all_trading_days = get_trading_days_between(latest_date + timedelta(days=1), target_date)
    
    # Limit to lookback_days
    if len(all_trading_days) > lookback_days:
        return []
    
    return all_trading_days


def should_fetch_marketstack(
    dataset: "xr.Dataset",
    lookback_days: Optional[int] = None,
    target_date: Optional[pd.Timestamp] = None,
    force_live_data: bool = False,
) -> bool:
    """Determine if Marketstack data should be fetched to fill gaps.
    
    Args:
        dataset: Quantiacs xarray Dataset
        lookback_days: Maximum number of days to look back (defaults to cfg.marketstack.DEFAULT_LOOKBACK_DAYS)
        target_date: Target date to check against (defaults to most_recent_trading_day for live data)
        force_live_data: If True, always fetch latest data even if no gap detected
    
    Returns:
        True if Marketstack fetch is needed, False otherwise
    """
    _require_xr()
    _require_pd()
    
    if dataset is None or 'time' not in dataset.dims:
        return force_live_data
    
    latest_date = get_latest_quantiacs_date(dataset)
    
    if latest_date is None:
        return force_live_data
    
    # For live data requests, use most recent trading day as target
    if target_date is None or force_live_data:
        target_date = get_most_recent_trading_day()
    
    missing_dates = get_missing_dates(latest_date, target_date, lookback_days)
    
    # If force_live_data is True, also check if we're at the boundary (latest_date < target_date)
    if force_live_data:
        latest_date_norm = latest_date.normalize()
        target_date_norm = target_date.normalize()
        if latest_date_norm < target_date_norm:
            return True
    
    return len(missing_dates) > 0


def get_missing_date_range(
    dataset: "xr.Dataset",
    lookback_days: Optional[int] = None,
    target_date: Optional[pd.Timestamp] = None,
    force_live_data: bool = False,
) -> Optional[tuple[str, str]]:
    """Get date range for missing days (for Marketstack API).
    
    Args:
        dataset: Quantiacs xarray Dataset
        lookback_days: Maximum number of days to look back
        target_date: Target date to check against (defaults to most_recent_trading_day for live data)
        force_live_data: If True, always return range for latest trading day even if no gap
    
    Returns:
        Tuple of (date_from, date_to) in YYYY-MM-DD format, or None if no gap
    """
    _require_xr()
    _require_pd()
    
    latest_date = get_latest_quantiacs_date(dataset)
    
    # #region agent log
    try:
        import json
        import time
        with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "B",
                "location": "data_gap_detector.py:263",
                "message": "get_missing_date_range entry",
                "data": {
                    "latest_date": str(latest_date) if latest_date else None,
                    "target_date": str(target_date) if target_date else None,
                    "force_live_data": force_live_data,
                    "lookback_days": lookback_days
                },
                "timestamp": int(time.time() * 1000)
            }) + "\n")
    except: pass
    # #endregion
    
    if latest_date is None:
        if force_live_data:
            # Return range for most recent trading day
            target = get_most_recent_trading_day()
            date_str = target.strftime('%Y-%m-%d')
            # #region agent log
            try:
                import json
                import time
                with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "B",
                        "location": "data_gap_detector.py:291",
                        "message": "Returning target (no latest_date)",
                        "data": {"date_str": date_str},
                        "timestamp": int(time.time() * 1000)
                    }) + "\n")
            except: pass
            # #endregion
            return (date_str, date_str)
        return None
    
    # For live data requests, use most recent trading day as target
    if target_date is None or force_live_data:
        target_date = get_most_recent_trading_day()
        # #region agent log
        try:
            import json
            import time
            with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "data_gap_detector.py:299",
                    "message": "Computed target_date",
                    "data": {
                        "target_date": str(target_date),
                        "latest_date": str(latest_date)
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except: pass
        # #endregion
    
    missing_dates = get_missing_dates(latest_date, target_date, lookback_days)
    
    # If force_live_data is True, always return target date range (even if no gap)
    # This ensures we always fetch the latest available data
    if force_live_data:
        target_date_norm = target_date.normalize()
        date_str = target_date_norm.strftime('%Y-%m-%d')
        # #region agent log
        try:
            import json
            import time
            with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "data_gap_detector.py:310",
                    "message": "Returning target (force_live_data=True)",
                    "data": {
                        "date_str": date_str,
                        "latest_date": str(latest_date),
                        "target_date": str(target_date),
                        "missing_dates_count": len(missing_dates)
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except: pass
        # #endregion
        return (date_str, date_str)
    
    if not missing_dates:
        # #region agent log
        try:
            import json
            import time
            with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "data_gap_detector.py:330",
                    "message": "No missing dates, returning None",
                    "data": {
                        "latest_date": str(latest_date),
                        "target_date": str(target_date)
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except: pass
        # #endregion
        return None
    
    # Get first and last missing dates
    date_from = missing_dates[0].strftime('%Y-%m-%d')
    date_to = missing_dates[-1].strftime('%Y-%m-%d')
    
    # #region agent log
    try:
        import json
        import time
        with open('/Users/svenbenson/Q23_QUANT_SYSTEM 2/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "B",
                "location": "data_gap_detector.py:340",
                "message": "Returning date range from missing_dates",
                "data": {
                    "date_from": date_from,
                    "date_to": date_to,
                    "missing_dates_count": len(missing_dates)
                },
                "timestamp": int(time.time() * 1000)
            }) + "\n")
    except: pass
    # #endregion
    
    return (date_from, date_to)
