"""q23.strategy.data_cache

Daily file-based data caching for Quantiacs data.

Caches data to .cache/data/YYYYMMDD/ to avoid redundant API calls
within the same day. Cache is automatically invalidated daily.

Benefits:
- Single API call per day: First strategy run fetches data, subsequent use cache
- Automatic daily refresh: Cache key includes date, so fresh data is pulled each day
- Disk-based persistence: Cache survives Python session restarts
- Backward compatible: Falls back to API if cache unavailable
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple

from q23.strategy.data_gap_detector import (
    get_latest_quantiacs_date,
    get_most_recent_trading_day,
)

try:
    import xarray as xr  # type: ignore
    import pandas as pd  # type: ignore
except ImportError:
    xr = None
    pd = None


def _require_xr() -> None:
    if xr is None:
        raise ImportError("xarray is required for q23.strategy.data_cache")


def _require_pd() -> None:
    if pd is None:
        raise ImportError("pandas is required for Marketstack refresh detection")


def _get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate up from src/q23/strategy to project root
    return Path(__file__).resolve().parents[3]


def _get_cache_dir() -> Path:
    """Get the cache directory path, creating it if necessary."""
    cache_dir = _get_project_root() / ".cache" / "data"
    return cache_dir


def get_today_prefix() -> str:
    """Get today's date as YYYYMMDD string for cache key."""
    return datetime.now().strftime("%Y%m%d")


def get_cache_path(index_type: str, date_prefix: Optional[str] = None) -> Path:
    """Get cache file path for an index type.
    
    Args:
        index_type: Index type ("SPX", "SP500", "NDX", "NAS100", etc.)
        date_prefix: Date prefix for cache (defaults to today's YYYYMMDD)
    
    Returns:
        Path to cache file: .cache/data/YYYYMMDD/<index_type>_data.nc
    """
    if date_prefix is None:
        date_prefix = get_today_prefix()
    
    cache_dir = _get_cache_dir() / date_prefix
    
    # Normalize index type to lowercase for consistent filenames
    filename = f"{index_type.lower()}_data.nc"
    
    return cache_dir / filename


def is_cache_valid(index_type: str) -> bool:
    """Check if cache exists and is from today.
    
    Args:
        index_type: Index type to check
    
    Returns:
        True if valid cache exists for today
    """
    cache_path = get_cache_path(index_type)
    return cache_path.exists()


def _validate_dataset(ds: "xr.Dataset") -> bool:
    """Validate that a dataset has the required structure for market data.
    
    Args:
        ds: Dataset to validate
    
    Returns:
        True if valid, False otherwise
    """
    _require_xr()
    
    # Must have time and asset dimensions
    if "time" not in ds.dims or "asset" not in ds.dims:
        return False
    
    # Must have at least some data
    if ds.sizes.get("time", 0) == 0 or ds.sizes.get("asset", 0) == 0:
        return False
    
    # Must have close price at minimum
    if "close" not in ds.variables:
        return False
    
    return True


def load_cached_data(index_type: str) -> Optional["xr.Dataset"]:
    """Load data from today's cache if exists and is valid.
    
    Automatically tries different netCDF engines if the default fails.
    
    Args:
        index_type: Index type ("SPX", "SP500", "NDX", "NAS100", etc.)
    
    Returns:
        xr.Dataset if cache exists and is valid, None otherwise
    """
    _require_xr()
    
    cache_path = get_cache_path(index_type)
    
    if not cache_path.exists():
        return None
    
    # Try loading with different engines
    engines_to_try = ["netcdf4", "h5netcdf", "scipy", None]  # None = auto-detect
    
    ds = None
    last_error = None
    
    for engine in engines_to_try:
        try:
            if engine is None:
                ds = xr.open_dataset(cache_path)
            else:
                ds = xr.open_dataset(cache_path, engine=engine)
            
            # Load into memory and close file handle
            ds = ds.load()
            break  # Success, exit loop
        except ImportError:
            # Engine not available, try next
            continue
        except Exception as e:
            last_error = e
            continue
    
    if ds is None:
        # All engines failed
        print(f"Warning: Failed to load cache for {index_type}: {last_error}")
        # Try to remove corrupted cache file
        try:
            cache_path.unlink()
        except Exception:
            pass
        return None
    
    # Validate the dataset structure
    if not _validate_dataset(ds):
        print(f"Warning: Invalid cache structure for {index_type}, will re-fetch")
        # Remove invalid cache file
        try:
            cache_path.unlink()
        except Exception:
            pass
        return None
    
    # Validate recency (if stale, treat as invalid so we re-fetch)
    try:
        latest_date = get_latest_quantiacs_date(ds)
        target_date = get_most_recent_trading_day()
        if latest_date is None or (latest_date.normalize() < target_date.normalize()):
            print(
                f"Warning: Cache for {index_type} is stale "
                f"(latest={latest_date}, target={target_date}), will re-fetch"
            )
            return None
    except Exception:
        # If we can't determine recency, keep cache to avoid blocking runs
        pass

    return ds


def _get_available_netcdf_engine() -> Tuple[Optional[str], bool]:
    """Detect which netCDF engine is available.
    
    Returns:
        Tuple of (engine_name, supports_compression)
        engine_name is None if no engine available
    """
    # Try netCDF4 first (best compression support)
    try:
        import netCDF4  # noqa: F401
        return "netcdf4", True
    except ImportError:
        pass
    
    # Try h5netcdf second (good compression support)
    try:
        import h5netcdf  # noqa: F401
        return "h5netcdf", True
    except ImportError:
        pass
    
    # Fall back to scipy (no compression support)
    try:
        import scipy  # noqa: F401
        return "scipy", False
    except ImportError:
        pass
    
    return None, False


def save_to_cache(index_type: str, data: "xr.Dataset") -> bool:
    """Save data to today's cache.
    
    Automatically selects the best available netCDF engine:
    - netcdf4: Best (supports compression)
    - h5netcdf: Good (supports compression)
    - scipy: Basic (no compression, larger files)
    
    Args:
        index_type: Index type ("SPX", "SP500", "NDX", "NAS100", etc.)
        data: xr.Dataset to cache
    
    Returns:
        True if successfully saved, False otherwise
    """
    _require_xr()
    
    # Validate data before saving
    if not _validate_dataset(data):
        print(f"Warning: Cannot cache invalid dataset for {index_type}")
        return False
    
    cache_path = get_cache_path(index_type)
    
    try:
        # Create cache directory if it doesn't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Detect available engine
        engine, supports_compression = _get_available_netcdf_engine()
        
        if engine is None:
            print(f"Warning: No netCDF engine available for caching {index_type}")
            return False
        
        # Build encoding based on engine capabilities
        encoding = {}
        if supports_compression:
            for var in data.data_vars:
                encoding[var] = {"zlib": True, "complevel": 4}
        
        # Save with detected engine
        if supports_compression:
            data.to_netcdf(cache_path, engine=engine, encoding=encoding)
        else:
            # scipy doesn't support compression encoding
            data.to_netcdf(cache_path, engine=engine)
        
        return True
    except Exception as e:
        print(f"Warning: Failed to save cache for {index_type}: {e}")
        # Try fallback without compression
        try:
            data.to_netcdf(cache_path)
            return True
        except Exception as e2:
            print(f"Warning: Fallback save also failed for {index_type}: {e2}")
            return False


def cleanup_old_cache(keep_days: int = 3) -> int:
    """Remove cache directories older than keep_days.
    
    Args:
        keep_days: Number of days of cache to keep (default 3)
    
    Returns:
        Number of directories removed
    """
    cache_base = _get_cache_dir()
    
    if not cache_base.exists():
        return 0
    
    removed = 0
    cutoff_date = datetime.now() - timedelta(days=keep_days)
    cutoff_str = cutoff_date.strftime("%Y%m%d")
    
    try:
        for item in cache_base.iterdir():
            if item.is_dir():
                # Check if directory name is a valid date prefix
                dir_name = item.name
                if len(dir_name) == 8 and dir_name.isdigit():
                    # Compare date strings (works because YYYYMMDD sorts correctly)
                    if dir_name < cutoff_str:
                        try:
                            shutil.rmtree(item)
                            removed += 1
                        except Exception as e:
                            print(f"Warning: Failed to remove old cache {item}: {e}")
    except Exception as e:
        print(f"Warning: Error during cache cleanup: {e}")
    
    return removed


def get_cache_info() -> dict:
    """Get information about current cache state.
    
    Returns:
        Dict with cache info (directories, sizes, engine, etc.)
    """
    cache_base = _get_cache_dir()
    
    # Get engine info
    engine, supports_compression = _get_available_netcdf_engine()
    
    info = {
        "cache_dir": str(cache_base),
        "exists": cache_base.exists(),
        "today": get_today_prefix(),
        "today_cache_exists": False,
        "cached_indices": [],
        "total_size_mb": 0.0,
        "directories": [],
        "netcdf_engine": engine or "none",
        "compression_supported": supports_compression,
    }
    
    if not cache_base.exists():
        return info
    
    total_size = 0
    today = get_today_prefix()
    
    for item in sorted(cache_base.iterdir()):
        if item.is_dir():
            dir_size = sum(f.stat().st_size for f in item.glob("*") if f.is_file())
            total_size += dir_size
            
            info["directories"].append({
                "date": item.name,
                "size_mb": round(dir_size / (1024 * 1024), 2),
                "files": [f.name for f in item.glob("*") if f.is_file()],
            })
            
            if item.name == today:
                info["today_cache_exists"] = True
                info["cached_indices"] = [
                    f.stem.replace("_data", "").upper()
                    for f in item.glob("*.nc")
                ]
    
    info["total_size_mb"] = round(total_size / (1024 * 1024), 2)
    
    return info


def clear_today_cache() -> bool:
    """Clear today's cache (forces re-fetch on next load).
    
    Returns:
        True if cache was cleared, False if no cache existed
    """
    cache_path = get_cache_path("dummy")  # Get today's cache dir
    cache_dir = cache_path.parent
    
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            return True
        except Exception as e:
            print(f"Warning: Failed to clear today's cache: {e}")
            return False
    
    return False


def clear_all_cache() -> bool:
    """Clear all cached data.
    
    Returns:
        True if cache was cleared successfully
    """
    cache_base = _get_cache_dir()
    
    if cache_base.exists():
        try:
            shutil.rmtree(cache_base)
            return True
        except Exception as e:
            print(f"Warning: Failed to clear all cache: {e}")
            return False
    
    return False


def is_marketstack_refresh_needed(
    cached_ds: "xr.Dataset",
    lookback_days: Optional[int] = None,
) -> bool:
    """Check if cached dataset needs Marketstack refresh for recent days.
    
    Args:
        cached_ds: Cached xarray Dataset
        lookback_days: Maximum number of days to look back (defaults to cfg.marketstack.DEFAULT_LOOKBACK_DAYS)
    
    Returns:
        True if Marketstack refresh is needed, False otherwise
    """
    _require_xr()
    _require_pd()
    
    if cached_ds is None or 'time' not in cached_ds.dims:
        return False
    
    if cached_ds.sizes.get('time', 0) == 0:
        return False
    
    try:
        from q23.strategy.data_gap_detector import should_fetch_marketstack
        from q23.shared.config import cfg
        
        if lookback_days is None:
            lookback_days = cfg.marketstack.DEFAULT_LOOKBACK_DAYS
        
        return should_fetch_marketstack(cached_ds, lookback_days=lookback_days)
    except Exception:
        # If we can't determine, assume refresh not needed
        return False


def validate_and_repair_cache() -> dict:
    """Validate all cache files and remove corrupted ones.
    
    Returns:
        Dict with validation results:
        - valid_files: List of valid cache files
        - removed_files: List of removed corrupted files
        - errors: List of error messages
    """
    _require_xr()
    
    cache_base = _get_cache_dir()
    
    result = {
        "valid_files": [],
        "removed_files": [],
        "errors": [],
    }
    
    if not cache_base.exists():
        return result
    
    for date_dir in cache_base.iterdir():
        if not date_dir.is_dir():
            continue
        
        for cache_file in date_dir.glob("*.nc"):
            try:
                # Try to load and validate
                ds = xr.open_dataset(cache_file)
                ds = ds.load()
                
                if _validate_dataset(ds):
                    result["valid_files"].append(str(cache_file))
                else:
                    # Invalid structure, remove
                    try:
                        cache_file.unlink()
                        result["removed_files"].append(str(cache_file))
                    except Exception as e:
                        result["errors"].append(f"Failed to remove {cache_file}: {e}")
            except Exception as e:
                # Corrupted file, remove
                try:
                    cache_file.unlink()
                    result["removed_files"].append(str(cache_file))
                except Exception as e2:
                    result["errors"].append(f"Failed to remove {cache_file}: {e2}")
        
        # Clean up empty date directories
        try:
            if date_dir.exists() and not any(date_dir.iterdir()):
                date_dir.rmdir()
        except Exception:
            pass
    
    return result
