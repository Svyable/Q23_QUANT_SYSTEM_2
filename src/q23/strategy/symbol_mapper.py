"""q23.strategy.symbol_mapper

Utility to map Quantiacs asset IDs to Marketstack ticker symbols.

Uses id-translation.csv to convert between Quantiacs server_id format
and Marketstack ticker symbols.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, List

try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None


def _require_pd() -> None:
    """Check that pandas is available."""
    if pd is None:
        raise ImportError("pandas is required for symbol_mapper")


def _get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate up from src/q23/strategy to project root
    return Path(__file__).resolve().parents[3]


def _get_id_translation_path() -> Path:
    """Get path to id-translation.csv file."""
    return _get_project_root() / "id-translation.csv"


# Cache for translation mappings
_translation_cache: Optional[pd.DataFrame] = None
_asset_to_ticker_cache: Optional[Dict[str, str]] = None
_ticker_to_asset_cache: Optional[Dict[str, str]] = None


def load_id_translation() -> pd.DataFrame:
    """Load id-translation.csv file.
    
    Returns:
        DataFrame with columns: server_id, user_id
    
    Raises:
        FileNotFoundError: If id-translation.csv doesn't exist
        ImportError: If pandas is not available
    """
    _require_pd()
    
    global _translation_cache
    
    if _translation_cache is not None:
        return _translation_cache
    
    translation_path = _get_id_translation_path()
    
    if not translation_path.exists():
        raise FileNotFoundError(
            f"id-translation.csv not found at {translation_path}. "
            "This file is required for symbol mapping."
        )
    
    df = pd.read_csv(translation_path)
    
    # Validate required columns
    required_cols = ['server_id', 'user_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"id-translation.csv missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )
    
    # Cache the result
    _translation_cache = df.copy()
    
    return _translation_cache


def _build_mapping_caches() -> None:
    """Build in-memory caches for fast lookups."""
    global _asset_to_ticker_cache, _ticker_to_asset_cache
    
    if _asset_to_ticker_cache is not None and _ticker_to_asset_cache is not None:
        return
    
    df = load_id_translation()
    
    _asset_to_ticker_cache = {}
    _ticker_to_asset_cache = {}
    
    for _, row in df.iterrows():
        server_id = str(row['server_id']).strip()
        user_id = str(row['user_id']).strip()
        
        # Extract ticker from user_id (format: "EXCHANGE:TICKER")
        if ':' in user_id:
            exchange, ticker = user_id.split(':', 1)
            ticker = ticker.strip()
            
            # Map asset ID to ticker
            _asset_to_ticker_cache[server_id] = ticker
            
            # Map ticker to asset ID (handle duplicates by keeping first)
            if ticker not in _ticker_to_asset_cache:
                _ticker_to_asset_cache[ticker] = server_id


def asset_id_to_ticker(asset_id: str) -> Optional[str]:
    """Convert Quantiacs asset ID to Marketstack ticker symbol.
    
    Args:
        asset_id: Quantiacs server_id (e.g., "tts-99960993")
    
    Returns:
        Ticker symbol (e.g., "AAPL") or None if not found
    """
    _build_mapping_caches()
    
    if _asset_to_ticker_cache is None:
        return None
    
    asset_id_str = str(asset_id).strip()
    return _asset_to_ticker_cache.get(asset_id_str)


def ticker_to_asset_id(ticker: str) -> Optional[str]:
    """Convert Marketstack ticker symbol to Quantiacs asset ID.
    
    Args:
        ticker: Ticker symbol (e.g., "AAPL")
    
    Returns:
        Quantiacs server_id (e.g., "tts-99960993") or None if not found
    
    Note:
        This is a reverse lookup. If multiple asset IDs map to the same ticker,
        only the first one found will be returned.
    """
    _build_mapping_caches()
    
    if _ticker_to_asset_cache is None:
        return None
    
    ticker_str = str(ticker).strip().upper()
    return _ticker_to_asset_cache.get(ticker_str)


def batch_convert_assets(asset_ids: List[str]) -> Dict[str, Optional[str]]:
    """Convert a list of asset IDs to ticker symbols.
    
    Args:
        asset_ids: List of Quantiacs asset IDs
    
    Returns:
        Dictionary mapping asset_id -> ticker (None if not found)
    """
    _build_mapping_caches()
    
    result = {}
    
    if _asset_to_ticker_cache is None:
        return {asset_id: None for asset_id in asset_ids}
    
    for asset_id in asset_ids:
        asset_id_str = str(asset_id).strip()
        result[asset_id_str] = _asset_to_ticker_cache.get(asset_id_str)
    
    return result


def batch_convert_tickers(tickers: List[str]) -> Dict[str, Optional[str]]:
    """Convert a list of ticker symbols to asset IDs.
    
    Args:
        tickers: List of ticker symbols
    
    Returns:
        Dictionary mapping ticker -> asset_id (None if not found)
    
    Note:
        If multiple asset IDs map to the same ticker, only the first one
        will be included in the result.
    """
    _build_mapping_caches()
    
    result = {}
    
    if _ticker_to_asset_cache is None:
        return {ticker: None for ticker in tickers}
    
    for ticker in tickers:
        ticker_str = str(ticker).strip().upper()
        result[ticker] = _ticker_to_asset_cache.get(ticker_str)
    
    return result


def get_available_tickers() -> List[str]:
    """Get list of all available ticker symbols from translation file.
    
    Returns:
        List of ticker symbols
    """
    _build_mapping_caches()
    
    if _ticker_to_asset_cache is None:
        return []
    
    return list(_ticker_to_asset_cache.keys())


def get_available_asset_ids() -> List[str]:
    """Get list of all available asset IDs from translation file.
    
    Returns:
        List of asset IDs
    """
    _build_mapping_caches()
    
    if _asset_to_ticker_cache is None:
        return []
    
    return list(_asset_to_ticker_cache.keys())


def clear_cache() -> None:
    """Clear the translation cache (useful for testing or reloading data)."""
    global _translation_cache, _asset_to_ticker_cache, _ticker_to_asset_cache
    _translation_cache = None
    _asset_to_ticker_cache = None
    _ticker_to_asset_cache = None
