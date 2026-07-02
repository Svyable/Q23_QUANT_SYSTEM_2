# Quantiacs + MarketStack Data Integration Setup

**Goal**: Run Quantiacs strategies locally by stitching recent OHLCV data from MarketStack API onto Quantiacs historical data, eliminating the 2-3 trading day lag while maintaining xarray DataArray compatibility.

## Prerequisites & Setup

### Dependencies
```bash
pip install qnt requests pandas xarray python-dotenv pytz
```

### Environment Configuration
Create a `.env` file in your project root:
```
MARKETSTACK_API_KEY=your_api_key_here
```

### MarketStack API Notes
- **Free tier**: 1000 requests/month, 100/day
- **Symbols**: SPX = `XNAS:SPX`, NDX = `XNAS:NDX` (verify with your MarketStack endpoint or symbol search)
- **Timezone**: Returns exchange local time (typically EST for US markets)
- **Rate limiting**: Implement exponential backoff for production use

## Data Fetching Functions

### Fetch Recent MarketStack Data
```python
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import pytz
import time

def fetch_marketstack_recent(symbols=['XNAS:SPX', 'XNAS:NDX'],
                           days_back=10,
                           max_retries=3):
    """
    Fetch recent EOD data from MarketStack with error handling and rate limiting.

    Args:
        symbols: List of MarketStack symbol identifiers
        days_back: Number of days to fetch (with overlap buffer)
        max_retries: Maximum API retry attempts

    Returns:
        pd.DataFrame: Cleaned OHLCV data with datetime index
    """
    load_dotenv()
    api_key = os.getenv('MARKETSTACK_API_KEY')
    if not api_key:
        raise ValueError("MARKETSTACK_API_KEY not found in environment")

    base_url = 'http://api.marketstack.com/v1/eod'
    eastern = pytz.timezone('US/Eastern')

    # Calculate date range with buffer
    end_date = datetime.now(eastern).date()
    start_date = end_date - timedelta(days=days_back)

    all_data = []

    for symbol in symbols:
        for attempt in range(max_retries):
            try:
                params = {
                    'access_key': api_key,
                    'symbols': symbol,
                    'date_from': start_date.strftime('%Y-%m-%d'),
                    'date_to': end_date.strftime('%Y-%m-%d'),
                    'limit': 1000  # MarketStack max per request
                }

                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()

                data = response.json().get('data', [])
                if not data:
                    print(f"Warning: No data returned for {symbol}")
                    continue

                # Convert to DataFrame
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize('US/Eastern')
                df['symbol'] = symbol.split(':')[-1]  # Extract SPX/NDX from IEXG:SPX
                df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'adj_close']]

                all_data.append(df)
                break  # Success, exit retry loop

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to fetch {symbol} after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

    if not all_data:
        raise ValueError("No data fetched from MarketStack")

    # Combine all symbols
    combined_df = pd.concat(all_data, ignore_index=True)

    # Pivot to wide format (date index, OHLCV columns per symbol)
    pivoted = combined_df.pivot(
        index='date',
        columns='symbol',
        values=['open', 'high', 'low', 'close', 'volume', 'adj_close']
    )

    # Flatten column names
    pivoted.columns = [f'{field}_{symbol}' for field, symbol in pivoted.columns]

    return pivoted.sort_index()
```

## Custom Data Loader

### Complete Integration Function
```python
import qnt.data as qndata
import xarray as xr
from datetime import timedelta
import numpy as np

def create_enhanced_data_loader(marketstack_symbols=['XNAS:SPX', 'XNAS:NDX']):
    """
    Factory function that returns a data loader with MarketStack integration.

    Args:
        marketstack_symbols: Symbols to fetch from MarketStack

    Returns:
        function: Data loader compatible with qnt.backtest()
    """
    def load_data(min_date="2000-01-01"):
        """
        Load Quantiacs data and stitch in recent MarketStack data.

        Args:
            min_date: Start date for historical data (YYYY-MM-DD)

        Returns:
            xr.DataArray: Combined dataset with dimensions [time, asset, field]
        """
        try:
            # Load Quantiacs historical data
            print("Loading Quantiacs historical data...")
            # For SPX/NDX specifically, use NDX universe:
            qnt_data = qndata.stocks.load_ndx_data(min_date=min_date)
            # OR for broader S&P500 universe:
            # qnt_data = qndata.stocks.load_data(min_date=min_date, assets=['SPX', 'NDX'])
            # Check qnt_data.asset.values post-load to confirm available assets

            # Fetch recent MarketStack data
            print("Fetching recent data from MarketStack...")
            recent_df = fetch_marketstack_recent(marketstack_symbols)

            # Convert to xarray format matching Quantiacs structure
            recent_xr = convert_marketstack_to_xarray(recent_df, qnt_data)

            # Merge datasets
            merged_data = merge_datasets(qnt_data, recent_xr)

            print(f"Data merged successfully. Final shape: {merged_data.shape}")
            return merged_data

        except Exception as e:
            print(f"Error in data loading: {e}")
            # Fallback to Quantiacs data only
            print("Falling back to Quantiacs data only...")
            return qndata.stocks.load_ndx_data(min_date=min_date)

    return load_data

def convert_marketstack_to_xarray(df, qnt_template):
    """
    Convert MarketStack DataFrame to xarray format matching Quantiacs.

    Args:
        df: MarketStack DataFrame with flattened OHLCV columns
        qnt_template: Sample Quantiacs DataArray to match structure

    Returns:
        xr.DataArray: Formatted data array
    """
    # Extract symbol names from column suffixes
    symbols = []
    for col in df.columns:
        if '_' in col:
            symbol = col.split('_')[-1]
            if symbol not in symbols:
                symbols.append(symbol)

    # Create time coordinate
    times = df.index.values

    # Create data array with correct dimensions
    fields = ['open', 'high', 'low', 'close', 'vol']  # Match Quantiacs field names
    data_arrays = []

    for symbol in symbols:
        symbol_data = []
        for field in fields:
            col_name = f'{field}_{symbol}'
            if field == 'vol':
                col_name = f'volume_{symbol}'  # MarketStack uses 'volume'

            if col_name in df.columns:
                values = df[col_name].fillna(method='ffill').values
            else:
                print(f"Warning: Column {col_name} not found, using NaN")
                values = np.full(len(df), np.nan)

            symbol_data.append(values)

        # Stack fields for this symbol
        symbol_array = np.stack(symbol_data, axis=0)
        data_arrays.append(symbol_array)

    # Combine all symbols
    combined_data = np.stack(data_arrays, axis=1)  # [time, asset, field]

    # Create xarray DataArray
    data_array = xr.DataArray(
        combined_data,
        dims=['time', 'asset', 'field'],
        coords={
            'time': times,
            'asset': symbols,
            'field': fields
        }
    )

    return data_array

def merge_datasets(qnt_data, recent_data):
    """
    Merge Quantiacs historical data with recent MarketStack data.

    Args:
        qnt_data: Quantiacs xarray DataArray
        recent_data: MarketStack xarray DataArray

    Returns:
        xr.DataArray: Merged dataset
    """
    # Find overlap period and trim Quantiacs data
    if len(recent_data.time) > 0:
        recent_start = recent_data.time.min()
        # Keep Quantiacs data up to (but not including) recent start date
        qnt_trimmed = qnt_data.sel(time=slice(None, recent_start.values - np.timedelta64(1, 'D')))
    else:
        qnt_trimmed = qnt_data

    # Concatenate along time dimension
    merged = xr.concat([qnt_trimmed, recent_data], dim='time')

    # Sort by time and forward fill any gaps
    merged = merged.sortby('time')

    # Handle any NaN values by forward filling along time
    merged = merged.ffill('time')

    return merged
```

## Usage Example

```python
import qnt

# Create enhanced data loader
data_loader = create_enhanced_data_loader()

# Use in backtest
qnt.backtest(
    competition_type="stocks",
    data_loader=data_loader,
    strategy=my_strategy
)
```

## Optimized Usage Pattern

```python
# One-liner for your workflow
data_loader = create_enhanced_data_loader(['XNAS:SPX', 'XNAS:NDX'])
weights = qnt.backtest(competition_type="stocks", data_loader=data_loader, strategy=my_strategy)

# Quick validation
validate_merged_data(weights.data)
print(f"Latest date: {weights.data.time.isel(time=-1).values}")  # Should be recent trading day
```

## Validation & Testing

### Data Quality Checks
```python
def validate_merged_data(data):
    """Validate the merged dataset quality."""
    print("Data validation:")
    print(f"Shape: {data.shape}")
    print(f"Time range: {data.time.min().values} to {data.time.max().values}")
    print(f"Assets: {list(data.asset.values)}")
    print(f"Fields: {list(data.field.values)}")

    # Check for gaps
    time_diff = np.diff(data.time.values.astype('datetime64[D]'))
    gaps = np.where(time_diff > np.timedelta64(1, 'D'))[0]
    if len(gaps) > 0:
        print(f"Warning: Found {len(gaps)} time gaps > 1 day")

    # Check data completeness
    nan_count = data.isnull().sum().values
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values found")

    return True
```

### Test Strategy Continuity
```python
# Test strategy outputs before/after stitch point
stitch_date = "2024-01-15"  # Example date
pre_data = data.sel(time=slice(None, stitch_date))
post_data = data.sel(time=slice(stitch_date, None))

# Run strategy on both periods and compare outputs
pre_weights = my_strategy(pre_data)
post_weights = my_strategy(post_data)
```

## Production Polish

### Caching Recent Data
```python
import pandas as pd

def fetch_with_cache(symbols=['XNAS:SPX', 'XNAS:NDX'], cache_file='marketstack_cache.parquet'):
    """Cache recent data to avoid repeated API calls."""
    try:
        # Check if cache exists and is from today
        if os.path.exists(cache_file):
            cached_data = pd.read_parquet(cache_file)
            cache_date = pd.to_datetime(cached_data.index.max().date())
            today = pd.Timestamp.now().date()

            if cache_date == today:
                print("Using cached data")
                return cached_data

        # Fetch fresh data
        data = fetch_marketstack_recent(symbols)
        data.to_parquet(cache_file)
        return data

    except Exception as e:
        print(f"Cache error: {e}")
        return fetch_marketstack_recent(symbols)
```

### Multi-Index Handling
```python
# If using adjusted prices, map MarketStack adj_close to Quantiacs close
def convert_marketstack_to_xarray_adjusted(df, qnt_template, use_adjusted=False):
    # ... (same as before, but modify field mapping)
    fields = ['open', 'high', 'low', 'close', 'vol']

    for symbol in symbols:
        for field in fields:
            if field == 'close' and use_adjusted:
                col_name = f'adj_close_{symbol}'  # Use adjusted close
            elif field == 'vol':
                col_name = f'volume_{symbol}'
            else:
                col_name = f'{field}_{symbol}'
            # ... rest of logic
```

### Asset Mapping
```python
# Robust symbol alignment
asset_map = {'SPX': 'SPX', 'NDX': 'NDX'}  # Add more mappings as needed

def validate_asset_alignment(qnt_assets, marketstack_assets):
    """Ensure asset names match between datasets."""
    missing = set(marketstack_assets) - set(qnt_assets)
    if missing:
        print(f"Warning: MarketStack assets not in Quantiacs: {missing}")
    return len(missing) == 0
```

## Important Considerations

1. **Symbol Mapping**: Verify MarketStack symbols (`XNAS:SPX`, `XNAS:NDX`) via endpoint or symbol search
2. **Field Name Mapping**: Quantiacs expects 'vol' but MarketStack returns 'volume' - handled correctly in `convert_marketstack_to_xarray()`
3. **Quantiacs Data Loader Selection**: Use `load_ndx_data()` for NDX universe or `load_data()` with specific assets. Check `qnt_data.asset.values` post-load
4. **Timezone Alignment**: MarketStack returns EST, ensure consistency with strategy expectations
5. **Data Quality**: Always validate merged data for continuity
6. **API Limits**: Monitor MarketStack usage to avoid rate limits
7. **Error Handling**: Implement fallbacks for API failures
8. **Backtesting**: Test thoroughly with different date ranges to ensure stability
9. **Caching**: Implement data caching for production use
10. **Adjusted Prices**: Consider `adj_close` if strategy uses adjusted prices