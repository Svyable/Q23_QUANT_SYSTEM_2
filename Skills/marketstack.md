# Marketstack API Integration

## Overview

Marketstack is integrated into Q23 to supplement Quantiacs data with recent market days (last X days) that may be missing. This enables real-time analysis for the next trading day while maintaining backward compatibility with existing data loading and caching mechanisms.

## API Endpoints

### End-of-Day (EOD) Data

The primary endpoint used is the EOD (End-of-Day) data endpoint:

**Base URL**: `https://api.marketstack.com/v2/eod`

**Required Parameters**:
- `access_key`: Your Marketstack API key
- `symbols`: Comma-separated list of ticker symbols (e.g., "AAPL,MSFT,GOOGL")

**Optional Parameters**:
- `date_from`: Start date in YYYY-MM-DD format
- `date_to`: End date in YYYY-MM-DD format
- `sort`: Sort order ("ASC" or "DESC", default "DESC")
- `limit`: Maximum number of records per symbol
- `offset`: Pagination offset

**Example Request**:
```
GET https://api.marketstack.com/v2/eod?access_key=YOUR_KEY&symbols=AAPL&date_from=2025-01-01&date_to=2025-01-10&sort=DESC
```

**Response Format**:
```json
{
  "pagination": {
    "limit": 100,
    "offset": 0,
    "count": 10,
    "total": 10
  },
  "data": [
    {
      "open": 150.25,
      "high": 152.30,
      "low": 149.80,
      "close": 151.50,
      "volume": 45000000,
      "adj_high": 152.30,
      "adj_low": 149.80,
      "adj_close": 151.50,
      "adj_open": 150.25,
      "adj_volume": 45000000,
      "split_factor": 1.0,
      "dividend": 0.0,
      "symbol": "AAPL",
      "exchange": "NASDAQ",
      "date": "2025-01-10T00:00:00+0000"
    }
  ]
}
```

## Authentication

The API uses an access key for authentication. The API key is configured in `q23.shared.config.MarketstackConfig`:

- **Default API Key**: `fbaa2b5406d67968b93af34afc694f70`
- **Environment Variable**: `MARKETSTACK_API_KEY` (overrides default)

## Configuration

Marketstack integration is configured via `MarketstackConfig` in `src/q23/shared/config.py`:

```python
@dataclass
class MarketstackConfig:
    API_KEY: str = "fbaa2b5406d67968b93af34afc694f70"
    ENABLED: bool = True
    DEFAULT_LOOKBACK_DAYS: int = 5
    BASE_URL: str = "https://api.marketstack.com/v2"
    RATE_LIMIT_DELAY: float = 0.1  # seconds between requests
```

### Configuration Options

- **API_KEY**: Marketstack API access key
- **ENABLED**: Enable/disable Marketstack integration (default: True)
- **DEFAULT_LOOKBACK_DAYS**: Default number of recent days to check for missing data (default: 5)
- **BASE_URL**: Base URL for Marketstack API
- **RATE_LIMIT_DELAY**: Delay between API requests in seconds (default: 0.1)

## Usage in Q23

### Basic Usage

Marketstack integration is automatically enabled when loading market data:

```python
from q23.strategy.data_loader import load_market_data

# Load data with Marketstack integration (default)
bundle = load_market_data(
    min_date="2024-01-01",
    fill_recent_days=5,  # Fill last 5 days from Marketstack if missing
    use_marketstack=True,  # Enable Marketstack (default)
)
```

### Disabling Marketstack

To disable Marketstack integration:

```python
bundle = load_market_data(
    min_date="2024-01-01",
    use_marketstack=False,  # Disable Marketstack
)
```

### Custom Lookback Days

Specify custom number of days to look back:

```python
bundle = load_market_data(
    min_date="2024-01-01",
    fill_recent_days=10,  # Check last 10 days
)
```

## Symbol Mapping

Quantiacs uses asset IDs (e.g., "tts-99960993") while Marketstack uses ticker symbols (e.g., "AAPL"). The system automatically maps between these formats using `id-translation.csv`.

### Mapping Process

1. **Asset ID to Ticker**: Quantiacs asset IDs are mapped to Marketstack tickers using `id-translation.csv`
   - Format: `server_id,user_id`
   - Example: `tts-99960993,NAS:AAPL`
   - Ticker extracted from `user_id` by splitting on `:`

2. **Ticker to Asset ID**: Reverse mapping for converting Marketstack responses back to Quantiacs asset IDs

### Symbol Mapper Functions

```python
from q23.strategy.symbol_mapper import (
    asset_id_to_ticker,
    ticker_to_asset_id,
    batch_convert_assets,
)

# Convert single asset ID
ticker = asset_id_to_ticker("tts-99960993")  # Returns "AAPL"

# Convert single ticker
asset_id = ticker_to_asset_id("AAPL")  # Returns "tts-99960993"

# Batch conversion
asset_ids = ["tts-99960993", "tts-98772733"]
ticker_map = batch_convert_assets(asset_ids)
# Returns: {"tts-99960993": "AAPL", "tts-98772733": "HUBB"}
```

## Data Gap Detection

The system automatically detects missing recent trading days:

### Gap Detection Logic

1. **Get Latest Date**: Extract latest date from Quantiacs dataset
2. **Compare to Today**: Compare latest date to today's date
3. **Calculate Missing Days**: Identify missing trading days (Monday-Friday only)
4. **Check Lookback Window**: Only fetch if gap is within `lookback_days` limit

### Gap Detector Functions

```python
from q23.strategy.data_gap_detector import (
    get_latest_quantiacs_date,
    should_fetch_marketstack,
    get_missing_date_range,
)

# Get latest date from dataset
latest_date = get_latest_quantiacs_date(dataset)

# Check if Marketstack fetch is needed
needs_fetch = should_fetch_marketstack(dataset, lookback_days=5)

# Get date range for missing days
date_range = get_missing_date_range(dataset, lookback_days=5)
# Returns: ("2025-01-08", "2025-01-10") or None
```

## Data Merging

Marketstack data is converted to xarray format and merged with Quantiacs data:

### Merging Strategy

1. **Convert to xarray**: Marketstack DataFrame → xarray Dataset
2. **Align Assets**: Ensure both datasets have same asset dimension
3. **Concatenate**: Concatenate along time dimension
4. **Remove Duplicates**: Prefer Marketstack data for overlapping dates
5. **Sort**: Sort by time ascending
6. **Validate**: Ensure merged dataset has correct structure

### Data Merger Functions

```python
from q23.strategy.data_merger import (
    marketstack_to_xarray,
    merge_datasets,
    validate_merged_data,
)

# Convert Marketstack DataFrame to xarray
marketstack_ds = marketstack_to_xarray(marketstack_df, asset_ids)

# Merge datasets
merged_ds = merge_datasets(quantiacs_ds, marketstack_ds)

# Validate merged data
is_valid = validate_merged_data(merged_ds)
```

## Error Handling

The system handles errors gracefully:

- **Marketstack API Failures**: Falls back to Quantiacs data only (logs warning)
- **Missing Symbol Mappings**: Skips assets without mappings (logs warning)
- **Partial Marketstack Data**: Merges what's available, fills missing with NaN
- **Rate Limiting**: Implements exponential backoff retry
- **Network Errors**: Retries with backoff, then falls back to Quantiacs only

## Rate Limits

Marketstack API has rate limits. The client enforces rate limiting:

- **Default Delay**: 0.1 seconds between requests
- **Configurable**: Set via `MarketstackConfig.RATE_LIMIT_DELAY`
- **Retry Logic**: Exponential backoff on failures (3 retries by default)

## Caching

Marketstack data is cached along with Quantiacs data:

- **Cache Location**: `.cache/data/YYYYMMDD/`
- **Cache Key**: Same as Quantiacs cache (e.g., "SPX")
- **Refresh Detection**: Cache checks if Marketstack refresh is needed
- **Daily Invalidation**: Cache automatically invalidated daily

### Cache Refresh Detection

```python
from q23.strategy.data_cache import is_marketstack_refresh_needed

# Check if cached dataset needs Marketstack refresh
needs_refresh = is_marketstack_refresh_needed(cached_ds, lookback_days=5)
```

## Data Flow

```
load_market_data()
  └─> load_quantiacs_stocks()
       ├─> Load from Quantiacs API (or cache)
       ├─> Detect missing recent days
       ├─> Fetch from Marketstack (if needed)
       │    ├─> Convert asset IDs to tickers
       │    ├─> Call Marketstack API
       │    └─> Convert response to xarray Dataset
       ├─> Merge datasets
       └─> Cache merged dataset
```

## Example: Complete Workflow

```python
from q23.strategy.data_loader import load_market_data
from q23.strategy.marketstack_client import MarketstackClient
from q23.strategy.symbol_mapper import batch_convert_assets

# Load market data with Marketstack integration
bundle = load_market_data(
    min_date="2024-01-01",
    fill_recent_days=5,
    use_marketstack=True,
)

# Access merged data
dataset = bundle.data
print(f"Latest date: {dataset.time.values[-1]}")
print(f"Assets: {len(dataset.asset.values)}")

# Direct Marketstack client usage (if needed)
client = MarketstackClient()
tickers = ["AAPL", "MSFT", "GOOGL"]
marketstack_df = client.fetch_latest_eod(tickers, limit=5)
print(marketstack_df)
```

## Best Practices

1. **Use Default Settings**: Default configuration works well for most use cases
2. **Monitor Rate Limits**: Adjust `RATE_LIMIT_DELAY` if hitting rate limits
3. **Check Symbol Mappings**: Ensure `id-translation.csv` is up to date
4. **Handle Errors Gracefully**: System falls back to Quantiacs data on errors
5. **Cache Management**: Let the system handle caching automatically

## Troubleshooting

### Marketstack API Errors

- **401 Unauthorized**: Check API key configuration
- **429 Too Many Requests**: Increase `RATE_LIMIT_DELAY`
- **500 Server Error**: Retry logic handles this automatically

### Missing Symbol Mappings

- Check `id-translation.csv` for asset ID
- Verify ticker symbol format (should match Marketstack format)
- Update translation file if needed

### Data Quality Issues

- Validate merged datasets using `validate_merged_data()`
- Check for NaN values in Marketstack data
- Verify date ranges match expected trading days

## Future Enhancements

Potential future improvements:

- Support for intraday data
- Support for other Marketstack endpoints (tickers, exchanges)
- Automatic symbol mapping updates
- Marketstack data quality validation
- Configurable data source priority (Quantiacs vs Marketstack)
