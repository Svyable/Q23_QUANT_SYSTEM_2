# Cryptocurrency Database Module

PostgreSQL database module for storing cryptocurrency historical market data locally.

## Overview

This module extends the NDX database system to support cryptocurrency data from Quantiacs. It provides:
- Storage for crypto OHLCV data in PostgreSQL
- Ingesting data from Quantiacs `cryptodaily.load_data()` API
- Querying cryptocurrency market data efficiently

## Setup

### 1. Initialize Crypto Schema

```bash
python scripts/init_crypto_db.py
```

This creates the following tables:
- `crypto_universe` - Cryptocurrency metadata
- `crypto_ohlcv` - Daily OHLCV data
- `crypto_ingestion_log` - Ingestion tracking

## Usage

### Ingest Cryptocurrency Data

Ingest data for specific cryptocurrencies:

```bash
python scripts/ingest_crypto.py \
    --start-date 2014-01-01 \
    --end-date 2025-01-24 \
    --assets BTC ETH
```

Or ingest all available cryptocurrencies:

```bash
python scripts/ingest_crypto.py \
    --start-date 2014-01-01 \
    --end-date 2025-01-24
```

Options:
- `--assets`: List of asset symbols (e.g., BTC ETH). Default: all available
- `--force-refresh`: Re-ingest existing data
- `--chunk-days`: Calendar days per chunk (default: 730 = ~2 years)

### Query Data (Python API)

```python
from q23.database import get_client

client = get_client()

# Get crypto universe
with client.get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM crypto_universe WHERE is_active = TRUE")
        universe = cur.fetchall()

# Get OHLCV for a cryptocurrency
from datetime import date
with client.get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT date, open, high, low, close, volume, is_liquid
            FROM crypto_ohlcv
            WHERE asset_id = 'BTC' 
            AND date >= %s AND date <= %s
            ORDER BY date
        """, (date(2020, 1, 1), date(2024, 12, 31)))
        btc_data = cur.fetchall()
```

## Database Schema

### crypto_universe

Cryptocurrency metadata.

- `asset_id` (PK): Asset symbol (e.g., "BTC", "ETH")
- `symbol`: Ticker symbol (same as asset_id)
- `name`: Cryptocurrency name
- `first_date`: First date with data
- `last_date`: Last date with data (NULL if active)
- `is_active`: Currently active

### crypto_ohlcv

Daily OHLCV data (time-series optimized).

- `date` (PK): Trading date
- `asset_id` (PK, FK): Asset symbol
- `open`, `high`, `low`, `close`: Prices (DECIMAL(18,8) for precision)
- `volume`: Trading volume (DECIMAL(20,8))
- `is_liquid`: Liquidity flag

**Note**: Crypto prices use higher precision (18,8) than stocks (12,4) to handle very small or very large values.

Indexes:
- `idx_crypto_ohlcv_date`: For time-range queries
- `idx_crypto_ohlcv_asset_date`: For asset history
- `idx_crypto_ohlcv_date_asset`: For cross-sectional queries

### crypto_ingestion_log

Ingestion run tracking (same structure as NDX).

## Differences from NDX

1. **No Universe List API**: Cryptocurrency data doesn't have a separate `load_list()` function. Assets are discovered from the data itself.

2. **Simpler Schema**: No dividends, splits, or adjusted prices (crypto doesn't have these concepts).

3. **Higher Precision**: Prices stored as DECIMAL(18,8) instead of DECIMAL(12,4) to handle wide price ranges.

4. **Calendar Days**: Crypto data uses calendar days (not just trading days), so chunking uses calendar days instead of trading days.

5. **DataArray vs Dataset**: Quantiacs crypto API returns a `DataArray` with dimensions `(field, time, asset)` instead of a `Dataset`.

## Data Volume

For 6 years of crypto data (typical Quantiacs default):
- **Universe**: ~50-100+ cryptocurrencies
- **Time range**: ~2,190 calendar days
- **Total rows**: ~100 assets × 2,190 days = ~219,000 rows
- **Storage**: ~30-50 MB (with indexes)

## Performance

Expected performance:
- **Ingestion**: ~6 years in 10-20 minutes
- **Query**: Single asset 6-year history in < 100ms
- **Cross-sectional**: All assets on a date in < 500ms

## Example: Load BTC and ETH Data

```python
import qnt.data as qndata

# Load data from Quantiacs
data = qndata.cryptodaily.load_data(
    assets=['BTC', 'ETH'],
    tail=365*5  # 5 years
)

# Extract fields
open_prices = data.sel(field='open')
close_prices = data.sel(field='close')
is_liquid = data.sel(field='is_liquid')

# Filter to liquid only
liquid_open = open_prices * is_liquid
```

## Integration with Strategy Engine

Future integration points:
- Add crypto data loader to `data_loader.py`
- Support crypto in strategy universe selection
- Extend factor computation for crypto-specific metrics
- Add crypto benchmarks
