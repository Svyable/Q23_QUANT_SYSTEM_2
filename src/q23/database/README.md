# NDX Database Module

PostgreSQL database module for storing NDX (NASDAQ 100) historical market data locally.

## Overview

This module provides a complete solution for:
- Storing NDX market data in PostgreSQL
- Ingesting data from Quantiacs API
- Querying market data efficiently
- Validating data quality

## Setup

### 1. Install Dependencies

```bash
pip install psycopg2-binary pandas numpy xarray
```

Note: You also need the Quantiacs package (`qnt`) for data ingestion.

### 2. Configure Database

Set environment variables or use defaults:

```bash
export Q23_DB_HOST=localhost
export Q23_DB_PORT=5432
export Q23_DB_NAME=q23_ndx
export Q23_DB_USER=postgres
export Q23_DB_PASSWORD=your_password
```

Or edit `src/q23/shared/config.py` to set defaults in `DatabaseConfig`.

### 3. Create PostgreSQL Database

```bash
createdb q23_ndx
```

### 4. Initialize Schema

```bash
python scripts/init_ndx_db.py
```

This creates the following tables:
- `ndx_universe` - Stock metadata and membership
- `ndx_ohlcv` - Daily OHLCV data
- `ndx_ingestion_log` - Ingestion tracking

## Usage

### Ingest Data

Ingest 10 years of NDX data:

```bash
python scripts/ingest_ndx.py \
    --start-date 2015-01-01 \
    --end-date 2025-01-24
```

Options:
- `--force-refresh`: Re-ingest existing data
- `--chunk-days`: Trading days per chunk (default: 756 = ~3 years)
- `--host`, `--port`, `--dbname`, `--user`, `--password`: Override config

### Validate Data

Run data quality checks:

```bash
python scripts/validate_ndx_db.py
```

Options:
- `--start-date`, `--end-date`: Date range to validate
- `--sample-size`: Number of assets to validate (default: 10)

### Query Data (Python API)

```python
from q23.database import get_client

client = get_client()

# Get universe
universe = client.get_universe()
print(f"Found {len(universe)} NDX stocks")

# Get OHLCV for a stock
from datetime import date
ohlcv = client.get_ohlcv(
    asset_id="NAS:AAPL",
    start_date=date(2020, 1, 1),
    end_date=date(2024, 12, 31),
)

# Get cross-sectional data for a date
cross_section = client.get_cross_sectional(
    as_of_date=date(2024, 12, 31),
    fields=["close", "volume"],
)

# Get latest date
latest = client.get_latest_date()
print(f"Latest data date: {latest}")
```

## Database Schema

### ndx_universe

Stock metadata and NDX membership tracking.

- `asset_id` (PK): Quantiacs asset ID (e.g., "NAS:AAPL")
- `symbol`: Ticker symbol
- `name`: Company name
- `sector`: Industry sector
- `figi`: Financial Instrument Global Identifier
- `first_date`: First date in NDX
- `last_date`: Last date in NDX (NULL if active)
- `is_active`: Currently in NDX

### ndx_ohlcv

Daily OHLCV data (time-series optimized).

- `date` (PK): Trading date
- `asset_id` (PK, FK): Asset ID
- `open`, `high`, `low`, `close`: Prices
- `volume`: Trading volume
- `adj_open`, `adj_high`, `adj_low`, `adj_close`, `adj_volume`: Adjusted values
- `dividend`: Dividend amount
- `split_factor`: Stock split factor
- `is_liquid`, `is_stock`: Flags

Indexes:
- `idx_ndx_ohlcv_date`: For time-range queries
- `idx_ndx_ohlcv_asset_date`: For asset history
- `idx_ndx_ohlcv_date_asset`: For cross-sectional queries

### ndx_ingestion_log

Ingestion run tracking.

- `id`: Log entry ID
- `start_date`, `end_date`: Date range ingested
- `assets_count`: Number of assets processed
- `rows_inserted`: Number of OHLCV rows inserted
- `status`: 'success', 'partial', or 'failed'
- `error_message`: Error details if failed
- `duration_seconds`: Ingestion duration

## Performance

Expected performance:
- **Ingestion**: ~10 years in 20-30 minutes
- **Query**: Single asset 10-year history in < 100ms
- **Cross-sectional**: All assets on a date in < 500ms

## Data Volume

For 10 years of NDX data:
- **Universe**: ~230-250 stocks
- **Time range**: ~2,520 trading days
- **Total rows**: ~580,000 OHLCV records
- **Storage**: ~50-100 MB (with indexes)

## Future Integration

This module is designed for future integration with:
- qntlab.app website (via API layer)
- Strategy engine (replace Quantiacs API calls)
- Real-time updates (daily ingestion jobs)
