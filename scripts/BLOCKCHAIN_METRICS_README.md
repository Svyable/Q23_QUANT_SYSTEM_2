# Blockchain Metrics Database Integration

This document describes the blockchain metrics integration that augments the cryptocurrency database with fundamental blockchain data from Quantiacs.

## Overview

The crypto database now supports both:
1. **Price Data (OHLCV)**: Daily cryptocurrency prices and volumes
2. **Blockchain Fundamentals**: Time-series blockchain metrics (e.g., miners revenue, hash rate, transaction volume)

## Available Metrics

To see all available blockchain metrics from Quantiacs:

```bash
python scripts/list_blockchain_metrics.py
```

This will:
- Fetch the list of available metrics from Quantiacs
- Display them in a table
- Save to `scripts/blockchain_metrics_list.csv` for reference

Example metrics include:
- `miners-revenue`: Bitcoin miners revenue
- `hash-rate`: Network hash rate
- `transaction-volume`: Transaction volume
- And many more...

## Database Schema

### New Tables

1. **`crypto_blockchain_metrics`**: Metadata about available metrics
   - `metric_id` (PK): Unique identifier (e.g., 'miners-revenue')
   - `name`: Human-readable name
   - `description`: Metric description
   - `unit`: Unit of measurement
   - `source`: Data source (default: 'blockchain.com')
   - `first_seen_date`, `last_seen_date`: Date range of available data
   - `is_active`: Whether metric is currently active

2. **`crypto_blockchain_data`**: Time-series metric values
   - `date` (PK): Date of observation
   - `metric_id` (PK, FK): Metric identifier
   - `value`: Metric value (DECIMAL(20,8) for precision)
   - Indexed on `date`, `metric_id`, and `(date, metric_id)`

3. **`crypto_blockchain_ingestion_log`**: Ingestion tracking
   - Tracks each ingestion run with status, row counts, and errors

## Setup

### 1. Initialize Schema

The blockchain schema is automatically created when you run:

```bash
python scripts/init_crypto_db.py
```

If you already have the crypto schema, it will add the blockchain tables without affecting existing data.

### 2. Update Metrics List

Fetch and store the list of available metrics:

```bash
python scripts/ingest_blockchain.py --update-metrics-list
```

Or it will be automatically updated when you run smart ingestion.

## Usage

### Ingest All Blockchain Metrics

Ingest all available blockchain metrics:

```bash
python scripts/ingest_blockchain.py --all
```

This will:
1. Fetch the list of available metrics
2. Ingest each metric's time-series data
3. Store everything in the database

### Ingest Specific Metric

Ingest a single metric:

```bash
python scripts/ingest_blockchain.py --metric miners-revenue
```

### Ingest with Date Range

```bash
python scripts/ingest_blockchain.py \
    --metric miners-revenue \
    --start-date 2015-01-01 \
    --end-date 2025-01-24
```

### Integrated Smart Ingestion

The blockchain metrics are now included in the master smart ingestion script:

```bash
python scripts/smart_ingest_all.py
```

This will ingest:
- NDX stock data
- Cryptocurrency price data (OHLCV)
- Blockchain metrics data

To skip blockchain ingestion:

```bash
python scripts/smart_ingest_all.py --skip-blockchain
```

To ingest specific metrics only:

```bash
python scripts/smart_ingest_all.py --blockchain-metrics miners-revenue hash-rate
```

## Querying Data

### List Available Metrics

```sql
SELECT metric_id, name, description, unit, first_seen_date, last_seen_date
FROM crypto_blockchain_metrics
WHERE is_active = TRUE
ORDER BY name;
```

### Get Time-Series for a Metric

```sql
SELECT date, value
FROM crypto_blockchain_data
WHERE metric_id = 'miners-revenue'
  AND date >= '2020-01-01'
  AND date <= '2024-12-31'
ORDER BY date;
```

### Join Price and Fundamental Data

```sql
-- Get Bitcoin price and miners revenue for the same dates
SELECT 
    p.date,
    p.close AS btc_price,
    b.value AS miners_revenue
FROM crypto_ohlcv p
JOIN crypto_blockchain_data b ON p.date = b.date
WHERE p.asset_id = 'BTC'
  AND b.metric_id = 'miners-revenue'
  AND p.date >= '2020-01-01'
ORDER BY p.date;
```

### Multiple Metrics

```sql
-- Get multiple metrics for the same dates
SELECT 
    date,
    MAX(CASE WHEN metric_id = 'miners-revenue' THEN value END) AS miners_revenue,
    MAX(CASE WHEN metric_id = 'hash-rate' THEN value END) AS hash_rate,
    MAX(CASE WHEN metric_id = 'transaction-volume' THEN value END) AS transaction_volume
FROM crypto_blockchain_data
WHERE metric_id IN ('miners-revenue', 'hash-rate', 'transaction-volume')
  AND date >= '2020-01-01'
GROUP BY date
ORDER BY date;
```

## Python API

```python
from q23.database import get_client
from q23.database.blockchain_ingestion import (
    ingest_blockchain_metrics_list,
    ingest_blockchain_data,
    fetch_blockchain_metrics_list,
)

client = get_client()

# Update metrics list
count = ingest_blockchain_metrics_list(client)

# Ingest specific metric
result = ingest_blockchain_data(
    client,
    metric_id='miners-revenue',
    start_date=date(2020, 1, 1),
    end_date=date.today(),
)

print(f"Inserted {result.rows_inserted} rows")
```

## Data Characteristics

- **Time Granularity**: Daily (end-of-day values)
- **Date Range**: Varies by metric (typically 2010+ for Bitcoin metrics)
- **Precision**: DECIMAL(20,8) for high precision values
- **Updates**: Metrics are updated as new data becomes available

## Integration with Strategies

You can now use blockchain fundamentals alongside price data in your Quantiacs strategies:

```python
import qnt.data as qndata

# Price data
crypto_prices = qndata.cryptodaily.load_data(...)

# Blockchain fundamentals (from your local DB)
# Query miners revenue, hash rate, etc. to enhance your strategy
```

## Files Created

1. **`src/q23/database/blockchain_ingestion.py`**: Core ingestion logic
2. **`scripts/list_blockchain_metrics.py`**: List available metrics
3. **`scripts/ingest_blockchain.py`**: Ingest blockchain data
4. **Schema extensions in `src/q23/database/crypto_schema.py`**: Database tables

## Notes

- Blockchain metrics are **not asset-specific** - they're network-level metrics (e.g., Bitcoin network hash rate, not per-asset)
- Some metrics may have gaps or missing dates
- The ingestion uses `ON CONFLICT DO NOTHING` so it's safe to re-run
- Smart ingestion will only fetch missing date ranges

## Troubleshooting

If you get errors about missing schema:

```bash
python scripts/init_crypto_db.py
```

If metrics list is empty:

```bash
python scripts/ingest_blockchain.py --update-metrics-list
```

If a specific metric fails:

```bash
# Check the metric ID is correct
python scripts/list_blockchain_metrics.py

# Try ingesting with verbose logging
python scripts/ingest_blockchain.py --metric <metric-id> --start-date 2020-01-01
```
