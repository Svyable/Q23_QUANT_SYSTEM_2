# Quick SQL Queries for Single Symbols

## Python Script (Easiest)

```bash
# Query NDX stock
python scripts/query_symbol.py AAPL

# Query crypto
python scripts/query_symbol.py BTC --crypto

# More rows
python scripts/query_symbol.py AAPL --limit 100
```

## Direct SQL Commands

### NDX Stock (via psql)

```bash
# Connect to database
/opt/homebrew/opt/postgresql@18/bin/psql -d q23_ndx

# Then run:
```

```sql
-- Single symbol - recent data
SELECT 
    o.date,
    u.symbol,
    u.name,
    o.open,
    o.high,
    o.low,
    o.close,
    o.volume,
    o.dividend
FROM ndx_ohlcv o
JOIN ndx_universe u ON o.asset_id = u.asset_id
WHERE UPPER(u.symbol) = 'AAPL'
ORDER BY o.date DESC
LIMIT 50;

-- Single symbol - date range
SELECT 
    o.date,
    u.symbol,
    o.open,
    o.high,
    o.low,
    o.close,
    o.volume
FROM ndx_ohlcv o
JOIN ndx_universe u ON o.asset_id = u.asset_id
WHERE UPPER(u.symbol) = 'AAPL'
  AND o.date >= '2024-01-01'
  AND o.date <= '2024-12-31'
ORDER BY o.date ASC;

-- Single symbol - summary stats
SELECT 
    u.symbol,
    u.name,
    COUNT(*) as total_days,
    MIN(o.date) as first_date,
    MAX(o.date) as last_date,
    AVG(o.close) as avg_close,
    MAX(o.high) as max_high,
    MIN(o.low) as min_low
FROM ndx_ohlcv o
JOIN ndx_universe u ON o.asset_id = u.asset_id
WHERE UPPER(u.symbol) = 'AAPL'
GROUP BY u.symbol, u.name;
```

### Cryptocurrency (via psql)

```sql
-- Single crypto symbol - recent data
SELECT 
    o.date,
    u.symbol,
    u.name,
    o.open,
    o.high,
    o.low,
    o.close,
    o.volume
FROM crypto_ohlcv o
JOIN crypto_universe u ON o.asset_id = u.asset_id
WHERE UPPER(u.symbol) = 'BTC'
ORDER BY o.date DESC
LIMIT 50;

-- Single crypto - date range
SELECT 
    o.date,
    u.symbol,
    o.open,
    o.high,
    o.low,
    o.close,
    o.volume
FROM crypto_ohlcv o
JOIN crypto_universe u ON o.asset_id = u.asset_id
WHERE UPPER(u.symbol) = 'BTC'
  AND o.date >= '2024-01-01'
  AND o.date <= '2024-12-31'
ORDER BY o.date ASC;
```

## One-Liner SQL (Terminal)

```bash
# NDX - Single symbol
/opt/homebrew/opt/postgresql@18/bin/psql -d q23_ndx -c "
SELECT o.date, u.symbol, o.open, o.high, o.low, o.close, o.volume
FROM ndx_ohlcv o
JOIN ndx_universe u ON o.asset_id = u.asset_id
WHERE UPPER(u.symbol) = 'AAPL'
ORDER BY o.date DESC
LIMIT 20;
"

# Crypto - Single symbol
/opt/homebrew/opt/postgresql@18/bin/psql -d q23_ndx -c "
SELECT o.date, u.symbol, o.open, o.high, o.low, o.close, o.volume
FROM crypto_ohlcv o
JOIN crypto_universe u ON o.asset_id = u.asset_id
WHERE UPPER(u.symbol) = 'BTC'
ORDER BY o.date DESC
LIMIT 20;
"
```

## List Available Symbols

```sql
-- NDX symbols
SELECT symbol, name FROM ndx_universe ORDER BY symbol;

-- Crypto symbols
SELECT symbol, name FROM crypto_universe ORDER BY symbol;
```

## Quick Stats for Symbol

```sql
-- NDX stats
SELECT 
    u.symbol,
    COUNT(*) as days,
    MIN(o.date) as first,
    MAX(o.date) as last,
    ROUND(AVG(o.close), 2) as avg_close
FROM ndx_ohlcv o
JOIN ndx_universe u ON o.asset_id = u.asset_id
WHERE UPPER(u.symbol) = 'AAPL'
GROUP BY u.symbol;
```
