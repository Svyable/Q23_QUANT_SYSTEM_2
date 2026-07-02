# Smart Data Ingestion System

## Overview

The smart ingestion system intelligently checks what data already exists in the database and only pulls missing date ranges. This makes it safe to run repeatedly without duplicating work or wasting API calls.

## Key Features

### 🧠 Intelligence Layer
- **Gap Detection**: Automatically identifies missing date ranges
- **Incremental Updates**: Only fetches new data, not existing data
- **Idempotent**: Safe to run multiple times
- **Progress Tracking**: Shows exactly what gaps will be filled

### 📊 Dual Database Support
- **NDX (NASDAQ 100)**: Stock market data
- **Cryptocurrency**: Crypto market data
- **Unified Interface**: Master script runs both intelligently

### 🚀 Best Practices
- Connection pooling for performance
- Chunked processing for memory efficiency
- Error handling with partial success support
- Comprehensive logging and progress reporting

## Quick Start

**🎯 Default Behavior: Gets ALL available data from start date to TODAY!**

By default, all scripts automatically use **today's date** as the end date, so you get the maximum data available without specifying anything.

### Double-Click Execution (macOS)

Simply double-click any of these files in Finder:

1. **`Smart_Ingest_All.command`** - Runs both NDX and Crypto ingestion (2015-01-01 to today / 2014-01-01 to today)
2. **`Smart_Ingest_NDX.command`** - Runs only NDX ingestion (2015-01-01 to today)
3. **`Smart_Ingest_Crypto.command`** - Runs only Crypto ingestion (2014-01-01 to today)

These files will:
- Automatically activate your conda environment
- Run the smart ingestion (gets ALL data to today by default)
- Keep the terminal open to show results

### Command Line Execution

```bash
# Run all databases
python scripts/smart_ingest_all.py

# Run only NDX
python scripts/smart_ingest_ndx.py

# Run only Crypto
python scripts/smart_ingest_crypto.py

# Default behavior: Gets ALL data from start date to TODAY
# No need to specify end dates - it automatically uses today!
python scripts/smart_ingest_all.py

# With custom date ranges (if you want to limit to a specific end date)
python scripts/smart_ingest_all.py \
    --ndx-start 2015-01-01 \
    --ndx-end 2025-01-24 \
    --crypto-start 2014-01-01 \
    --crypto-end 2025-01-24

# Force refresh (re-ingest everything)
python scripts/smart_ingest_all.py --force-refresh

# Skip one database
python scripts/smart_ingest_all.py --skip-ndx
python scripts/smart_ingest_all.py --skip-crypto
```

## How It Works

### 1. Gap Detection

The system queries the database to find:
- **Earliest date** with data
- **Latest date** with data
- **Gaps before** existing data (if target start is earlier)
- **Gaps after** existing data (if target end is later)

### 2. Smart Ingestion

For each detected gap:
- Fetches data from Quantiacs API
- Processes in memory-efficient chunks
- Inserts only new records (uses `ON CONFLICT DO NOTHING`)
- Tracks progress and errors

### 3. Result Reporting

Shows:
- Number of gaps found
- Rows inserted per gap
- Overall status (success/partial/failed)
- Any errors encountered

## Example Output

```
==========================================
Smart Data Ingestion - All Databases
==========================================

==========================================
NDX DATA INGESTION
==========================================
Found 1 date gap(s) in NDX data:
  2025-01-20 to 2025-01-24
Ingesting gap: 2025-01-20 to 2025-01-24
Processing 1 date chunks
  Inserted 1,150 rows from time slice 2025-01-20 to 2025-01-24
NDX Status: success
NDX Rows inserted: 1,150

==========================================
CRYPTOCURRENCY DATA INGESTION
==========================================
Found 1 date gap(s) in crypto data:
  2025-01-20 to 2025-01-24
Assets: all available
Ingesting gap: 2025-01-20 to 2025-01-24
Crypto Status: success
Crypto Rows inserted: 500

==========================================
OVERALL SUMMARY
==========================================
All ingestion tasks completed successfully!
```

## Configuration

### Environment Variables

Set in `.env` or environment:
- `Q23_DB_HOST` - Database host (default: localhost)
- `Q23_DB_PORT` - Database port (default: 5432)
- `Q23_DB_NAME` - Database name (default: q23_ndx)
- `Q23_DB_USER` - Database user (default: postgres)
- `Q23_DB_PASSWORD` - Database password

### Default Date Ranges (🎯 Gets ALL Available Data!)

- **NDX**: 2015-01-01 to **today** (automatic - no end date needed!)
- **Crypto**: 2014-01-01 to **today** (automatic - no end date needed!)

**The system automatically uses today's date as the end date**, so you get the maximum available data. Just run the script without any end date arguments!

Override with command-line arguments if you want to limit to a specific end date.

## Advanced Usage

### Custom Asset Lists (Crypto)

```bash
python scripts/smart_ingest_crypto.py --assets BTC ETH SOL
```

### Force Refresh

```bash
# Re-ingest all data (ignores existing data)
python scripts/smart_ingest_all.py --force-refresh
```

### Chunk Size Tuning

```bash
# Larger chunks = faster but more memory
python scripts/smart_ingest_ndx.py --chunk-days 756  # ~3 years
python scripts/smart_ingest_crypto.py --chunk-days 730  # ~2 years
```

## Error Handling

The system handles:
- **Missing schemas**: Warns and skips gracefully
- **API failures**: Logs error and continues with next gap
- **Partial success**: Reports status as "partial" with error details
- **Interruptions**: Handles Ctrl+C gracefully

## Performance

### Typical Performance
- **Gap detection**: < 1 second
- **Daily updates**: 1-5 minutes (depending on gap size)
- **Full initial load**: 20-30 minutes for 10 years

### Optimization Tips
1. Run daily to keep gaps small
2. Use `--force-refresh` sparingly (only when needed)
3. Adjust `--chunk-days` based on available memory
4. Run NDX and Crypto separately if one fails

## Troubleshooting

### "Schema does not exist"
```bash
# Initialize NDX schema
python scripts/init_ndx_db.py

# Initialize Crypto schema
python scripts/init_crypto_db.py
```

### "Failed to connect to database"
- Check PostgreSQL is running
- Verify connection settings in `.env`
- Test connection: `psql -h localhost -U postgres -d q23_ndx`

### "No gaps found but data seems incomplete"
- Check date ranges in database: `SELECT MIN(date), MAX(date) FROM ndx_ohlcv`
- Use `--force-refresh` to re-ingest if needed
- Check ingestion logs: `SELECT * FROM ndx_ingestion_log ORDER BY created_at DESC`

## Integration with Existing System

The smart ingestion system is designed to work alongside:
- Existing `ingest_ndx.py` and `ingest_crypto.py` scripts
- Manual data loading
- Scheduled cron jobs

All scripts use the same database schema and are fully compatible.

## Best Practices

1. **Run Daily**: Set up a cron job to run `Smart_Ingest_All.command` daily
2. **Monitor Logs**: Check `ndx_ingestion_log` and `crypto_ingestion_log` tables
3. **Validate Data**: Run `validate_ndx_db.py` periodically
4. **Backup Database**: Regular backups before large updates
5. **Use Force Refresh Sparingly**: Only when data corruption is suspected

## Future Enhancements

Potential improvements:
- Internal gap detection (missing dates within range)
- Parallel gap processing
- Progress bars for long-running ingestions
- Email/Slack notifications on completion
- Automatic retry on transient failures
