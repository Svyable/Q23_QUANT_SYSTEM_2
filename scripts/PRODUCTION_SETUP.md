# Production Setup Guide

## Quick Start

### 1. Ensure .env File is Configured

Your `.env` file should contain:
```bash
API_KEY=84f1097a-e19c-4227-bba3-5be55812a45e
MARKETSTACK_API_KEY=fbaa2b5406d67968b93af34afc694f70

# Database (optional - defaults provided)
Q23_DB_HOST=localhost
Q23_DB_PORT=5432
Q23_DB_NAME=q23_ndx
Q23_DB_USER=postgres
Q23_DB_PASSWORD=your_password

# Conda (optional - defaults provided)
Q23_CONDA_SH=/opt/anaconda3/etc/profile.d/conda.sh
Q23_CONDA_ENV=qntdev
```

### 2. Initialize Database

```bash
# Create database
createdb q23_ndx

# Initialize schemas
python scripts/init_ndx_db.py
python scripts/init_crypto_db.py
```

### 3. Run Smart Ingestion

**Option A: Double-click (easiest)**
- Double-click `Smart_Ingest_All.command` in Finder

**Option B: Command line**
```bash
python scripts/smart_ingest_all.py
```

**Option C: Individual databases**
```bash
python scripts/smart_ingest_ndx.py
python scripts/smart_ingest_crypto.py
```

## What's Production-Ready

### ✅ Environment Management
- All scripts load `.env` file automatically
- API_KEY validated before Quantiacs calls
- Database credentials from environment
- Proper error messages if configuration missing

### ✅ Error Handling
- Retry logic with exponential backoff (3 attempts)
- Connection pool retry on failures
- Partial success tracking
- Transaction safety (rollback on error)

### ✅ Data Alignment
- Schema matches Quantiacs API exactly
- Only stores fields that exist
- Proper field name mappings
- Date ranges derived from actual data

### ✅ Performance
- Memory-efficient chunked processing
- Fast batch inserts (1000 rows/batch)
- Connection pooling (5 base + 10 overflow)
- Optimized indexes for queries
- Smart gap detection (only pulls missing data)

### ✅ Monitoring
- Ingestion logs in database
- Progress reporting
- Error tracking
- Duration metrics

## Verification

After setup, verify everything works:

```bash
# 1. Test database connection
python -c "from q23.database import get_client; client = get_client(); print('✅ Connected' if client.test_connection() else '❌ Failed')"

# 2. Check schemas exist
python scripts/init_ndx_db.py  # Should say "already exists"
python scripts/init_crypto_db.py  # Should say "already exists"

# 3. Run smart ingestion (will show gaps or "already complete")
python scripts/smart_ingest_all.py

# 4. Validate data
python scripts/validate_ndx_db.py
```

## Production Deployment

### Scheduled Daily Updates

Add to crontab:
```bash
# Daily at 6 AM
0 6 * * * cd /path/to/Q23_QUANT_SYSTEM\ 2 && /path/to/python scripts/smart_ingest_all.py >> /var/log/q23_ingestion.log 2>&1
```

### Health Monitoring

Check ingestion status:
```sql
SELECT 
    start_date,
    end_date,
    status,
    rows_inserted,
    duration_seconds,
    created_at
FROM ndx_ingestion_log
ORDER BY created_at DESC
LIMIT 5;
```

Check data freshness:
```sql
SELECT MAX(date) as latest_date FROM ndx_ohlcv;
SELECT MAX(date) as latest_date FROM crypto_ohlcv;
```

## Troubleshooting

### "API_KEY is not set"
- Check `.env` file exists in project root
- Verify `API_KEY=...` line is present
- Check file permissions

### "ModuleNotFoundError: No module named 'q23'"
- Verify conda environment is activated
- Check PYTHONPATH includes `src/` directory
- Run: `export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"`

### "Failed to connect to database"
- Check PostgreSQL is running: `pg_isready`
- Verify connection settings in `.env`
- Test: `psql -h localhost -U postgres -d q23_ndx`

### "Schema does not exist"
- Run: `python scripts/init_ndx_db.py`
- Verify database name is correct
- Check user has CREATE TABLE permissions

## Best Practices

1. **Run daily** - Keeps gaps small, updates fast
2. **Monitor logs** - Check `ndx_ingestion_log` table regularly
3. **Validate data** - Run validation scripts weekly
4. **Backup database** - Regular backups before major updates
5. **Use smart ingestion** - Only pulls missing data, safe to run repeatedly
