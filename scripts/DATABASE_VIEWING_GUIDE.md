# Viewing Your Database Data

## Option 1: Quick Python Script (Easiest - In Cursor)

I've created a simple viewer script you can run:

```bash
# View database summary
python scripts/view_database.py

# View specific table
python scripts/view_database.py --table ndx_ohlcv --limit 100

# View universe
python scripts/view_database.py --table ndx_universe
```

This shows data directly in your terminal/Cursor.

## Option 2: GUI Tools (Recommended for Exploration)

### A. TablePlus (Best for macOS - $99, but free trial)
- **Download**: https://tableplus.com/
- **Why**: Beautiful, fast, native macOS app
- **Setup**: 
  1. Install TablePlus
  2. Click "Create a new connection"
  3. Select "PostgreSQL"
  4. Enter:
     - Host: `localhost`
     - Port: `5432`
     - User: `svenbenson` (your macOS username)
     - Database: `q23_ndx`
     - Password: (leave empty)

### B. DBeaver (Free, Cross-platform)
- **Download**: https://dbeaver.io/download/
- **Why**: Free, powerful, open-source
- **Setup**: 
  1. Install DBeaver
  2. New Database Connection → PostgreSQL
  3. Enter connection details (same as above)

### C. Postico (macOS only - $49, free trial)
- **Download**: https://eggerapps.at/postico/
- **Why**: Simple, elegant, macOS-native
- **Setup**: Same connection details as TablePlus

### D. pgAdmin (Official PostgreSQL GUI - Free)
- **Download**: https://www.pgadmin.org/download/
- **Why**: Official tool, very powerful
- **Setup**: More complex but full-featured

## Option 3: Command Line (psql)

```bash
# Connect to database
/opt/homebrew/opt/postgresql@18/bin/psql -d q23_ndx

# Then run SQL queries:
SELECT COUNT(*) FROM ndx_universe;
SELECT * FROM ndx_ohlcv LIMIT 10;
SELECT * FROM ndx_universe ORDER BY symbol;
```

## Option 4: VS Code Extension

If you use VS Code (or Cursor supports extensions):
- Install "PostgreSQL" extension by Chris Kolkman
- Connect to your database
- Run queries in the editor

## Quick SQL Queries

Once connected, try these:

```sql
-- View all NDX stocks
SELECT * FROM ndx_universe ORDER BY symbol;

-- View recent OHLCV data
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
WHERE o.date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY o.date DESC, u.symbol
LIMIT 100;

-- Count data by asset
SELECT 
    u.symbol,
    COUNT(*) as rows,
    MIN(o.date) as first_date,
    MAX(o.date) as last_date
FROM ndx_ohlcv o
JOIN ndx_universe u ON o.asset_id = u.asset_id
GROUP BY u.symbol
ORDER BY u.symbol;

-- Database summary
SELECT 
    'NDX Universe' as table_name,
    COUNT(*) as row_count
FROM ndx_universe
UNION ALL
SELECT 
    'NDX OHLCV' as table_name,
    COUNT(*) as row_count
FROM ndx_ohlcv;
```

## My Recommendation

1. **Quick checks**: Use `python scripts/view_database.py` (fastest)
2. **Data exploration**: Use **TablePlus** or **DBeaver** (best GUI experience)
3. **Command line**: Use `psql` for quick queries

## Connection Details Summary

- **Host**: `localhost`
- **Port**: `5432`
- **User**: `svenbenson` (your macOS username)
- **Database**: `q23_ndx`
- **Password**: (empty for local development)
