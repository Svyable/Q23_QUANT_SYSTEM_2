# PostgreSQL@18 Setup - Quick Commands

## Step 1: Add PostgreSQL to PATH (One-time setup)

Run this in your terminal:

```bash
echo 'export PATH="/opt/homebrew/opt/postgresql@18/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Or if you prefer to add it temporarily (just for this session):

```bash
export PATH="/opt/homebrew/opt/postgresql@18/bin:$PATH"
```

## Step 2: Create Database

After adding to PATH:

```bash
createdb q23_ndx
```

## Alternative: Use Full Path (No PATH changes needed)

If you don't want to modify PATH, you can use the full path:

```bash
/opt/homebrew/opt/postgresql@18/bin/createdb q23_ndx
```

## Step 3: Verify PostgreSQL is Running

```bash
# Check if service is running
brew services list | grep postgresql@18

# Test connection
/opt/homebrew/opt/postgresql@18/bin/psql -d q23_ndx -c "SELECT version();"
```

## Step 4: Initialize Database Schemas

Once the database is created, run (in your conda environment):

```bash
conda activate qntdev
cd "/Users/svenbenson/Q23_QUANT_SYSTEM 2"
python scripts/init_ndx_db.py
python scripts/init_crypto_db.py
```

## Step 5: Run Ingestion

```bash
python scripts/smart_ingest_all.py
```

## Troubleshooting

### If `createdb` still not found:
Use the full path:
```bash
/opt/homebrew/opt/postgresql@18/bin/createdb q23_ndx
```

### If connection fails:
Check PostgreSQL is running:
```bash
brew services list | grep postgresql@18
```

Start it if needed:
```bash
brew services start postgresql@18
```

### Check PostgreSQL port:
PostgreSQL@18 might use a different port. Check:
```bash
/opt/homebrew/opt/postgresql@18/bin/psql -l
```

If it works, the default port is 5432, which matches your config.
