# Quick Start Guide - After Restart

## 1. Start PostgreSQL

### Option A: Homebrew Service (Recommended - Auto-starts)
```bash
brew services start postgresql@18
```

### Option B: Check if Already Running
```bash
# Check status
brew services list | grep postgresql@18

# If it shows "started", you're good!
# If it shows "stopped", run:
brew services start postgresql@18
```

### Verify PostgreSQL is Running
```bash
# Quick test
/opt/homebrew/opt/postgresql@18/bin/pg_isready

# Or test connection
/opt/homebrew/opt/postgresql@18/bin/psql -d q23_ndx -c "SELECT version();"
```

## 2. Activate Conda Environment

```bash
# Activate conda base (if not already active)
conda activate base

# Then activate your qntdev environment
conda activate qntdev
```

### Or Use Your .command Files
The `.command` files automatically activate conda, so you can just:
- Double-click `Smart_Ingest_All.command` (it handles conda activation)

## 3. Quick Verification

```bash
# Test PostgreSQL
/opt/homebrew/opt/postgresql@18/bin/pg_isready

# Test conda environment
conda activate qntdev
python -c "import psycopg2; print('✅ PostgreSQL driver OK')"
python -c "import qnt; print('✅ Quantiacs OK')"
```

## 4. Run Your Scripts

Once both are running:

```bash
# Navigate to project
cd "/Users/svenbenson/Q23_QUANT_SYSTEM 2"

# Activate conda
conda activate qntdev

# Run ingestion
python scripts/smart_ingest_all.py
```

## Auto-Start PostgreSQL on Boot (Optional)

If you want PostgreSQL to start automatically on boot:

```bash
# It should already be set up, but verify:
brew services list | grep postgresql@18

# If it shows "started" with a checkmark, it will auto-start on boot
```

## Troubleshooting

### PostgreSQL Won't Start
```bash
# Check logs
tail -f /opt/homebrew/var/log/postgresql@18.log

# Try manual start
/opt/homebrew/opt/postgresql@18/bin/postgres -D /opt/homebrew/var/postgresql@18
```

### Conda Not Found
```bash
# Initialize conda (if needed)
source /opt/anaconda3/etc/profile.d/conda.sh

# Then activate
conda activate qntdev
```

### Database Connection Error
```bash
# Make sure PostgreSQL is running
brew services start postgresql@18

# Test connection
/opt/homebrew/opt/postgresql@18/bin/psql -d q23_ndx
```
