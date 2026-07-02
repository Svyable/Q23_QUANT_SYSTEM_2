# PostgreSQL Setup for Q23 Database

## Quick Installation (macOS)

### Option 1: Homebrew (Recommended)

```bash
# Install PostgreSQL
brew install postgresql@16

# Start PostgreSQL service
brew services start postgresql@16

# Verify it's running
pg_isready
```

### Option 2: Postgres.app (GUI - Easiest)

1. Download from: https://postgresapp.com/
2. Install and open Postgres.app
3. Click "Initialize" to create a new server
4. The default connection will be:
   - Host: localhost
   - Port: 5432
   - User: your macOS username
   - Database: (create one)

### Option 3: Conda (If you prefer conda)

```bash
conda activate qntdev
conda install -c conda-forge postgresql
```

## Create Database

After PostgreSQL is installed and running:

```bash
# Create the database
createdb q23_ndx

# Or if using Postgres.app, connect and run:
# psql postgres
# CREATE DATABASE q23_ndx;
```

## Verify Installation

```bash
# Check if PostgreSQL is running
pg_isready

# Test connection
psql -h localhost -U $(whoami) -d q23_ndx -c "SELECT version();"
```

## Start PostgreSQL (if not running)

### Homebrew:
```bash
brew services start postgresql@16
```

### Postgres.app:
- Just open the app - it starts automatically

### Manual start (if installed via Homebrew):
```bash
pg_ctl -D /opt/homebrew/var/postgresql@16 start
```

## Connection Settings

Default settings (update in `.env` if different):
- Host: localhost
- Port: 5432
- User: your macOS username (or 'postgres')
- Database: q23_ndx
- Password: (usually none for local development)

## Troubleshooting

### "Connection refused"
- PostgreSQL is not running
- Start it: `brew services start postgresql@16` or open Postgres.app

### "Database does not exist"
- Create it: `createdb q23_ndx`

### "Password authentication failed"
- For local development, you may need to configure `pg_hba.conf` to allow local connections without password
- Or set password in `.env`: `Q23_DB_PASSWORD=your_password`

### "Permission denied"
- Make sure your user has permission to create databases
- Or use: `createdb -U postgres q23_ndx`

## Next Steps

Once PostgreSQL is running:

1. Create database:
   ```bash
   createdb q23_ndx
   ```

2. Initialize schemas:
   ```bash
   python scripts/init_ndx_db.py
   python scripts/init_crypto_db.py
   ```

3. Run ingestion:
   ```bash
   python scripts/smart_ingest_all.py
   ```
