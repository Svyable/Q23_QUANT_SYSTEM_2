# Fix: PostgreSQL User Configuration

## Issue
PostgreSQL@18 on macOS uses your **macOS username** (`svenbenson`) as the default superuser, not `postgres`.

## Solution Applied

I've updated your `.env` file to use the correct username:
```bash
Q23_DB_USER=svenbenson
```

And updated the default in `config.py` to automatically use your macOS username if not specified.

## About the Quantiacs Notices

The notices about `DATA_BASE_URL`, `CACHE_RETENTION`, and `CACHE_DIR` are from the Quantiacs package itself - they're just informational messages about their API endpoint defaults. You can ignore them - they don't affect your local PostgreSQL database.

## Verify Database Connection

Now try again:

```bash
conda activate qntdev
cd "/Users/svenbenson/Q23_QUANT_SYSTEM 2"
python scripts/init_ndx_db.py
```

## If You Still Get Errors

If you want to create a `postgres` user instead, you can:

```bash
# Connect to PostgreSQL
/opt/homebrew/opt/postgresql@18/bin/psql postgres

# Create postgres user
CREATE USER postgres WITH SUPERUSER;

# Then update .env to use:
# Q23_DB_USER=postgres
```

But using your macOS username (svenbenson) is the standard approach and should work fine.
