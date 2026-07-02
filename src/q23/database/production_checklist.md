# Production Readiness Checklist

## Environment Configuration

### ✅ Completed
- [x] `.env` file loading in all scripts
- [x] API_KEY validation before Quantiacs calls
- [x] Database connection configuration via environment variables
- [x] Proper Python path setup
- [x] Conda environment activation in .command files

### Required Environment Variables

**Quantiacs API:**
```bash
API_KEY=your-quantiacs-api-key
```

**Database (optional - defaults provided):**
```bash
Q23_DB_HOST=localhost
Q23_DB_PORT=5432
Q23_DB_NAME=q23_ndx
Q23_DB_USER=postgres
Q23_DB_PASSWORD=your_password
```

**Conda (optional - defaults provided):**
```bash
Q23_CONDA_SH=/opt/anaconda3/etc/profile.d/conda.sh
Q23_CONDA_ENV=qntdev
```

## Production Enhancements

### 1. Error Handling & Resilience

**Current Status:**
- ✅ Connection pooling with retry logic
- ✅ Partial success tracking (logs what succeeded)
- ✅ Graceful degradation on API failures
- ✅ Transaction rollback on errors

**Recommendations:**
- Add exponential backoff for Quantiacs API calls
- Implement circuit breaker pattern for repeated failures
- Add health check endpoint for monitoring

### 2. Monitoring & Observability

**Current Status:**
- ✅ Ingestion logging to database tables
- ✅ Progress reporting in logs
- ✅ Error message tracking

**Recommendations:**
- Add metrics export (Prometheus/StatsD)
- Structured logging (JSON format)
- Alerting on ingestion failures
- Dashboard for ingestion status

### 3. Performance Optimization

**Current Status:**
- ✅ Chunked processing for memory efficiency
- ✅ Batch inserts for database writes
- ✅ Indexed queries for fast lookups
- ✅ Connection pooling

**Recommendations:**
- Parallel chunk processing (multiprocessing)
- Use COPY FROM for bulk inserts (faster than INSERT)
- Partition tables by year for very large datasets
- Add query result caching for metadata

### 4. Data Integrity

**Current Status:**
- ✅ Primary key constraints prevent duplicates
- ✅ Foreign key constraints maintain referential integrity
- ✅ Data validation scripts
- ✅ Smart gap detection

**Recommendations:**
- Add CHECK constraints for price relationships
- Implement data checksums for corruption detection
- Regular automated validation runs
- Backup strategy documentation

### 5. Security

**Current Status:**
- ✅ API keys in .env (not in code)
- ✅ Database credentials via environment variables
- ✅ No hardcoded secrets

**Recommendations:**
- Use secrets management (AWS Secrets Manager, HashiCorp Vault)
- Encrypt database connections (SSL/TLS)
- Rotate API keys regularly
- Audit log access

### 6. Scalability

**Current Status:**
- ✅ Handles 10 years of data efficiently
- ✅ Memory-efficient chunked processing
- ✅ Connection pooling for concurrent access

**Recommendations:**
- Horizontal scaling: Read replicas for queries
- Vertical scaling: Larger connection pools
- Partitioning: Time-based table partitioning
- Archival: Move old data to archive tables

## Deployment Checklist

### Pre-Production
- [ ] Test with production-like data volume
- [ ] Load test database queries
- [ ] Verify backup/restore procedures
- [ ] Document rollback procedures
- [ ] Set up monitoring alerts
- [ ] Configure log aggregation
- [ ] Test disaster recovery

### Production Deployment
- [ ] Initialize database schema
- [ ] Configure environment variables
- [ ] Set up scheduled ingestion (cron/systemd)
- [ ] Configure monitoring dashboards
- [ ] Set up alerting rules
- [ ] Document runbooks
- [ ] Train operations team

## Operational Procedures

### Daily Operations
1. Run smart ingestion: `Smart_Ingest_All.command` or cron job
2. Monitor ingestion logs: `SELECT * FROM ndx_ingestion_log ORDER BY created_at DESC LIMIT 10;`
3. Check for errors: Review logs for failed/partial status
4. Validate data: Run `validate_ndx_db.py` weekly

### Weekly Maintenance
1. Review ingestion performance metrics
2. Check database size and growth
3. Validate data quality
4. Review error logs

### Monthly Maintenance
1. Database backup verification
2. Performance tuning (analyze slow queries)
3. Index maintenance (VACUUM ANALYZE)
4. Capacity planning review

## Performance Benchmarks

### Expected Performance (Production)
- **Ingestion**: 10 years in 20-30 minutes
- **Query**: Single asset 10-year history < 100ms
- **Cross-sectional**: All assets on date < 500ms
- **Gap detection**: < 1 second
- **Daily updates**: 1-5 minutes

### Monitoring Thresholds
- Alert if ingestion > 60 minutes
- Alert if query > 1 second
- Alert if error rate > 5%
- Alert if database size > 10GB

## Troubleshooting Guide

### Common Issues

**1. "API_KEY is not set"**
- Check `.env` file exists and contains `API_KEY=...`
- Verify .env file is in project root
- Check file permissions

**2. "ModuleNotFoundError: No module named 'q23'"**
- Verify PYTHONPATH includes `src/` directory
- Check conda environment is activated
- Verify `q23` package structure

**3. "Failed to connect to database"**
- Check PostgreSQL is running: `pg_isready`
- Verify connection settings in `.env`
- Test connection: `psql -h localhost -U postgres -d q23_ndx`

**4. "Schema does not exist"**
- Run: `python scripts/init_ndx_db.py`
- Check database name is correct
- Verify user has CREATE TABLE permissions

**5. "No data returned from Quantiacs"**
- Verify API_KEY is valid
- Check Quantiacs service status
- Review API rate limits
- Check date range is valid

## Production Runbook

### Initial Setup
```bash
# 1. Create database
createdb q23_ndx

# 2. Initialize schemas
python scripts/init_ndx_db.py
python scripts/init_crypto_db.py

# 3. Initial data load
python scripts/smart_ingest_all.py
```

### Daily Updates
```bash
# Run smart ingestion (only pulls new data)
python scripts/smart_ingest_all.py

# Or use double-click:
# Smart_Ingest_All.command
```

### Scheduled Job (cron)
```bash
# Add to crontab (runs daily at 6 AM)
0 6 * * * cd /path/to/project && /path/to/python scripts/smart_ingest_all.py >> /var/log/q23_ingestion.log 2>&1
```

### Health Check Script
```python
# Check database health
from q23.database import get_client
client = get_client()
if client.test_connection():
    latest = client.get_latest_date()
    print(f"Database healthy. Latest data: {latest}")
else:
    print("Database connection failed!")
```

## Security Best Practices

1. **Never commit .env file** - Already in .gitignore ✅
2. **Use strong database passwords**
3. **Limit database user permissions** (read/write only, no DROP)
4. **Enable SSL for database connections** (production)
5. **Rotate API keys regularly**
6. **Audit log access** (who accessed what data)

## Backup Strategy

### Database Backups
```bash
# Daily backup
pg_dump q23_ndx > backup_$(date +%Y%m%d).sql

# Restore
psql q23_ndx < backup_20250124.sql
```

### Recommended Schedule
- **Full backup**: Weekly
- **Incremental**: Daily
- **Retention**: 30 days full, 7 days incremental

## Performance Tuning

### Database Configuration
```sql
-- Optimize for time-series queries
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '128MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
```

### Query Optimization
- Use EXPLAIN ANALYZE for slow queries
- Monitor index usage
- Consider materialized views for common queries
- Partition large tables by year

## Monitoring Metrics

### Key Metrics to Track
1. **Ingestion rate**: Rows per second
2. **Error rate**: Failed ingestions / total
3. **Data freshness**: Days behind current date
4. **Database size**: Growth rate
5. **Query performance**: P95 latency
6. **API usage**: Quantiacs API calls per day

### Alerting Rules
- Ingestion failure → Immediate alert
- Data > 2 days stale → Warning
- Error rate > 5% → Warning
- Database size > 10GB → Info
- Query P95 > 1s → Warning
