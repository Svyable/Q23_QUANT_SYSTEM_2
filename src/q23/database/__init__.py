"""q23.database

PostgreSQL database module for local NDX market data storage.

This module provides:
- Database schema definitions and migrations
- Database client with connection pooling
- Data ingestion pipeline from Quantiacs
- Query utilities for market data access
"""

from q23.database.schema import (
    create_schema,
    drop_schema,
    schema_exists,
)
from q23.database.client import (
    DatabaseClient,
    get_client,
)
from q23.database.ingestion import (
    ingest_ndx_data,
    IngestionResult,
)
from q23.database.validation import (
    validate_all,
    validate_date_continuity,
    validate_price_relationships,
    validate_volume,
    validate_duplicates,
    ValidationResult,
)
from q23.database.crypto_schema import (
    create_crypto_schema,
    drop_crypto_schema,
    crypto_schema_exists,
)
from q23.database.crypto_ingestion import (
    ingest_crypto_data,
    CryptoIngestionResult,
)
from q23.database.smart_ingestion import (
    smart_ingest_ndx,
    smart_ingest_crypto,
    get_ndx_date_gaps,
    get_crypto_date_gaps,
)
from q23.database.crypto_schema import (
    create_blockchain_schema,
    blockchain_schema_exists,
)
from q23.database.blockchain_ingestion import (
    ingest_blockchain_metrics_list,
    ingest_blockchain_data,
    fetch_blockchain_metrics_list,
    BlockchainIngestionResult,
)

__all__ = [
    "create_schema",
    "drop_schema",
    "schema_exists",
    "DatabaseClient",
    "get_client",
    "ingest_ndx_data",
    "IngestionResult",
    "validate_all",
    "validate_date_continuity",
    "validate_price_relationships",
    "validate_volume",
    "validate_duplicates",
    "ValidationResult",
    "create_crypto_schema",
    "drop_crypto_schema",
    "crypto_schema_exists",
    "ingest_crypto_data",
    "CryptoIngestionResult",
    "smart_ingest_ndx",
    "smart_ingest_crypto",
    "get_ndx_date_gaps",
    "get_crypto_date_gaps",
    "create_blockchain_schema",
    "blockchain_schema_exists",
    "ingest_blockchain_metrics_list",
    "ingest_blockchain_data",
    "fetch_blockchain_metrics_list",
    "BlockchainIngestionResult",
]
