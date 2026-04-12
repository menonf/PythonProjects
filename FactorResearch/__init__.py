"""
Refinitiv Security Master Ingestion Package
============================================

A modular, configurable pipeline for fetching equity data from Refinitiv
and upserting it into a security master database.

Modules:
    - config: Dataclass-based configuration for all pipeline parameters
    - filters: OData filter expression builder
    - api_client: Refinitiv API interactions with retry logic
    - transformers: Data cleaning and transformation
    - db_operations: Database matching and upsert operations

Quick Start:
    from refinitiv_ingest import run_ingestion, IngestionConfig, FilterCriteria
    
    # Use default configuration (Indian equities)
    run_ingestion()
    
    # Custom configuration
    config = IngestionConfig(
        filter_criteria=FilterCriteria(
            asset_type="equity",
            countries=["United States", "United Kingdom"]
        )
    )
    run_ingestion(config)
"""

from .config import (
    IngestionConfig,
    APIConfig,
    DatabaseConfig,
    SearchFields,
    EnrichmentFields,
    FilterCriteria,
    PartitionConfig,
    IdentifierPriority,
    # Pre-built configurations
    indian_equities_config,
    us_equities_config,
    global_equities_config,
    test_config,
)

from .filters import (
    FilterBuilder,
    build_filter_from_criteria,
    get_partition_filters,
)

from .api_client import RefinitivClient

from .transformers import (
    SecurityDataTransformer,
    clean_value,
    safe_column,
)

from .db_operations import (
    LookupMapBuilder,
    IdentifierMatcher,
    SecurityRepository,
    SecurityMasterUpserter,
)

from .pipeline import run_ingestion, IngestionPipeline


__all__ = [
    # Config
    "IngestionConfig",
    "APIConfig",
    "DatabaseConfig",
    "SearchFields",
    "EnrichmentFields",
    "FilterCriteria",
    "PartitionConfig",
    "IdentifierPriority",
    "indian_equities_config",
    "us_equities_config",
    "global_equities_config",
    "test_config",
    # Filters
    "FilterBuilder",
    "build_filter_from_criteria",
    "get_partition_filters",
    # API
    "RefinitivClient",
    # Transformers
    "SecurityDataTransformer",
    "clean_value",
    "safe_column",
    # DB Operations
    "LookupMapBuilder",
    "IdentifierMatcher",
    "SecurityRepository",
    "SecurityMasterUpserter",
    # Pipeline
    "run_ingestion",
    "IngestionPipeline",
]
