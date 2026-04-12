#!/usr/bin/env python
"""
Usage Examples for Refinitiv Security Master Ingestion
=======================================================

This file demonstrates various ways to use the refactored ingestion package.
"""

from refinitiv_ingest import (
    # Core entry points
    run_ingestion,
    IngestionPipeline,
    # Configuration classes
    IngestionConfig,
    APIConfig,
    DatabaseConfig,
    FilterCriteria,
    PartitionConfig,
    SearchFields,
    # Pre-built configurations
    indian_equities_config,
    us_equities_config,
    global_equities_config,
    test_config,
    # Filter utilities
    FilterBuilder,
    build_filter_from_criteria,
    # Setup
    setup_logging,
)


# ---------------------------------------------------------------------------
# Example 1: Simple Usage with Defaults (Indian Equities)
# ---------------------------------------------------------------------------

def example_default_run():
    """Run with default configuration (Indian equities)."""
    setup_logging()
    stats = run_ingestion()
    print(f"Completed: {stats}")


# ---------------------------------------------------------------------------
# Example 2: Using Pre-built Configurations
# ---------------------------------------------------------------------------

def example_prebuilt_configs():
    """Use pre-built configuration functions."""
    setup_logging()
    
    # Option A: Indian equities
    config = indian_equities_config()
    
    # Option B: US equities
    # config = us_equities_config()
    
    # Option C: Global equities (all countries)
    # config = global_equities_config()
    
    # Option D: Test run (limited pages)
    # config = test_config(max_pages=2)
    
    stats = run_ingestion(config)
    print(f"Completed: {stats}")


# ---------------------------------------------------------------------------
# Example 3: Custom Filter Criteria
# ---------------------------------------------------------------------------

def example_custom_filters():
    """Create custom filter configurations."""
    setup_logging()
    
    # Filter by multiple countries
    config = IngestionConfig(
        filter_criteria=FilterCriteria(
            asset_type="equity",
            countries=["United States", "United Kingdom", "Germany"],
        )
    )
    
    stats = run_ingestion(config)
    print(f"Completed: {stats}")


def example_currency_filter():
    """Filter by specific currencies."""
    setup_logging()
    
    config = IngestionConfig(
        filter_criteria=FilterCriteria(
            asset_type="equity",
            countries=["India"],
            currencies=["INR"],  # Only INR-denominated securities
        )
    )
    
    stats = run_ingestion(config)
    print(f"Completed: {stats}")


def example_exchange_filter():
    """Filter by specific exchanges."""
    setup_logging()
    
    config = IngestionConfig(
        filter_criteria=FilterCriteria(
            asset_type="equity",
            exchanges=["NSE", "BSE"],  # National Stock Exchange, Bombay Stock Exchange
        )
    )
    
    stats = run_ingestion(config)
    print(f"Completed: {stats}")


def example_isin_prefix():
    """Filter by ISIN country prefix."""
    setup_logging()
    
    config = IngestionConfig(
        filter_criteria=FilterCriteria(
            asset_type="equity",
            isin_prefix="US",  # US-issued securities
        ),
        partition=PartitionConfig(
            partition_field="TickerSymbol",  # Partition by ticker instead of RIC
        )
    )
    
    stats = run_ingestion(config)
    print(f"Completed: {stats}")


def example_combined_filters():
    """Combine multiple filter criteria."""
    setup_logging()
    
    config = IngestionConfig(
        filter_criteria=FilterCriteria(
            asset_type="equity",
            countries=["Japan"],
            currencies=["JPY"],
            exchanges=["TYO"],  # Tokyo Stock Exchange
            # You can also add custom OData expressions:
            # custom_filter="IssuerCommonName ne 'Test Company'"
        )
    )
    
    stats = run_ingestion(config)
    print(f"Completed: {stats}")


# ---------------------------------------------------------------------------
# Example 4: Custom Partitioning Strategy
# ---------------------------------------------------------------------------

def example_custom_partitioning():
    """Customize how large requests are partitioned."""
    setup_logging()
    
    config = IngestionConfig(
        filter_criteria=FilterCriteria(
            asset_type="equity",
            countries=["United States"],
        ),
        partition=PartitionConfig(
            enabled=True,
            partition_field="TickerSymbol",  # Partition by ticker
            partition_values=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            use_startswith=True,
        )
    )
    
    stats = run_ingestion(config)
    print(f"Completed: {stats}")


def example_no_partitioning():
    """Disable partitioning for small datasets."""
    setup_logging()
    
    config = IngestionConfig(
        filter_criteria=FilterCriteria(
            asset_type="equity",
            countries=["Luxembourg"],  # Small market
        ),
        partition=PartitionConfig(
            enabled=False,  # No partitioning needed
        )
    )
    
    stats = run_ingestion(config)
    print(f"Completed: {stats}")


# ---------------------------------------------------------------------------
# Example 5: API Configuration
# ---------------------------------------------------------------------------

def example_api_config():
    """Customize API behavior."""
    setup_logging()
    
    config = IngestionConfig(
        api=APIConfig(
            page_size=5000,         # Smaller pages
            gics_batch_size=2000,   # Smaller enrichment batches
            max_api_retries=5,      # More retries
            retry_sleep_sec=10,     # Longer pause between retries
            max_universe_pages=10,  # Limit for testing
        ),
        filter_criteria=FilterCriteria(
            asset_type="equity",
            countries=["India"],
        )
    )
    
    stats = run_ingestion(config)
    print(f"Completed: {stats}")


# ---------------------------------------------------------------------------
# Example 6: Using the FilterBuilder Directly
# ---------------------------------------------------------------------------

def example_filter_builder():
    """Build filters programmatically using FilterBuilder."""
    
    # Fluent API for building complex filters
    builder = FilterBuilder()
    filter_expr = (
        builder
        .with_asset_type("equity")
        .with_countries(["India", "Japan"])
        .with_currencies(["INR", "JPY"])
        .with_partition("RIC", "A")
        .build()
    )
    
    print(f"Generated filter: {filter_expr}")
    # Output: AssetType eq 'equity' and (RCSExchangeCountryLeaf eq 'India' or 
    #         RCSExchangeCountryLeaf eq 'Japan') and (RCSCurrencyLeaf eq 'INR' or 
    #         RCSCurrencyLeaf eq 'JPY') and startswith(RIC, 'A')


# ---------------------------------------------------------------------------
# Example 7: Custom Search Fields
# ---------------------------------------------------------------------------

def example_custom_search_fields():
    """Customize which fields are fetched in the search."""
    setup_logging()
    
    # Only fetch specific fields (reduces API payload)
    search_fields = SearchFields(
        ticker_symbol=True,
        ric=True,
        isin=True,
        exchange_country=True,
        # Disable fields you don't need:
        sedol=False,
        cusip=False,
        issuer_common_name=True,
        currency=True,
        asset_category=False,
        asset_class=False,
        exchange_code=True,
    )
    
    config = IngestionConfig(
        search_fields=search_fields,
        filter_criteria=FilterCriteria(
            asset_type="equity",
            countries=["India"],
        )
    )
    
    # Check what fields will be requested
    print(f"Search fields: {search_fields.get_select_string()}")
    
    stats = run_ingestion(config)
    print(f"Completed: {stats}")


# ---------------------------------------------------------------------------
# Example 8: Using the Pipeline Directly for More Control
# ---------------------------------------------------------------------------

def example_direct_pipeline():
    """Use IngestionPipeline directly for more control."""
    import logging
    
    setup_logging(level=logging.DEBUG)  # More verbose logging
    
    config = IngestionConfig(
        filter_criteria=FilterCriteria(
            asset_type="equity",
            countries=["India"],
        ),
        api=APIConfig(
            max_universe_pages=2,  # Test mode
        )
    )
    
    pipeline = IngestionPipeline(config)
    
    # You can access components directly if needed
    print(f"Using vendor: {config.database.vendor}")
    print(f"Filter select: {config.search_fields.get_select_string()}")
    
    # Run the pipeline
    stats = pipeline.run()
    
    # Process results
    print("\n=== Ingestion Complete ===")
    print(f"RICs fetched: {stats['rics_fetched']}")
    print(f"Securities matched: {stats['securities_matched']}")
    print(f"Securities inserted: {stats['securities_inserted']}")
    print(f"Xref rows written: {stats['xref_rows_written']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Uncomment the example you want to run:
    
    # example_default_run()
    # example_prebuilt_configs()
    # example_custom_filters()
    # example_currency_filter()
    # example_exchange_filter()
    # example_isin_prefix()
    # example_combined_filters()
    # example_custom_partitioning()
    # example_no_partitioning()
    # example_api_config()
    example_filter_builder()  # Doesn't require API connection
    # example_custom_search_fields()
    # example_direct_pipeline()
