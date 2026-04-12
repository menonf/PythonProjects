"""
Pipeline orchestrator for Refinitiv Security Master Ingestion.

Provides high-level entry points for running the complete ingestion
workflow with configurable parameters.
"""

import logging
from typing import Optional

from .config import IngestionConfig, indian_equities_config
from .api_client import RefinitivClient
from .transformers import SecurityDataTransformer
from .db_operations import SecurityRepository, SecurityMasterUpserter

# Note: Adjust this import to match your actual database module location
from data_engineering.database import database

log = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestrates the complete ingestion workflow.
    
    Coordinates the API client, data transformer, and database operations
    to execute the end-to-end pipeline.
    
    Example:
        config = IngestionConfig(
            filter_criteria=FilterCriteria(countries=["India"])
        )
        pipeline = IngestionPipeline(config)
        pipeline.run()
    """
    
    def __init__(self, config: IngestionConfig):
        """
        Initialize the pipeline.
        
        Args:
            config: Complete ingestion configuration
        """
        self.config = config
        self.client = RefinitivClient(config)
        self.transformer = SecurityDataTransformer(config.database)
    
    def run(self) -> dict:
        """
        Execute the complete ingestion pipeline.
        
        Returns:
            Dict with execution statistics
        """
        stats = {
            "rics_fetched": 0,
            "securities_matched": 0,
            "securities_inserted": 0,
            "xref_rows_written": 0,
        }
        
        log.info("=" * 60)
        log.info("Refinitiv Security Master Ingestion - START")
        log.info("=" * 60)
        
        # --- Phase 1: Extract from Refinitiv ---
        with self.client.session():
            universe_df = self.client.fetch_universe()
            stats["rics_fetched"] = len(universe_df)
            
            rics = universe_df["RIC"].dropna().unique().tolist()
            enrich_df = self.client.fetch_enrichment(rics)
            
            master_df, xref_df = self.transformer.build_master_and_xref(
                universe_df, enrich_df
            )
            log.info(
                "Built %d candidate master rows and %d xref rows.",
                len(master_df),
                len(xref_df),
            )
        
        # --- Phase 2: Load into database ---
        orm_engine, orm_conn, _, orm_session = database.get_db_connection()
        
        try:
            # Initialize repository and upserter
            repository = SecurityRepository(
                self.config.database,
                orm_engine,
                orm_session
            )
            upserter = SecurityMasterUpserter(
                repository,
                self.config.identifier_priority,
                self.config.database.vendor
            )
            
            # Upsert security_master
            initial_count = len(master_df)
            master_df = upserter.upsert(master_df)
            
            stats["securities_matched"] = len(master_df)
            stats["securities_inserted"] = initial_count - stats["securities_matched"]
            
            log.info(
                "security_master upsert complete. Resolved %d security_ids.",
                len(master_df)
            )
            
            # Upsert vendor xref
            ric_to_sid = self.transformer.build_ric_to_security_id_map(master_df)
            xref_df = self.transformer.attach_security_ids_to_xref(
                xref_df, ric_to_sid
            )
            xref_clean = self.transformer.strip_internal_columns(xref_df)
            
            repository.write_vendor_xref(xref_clean)
            stats["xref_rows_written"] = len(xref_clean)
            
            log.info("security_vendor_xref upsert complete.")
            
        finally:
            orm_conn.close()
            log.info("DB connection closed.")
        
        log.info("=" * 60)
        log.info("Refinitiv Security Master Ingestion - DONE")
        log.info("Statistics: %s", stats)
        log.info("=" * 60)
        
        return stats


def run_ingestion(config: Optional[IngestionConfig] = None) -> dict:
    """
    Run the complete ingestion pipeline.
    
    This is the main entry point for CLI usage or simple programmatic calls.
    
    Args:
        config: Optional configuration. If None, uses indian_equities_config().
    
    Returns:
        Dict with execution statistics
    
    Example:
        # Use defaults (Indian equities)
        run_ingestion()
        
        # Custom configuration
        from refinitiv_ingest import FilterCriteria, IngestionConfig
        
        config = IngestionConfig(
            filter_criteria=FilterCriteria(
                asset_type="equity",
                countries=["United States"],
                currencies=["USD"]
            )
        )
        stats = run_ingestion(config)
        print(f"Fetched {stats['rics_fetched']} RICs")
    """
    if config is None:
        config = indian_equities_config()
    
    pipeline = IngestionPipeline(config)
    return pipeline.run()


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure logging for the ingestion pipeline.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point with default configuration."""
    setup_logging()
    run_ingestion()


if __name__ == "__main__":
    main()
