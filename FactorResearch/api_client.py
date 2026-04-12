"""
Refinitiv API client module.

Handles all interactions with the Refinitiv Data Library API,
including session management, retry logic, and data fetching.
"""

import logging
import math
import time
from contextlib import contextmanager
from typing import Callable, List, Optional, TypeVar

import pandas as pd
import lseg.data as rd

from .config import APIConfig, IngestionConfig, SearchFields, EnrichmentFields
from .filters import build_filter_from_criteria

log = logging.getLogger(__name__)

T = TypeVar("T")


class RefinitivClient:
    """
    Client for interacting with Refinitiv Data Library API.
    
    Provides methods for fetching universe data and enrichment data
    with automatic retry logic and batch processing.
    
    Example:
        config = IngestionConfig()
        client = RefinitivClient(config)
        
        with client.session():
            universe_df = client.fetch_universe()
            enrich_df = client.fetch_enrichment(rics)
    """
    
    def __init__(self, config: IngestionConfig):
        """
        Initialize the Refinitiv client.
        
        Args:
            config: IngestionConfig containing all settings
        """
        self.config = config
        self._session_open = False
    
    @contextmanager
    def session(self):
        """
        Context manager for Refinitiv session lifecycle.
        
        Automatically opens and closes the session, ensuring
        proper cleanup even if an exception occurs.
        
        Example:
            with client.session():
                data = client.fetch_universe()
        """
        self.open_session()
        try:
            yield self
        finally:
            self.close_session()
    
    def open_session(self) -> None:
        """Open a Refinitiv API session."""
        if not self._session_open:
            rd.open_session()
            self._session_open = True
            log.info("Refinitiv session opened.")
    
    def close_session(self) -> None:
        """Close the Refinitiv API session."""
        if self._session_open:
            rd.close_session()
            self._session_open = False
            log.info("Refinitiv session closed.")
    
    def _api_call_with_retry(
        self,
        func: Callable[[], T],
        description: str = "API call"
    ) -> T:
        """
        Execute an API call with automatic retry logic.
        
        Args:
            func: Callable that performs the API call
            description: Description for logging purposes
        
        Returns:
            Result of the API call
        
        Raises:
            Last exception if all retries fail
        """
        api_config = self.config.api
        
        for attempt in range(1, api_config.max_api_retries + 1):
            try:
                return func()
            except Exception as exc:
                log.warning(
                    "%s - attempt %d/%d failed: %s",
                    description,
                    attempt,
                    api_config.max_api_retries,
                    exc
                )
                if attempt == api_config.max_api_retries:
                    raise
                time.sleep(api_config.retry_sleep_sec)
    
    def _fetch_page(
        self,
        filter_expr: str,
        skip: int,
        top: int,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch a single page of search results.
        
        Args:
            filter_expr: OData filter expression
            skip: Number of results to skip
            top: Maximum number of results to return
        
        Returns:
            DataFrame with search results, or None if empty
        """
        select_str = self.config.search_fields.get_select_string()
        
        result = self._api_call_with_retry(
            lambda: rd.discovery.search(
                view=rd.discovery.Views.EQUITY_QUOTES,
                filter=filter_expr,
                select=select_str,
                top=top,
                skip=skip,
            ),
            description=f"Universe search (skip={skip}, top={top})",
        )
        
        return result if result is not None and not result.empty else None
    
    def fetch_universe(self) -> pd.DataFrame:
        """
        Fetch the complete equity universe based on configuration.
        
        Uses filter-based partitioning to work around the API's
        10,000 offset limit, then paginates within each partition.
        
        Returns:
            DataFrame with deduplicated universe data (one row per RIC)
        """
        api_config = self.config.api
        filter_criteria = self.config.filter_criteria
        partition_config = self.config.partition
        
        all_pages: List[pd.DataFrame] = []
        total_fetched = 0
        
        log.info(
            "Starting Refinitiv universe fetch (page_size=%d)…",
            api_config.page_size
        )
        
        partitions = (
            partition_config.partition_values
            if partition_config.enabled
            else [None]
        )
        
        for partition in partitions:
            skip = 0
            page_count = 0
            
            # Build filter for this partition
            filter_expr = build_filter_from_criteria(
                filter_criteria,
                partition_value=partition,
                partition_config=partition_config if partition else None
            )
            
            if partition:
                log.info("Fetching partition '%s'…", partition)
            
            while True:
                # Check offset limit
                if skip >= api_config.max_offset:
                    log.warning(
                        "Partition '%s' hit MAX_OFFSET=%d; data may be incomplete.",
                        partition,
                        api_config.max_offset
                    )
                    break
                
                # Calculate page size (respect offset limit)
                top = min(api_config.page_size, api_config.max_offset - skip)
                
                # Fetch page
                page = self._fetch_page(filter_expr, skip, top)
                
                if page is None:
                    break
                
                page_count += 1
                all_pages.append(page)
                fetched = len(page)
                skip += fetched
                total_fetched += fetched
                
                log.info(
                    "Partition '%s' page %d: fetched %d rows "
                    "(partition total: %d, running total: %d).",
                    partition, page_count, fetched, skip, total_fetched
                )
                
                # Check test run limit
                if (api_config.max_universe_pages and
                        page_count >= api_config.max_universe_pages):
                    log.info(
                        "Reached MAX_UNIVERSE_PAGES=%d; stopping test run.",
                        api_config.max_universe_pages
                    )
                    break
                
                # Check if this was the last page
                if fetched < top:
                    break
            
            # Check test run limit at partition level
            if (api_config.max_universe_pages and
                    page_count >= api_config.max_universe_pages):
                break
        
        if not all_pages:
            raise RuntimeError("Refinitiv search returned no data.")
        
        # Combine and deduplicate
        df = pd.concat(all_pages, ignore_index=True).drop_duplicates(subset=["RIC"])
        log.info("Universe: %d unique RICs after de-duplication.", len(df))
        return df
    
    def fetch_enrichment(self, rics: List[str]) -> pd.DataFrame:
        """
        Fetch enrichment data (GICS sectors, additional identifiers) for RICs.
        
        Processes in batches to respect API limits.
        
        Args:
            rics: List of RICs to fetch enrichment data for
        
        Returns:
            DataFrame with enrichment data indexed by RIC
        """
        api_config = self.config.api
        fields = self.config.enrichment_fields.fields
        
        n_batches = math.ceil(len(rics) / api_config.gics_batch_size)
        log.info(
            "Fetching enrichment data in %d batch(es) of up to %d RICs…",
            n_batches,
            api_config.gics_batch_size
        )
        
        pages: List[pd.DataFrame] = []
        
        for batch_idx in range(n_batches):
            start = batch_idx * api_config.gics_batch_size
            end = start + api_config.gics_batch_size
            batch = rics[start:end]
            
            try:
                result = self._api_call_with_retry(
                    lambda b=batch: rd.get_data(universe=b, fields=fields),
                    description=f"Enrichment batch {batch_idx + 1}/{n_batches}",
                )
                pages.append(result)
                log.info(
                    "Enrichment batch %d/%d done (%d RICs).",
                    batch_idx + 1, n_batches, len(batch)
                )
            except Exception:
                log.error(
                    "Giving up on enrichment batch %d - skipping.",
                    batch_idx + 1
                )
        
        if not pages:
            log.warning(
                "No enrichment data returned - sector/industry & extra IDs will be NULL."
            )
            return pd.DataFrame(columns=["RIC"] + fields)
        
        enrich_df = pd.concat(pages, ignore_index=True).rename(
            columns={"Instrument": "RIC"}
        )
        return enrich_df
