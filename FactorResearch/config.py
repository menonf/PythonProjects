"""
Configuration module for Refinitiv Security Master Ingestion.

Contains all configurable parameters as dataclasses for type safety,
validation, and easy modification.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class APIConfig:
    """API-related configuration settings."""
    
    page_size: int = 10_000
    """Maximum rows per search page."""
    
    gics_batch_size: int = 5_000
    """RICs per get_data call (API limit)."""
    
    retry_sleep_sec: int = 5
    """Pause between failed API calls in seconds."""
    
    max_api_retries: int = 3
    """Maximum number of retry attempts for API calls."""
    
    max_universe_pages: Optional[int] = None
    """Set to an int (e.g., 3) for test runs; None = full run."""
    
    max_offset: int = 10_000
    """Maximum offset for pagination (API limit)."""


@dataclass
class DatabaseConfig:
    """Database-related configuration settings."""
    
    batch_size: int = 1_000
    """Rows per DB commit."""
    
    vendor: str = "Refinitiv"
    """Vendor name for xref table."""
    
    script_name: str = "refinitiv_ingest.py"
    """Script identifier for audit columns."""


@dataclass
class SearchFields:
    """
    Fields returned by rd.discovery.search.
    
    Each field can be toggled on/off and used in filters.
    """
    
    ticker_symbol: bool = True
    issuer_common_name: bool = True
    sedol: bool = True
    cusip: bool = True
    ric: bool = True
    isin: bool = True
    exchange_country: bool = True
    currency: bool = True
    asset_category: bool = True
    asset_class: bool = True
    exchange_code: bool = True
    
    # Mapping of internal names to Refinitiv API field names
    FIELD_MAPPING: Dict[str, str] = field(default_factory=lambda: {
        "ticker_symbol": "TickerSymbol",
        "issuer_common_name": "IssuerCommonName",
        "sedol": "SEDOL",
        "cusip": "CUSIP",
        "ric": "RIC",
        "isin": "ISIN",
        "exchange_country": "RCSExchangeCountryLeaf",
        "currency": "RCSCurrencyLeaf",
        "asset_category": "RCSAssetCategoryLeaf",
        "asset_class": "RCSAssetClass",
        "exchange_code": "ExchangeCode",
    })
    
    def get_select_string(self) -> str:
        """Generate the SELECT clause from enabled fields."""
        enabled = []
        for internal_name, api_name in self.FIELD_MAPPING.items():
            if getattr(self, internal_name, False):
                enabled.append(api_name)
        return ", ".join(enabled)
    
    def get_enabled_fields(self) -> List[str]:
        """Return list of enabled API field names."""
        return [
            api_name
            for internal_name, api_name in self.FIELD_MAPPING.items()
            if getattr(self, internal_name, False)
        ]


@dataclass
class EnrichmentFields:
    """Fields fetched via rd.get_data for enrichment."""
    
    fields: List[str] = field(default_factory=lambda: [
        "TR.ISIN",
        "TR.GICSSector",
        "TR.GICSIndustry",
        "TR.GICSSubIndustry",
        "TR.VALOR",       # Swiss Valorennummer
        "TR.WKN",         # German Wertpapierkennnummer
        "TR.CommonCode",  # Euroclear / Clearstream Common Code
        "TR.PermID",      # Refinitiv permanent identifier
    ])


@dataclass
class FilterCriteria:
    """
    Parameterized filter criteria for the Refinitiv search.
    
    Use this to customize which securities are fetched based on
    any combination of filterable fields from SEARCH_SELECT.
    
    Examples:
        # Fetch Indian equities only
        FilterCriteria(asset_type="equity", countries=["India"])
        
        # Fetch US and UK equities in USD or GBP
        FilterCriteria(
            asset_type="equity",
            countries=["United States", "United Kingdom"],
            currencies=["USD", "GBP"]
        )
        
        # Fetch all equities (no country filter)
        FilterCriteria(asset_type="equity")
    """
    
    asset_type: str = "equity"
    """Asset type filter (e.g., 'equity', 'bond')."""
    
    countries: Optional[List[str]] = None
    """Filter by RCSExchangeCountryLeaf. None = all countries."""
    
    currencies: Optional[List[str]] = None
    """Filter by RCSCurrencyLeaf. None = all currencies."""
    
    asset_categories: Optional[List[str]] = None
    """Filter by RCSAssetCategoryLeaf. None = all categories."""
    
    asset_classes: Optional[List[str]] = None
    """Filter by RCSAssetClass. None = all classes."""
    
    exchanges: Optional[List[str]] = None
    """Filter by ExchangeCode. None = all exchanges."""
    
    ticker_prefix: Optional[str] = None
    """Filter by ticker symbol prefix."""
    
    isin_prefix: Optional[str] = None
    """Filter by ISIN prefix (e.g., 'US' for US securities)."""
    
    custom_filter: Optional[str] = None
    """Additional custom OData filter expression to AND with built filters."""


@dataclass
class PartitionConfig:
    """
    Configuration for partitioning large datasets.
    
    Partitioning helps work around API offset limits by splitting
    requests into smaller chunks based on field prefixes.
    """
    
    enabled: bool = True
    """Whether to use partitioning."""
    
    partition_field: str = "RIC"
    """Field to partition on (e.g., 'RIC', 'ISIN', 'TickerSymbol')."""
    
    partition_values: List[str] = field(default_factory=lambda: list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
    """Values to partition by (first characters)."""
    
    use_startswith: bool = True
    """Use startswith() for partitioning. If False, uses 'eq'."""


@dataclass
class IdentifierPriority:
    """
    Configuration for the identifier matching cascade.
    
    Defines the order in which identifiers are checked when matching
    incoming securities to existing database records.
    """
    
    cascade: List[tuple] = field(default_factory=lambda: [
        ("xref_ric",    "_ric"),         # RIC via vendor xref (highest priority)
        ("isin",        "_isin"),        # ISIN
        ("cusip",       "_cusip"),       # CUSIP
        ("figi",        "figi"),         # FIGI
        ("sedol",       "_sedol"),       # SEDOL
        ("valor",       "_valor"),       # Swiss Valorennummer
        ("wkn",         "_wkn"),         # German WKN
        ("common_code", "_common_code"), # Euroclear Common Code
        ("perm_id",     "_perm_id"),     # Refinitiv PermID
        ("ticker_exch", "_ticker_exch"), # Ticker + Exchange composite
        ("name_country", "_name_country"), # Name + Country (lowest priority)
    ])
    """
    Ordered list of (lookup_map_name, row_key) tuples.
    First match wins; order defines priority.
    """


@dataclass
class IngestionConfig:
    """
    Master configuration for the entire ingestion pipeline.
    
    Combines all sub-configurations into a single, easily passable object.
    
    Example:
        # Default configuration for Indian equities
        config = IngestionConfig()
        
        # Custom configuration for US equities
        config = IngestionConfig(
            filter_criteria=FilterCriteria(
                asset_type="equity",
                countries=["United States"]
            ),
            partition=PartitionConfig(
                partition_field="ISIN",
                partition_values=["US"]  # ISIN prefix
            )
        )
    """
    
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    search_fields: SearchFields = field(default_factory=SearchFields)
    enrichment_fields: EnrichmentFields = field(default_factory=EnrichmentFields)
    filter_criteria: FilterCriteria = field(default_factory=FilterCriteria)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    identifier_priority: IdentifierPriority = field(default_factory=IdentifierPriority)


# ---------------------------------------------------------------------------
# Pre-defined configurations for common use cases
# ---------------------------------------------------------------------------

def indian_equities_config() -> IngestionConfig:
    """Configuration for fetching Indian equities only."""
    return IngestionConfig(
        filter_criteria=FilterCriteria(
            asset_type="equity",
            countries=["India"]
        )
    )


def us_equities_config() -> IngestionConfig:
    """Configuration for fetching US equities only."""
    return IngestionConfig(
        filter_criteria=FilterCriteria(
            asset_type="equity",
            countries=["United States"]
        ),
        partition=PartitionConfig(
            partition_field="TickerSymbol"
        )
    )


def global_equities_config() -> IngestionConfig:
    """Configuration for fetching all global equities."""
    return IngestionConfig(
        filter_criteria=FilterCriteria(
            asset_type="equity",
            countries=None  # All countries
        )
    )


def test_config(max_pages: int = 2) -> IngestionConfig:
    """Configuration for test runs with limited data."""
    return IngestionConfig(
        api=APIConfig(max_universe_pages=max_pages),
        filter_criteria=FilterCriteria(
            asset_type="equity",
            countries=["India"]
        )
    )
