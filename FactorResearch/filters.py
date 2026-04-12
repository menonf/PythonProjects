"""
Filter builder module for constructing OData filter expressions.

Provides a fluent API for building complex filter queries from
the parameterized FilterCriteria configuration.
"""

from typing import List, Optional
from .config import FilterCriteria, PartitionConfig


class FilterBuilder:
    """
    Builds OData filter expressions for Refinitiv discovery search.
    
    Supports all filterable fields from SEARCH_SELECT and provides
    a clean API for combining multiple filter conditions.
    
    Example:
        builder = FilterBuilder()
        filter_expr = (
            builder
            .with_asset_type("equity")
            .with_countries(["India", "United States"])
            .with_currencies(["INR", "USD"])
            .with_partition("RIC", "A")
            .build()
        )
        # Result: "AssetType eq 'equity' and (RCSExchangeCountryLeaf eq 'India' or 
        #          RCSExchangeCountryLeaf eq 'United States') and 
        #          (RCSCurrencyLeaf eq 'INR' or RCSCurrencyLeaf eq 'USD') and 
        #          startswith(RIC, 'A')"
    """
    
    # Mapping of filter attributes to OData field names
    FIELD_MAPPING = {
        "asset_type": "AssetType",
        "countries": "RCSExchangeCountryLeaf",
        "currencies": "RCSCurrencyLeaf",
        "asset_categories": "RCSAssetCategoryLeaf",
        "asset_classes": "RCSAssetClass",
        "exchanges": "ExchangeCode",
        "ticker_prefix": "TickerSymbol",
        "isin_prefix": "ISIN",
    }
    
    def __init__(self):
        """Initialize an empty filter builder."""
        self._conditions: List[str] = []
    
    def _escape_value(self, value: str) -> str:
        """Escape single quotes in OData string values."""
        return value.replace("'", "''")
    
    def _build_or_condition(self, field: str, values: List[str]) -> str:
        """Build an OR condition for multiple values on the same field."""
        if len(values) == 1:
            return f"{field} eq '{self._escape_value(values[0])}'"
        
        conditions = [f"{field} eq '{self._escape_value(v)}'" for v in values]
        return f"({' or '.join(conditions)})"
    
    def with_asset_type(self, asset_type: str) -> "FilterBuilder":
        """Add asset type filter (e.g., 'equity', 'bond')."""
        self._conditions.append(
            f"AssetType eq '{self._escape_value(asset_type)}'"
        )
        return self
    
    def with_countries(self, countries: List[str]) -> "FilterBuilder":
        """Add country filter (RCSExchangeCountryLeaf)."""
        if countries:
            self._conditions.append(
                self._build_or_condition("RCSExchangeCountryLeaf", countries)
            )
        return self
    
    def with_currencies(self, currencies: List[str]) -> "FilterBuilder":
        """Add currency filter (RCSCurrencyLeaf)."""
        if currencies:
            self._conditions.append(
                self._build_or_condition("RCSCurrencyLeaf", currencies)
            )
        return self
    
    def with_asset_categories(self, categories: List[str]) -> "FilterBuilder":
        """Add asset category filter (RCSAssetCategoryLeaf)."""
        if categories:
            self._conditions.append(
                self._build_or_condition("RCSAssetCategoryLeaf", categories)
            )
        return self
    
    def with_asset_classes(self, classes: List[str]) -> "FilterBuilder":
        """Add asset class filter (RCSAssetClass)."""
        if classes:
            self._conditions.append(
                self._build_or_condition("RCSAssetClass", classes)
            )
        return self
    
    def with_exchanges(self, exchanges: List[str]) -> "FilterBuilder":
        """Add exchange filter (ExchangeCode)."""
        if exchanges:
            self._conditions.append(
                self._build_or_condition("ExchangeCode", exchanges)
            )
        return self
    
    def with_ticker_prefix(self, prefix: str) -> "FilterBuilder":
        """Add ticker symbol prefix filter using startswith()."""
        self._conditions.append(
            f"startswith(TickerSymbol, '{self._escape_value(prefix)}')"
        )
        return self
    
    def with_isin_prefix(self, prefix: str) -> "FilterBuilder":
        """Add ISIN prefix filter using startswith()."""
        self._conditions.append(
            f"startswith(ISIN, '{self._escape_value(prefix)}')"
        )
        return self
    
    def with_partition(
        self,
        field: str,
        value: str,
        use_startswith: bool = True
    ) -> "FilterBuilder":
        """
        Add partition filter for paginating large datasets.
        
        Args:
            field: The field to partition on (e.g., 'RIC', 'ISIN')
            value: The partition value (e.g., 'A' for RICs starting with A)
            use_startswith: If True, use startswith(); otherwise use eq
        """
        if use_startswith:
            self._conditions.append(
                f"startswith({field}, '{self._escape_value(value)}')"
            )
        else:
            self._conditions.append(
                f"{field} eq '{self._escape_value(value)}'"
            )
        return self
    
    def with_custom_filter(self, filter_expr: str) -> "FilterBuilder":
        """Add a custom OData filter expression."""
        if filter_expr:
            self._conditions.append(f"({filter_expr})")
        return self
    
    def build(self) -> str:
        """Combine all conditions with AND and return the filter string."""
        return " and ".join(self._conditions)
    
    def reset(self) -> "FilterBuilder":
        """Clear all conditions and start fresh."""
        self._conditions = []
        return self


def build_filter_from_criteria(
    criteria: FilterCriteria,
    partition_value: Optional[str] = None,
    partition_config: Optional[PartitionConfig] = None,
) -> str:
    """
    Build OData filter expression from FilterCriteria configuration.
    
    This is the main entry point for filter generation, combining
    all configured criteria with optional partitioning.
    
    Args:
        criteria: FilterCriteria dataclass with filter parameters
        partition_value: Optional partition value (e.g., 'A' for RIC prefix)
        partition_config: Optional PartitionConfig for partition settings
    
    Returns:
        OData filter expression string
    
    Example:
        criteria = FilterCriteria(
            asset_type="equity",
            countries=["India"],
            currencies=["INR"]
        )
        filter_expr = build_filter_from_criteria(criteria, partition_value="A")
        # Returns: "AssetType eq 'equity' and RCSExchangeCountryLeaf eq 'India' 
        #           and RCSCurrencyLeaf eq 'INR' and startswith(RIC, 'A')"
    """
    builder = FilterBuilder()
    
    # Required: Asset type
    builder.with_asset_type(criteria.asset_type)
    
    # Optional: Country filter
    if criteria.countries:
        builder.with_countries(criteria.countries)
    
    # Optional: Currency filter
    if criteria.currencies:
        builder.with_currencies(criteria.currencies)
    
    # Optional: Asset category filter
    if criteria.asset_categories:
        builder.with_asset_categories(criteria.asset_categories)
    
    # Optional: Asset class filter
    if criteria.asset_classes:
        builder.with_asset_classes(criteria.asset_classes)
    
    # Optional: Exchange filter
    if criteria.exchanges:
        builder.with_exchanges(criteria.exchanges)
    
    # Optional: Ticker prefix filter
    if criteria.ticker_prefix:
        builder.with_ticker_prefix(criteria.ticker_prefix)
    
    # Optional: ISIN prefix filter
    if criteria.isin_prefix:
        builder.with_isin_prefix(criteria.isin_prefix)
    
    # Optional: Partition filter
    if partition_value and partition_config:
        builder.with_partition(
            field=partition_config.partition_field,
            value=partition_value,
            use_startswith=partition_config.use_startswith
        )
    elif partition_value:
        # Default to RIC partitioning if no config provided
        builder.with_partition("RIC", partition_value)
    
    # Optional: Custom filter
    if criteria.custom_filter:
        builder.with_custom_filter(criteria.custom_filter)
    
    return builder.build()


def get_partition_filters(
    criteria: FilterCriteria,
    partition_config: PartitionConfig,
) -> List[str]:
    """
    Generate a list of filter expressions, one per partition.
    
    Useful for distributing large queries across multiple smaller
    requests to avoid API offset limits.
    
    Args:
        criteria: FilterCriteria with base filter parameters
        partition_config: PartitionConfig with partition settings
    
    Returns:
        List of OData filter expressions
    
    Example:
        filters = get_partition_filters(
            FilterCriteria(asset_type="equity", countries=["India"]),
            PartitionConfig(partition_values=["A", "B", "C"])
        )
        # Returns list of 3 filters, one for each partition
    """
    if not partition_config.enabled:
        return [build_filter_from_criteria(criteria)]
    
    return [
        build_filter_from_criteria(criteria, partition_value, partition_config)
        for partition_value in partition_config.partition_values
    ]
