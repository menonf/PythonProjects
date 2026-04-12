"""
Data transformation module for Refinitiv security data.

Handles merging, cleaning, and transforming raw API data into
formats suitable for database ingestion.
"""

from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

from .config import DatabaseConfig


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def clean_value(val) -> Optional[str]:
    """
    Normalize values for database storage.
    
    Returns None for blank/null/NaN/NA values; stripped string otherwise.
    
    Args:
        val: Any value to clean
    
    Returns:
        Cleaned string or None
    """
    if val is None:
        return None
    text = str(val).strip()
    return None if text in ("", "nan", "None", "<NA>") else text


def safe_column(df: pd.DataFrame, name: str) -> pd.Series:
    """
    Safely extract a column from DataFrame with cleaning applied.
    
    Returns an all-None series if the column doesn't exist.
    
    Args:
        df: Source DataFrame
        name: Column name to extract
    
    Returns:
        Series with cleaned values
    """
    if name in df.columns:
        return df[name].map(clean_value)
    return pd.Series([None] * len(df), dtype=object)


# Internal column prefix (stripped before any DB write)
INTERNAL_PREFIX = "_"


# ---------------------------------------------------------------------------
# Data Transformer Class
# ---------------------------------------------------------------------------

class SecurityDataTransformer:
    """
    Transforms raw Refinitiv API data into database-ready formats.
    
    Handles the merge of universe data with enrichment data and
    produces separate DataFrames for security_master and security_vendor_xref.
    
    Example:
        transformer = SecurityDataTransformer(db_config)
        master_df, xref_df = transformer.build_master_and_xref(
            universe_df, enrich_df
        )
    """
    
    # Mapping of internal column names to Refinitiv field names
    UNIVERSE_FIELD_MAP = {
        "name": "IssuerCommonName",
        "ticker": "TickerSymbol",
        "sedol": "SEDOL",
        "cusip": "CUSIP",
        "ric": "RIC",
        "isin": "ISIN",
        "country": "RCSExchangeCountryLeaf",
        "currency": "RCSCurrencyLeaf",
        "security_type": "RCSAssetCategoryLeaf",
        "asset_class": "RCSAssetClass",
        "exchange_code": "ExchangeCode",
    }
    
    ENRICH_FIELD_MAP = {
        "isin_enriched": "TR.ISIN",
        "sector": "TR.GICSSector",
        "industry_group": "TR.GICSIndustry",
        "industry": "TR.GICSSubIndustry",
        "valor": "TR.VALOR",
        "wkn": "TR.WKN",
        "common_code": "TR.CommonCode",
        "perm_id": "TR.PermID",
    }
    
    def __init__(self, db_config: DatabaseConfig):
        """
        Initialize the transformer.
        
        Args:
            db_config: Database configuration containing vendor and script info
        """
        self.db_config = db_config
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string for upsert_date."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def build_master_and_xref(
        self,
        universe_df: pd.DataFrame,
        enrich_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merge universe and enrichment data into master and xref DataFrames.
        
        Produces two DataFrames:
        - master_df: Columns matching dbo.security_master (no security_id yet)
        - xref_df: Columns matching dbo.security_vendor_xref (no security_id yet)
        
        Columns prefixed with '_' are internal join keys and should be
        stripped before any DB write.
        
        Args:
            universe_df: Raw universe data from search API
            enrich_df: Enrichment data from get_data API
        
        Returns:
            Tuple of (master_df, xref_df)
        """
        now_str = self._get_timestamp()
        merged = universe_df.merge(enrich_df, on="RIC", how="left")
        
        # Helper for safely pulling cleaned columns
        col = lambda name: safe_column(merged, name)
        
        # --- Build security_master rows ---
        master_df = pd.DataFrame({
            # Core identifiers
            "name":           col("IssuerCommonName"),
            "isin":           col("ISIN"),
            "sedol":          col("SEDOL"),
            "cusip":          col("CUSIP"),
            "figi":           None,  # Not available from Refinitiv
            "valor":          col("TR.VALOR"),
            "wkn":            col("TR.WKN"),
            "common_code":    col("TR.CommonCode"),
            "perm_id":        col("TR.PermID"),
            
            # Trading / listing info
            "ticker":         col("TickerSymbol"),
            "exchange_code":  col("ExchangeCode"),
            "country":        col("RCSExchangeCountryLeaf"),
            "currency":       col("RCSCurrencyLeaf"),
            
            # Classification
            "sector":         col("TR.GICSSector"),
            "industry_group": col("TR.GICSIndustry"),
            "industry":       col("TR.GICSSubIndustry"),
            "security_type":  col("RCSAssetCategoryLeaf"),
            "asset_class":    col("RCSAssetClass"),
            
            # Housekeeping
            "is_active":   1,
            "upsert_date": now_str,
            "upsert_by":   self.db_config.script_name,
            
            # Internal join keys (dropped before DB writes)
            "_ric":          merged["RIC"],
            "_isin":         col("ISIN"),
            "_cusip":        col("CUSIP"),
            "_sedol":        col("SEDOL"),
            "_valor":        col("TR.VALOR"),
            "_wkn":          col("TR.WKN"),
            "_common_code":  col("TR.CommonCode"),
            "_perm_id":      col("TR.PermID"),
            "_ticker_exch":  (col("TickerSymbol").fillna("") + "|" + 
                             col("ExchangeCode").fillna("")),
            "_name_country": (col("IssuerCommonName").fillna("") + "|" + 
                             col("RCSExchangeCountryLeaf").fillna("")),
        })
        
        # --- Build security_vendor_xref rows ---
        xref_df = pd.DataFrame({
            "vendor":               self.db_config.vendor,
            "vendor_ticker":        merged["RIC"].map(clean_value),
            "vendor_exchange_code": col("ExchangeCode"),
            "vendor_currency":      col("RCSCurrencyLeaf"),
            "loanxid":              None,
            "is_primary":           0,
            "is_active":            1,
            "upsert_date":          now_str,
            "upsert_by":            self.db_config.script_name,
            
            # Internal join key
            "_ric": merged["RIC"],
        })
        
        return master_df, xref_df
    
    @staticmethod
    def strip_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove internal join-key columns (prefixed with '_') from DataFrame.
        
        Use this before writing to the database.
        
        Args:
            df: DataFrame with potential internal columns
        
        Returns:
            DataFrame with internal columns removed
        """
        return df.drop(
            columns=[c for c in df.columns if c.startswith(INTERNAL_PREFIX)]
        )
    
    @staticmethod
    def build_ric_to_security_id_map(
        master_df: pd.DataFrame
    ) -> dict:
        """
        Create a mapping from RIC to security_id from resolved master data.
        
        Args:
            master_df: Master DataFrame with _ric and security_id columns
        
        Returns:
            Dict mapping RIC values to security_id
        """
        return master_df.set_index("_ric")["security_id"].to_dict()
    
    def attach_security_ids_to_xref(
        self,
        xref_df: pd.DataFrame,
        ric_to_sid: dict,
    ) -> pd.DataFrame:
        """
        Join resolved security_id values into xref DataFrame.
        
        Args:
            xref_df: Xref DataFrame with _ric column
            ric_to_sid: Dict mapping RIC to security_id
        
        Returns:
            Xref DataFrame with security_id column added
        """
        xref_df = xref_df.copy()
        xref_df["security_id"] = xref_df["_ric"].map(ric_to_sid)
        xref_df = xref_df.dropna(subset=["security_id"])
        xref_df["security_id"] = xref_df["security_id"].astype(int)
        return xref_df
