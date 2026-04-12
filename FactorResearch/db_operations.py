"""
Database operations module for security master management.

Handles identifier matching, lookup map building, and upsert operations
for security_master and security_vendor_xref tables.
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import sqlalchemy as sql
from sqlalchemy.orm import Session

from .config import DatabaseConfig, IdentifierPriority
from .transformers import clean_value, SecurityDataTransformer, INTERNAL_PREFIX

# Note: This assumes data_engineering.database is your actual database module
# Adjust the import path as needed for your project structure
from data_engineering.database import database

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lookup Map Builders
# ---------------------------------------------------------------------------

class LookupMapBuilder:
    """
    Builds identifier lookup maps from existing database records.
    
    These maps are used by the IdentifierMatcher to resolve incoming
    securities to existing database records.
    """
    
    def __init__(self, vendor: str):
        """
        Initialize the builder.
        
        Args:
            vendor: Vendor name for filtering xref records
        """
        self.vendor = vendor
    
    def build_simple_id_map(
        self,
        df: pd.DataFrame,
        column: str
    ) -> Dict[str, int]:
        """
        Build {identifier_value: security_id} map for a single column.
        
        Args:
            df: DataFrame with security_id and identifier columns
            column: Column name to build map for
        
        Returns:
            Dict mapping identifier values to security_ids
        """
        if column not in df.columns:
            return {}
        series = df.set_index("security_id")[column].dropna()
        return {value: sid for sid, value in series.items()}
    
    def build_xref_ric_map(
        self,
        existing_xref: pd.DataFrame
    ) -> Dict[str, int]:
        """
        Build {RIC: security_id} map from vendor cross-reference table.
        
        Args:
            existing_xref: DataFrame from security_vendor_xref table
        
        Returns:
            Dict mapping RIC values to security_ids
        """
        if existing_xref.empty:
            return {}
        mask = (
            (existing_xref["vendor"] == self.vendor) &
            existing_xref["vendor_ticker"].notna()
        )
        subset = existing_xref.loc[mask]
        return dict(zip(subset["vendor_ticker"], subset["security_id"]))
    
    def build_composite_maps(
        self,
        existing_master: pd.DataFrame
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Build composite-key maps for fallback matching.
        
        Returns:
            Tuple of (ticker_exch_map, name_country_map)
        """
        ticker_exch_map: Dict[str, int] = {}
        name_country_map: Dict[str, int] = {}
        
        for _, row in existing_master.iterrows():
            sid = row["security_id"]
            
            ticker = clean_value(row.get("ticker"))
            exchange = clean_value(row.get("exchange_code"))
            if ticker and exchange:
                ticker_exch_map[f"{ticker}|{exchange}"] = sid
            
            name = clean_value(row.get("name"))
            country = clean_value(row.get("country"))
            if name and country:
                name_country_map[f"{name}|{country}"] = sid
        
        return ticker_exch_map, name_country_map
    
    def build_all_maps(
        self,
        existing_master: pd.DataFrame,
        existing_xref: pd.DataFrame
    ) -> Dict[str, Dict[str, int]]:
        """
        Build all lookup maps for the identifier matching cascade.
        
        Args:
            existing_master: DataFrame from security_master table
            existing_xref: DataFrame from security_vendor_xref table
        
        Returns:
            Dict of {identifier_type: {value: security_id}}
        """
        xref_ric_map = self.build_xref_ric_map(existing_xref)
        log.info(
            "Xref RIC map: %d existing %s mappings.",
            len(xref_ric_map),
            self.vendor
        )
        
        ticker_exch_map, name_country_map = self.build_composite_maps(
            existing_master
        )
        
        return {
            # Priority 1 - RIC via xref
            "xref_ric":    xref_ric_map,
            # Priority 2 - standard identifiers
            "isin":        self.build_simple_id_map(existing_master, "isin"),
            "cusip":       self.build_simple_id_map(existing_master, "cusip"),
            "figi":        self.build_simple_id_map(existing_master, "figi"),
            "sedol":       self.build_simple_id_map(existing_master, "sedol"),
            # Priority 3 - supplementary identifiers
            "valor":       self.build_simple_id_map(existing_master, "valor"),
            "wkn":         self.build_simple_id_map(existing_master, "wkn"),
            "common_code": self.build_simple_id_map(existing_master, "common_code"),
            "perm_id":     self.build_simple_id_map(existing_master, "perm_id"),
            # Priority 4 - composite keys (last resort)
            "ticker_exch":  ticker_exch_map,
            "name_country": name_country_map,
        }


# ---------------------------------------------------------------------------
# Identifier Matcher
# ---------------------------------------------------------------------------

class IdentifierMatcher:
    """
    Matches incoming securities to existing database records.
    
    Uses a configurable cascade of identifiers to find matches,
    checking each identifier type in priority order.
    """
    
    def __init__(self, identifier_priority: IdentifierPriority):
        """
        Initialize with identifier priority configuration.
        
        Args:
            identifier_priority: Configuration defining cascade order
        """
        self.cascade = identifier_priority.cascade
    
    def find_match(
        self,
        row: pd.Series,
        maps: Dict[str, Dict[str, int]]
    ) -> Optional[int]:
        """
        Walk the identifier cascade to find a matching security_id.
        
        Args:
            row: Incoming security row with identifier columns
            maps: Lookup maps from LookupMapBuilder
        
        Returns:
            Matching security_id, or None if no match found
        """
        for map_name, row_key in self.cascade:
            key_value = row.get(row_key)
            if key_value is not None:
                sid = maps.get(map_name, {}).get(key_value)
                if sid is not None:
                    return sid
        return None
    
    def match_all(
        self,
        df: pd.DataFrame,
        maps: Dict[str, Dict[str, int]]
    ) -> List[Optional[int]]:
        """
        Match all rows in a DataFrame to security_ids.
        
        Args:
            df: DataFrame with identifier columns
            maps: Lookup maps from LookupMapBuilder
        
        Returns:
            List of security_ids (None for unmatched rows)
        """
        return [
            self.find_match(row, maps)
            for _, row in df.iterrows()
        ]


# ---------------------------------------------------------------------------
# Database Repository
# ---------------------------------------------------------------------------

class SecurityRepository:
    """
    Repository for security_master and security_vendor_xref operations.
    
    Encapsulates all database interactions for cleaner separation
    of concerns and easier testing.
    """
    
    def __init__(
        self,
        db_config: DatabaseConfig,
        orm_engine: sql.Engine,
        orm_session: Session
    ):
        """
        Initialize the repository.
        
        Args:
            db_config: Database configuration
            orm_engine: SQLAlchemy engine
            orm_session: SQLAlchemy session
        """
        self.config = db_config
        self.engine = orm_engine
        self.session = orm_session
    
    def read_security_master(self) -> pd.DataFrame:
        """Read existing security_master records."""
        return database.read_security_master(self.session, self.engine)
    
    def read_security_vendor_xref(self) -> pd.DataFrame:
        """Read existing security_vendor_xref records for this vendor."""
        return database.read_security_vendor_xref(
            self.session, self.engine, vendor=self.config.vendor
        )
    
    def insert_new_securities(self, rows: List[dict]) -> None:
        """
        Batch-insert new securities into security_master.
        
        Args:
            rows: List of row dicts (without internal columns)
        """
        log.info("Inserting %d new securities into security_master…", len(rows))
        
        for start in range(0, len(rows), self.config.batch_size):
            batch = rows[start:start + self.config.batch_size]
            with Session(self.engine) as session:
                database._execute_with_session(
                    session,
                    lambda data: session.bulk_insert_mappings(
                        database.SecurityMaster, data
                    ),
                    batch,
                )
            log.info("  Inserted rows %d-%d.", start + 1, start + len(batch))
    
    def write_vendor_xref(self, xref_df: pd.DataFrame) -> None:
        """
        Write vendor xref records in batches.
        
        Args:
            xref_df: DataFrame with xref records (internal columns stripped)
        """
        log.info("Writing %d vendor xref rows…", len(xref_df))
        
        for start in range(0, len(xref_df), self.config.batch_size):
            batch_df = xref_df.iloc[start:start + self.config.batch_size]
            with Session(self.engine) as session:
                database.write_security_vendor_xref(
                    batch_df, session, vendor=self.config.vendor
                )
            log.info("  Xref batch %d-%d written.", start + 1, start + len(batch_df))


# ---------------------------------------------------------------------------
# Security Master Upserter
# ---------------------------------------------------------------------------

class SecurityMasterUpserter:
    """
    High-level upserter that orchestrates the match-insert-resolve workflow.
    
    Uses IdentifierMatcher to find existing securities, inserts new ones,
    and resolves all security_ids for downstream processing.
    """
    
    def __init__(
        self,
        repository: SecurityRepository,
        identifier_priority: IdentifierPriority,
        vendor: str
    ):
        """
        Initialize the upserter.
        
        Args:
            repository: SecurityRepository for DB operations
            identifier_priority: Cascade configuration
            vendor: Vendor name for lookup maps
        """
        self.repository = repository
        self.matcher = IdentifierMatcher(identifier_priority)
        self.map_builder = LookupMapBuilder(vendor)
    
    def _build_lookup_maps(self) -> Dict[str, Dict[str, int]]:
        """Load existing data and build lookup maps."""
        log.info("Loading existing security_master and vendor xref from DB…")
        existing_master = self.repository.read_security_master()
        existing_xref = self.repository.read_security_vendor_xref()
        return self.map_builder.build_all_maps(existing_master, existing_xref)
    
    def _extract_rows_to_insert(
        self,
        master_df: pd.DataFrame,
        security_ids: List[Optional[int]]
    ) -> List[dict]:
        """Extract rows that need to be inserted (unmatched rows)."""
        rows_to_insert = []
        for idx, sid in enumerate(security_ids):
            if sid is None:
                row = master_df.iloc[idx]
                rows_to_insert.append({
                    k: v for k, v in row.items()
                    if not k.startswith(INTERNAL_PREFIX)
                })
        return rows_to_insert
    
    def upsert(self, master_df: pd.DataFrame) -> pd.DataFrame:
        """
        Resolve security_ids for all rows, inserting new securities as needed.
        
        Args:
            master_df: DataFrame with security data and internal join keys
        
        Returns:
            DataFrame with security_id column populated
        """
        # First pass: match existing securities
        maps = self._build_lookup_maps()
        security_ids = self.matcher.match_all(master_df, maps)
        
        master_df = master_df.copy()
        master_df["security_id"] = security_ids
        
        matched = sum(1 for s in security_ids if s is not None)
        rows_to_insert = self._extract_rows_to_insert(master_df, security_ids)
        
        log.info(
            "Cascade results: %d matched, %d new (to insert).",
            matched,
            len(rows_to_insert)
        )
        
        # Insert new securities and re-resolve
        if rows_to_insert:
            self.repository.insert_new_securities(rows_to_insert)
            
            log.info("Reloading tables to resolve newly created security_ids…")
            maps = self._build_lookup_maps()
            
            for idx, row in master_df.iterrows():
                if pd.isna(master_df.at[idx, "security_id"]):
                    master_df.at[idx, "security_id"] = self.matcher.find_match(
                        row, maps
                    )
        else:
            log.info(
                "All %d securities already exist in security_master.",
                len(master_df)
            )
        
        # Drop unresolvable rows
        unresolved = master_df["security_id"].isna().sum()
        if unresolved:
            log.warning(
                "%d rows could not be resolved to a security_id and will be "
                "skipped. These have no identifiers whatsoever - check source "
                "data quality.",
                unresolved,
            )
        
        master_df = master_df.dropna(subset=["security_id"])
        master_df["security_id"] = master_df["security_id"].astype(int)
        return master_df
