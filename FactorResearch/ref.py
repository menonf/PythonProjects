"""
Refinitiv Security Master Ingestion
=====================================
Fetches the full equity universe from Refinitiv Data Library, enriches it with
GICS sector/industry data and additional identifiers, then upserts into the
two-table security master:

    dbo.security_master      - one row per canonical security
    dbo.security_vendor_xref - one row per vendor × security (Refinitiv-specific codes)

Matching cascade (highest → lowest priority):
    1. RIC via security_vendor_xref          ← catches ALL previously ingested securities
    2. ISIN → CUSIP → FIGI → SEDOL
    3. VALOR → WKN → Common Code → PermID
    4. Ticker+Exchange → Name+Country        ← fuzzy last-resort

Why RIC is first:
    RIC is always present in Refinitiv data.  Any security ingested previously
    already has its RIC stored in security_vendor_xref.vendor_ticker.  Checking
    the xref table first means a security with no ISIN/CUSIP is still matched
    on every subsequent run rather than being inserted as a duplicate.

Usage:
    python refinitiv_ingest.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import logging
import math
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import lseg.data as rd
import sqlalchemy as sql
from sqlalchemy.orm import Session

from data_engineering.database import database

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / Configuration
# ---------------------------------------------------------------------------

VENDOR = "Refinitiv"
SCRIPT_NAME = "refinitiv_ingest.py"

PAGE_SIZE = 10000           # max rows per search page
GICS_BATCH_SIZE = 5000      # RICs per get_data call (API limit)
DB_BATCH_SIZE = 1000         # rows per DB commit
RETRY_SLEEP_SEC = 5           # pause between failed API calls
MAX_API_RETRIES = 3
MAX_UNIVERSE_PAGES = None     # set to an int (e.g. 3) for test runs; None = full run

# Fields returned by rd.discovery.search
SEARCH_SELECT = (
    "TickerSymbol, "
    "IssuerCommonName, "
    "SEDOL, "
    "CUSIP, "
    "RIC, "
    "ISIN, "
    "RCSExchangeCountryLeaf, "
    "RCSCurrencyLeaf, "
    "RCSAssetCategoryLeaf, "
    "RCSAssetClass, "
    "ExchangeCode"
)

# Fields fetched via rd.get_data (keyed on RIC)
ENRICH_FIELDS = [
    "TR.ISIN",
    "TR.GICSSector",
    "TR.GICSIndustry",
    "TR.GICSSubIndustry",
    "TR.VALOR",       # Swiss Valorennummer
    "TR.WKN",         # German Wertpapierkennnummer
    "TR.CommonCode",  # Euroclear / Clearstream Common Code
    "TR.PermID",      # Refinitiv permanent identifier (open, non-recycled)
]

# Internal join-key column prefix (stripped before any DB write)
_INTERNAL_PREFIX = "_"

# Ordered cascade of (lookup_map_name, row_key) pairs used for matching.
# This list drives _cascade_lookup and must stay in priority order.
IDENTIFIER_CASCADE = [
    ("xref_ric",    "_ric"),
    ("isin",        "_isin"),
    ("cusip",       "_cusip"),
    ("figi",        "figi"),
    ("sedol",       "_sedol"),
    ("valor",       "_valor"),
    ("wkn",         "_wkn"),
    ("common_code", "_common_code"),
    ("perm_id",     "_perm_id"),
    ("ticker_exch", "_ticker_exch"),
    ("name_country", "_name_country"),
]


# ===================================================================
# Helpers
# ===================================================================

def _clean(val) -> Optional[str]:
    """Return ``None`` for blank / null / NaN / NA values; stripped string otherwise."""
    if val is None:
        return None
    text = str(val).strip()
    return None if text in ("", "nan", "None", "<NA>") else text


def _safe_column(df: pd.DataFrame, name: str) -> pd.Series:
    """Return the column *name* from *df* with ``_clean`` applied, or an all-None series."""
    if name in df.columns:
        return df[name].map(_clean)
    return pd.Series([None] * len(df), dtype=object)


def _api_call_with_retry(func, *, description: str = "API call"):
    """
    Execute *func* with up to ``MAX_API_RETRIES`` attempts.

    Returns the result of *func* on success or re-raises the last exception.
    """
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            return func()
        except Exception as exc:
            log.warning("%s - attempt %d/%d failed: %s", description, attempt, MAX_API_RETRIES, exc)
            if attempt == MAX_API_RETRIES:
                raise
            time.sleep(RETRY_SLEEP_SEC)


# ===================================================================
# Step 1 - Fetch universe from Refinitiv (paginated)
# ===================================================================

def fetch_refinitiv_universe() -> pd.DataFrame:
    """
    Page through ``rd.discovery.search`` for all equity quotes.

    Uses filter-based partitioning to work around the API's 10,000 offset limit,
    then paginates within each partition.
    """
    MAX_OFFSET = 10000
    
    # Partition by first letter of RIC to stay under 10k per partition
    # Adjust partitions based on your data distribution
    PARTITIONS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    all_pages: List[pd.DataFrame] = []
    total_fetched = 0

    log.info("Starting Refinitiv universe fetch (page_size=%d)…", PAGE_SIZE)

    for partition in PARTITIONS:
        skip = 0
        page_count = 0
        partition_filter = f"AssetType eq 'equity' and RCSExchangeCountryLeaf eq 'India' and startswith(RIC, '{partition}')"

        log.info("Fetching partition '%s'…", partition)

        while True:
            if skip >= MAX_OFFSET:
                log.warning("Partition '%s' hit MAX_OFFSET=%d; data may be incomplete.", partition, MAX_OFFSET)
                break

            top = min(PAGE_SIZE, MAX_OFFSET - skip)

            page = _api_call_with_retry(
                lambda s=skip, t=top, f=partition_filter: rd.discovery.search(
                    view=rd.discovery.Views.EQUITY_QUOTES,
                    filter=f,
                    select=SEARCH_SELECT,
                    top=t,
                    skip=s,
                ),
                description=f"Universe search (partition={partition}, skip={skip}, top={top})",
            )

            if page is None or page.empty:
                break

            page_count += 1
            all_pages.append(page)
            fetched = len(page)
            skip += fetched
            total_fetched += fetched
            log.info("Partition '%s' page %d: fetched %d rows (partition total: %d, running total: %d).",
                     partition, page_count, fetched, skip, total_fetched)

            if MAX_UNIVERSE_PAGES and page_count >= MAX_UNIVERSE_PAGES:
                log.info("Reached MAX_UNIVERSE_PAGES=%d; stopping test run.", MAX_UNIVERSE_PAGES)
                break

            if fetched < top:
                break

        if MAX_UNIVERSE_PAGES and page_count >= MAX_UNIVERSE_PAGES:
            break

    if not all_pages:
        raise RuntimeError("Refinitiv search returned no data.")

    df = pd.concat(all_pages, ignore_index=True).drop_duplicates(subset=["RIC"])
    log.info("Universe: %d unique RICs after de-duplication.", len(df))
    return df



# ===================================================================
# Step 2 - Enrich with GICS + extra identifiers (batched)
# ===================================================================

def fetch_enrichment_data(rics: List[str]) -> pd.DataFrame:
    """
    Fetch GICS sector/industry fields and extra identifiers
    (VALOR, WKN, CommonCode, PermID) for a list of RICs in batches.
    """
    n_batches = math.ceil(len(rics) / GICS_BATCH_SIZE)
    log.info("Fetching enrichment data in %d batch(es) of up to %d RICs…", n_batches, GICS_BATCH_SIZE)

    pages: List[pd.DataFrame] = []

    for batch_idx in range(n_batches):
        batch = rics[batch_idx * GICS_BATCH_SIZE: (batch_idx + 1) * GICS_BATCH_SIZE]

        try:
            result = _api_call_with_retry(
                lambda b=batch: rd.get_data(universe=b, fields=ENRICH_FIELDS),
                description=f"Enrichment batch {batch_idx + 1}/{n_batches}",
            )
            pages.append(result)
            log.info("Enrichment batch %d/%d done (%d RICs).", batch_idx + 1, n_batches, len(batch))
        except Exception:
            log.error("Giving up on enrichment batch %d - skipping.", batch_idx + 1)

    if not pages:
        log.warning("No enrichment data returned - sector/industry & extra IDs will be NULL.")
        return pd.DataFrame(columns=["RIC"] + ENRICH_FIELDS)

    enrich_df = pd.concat(pages, ignore_index=True).rename(columns={"Instrument": "RIC"})
    return enrich_df


# ===================================================================
# Step 3 - Merge search + enrichment data and split into master / xref
# ===================================================================

def build_master_and_xref(
    universe_df: pd.DataFrame,
    enrich_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge search results with enrichment fields and produce two DataFrames:

    * **master_df** - columns matching ``dbo.security_master`` (no ``security_id`` yet).
    * **xref_df**   - columns matching ``dbo.security_vendor_xref`` (no ``security_id`` yet).

    Columns prefixed with ``_`` are internal join keys and are stripped before
    any DB write.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    merged = universe_df.merge(enrich_df, on="RIC", how="left")

    # Shorthand for safely pulling cleaned columns from the merged frame
    col = lambda name: _safe_column(merged, name)  # noqa: E731

    # --- security_master rows ---
    master_df = pd.DataFrame({
        # Core identifiers
        "name":           col("IssuerCommonName"),
        "isin":           col("ISIN"),
        "sedol":          col("SEDOL"),
        "cusip":          col("CUSIP"),
        "figi":           None,
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
        "upsert_by":   SCRIPT_NAME,

        # Internal join keys (dropped before DB writes)
        "_ric":          merged["RIC"],
        "_isin":         col("ISIN"),
        "_cusip":        col("CUSIP"),
        "_sedol":        col("SEDOL"),
        "_valor":        col("TR.VALOR"),
        "_wkn":          col("TR.WKN"),
        "_common_code":  col("TR.CommonCode"),
        "_perm_id":      col("TR.PermID"),
        "_ticker_exch":  col("TickerSymbol").fillna("") + "|" + col("ExchangeCode").fillna(""),
        "_name_country": col("IssuerCommonName").fillna("") + "|" + col("RCSExchangeCountryLeaf").fillna(""),
    })

    # --- security_vendor_xref rows ---
    xref_df = pd.DataFrame({
        "vendor":               VENDOR,
        "vendor_ticker":        merged["RIC"].map(_clean),
        "vendor_exchange_code": col("ExchangeCode"),
        "vendor_currency":      col("RCSCurrencyLeaf"),
        "loanxid":              None,
        "is_primary":           0,
        "is_active":            1,
        "upsert_date":          now_str,
        "upsert_by":            SCRIPT_NAME,

        # Internal join key
        "_ric": merged["RIC"],
    })

    return master_df, xref_df


# ===================================================================
# Step 4 - Build lookup maps for the matching cascade
# ===================================================================

def _build_simple_id_map(df: pd.DataFrame, column: str) -> Dict[str, int]:
    """Return ``{identifier_value: security_id}`` for a single column in *df*."""
    if column not in df.columns:
        return {}
    series = df.set_index("security_id")[column].dropna()
    return {value: sid for sid, value in series.items()}


def _build_xref_ric_map(existing_xref: pd.DataFrame) -> Dict[str, int]:
    """Return ``{RIC: security_id}`` from the vendor cross-reference table."""
    if existing_xref.empty:
        return {}
    mask = (existing_xref["vendor"] == VENDOR) & existing_xref["vendor_ticker"].notna()
    subset = existing_xref.loc[mask]
    return dict(zip(subset["vendor_ticker"], subset["security_id"]))


def _build_composite_maps(
    existing_master: pd.DataFrame,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Build composite-key maps:

    * ``ticker|exchange`` → security_id
    * ``name|country``    → security_id
    """
    ticker_exch_map: Dict[str, int] = {}
    name_country_map: Dict[str, int] = {}

    for _, row in existing_master.iterrows():
        sid = row["security_id"]

        ticker = _clean(row.get("ticker"))
        exchange = _clean(row.get("exchange_code"))
        if ticker and exchange:
            ticker_exch_map[f"{ticker}|{exchange}"] = sid

        name = _clean(row.get("name"))
        country = _clean(row.get("country"))
        if name and country:
            name_country_map[f"{name}|{country}"] = sid

    return ticker_exch_map, name_country_map


def _build_lookup_maps(
    existing_master: pd.DataFrame,
    existing_xref: pd.DataFrame,
) -> Dict[str, Dict[str, int]]:
    """
    Build ``{identifier_type: {value: security_id}}`` lookup dicts used by
    the matching cascade.
    """
    xref_ric_map = _build_xref_ric_map(existing_xref)
    log.info("Xref RIC map: %d existing Refinitiv mappings.", len(xref_ric_map))

    ticker_exch_map, name_country_map = _build_composite_maps(existing_master)

    return {
        # Priority 1 - RIC via xref
        "xref_ric":    xref_ric_map,
        # Priority 2 - standard identifiers
        "isin":        _build_simple_id_map(existing_master, "isin"),
        "cusip":       _build_simple_id_map(existing_master, "cusip"),
        "figi":        _build_simple_id_map(existing_master, "figi"),
        "sedol":       _build_simple_id_map(existing_master, "sedol"),
        # Priority 3 - supplementary identifiers
        "valor":       _build_simple_id_map(existing_master, "valor"),
        "wkn":         _build_simple_id_map(existing_master, "wkn"),
        "common_code": _build_simple_id_map(existing_master, "common_code"),
        "perm_id":     _build_simple_id_map(existing_master, "perm_id"),
        # Priority 4 - composite keys (last resort)
        "ticker_exch":  ticker_exch_map,
        "name_country": name_country_map,
    }


def _cascade_lookup(
    row: pd.Series,
    maps: Dict[str, Dict[str, int]],
) -> Optional[int]:
    """
    Walk the identifier cascade for a single incoming row and return the first
    matching ``security_id``, or ``None`` if no match is found.
    """
    for map_name, row_key in IDENTIFIER_CASCADE:
        key_value = row.get(row_key)
        if key_value is not None:
            sid = maps[map_name].get(key_value)
            if sid is not None:
                return sid
    return None


# ===================================================================
# Step 5 - Upsert security_master and resolve security_ids
# ===================================================================

def _insert_new_securities(
    rows: List[dict],
    orm_engine: sql.Engine,
) -> None:
    """Batch-insert new securities into ``security_master``."""
    log.info("Inserting %d new securities into security_master…", len(rows))
    for start in range(0, len(rows), DB_BATCH_SIZE):
        batch = rows[start: start + DB_BATCH_SIZE]
        with Session(orm_engine) as session:
            database._execute_with_session(
                session,
                lambda data: session.bulk_insert_mappings(database.SecurityMaster, data),
                batch,
            )
        log.info("  Inserted rows %d-%d.", start + 1, start + len(batch))


def upsert_security_master(
    master_df: pd.DataFrame,
    orm_session: Session,
    orm_engine: sql.Engine,
) -> pd.DataFrame:
    """
    Resolve each incoming row to an existing ``security_id`` via the cascade,
    insert genuinely new securities, then return *master_df* with ``security_id``
    fully populated.
    """
    log.info("Loading existing security_master and vendor xref from DB…")
    existing_master = database.read_security_master(orm_session, orm_engine)
    existing_xref = database.read_security_vendor_xref(orm_session, orm_engine, vendor=VENDOR)
    maps = _build_lookup_maps(existing_master, existing_xref)

    # --- First pass: match existing securities ---
    security_ids: List[Optional[int]] = []
    rows_to_insert: List[dict] = []

    for _, row in master_df.iterrows():
        sid = _cascade_lookup(row, maps)
        security_ids.append(sid)
        if sid is None:
            rows_to_insert.append({
                k: v for k, v in row.items()
                if not k.startswith(_INTERNAL_PREFIX)
            })

    master_df = master_df.copy()
    master_df["security_id"] = security_ids

    matched = sum(1 for s in security_ids if s is not None)
    log.info("Cascade results: %d matched, %d new (to insert).", matched, len(rows_to_insert))

    # --- Insert new securities and re-resolve ---
    if rows_to_insert:
        _insert_new_securities(rows_to_insert, orm_engine)

        log.info("Reloading tables to resolve newly created security_ids…")
        existing_master = database.read_security_master(orm_session, orm_engine)
        existing_xref = database.read_security_vendor_xref(orm_session, orm_engine, vendor=VENDOR)
        maps = _build_lookup_maps(existing_master, existing_xref)

        for idx, row in master_df.iterrows():
            if pd.isna(master_df.at[idx, "security_id"]):
                master_df.at[idx, "security_id"] = _cascade_lookup(row, maps)
    else:
        log.info("All %d securities already exist in security_master.", len(master_df))

    # --- Drop unresolvable rows ---
    unresolved = master_df["security_id"].isna().sum()
    if unresolved:
        log.warning(
            "%d rows could not be resolved to a security_id and will be skipped. "
            "These have no identifiers whatsoever - check source data quality.",
            unresolved,
        )

    master_df = master_df.dropna(subset=["security_id"])
    master_df["security_id"] = master_df["security_id"].astype(int)
    return master_df


# ===================================================================
# Step 6 - Attach resolved security_ids to xref and write
# ===================================================================

def upsert_vendor_xref(
    master_df: pd.DataFrame,
    xref_df: pd.DataFrame,
    orm_engine: sql.Engine,
) -> None:
    """
    Join resolved ``security_id`` values into *xref_df* (keyed on ``_ric``),
    then batch-upsert into ``security_vendor_xref``.
    """
    ric_to_sid = master_df.set_index("_ric")["security_id"].to_dict()

    xref_df = xref_df.copy()
    xref_df["security_id"] = xref_df["_ric"].map(ric_to_sid)
    xref_df = xref_df.dropna(subset=["security_id"])
    xref_df["security_id"] = xref_df["security_id"].astype(int)

    # Drop internal columns before writing to the database
    xref_clean = xref_df.drop(
        columns=[c for c in xref_df.columns if c.startswith(_INTERNAL_PREFIX)]
    )

    log.info("Writing %d vendor xref rows…", len(xref_clean))
    for start in range(0, len(xref_clean), DB_BATCH_SIZE):
        batch_df = xref_clean.iloc[start: start + DB_BATCH_SIZE]
        with Session(orm_engine) as session:
            database.write_security_vendor_xref(batch_df, session, vendor=VENDOR)
        log.info("  Xref batch %d-%d written.", start + 1, start + len(batch_df))


# ===================================================================
# Orchestrator
# ===================================================================

def main() -> None:
    """End-to-end pipeline: fetch → enrich → merge → upsert."""
    log.info("=" * 60)
    log.info("Refinitiv Security Master Ingestion - START")
    log.info("=" * 60)

    # --- Phase 1: Extract from Refinitiv ---
    rd.open_session()
    log.info("Refinitiv session opened.")

    try:
        universe_df = fetch_refinitiv_universe()
        rics = universe_df["RIC"].dropna().unique().tolist()
        enrich_df = fetch_enrichment_data(rics)
        master_df, xref_df = build_master_and_xref(universe_df, enrich_df)
        log.info(
            "Built %d candidate master rows and %d xref rows.",
            len(master_df),
            len(xref_df),
        )
    finally:
        rd.close_session()
        log.info("Refinitiv session closed.")

    # --- Phase 2: Load into database ---
    orm_engine, orm_conn, _, orm_session = database.get_db_connection()

    try:
        master_df = upsert_security_master(master_df, orm_session, orm_engine)
        log.info("security_master upsert complete. Resolved %d security_ids.", len(master_df))

        upsert_vendor_xref(master_df, xref_df, orm_engine)
        log.info("security_vendor_xref upsert complete.")
    finally:
        orm_conn.close()
        log.info("DB connection closed.")

    log.info("=" * 60)
    log.info("Refinitiv Security Master Ingestion - DONE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
