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

PAGE_SIZE = 1000            # rows per search page — kept small so top+skip never exceeds 10k
GICS_BATCH_SIZE = 5000      # RICs per get_data call (API limit)
DB_BATCH_SIZE = 1000        # rows per DB commit
RETRY_SLEEP_SEC = 5         # pause between failed API calls
MAX_API_RETRIES = 3
MAX_UNIVERSE_PAGES = None   # set to an int (e.g. 3) for test runs; None = full run

# ---------------------------------------------------------------------------
# Partition list — one search shard per exchange.
# Each exchange must have fewer than 10,000 securities so that
# (top + skip) never exceeds the Elasticsearch hard ceiling.
# Add or remove exchanges to match your target universe.
# If any single exchange has >9,000 securities, split it further
# (e.g. by asset class or alphabetical RIC range).
# ---------------------------------------------------------------------------
EXCHANGE_PARTITIONS = [
   # "NSE",   # National Stock Exchange of India
    "BSE",   # Bombay Stock Exchange
    # Add more: "MCX", "NYSE", "NASDAQ", "LSE", etc.
]

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
    ("xref_ric",     "_ric"),
    ("isin",         "_isin"),
    ("cusip",        "_cusip"),
    ("figi",         "figi"),
    ("sedol",        "_sedol"),
    ("valor",        "_valor"),
    ("wkn",          "_wkn"),
    ("common_code",  "_common_code"),
    ("perm_id",      "_perm_id"),
    ("ticker_exch",  "_ticker_exch"),
    ("name_country", "_name_country"),
]


# ===========================================================================
# Helpers
# ===========================================================================

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
            log.warning(
                "%s - attempt %d/%d failed: %s",
                description, attempt, MAX_API_RETRIES, exc,
            )
            if attempt == MAX_API_RETRIES:
                raise
            time.sleep(RETRY_SLEEP_SEC)


# ===========================================================================
# Step 1 — Fetch universe from Refinitiv (partitioned by exchange)
# ===========================================================================

def _fetch_partition(exchange: str) -> pd.DataFrame:
    """
    Fetch all equity quotes for a single *exchange* code using offset pagination.

    Refinitiv's search endpoint is backed by Elasticsearch and enforces a hard
    ceiling of ``top + skip <= 10_000``.  By fetching one exchange at a time we
    keep each shard well within that limit.  If a partition approaches 9,000
    rows a warning is emitted — split that exchange into sub-partitions.
    """
    pages: List[pd.DataFrame] = []
    skip = 0
    page_count = 0
    API_CEILING = 10_000

    while True:
        current_skip = skip  # freeze value before entering retry closure

        # Guard: abort before the API rejects the request
        if current_skip + PAGE_SIZE > API_CEILING:
            log.warning(
                "Exchange '%s' has ≥%d securities — cannot fetch beyond the "
                "Elasticsearch 10k ceiling.  Split this partition into smaller "
                "sub-groups (e.g. by asset class or alphabetical RIC range).",
                exchange,
                API_CEILING,
            )
            break

        page = _api_call_with_retry(
            lambda s=current_skip, ex=exchange: rd.discovery.search(
                view=rd.discovery.Views.EQUITY_QUOTES,
                filter=f"AssetType eq 'equity' and ExchangeCode eq '{ex}'",
                select=SEARCH_SELECT,
                top=PAGE_SIZE,
                skip=s,
            ),
            description=f"Search exchange={exchange} skip={current_skip}",
        )

        if page is None or page.empty:
            log.info("Exchange '%s': empty page at skip=%d — partition complete.", exchange, current_skip)
            break

        page_count += 1
        pages.append(page)
        fetched = len(page)
        skip += fetched
        log.info("Exchange '%s' | page %d | skip=%d | fetched=%d | running total=%d",
                 exchange, page_count, current_skip, fetched, skip)

        if MAX_UNIVERSE_PAGES and page_count >= MAX_UNIVERSE_PAGES:
            log.info("Reached MAX_UNIVERSE_PAGES=%d for exchange '%s'; stopping.", MAX_UNIVERSE_PAGES, exchange)
            break

        if fetched < PAGE_SIZE:
            break  # last partial page — we are done

    if not pages:
        log.warning("Exchange '%s' returned no data.", exchange)
        return pd.DataFrame()

    return pd.concat(pages, ignore_index=True)


def fetch_refinitiv_universe() -> pd.DataFrame:
    """
    Iterate over ``EXCHANGE_PARTITIONS`` and concatenate all results into a
    single de-duplicated DataFrame.

    Each partition is fetched independently so that no individual shard
    approaches the Refinitiv / Elasticsearch 10,000-row (top + skip) ceiling.
    """
    all_frames: List[pd.DataFrame] = []

    log.info(
        "Starting Refinitiv universe fetch across %d exchange partition(s): %s",
        len(EXCHANGE_PARTITIONS),
        EXCHANGE_PARTITIONS,
    )

    for exchange in EXCHANGE_PARTITIONS:
        df_partition = _fetch_partition(exchange)
        if not df_partition.empty:
            all_frames.append(df_partition)
            log.info("Partition '%s' complete: %d rows.", exchange, len(df_partition))

    if not all_frames:
        raise RuntimeError("Refinitiv search returned no data across all partitions.")

    df = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset=["RIC"])
    log.info(
        "Universe fetch complete: %d unique RICs across %d partition(s).",
        len(df),
        len(EXCHANGE_PARTITIONS),
    )
    return df


# ===========================================================================
# Main
# ===========================================================================

def main():

    log.info("START Refinitiv ingestion")

    rd.open_session()

    try:
        universe = fetch_refinitiv_universe()
        rics = universe["RIC"].dropna().unique().tolist()

    finally:
        rd.close_session()


    log.info("DONE")


if __name__ == "__main__":
    main()
