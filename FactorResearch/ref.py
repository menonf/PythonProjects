def fetch_refinitiv_universe() -> pd.DataFrame:
    """
    Page through ``rd.discovery.search`` for all equity quotes.

    Uses ``skip`` for offset-based pagination and stops when a page is empty
    or shorter than ``PAGE_SIZE``.
    """
    all_pages: List[pd.DataFrame] = []
    skip = 0
    page_count = 0

    log.info("Starting Refinitiv universe fetch (page_size=%d)…", PAGE_SIZE)

    while True:
        page = _api_call_with_retry(
            lambda: rd.discovery.search(
                view=rd.discovery.Views.EQUITY_QUOTES,
                filter="AssetType eq 'equity' and RCSExchangeCountryLeaf eq 'India'",  # example filter - adjust as needed
                select=SEARCH_SELECT,
                top=PAGE_SIZE,
            ),
            description=f"Universe search (skip={skip})",
        )

        if page is None or page.empty:
            log.info("Empty page at skip=%d - fetch complete.", skip)
            break 

        page_count += 1
        all_pages.append(page)
        fetched = len(page)
        skip += fetched
        log.info("Page %d: fetched %d rows (running total: %d).", page_count, fetched, skip)

        if MAX_UNIVERSE_PAGES and page_count >= MAX_UNIVERSE_PAGES:
            log.info("Reached MAX_UNIVERSE_PAGES=%d; stopping test run.", MAX_UNIVERSE_PAGES)
            break

        if fetched < PAGE_SIZE:
            break  # last partial page

    if not all_pages:
        raise RuntimeError("Refinitiv search returned no data.")

    df = pd.concat(all_pages, ignore_index=True).drop_duplicates(subset=["RIC"])
    log.info("Universe: %d unique RICs after de-duplication.", len(df))
    return df
