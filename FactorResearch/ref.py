def fetch_refinitiv_universe() -> pd.DataFrame:
    """
    Page through ``rd.discovery.search`` for all equity quotes.

    Uses filter-based partitioning to work around the API's 10,000 offset limit,
    then paginates within each partition.
    """
    MAX_OFFSET = 10000
    
    # Partition by first letter of RIC to stay under 10k per partition
    # Adjust partitions based on your data distribution
    PARTITIONS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    
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
