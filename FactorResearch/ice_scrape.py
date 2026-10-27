"""
ICE Index Viewer -> Constituents download, for every month-end in a date range.
RUN FROM A TERMINAL (not the VS Code Interactive Window):
    python ice_scrape.py

Prereq: Chrome started with a debug port (close all Chrome first):
"C:\Program Files\Google\Chrome\Application\chrome.exe" ^
  --remote-debugging-port=9222 ^
  --user-data-dir="C:\chrome-ice-profile""
Log into ICE once in that window.
"""

import calendar
import time
from datetime import date
from pathlib import Path
from playwright.sync_api import sync_playwright

# ----------------- parameters -----------------
INDICES = ["HPID"]                 # e.g. ["H0A0", "C0A0", "G0BC"]
FROM_DATE = date(2005, 1, 1)       # inclusive
TO_DATE   = date(2012, 1, 1)      # inclusive
BUSINESS_DAY_END = False           # True = last business day instead of last calendar day
DOWNLOAD_DIR = Path(r"C:\Users\feme\Downloads\ice")
# ----------------------------------------------

URL = ("https://indices.ice.com/bondIndexAnalysis/showIndexViewerPage"
       "?pageTitle=Index%20Viewer")

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def month_ends(start: date, end: date, business: bool = False):
    """
    Every month-end falling within [start, end], oldest first.
    A month-end is only included if it lies inside the range, so a partial
    first or last month is skipped rather than half-reported.
    """
    out = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        last = calendar.monthrange(y, m)[1]
        d = date(y, m, last)
        if business:
            while d.weekday() >= 5:          # 5=Sat, 6=Sun
                d = d.replace(day=d.day - 1)
        if start <= d <= end:
            out.append(d)
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def fmt(d: date) -> str:
    return d.strftime("%m/%d/%Y")          # the format the page expects


def toolbar_buttons(page):
    """Row holding the icons: 0 = eye (view), 1 = download, 2 = print."""
    date_input = page.locator("input[type='text']").nth(1)
    row = date_input.locator(
        "xpath=ancestor::*[self::div or self::form][.//button][1]"
    )
    return row.locator("button")


def set_parameters(page, index_code: str, date_str: str):
    inputs = page.locator("input[type='text']")

    idx = inputs.nth(0)
    idx.click(); idx.fill("")
    idx.type(index_code, delay=60)
    idx.press("Tab")
    page.wait_for_timeout(800)

    dt = inputs.nth(1)
    dt.click(); dt.fill("")
    dt.type(date_str, delay=60)
    dt.press("Tab")
    page.wait_for_timeout(500)


def click_eye(page):
    eye = page.locator("button[title*='View' i], button[aria-label*='view' i]")
    (eye.first if eye.count() else toolbar_buttons(page).nth(0)).click()
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(3000)


def open_constituents(page):
    tab = page.get_by_role("button", name="Constituents")
    if tab.count() == 0:
        tab = page.locator("#controlled-tab-navigator-tab-2")
    tab.first.click()
    page.wait_for_timeout(3000)


def click_download(page, index_code: str, d: date):
    stamp = d.strftime("%Y%m%d")
    dl_btn = page.locator(
        "button[title*='Download' i], button[title*='Export' i], "
        "button[aria-label*='download' i], button[aria-label*='export' i]"
    )
    target_btn = dl_btn.first if dl_btn.count() else toolbar_buttons(page).nth(1)

    try:
        with page.expect_download(timeout=180_000) as dl:
            target_btn.click()
        f = dl.value
        out = DOWNLOAD_DIR / f"{index_code}_{stamp}_{f.suggested_filename}"
        f.save_as(out)
        return out
    except Exception:
        # Some builds pop a format dialog (CSV / Excel) before downloading.
        for label in ("CSV", "Excel", "Download", "OK"):
            opt = page.get_by_role("button", name=label)
            if opt.count():
                with page.expect_download(timeout=180_000) as dl:
                    opt.first.click()
                f = dl.value
                out = DOWNLOAD_DIR / f"{index_code}_{stamp}_{f.suggested_filename}"
                f.save_as(out)
                return out
        raise


def main():
    dates = month_ends(FROM_DATE, TO_DATE, BUSINESS_DAY_END)
    print(f"{len(dates)} month-end(s) x {len(INDICES)} index/indices "
          f"= {len(dates) * len(INDICES)} download(s)")
    print("Dates:", ", ".join(fmt(d) for d in dates))

    done, failed = [], []

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        context.set_default_timeout(60_000)
        page = context.pages[0] if context.pages else context.new_page()

        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(2000)

        for code in INDICES:
            for d in dates:
                label = f"{code} @ {fmt(d)}"
                print(f"\n=== {label} ===")
                try:
                    set_parameters(page, code, fmt(d))
                    click_eye(page)
                    open_constituents(page)
                    out = click_download(page, code, d)
                    print(f"  saved -> {out}")
                    done.append(label)
                except Exception as e:
                    # One bad date shouldn't kill the whole run.
                    print(f"  FAILED: {type(e).__name__}: {e}")
                    failed.append(label)
                    page.screenshot(path=str(DOWNLOAD_DIR /
                                    f"error_{code}_{d:%Y%m%d}.png"), full_page=True)
                    page.goto(URL, wait_until="networkidle")   # reset the form
                    page.wait_for_timeout(2000)
                time.sleep(2)          # be polite to the server

        browser.close()

    print(f"\nDone: {len(done)} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed:", ", ".join(failed))


if __name__ == "__main__":
    main()