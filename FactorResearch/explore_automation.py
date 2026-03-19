"""Explore Automation – Aladdin/Blackrock Explore workspace scraper."""

import os
import time
import datetime as dt

import keyring
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge import service

SHORT_WAIT = 2
MEDIUM_WAIT = 10
LONG_WAIT = 20
LOGIN_WAIT = 45

# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_shadow_root(driver, element):
    """Return the shadow root of a web element."""
    return driver.execute_script("return arguments[0].shadowRoot", element)


def _parse_date_string(date_str: str):
    """Parse an 'MMddYYYY' string and return (month_name, year_str, day_int)."""
    month_int = int(date_str[:2])
    day = int(date_str[2:4])
    year = date_str[4:]
    month_name = dt.date(1900, month_int, 1).strftime("%B")
    return month_name, year, day


# ── Browser wrapper ──────────────────────────────────────────────────────────


class ExploreAutomation:
    """Drives an Edge browser session against the Aladdin Explore workspace."""

    def __init__(self, driver_path: str):
        svc = service.Service(driver_path)
        self.driver = webdriver.Edge(service=svc)
        self.username = keyring.get_password("aladdin", "username")
        time.sleep(15)

    # ── Authentication ───────────────────────────────────────────────────

    def login(self, url: str) -> None:
        """Open the workspace URL and authenticate via Okta."""
        print("Opening Browser")
        self.driver.get(url)
        print("Waiting for browser to load")
        time.sleep(MEDIUM_WAIT)

        username_field = self.driver.find_element(By.ID, "idp-discovery-username")
        username_field.click()
        username_field.send_keys(self.username)
        username_field.send_keys(Keys.ENTER)
        time.sleep(LOGIN_WAIT)

    # ── Navigation ───────────────────────────────────────────────────────

    def select_report_tab(self, report_name: str = "Flat Download") -> None:
        """Click a named tab in the workspace tab-bar."""
        print(f"Selecting Tab: {report_name}")
        tab_bar = self.driver.find_element(By.XPATH, "//aux-tab-bar")
        root = _get_shadow_root(self.driver, tab_bar)
        root.find_element(
            By.CSS_SELECTOR, f"aux-tab-bar-item[label='{report_name}']"
        ).click()
        time.sleep(MEDIUM_WAIT)

    def select_portfolio(self, portfolio_name: str) -> None:
        """Select a portfolio from the workspace side-bar."""
        print(f"Selecting Portfolio: {portfolio_name}")
        ws_drawer = self.driver.find_element(
            By.CSS_SELECTOR, "aux-drawer[header='Workspace']"
        )
        portfolio_list = ws_drawer.find_element(
            By.CSS_SELECTOR, "div[class='report-group-content-container']"
        )
        for selector in portfolio_list.find_elements(
            By.CSS_SELECTOR, "app-side-bar-portfolio"
        ):
            port_div = selector.find_element(
                By.CSS_SELECTOR,
                "div.portfolio-in-report-group.portfolio-name.aux-primary.aux-p13.portfolio-ticker",
            )
            if port_div.get_attribute("innerHTML").strip() == portfolio_name:
                port_div.location_once_scrolled_into_view
                port_div.click()
                break
        time.sleep(LONG_WAIT)

    # ── Date picker ──────────────────────────────────────────────────────

    def pick_date(self, date_str: str = "04152024") -> None:
        """Set a date via the aux-date-picker (format: MMddYYYY)."""
        month, year, day = _parse_date_string(date_str)
        print(f"Pick Date: {date_str}  →  {month} {day}, {year}")

        datepicker = self.driver.find_element(By.XPATH, "//aux-date-picker")
        root = _get_shadow_root(self.driver, datepicker)
        root.find_element(By.CSS_SELECTOR, "aux-button").click()
        time.sleep(SHORT_WAIT)

        calendar = self.driver.find_element(By.XPATH, "//aux-calendar")
        cal_root = _get_shadow_root(self.driver, calendar)

        # Year
        selects = cal_root.find_elements(By.CSS_SELECTOR, "aux-select")
        selects[1].click()
        time.sleep(0.5)
        for btn in self.driver.find_elements(
            By.CSS_SELECTOR, "span.aux-select__option-span"
        ):
            if btn.get_attribute("title") == year:
                btn.click()
                break
        time.sleep(0.5)

        # Month
        selects = cal_root.find_elements(By.CSS_SELECTOR, "aux-select")
        selects[0].click()
        time.sleep(0.5)
        month_buttons = self.driver.find_elements(
            By.CSS_SELECTOR, "span.aux-select__option-span"
        )
        for btn in month_buttons:
            if btn.get_attribute("title") == month:
                btn.click()
                break
        time.sleep(0.5)

        # Day
        for btn in cal_root.find_elements(
            By.CSS_SELECTOR, f"button[data-month='{month}']"
        ):
            if btn.get_attribute("data-date") == str(day):
                btn.click()
                break
        time.sleep(MEDIUM_WAIT)

    # ── Actions ──────────────────────────────────────────────────────────

    def hit_reload(self) -> None:
        """Click the 'Reload' notification button (with fallback)."""
        print("Reloading")
        try:
            report = self.driver.find_element(By.XPATH, "//app-report-presenter")
            notif_group = report.find_element(By.CSS_SELECTOR, "aux-notification-group")
            root = _get_shadow_root(self.driver, notif_group)
            time.sleep(SHORT_WAIT)
            notif = root.find_element(By.CSS_SELECTOR, "aux-notification")
            root2 = _get_shadow_root(self.driver, notif)
            time.sleep(SHORT_WAIT)
            root2.find_element(By.CSS_SELECTOR, "aux-button").click()
            time.sleep(MEDIUM_WAIT)
        except Exception:
            print("Fallback: clicking generic Reload button")
            self.driver.find_element(
                By.CSS_SELECTOR, "aux-button[label='Reload']"
            ).click()
            time.sleep(LONG_WAIT)

    def export_widgets(
        self,
        download_dir: str | None = None,
        expanded_reports: list[str] | None = None,
    ) -> None:
        """Export every widget on the current report tab as Excel."""
        if expanded_reports is None:
            expanded_reports = ["All_Data(Ex-Cash & Derivatives)"]

        download_dir = os.path.abspath(
            download_dir or os.path.expanduser("~\\Downloads")
        )
        os.makedirs(download_dir, exist_ok=True)

        try:
            self.driver.execute_cdp_cmd(
                "Page.setDownloadBehavior",
                {"behavior": "allow", "downloadPath": download_dir},
            )
        except Exception:
            pass

        print("Exporting widgets")
        report = self.driver.find_element(By.XPATH, "//app-report-presenter")
        widgets = report.find_elements(
            By.CSS_SELECTOR, "div.aux-widget-container.sc-aux-widget"
        )

        for i, widget in enumerate(widgets, start=1):
            print(f"  Widget {i}/{len(widgets)}")
            widget.find_element(By.CSS_SELECTOR, "aux-button[title='Export']").click()
            time.sleep(3)

            overlay = self.driver.find_element(
                By.CSS_SELECTOR, "div.aux-overlay__menu-container"
            )
            items = overlay.find_elements(
                By.CSS_SELECTOR, "li.aux-inline-menu__item-wrapper.sc-aux-inline-menu"
            )
            items[-1].click()  # Excel option
            time.sleep(SHORT_WAIT)

            widget_text = widget.text.split("\n")[0].strip()
            # Select the appropriate radio button based on widget text
            radio_group = self.driver.find_element(By.CSS_SELECTOR, "aux-radio-group")
            rg_root = _get_shadow_root(self.driver, radio_group)
            if widget_text in expanded_reports:
                target_label = "All data, expanded"
            else:
                target_label = "Currently visible data only"
            target_radio = rg_root.find_element(
                By.CSS_SELECTOR, f"aux-radio[label='{target_label}']"
            )
            radio_root = _get_shadow_root(self.driver, target_radio)
            radio_input = radio_root.find_element(By.CSS_SELECTOR, "input[type='radio']")
            self.driver.execute_script("arguments[0].click();", radio_input)
            time.sleep(SHORT_WAIT)

            self.driver.find_elements(By.XPATH, "//aux-button[@label='OK']")[0].click()
            time.sleep(LONG_WAIT)

    # ── Cleanup ──────────────────────────────────────────────────────────

    def quit(self) -> None:
        self.driver.quit()


# ── Data helpers ─────────────────────────────────────────────────────────────


def read_excel_to_df(
    file_path: str,
    skiprows: int | None = None,
    sheet_name: int | str = 0,
    usecols=None,
    dtype=None,
) -> pd.DataFrame:
    """Read an Excel file into a DataFrame, trimming at the first blank row."""
    path = os.path.abspath(os.path.expanduser(file_path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel file not found: {path}")

    df = pd.read_excel(
        path,
        sheet_name=sheet_name,
        skiprows=skiprows,
        usecols=usecols,
        dtype=dtype,
    )

    first_blank = df.index[df.isna().all(axis=1)]
    if len(first_blank):
        df = df.loc[: first_blank[0] - 1]

    return df


def rename_columns(df: pd.DataFrame, new_cols: list[str]) -> pd.DataFrame:
    """Rename DataFrame columns; warn if lengths differ."""
    if len(new_cols) == len(df.columns):
        df.columns = new_cols
    else:
        mapping = dict(zip(df.columns, new_cols))
        df.rename(columns=mapping, inplace=True)
        print(
            f"Warning: column count mismatch "
            f"(have {len(df.columns)}, got {len(new_cols)}). "
            f"Renamed {len(mapping)} columns."
        )
    return df


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    portfolios = ["BCGHYBU", "BCEHYBF", "BCUSHYB"]  # Add your portfolios here

    # ── Constants ────────────────────────────────────────────────────────────────

    DEFAULT_DOWNLOAD_DIR = os.path.join("explore_output", "two_week_changes")
    WORKSPACE_URL = r"https://barings.blackrock.com/apps/explore/?workspace=53605"
    EDGE_DRIVER_PATH = r".\msedgedriver.exe"
    # ────────────────────────────────────────────────────────────────────────────
    
    bot = ExploreAutomation(EDGE_DRIVER_PATH)
    try:
        bot.login(WORKSPACE_URL)
        bot.pick_date("03032026")
        bot.hit_reload()
        bot.select_report_tab("Two Week Changes all_data_ex_cash")

        for portfolio in portfolios:
            print(f"\n{'='*60}")
            print(f"Processing portfolio: {portfolio}")
            print(f"{'='*60}")

            bot.select_portfolio(portfolio)
            download_dir = DEFAULT_DOWNLOAD_DIR

            # Record existing files before export so we only rename new ones
            existing_files = set(os.listdir(download_dir)) if os.path.exists(download_dir) else set()

            bot.export_widgets(DEFAULT_DOWNLOAD_DIR)

            # Rename only newly downloaded Excel files for this portfolio
            for fname in os.listdir(download_dir):
                if fname.endswith(".xlsx") and fname not in existing_files:
                    original_file = os.path.join(download_dir, fname)
                    name, ext = os.path.splitext(fname)
                    renamed_file = os.path.join(download_dir, f"{name}_{portfolio}{ext}")
                    os.rename(original_file, renamed_file)
                    df = read_excel_to_df(renamed_file, skiprows=5)
                    print(df)

    finally:
        bot.quit()


if __name__ == "__main__":
    main()
