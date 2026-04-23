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

# ── Browser wrapper ──────────────────────────────────────────────────────────

class ExploreAutomation:
    """Drives an Edge browser session against the Aladdin Explore workspace."""

    def __init__(self, driver_path: str):
        svc = service.Service(driver_path)
        self.driver = webdriver.Edge(service=svc)
        self.driver.maximize_window()
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
   

    # ── Templates ───────────────────────────────────────────────────────

    def load_template(self,
        download_dir: str | None = None,
    ) -> None:
        """Click the three-dot kebab menu icon, then select 'Batch Exporting'."""
        print("Clicking template menu icon")
        icon = self.driver.find_element(
            By.CSS_SELECTOR,
            "aux-icon[type='action'][state='primary'][slot='target']",
        )
        icon.click()
        time.sleep(MEDIUM_WAIT)

        print("Selecting 'Batch Exporting'")
        menu_items = self.driver.find_elements(
            By.CSS_SELECTOR, "li.aux-inline-menu__item-wrapper"
        )
        for item in menu_items:
            if "Batch Exporting" in item.text:
                item.click()
                break
        time.sleep(MEDIUM_WAIT)

        print("Clicking 'Load'")
        load_btn = self.driver.find_element(
            By.CSS_SELECTOR, "aux-button[label='Load']"
        )
        shadow = _get_shadow_root(self.driver, load_btn)
        shadow.find_element(By.CSS_SELECTOR, "button").click()
        time.sleep(MEDIUM_WAIT)

        print("Expanding 'My batch reports' accordion")
        accordion = self.driver.find_element(
            By.CSS_SELECTOR,
            "aux-accordion-expansion-panel[header='My batch reports']",
        )
        acc_root = _get_shadow_root(self.driver, accordion)
        acc_root.find_element(
            By.CSS_SELECTOR,
            "button[data-test='aux-accordion-expansion-panel__header-button']",
        ).click()
        time.sleep(MEDIUM_WAIT)

        print("Selecting 'Limits QC Batch v12'")
        tree_list = accordion.find_element(
            By.CSS_SELECTOR, "aux-advanced-tree-list"
        )
        tree_root = _get_shadow_root(self.driver, tree_list)
        tree_root.find_element(
            By.CSS_SELECTOR, "div[role='treeitem']"
        ).click()
        time.sleep(MEDIUM_WAIT)

        print("Clicking batch settings checkbox")
        checkbox_div = self.driver.find_element(
            By.CSS_SELECTOR, "div.batch-settings-active-checkbox"
        )
        aux_checkbox = checkbox_div.find_element(
            By.CSS_SELECTOR, "aux-checkbox"
        )
        cb_root = _get_shadow_root(self.driver, aux_checkbox)
        cb_input = cb_root.find_element(
            By.CSS_SELECTOR, "input[data-test='aux-checkbox--input']"
        )
        self.driver.execute_script("arguments[0].click();", cb_input)
        time.sleep(MEDIUM_WAIT)

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

    def run_batch(self, poll_interval: int = 120) -> None:
        """Click 'Run', confirm it changes to 'Cancel', then poll until it reverts to 'Run'."""
        print("Clicking 'Run' button")
        run_btn = self.driver.find_element(
            By.CSS_SELECTOR, "aux-button[label='Run']"
        )
        run_shadow = _get_shadow_root(self.driver, run_btn)
        run_shadow.find_element(By.CSS_SELECTOR, "button").click()
        time.sleep(LONG_WAIT)

        # Verify the button changed to Cancel
        try:
            self.driver.find_element(
                By.CSS_SELECTOR, "aux-button[label='Cancel']"
            )
            print("Batch is running (button shows 'Cancel')")
        except Exception:
            print("Warning: 'Cancel' button not found – batch may not have started")
            return

        # Poll every poll_interval seconds until Run button reappears
        while True:
            print(f"Waiting {poll_interval // 60} minutes before next check...")
            time.sleep(poll_interval)
            run_buttons = self.driver.find_elements(
                By.CSS_SELECTOR, "aux-button[label='Run']"
            )
            if run_buttons:
                print("Batch complete – 'Run' button has reappeared")
                break
            else:
                print("Batch still running (button still shows 'Cancel')")

    # ── Cleanup ──────────────────────────────────────────────────────────

    def quit(self) -> None:
        self.driver.quit()


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:

    DEFAULT_DOWNLOAD_DIR = os.path.join("explore_output", "two_week_changes")
    WORKSPACE_URL = r"https://barings.blackrock.com/apps/explore/?workspace=36665"
    EDGE_DRIVER_PATH = r".\msedgedriver.exe"

    bot = ExploreAutomation(EDGE_DRIVER_PATH)
    try:
        bot.login(WORKSPACE_URL)
        time.sleep(SHORT_WAIT)
        bot.select_report_tab("Limits Export")
        bot.load_template(DEFAULT_DOWNLOAD_DIR)
        bot.run_batch()
        print("finished")
        

    finally:
        bot.quit()


if __name__ == "__main__":
    main()
