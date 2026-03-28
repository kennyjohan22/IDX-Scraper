"""
scraper_browser.py — IDX Scraper with Playwright (Cloudflare bypass)
=====================================================================
Uses a real Chromium browser to establish a Cloudflare session, then
makes the IDX API calls from *inside* the browser context so CF cookies
are automatically included.

Run manually:    python scraper_browser.py
Run with date:   python scraper_browser.py --date 2025-11-07
Backfill range:  python scraper_browser.py --from 2025-11-07 --to 2026-03-28
"""

import os
import re
import json
import argparse
import time
import pandas as pd
from datetime import datetime, date, timedelta
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RAW_DIR    = os.path.join(BASE_DIR, "raw")
MASTER_CSV = os.path.join(BASE_DIR, "master.csv")
LOG_FILE   = os.path.join(BASE_DIR, "scraper.log")

os.makedirs(RAW_DIR, exist_ok=True)

IDX_PAGE = "https://www.idx.co.id/id/data-pasar/ringkasan-perdagangan/ringkasan-saham/"
IDX_API  = "https://www.idx.co.id/primary/TradingSummary/GetStockSummary"

COLUMN_MAP = {
    "No":                   "No",
    "StockCode":            "Kode Saham",
    "StockName":            "Nama Perusahaan",
    "Remarks":              "Remarks",
    "Previous":             "Sebelumnya",
    "OpenPrice":            "Open Price",
    "LastTradingDate":      "Tanggal Perdagangan Terakhir",
    "FirstTrade":           "First Trade",
    "High":                 "Tertinggi",
    "Low":                  "Terendah",
    "Close":                "Penutupan",
    "Change":               "Selisih",
    "Volume":               "Volume",
    "Value":                "Nilai",
    "Frequency":            "Frekuensi",
    "IndexIndividual":      "Index Individual",
    "Offer":                "Offer",
    "OfferVolume":          "Offer Volume",
    "Bid":                  "Bid",
    "BidVolume":            "Bid Volume",
    "ListedShares":         "Listed Shares",
    "TradebleShares":       "Tradeble Shares",
    "WeightForIndex":       "Weight For Index",
    "ForeignSell":          "Foreign Sell",
    "ForeignBuy":           "Foreign Buy",
    "NonRegularVolume":     "Non Regular Volume",
    "NonRegularValue":      "Non Regular Value",
    "NonRegularFrequency":  "Non Regular Frequency",
}


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def _fetch_page_in_browser(page, idx_date: str, start: int, length: int = 100) -> dict:
    """
    Execute a fetch() call from inside the Playwright browser context.
    This reuses the browser's Cloudflare session cookies automatically.
    """
    url = IDX_API
    js = f"""
    async () => {{
        const params = new URLSearchParams({{
            start: '{start}',
            length: '{length}',
            date: '{idx_date}'
        }});
        const resp = await fetch(`{url}?${{params}}`, {{
            headers: {{
                'Accept': 'application/json, text/plain, */*',
                'Referer': '{IDX_PAGE}',
            }}
        }});
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        return await resp.json();
    }}
    """
    result = page.evaluate(js)
    return result


def fetch_all_stocks_browser(page, target_date: str) -> pd.DataFrame:
    """Paginate through all IDX stocks for target_date using in-browser fetch."""
    dt = datetime.strptime(target_date, "%Y-%m-%d")
    idx_date = dt.strftime("%d %b %Y")

    all_rows = []
    start = 0
    page_size = 100
    total = None

    log(f"Fetching {target_date} (IDX: {idx_date})...")

    while True:
        try:
            data = _fetch_page_in_browser(page, idx_date, start, page_size)
        except Exception as e:
            log(f"ERROR fetching page start={start}: {e}")
            raise

        rows = data.get("data", [])
        if total is None:
            total = data.get("recordsTotal", 0)
            log(f"  Total stocks reported: {total}")

        if not rows:
            break

        all_rows.extend(rows)
        log(f"  Fetched {len(all_rows)}/{total}...")

        # Stop if we have all records, or if the API returned a short page
        # (some IDX API versions return all rows in one shot ignoring `length`)
        if len(all_rows) >= total or len(rows) < page_size:
            break

        start += page_size
        if start >= total:
            break

        time.sleep(0.3)

    if not all_rows:
        raise ValueError(
            f"No data returned for {target_date}. "
            "Market may be closed or date not available."
        )

    df = pd.DataFrame(all_rows)
    df = df.rename(columns=COLUMN_MAP)
    known = list(COLUMN_MAP.values())
    df = df[[c for c in known if c in df.columns]]
    log(f"  Successfully fetched {len(df)} stocks.")
    return df


def save_raw_xlsx(df: pd.DataFrame, target_date: str) -> str:
    date_str = datetime.strptime(target_date, "%Y-%m-%d").strftime("%Y%m%d")
    filename = f"Ringkasan Saham-{date_str}.xlsx"
    path = os.path.join(RAW_DIR, filename)
    df.to_excel(path, index=False)
    log(f"Saved: {path}")
    return path


def append_to_master(df: pd.DataFrame, target_date: str):
    df_copy = df.copy()
    df_copy.insert(0, "DATE", target_date)

    if os.path.exists(MASTER_CSV):
        existing = pd.read_csv(MASTER_CSV, usecols=["DATE"], nrows=10000)
        if target_date in existing["DATE"].values:
            log(f"SKIP: {target_date} already in master.csv.")
            return
        # Align to master schema — fill any missing columns with None so
        # rows are never position-shifted when the IDX API drops a field.
        master_cols = pd.read_csv(MASTER_CSV, nrows=0).columns.tolist()
        for col in master_cols:
            if col not in df_copy.columns:
                df_copy[col] = None
        df_copy = df_copy[master_cols]
        df_copy.to_csv(MASTER_CSV, mode="a", header=False, index=False)
        log(f"Appended {len(df_copy)} rows to master.csv for {target_date}.")
    else:
        df_copy.to_csv(MASTER_CSV, index=False)
        log(f"Created master.csv with {len(df_copy)} rows for {target_date}.")


def run_dates(dates: list[str]):
    """
    Fetch a list of date strings using a single persistent browser session.
    Skips weekends and dates that already have a raw XLSX.
    """
    # Filter out weekends and already-done dates upfront
    to_fetch = []
    for d in dates:
        dt = datetime.strptime(d, "%Y-%m-%d")
        if dt.weekday() >= 5:
            continue
        date_str = dt.strftime("%Y%m%d")
        raw_path = os.path.join(RAW_DIR, f"Ringkasan Saham-{date_str}.xlsx")
        if os.path.exists(raw_path):
            log(f"SKIP {d}: raw file exists, appending to master if needed.")
            existing_df = pd.read_excel(raw_path)
            append_to_master(existing_df, d)
            continue
        to_fetch.append(d)

    if not to_fetch:
        log("Nothing new to fetch.")
        return

    log(f"Starting browser session for {len(to_fetch)} date(s)...")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="id-ID",
        )
        page = context.new_page()

        # Visit the IDX page once to get Cloudflare clearance cookies
        log("Loading IDX page to establish Cloudflare session...")
        try:
            page.goto(IDX_PAGE, wait_until="domcontentloaded", timeout=45000)
            # Wait a moment for CF JS challenge to complete
            page.wait_for_timeout(3000)
            log("IDX page loaded.")
        except PWTimeout:
            log("WARNING: IDX page load timed out — attempting API calls anyway.")

        failed = []
        for target_date in to_fetch:
            log(f"=== START {target_date} ===")
            try:
                df = fetch_all_stocks_browser(page, target_date)
                save_raw_xlsx(df, target_date)
                append_to_master(df, target_date)
                log(f"=== DONE {target_date} ===\n")
            except Exception as e:
                log(f"=== FAIL {target_date}: {e} ===\n")
                failed.append(target_date)

            # Brief pause between dates to avoid hammering the server
            time.sleep(1.5)

        browser.close()

    if failed:
        log(f"Backfill complete. {len(failed)} date(s) failed: {failed}")
    else:
        log("All dates fetched successfully.")


def run_single(target_date: str):
    run_dates([target_date])


def run_range(from_date: str, to_date: str):
    start = datetime.strptime(from_date, "%Y-%m-%d")
    end   = datetime.strptime(to_date,   "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    run_dates(dates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDX browser scraper (Cloudflare bypass)")
    parser.add_argument("--date",      type=str, default=None)
    parser.add_argument("--from",      dest="from_date", type=str, default=None)
    parser.add_argument("--to",        dest="to_date",   type=str, default=None)
    args = parser.parse_args()

    if args.from_date:
        to = args.to_date or date.today().strftime("%Y-%m-%d")
        run_range(args.from_date, to)
    else:
        run_single(args.date or date.today().strftime("%Y-%m-%d"))
