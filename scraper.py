"""
IDX Daily Scraper - Ringkasan Saham
=====================================
Downloads today's stock summary from idx.co.id,
saves a dated XLSX to /raw, and appends to master.csv.

Run manually:    python scraper.py
Run with date:   python scraper.py --date 2025-01-02
Backfill range:  python scraper.py --from 2025-10-01 --to 2025-10-31
"""

import requests
import pandas as pd
import os
import sys
import argparse
from datetime import datetime, date, timedelta
import time

from master_store import master_contains_date

# ── Folders ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RAW_DIR    = os.path.join(BASE_DIR, "raw")
MASTER_CSV = os.path.join(BASE_DIR, "master.csv")
LOG_FILE   = os.path.join(BASE_DIR, "scraper.log")

os.makedirs(RAW_DIR, exist_ok=True)

# ── IDX API endpoint (reverse-engineered from idx.co.id) ─────────────────────
IDX_API = "https://www.idx.co.id/primary/TradingSummary/GetStockSummary"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.idx.co.id/data-pasar/ringkasan-perdagangan/ringkasan-saham/",
    "Accept": "application/json, text/plain, */*",
}

# Column rename map: IDX API field → friendly name
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


def fetch_all_stocks(target_date: str) -> pd.DataFrame:
    """
    Fetches all stocks for a given date (YYYY-MM-DD) from the IDX API.
    The API returns 100 rows per page; we paginate until done.
    """
    dt = datetime.strptime(target_date, "%Y-%m-%d")
    idx_date = dt.strftime("%d %b %Y")

    all_rows = []
    start = 0
    page_size = 100
    total = None

    log(f"Fetching data for {target_date} (IDX format: {idx_date})...")

    while True:
        params = {
            "start":  start,
            "length": page_size,
            "date":   idx_date,
        }

        try:
            resp = requests.get(IDX_API, headers=HEADERS, params=params, timeout=30)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            log(f"ERROR: Request failed on page start={start}: {e}")
            raise

        data = resp.json()

        rows = data.get("data", [])
        if total is None:
            total = data.get("recordsTotal", 0)
            log(f"Total stocks reported by IDX: {total}")

        if not rows:
            break

        all_rows.extend(rows)
        log(f"  Fetched {len(all_rows)}/{total} stocks...")

        start += page_size
        if start >= total:
            break

        time.sleep(0.5)

    if not all_rows:
        raise ValueError(
            f"No data returned for {target_date}. "
            "Market may be closed or IDX API structure changed."
        )

    df = pd.DataFrame(all_rows)
    df = df.rename(columns=COLUMN_MAP)
    known_cols = list(COLUMN_MAP.values())
    df = df[[c for c in known_cols if c in df.columns]]

    log(f"Successfully fetched {len(df)} stocks for {target_date}.")
    return df


def save_raw_xlsx(df: pd.DataFrame, target_date: str):
    """Saves a dated XLSX to the /raw folder."""
    date_str = datetime.strptime(target_date, "%Y-%m-%d").strftime("%Y%m%d")
    filename = f"Ringkasan Saham-{date_str}.xlsx"
    path = os.path.join(RAW_DIR, filename)
    df.to_excel(path, index=False)
    log(f"Saved raw file: {path}")
    return path


def append_to_master(df: pd.DataFrame, target_date: str):
    """
    Appends today's data to master.csv.
    Adds a DATE column as the first column.
    Skips if this date already exists in master.csv.
    """
    df_copy = df.copy()
    df_copy.insert(0, "DATE", target_date)

    if os.path.exists(MASTER_CSV):
        if master_contains_date(MASTER_CSV, target_date):
            log(f"SKIP: {target_date} already exists in master.csv.")
            return
        df_copy.to_csv(MASTER_CSV, mode="a", header=False, index=False)
        log(f"Appended {len(df_copy)} rows to master.csv for {target_date}.")
    else:
        df_copy.to_csv(MASTER_CSV, index=False)
        log(f"Created master.csv with {len(df_copy)} rows for {target_date}.")


def run(target_date: str):
    log(f"=== IDX Scraper START for {target_date} ===")

    dt = datetime.strptime(target_date, "%Y-%m-%d")
    if dt.weekday() >= 5:
        log(f"SKIP: {target_date} is a weekend. IDX is closed.")
        return

    # Check if already scraped (raw XLSX exists)
    date_str = dt.strftime("%Y%m%d")
    raw_path = os.path.join(RAW_DIR, f"Ringkasan Saham-{date_str}.xlsx")
    if os.path.exists(raw_path):
        log(f"Raw file already exists for {target_date}, skipping fetch.")
        # Still try to append to master if missing
        df = pd.read_excel(raw_path)
        append_to_master(df, target_date)
        log(f"=== IDX Scraper DONE for {target_date} ===\n")
        return

    df = fetch_all_stocks(target_date)
    save_raw_xlsx(df, target_date)
    append_to_master(df, target_date)

    log(f"=== IDX Scraper DONE for {target_date} ===\n")


def run_range(from_date: str, to_date: str):
    """Backfill a date range, skipping weekends and already-scraped dates.
    Errors on individual days are logged and skipped so the range continues."""
    start = datetime.strptime(from_date, "%Y-%m-%d")
    end   = datetime.strptime(to_date,   "%Y-%m-%d")
    current = start
    failed = []
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        try:
            run(date_str)
        except Exception as e:
            log(f"SKIP {date_str}: {e}")
            failed.append(date_str)
        current += timedelta(days=1)
        time.sleep(1)
    if failed:
        log(f"Backfill complete. {len(failed)} dates failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDX Ringkasan Saham daily scraper")
    parser.add_argument("--date", type=str, default=None,
                        help="Single date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--from", dest="from_date", type=str, default=None,
                        help="Start date for backfill range (YYYY-MM-DD)")
    parser.add_argument("--to",   dest="to_date",   type=str, default=None,
                        help="End date for backfill range (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.from_date and args.to_date:
        run_range(args.from_date, args.to_date)
    elif args.from_date:
        run_range(args.from_date, date.today().strftime("%Y-%m-%d"))
    else:
        run(args.date or date.today().strftime("%Y-%m-%d"))
