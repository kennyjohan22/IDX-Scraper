"""Daily IDX data refresh orchestration.

Runs the browser scraper for every missing weekday between the latest date in
master.csv and today, then refreshes the daily screener output.
"""

from __future__ import annotations

import os
import subprocess
import sys
import gzip
import shutil
from datetime import date, datetime, timedelta
from typing import Optional, Tuple

import pandas as pd

import scraper_browser


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV = os.path.join(BASE_DIR, "master.csv")
MASTER_SNAPSHOT_CSV_GZ = os.path.join(BASE_DIR, "data", "master.csv.gz")
INITIAL_BACKFILL_FROM = os.environ.get("IDX_INITIAL_BACKFILL_FROM")


def ensure_master_csv() -> None:
    if os.path.exists(MASTER_CSV) or not os.path.exists(MASTER_SNAPSHOT_CSV_GZ):
        return

    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Restoring master.csv from compressed snapshot.",
        flush=True,
    )
    with gzip.open(MASTER_SNAPSHOT_CSV_GZ, "rb") as source:
        with open(MASTER_CSV, "wb") as target:
            shutil.copyfileobj(source, target)


def write_snapshot() -> None:
    if not os.path.exists(MASTER_CSV):
        return

    os.makedirs(os.path.dirname(MASTER_SNAPSHOT_CSV_GZ), exist_ok=True)
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Updating compressed master snapshot.",
        flush=True,
    )
    with open(MASTER_CSV, "rb") as source:
        with gzip.open(MASTER_SNAPSHOT_CSV_GZ, "wb") as target:
            shutil.copyfileobj(source, target)


def latest_master_date() -> Optional[date]:
    if not os.path.exists(MASTER_CSV):
        return None

    dates = pd.read_csv(
        MASTER_CSV,
        usecols=["DATE"],
        dtype={"DATE": "string"},
        low_memory=False,
    )["DATE"]
    parsed = pd.to_datetime(dates, errors="coerce")
    latest = parsed.max()
    if pd.isna(latest):
        return None
    return latest.date()


def next_refresh_window(today: Optional[date] = None) -> Tuple[date, date]:
    today = today or date.today()
    latest = latest_master_date()
    if latest is None:
        if INITIAL_BACKFILL_FROM:
            return datetime.strptime(INITIAL_BACKFILL_FROM, "%Y-%m-%d").date(), today
        return today, today
    return latest + timedelta(days=1), today


def has_weekday(start: date, end: date) -> bool:
    current = start
    while current <= end:
        if current.weekday() < 5:
            return True
        current += timedelta(days=1)
    return False


def run_screener() -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Running screener...", flush=True)
    subprocess.run([sys.executable, "screener.py"], cwd=BASE_DIR, check=False)


def main() -> int:
    ensure_master_csv()
    start, end = next_refresh_window()
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Daily refresh window: {start} -> {end}", flush=True)

    if start <= end and has_weekday(start, end):
        scraper_browser.run_range(start.isoformat(), end.isoformat())
    else:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] No missing weekday data to fetch.", flush=True)

    run_screener()
    write_snapshot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
