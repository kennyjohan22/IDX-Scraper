"""
build_master.py
===============
One-time (and idempotent) script that reads ALL Ringkasan Saham-*.xlsx files
in the raw/ folder and consolidates them into master.csv.

Run:  python build_master.py
      python build_master.py --rebuild      # wipe master.csv and rebuild from scratch
"""

import os
import re
import argparse
import pandas as pd
from datetime import datetime

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RAW_DIR    = os.path.join(BASE_DIR, "raw")
MASTER_CSV = os.path.join(BASE_DIR, "master.csv")


def extract_date_from_filename(filename: str):
    """Parse YYYYMMDD from 'Ringkasan Saham-YYYYMMDD.xlsx'."""
    m = re.search(r"(\d{8})", filename)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d").strftime("%Y-%m-%d")


def load_existing_dates() -> set:
    if not os.path.exists(MASTER_CSV):
        return set()
    df = pd.read_csv(MASTER_CSV, usecols=["DATE"])
    return set(df["DATE"].unique())


def build_master(rebuild: bool = False):
    if rebuild and os.path.exists(MASTER_CSV):
        os.remove(MASTER_CSV)
        print(f"Wiped existing master.csv for full rebuild.")

    existing_dates = load_existing_dates()

    # Find all XLSX files in raw/
    files = sorted([
        f for f in os.listdir(RAW_DIR)
        if f.endswith(".xlsx") and re.search(r"\d{8}", f)
    ])
    print(f"Found {len(files)} XLSX files in raw/")

    appended = 0
    skipped  = 0

    for filename in files:
        target_date = extract_date_from_filename(filename)
        if target_date is None:
            print(f"  SKIP (unparseable date): {filename}")
            continue

        if target_date in existing_dates:
            skipped += 1
            continue

        path = os.path.join(RAW_DIR, filename)
        try:
            df = pd.read_excel(path)
        except Exception as e:
            print(f"  ERROR reading {filename}: {e}")
            continue

        df.insert(0, "DATE", target_date)

        write_header = not os.path.exists(MASTER_CSV)
        df.to_csv(MASTER_CSV, mode="a", header=write_header, index=False)
        existing_dates.add(target_date)
        appended += 1
        print(f"  Added {target_date} ({len(df)} stocks)")

    print(f"\nDone. Added {appended} dates, skipped {skipped} already-present dates.")
    print(f"master.csv now covers {len(existing_dates)} trading days.")

    # Print date range summary
    all_dates = sorted(existing_dates)
    if all_dates:
        print(f"Date range: {all_dates[0]}  →  {all_dates[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build master.csv from raw XLSX files")
    parser.add_argument("--rebuild", action="store_true",
                        help="Delete existing master.csv and rebuild from scratch")
    args = parser.parse_args()
    build_master(rebuild=args.rebuild)
