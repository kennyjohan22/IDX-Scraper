"""Shared helpers for reading and updating the scraper's master dataset."""

import pandas as pd


def master_contains_date(master_csv: str, target_date: str) -> bool:
    """Return whether target_date exists anywhere in a potentially large CSV."""
    date_chunks = pd.read_csv(
        master_csv,
        usecols=["DATE"],
        dtype={"DATE": "string"},
        chunksize=100_000,
    )
    with date_chunks:
        for date_chunk in date_chunks:
            if date_chunk["DATE"].eq(target_date).any():
                return True
    return False
