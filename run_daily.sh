#!/bin/bash
# run_daily.sh — called by cron every weekday morning
# Fetches today's IDX data and appends to master.csv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$(which python3)"
LOG="$SCRIPT_DIR/scraper.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_daily.sh triggered" >> "$LOG"
cd "$SCRIPT_DIR"
"$PYTHON" scraper_browser.py >> "$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_daily.sh done" >> "$LOG"
