#!/bin/bash
# run_daily.sh - called by launchd/cron on trading days
# Fetches any missing IDX data through today and refreshes screen outputs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
PYTHON="$(command -v python3)"
LOG="$SCRIPT_DIR/scraper.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_daily.sh triggered" >> "$LOG"
cd "$SCRIPT_DIR"

if [ -z "$PYTHON" ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: python3 not found" >> "$LOG"
  exit 1
fi

"$PYTHON" daily_update.py >> "$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_daily.sh done" >> "$LOG"
