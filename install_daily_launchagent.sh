#!/bin/bash
# Installs the daily IDX scraper schedule for the current macOS user.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABEL="com.moladin.idx-scraper.daily"
SOURCE_PLIST="$SCRIPT_DIR/automation/$LABEL.plist"
TARGET_DIR="$HOME/Library/LaunchAgents"
TARGET_PLIST="$TARGET_DIR/$LABEL.plist"

mkdir -p "$TARGET_DIR"
cp "$SOURCE_PLIST" "$TARGET_PLIST"
chmod 644 "$TARGET_PLIST"
chmod +x "$SCRIPT_DIR/run_daily.sh"

launchctl unload "$TARGET_PLIST" >/dev/null 2>&1 || true
launchctl load "$TARGET_PLIST"

echo "Installed $LABEL"
echo "Schedule: Monday-Friday at 18:15 local time"
echo "Log: $SCRIPT_DIR/scraper.log"
