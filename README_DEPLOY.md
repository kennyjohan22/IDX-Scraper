# IDX Stock Recommendation Bot Deployment

This project is a Streamlit dashboard backed by `master.csv`.

## Daily local refresh

Install the macOS daily schedule:

```bash
./install_daily_launchagent.sh
```

It runs Monday-Friday at 18:15 local time, fetches any missing IDX weekdays
through today, appends them to `master.csv`, and refreshes the screener output.
Logs are written to `scraper.log`.

## Streamlit website

The fastest hosted path for the current app is Streamlit Community Cloud:

1. Push this repo to GitHub.
2. Create a new Streamlit app from the repo.
3. Set the main file to `dashboard.py`.
4. Keep `requirements.txt` and `packages.txt` in the repo root.

The dashboard prefers local `master.csv` when present. On the hosted app it can
read `data/master.csv.gz`, a compressed snapshot that is safe to keep in git.

## Hosted daily refresh

`.github/workflows/daily-idx-refresh.yml` runs Monday-Friday at 18:30 WIB. It
restores `master.csv` from `data/master.csv.gz` when available, fetches missing
IDX data, updates the compressed snapshot, refreshes screener CSVs, and commits
the new snapshot back to GitHub. On a fresh hosted repo with no snapshot yet, it
backfills from `2026-03-01` on its first run.
