"""
screener.py — Pre-Rocket Pattern Daily Screener
================================================
Scans all IDX stocks for the two conditions statistically shown to
precede a rocket (50%+ gain in 20 days):

  Signal 1 — Volume accumulation  : 10-day avg volume > 1.5x prior 30-day baseline
  Signal 2 — Range compression    : 10-day avg daily range % < prior 30-day baseline
  Signal 3 — Price NOT already up : 10-day price trend flat or slightly down

Each stock gets a SCORE 0–100 and an ALERT level:
  🔴  STRONG  — all 3 signals active, high confidence
  🟡  WATCH   — 2 signals active
  ⚪  RADAR   — 1 signal (volume spike only)

Run:
    python screener.py               # screen as of latest date in master.csv
    python screener.py --date 2026-03-27
    python screener.py --top 30      # show top N results (default 20)
    python screener.py --min-score 60
"""

import os
import argparse
import warnings
from datetime import datetime, date

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV  = os.path.join(BASE_DIR, "master.csv")
SCREENS_DIR = os.path.join(BASE_DIR, "screens")
os.makedirs(SCREENS_DIR, exist_ok=True)

# ── Thresholds (tuned from back-analysis) ────────────────────────────────────
MIN_BASE_PRICE     = 100       # IDR — ignore penny stocks
MIN_BASELINE_NILAI = 500_000_000  # avg 500M/day value to ensure liquidity
BASELINE_DAYS      = 30        # days used to compute "normal" metrics
SIGNAL_WINDOW      = 10        # days we measure for signals
VOL_SPIKE_STRONG   = 2.0       # volume ratio for strong signal
VOL_SPIKE_MILD     = 1.5       # volume ratio for mild signal
RANGE_COMPRESS_PCT = 0.90      # pre-window range must be < 90% of baseline range


def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV, low_memory=False)
    df["DATE"] = pd.to_datetime(df["DATE"])
    numeric = [
        "Sebelumnya", "Open Price", "Tertinggi", "Terendah", "Penutupan",
        "Selisih", "Volume", "Nilai", "Frekuensi",
        "Foreign Buy", "Foreign Sell",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Net Foreign"] = df["Foreign Buy"] - df["Foreign Sell"]
    return df.sort_values(["Kode Saham", "DATE"]).reset_index(drop=True)


def score_stock(grp: pd.DataFrame, as_of_pos: int):
    """
    Compute screener signals for one stock up to position `as_of_pos`.
    Returns a result dict or None if the stock doesn't qualify.
    """
    total_needed = BASELINE_DAYS + SIGNAL_WINDOW + 5   # buffer
    if as_of_pos < total_needed:
        return None

    # Slice windows
    baseline = grp.iloc[as_of_pos - BASELINE_DAYS - SIGNAL_WINDOW : as_of_pos - SIGNAL_WINDOW]
    signal   = grp.iloc[as_of_pos - SIGNAL_WINDOW : as_of_pos]
    latest   = grp.iloc[as_of_pos - 1]  # most recent day

    if len(baseline) < 20 or len(signal) < 7:
        return None

    # Liquidity gate
    if baseline["Nilai"].mean() < MIN_BASELINE_NILAI:
        return None
    if latest["Penutupan"] < MIN_BASE_PRICE:
        return None
    # Must have traded recently
    if signal["Volume"].sum() == 0:
        return None

    # ── Signal 1: Volume accumulation ────────────────────────────────────────
    b_vol = baseline["Volume"].mean()
    s_vol = signal["Volume"].mean()
    vol_ratio = s_vol / b_vol if b_vol > 0 else 0

    # ── Signal 2: Range compression ──────────────────────────────────────────
    baseline["range_pct"] = (
        (baseline["Tertinggi"] - baseline["Terendah"])
        / baseline["Sebelumnya"].replace(0, np.nan)
        * 100
    )
    signal["range_pct"] = (
        (signal["Tertinggi"] - signal["Terendah"])
        / signal["Sebelumnya"].replace(0, np.nan)
        * 100
    )
    b_range = baseline["range_pct"].mean()
    s_range = signal["range_pct"].mean()
    range_ratio = s_range / b_range if b_range > 0 else 1

    # ── Signal 3: Price trend — flat or slightly down ────────────────────────
    x = np.arange(len(signal))
    slope = np.polyfit(x, signal["Penutupan"].values, 1)[0]
    price_trend_pct_per_day = slope / signal["Penutupan"].mean() * 100

    # 10-day cumulative price change
    first_close = signal.iloc[0]["Sebelumnya"]
    last_close  = signal.iloc[-1]["Penutupan"]
    price_chg_10d = (last_close - first_close) / first_close * 100 if first_close > 0 else 0

    # ── Frequency trend ───────────────────────────────────────────────────────
    b_freq = baseline["Frekuensi"].mean()
    s_freq = signal["Frekuensi"].mean()
    freq_ratio = s_freq / b_freq if b_freq > 0 else 0

    # ── Foreign flow ─────────────────────────────────────────────────────────
    net_foreign_signal = signal["Net Foreign"].sum()
    net_foreign_days   = (signal["Net Foreign"] > 0).sum()

    # ── Scoring (0–100) ──────────────────────────────────────────────────────
    score = 0

    # Volume component (0–40 pts)
    if vol_ratio >= VOL_SPIKE_STRONG:
        score += 40
    elif vol_ratio >= VOL_SPIKE_MILD:
        score += 25
    elif vol_ratio >= 1.2:
        score += 10

    # Range compression (0–35 pts)
    if range_ratio < 0.70:
        score += 35
    elif range_ratio < RANGE_COMPRESS_PCT:
        score += 20
    elif range_ratio < 1.0:
        score += 8

    # Discard stocks with extreme price moves (corporate actions / suspensions)
    if price_chg_10d < -30 or range_ratio < 0:
        return None

    # Price not already running (0–15 pts)
    # Best: flat to slightly down (-5% to +5%). Penalise if already up big.
    if -5 <= price_chg_10d <= 5:
        score += 15
    elif -10 <= price_chg_10d <= 10:
        score += 8
    elif price_chg_10d > 20:
        score -= 10   # Already running — lower priority

    # Frequency uptick bonus (0–10 pts)
    if freq_ratio >= 1.5:
        score += 10
    elif freq_ratio >= 1.2:
        score += 5

    score = max(0, min(100, score))

    # ── Alert level ───────────────────────────────────────────────────────────
    has_vol    = vol_ratio >= VOL_SPIKE_MILD       # mandatory
    has_range  = range_ratio < RANGE_COMPRESS_PCT
    has_flat   = -10 <= price_chg_10d <= 10

    # Volume is mandatory — no volume spike = not interesting
    if not has_vol:
        return None

    secondary = sum([has_range, has_flat])
    if secondary == 2:
        alert = "STRONG"
    elif secondary == 1:
        alert = "WATCH"
    else:
        alert = "RADAR"

    return {
        "Kode Saham":     latest["Kode Saham"],
        "Nama Perusahaan": latest["Nama Perusahaan"],
        "Score":          score,
        "Alert":          alert,
        "Close":          int(latest["Penutupan"]),
        "Vol Ratio":      round(vol_ratio, 2),
        "Range Ratio":    round(range_ratio, 2),
        "10d Chg%":       round(price_chg_10d, 1),
        "Freq Ratio":     round(freq_ratio, 2),
        "Net Fgn (B)":    round(net_foreign_signal / 1e9, 2),
        "Avg Val/day (B)": round(baseline["Nilai"].mean() / 1e9, 2),
    }


def run_screener(as_of_date: str = None, top_n: int = 20, min_score: int = 0) -> pd.DataFrame:
    df = load_master()

    if as_of_date:
        cutoff = pd.to_datetime(as_of_date)
        df = df[df["DATE"] <= cutoff]
        if df.empty:
            print(f"No data on or before {as_of_date}")
            return pd.DataFrame()

    screen_date = df["DATE"].max().strftime("%Y-%m-%d")
    print(f"\nRunning screener as of {screen_date} ...")

    results = []
    for ticker, grp in df.groupby("Kode Saham"):
        grp = grp[grp["Penutupan"] > 0].reset_index(drop=True)
        result = score_stock(grp, len(grp))
        if result:
            results.append(result)

    if not results:
        print("No candidates found.")
        return pd.DataFrame()

    out = (
        pd.DataFrame(results)
        .sort_values(["Alert", "Score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    out = out[out["Score"] >= min_score]

    # ── Print ─────────────────────────────────────────────────────────────────
    alert_order = {"STRONG": 0, "WATCH": 1, "RADAR": 2}
    out["_order"] = out["Alert"].map(alert_order)
    out = out.sort_values(["_order", "Score"], ascending=[True, False]).drop(columns="_order")

    strong = out[out["Alert"] == "STRONG"]
    watch  = out[out["Alert"] == "WATCH"]
    radar  = out[out["Alert"] == "RADAR"]

    print(f"\n{'='*80}")
    print(f"  IDX PRE-ROCKET SCREENER  —  {screen_date}")
    print(f"  Signals: Vol >1.5x baseline | Range compression | Price flat/down")
    print(f"{'='*80}")

    def print_section(label, emoji, section_df, n):
        if section_df.empty:
            return
        print(f"\n  {emoji} {label} ({len(section_df)} stocks)  — showing top {min(n, len(section_df))}")
        print(f"  {'Ticker':<8} {'Score':>5} {'Close':>7} {'Vol Ratio':>9} {'RangeRatio':>10} {'10d Chg%':>9} {'FreqRatio':>9} {'NetFgn(B)':>10}  Company")
        print(f"  {'-'*105}")
        for _, r in section_df.head(n).iterrows():
            print(
                f"  {r['Kode Saham']:<8} {r['Score']:>5} {r['Close']:>7,} "
                f"{r['Vol Ratio']:>9.2f} {r['Range Ratio']:>10.2f} "
                f"{r['10d Chg%']:>8.1f}% {r['Freq Ratio']:>9.2f} "
                f"{r['Net Fgn (B)']:>10.2f}  {r['Nama Perusahaan'][:35]}"
            )

    show_n = max(top_n // 3, 5)
    print_section("STRONG — All 3 signals active", "🔴", strong, show_n * 2)
    print_section("WATCH  — 2 signals active",     "🟡", watch,  show_n)
    print_section("RADAR  — Volume spike only",    "⚪", radar,  show_n)

    print(f"\n  Total candidates: {len(out)}  (STRONG: {len(strong)}, WATCH: {len(watch)}, RADAR: {len(radar)})")
    print()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_path = os.path.join(SCREENS_DIR, f"screen_{screen_date}.csv")
    out.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDX Pre-Rocket Screener")
    parser.add_argument("--date",      type=str, default=None,   help="Screen as of date (YYYY-MM-DD)")
    parser.add_argument("--top",       type=int, default=20,     help="Total rows to show (default 20)")
    parser.add_argument("--min-score", type=int, default=0,      help="Minimum score filter (0-100)")
    args = parser.parse_args()

    run_screener(
        as_of_date=args.date,
        top_n=args.top,
        min_score=args.min_score,
    )
