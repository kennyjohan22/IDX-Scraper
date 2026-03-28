"""
screener.py — Pre-Rocket Pattern Daily Screener  (v2)
======================================================
Rebuilt after back-analysis of historical accuracy showed original signals
(range compression + flat price) were HURTING performance — STRONG tier
underperformed RADAR tier, meaning the extra filters discarded winners.

New signals (replacing range compression & flat price):
  Signal 1 — Volume surge       : 10-day avg volume > 1.5x prior 30-day baseline
  Signal 2 — Up-day volume bias : volume on up-days / volume on down-days > 1.5
                                   distinguishes accumulation from distribution
  Signal 3 — Price above MA20   : closing price above 20-day moving average
  Signal 4 — Breakout proximity : close within 8% of 20-day high (coiling near resistance)

Each stock gets a SCORE 0–100 and an ALERT level:
  🔴  STRONG  — vol spike + 2 of (up-bias, above-MA20, near-breakout)
  🟡  WATCH   — vol spike + 1 of the above
  ⚪  RADAR   — vol spike only

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

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_BASE_PRICE     = 100           # IDR — ignore penny stocks
MIN_BASELINE_NILAI = 500_000_000   # avg 500M/day value to ensure liquidity
BASELINE_DAYS      = 30            # days used to compute "normal" metrics
SIGNAL_WINDOW      = 10            # days we measure for signals
VOL_SPIKE_THRESH   = 1.5           # minimum volume ratio (mandatory gate)
UP_BIAS_STRONG     = 1.5           # up-vol / down-vol for accumulation signal
BREAK_PROX_THRESH  = 0.92          # close / 20d-high for near-breakout signal


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
    Compute screener v2 signals for one stock up to position `as_of_pos`.
    Returns a result dict or None if the stock doesn't qualify.

    New signals: up-day volume bias, price vs MA20, breakout proximity.
    Removed: range compression, flat-price filter (showed negative predictive value).
    """
    total_needed = BASELINE_DAYS + SIGNAL_WINDOW + 5
    if as_of_pos < total_needed:
        return None

    baseline = grp.iloc[as_of_pos - BASELINE_DAYS - SIGNAL_WINDOW : as_of_pos - SIGNAL_WINDOW]
    signal   = grp.iloc[as_of_pos - SIGNAL_WINDOW : as_of_pos]
    latest   = grp.iloc[as_of_pos - 1]

    if len(baseline) < 20 or len(signal) < 7:
        return None

    # ── Gates ─────────────────────────────────────────────────────────────────
    if baseline["Nilai"].mean() < MIN_BASELINE_NILAI:
        return None
    if latest["Penutupan"] < MIN_BASE_PRICE:
        return None
    if signal["Volume"].sum() == 0:
        return None

    b_vol = baseline["Volume"].mean()
    s_vol = signal["Volume"].mean()
    vol_ratio = s_vol / b_vol if b_vol > 0 else 0
    if vol_ratio < VOL_SPIKE_THRESH:
        return None

    # Corporate action / suspension filter
    first_close = signal.iloc[0]["Sebelumnya"]
    last_close  = signal.iloc[-1]["Penutupan"]
    price_chg_10d = (last_close - first_close) / first_close * 100 if first_close > 0 else 0
    if price_chg_10d < -30:
        return None

    b_freq = baseline["Frekuensi"].mean()
    s_freq = signal["Frekuensi"].mean()
    freq_ratio = s_freq / b_freq if b_freq > 0 else 0

    net_foreign_signal = signal["Net Foreign"].sum()

    # ── Signal 1: Up-day volume bias (accumulation vs distribution) ───────────
    up_mask  = signal["Penutupan"] >= signal["Sebelumnya"]
    up_vol   = signal.loc[ up_mask, "Volume"].sum()
    down_vol = signal.loc[~up_mask, "Volume"].sum()
    up_bias  = up_vol / down_vol if down_vol > 0 else 5.0

    # ── Signal 2: Price above 20-day MA (trend alignment) ─────────────────────
    ma20_window = grp.iloc[max(0, as_of_pos - 20) : as_of_pos]
    ma20        = ma20_window["Penutupan"].mean()
    above_ma20  = bool(latest["Penutupan"] > ma20)

    # ── Signal 3: Breakout proximity (coiling near resistance) ────────────────
    high_20d   = ma20_window["Tertinggi"].max()
    break_prox = latest["Penutupan"] / high_20d if high_20d > 0 else 0
    near_break = break_prox >= BREAK_PROX_THRESH

    # ── Scoring (0–100) ───────────────────────────────────────────────────────
    score = 0
    score += 35 if vol_ratio >= 2.5 else 25 if vol_ratio >= 2.0 else 15   # vol (15-35)
    score += 30 if up_bias >= 3.0 else 20 if up_bias >= 2.0 else 12 if up_bias >= UP_BIAS_STRONG else 5 if up_bias >= 1.0 else 0
    score += 15 if above_ma20 else 0
    score += 12 if break_prox >= 0.95 else 8 if near_break else 3 if break_prox >= 0.85 else 0
    score += 8 if freq_ratio >= 1.5 else 4 if freq_ratio >= 1.2 else 0
    score = max(0, min(100, score))

    # ── Alert level ───────────────────────────────────────────────────────────
    secondary = sum([up_bias >= UP_BIAS_STRONG, above_ma20, near_break])
    if secondary >= 2:
        alert = "STRONG"
    elif secondary >= 1:
        alert = "WATCH"
    else:
        alert = "RADAR"

    return {
        "Kode Saham":      latest["Kode Saham"],
        "Nama Perusahaan": latest["Nama Perusahaan"],
        "Score":           score,
        "Alert":           alert,
        "Close":           int(latest["Penutupan"]),
        "Vol Ratio":       round(vol_ratio, 2),
        "Up-Vol Bias":     round(up_bias, 2),
        "Above MA20":      above_ma20,
        "Brk Prox%":       round(break_prox * 100, 1),
        "10d Chg%":        round(price_chg_10d, 1),
        "Freq Ratio":      round(freq_ratio, 2),
        "Net Fgn (B)":     round(net_foreign_signal / 1e9, 2),
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
        print(f"  {'Ticker':<8} {'Score':>5} {'Close':>7} {'VolRatio':>8} {'UpBias':>7} {'AbvMA20':>7} {'BrkPrx%':>8} {'10dChg%':>8} {'FreqR':>6} {'NetFgn(B)':>9}  Company")
        print(f"  {'-'*110}")
        for _, r in section_df.head(n).iterrows():
            ma_flag = "YES" if r.get("Above MA20", False) else "no"
            print(
                f"  {r['Kode Saham']:<8} {r['Score']:>5} {r['Close']:>7,} "
                f"{r['Vol Ratio']:>8.2f} {r['Up-Vol Bias']:>7.2f} {ma_flag:>7} "
                f"{r['Brk Prox%']:>7.1f}% {r['10d Chg%']:>7.1f}% "
                f"{r['Freq Ratio']:>6.2f} {r['Net Fgn (B)']:>9.2f}  {r['Nama Perusahaan'][:35]}"
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
