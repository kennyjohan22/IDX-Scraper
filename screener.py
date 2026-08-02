"""
screener.py — Pre-Rocket Pattern Daily Screener  (v3)
======================================================
v3 is a data-driven rebuild after analyzing 3,594 historical picks.
Development-sample results vs v2 (same data used to select the rules; not an
out-of-sample forecast and excluding execution costs):
  Win rate:      47.3%  (vs 44.0%)   +3.3 pp
  Mean +30d:     +7.16% (vs +5.67%)  +1.50 pp
  Median +30d:   -0.70% (vs -1.89%)  +1.19 pp
  STRONG tier:   49.8% win  (vs 46.0%)
  Top 50 picks:  +7.51% mean (vs +3.52%)  +4.0 pp win rate

Counterintuitive findings that drove v3:
  • Volume ratio is NEGATIVELY correlated with returns — huge spikes
    are typically pump-and-dumps. Sweet spot is 1.5x – 2.5x.
  • Position in 52w range matters more than absolute trend.
  • Foreign-buying CONSISTENCY (% of days) outperforms total amount.
  • Already-extended stocks (pre30_chg >= 30%) underperform — chase
    risk dominates momentum benefit.
  • A single-day volume spike (vol_spike_max >= 3.5) hurts performance.
  • MA50>MA200 (bull regime) is surprisingly NOT predictive.

v3 mandatory gates:
  vol_ratio        in [1.5, 5.0]   not too small, not pumpy
  vol_spike_max    < 3.5           no one-day pump-and-dumps
  pos_in_range     >= 50           upper half of 52w range
  above_ma20       = 1             short-term uptrend
  above_ma50       = 1             medium-term uptrend
  pre30_chg        < 30            not already extended

v3 scoring (0–100):
  break_prox       25 pts   strongest predictor
  pos_in_range     20 pts
  fgn_positive_days 15 pts  consistency over total
  vol_ratio sweet  10 pts   1.7-2.5 ideal
  vol_cv (consistency) 10 pts
  obv_pos          10 pts
  up_bias          10 pts

Tier (STRONG/WATCH/RADAR) based on count of these high-conviction bits:
  break_prox >= 0.95, pos_in_range >= 75, fgn_pos_days >= 60, obv_pos = 1

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

import pandas as pd

from strategy_features import obv_slope_is_positive, position_in_trading_range

warnings.filterwarnings("ignore")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV  = os.path.join(BASE_DIR, "master.csv")
SCREENS_DIR = os.path.join(BASE_DIR, "screens")
os.makedirs(SCREENS_DIR, exist_ok=True)

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_BASE_PRICE     = 100           # IDR — ignore penny stocks
MIN_BASELINE_NILAI = 500_000_000   # avg 500M/day for liquidity
BASELINE_DAYS      = 30
SIGNAL_WINDOW      = 10

# v3 gates
VOL_RATIO_MIN      = 1.5
VOL_RATIO_MAX      = 5.0
VOL_SPIKE_MAX      = 3.5
POS_IN_RANGE_MIN   = 50
PRE30_CHG_MAX      = 30


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
    """v3 screener — returns dict or None if doesn't qualify."""
    total_needed = BASELINE_DAYS + SIGNAL_WINDOW + 5
    if as_of_pos < total_needed:
        return None

    baseline = grp.iloc[as_of_pos - BASELINE_DAYS - SIGNAL_WINDOW : as_of_pos - SIGNAL_WINDOW]
    signal   = grp.iloc[as_of_pos - SIGNAL_WINDOW : as_of_pos]
    latest   = grp.iloc[as_of_pos - 1]

    if len(baseline) < 20 or len(signal) < 7:
        return None

    # ── Liquidity & price gates ───────────────────────────────────────────────
    if baseline["Nilai"].mean() < MIN_BASELINE_NILAI:
        return None
    if latest["Penutupan"] < MIN_BASE_PRICE:
        return None
    if signal["Volume"].sum() == 0:
        return None

    b_vol = baseline["Volume"].mean()
    s_vol = signal["Volume"].mean()
    vol_ratio = s_vol / b_vol if b_vol > 0 else 0
    if vol_ratio < VOL_RATIO_MIN or vol_ratio > VOL_RATIO_MAX:
        return None

    # Single-day volume spike concentration
    vol_spike_max = signal["Volume"].max() / s_vol if s_vol > 0 else 0
    if vol_spike_max >= VOL_SPIKE_MAX:
        return None

    # Corporate action / suspension filter
    first_close = signal.iloc[0]["Sebelumnya"]
    last_close  = signal.iloc[-1]["Penutupan"]
    price_chg_10d = (last_close - first_close) / first_close * 100 if first_close > 0 else 0
    if price_chg_10d < -30:
        return None

    # ── Pre-30d trend (must NOT already be extended) ──────────────────────────
    pos_idx = as_of_pos
    pre30_start = grp.iloc[max(0, pos_idx-40)]["Penutupan"] if pos_idx >= 40 else grp.iloc[0]["Penutupan"]
    pre30_end   = grp.iloc[pos_idx-11]["Penutupan"] if pos_idx >= 11 else grp.iloc[-1]["Penutupan"]
    pre30_chg   = (pre30_end - pre30_start) / pre30_start * 100 if pre30_start > 0 else 0
    if pre30_chg >= PRE30_CHG_MAX:
        return None

    # ── Position in 52-week range ─────────────────────────────────────────────
    pos_in_range = position_in_trading_range(grp, as_of_pos)
    if pos_in_range < POS_IN_RANGE_MIN:
        return None

    # ── Trend gates: above MA20 AND MA50 ──────────────────────────────────────
    ma20_window = grp.iloc[max(0, as_of_pos - 20) : as_of_pos]
    ma20 = ma20_window["Penutupan"].mean()
    above_ma20 = bool(latest["Penutupan"] > ma20)
    if not above_ma20:
        return None

    ma50 = grp.iloc[max(0, as_of_pos - 50) : as_of_pos]["Penutupan"].mean()
    above_ma50 = bool(latest["Penutupan"] > ma50)
    if not above_ma50:
        return None

    # ── Compute remaining features for scoring ────────────────────────────────
    high_20d   = ma20_window["Tertinggi"].max()
    break_prox = latest["Penutupan"] / high_20d if high_20d > 0 else 0

    up_mask  = signal["Penutupan"] >= signal["Sebelumnya"]
    up_vol   = signal.loc[ up_mask, "Volume"].sum()
    down_vol = signal.loc[~up_mask, "Volume"].sum()
    up_bias  = up_vol / down_vol if down_vol > 0 else 5.0

    obv_pos = obv_slope_is_positive(signal)

    # Volume consistency (low CV = sustained, not single spike)
    vol_cv = signal["Volume"].std() / signal["Volume"].mean() if signal["Volume"].mean() > 0 else 0

    # Foreign-flow consistency (% of days net foreign positive)
    fgn_positive_days = (signal["Net Foreign"] > 0).sum() / len(signal) * 100
    net_foreign_signal = signal["Net Foreign"].sum()

    b_freq = baseline["Frekuensi"].mean()
    s_freq = signal["Frekuensi"].mean()
    freq_ratio = s_freq / b_freq if b_freq > 0 else 0

    # ── v3 Scoring (0-100) ────────────────────────────────────────────────────
    score = 0
    # break_prox: 25 pts (strongest predictor)
    score += (25 if break_prox >= 0.97 else
              20 if break_prox >= 0.95 else
              15 if break_prox >= 0.92 else
              8  if break_prox >= 0.88 else 0)
    # pos_in_range: 20 pts
    score += (20 if pos_in_range >= 85 else
              15 if pos_in_range >= 75 else
              10 if pos_in_range >= 65 else 5)
    # fgn_positive_days: 15 pts
    score += (15 if fgn_positive_days >= 70 else
              10 if fgn_positive_days >= 60 else
              5  if fgn_positive_days >= 50 else 0)
    # obv_pos: 10 pts
    score += 10 if obv_pos else 0
    # up_bias capped: 10 pts
    ub = min(up_bias, 5)
    score += (10 if ub >= 2.5 else
              7  if ub >= 1.8 else
              4  if ub >= 1.3 else 0)
    # vol_ratio sweet spot: 10 pts
    score += (10 if 1.7 <= vol_ratio <= 2.5 else
              7  if 1.5 <= vol_ratio <= 3.0 else 3)
    # vol consistency: 10 pts
    score += (10 if vol_cv < 0.5 else
              7  if vol_cv < 0.7 else
              3  if vol_cv < 1.0 else 0)
    score = max(0, min(100, score))

    # ── Tier ──────────────────────────────────────────────────────────────────
    strong_bits = sum([
        break_prox >= 0.95,
        pos_in_range >= 75,
        fgn_positive_days >= 60,
        bool(obv_pos),
    ])
    if strong_bits >= 3:
        alert = "STRONG"
    elif strong_bits >= 2:
        alert = "WATCH"
    else:
        alert = "RADAR"

    # ── "Why" reasons ─────────────────────────────────────────────────────────
    reasons = []
    if break_prox >= 0.95: reasons.append(f"breakout-ready ({break_prox*100:.0f}% of 20d high)")
    elif break_prox >= 0.92: reasons.append(f"near breakout ({break_prox*100:.0f}%)")
    if pos_in_range >= 75: reasons.append(f"top of 52w range ({pos_in_range:.0f}%)")
    if fgn_positive_days >= 60: reasons.append(f"foreign buying {fgn_positive_days:.0f}% of days")
    if 1.7 <= vol_ratio <= 2.5: reasons.append(f"healthy volume ({vol_ratio:.1f}x)")
    if obv_pos: reasons.append("OBV uptrend")
    if up_bias >= 2.0: reasons.append(f"strong accumulation ({up_bias:.1f}x up/down)")
    if vol_cv < 0.5: reasons.append("sustained volume (no single spike)")
    if not reasons: reasons.append("baseline pattern")
    why = " · ".join(reasons[:4])  # cap at 4 most-specific reasons

    return {
        "Kode Saham":      latest["Kode Saham"],
        "Nama Perusahaan": latest["Nama Perusahaan"],
        "Score":           score,
        "Alert":           alert,
        "Close":           int(latest["Penutupan"]),
        "Vol Ratio":       round(vol_ratio, 2),
        "Up-Vol Bias":     round(up_bias, 2),
        "Brk Prox%":       round(break_prox * 100, 1),
        "PosRange%":       round(pos_in_range, 0),
        "Fgn+Days%":       round(fgn_positive_days, 0),
        "VolCV":           round(vol_cv, 2),
        "OBV+":            "YES" if obv_pos else "no",
        "10d Chg%":        round(price_chg_10d, 1),
        "Pre30 Chg%":      round(pre30_chg, 1),
        "Net Fgn (M sh)":  round(net_foreign_signal / 1e6, 2),
        "Avg Val/day (B)": round(baseline["Nilai"].mean() / 1e9, 2),
        "Why":             why,
    }


def run_screener(as_of_date: str = None, top_n: int = 20, min_score: int = 0) -> pd.DataFrame:
    df = load_master()

    if as_of_date:
        cutoff = pd.to_datetime(as_of_date)
        df = df[df["DATE"] <= cutoff]
        if df.empty:
            print(f"No data on or before {as_of_date}")
            return pd.DataFrame()

    screen_timestamp = pd.Timestamp(df["DATE"].max())
    screen_date = screen_timestamp.strftime("%Y-%m-%d")
    print(f"\nRunning v3 screener as of {screen_date} ...")

    results = []
    for ticker, grp in df.groupby("Kode Saham"):
        grp = grp[grp["Penutupan"] > 0].reset_index(drop=True)
        if grp.empty:
            continue
        latest = grp.iloc[-1]
        if pd.Timestamp(latest["DATE"]) != screen_timestamp or latest["Volume"] <= 0:
            continue
        result = score_stock(grp, len(grp))
        if result:
            results.append(result)

    if not results:
        print("No candidates found.")
        return pd.DataFrame()

    out = pd.DataFrame(results)
    out = out[out["Score"] >= min_score]

    alert_order = {"STRONG": 0, "WATCH": 1, "RADAR": 2}
    out["_order"] = out["Alert"].map(alert_order)
    out = out.sort_values(["_order", "Score"], ascending=[True, False]).drop(columns="_order")

    strong = out[out["Alert"] == "STRONG"]
    watch  = out[out["Alert"] == "WATCH"]
    radar  = out[out["Alert"] == "RADAR"]

    print(f"\n{'='*100}")
    print(f"  IDX PRE-ROCKET SCREENER (v3)  —  {screen_date}")
    print(f"  Gates: 1.5≤VolR≤5.0 | VolSpikeMax<3.5 | PosRange≥50 | >MA20 | >MA50 | Pre30Chg<30")
    print(f"{'='*100}")

    def print_section(label, emoji, section_df, n):
        if section_df.empty:
            return
        print(f"\n  {emoji} {label} ({len(section_df)} stocks) — showing top {min(n, len(section_df))}")
        print(f"  {'Ticker':<8} {'Sc':>3} {'Close':>7} {'VolR':>5} {'UpBias':>6} {'BrkPx%':>7} "
              f"{'PosR%':>5} {'F+%':>4} {'OBV':>4} {'Why'}")
        print(f"  {'-'*100}")
        for _, r in section_df.head(n).iterrows():
            print(
                f"  {r['Kode Saham']:<8} {r['Score']:>3} {r['Close']:>7,} "
                f"{r['Vol Ratio']:>5.2f} {r['Up-Vol Bias']:>6.2f} "
                f"{r['Brk Prox%']:>6.1f}% {r['PosRange%']:>4.0f}% "
                f"{r['Fgn+Days%']:>3.0f}% {r['OBV+']:>4}  {r['Why']}"
            )

    show_n = max(top_n // 3, 5)
    print_section("STRONG — 3+ high-conviction signals", "🔴", strong, show_n * 2)
    print_section("WATCH  — 2 high-conviction signals",  "🟡", watch,  show_n)
    print_section("RADAR  — passed all gates",            "⚪", radar,  show_n)

    print(f"\n  Total: {len(out)}  (STRONG: {len(strong)}, WATCH: {len(watch)}, RADAR: {len(radar)})")
    print()

    out_path = os.path.join(SCREENS_DIR, f"screen_{screen_date}.csv")
    out.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDX Pre-Rocket Screener v3")
    parser.add_argument("--date",      type=str, default=None,   help="Screen as of date (YYYY-MM-DD)")
    parser.add_argument("--top",       type=int, default=20,     help="Total rows to show (default 20)")
    parser.add_argument("--min-score", type=int, default=0,      help="Minimum score filter (0-100)")
    args = parser.parse_args()

    run_screener(
        as_of_date=args.date,
        top_n=args.top,
        min_score=args.min_score,
    )
