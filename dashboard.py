"""
IDX Stock Market Dashboard
===========================
Run:  streamlit run dashboard.py

Pages:
  1. Market Overview  — breadth, value, foreign flow for any date
  2. Screener         — pre-rocket STRONG / WATCH / RADAR picks
  3. Stock Deep Dive  — full price + volume + foreign flow chart for any ticker
  4. Portfolio        — track your positions with P&L
"""

import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from strategy_features import obv_slope_is_positive, position_in_trading_range
from weekly_strategy import (
    assess_weekly_regime,
    build_weekly_trade_plan,
    estimate_foreign_value,
    rank_weekly_candidates,
    select_actionable_candidates,
)

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV  = os.path.join(BASE_DIR, "master.csv")
MASTER_SNAPSHOT_CSV_GZ = os.path.join(BASE_DIR, "data", "master.csv.gz")
SCREENS_DIR = os.path.join(BASE_DIR, "screens")

st.set_page_config(
    page_title="IDX Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loading (cached) ─────────────────────────────────────────────────────
def master_data_path():
    """Prefer the live local master.csv, fallback to the deployable snapshot."""
    if os.path.exists(MASTER_CSV):
        return MASTER_CSV
    if os.path.exists(MASTER_SNAPSHOT_CSV_GZ):
        return MASTER_SNAPSHOT_CSV_GZ
    return MASTER_CSV


@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(master_data_path(), low_memory=False)
    df["DATE"] = pd.to_datetime(df["DATE"])
    num_cols = [
        "Sebelumnya", "Open Price", "Tertinggi", "Terendah", "Penutupan",
        "Selisih", "Volume", "Nilai", "Frekuensi",
        "Foreign Buy", "Foreign Sell",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Net Foreign"] = df["Foreign Buy"] - df["Foreign Sell"]
    df["Net Foreign Est Value"] = estimate_foreign_value(
        df["Net Foreign"], df["Penutupan"]
    )
    df["pct_chg"] = (df["Selisih"] / df["Sebelumnya"].replace(0, np.nan) * 100).round(2)
    return df.sort_values(["Kode Saham", "DATE"]).reset_index(drop=True)

@st.cache_data(ttl=300)
def get_trading_days():
    df = load_data()
    return sorted(df["DATE"].unique(), reverse=True)


@st.cache_data(ttl=3600)
def get_weekly_rankings(_df, as_of_date, data_version):
    """Cache rankings while invalidating when master.csv changes."""
    return rank_weekly_candidates(_df, as_of_date=as_of_date)

# ── Screener logic v3 (data-driven) ───────────────────────────────────────────
# v3 development sample: 47.3% win, +7.16% mean. These figures were measured
# on the same history used to select the rules and are not out-of-sample proof.
# Top 50 picks: 48% win, +7.51% mean (vs v2 44%, +3.52%)
#
# Counterintuitive findings driving v3 design:
#   • vol_ratio negatively correlates with returns — huge spikes = pump-and-dumps
#   • Position in 52w range matters more than absolute trend
#   • Foreign-buying CONSISTENCY beats total amount
#   • Already-extended stocks (pre30_chg ≥ 30%) underperform
#   • MA50>MA200 (bull regime) is NOT predictive
#   • break_prox is the single strongest predictor
def run_screener(df, as_of_date):
    BASELINE = 30; SIGNAL = 10
    MIN_PRICE = 100; MIN_NILAI = 500_000_000
    VOL_RATIO_MIN = 1.5; VOL_RATIO_MAX = 5.0
    VOL_SPIKE_MAX = 3.5
    POS_IN_RANGE_MIN = 50
    PRE30_CHG_MAX = 30

    cutoff = pd.to_datetime(as_of_date)
    d = df[df["DATE"] <= cutoff]
    results = []
    for ticker, grp in d.groupby("Kode Saham"):
        g = grp[grp["Penutupan"] > 0].reset_index(drop=True)
        if len(g) < BASELINE + SIGNAL + 5: continue
        base = g.iloc[-(BASELINE+SIGNAL):-SIGNAL].copy()
        sig  = g.iloc[-SIGNAL:].copy()
        lat  = g.iloc[-1]

        # ── Gates ─────────────────────────────────────────────────────────────
        if pd.Timestamp(lat["DATE"]) != cutoff or lat["Volume"] <= 0: continue
        if base["Nilai"].mean() < MIN_NILAI: continue
        if lat["Penutupan"] < MIN_PRICE: continue
        if sig["Volume"].sum() == 0: continue

        b_vol  = base["Volume"].mean(); s_vol  = sig["Volume"].mean()
        b_val  = base["Nilai"].mean()
        vr    = s_vol / b_vol if b_vol > 0 else 0
        if vr < VOL_RATIO_MIN or vr > VOL_RATIO_MAX: continue

        # Single-day spike concentration (no pump-and-dump)
        vol_spike_max = sig["Volume"].max() / s_vol if s_vol > 0 else 0
        if vol_spike_max >= VOL_SPIKE_MAX: continue

        # Corporate action filter
        f  = sig.iloc[0]["Sebelumnya"]; l = sig.iloc[-1]["Penutupan"]
        pc = (l - f) / f * 100 if f > 0 else 0
        if pc < -30: continue

        # Pre-30d trend (must NOT already be extended)
        pos_idx = len(g)
        pre30_start = g.iloc[max(0, pos_idx-40)]["Penutupan"] if pos_idx >= 40 else g.iloc[0]["Penutupan"]
        pre30_end   = g.iloc[pos_idx-11]["Penutupan"] if pos_idx >= 11 else g.iloc[-1]["Penutupan"]
        pre30_chg   = (pre30_end - pre30_start) / pre30_start * 100 if pre30_start > 0 else 0
        if pre30_chg >= PRE30_CHG_MAX: continue

        # Position in the latest 252 valid trading observations (~52 weeks)
        pos_in_range = position_in_trading_range(g, len(g))
        if pos_in_range < POS_IN_RANGE_MIN: continue

        # Trend gates
        ma20       = g.tail(20)["Penutupan"].mean()
        above_ma20 = bool(lat["Penutupan"] > ma20)
        if not above_ma20: continue

        ma50       = g.tail(50)["Penutupan"].mean()
        above_ma50 = bool(lat["Penutupan"] > ma50)
        if not above_ma50: continue

        # ── Compute features for scoring ──────────────────────────────────────
        high_20d   = g.tail(20)["Tertinggi"].max()
        break_prox = lat["Penutupan"] / high_20d if high_20d > 0 else 0

        up_mask  = sig["Penutupan"] >= sig["Sebelumnya"]
        up_vol   = sig.loc[ up_mask, "Volume"].sum()
        down_vol = sig.loc[~up_mask, "Volume"].sum()
        up_bias  = up_vol / down_vol if down_vol > 0 else 5.0

        obv_pos = obv_slope_is_positive(sig)

        vol_cv = sig["Volume"].std() / sig["Volume"].mean() if sig["Volume"].mean() > 0 else 0
        fgn_positive_days = (sig["Net Foreign"] > 0).sum() / len(sig) * 100
        nf = sig["Net Foreign"].sum()

        # ── v3 Scoring (0–100) ────────────────────────────────────────────────
        sc = 0
        sc += 25 if break_prox >= 0.97 else 20 if break_prox >= 0.95 else 15 if break_prox >= 0.92 else 8 if break_prox >= 0.88 else 0
        sc += 20 if pos_in_range >= 85 else 15 if pos_in_range >= 75 else 10 if pos_in_range >= 65 else 5
        sc += 15 if fgn_positive_days >= 70 else 10 if fgn_positive_days >= 60 else 5 if fgn_positive_days >= 50 else 0
        sc += 10 if obv_pos else 0
        ub = min(up_bias, 5)
        sc += 10 if ub >= 2.5 else 7 if ub >= 1.8 else 4 if ub >= 1.3 else 0
        sc += 10 if 1.7 <= vr <= 2.5 else 7 if 1.5 <= vr <= 3.0 else 3
        sc += 10 if vol_cv < 0.5 else 7 if vol_cv < 0.7 else 3 if vol_cv < 1.0 else 0
        sc = max(0, min(100, sc))

        # ── Tier (high-conviction bits) ───────────────────────────────────────
        strong_bits = sum([
            break_prox >= 0.95,
            pos_in_range >= 75,
            fgn_positive_days >= 60,
            bool(obv_pos),
        ])
        alert = "🔴 STRONG" if strong_bits >= 3 else "🟡 WATCH" if strong_bits >= 2 else "⚪ RADAR"

        # ── Why ───────────────────────────────────────────────────────────────
        reasons = []
        if break_prox >= 0.95:    reasons.append(f"Breakout-ready ({break_prox*100:.0f}% of 20d high)")
        elif break_prox >= 0.92:  reasons.append(f"Near breakout ({break_prox*100:.0f}%)")
        if pos_in_range >= 75:    reasons.append(f"Top of 52w range ({pos_in_range:.0f}%)")
        if fgn_positive_days >= 60: reasons.append(f"Foreign buying {fgn_positive_days:.0f}% of days")
        if 1.7 <= vr <= 2.5:      reasons.append(f"Healthy volume ({vr:.1f}x)")
        if obv_pos:               reasons.append("OBV uptrend")
        if up_bias >= 2.0:        reasons.append(f"Strong accumulation ({up_bias:.1f}x up/down)")
        if vol_cv < 0.5:          reasons.append("Sustained volume (no single spike)")
        if pre30_chg < 5:         reasons.append("Fresh setup (not yet run)")
        if not reasons:           reasons.append("All gates passed")

        results.append({
            "Ticker": ticker, "Company": str(lat["Nama Perusahaan"])[:35],
            "Alert": alert, "Score": sc,
            "Why": " | ".join(reasons[:4]),
            "Close": int(lat["Penutupan"]),
            "Vol Ratio":   round(vr, 2),
            "Up-Vol Bias": round(up_bias, 2),
            "Brk Prox%":   round(break_prox * 100, 1),
            "PosRange%":   round(pos_in_range, 0),
            "Fgn+Days%":   round(fgn_positive_days, 0),
            "VolCV":       round(vol_cv, 2),
            "OBV+":        "YES" if obv_pos else "no",
            "10d Chg%":    round(pc, 1),
            "Pre30 Chg%":  round(pre30_chg, 1),
            "Net Fgn (M sh)": round(nf / 1e6, 2),
            "Avg Val/day (B)": round(b_val / 1e9, 2),
        })
    if not results:
        return pd.DataFrame()
    out = pd.DataFrame(results)
    order = {"🔴 STRONG": 0, "🟡 WATCH": 1, "⚪ RADAR": 2}
    out["_ord"] = out["Alert"].map(order)
    return out.sort_values(["_ord", "Score"], ascending=[True, False]).drop(columns="_ord").reset_index(drop=True)


def compute_forward_returns(df, tickers, screen_date):
    """Return a DataFrame with +1d, +7d, +30d and 'now' returns for each ticker."""
    all_dates = sorted(df["DATE"].unique())
    cutoff = pd.to_datetime(screen_date)
    # Find index of screen_date (or last date <= cutoff)
    idx = max((i for i, d in enumerate(all_dates) if d <= cutoff), default=None)
    if idx is None:
        return pd.DataFrame()

    price_pivot = df.pivot_table(index="DATE", columns="Kode Saham", values="Penutupan")
    latest_date = all_dates[-1]

    rows = []
    for ticker in tickers:
        if ticker not in price_pivot.columns:
            continue
        entry = price_pivot[ticker].iloc[idx] if idx < len(price_pivot) else None
        if not entry or pd.isna(entry) or entry <= 0:
            continue

        def fwd_ret(offset):
            fi = idx + offset
            if fi >= len(all_dates):
                return None
            p = price_pivot[ticker].get(all_dates[fi])
            if p is None or pd.isna(p) or p <= 0:
                return None
            return round((p - entry) / entry * 100, 1)

        now_p = price_pivot[ticker].get(latest_date)
        now_ret = round((now_p - entry) / entry * 100, 1) if (now_p and not pd.isna(now_p) and entry > 0) else None

        rows.append({
            "Ticker": ticker,
            "Entry Close": int(entry),
            "+1d %":  fwd_ret(1),
            "+7d %":  fwd_ret(7),
            "+30d %": fwd_ret(30),
            "Now %":  now_ret if latest_date != all_dates[idx] else None,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@st.cache_data(ttl=3600)
def get_screener_accuracy_trend(_df, horizon=30):
    """
    Sample every 5 trading days, run the screener, record forward return per pick.
    Returns a long-form DataFrame: date | Alert | ret_<horizon>d
    Cached for 1 hour, keyed on horizon — switching horizon recomputes.
    """
    all_dates = sorted(_df["DATE"].unique())
    min_idx  = 45
    max_idx  = len(all_dates) - horizon - 1
    if max_idx <= min_idx:
        return pd.DataFrame()

    price_pivot = _df.pivot_table(index="DATE", columns="Kode Saham", values="Penutupan")

    rows = []
    for i in range(min_idx, max_idx, 5):
        screen_date = all_dates[i]
        res = run_screener(_df, screen_date)
        if res.empty:
            continue
        for _, rec in res.iterrows():
            t = rec["Ticker"]
            if t not in price_pivot.columns:
                continue
            entry = price_pivot[t].get(all_dates[i])
            if not entry or pd.isna(entry) or entry <= 0:
                continue
            fp = price_pivot[t].get(all_dates[i + horizon])
            if fp is None or pd.isna(fp) or fp <= 0:
                continue
            rows.append({
                "date":    pd.Timestamp(screen_date),
                "Alert":   rec["Alert"],
                "ret_fwd": round((fp - entry) / entry * 100, 2),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@st.cache_data(ttl=3600)
def backtest_tp_sl(_df, tp_pct, sl_pct, max_hold_days, tiers):
    """
    Simulate: for every historical pick in `tiers`, buy at close, then walk
    forward up to max_hold_days using daily High/Low to see whether TP or SL
    was hit first. If neither, exit at last close.

    Conservative rule: if a single day's bar contains both TP and SL levels,
    assume SL hit first (worst case, treats gaps as bad).

    Returns: DataFrame of trades — date, ticker, alert, entry, exit, return,
             outcome (TP/SL/TIME), hold_days
    """
    all_dates = sorted(_df["DATE"].unique())
    min_idx = 45
    max_idx = len(all_dates) - 2  # need at least 1 forward day
    if max_idx <= min_idx:
        return pd.DataFrame()

    high_pivot  = _df.pivot_table(index="DATE", columns="Kode Saham", values="Tertinggi")
    low_pivot   = _df.pivot_table(index="DATE", columns="Kode Saham", values="Terendah")
    close_pivot = _df.pivot_table(index="DATE", columns="Kode Saham", values="Penutupan")

    tp_enabled = tp_pct is not None and tp_pct > 0
    sl_enabled = sl_pct is not None and sl_pct > 0
    hold_enabled = max_hold_days is not None and max_hold_days > 0

    tp_mult = 1.0 + (tp_pct / 100.0) if tp_enabled else None
    sl_mult = 1.0 - (sl_pct / 100.0) if sl_enabled else None
    walk_horizon = int(max_hold_days) if hold_enabled else len(all_dates)

    trades = []
    for i in range(min_idx, max_idx, 5):
        screen_date = all_dates[i]
        res = run_screener(_df, screen_date)
        if res.empty:
            continue
        # Filter to selected tiers
        res = res[res["Alert"].isin(tiers)]
        if res.empty:
            continue
        for _, rec in res.iterrows():
            t = rec["Ticker"]
            if t not in close_pivot.columns:
                continue
            entry = close_pivot[t].get(all_dates[i])
            if not entry or pd.isna(entry) or entry <= 0:
                continue

            tp_price = entry * tp_mult if tp_enabled else None
            sl_price = entry * sl_mult if sl_enabled else None

            outcome = "TIME"
            exit_price = entry
            hold = 0
            # Walk forward day-by-day. Track whether the walk ended naturally
            # (TP/SL hit, or full hold horizon elapsed) vs ran out of data.
            target_walk_end = i + 1 + walk_horizon
            walk_end = min(target_walk_end, len(all_dates))
            ran_out_of_data = target_walk_end > len(all_dates)
            triggered = False
            for j in range(i + 1, walk_end):
                hold = j - i
                hi = high_pivot[t].get(all_dates[j])
                lo = low_pivot[t].get(all_dates[j])
                cl = close_pivot[t].get(all_dates[j])
                if hi is None or pd.isna(hi):
                    continue

                # Check SL first (conservative)
                if sl_enabled and lo is not None and not pd.isna(lo) and lo <= sl_price:
                    outcome = "SL"; exit_price = sl_price
                    triggered = True; break
                if tp_enabled and hi >= tp_price:
                    outcome = "TP"; exit_price = tp_price
                    triggered = True; break
                exit_price = cl if cl is not None and not pd.isna(cl) else exit_price

            # If we never triggered AND the simulation simply ran out of data,
            # the position is still floating — mark-to-market at last close.
            if not triggered and ran_out_of_data:
                outcome = "OPEN"

            ret_pct = (exit_price - entry) / entry * 100
            trades.append({
                "date":      pd.Timestamp(screen_date),
                "Ticker":    t,
                "Alert":     rec["Alert"],
                "Score":     rec["Score"],
                "Entry":     entry,
                "Exit":      exit_price,
                "Return %":  round(ret_pct, 2),
                "Outcome":   outcome,
                "Hold (d)":  hold,
                "Status":    "Floating" if outcome == "OPEN" else "Realized",
            })

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def compute_atr(g, periods=14):
    """14-day Average True Range. Expects sorted ticker DataFrame with Tertinggi/Terendah/Sebelumnya."""
    if len(g) < 2:
        return 0
    tr = pd.concat([
        (g["Tertinggi"] - g["Terendah"]).abs(),
        (g["Tertinggi"] - g["Sebelumnya"]).abs(),
        (g["Terendah"]  - g["Sebelumnya"]).abs(),
    ], axis=1).max(axis=1)
    return tr.tail(periods).mean()


def compute_action_levels(g, account_size=100_000_000, risk_pct=1.0):
    """
    Returns dict with entry, stop, target1 (2R), target2 (3.5R), risk_pct, rr_ratio,
    suggested_lots, capital_required.

    Stop logic: max(close - 2*ATR, MA20 * 0.98), capped between 3-8% below entry.
    Position sizing: risk_pct% of account / risk_per_share, rounded down to 100-lot blocks.
    """
    if len(g) < 20:
        return None
    close = g.iloc[-1]["Penutupan"]
    if not close or close <= 0:
        return None

    atr = compute_atr(g.tail(40), periods=14)
    ma20 = g.tail(20)["Penutupan"].mean()

    stop_atr = close - 2 * atr
    stop_ma  = ma20 * 0.98
    stop = max(stop_atr, stop_ma)
    stop = max(stop, close * 0.92)  # cap loss at 8%
    stop = min(stop, close * 0.97)  # min 3% gap

    risk_per_share = close - stop
    if risk_per_share <= 0:
        return None

    target1 = close + 2.0 * risk_per_share
    target2 = close + 3.5 * risk_per_share

    risk_idr = account_size * (risk_pct / 100.0)
    raw_shares = risk_idr / risk_per_share
    lots = max(int(raw_shares // 100), 0)
    actual_risk = lots * 100 * risk_per_share

    return {
        "entry":         round(close, 0),
        "stop":          round(stop, 0),
        "target_1":      round(target1, 0),
        "target_2":      round(target2, 0),
        "atr":           round(atr, 1),
        "risk_pct":      round((close - stop) / close * 100, 1),
        "reward_pct":    round((target1 - close) / close * 100, 1),
        "rr_ratio":      round((target1 - close) / risk_per_share, 1),
        "suggested_lots": lots,
        "actual_risk_idr": round(actual_risk, 0),
        "shares_total":  lots * 100,
        "capital_required": round(lots * 100 * close, 0),
    }


def detect_exit_signals(g, entry_price=None):
    """
    Returns list of (severity_emoji_label, message) for an open position.
    """
    sigs = []
    if len(g) < 25:
        return sigs

    last = g.iloc[-1]
    close = last["Penutupan"]
    ma20 = g.tail(20)["Penutupan"].mean()
    ma50 = g.tail(50)["Penutupan"].mean() if len(g) >= 50 else ma20

    if entry_price and entry_price > 0:
        atr = compute_atr(g.tail(40))
        if close < entry_price - 2 * atr:
            sigs.append(("🔴 STOP", f"-2 ATR from entry {int(entry_price):,}"))

    if close < ma20:
        sigs.append(("🟠 MA20 break", f"{(close/ma20-1)*100:+.1f}% vs MA20"))
    if close < ma50:
        sigs.append(("🔴 MA50 break", "Below MA50 — medium trend broken"))

    last3 = g.tail(3)
    if len(last3) == 3 and (last3["Penutupan"] < last3["Sebelumnya"]).all():
        sigs.append(("🟠 3 red", "3 consecutive red days"))

    last7 = g.tail(7)
    nf_neg = (last7["Net Foreign"] < 0).sum()
    if nf_neg >= 5:
        sigs.append(("🟠 Fgn sell", f"Foreign sold {nf_neg}/7 days"))

    avg_vol = g.tail(20)["Volume"].mean()
    for _, r in g.tail(3).iterrows():
        if r["Penutupan"] < r["Sebelumnya"] and r["Volume"] > avg_vol * 1.8:
            sigs.append(("🔴 Distribution", f"High vol red day {r['DATE'].strftime('%m-%d')}"))
            break

    low_20 = g.tail(20)["Terendah"].min()
    if last["Terendah"] <= low_20:
        sigs.append(("🔴 20d low", "New 20-day low"))

    return sigs


@st.cache_data(ttl=3600)
def assess_market_regime(_df, data_version=None):
    """Compatibility wrapper around the liquid-universe weekly regime."""
    snapshot = assess_weekly_regime(_df, as_of_date=_df["DATE"].max())
    labels = {
        "RISK-ON": "🟢 BULL",
        "SELECTIVE": "🟡 NEUTRAL",
        "DEFENSIVE": "🔴 BEAR",
        "UNKNOWN": "❓ UNKNOWN",
    }
    detail = (
        f"{snapshot['pct_above_ma20']:.0f}% liquid stocks > MA20 · "
        f"{snapshot['pct_above_ma50']:.0f}% > MA50 · "
        f"median 20d {snapshot['median_return_20d_pct']:+.1f}%"
    )
    return labels[snapshot["label"]], snapshot["score"], detail


def current_v3_verdict(g):
    """Run v3 logic on a single ticker, return dict or None.
    Used in Deep Dive page to show 'what does v3 say about this stock today?'"""
    BASELINE = 30; SIGNAL = 10
    if len(g) < BASELINE + SIGNAL + 5:
        return None
    base = g.iloc[-(BASELINE+SIGNAL):-SIGNAL]
    sig  = g.iloc[-SIGNAL:]
    lat  = g.iloc[-1]

    b_vol = base["Volume"].mean(); s_vol = sig["Volume"].mean()
    vr = s_vol / b_vol if b_vol > 0 else 0
    vol_spike_max = sig["Volume"].max() / s_vol if s_vol > 0 else 0

    pos_idx = len(g)
    pre30_start = g.iloc[max(0, pos_idx-40)]["Penutupan"] if pos_idx >= 40 else g.iloc[0]["Penutupan"]
    pre30_end   = g.iloc[pos_idx-11]["Penutupan"] if pos_idx >= 11 else g.iloc[-1]["Penutupan"]
    pre30_chg   = (pre30_end - pre30_start) / pre30_start * 100 if pre30_start > 0 else 0

    pos_in_range = position_in_trading_range(g, len(g))

    ma20 = g.tail(20)["Penutupan"].mean()
    ma50 = g.tail(50)["Penutupan"].mean()
    above_ma20 = lat["Penutupan"] > ma20
    above_ma50 = lat["Penutupan"] > ma50

    high_20d = g.tail(20)["Tertinggi"].max()
    break_prox = lat["Penutupan"] / high_20d if high_20d > 0 else 0

    fgn_pos_days = (sig["Net Foreign"] > 0).sum() / len(sig) * 100

    # Gate failures
    fails = []
    if vr < 1.5: fails.append(f"Vol ratio {vr:.2f}x < 1.5x (no surge)")
    elif vr > 5.0: fails.append(f"Vol ratio {vr:.2f}x > 5.0x (likely pump)")
    if vol_spike_max >= 3.5: fails.append(f"Single-day vol spike {vol_spike_max:.1f}x")
    if pos_in_range < 50: fails.append(f"Pos in 52w range only {pos_in_range:.0f}%")
    if not above_ma20: fails.append(f"Below MA20")
    if not above_ma50: fails.append(f"Below MA50")
    if pre30_chg >= 30: fails.append(f"Pre-30d chg {pre30_chg:+.0f}% (already extended)")

    return {
        "passes_v3": len(fails) == 0,
        "fails": fails,
        "vol_ratio": vr, "vol_spike_max": vol_spike_max,
        "break_prox": break_prox, "pos_in_range": pos_in_range,
        "above_ma20": above_ma20, "above_ma50": above_ma50,
        "pre30_chg": pre30_chg, "fgn_pos_days": fgn_pos_days,
        "ma20": ma20, "ma50": ma50,
    }


@st.cache_data(ttl=3600)
def get_ticker_flags(_df, ticker):
    """
    Find every date where the v3 screener would have flagged this ticker.
    Mirrors run_screener gate/scoring logic.
    """
    BASELINE = 30; SIGNAL = 10
    MIN_PRICE = 100; MIN_NILAI = 500_000_000
    VOL_RATIO_MIN = 1.5; VOL_RATIO_MAX = 5.0
    VOL_SPIKE_MAX = 3.5
    POS_IN_RANGE_MIN = 50
    PRE30_CHG_MAX = 30

    g = _df[_df["Kode Saham"] == ticker].copy()
    g = g[g["Penutupan"] > 0].sort_values("DATE").reset_index(drop=True)

    flags = []
    for pos in range(BASELINE + SIGNAL + 5, len(g) + 1):
        base = g.iloc[pos - BASELINE - SIGNAL : pos - SIGNAL].copy()
        sig  = g.iloc[pos - SIGNAL : pos].copy()
        lat  = g.iloc[pos - 1]

        if len(base) < 20 or len(sig) < 7:   continue
        if base["Nilai"].mean() < MIN_NILAI:  continue
        if lat["Penutupan"] < MIN_PRICE:      continue
        if sig["Volume"].sum() == 0:          continue

        b_vol = base["Volume"].mean(); s_vol = sig["Volume"].mean()
        vr = s_vol / b_vol if b_vol > 0 else 0
        if vr < VOL_RATIO_MIN or vr > VOL_RATIO_MAX: continue

        vol_spike_max = sig["Volume"].max() / s_vol if s_vol > 0 else 0
        if vol_spike_max >= VOL_SPIKE_MAX: continue

        f  = sig.iloc[0]["Sebelumnya"]; l = sig.iloc[-1]["Penutupan"]
        pc = (l - f) / f * 100 if f > 0 else 0
        if pc < -30:                          continue

        pre30_start = g.iloc[max(0, pos-40)]["Penutupan"] if pos >= 40 else g.iloc[0]["Penutupan"]
        pre30_end   = g.iloc[pos-11]["Penutupan"] if pos >= 11 else g.iloc[-1]["Penutupan"]
        pre30_chg   = (pre30_end - pre30_start) / pre30_start * 100 if pre30_start > 0 else 0
        if pre30_chg >= PRE30_CHG_MAX: continue

        pos_in_range = position_in_trading_range(g, pos)
        if pos_in_range < POS_IN_RANGE_MIN: continue

        ma20       = g.iloc[max(0, pos - 20):pos]["Penutupan"].mean()
        above_ma20 = bool(lat["Penutupan"] > ma20)
        if not above_ma20: continue

        ma50       = g.iloc[max(0, pos - 50):pos]["Penutupan"].mean()
        above_ma50 = bool(lat["Penutupan"] > ma50)
        if not above_ma50: continue

        high_20d   = g.iloc[max(0, pos - 20):pos]["Tertinggi"].max()
        break_prox = lat["Penutupan"] / high_20d if high_20d > 0 else 0

        up_mask  = sig["Penutupan"] >= sig["Sebelumnya"]
        up_vol   = sig.loc[ up_mask, "Volume"].sum()
        down_vol = sig.loc[~up_mask, "Volume"].sum()
        up_bias  = up_vol / down_vol if down_vol > 0 else 5.0

        obv_pos = obv_slope_is_positive(sig)

        fgn_positive_days = (sig["Net Foreign"] > 0).sum() / len(sig) * 100

        strong_bits = sum([
            break_prox >= 0.95,
            pos_in_range >= 75,
            fgn_positive_days >= 60,
            bool(obv_pos),
        ])
        alert = "🔴 STRONG" if strong_bits >= 3 else "🟡 WATCH" if strong_bits >= 2 else "⚪ RADAR"

        flags.append({"DATE": lat["DATE"], "Alert": alert,
                      "Close": lat["Penutupan"], "Vol Ratio": round(vr, 2),
                      "Up-Vol Bias": round(up_bias, 2)})

    return pd.DataFrame(flags) if flags else pd.DataFrame()


# ── Sidebar nav ───────────────────────────────────────────────────────────────
st.sidebar.title("IDX Dashboard")
page = st.sidebar.radio(
    "Navigate",
    [
        "🎯 Daily Plan",
        "🗓️ Next Week",
        "📊 Market Overview",
        "🔍 Screener",
        "📈 Stock Deep Dive",
        "💼 Portfolio",
    ],
)

df = load_data()
trading_days = get_trading_days()
latest_date  = trading_days[0]
master_version = os.path.getmtime(master_data_path())
existing_holdings = {}
for position in st.session_state.get("portfolio", []):
    ticker = str(position.get("ticker", ""))
    lots = max(int(position.get("shares", 0) or 0), 0)
    if ticker and lots > 0:
        existing_holdings[ticker] = existing_holdings.get(ticker, 0) + lots * 100

st.sidebar.markdown("---")
st.sidebar.markdown("**⚙️ Trading settings**")
account_size = st.sidebar.number_input(
    "Account size (IDR)", value=100_000_000, step=10_000_000,
    min_value=1_000_000, format="%d", help="Used for position sizing in Daily Plan, Next Week, and Screener.",
)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.25, 3.0, 1.0, 0.25,
                             help="Requested account risk; the weekly model caps this at 0.25% in RISK-ON and 0.15% in SELECTIVE regimes.")

st.sidebar.markdown("---")
st.sidebar.caption(f"Data: {df['DATE'].min().date()} → {df['DATE'].max().date()}")
st.sidebar.caption(f"Trading days: {len(trading_days)}")
st.sidebar.caption(f"Stocks: {df['Kode Saham'].nunique()}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — Daily Plan (action-oriented home)
# ══════════════════════════════════════════════════════════════════════════════
if page == "🎯 Daily Plan":
    st.title("🎯 Daily Plan")
    st.caption(f"As of {pd.Timestamp(latest_date).strftime('%A, %Y-%m-%d')} — your daily action briefing")

    # ── Market Regime ────────────────────────────────────────────────────────
    regime, regime_score, regime_detail = assess_market_regime(df, master_version)
    rc1, rc2, rc3, rc4 = st.columns([1.2, 1, 1, 1])
    rc1.metric("Market Regime", regime)
    last_day = df[df["DATE"] == latest_date]
    active_today = last_day[last_day["Volume"] > 0]
    rc2.metric("Gainers / Losers",
               f"{(active_today['pct_chg']>0).sum()} / {(active_today['pct_chg']<0).sum()}")
    rc3.metric("Total Value",
               f"Rp {active_today['Nilai'].sum()/1e12:.1f}T")
    nf_today = active_today["Net Foreign Est Value"].sum() / 1e9
    rc4.metric("Est. Net Foreign Value",
               f"Rp {nf_today:+.1f}B",
               delta="BUY" if nf_today > 0 else "SELL",
               delta_color="normal" if nf_today > 0 else "inverse")
    st.caption(f"📊 {regime_detail}")

    # Regime guidance
    if regime.startswith("🔴"):
        st.warning("🔴 **Bear regime** — no new weekly long entries. Protect capital and wait for "
                   "liquid-market breadth to recover.")
    elif regime.startswith("🟢"):
        st.success("🟢 **Bull regime** — breadth supports a pilot long book, capped at 25% gross "
                   "exposure and 0.25% account risk per position.")
    else:
        st.info("🟡 **Neutral regime** — stay selective. The model caps the pilot book at 15% "
                "gross exposure and 0.15% account risk per position.")

    st.markdown("---")

    # ── Actionable recommendations (five-session model) ──────────────────────
    st.subheader("🔥 Actionable Recommendations — next-session conditions")

    with st.spinner("Ranking the active, liquid universe and applying risk guards..."):
        daily_regime = assess_weekly_regime(df, as_of_date=latest_date)
        daily_rankings = get_weekly_rankings(df, latest_date, master_version)
        daily_candidates = select_actionable_candidates(daily_rankings)
        daily_plan = build_weekly_trade_plan(
            df,
            daily_candidates,
            daily_regime,
            as_of_date=latest_date,
            account_size=account_size,
            requested_risk_pct=risk_pct,
            max_positions=5,
            existing_holdings=existing_holdings,
        )

    if daily_plan.empty:
        st.info("No new long entries pass the model, execution guards, and current regime limits.")
    else:
        setup_map = daily_candidates.set_index("Ticker")["Recommendation"]
        action_df = daily_plan.copy()
        action_df.insert(1, "Setup", action_df["Ticker"].map(setup_map))
        action_df["Capital"] = action_df["Capital IDR"].map(
            lambda value: f"Rp {value / 1e6:.1f}M"
        )
        action_df["Risk"] = action_df["Risk IDR"].map(
            lambda value: f"Rp {value / 1e3:.0f}K"
        )
        action_columns = [
            "Ticker", "Setup", "Weekly Score", "Trigger", "Max Entry", "Stop",
            "Target 1R", "Target 2R", "ATR %", "Lots", "Capital", "Risk",
        ]
        st.dataframe(action_df[action_columns], width="stretch", hide_index=True)
        deployed = daily_plan["Capital IDR"].sum()
        st.caption(
            f"Five-session model · Rp {deployed / 1e6:.1f}M proposed "
            f"({deployed / account_size * 100:.1f}% of account). The sidebar risk request "
            f"is capped at {daily_regime['risk_cap_pct']:.2f}% per position; pairwise "
            "60-session correlation is capped at 0.65. Trigger is a confirmation level, "
            "not an assumed fill; sizing uses the worst allowed entry. Existing portfolio "
            "market value and correlations consume the same exposure budget."
        )

    st.markdown("---")

    # ── Recent Picks Tracker (did they work?) ─────────────────────────────────
    st.subheader("📈 Recent Picks Tracker — last 5 / 10 / 20 trading days")
    st.caption("Legacy v3 diagnostics only: how did its earlier picks perform? This is "
               "monitoring evidence, not the current actionable engine or proof of an edge.")

    picks_history = []
    lookback_dates = [trading_days[i] for i in [4, 9, 19] if i < len(trading_days)]
    lookback_labels = ["5d ago", "10d ago", "20d ago"][:len(lookback_dates)]
    price_pivot = df.pivot_table(index="DATE", columns="Kode Saham", values="Penutupan")
    latest_prices = price_pivot.iloc[-1] if not price_pivot.empty else pd.Series()

    for lbl, dt in zip(lookback_labels, lookback_dates):
        s = run_screener(df, dt)
        if s.empty:
            continue
        for _, r in s.iterrows():
            t = r["Ticker"]
            entry_p = price_pivot[t].get(pd.Timestamp(dt)) if t in price_pivot.columns else None
            now_p   = latest_prices.get(t) if t in latest_prices.index else None
            if not entry_p or pd.isna(entry_p) or entry_p <= 0:
                continue
            if not now_p or pd.isna(now_p):
                ret = None
            else:
                ret = (now_p - entry_p) / entry_p * 100
            picks_history.append({
                "Picked":  lbl,
                "Date":    pd.Timestamp(dt).strftime("%Y-%m-%d"),
                "Ticker":  t,
                "Alert":   r["Alert"],
                "Score":   r["Score"],
                "Entry":   f"{int(entry_p):,}",
                "Now":     f"{int(now_p):,}" if now_p else "—",
                "Return":  ret,
            })

    if picks_history:
        ph_df = pd.DataFrame(picks_history)
        # Summary by lookback
        summary_rows = []
        for lbl in lookback_labels:
            sub = ph_df[ph_df["Picked"] == lbl]
            if sub.empty: continue
            rets = sub["Return"].dropna()
            if rets.empty: continue
            summary_rows.append({
                "Picked":      lbl,
                "Date":        sub["Date"].iloc[0],
                "Picks":       len(sub),
                "Avg Return":  f"{rets.mean():+.2f}%",
                "Median":      f"{rets.median():+.2f}%",
                "Win Rate":    f"{(rets > 0).mean()*100:.0f}%",
                "Best":        f"{rets.max():+.1f}%",
                "Worst":       f"{rets.min():+.1f}%",
            })

        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        with st.expander(f"See all {len(ph_df)} historical picks"):
            disp = ph_df.copy()
            disp["Return"] = disp["Return"].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough history to show recent picks.")

    st.markdown("---")

    # ── Portfolio Health Alerts ─────────────────────────────────────────────
    st.subheader("⚠️ Portfolio Health Alerts")
    portfolio = st.session_state.get("portfolio", [])
    portfolio = [p for p in portfolio if p.get("shares", 0) > 0 and p.get("ticker")]

    if not portfolio:
        st.info("No positions in portfolio. Add positions on the 💼 Portfolio page to see exit alerts here.")
    else:
        alert_rows = []
        for pos in portfolio:
            t = pos["ticker"]
            g_t = df[df["Kode Saham"] == t].sort_values("DATE").reset_index(drop=True)
            if g_t.empty: continue
            sigs = detect_exit_signals(g_t, pos.get("entry"))
            close = g_t.iloc[-1]["Penutupan"]
            entry = pos.get("entry", 0)
            pnl_pct = (close - entry) / entry * 100 if entry > 0 else 0
            alert_rows.append({
                "Ticker":   t,
                "Close":    f"{int(close):,}",
                "Entry":    f"{int(entry):,}" if entry else "—",
                "P&L":      f"{pnl_pct:+.1f}%" if entry else "—",
                "Alerts":   " · ".join([f"{s[0]} {s[1]}" for s in sigs]) if sigs else "✅ Healthy",
                "Severity": len(sigs),
            })
        alert_df = pd.DataFrame(alert_rows).sort_values("Severity", ascending=False).drop(columns=["Severity"])

        def color_alerts(val):
            if not isinstance(val, str): return ""
            if "🔴" in val: return "background-color: #ffcdd2; font-weight: bold"
            if "🟠" in val: return "background-color: #ffe0b2"
            if "✅" in val: return "background-color: #c8e6c9"
            return ""

        st.dataframe(
            alert_df.style.map(color_alerts, subset=["Alerts"]),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")

    # ── Top Foreign Movers Today ──────────────────────────────────────────────
    st.subheader("🌐 Foreign Money Today")
    fcc1, fcc2 = st.columns(2)
    active_today = active_today.copy()
    active_today["Net Fgn Est (B)"] = active_today["Net Foreign Est Value"] / 1e9
    with fcc1:
        st.markdown("**Top 5 Net Buy**")
        fb = (active_today.nlargest(5, "Net Foreign Est Value")
                          [["Kode Saham", "Penutupan", "pct_chg", "Net Fgn Est (B)"]]
                          .copy())
        fb.columns = ["Ticker", "Close", "Chg %", "Net Fgn Est (B)"]
        st.dataframe(fb.reset_index(drop=True), use_container_width=True, hide_index=True)
    with fcc2:
        st.markdown("**Top 5 Net Sell**")
        fs = (active_today.nsmallest(5, "Net Foreign Est Value")
                          [["Kode Saham", "Penutupan", "pct_chg", "Net Fgn Est (B)"]]
                          .copy())
        fs.columns = ["Ticker", "Close", "Chg %", "Net Fgn Est (B)"]
        st.dataframe(fs.reset_index(drop=True), use_container_width=True, hide_index=True)
    st.caption("Estimated value = net foreign share volume × closing price; actual fills vary intraday.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Next Week Strategy
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗓️ Next Week":
    next_session = pd.Timestamp(latest_date) + pd.offsets.BDay(1)
    week_end = next_session + pd.offsets.BDay(4)
    st.title("🗓️ Next Week Strategy")
    st.caption(
        f"Plan for {next_session:%-d %B}–{week_end:%-d %B %Y}, built only from data "
        f"available through {pd.Timestamp(latest_date):%Y-%m-%d}."
    )

    with st.spinner("Ranking the liquid universe on one-week factors..."):
        weekly_regime = assess_weekly_regime(df, as_of_date=latest_date)
        weekly_rankings = get_weekly_rankings(df, latest_date, master_version)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Market regime", weekly_regime["label"])
    mc2.metric("Liquid stocks > MA20", f"{weekly_regime['pct_above_ma20']:.0f}%")
    mc3.metric("Median 20d return", f"{weekly_regime['median_return_20d_pct']:+.1f}%")
    mc4.metric("Pilot exposure cap", f"{weekly_regime['max_exposure_pct']:.0f}%")

    st.info(
        "The 2026 holdout test supports this model as a relative ranker, not as a "
        "guaranteed positive-return system. Top-ten relative performance improved, "
        "but average absolute returns were negative after an assumed 0.50% round trip."
    )

    if weekly_rankings.empty:
        st.warning("No active stocks passed the weekly liquidity and history requirements.")
        st.stop()

    # A cautious execution universe: meaningful liquidity, positive recent foreign
    # share flow, and no sub-Rp500 names. The score itself remains unmodified.
    guarded = select_actionable_candidates(weekly_rankings)
    weekly_plan = build_weekly_trade_plan(
        df,
        guarded,
        weekly_regime,
        as_of_date=latest_date,
        account_size=account_size,
        requested_risk_pct=risk_pct,
        max_positions=5,
        existing_holdings=existing_holdings,
    )

    st.subheader("Conditional trade plan")
    if weekly_plan.empty:
        st.warning("No new long entries under the current regime and risk limits.")
    else:
        plan_display = weekly_plan.copy()
        setup_map = guarded.set_index("Ticker")["Recommendation"]
        plan_display.insert(1, "Setup", plan_display["Ticker"].map(setup_map))
        plan_display["Capital"] = plan_display["Capital IDR"].map(
            lambda value: f"Rp {value / 1e6:.1f}M"
        )
        plan_display["Risk"] = plan_display["Risk IDR"].map(
            lambda value: f"Rp {value / 1e3:.0f}K"
        )
        plan_columns = [
            "Ticker", "Setup", "Weekly Score", "Trigger", "Max Entry", "Stop",
            "Target 1R", "Target 2R", "ATR %", "Lots", "Capital", "Risk",
        ]
        st.dataframe(plan_display[plan_columns], width="stretch", hide_index=True)
        deployed = weekly_plan["Capital IDR"].sum()
        st.caption(
            f"Proposed capital: Rp {deployed / 1e6:.1f}M "
            f"({deployed / account_size * 100:.1f}% of account). Pairwise 60-day "
            "return correlation is capped at 0.65. Actionable names must be in the "
            "top 30 ranks, average at least Rp10B/day, close at Rp500 or above, and "
            "have non-negative 10-session foreign share flow. Existing portfolio market "
            "value and correlations consume the same exposure budget."
        )

    st.markdown(
        """
**Execution rules**

1. Wait through the first 30 minutes. Enter only if price holds at or reclaims **Trigger** without trading above **Max Entry**; skip a larger gap and do not catch a gap-down that cannot reclaim.
2. Risk no more than the displayed amount. Never widen the stop after entry.
3. Recalculate 1R and 2R from the actual fill; the table uses the worst allowed entry. At +1R, take partial profit or tighten risk.
4. Exit any remaining tactical position by the fifth IDX session close (Friday 7 August for this plan). Do not carry the weekly trade through the US payroll weekend event.
        """
    )
    st.caption(
        "Displayed levels are conservatively rounded to the official "
        "[IDX equity price fractions](https://testing.idx.id/en/investhub/trading-mechanism/)."
    )

    st.markdown("---")
    st.subheader("Liquid-universe factor leaders")
    factor_columns = [
        "Ticker", "Weekly Score", "Close", "Avg Value/day (B)", "Break60 %",
        "RV20 %", "Volume CV10", "Avg Trade (M)", "Abs Gap5 %",
        "Foreign Accel %Vol", "Foreign Net10 %Vol", "Momentum20 %",
    ]
    leaders = weekly_rankings[factor_columns].head(15).copy()
    numeric_display = [c for c in factor_columns if c not in ["Ticker", "Close"]]
    leaders[numeric_display] = leaders[numeric_display].round(2)
    st.dataframe(leaders, width="stretch", hide_index=True)
    st.caption(
        "Score = equal-weight percentile blend of proximity to the 60d high, low "
        "20d volatility, controlled 10d volume, larger average trade value, low "
        "five-day absolute opening-gap risk, and improving foreign flow as a share of volume."
    )
    with st.expander("How the factors map to upside support and downside risk"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Factor": "60d high proximity",
                        "Upside support": "Price holds close to its 60-session high",
                        "Downside risk": "Repeated rejection or loss of the recent range",
                    },
                    {
                        "Factor": "20d realized volatility",
                        "Upside support": "Lower, steadier return variability",
                        "Downside risk": "High volatility and frequent stop-outs",
                    },
                    {
                        "Factor": "10d volume consistency",
                        "Upside support": "Sustained participation across sessions",
                        "Downside risk": "One-off spikes that can fade quickly",
                    },
                    {
                        "Factor": "Average trade value",
                        "Upside support": "Larger average ticket size",
                        "Downside risk": "Small-ticket, easier-to-distort activity",
                    },
                    {
                        "Factor": "Five-day absolute opening gap",
                        "Upside support": "Controlled entry near the prior close",
                        "Downside risk": "Repeated gaps increase chase and reversal risk",
                    },
                    {
                        "Factor": "Foreign-flow acceleration",
                        "Upside support": "Net foreign share flow is improving",
                        "Downside risk": "Flow is deteriorating or persistently negative",
                    },
                    {
                        "Factor": "Liquid-market breadth",
                        "Upside support": "More liquid stocks hold above MA20 and MA50",
                        "Downside risk": "Narrow or weakening breadth cuts exposure",
                    },
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "These are probabilistic ranking signals, not deterministic predictions. "
            "The model combines them and then applies liquidity, gap, correlation, "
            "position-size, and market-regime controls."
        )

    st.markdown("---")
    st.subheader("How the existing v3 picks change on a five-day horizon")
    v3_now = run_screener(df, latest_date)
    if not v3_now.empty:
        weekly_overlay = weekly_rankings[
            [
                "Ticker", "Weekly Score", "Avg Value/day (B)", "RV20 %",
                "Foreign Net10 %Vol", "Foreign +Days %", "Momentum20 %",
            ]
        ]
        overlay = v3_now[
            ["Ticker", "Alert", "Score", "Close", "Avg Val/day (B)"]
        ].merge(weekly_overlay, on="Ticker", how="left")

        def weekly_action(row):
            if pd.isna(row["Weekly Score"]):
                return "EXCLUDE — below Rp5B/day or insufficient history"
            if row["RV20 %"] >= 4:
                return "DEMOTE — excessive weekly volatility"
            if row["Momentum20 %"] >= 20 and row["Foreign Net10 %Vol"] < 0:
                return "EVENT ONLY — extended while foreign flow is negative"
            if row["Foreign Net10 %Vol"] < 0:
                return "WATCH — foreign share flow is negative"
            if row["Avg Value/day (B)"] < 10:
                return "SMALL SATELLITE — thinner liquidity"
            return "RETAIN — next-open conditions still required"

        overlay["Weekly Action"] = overlay.apply(weekly_action, axis=1)
        overlay["Weekly Score"] = overlay["Weekly Score"].round(1)
        overlay["RV20 %"] = overlay["RV20 %"].round(2)
        overlay["Foreign Net10 %Vol"] = overlay["Foreign Net10 %Vol"].round(1)
        overlay["Momentum20 %"] = overlay["Momentum20 %"].round(1)
        st.dataframe(overlay, width="stretch", hide_index=True)

    if next_session == pd.Timestamp("2026-08-03"):
        st.markdown("---")
        st.subheader("Event-risk calendar · WIB")
        event_rows = [
            {
                "Date": "Mon 3 Aug",
                "Confirmed event": "Indonesia July PMI (07:30), July CPI; reported BFIN index inclusion effective",
                "Trading response": "Wait for the first reaction; do not chase a BFIN gap.",
            },
            {
                "Date": "Tue 4 Aug",
                "Confirmed event": "US JOLTS at 21:00, after IDX close",
                "Trading response": "Recheck global risk tone before Wednesday entries.",
            },
            {
                "Date": "Wed 5 Aug",
                "Confirmed event": "Indonesia Q2 GDP; US petroleum report after IDX close",
                "Trading response": "GDP is the main local macro test; oil is contextual for AKRA.",
            },
            {
                "Date": "Fri 7 Aug",
                "Confirmed event": "Indonesia reserves; US payrolls at 19:30, after IDX close",
                "Trading response": "Close the five-day tactical book before the weekend event.",
            },
        ]
        st.dataframe(pd.DataFrame(event_rows), width="stretch", hide_index=True)
        st.markdown(
            "Sources: [S&P PMI calendar](https://www.pmi.spglobal.com/Public/Release/ReleaseDates?language=en) · "
            "[Bank Indonesia calendar](https://www.bi.go.id/id/publikasi/Kalender/Default.aspx) · "
            "[BI 2026 meeting schedule](https://www.bi.go.id/en/publikasi/ruang-media/news-release/Pages/sp_2730825.aspx) · "
            "[US BLS August calendar](https://www.bls.gov/schedule/2026/08_sched_list.htm) · "
            "[EIA petroleum schedule](https://www.eia.gov/petroleum/supply/weekly/schedule.php) · "
            "[BFIN H1 2026 release](https://www.bfi.co.id/id/corporate/hubungan-investor/siaran-pers) · "
            "[reported BFIN index change](https://www.liputan6.com/saham/read/8255794/bei-ubah-penghuni-saham-indeks-lq45-hingga-idx30)"
        )
        st.caption(
            "BFIN has the clearest stock-specific catalyst, but it also gained 22.8% "
            "over 20 sessions and recorded negative foreign share flow over the last "
            "10 sessions. AKRA and SRTG have no confirmed issuer event this week."
        )

    with st.expander("Validation details and limitations"):
        st.markdown(
            """
- Development period: 2025. Holdout period: January–July 2026.
- Five-session label: next market open through the fifth subsequent close.
- Holdout cross-sectional information coefficient: **+0.130** for the six-factor blend.
- Top-ten relative spread: approximately **+1.21 to +1.90 percentage points per week** versus the liquid universe across five rebalance offsets.
- Absolute gross returns ranged from **−0.18% to +0.18% per week**; after an assumed **0.50% round trip**, mean net returns remained negative.
- The 130 daily holdout screens overlap; there are only about 26 independent weekly observations per rebalance schedule.
- The displayed range uses business weekdays. The 3–7 August plan has five expected sessions; confirm official IDX holiday notices before reusing it for another week.
            """
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Market Overview
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Market Overview":
    st.title("📊 Market Overview")

    sel_date = st.selectbox(
        "Trading date",
        options=trading_days,
        format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
    )

    day = df[df["DATE"] == sel_date].copy()
    active = day[day["Volume"] > 0]

    if active.empty:
        st.warning("No trading data for this date.")
        st.stop()

    # ── KPI row ───────────────────────────────────────────────────────────────
    gainers  = (active["pct_chg"] > 0).sum()
    losers   = (active["pct_chg"] < 0).sum()
    flat     = (active["pct_chg"] == 0).sum()
    tot_val  = active["Nilai"].sum()
    tot_vol  = active["Volume"].sum()
    net_fgn  = active["Net Foreign Est Value"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Value",    f"Rp {tot_val/1e12:.2f}T")
    c2.metric("Total Volume",   f"{tot_vol/1e9:.1f}B shares")
    c3.metric("Gainers",        f"{gainers:,}", delta=f"+{gainers}")
    c4.metric("Losers",         f"{losers:,}",  delta=f"-{losers}", delta_color="inverse")
    c5.metric("Est. Net Foreign Value",
              f"Rp {net_fgn/1e9:.1f}B",
              delta="NET BUY" if net_fgn>0 else "NET SELL",
              delta_color="normal" if net_fgn>0 else "inverse")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    # Advance / Decline pie
    with col_left:
        st.subheader("Market Breadth")
        fig = go.Figure(go.Pie(
            labels=["Gainers","Losers","Flat"],
            values=[gainers, losers, flat],
            marker_colors=["#00c853","#ff1744","#9e9e9e"],
            hole=0.4,
        ))
        fig.update_layout(height=300, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Top gainers bar
    with col_right:
        st.subheader("Top 10 Gainers")
        top_g = active.nlargest(10, "pct_chg")[["Kode Saham","pct_chg","Penutupan"]].copy()
        fig = px.bar(top_g, x="Kode Saham", y="pct_chg",
                     color="pct_chg", color_continuous_scale="Greens",
                     labels={"pct_chg":"Change %","Kode Saham":"Ticker"},
                     height=300)
        fig.update_layout(showlegend=False, margin=dict(t=20,b=20), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    # Top losers
    with col_l2:
        st.subheader("Top 10 Losers")
        top_l = active.nsmallest(10, "pct_chg")[["Kode Saham","pct_chg","Penutupan"]].copy()
        fig = px.bar(top_l, x="Kode Saham", y="pct_chg",
                     color="pct_chg", color_continuous_scale="Reds_r",
                     labels={"pct_chg":"Change %","Kode Saham":"Ticker"},
                     height=300)
        fig.update_layout(showlegend=False, margin=dict(t=20,b=20), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Top by value
    with col_r2:
        st.subheader("Top 10 by Trading Value")
        top_v = active.nlargest(10, "Nilai")[["Kode Saham","Nilai","pct_chg"]].copy()
        top_v["Nilai (B)"] = top_v["Nilai"]/1e9
        top_v["color"] = top_v["pct_chg"].apply(lambda x: "#00c853" if x>0 else "#ff1744" if x<0 else "#9e9e9e")
        fig = go.Figure(go.Bar(
            x=top_v["Kode Saham"], y=top_v["Nilai (B)"],
            marker_color=top_v["color"],
        ))
        fig.update_layout(height=300, margin=dict(t=20,b=20),
                          yaxis_title="Value (B IDR)", xaxis_title="Ticker")
        st.plotly_chart(fig, use_container_width=True)

    # Foreign flow
    st.subheader("Top Foreign Net Buy / Sell")
    active["Net Fgn Est (B)"] = active["Net Foreign Est Value"] / 1e9
    fc1, fc2 = st.columns(2)
    with fc1:
        fb = active.nlargest(10,"Net Foreign Est Value")[["Kode Saham","Nama Perusahaan","Penutupan","pct_chg","Net Fgn Est (B)"]].copy()
        st.dataframe(fb.reset_index(drop=True), use_container_width=True)
    with fc2:
        fs = active.nsmallest(10,"Net Foreign Est Value")[["Kode Saham","Nama Perusahaan","Penutupan","pct_chg","Net Fgn Est (B)"]].copy()
        st.dataframe(fs.reset_index(drop=True), use_container_width=True)
    st.caption("Estimated value = net foreign share volume × closing price; actual fills vary intraday.")

    # Market breadth over time
    st.markdown("---")
    st.subheader("Market Breadth Trend (last 60 days)")
    recent = df[df["DATE"] >= df["DATE"].max() - pd.Timedelta(days=90)]
    breadth = recent.groupby("DATE").apply(
        lambda g: pd.Series({
            "Gainers":   (g["pct_chg"]>0).sum(),
            "Losers":    (g["pct_chg"]<0).sum(),
            "Net Fgn Est (B)": g["Net Foreign Est Value"].sum()/1e9,
            "Value (T)": g["Nilai"].sum()/1e12,
        }), include_groups=False
    ).reset_index()
    breadth["A/D Line"] = (breadth["Gainers"] - breadth["Losers"]).cumsum()

    fig = go.Figure()
    fig.add_bar(x=breadth["DATE"], y=breadth["Gainers"], name="Gainers", marker_color="#00c853")
    fig.add_bar(x=breadth["DATE"], y=-breadth["Losers"], name="Losers", marker_color="#ff1744")
    fig.update_layout(barmode="relative", height=300, margin=dict(t=10,b=10),
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Screener
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Screener":
    st.title("🔍 Pre-Rocket Screener (v3)")
    st.warning(
        "Legacy research screen: use **Daily Plan** or **Next Week** for current "
        "actionable recommendations, portfolio caps, and next-session entry rules."
    )
    st.caption("Development sample: 47.3% win rate and +7.16% mean; these in-sample, "
               "same-close, cost-free figures are not a forecast. "
               "Gates: 1.5≤VolRatio≤5.0 · VolSpikeMax<3.5 · PosRange≥50 · "
               ">MA20 & MA50 · Pre30Chg<30")

    screen_date = st.selectbox(
        "Screen as of",
        options=trading_days,
        format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
    )

    with st.spinner("Scanning all stocks..."):
        results = run_screener(df, screen_date)

    if results.empty:
        st.warning("No candidates found for this date.")
        st.stop()

    strong = results[results["Alert"]=="🔴 STRONG"]
    watch  = results[results["Alert"]=="🟡 WATCH"]
    radar  = results[results["Alert"]=="⚪ RADAR"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Candidates", len(results))
    c2.metric("🔴 STRONG", len(strong))
    c3.metric("🟡 WATCH",  len(watch))
    c4.metric("⚪ RADAR",  len(radar))

    st.markdown("---")

    def color_score(val):
        if val >= 70: return "background-color: #b71c1c; color: white"
        if val >= 50: return "background-color: #e53935; color: white"
        if val >= 30: return "background-color: #ef9a9a"
        return ""

    # Columns to show in each section (v3)
    display_cols = ["Ticker", "Company", "Score", "Why", "Close",
                    "Vol Ratio", "Up-Vol Bias", "Brk Prox%", "PosRange%",
                    "Fgn+Days%", "VolCV", "OBV+",
                    "10d Chg%", "Pre30 Chg%", "Net Fgn (M sh)", "Avg Val/day (B)"]

    # ── Action Levels Table (entry / stop / target / position size) ──────────
    st.subheader("🎯 Trade Plan — Entry / Stop / Target")
    st.caption(f"Sized for **Rp {account_size/1e6:.0f}M** at **{risk_pct}% risk per trade** "
               f"(adjust in sidebar). Stops use max(2-ATR, MA20×0.98) capped 3-8%. Targets at 2R.")

    action_rows = []
    for _, r in results.iterrows():
        t = r["Ticker"]
        g_t = df[df["Kode Saham"] == t].sort_values("DATE").reset_index(drop=True)
        g_t = g_t[g_t["DATE"] <= screen_date]
        levels = compute_action_levels(g_t, account_size, risk_pct)
        if not levels:
            continue
        action_rows.append({
            "Ticker":     t,
            "Alert":      r["Alert"],
            "Score":      r["Score"],
            "Entry":      int(levels["entry"]),
            "Stop":       int(levels["stop"]),
            "Risk%":      levels["risk_pct"],
            "Target":     int(levels["target_1"]),
            "Reward%":    levels["reward_pct"],
            "R:R":        levels["rr_ratio"],
            "Lots":       levels["suggested_lots"],
            "Capital":    f"Rp {levels['capital_required']/1e6:.1f}M",
            "Risk IDR":   f"Rp {levels['actual_risk_idr']/1e6:.2f}M",
            "Why":        r["Why"][:55] + ("…" if len(r["Why"]) > 55 else ""),
        })

    if action_rows:
        action_df = pd.DataFrame(action_rows)

        def color_rr(val):
            if not isinstance(val, (int, float)): return ""
            if val >= 3.0: return "background-color: #1b5e20; color: white; font-weight: bold"
            if val >= 2.0: return "background-color: #66bb6a; color: white"
            if val >= 1.5: return "background-color: #fff59d"
            return "background-color: #ffcdd2"

        st.dataframe(
            action_df.style
                .map(color_score, subset=["Score"])
                .map(color_rr, subset=["R:R"]),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")
    st.subheader("📋 Full Diagnostics")

    if not strong.empty:
        st.markdown("**🔴 STRONG — 3+ high-conviction signals**")
        st.caption("Breakout-ready · top of 52w range · consistent foreign buying · OBV uptrend")
        show_cols = [c for c in display_cols if c in strong.columns]
        st.dataframe(
            strong[show_cols].style.map(color_score, subset=["Score"]),
            use_container_width=True, hide_index=True,
        )

    if not watch.empty:
        st.markdown("**🟡 WATCH — 2 high-conviction signals**")
        show_cols = [c for c in display_cols if c in watch.columns]
        st.dataframe(
            watch[show_cols].style.map(color_score, subset=["Score"]),
            use_container_width=True, hide_index=True,
        )

    if not radar.empty:
        with st.expander(f"⚪ RADAR — Passed all v3 gates ({len(radar)} stocks)"):
            show_cols = [c for c in display_cols if c in radar.columns]
            st.dataframe(radar[show_cols], use_container_width=True, hide_index=True)

    # Score distribution
    st.markdown("---")
    fig = px.histogram(results, x="Score", color="Alert",
                       color_discrete_map={"🔴 STRONG":"#e53935","🟡 WATCH":"#ffc107","⚪ RADAR":"#90a4ae"},
                       nbins=20, title="Score Distribution",
                       labels={"Score":"Composite Score","count":"# Stocks"})
    fig.update_layout(height=300, margin=dict(t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ── Forward Returns ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Recommendation Returns")
    st.caption("How these picks actually performed after the screen date. "
               "'Now %' = return from entry close to the latest available date.")

    fwd = compute_forward_returns(df, results["Ticker"].tolist(), screen_date)

    if fwd.empty:
        st.info("Forward return data not available (need data after the screen date).")
    else:
        # Merge with alert/score/why for context
        meta = results[["Ticker","Alert","Score","Why"]].copy()
        fwd = fwd.merge(meta, on="Ticker", how="left")
        col_order = ["Ticker","Alert","Score","Why","Entry Close","+1d %","+7d %","+30d %","Now %"]
        fwd = fwd[[c for c in col_order if c in fwd.columns]]

        def color_ret(val):
            if not isinstance(val, (int, float)) or pd.isna(val):
                return "color: #888"
            if val > 10:  return "background-color: #1b5e20; color: white; font-weight: bold"
            if val > 0:   return "color: #00c853; font-weight: bold"
            if val < -10: return "background-color: #b71c1c; color: white; font-weight: bold"
            if val < 0:   return "color: #ff1744; font-weight: bold"
            return ""

        ret_cols = [c for c in ["+1d %","+7d %","+30d %","Now %"] if c in fwd.columns]

        st.dataframe(
            fwd.style
               .map(color_score, subset=["Score"])
               .map(color_ret,   subset=ret_cols),
            use_container_width=True, hide_index=True,
        )

        # Summary stats by alert tier
        tiers = [
            ("🔴 STRONG", "🔴 STRONG"),
            ("🟡 WATCH",  "🟡 WATCH"),
            ("⚪ RADAR",  "⚪ RADAR"),
        ]
        st.markdown("**Win rate by signal tier** (% picks with positive return)")
        for tier_label, tier_key in tiers:
            tier_fwd = fwd[fwd["Alert"] == tier_key] if "Alert" in fwd.columns else pd.DataFrame()
            if tier_fwd.empty:
                continue
            st.markdown(f"**{tier_label}** — {len(tier_fwd)} stock(s)")
            cols = st.columns(len(ret_cols))
            for col_w, rc in zip(cols, ret_cols):
                series = tier_fwd[rc].dropna()
                if len(series) == 0:
                    col_w.metric(rc, "—", delta="no data")
                    continue
                win_rate = (series > 0).sum() / len(series) * 100
                avg_ret  = series.mean()
                col_w.metric(
                    rc,
                    f"{win_rate:.0f}% wins ({len(series)})",
                    delta=f"avg {avg_ret:+.1f}%",
                    delta_color="normal" if avg_ret >= 0 else "inverse",
                )

    # ── Screener Accuracy Over Time ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Screener Accuracy Over Time")

    hc1, hc2 = st.columns([1, 4])
    horizon = hc1.selectbox(
        "Forward horizon",
        options=[5, 10, 20, 30],
        index=1,
        format_func=lambda x: f"+{x} days",
        help="Shorter = more recent screens visible (need N future days per screen)."
    )
    hc2.caption(f"Win rate and average +{horizon}d return per signal tier, sampled weekly across history. "
                f"Most recent screens shown end ~{horizon} trading days before today.")

    with st.spinner(f"Computing screener accuracy history (+{horizon}d, cached)..."):
        trend = get_screener_accuracy_trend(df, horizon=horizon)

    if trend.empty:
        st.info(f"Not enough data to compute +{horizon}d trend (need at least {45+horizon+5} trading days).")
    else:
        # Aggregate: per date+alert → win rate & avg return
        agg = (
            trend.groupby(["date", "Alert"])
            .agg(
                win_rate=("ret_fwd", lambda x: (x > 0).mean() * 100),
                avg_ret =("ret_fwd", "mean"),
                n_picks =("ret_fwd", "count"),
            )
            .reset_index()
        )
        # Rolling 3-period smooth per tier
        smoothed = []
        for alert_tier, grp in agg.groupby("Alert"):
            grp = grp.sort_values("date").copy()
            grp["win_rate_smooth"] = grp["win_rate"].rolling(3, min_periods=1).mean()
            grp["avg_ret_smooth"]  = grp["avg_ret"].rolling(3, min_periods=1).mean()
            smoothed.append(grp)
        agg = pd.concat(smoothed).reset_index(drop=True)

        color_map = {"🔴 STRONG": "#e53935", "🟡 WATCH": "#ffc107", "⚪ RADAR": "#90a4ae"}

        tab1, tab2 = st.tabs(["Win Rate %", "Avg +30d Return %"])

        with tab1:
            fig_trend = go.Figure()
            fig_trend.add_hline(y=50, line_dash="dash", line_color="gray",
                                annotation_text="50% (coin flip)", annotation_position="bottom right")
            for alert_tier, grp in agg.groupby("Alert"):
                fig_trend.add_trace(go.Scatter(
                    x=grp["date"], y=grp["win_rate_smooth"],
                    name=alert_tier, mode="lines+markers",
                    line=dict(color=color_map.get(alert_tier, "#aaa"), width=2),
                    marker=dict(size=5),
                    hovertemplate="%{x|%Y-%m-%d}<br>Win rate: %{y:.1f}%<extra>" + alert_tier + "</extra>",
                ))
            fig_trend.update_layout(
                height=350, margin=dict(t=20, b=20),
                yaxis_title="Win Rate %", yaxis_range=[0, 100],
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with tab2:
            fig_ret = go.Figure()
            fig_ret.add_hline(y=0, line_dash="dash", line_color="gray")
            for alert_tier, grp in agg.groupby("Alert"):
                fig_ret.add_trace(go.Scatter(
                    x=grp["date"], y=grp["avg_ret_smooth"],
                    name=alert_tier, mode="lines+markers",
                    line=dict(color=color_map.get(alert_tier, "#aaa"), width=2),
                    marker=dict(size=5),
                    hovertemplate="%{x|%Y-%m-%d}<br>Avg return: %{y:.1f}%<extra>" + alert_tier + "</extra>",
                ))
            fig_ret.update_layout(
                height=350, margin=dict(t=20, b=20),
                yaxis_title=f"Avg +{horizon}d Return %",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_ret, use_container_width=True)

        # Overall summary table
        summary = (
            trend.groupby("Alert")
            .agg(
                Screens=("date", "nunique"),
                Total_Picks=("ret_fwd", "count"),
                Win_Rate=("ret_fwd", lambda x: f"{(x>0).mean()*100:.1f}%"),
                Avg_Return=("ret_fwd", lambda x: f"{x.mean():+.1f}%"),
                Median_Return=("ret_fwd", lambda x: f"{x.median():+.1f}%"),
                Best=("ret_fwd", lambda x: f"{x.max():+.1f}%"),
                Worst=("ret_fwd", lambda x: f"{x.min():+.1f}%"),
            )
            .reset_index()
            .rename(columns={"Alert":"Tier","Total_Picks":"Total Picks",
                             "Win_Rate":"Win Rate","Avg_Return":f"Avg +{horizon}d",
                             "Median_Return":f"Median +{horizon}d"})
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

    # ── TP/SL Backtest Simulator ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🧪 TP / SL Backtest Simulator")
    st.caption("If I buy these tier(s) and set Take-Profit / Stop-Loss / max hold days — "
               "what's my historical win rate? Conservative: same-day TP+SL = SL hit first.")

    bc1, bc2, bc3, bc4 = st.columns(4)
    bt_tiers = bc1.multiselect(
        "Tier(s)",
        options=["🔴 STRONG", "🟡 WATCH", "⚪ RADAR"],
        default=["🔴 STRONG"],
    )
    with bc2:
        tp_on = st.checkbox("Use TP", value=True, key="bt_tp_on")
        tp_pct_val = st.number_input("Take Profit %", min_value=1.0, max_value=200.0,
                                     value=10.0, step=0.5, disabled=not tp_on)
        tp_pct = float(tp_pct_val) if tp_on else None
    with bc3:
        sl_on = st.checkbox("Use SL", value=True, key="bt_sl_on")
        sl_pct_val = st.number_input("Stop Loss %", min_value=1.0, max_value=50.0,
                                     value=5.0, step=0.5, disabled=not sl_on)
        sl_pct = float(sl_pct_val) if sl_on else None
    with bc4:
        hold_on = st.checkbox("Use max hold", value=True, key="bt_hold_on")
        max_hold_val = st.number_input("Max hold days", min_value=1, max_value=365,
                                       value=30, step=1, disabled=not hold_on)
        max_hold = int(max_hold_val) if hold_on else None
    if tp_pct is None and sl_pct is None and max_hold is None:
        st.caption("⚠️ All three filters disabled — trades will hold to end of data (buy-and-hold).")

    bm1, bm2 = st.columns(2)
    money_per_trade = bm1.number_input(
        "💰 Money per trade (IDR)", min_value=100_000, max_value=10_000_000_000,
        value=10_000_000, step=1_000_000, format="%d",
        help="Fixed rupiah amount invested in each trade. P&L per trade = this × return%."
    )
    starting_capital = bm2.number_input(
        "🏦 Starting capital (IDR)", min_value=1_000_000, max_value=100_000_000_000,
        value=100_000_000, step=10_000_000, format="%d",
        help="For account-balance curve: starting capital. P&L is added on top.",
    )

    run_bt = st.button("▶ Run backtest", type="primary")

    if run_bt or st.session_state.get("bt_results") is not None:
        if run_bt:
            if not bt_tiers:
                st.warning("Select at least one tier.")
                st.stop()
            with st.spinner("Simulating trades across history..."):
                trades = backtest_tp_sl(df, tp_pct, sl_pct, max_hold, tuple(bt_tiers))
                st.session_state["bt_results"] = trades
                st.session_state["bt_params"]  = (tuple(bt_tiers), tp_pct, sl_pct, max_hold)
        trades = st.session_state.get("bt_results", pd.DataFrame())

        if trades.empty:
            st.warning("No trades generated.")
        else:
            params = st.session_state.get("bt_params")
            tiers_used, tp_used, sl_used, mh_used = params
            tp_str = f"TP +{tp_used:.1f}%" if tp_used is not None else "TP off"
            sl_str = f"SL -{sl_used:.1f}%" if sl_used is not None else "SL off"
            mh_str = f"Max hold {mh_used}d" if mh_used is not None else "No hold cap"
            st.caption(f"**Strategy:** Buy {' + '.join(tiers_used)} · {tp_str} · {sl_str} · {mh_str}")

            # Ensure Status column exists for older cached results
            trades = trades.copy()
            if "Status" not in trades.columns:
                trades["Status"] = trades["Outcome"].map(
                    lambda o: "Floating" if o == "OPEN" else "Realized"
                )

            # Top-line metrics
            n = len(trades)
            wins = (trades["Return %"] > 0).sum()
            tps  = (trades["Outcome"] == "TP").sum()
            sls  = (trades["Outcome"] == "SL").sum()
            tims = (trades["Outcome"] == "TIME").sum()
            opens= (trades["Outcome"] == "OPEN").sum()
            avg_ret = trades["Return %"].mean()
            avg_hold = trades["Hold (d)"].mean()
            expectancy = avg_ret

            # IDR P&L per trade (fixed-size sizing)
            trades["P&L (IDR)"] = trades["Return %"] / 100.0 * money_per_trade

            realized = trades[trades["Status"] == "Realized"]
            floating = trades[trades["Status"] == "Floating"]
            n_real, n_float = len(realized), len(floating)

            realized_pnl = realized["P&L (IDR)"].sum()
            floating_pnl = floating["P&L (IDR)"].sum()
            total_pnl    = realized_pnl + floating_pnl
            avg_pnl      = trades["P&L (IDR)"].mean()
            best_pnl     = trades["P&L (IDR)"].max()
            worst_pnl    = trades["P&L (IDR)"].min()
            realized_balance = starting_capital + realized_pnl
            total_balance    = starting_capital + total_pnl
            realized_growth  = (realized_balance / starting_capital - 1) * 100
            total_growth     = (total_balance / starting_capital - 1) * 100

            mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
            mc1.metric("Trades",     f"{n:,}",
                       delta=f"{n_real} closed · {n_float} open",
                       delta_color="off")
            mc2.metric("Win Rate",   f"{wins/n*100:.1f}%")
            mc3.metric("TP/SL/TIME/OPEN", f"{tps}/{sls}/{tims}/{opens}")
            mc4.metric("Avg Return", f"{avg_ret:+.2f}%")
            mc5.metric("Avg Hold",   f"{avg_hold:.1f}d")
            mc6.metric("Expectancy / trade", f"{expectancy:+.2f}%",
                       delta="Edge" if expectancy > 0 else "No edge",
                       delta_color="normal" if expectancy > 0 else "inverse")

            # ── 💰 Money summary (the headline result) ─────────────────────
            st.markdown("### 💰 Account Performance — Realized vs Floating")
            st.caption(
                f"**Realized** = trades that closed (TP, SL, or full hold elapsed). "
                f"**Floating** = positions still open because the simulation ran out "
                f"of data before max-hold; marked-to-market at last close."
            )
            mm1, mm2, mm3, mm4, mm5 = st.columns(5)
            mm1.metric(
                "Starting Capital",
                f"Rp {starting_capital/1e6:.1f}M",
                help=f"Money per trade: Rp {money_per_trade/1e6:.1f}M",
            )
            mm2.metric(
                "Realized P&L",
                f"Rp {realized_pnl/1e6:+.1f}M",
                delta=f"{n_real} closed trades",
                delta_color="normal" if realized_pnl >= 0 else "inverse",
                help="Locked-in profit/loss from trades that hit TP, SL, or max hold.",
            )
            mm3.metric(
                "Floating P&L",
                f"Rp {floating_pnl/1e6:+.1f}M",
                delta=f"{n_float} open trades" if n_float else "no open trades",
                delta_color="normal" if floating_pnl >= 0 else "inverse",
                help="Mark-to-market on positions still open at end of data.",
            )
            mm4.metric(
                "Total P&L (Real + Float)",
                f"Rp {total_pnl/1e6:+.1f}M",
                delta=f"{total_growth:+.1f}% on capital",
                delta_color="normal" if total_pnl >= 0 else "inverse",
            )
            mm5.metric(
                "Final Balance",
                f"Rp {total_balance/1e6:.1f}M",
                delta=f"Realized only: Rp {realized_balance/1e6:.1f}M ({realized_growth:+.1f}%)",
                delta_color="off",
            )

            # Mini per-trade extremes (uses all trades incl. floating)
            st.caption(
                f"📊 Per-trade extremes — Best: Rp {best_pnl/1e6:+.2f}M  ·  "
                f"Worst: Rp {worst_pnl/1e6:+.2f}M  ·  "
                f"Avg: Rp {avg_pnl/1e6:+.2f}M"
            )

            # Outcome breakdown
            ob1, ob2 = st.columns([1.2, 1])
            with ob1:
                outcome_df = pd.DataFrame({
                    "Outcome": ["TP hit (win)", "SL hit (loss)",
                                "TIME exit (hold elapsed)", "OPEN (floating)"],
                    "Count":   [tps, sls, tims, opens],
                })
                outcome_df = outcome_df[outcome_df["Count"] > 0]
                fig_o = px.pie(outcome_df, values="Count", names="Outcome",
                               color="Outcome",
                               color_discrete_map={
                                   "TP hit (win)": "#2e7d32",
                                   "SL hit (loss)": "#c62828",
                                   "TIME exit (hold elapsed)": "#9e9e9e",
                                   "OPEN (floating)": "#1976d2",
                               },
                               hole=0.4, title="Trade Outcomes")
                fig_o.update_layout(height=320, margin=dict(t=40, b=10))
                st.plotly_chart(fig_o, use_container_width=True)

            with ob2:
                st.markdown("**Risk / Reward Math**")
                if tp_used is not None and sl_used is not None:
                    st.markdown(f"- TP: **+{tp_used:.1f}%** · SL: **-{sl_used:.1f}%** · "
                                f"R:R = **{tp_used/sl_used:.2f}**")
                    bep = sl_used / (tp_used + sl_used) * 100
                    st.markdown(f"- Breakeven win rate (TP-vs-SL): **{bep:.1f}%**")
                    actual_wr = tps / max(tps+sls, 1) * 100
                    if tps + sls > 0:
                        edge = actual_wr - bep
                        st.markdown(f"- Actual TP-vs-SL win rate: **{actual_wr:.1f}%** "
                                    f"({'+' if edge >= 0 else ''}{edge:.1f}% vs breakeven)")
                else:
                    parts = []
                    if tp_used is not None: parts.append(f"TP **+{tp_used:.1f}%**")
                    else: parts.append("TP **off**")
                    if sl_used is not None: parts.append(f"SL **-{sl_used:.1f}%**")
                    else: parts.append("SL **off**")
                    st.markdown("- " + " · ".join(parts))
                    st.markdown(f"- Win rate (any positive return): **{wins/n*100:.1f}%**")
                if expectancy > 0:
                    st.success(f"✅ Positive expectancy in this simulation: **{expectancy:+.2f}%/trade**. "
                               f"Validate next-open execution, costs, and a later holdout before relying on it.")
                else:
                    st.error(f"❌ Negative expectancy: **{expectancy:+.2f}%/trade** — "
                             f"strategy loses money on average.")

            # Return distribution
            fig_d = px.histogram(trades, x="Return %", nbins=40,
                                 color="Outcome",
                                 color_discrete_map={
                                     "TP": "#2e7d32", "SL": "#c62828",
                                     "TIME": "#9e9e9e", "OPEN": "#1976d2",
                                 },
                                 title="Per-trade return distribution (incl. floating)")
            fig_d.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_d.add_vline(x=avg_ret, line_dash="dot", line_color="#1565c0",
                            annotation_text=f"Avg {avg_ret:+.1f}%",
                            annotation_position="top")
            fig_d.update_layout(height=280, margin=dict(t=40, b=10))
            st.plotly_chart(fig_d, use_container_width=True)

            # IDR Account balance curve — Realized line + Floating overlay
            trades_sorted = trades.sort_values("date").reset_index(drop=True)
            real_only = trades_sorted[trades_sorted["Status"] == "Realized"].copy()
            real_only["cum_pnl_idr"]   = real_only["P&L (IDR)"].cumsum()
            real_only["balance_idr"]   = starting_capital + real_only["cum_pnl_idr"]

            # "Total balance incl floating" = realized cumulative + sum of floating
            # P&L for trades opened on/before each date.
            float_only = trades_sorted[trades_sorted["Status"] == "Floating"].copy()
            float_only = float_only.sort_values("date")
            float_only["cum_float"] = float_only["P&L (IDR)"].cumsum()

            # Combined "total balance" series: at each unique date, balance
            # = starting + realized cumulative up to that date
            #              + cumulative floating opened up to that date.
            all_dates_curve = sorted(trades_sorted["date"].unique())
            real_lookup  = real_only.set_index("date")["cum_pnl_idr"]
            float_lookup = float_only.set_index("date")["cum_float"]
            cum_real, cum_float = 0.0, 0.0
            rows = []
            for d in all_dates_curve:
                if d in real_lookup.index:  cum_real  = real_lookup.loc[d] if not isinstance(real_lookup.loc[d], pd.Series) else real_lookup.loc[d].iloc[-1]
                if d in float_lookup.index: cum_float = float_lookup.loc[d] if not isinstance(float_lookup.loc[d], pd.Series) else float_lookup.loc[d].iloc[-1]
                rows.append({"date": d,
                             "realized_balance": starting_capital + cum_real,
                             "total_balance":    starting_capital + cum_real + cum_float})
            curve = pd.DataFrame(rows)

            running_dd = (curve["total_balance"] /
                          curve["total_balance"].cummax() - 1) * 100
            max_dd_pct = running_dd.min() if len(running_dd) else 0
            final_real  = curve["realized_balance"].iloc[-1]
            final_total = curve["total_balance"].iloc[-1]

            fig_e = go.Figure()
            fig_e.add_trace(go.Scatter(
                x=curve["date"], y=curve["total_balance"]/1e6,
                mode="lines", line=dict(color="#1976d2", width=2),
                name="Total balance (Realized + Floating)",
                hovertemplate="%{x|%Y-%m-%d}<br>Total: Rp %{y:.2f}M<extra></extra>",
            ))
            fig_e.add_trace(go.Scatter(
                x=curve["date"], y=curve["realized_balance"]/1e6,
                mode="lines", line=dict(color="#2e7d32", width=2, dash="dot"),
                name="Realized only",
                hovertemplate="%{x|%Y-%m-%d}<br>Realized: Rp %{y:.2f}M<extra></extra>",
            ))
            fig_e.add_hline(y=starting_capital/1e6, line_dash="dash", line_color="gray",
                            annotation_text=f"Start Rp {starting_capital/1e6:.0f}M",
                            annotation_position="bottom right")
            fig_e.update_layout(
                title=(f"💰 Account Balance Over Time (Rp {money_per_trade/1e6:.1f}M per trade) — "
                       f"Total: Rp {final_total/1e6:.1f}M  ·  "
                       f"Realized: Rp {final_real/1e6:.1f}M  ·  "
                       f"Max DD: {max_dd_pct:.1f}%"),
                height=340, margin=dict(t=50, b=10),
                yaxis_title="Balance (Rp Millions)",
                xaxis_title="Trade entry date",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_e, use_container_width=True)

            # Per-tier breakdown
            st.markdown("**By tier:**")
            tier_summary = (
                trades.groupby("Alert")
                .agg(
                    Trades=("Return %", "count"),
                    Win_Rate=("Return %", lambda x: f"{(x>0).mean()*100:.1f}%"),
                    Avg_Return=("Return %", lambda x: f"{x.mean():+.2f}%"),
                    Open_Cnt=("Status", lambda x: int((x=="Floating").sum())),
                    Realized_PnL=("P&L (IDR)",
                        lambda x: f"Rp {x[trades.loc[x.index,'Status']=='Realized'].sum()/1e6:+.1f}M"),
                    Floating_PnL=("P&L (IDR)",
                        lambda x: f"Rp {x[trades.loc[x.index,'Status']=='Floating'].sum()/1e6:+.1f}M"),
                    Total_PnL=("P&L (IDR)", lambda x: f"Rp {x.sum()/1e6:+.1f}M"),
                    TP_Rate=("Outcome", lambda x: f"{(x=='TP').mean()*100:.0f}%"),
                    SL_Rate=("Outcome", lambda x: f"{(x=='SL').mean()*100:.0f}%"),
                    Avg_Hold=("Hold (d)", lambda x: f"{x.mean():.1f}d"),
                )
                .reset_index()
                .rename(columns={"Alert":"Tier","Win_Rate":"Win Rate",
                                 "Avg_Return":"Avg Return",
                                 "Open_Cnt":"Open",
                                 "Realized_PnL":"Realized P&L",
                                 "Floating_PnL":"Floating P&L",
                                 "Total_PnL":"Total P&L",
                                 "TP_Rate":"TP %",
                                 "SL_Rate":"SL %","Avg_Hold":"Avg Hold"})
            )
            st.dataframe(tier_summary, use_container_width=True, hide_index=True)

            # Floating positions detail (live mark-to-market)
            if n_float > 0:
                with st.expander(f"🔵 {n_float} floating (still-open) positions — "
                                 f"live mark-to-market: Rp {floating_pnl/1e6:+.1f}M"):
                    f_disp = floating.sort_values("date", ascending=False).copy()
                    f_disp["date"] = f_disp["date"].dt.strftime("%Y-%m-%d")
                    f_disp["Entry"] = f_disp["Entry"].astype(int)
                    f_disp["Exit"]  = f_disp["Exit"].round(0).astype(int).rename("MTM Price")
                    f_disp["P&L (IDR)"] = f_disp["P&L (IDR)"].apply(
                        lambda v: f"Rp {v/1e6:+.2f}M"
                    )
                    f_disp = f_disp.rename(columns={"Exit":"MTM Price",
                                                    "Hold (d)":"Days held"})
                    st.dataframe(f_disp[["date","Ticker","Alert","Score","Entry",
                                         "MTM Price","Return %","Days held","P&L (IDR)"]],
                                 use_container_width=True, hide_index=True)

            # All trades expander
            with st.expander(f"📋 All {len(trades)} simulated trades "
                             f"({n_real} realized + {n_float} floating)"):
                trades_disp = trades.sort_values("date", ascending=False).copy()
                trades_disp["date"] = trades_disp["date"].dt.strftime("%Y-%m-%d")
                trades_disp["Entry"] = trades_disp["Entry"].astype(int)
                trades_disp["Exit"]  = trades_disp["Exit"].round(0).astype(int)
                trades_disp["P&L (IDR)"] = trades_disp["P&L (IDR)"].apply(
                    lambda v: f"Rp {v/1e6:+.2f}M"
                )
                st.dataframe(trades_disp, use_container_width=True, hide_index=True)
    else:
        st.info("👆 Set TP/SL/hold-days and tier(s), then click **Run backtest**.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Stock Deep Dive
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Stock Deep Dive":
    st.title("📈 Stock Deep Dive")

    all_tickers = sorted(df["Kode Saham"].unique())
    ticker = st.selectbox("Select ticker", all_tickers, index=all_tickers.index("BBCA") if "BBCA" in all_tickers else 0)

    g = df[df["Kode Saham"]==ticker].sort_values("DATE").reset_index(drop=True)
    if g.empty:
        st.warning("No data found.")
        st.stop()

    name = g.iloc[-1]["Nama Perusahaan"]
    latest = g.iloc[-1]

    # Date range filter
    min_d = g["DATE"].min().date()
    max_d = g["DATE"].max().date()
    d_from, d_to = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    g = g[(g["DATE"].dt.date >= d_from) & (g["DATE"].dt.date <= d_to)]

    # KPIs
    prev_close  = g.iloc[-2]["Penutupan"] if len(g)>1 else g.iloc[-1]["Sebelumnya"]
    close       = g.iloc[-1]["Penutupan"]
    chg_pct     = (close - prev_close) / prev_close * 100 if prev_close else 0
    hi52        = g["Tertinggi"].max()
    lo52        = g["Terendah"].min()
    avg_vol_20  = g.tail(20)["Volume"].mean()
    net_fgn_30  = g.tail(30)["Net Foreign Est Value"].sum()

    st.subheader(f"{ticker} — {name}")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Close",        f"{int(close):,}",    delta=f"{chg_pct:.2f}%")
    c2.metric("Period High",  f"{int(hi52):,}")
    c3.metric("Period Low",   f"{int(lo52):,}")
    c4.metric("Avg Vol (20d)",f"{avg_vol_20/1e6:.1f}M")
    c5.metric("Est. Net Fgn Value (30d)",f"Rp {net_fgn_30/1e9:.1f}B",
              delta="NET BUY" if net_fgn_30>0 else "NET SELL",
              delta_color="normal" if net_fgn_30>0 else "inverse")

    # ── Current v3 Verdict ────────────────────────────────────────────────────
    g_full = df[df["Kode Saham"] == ticker].sort_values("DATE").reset_index(drop=True)
    verdict = current_v3_verdict(g_full)
    levels = compute_action_levels(g_full, account_size, risk_pct)

    vc1, vc2 = st.columns([1, 1.5])
    with vc1:
        st.subheader("🎯 v3 Verdict")
        if verdict is None:
            st.info("Insufficient history for v3 analysis.")
        elif verdict["passes_v3"]:
            st.success("✅ **PASSES v3 gates** — qualifies as a screener candidate today.")
            st.caption(
                f"BrkProx {verdict['break_prox']*100:.1f}% · "
                f"PosRange {verdict['pos_in_range']:.0f}% · "
                f"VolRatio {verdict['vol_ratio']:.2f}x · "
                f"Pre30Chg {verdict['pre30_chg']:+.1f}% · "
                f"Fgn+Days {verdict['fgn_pos_days']:.0f}%"
            )
        else:
            st.error("❌ **Does NOT pass v3 gates**")
            for f in verdict["fails"]:
                st.markdown(f"- ⚠️ {f}")

    with vc2:
        st.subheader("🛒 Suggested Action Levels")
        if levels is None:
            st.info("Need ≥20 days of history to compute action levels.")
        else:
            ac1, ac2, ac3, ac4 = st.columns(4)
            ac1.metric("Entry",  f"{int(levels['entry']):,}")
            ac2.metric("Stop",   f"{int(levels['stop']):,}", delta=f"-{levels['risk_pct']:.1f}%",
                       delta_color="inverse")
            ac3.metric("Target", f"{int(levels['target_1']):,}", delta=f"+{levels['reward_pct']:.1f}%")
            ac4.metric("R:R",    f"{levels['rr_ratio']:.1f}",
                       delta="Quality" if levels['rr_ratio'] >= 2 else "Marginal",
                       delta_color="normal" if levels['rr_ratio'] >= 2 else "inverse")
            st.caption(
                f"Sized for **Rp {account_size/1e6:.0f}M @ {risk_pct}% risk** → "
                f"buy **{levels['suggested_lots']:,} lots** "
                f"(Rp {levels['capital_required']/1e6:.1f}M capital, "
                f"Rp {levels['actual_risk_idr']/1e6:.2f}M at risk)"
            )

    # Load screener flags for this ticker (full history, then filter to display range)
    with st.spinner(f"Loading screener history for {ticker}..."):
        flags_df = get_ticker_flags(df, ticker)

    # Candlestick + screener flags + MA overlays
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=g["DATE"], open=g["Open Price"],
        high=g["Tertinggi"], low=g["Terendah"], close=g["Penutupan"],
        name="Price", increasing_line_color="#00c853", decreasing_line_color="#ff1744",
    ))

    # Moving averages — context for entry/stop reasoning
    if len(g) >= 5:
        ma20_series = g["Penutupan"].rolling(20, min_periods=5).mean()
        ma50_series = g["Penutupan"].rolling(50, min_periods=10).mean()
        ma200_series = g["Penutupan"].rolling(200, min_periods=30).mean()
        fig.add_trace(go.Scatter(x=g["DATE"], y=ma20_series, name="MA20",
                                 line=dict(color="#42a5f5", width=1.2), opacity=0.85))
        fig.add_trace(go.Scatter(x=g["DATE"], y=ma50_series, name="MA50",
                                 line=dict(color="#ab47bc", width=1.2), opacity=0.85))
        fig.add_trace(go.Scatter(x=g["DATE"], y=ma200_series, name="MA200",
                                 line=dict(color="#ff9800", width=1.5, dash="dash"), opacity=0.85))

    # Suggested action levels (entry / stop / target) as horizontal lines
    if levels:
        last_dt = g["DATE"].max()
        fig.add_hline(y=levels["entry"], line_dash="dot", line_color="#1976d2",
                      annotation_text=f"Entry {int(levels['entry']):,}",
                      annotation_position="right")
        fig.add_hline(y=levels["stop"], line_dash="dot", line_color="#d32f2f",
                      annotation_text=f"Stop {int(levels['stop']):,}",
                      annotation_position="right")
        fig.add_hline(y=levels["target_1"], line_dash="dot", line_color="#388e3c",
                      annotation_text=f"Target {int(levels['target_1']):,}",
                      annotation_position="right")

    if not flags_df.empty:
        # Filter flags to the displayed date range
        flags_in_range = flags_df[
            (flags_df["DATE"] >= pd.Timestamp(d_from)) &
            (flags_df["DATE"] <= pd.Timestamp(d_to))
        ]
        flag_styles = {
            "🔴 STRONG": ("triangle-up",  "#e53935", 14),
            "🟡 WATCH":  ("diamond",      "#ffc107", 12),
            "⚪ RADAR":  ("circle",       "#90a4ae", 9),
        }
        for alert_tier, (symbol, color, size) in flag_styles.items():
            subset = flags_in_range[flags_in_range["Alert"] == alert_tier]
            if subset.empty:
                continue
            # Place marker slightly above the high for that candle
            marker_y = []
            for dt in subset["DATE"]:
                candle = g[g["DATE"] == dt]
                y_pos = candle["Tertinggi"].values[0] * 1.01 if not candle.empty else subset.loc[subset["DATE"]==dt, "Close"].values[0]
                marker_y.append(y_pos)
            fig.add_trace(go.Scatter(
                x=subset["DATE"], y=marker_y,
                mode="markers",
                marker=dict(symbol=symbol, color=color, size=size,
                            line=dict(color="white", width=1)),
                name=alert_tier,
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>" + alert_tier +
                    "<br>Vol Ratio: " + subset["Vol Ratio"].astype(str).values[0]
                    if len(subset) == 1 else
                    "%{x|%Y-%m-%d}<br>" + alert_tier
                ) + "<extra></extra>",
            ))

    total_flags = len(flags_df) if not flags_df.empty else 0
    fig.update_layout(
        title=f"{ticker} Price  •  Screener flags: {total_flags} total in history "
              f"({'▲ STRONG  ◆ WATCH  ● RADAR' if total_flags > 0 else 'none'})",
        height=450,
        xaxis_rangeslider_visible=False,
        margin=dict(t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Volume bars + 20-day avg line
    colors = ["#00c853" if p>=0 else "#ff1744" for p in g["pct_chg"]]
    vol_m = g["Volume"] / 1e6
    avg20 = vol_m.rolling(20, min_periods=1).mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=g["DATE"], y=vol_m, marker_color=colors, name="Volume (M)", opacity=0.8))
    fig2.add_trace(go.Scatter(x=g["DATE"], y=avg20, mode="lines",
                              line=dict(color="#ff9800", width=1.5, dash="dot"),
                              name="20d Avg Vol"))
    fig2.update_layout(title="Volume (M shares)", height=200, margin=dict(t=40,b=10),
                       yaxis_title="M shares", legend=dict(orientation="h"))
    st.plotly_chart(fig2, use_container_width=True)

    # Foreign flow. The source fields are shares; value is estimated at close.
    g["nf_est_b"] = g["Net Foreign Est Value"] / 1e9
    colors_fgn = ["#1565c0" if v > 0 else "#b71c1c" for v in g["nf_est_b"]]
    fig3 = go.Figure(go.Bar(
        x=g["DATE"], y=g["nf_est_b"], marker_color=colors_fgn,
        name="Est. Net Foreign Value (B IDR)",
    ))
    fig3.update_layout(
        title="Estimated Net Foreign Value (net shares × close)",
        height=200, margin=dict(t=40,b=10), yaxis_title="B IDR (estimated)",
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Raw data table
    with st.expander("Raw data"):
        show = g[["DATE","Sebelumnya","Open Price","Tertinggi","Terendah","Penutupan",
                   "pct_chg","Volume","Nilai","Frekuensi","Net Foreign",
                   "Net Foreign Est Value"]].copy()
        show["DATE"] = show["DATE"].dt.strftime("%Y-%m-%d")
        show["Nilai (B)"] = (show["Nilai"]/1e9).round(1)
        show["Net Fgn (M sh)"] = (show["Net Foreign"]/1e6).round(2)
        show["Net Fgn Est (B)"] = (show["Net Foreign Est Value"]/1e9).round(2)
        st.dataframe(show.drop(columns=["Nilai","Net Foreign", "Net Foreign Est Value"]).sort_values("DATE",ascending=False),
                     use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Portfolio Tracker
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💼 Portfolio":
    st.title("💼 Portfolio Tracker")

    st.info("Add your positions below. Entry price and shares are saved in your session.")

    # Session state for portfolio
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = [
            {"ticker": "BRPT", "shares": 0, "entry": 0},
            {"ticker": "IDEA", "shares": 0, "entry": 0},
            {"ticker": "INET", "shares": 0, "entry": 0},
        ]

    # Edit positions
    st.subheader("My Positions")
    all_tickers = sorted(df["Kode Saham"].unique())

    updated = []
    for i, pos in enumerate(st.session_state.portfolio):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        ticker  = c1.selectbox("Ticker",   all_tickers, key=f"t{i}",
                               index=all_tickers.index(pos["ticker"]) if pos["ticker"] in all_tickers else 0)
        shares  = c2.number_input("Lots (1 lot = 100 shares)", value=int(pos["shares"]), min_value=0, key=f"s{i}")
        entry   = c3.number_input("Avg Entry Price",      value=float(pos["entry"]), min_value=0.0, key=f"e{i}")
        remove  = c4.button("✕", key=f"r{i}")
        if not remove:
            updated.append({"ticker": ticker, "shares": shares, "entry": entry})

    if st.button("＋ Add Position"):
        updated.append({"ticker": all_tickers[0], "shares": 0, "entry": 0})

    st.session_state.portfolio = updated

    if not updated:
        st.stop()

    # Build P&L table
    st.markdown("---")
    st.subheader("P&L Summary")

    latest_prices = df[df["DATE"]==latest_date].set_index("Kode Saham")["Penutupan"].to_dict()

    pnl_rows = []
    for pos in updated:
        t = pos["ticker"]
        if t not in latest_prices or pos["shares"] == 0: continue
        cp    = latest_prices.get(t, 0)
        entry = pos["entry"]
        lots  = pos["shares"]
        mkt_val   = cp * lots * 100
        cost      = entry * lots * 100
        pnl_idr   = mkt_val - cost
        pnl_pct   = (cp - entry) / entry * 100 if entry > 0 else 0
        g_stock   = df[df["Kode Saham"]==t].sort_values("DATE").reset_index(drop=True)
        hi = g_stock["Tertinggi"].max()
        dd = (cp - hi) / hi * 100 if hi > 0 else 0

        # Stop loss & exit signals
        sigs = detect_exit_signals(g_stock, entry)
        action_levels = compute_action_levels(g_stock, account_size, risk_pct)
        suggested_stop = action_levels["stop"] if action_levels else 0
        dist_to_stop = (cp - suggested_stop) / cp * 100 if suggested_stop > 0 else 0

        # Health summary
        if any("🔴" in s[0] for s in sigs):
            health = "🔴 EXIT"
        elif any("🟠" in s[0] for s in sigs):
            health = "🟠 WATCH"
        else:
            health = "🟢 HOLD"

        pnl_rows.append({
            "Ticker": t,
            "Health": health,
            "Lots": lots,
            "Entry": int(entry),
            "Current": int(cp),
            "Chg%": round(pnl_pct, 1),
            "P&L (M)": round(pnl_idr/1e6, 2),
            "Mkt Val (M)": round(mkt_val/1e6, 1),
            "Sugg. Stop": int(suggested_stop) if suggested_stop else 0,
            "Dist Stop%": round(dist_to_stop, 1),
            "DD ATH%": round(dd, 1),
            "Alerts": " · ".join([f"{s[0]}" for s in sigs]) if sigs else "✅ Healthy",
        })

    if not pnl_rows:
        st.info("Enter your positions above to see P&L.")
        st.stop()

    pnl_df = pd.DataFrame(pnl_rows)
    total_cost   = sum(p["entry"]*p["shares"]*100 for p in updated if p["shares"]>0 and p["ticker"] in latest_prices)
    total_mktval = sum(latest_prices.get(p["ticker"],0)*p["shares"]*100 for p in updated if p["shares"]>0)
    total_pnl    = total_mktval - total_cost
    total_pct    = total_pnl/total_cost*100 if total_cost>0 else 0

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Total Cost",      f"Rp {total_cost/1e6:.1f}M")
    mc2.metric("Market Value",    f"Rp {total_mktval/1e6:.1f}M")
    mc3.metric("Total P&L",       f"Rp {total_pnl/1e6:.1f}M",
               delta=f"{total_pct:.1f}%",
               delta_color="normal" if total_pnl>=0 else "inverse")

    def color_pnl(val):
        if isinstance(val, (int, float)):
            color = "#00c853" if val > 0 else "#ff1744" if val < 0 else "inherit"
            return f"color: {color}; font-weight: bold"
        return ""

    def color_health(val):
        s = str(val)
        if "🔴 EXIT" in s: return "background-color: #c62828; color: white; font-weight: bold"
        if "🟠 WATCH" in s: return "background-color: #f9a825; color: black; font-weight: bold"
        if "🟢 HOLD" in s: return "background-color: #2e7d32; color: white; font-weight: bold"
        return ""

    def color_dist_stop(val):
        if not isinstance(val, (int, float)): return ""
        if val < 2:  return "background-color: #c62828; color: white; font-weight: bold"  # at/below stop
        if val < 5:  return "background-color: #ff9800; color: white"
        if val < 10: return "background-color: #ffeb3b"
        return ""

    st.dataframe(
        pnl_df.style
            .map(color_pnl, subset=["Chg%","P&L (M)","DD ATH%"])
            .map(color_health, subset=["Health"])
            .map(color_dist_stop, subset=["Dist Stop%"])
            .format({"Chg%":"{:.1f}%", "DD ATH%":"{:.1f}%",
                     "Dist Stop%":"{:.1f}%",
                     "P&L (M)":"{:+.2f}", "Mkt Val (M)":"{:.1f}",
                     "Entry":"{:,}", "Current":"{:,}", "Sugg. Stop":"{:,}"}),
        use_container_width=True, hide_index=True,
    )

    # ── Exit Signals Detail ───────────────────────────────────────────────────
    flagged = [r for r in pnl_rows if r["Alerts"] != "✅ Healthy"]
    if flagged:
        st.markdown("---")
        st.subheader("⚠️ Detailed Exit Signals")
        for row in flagged:
            t = row["Ticker"]
            g_t = df[df["Kode Saham"] == t].sort_values("DATE").reset_index(drop=True)
            entry_p = next((p["entry"] for p in updated if p["ticker"] == t), 0)
            sigs = detect_exit_signals(g_t, entry_p)
            if not sigs: continue
            with st.expander(f"{row['Health']}  **{t}** — {row['Chg%']:+.1f}% (P&L: Rp {row['P&L (M)']:+.2f}M)"):
                for sev, msg in sigs:
                    st.markdown(f"- {sev}  {msg}")

    # ── Portfolio Concentration ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Portfolio Concentration")
    if total_mktval > 0:
        conc_df = pnl_df[["Ticker", "Mkt Val (M)"]].copy()
        conc_df["Weight %"] = conc_df["Mkt Val (M)"] / (total_mktval/1e6) * 100
        conc_df = conc_df.sort_values("Weight %", ascending=False).reset_index(drop=True)
        cc1, cc2 = st.columns([1.2, 1])
        with cc1:
            fig_conc = px.pie(conc_df, values="Weight %", names="Ticker",
                              title="Position Weights", hole=0.4)
            fig_conc.update_layout(height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig_conc, use_container_width=True)
        with cc2:
            top3 = conc_df.head(3)["Weight %"].sum()
            largest = conc_df.iloc[0]
            st.metric("Top-3 Concentration", f"{top3:.0f}%",
                      delta="High risk" if top3 > 70 else "Diversified",
                      delta_color="inverse" if top3 > 70 else "normal")
            st.metric("Largest Position", f"{largest['Ticker']} {largest['Weight %']:.0f}%")
            st.metric("# Positions", len(conc_df))
            if top3 > 70:
                st.warning("⚠️ Heavy concentration. Consider diversifying.")

    # Price chart for all positions
    st.markdown("---")
    st.subheader("Price History — All Positions")
    fig = go.Figure()
    for pos in updated:
        t = pos["ticker"]
        g_p = df[df["Kode Saham"]==t].sort_values("DATE")
        if g_p.empty: continue
        fig.add_trace(go.Scatter(
            x=g_p["DATE"], y=g_p["Penutupan"],
            name=t, mode="lines", line=dict(width=2),
        ))
        if pos["entry"] > 0:
            fig.add_hline(
                y=pos["entry"], line_dash="dot",
                annotation_text=f"{t} entry", line_color="gray",
            )
    fig.update_layout(
        title="Portfolio Positions (with entry price lines)",
        height=450, margin=dict(t=50,b=10),
        legend=dict(orientation="h"),
        yaxis_title="Price (IDR)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Individual stock details
    st.markdown("---")
    st.subheader("Individual Stock Detail")
    sel = st.selectbox("Select stock", [p["ticker"] for p in updated])
    g2 = df[df["Kode Saham"]==sel].sort_values("DATE").tail(60)
    if not g2.empty:
        fig4 = go.Figure()
        fig4.add_trace(go.Candlestick(
            x=g2["DATE"], open=g2["Open Price"], high=g2["Tertinggi"],
            low=g2["Terendah"], close=g2["Penutupan"], name="Price",
            increasing_line_color="#00c853", decreasing_line_color="#ff1744",
        ))
        ep = next((p["entry"] for p in updated if p["ticker"]==sel), 0)
        if ep > 0:
            fig4.add_hline(y=ep, line_dash="dash", line_color="#ff9800",
                           annotation_text=f"Entry: {int(ep):,}")
        fig4.update_layout(height=400, xaxis_rangeslider_visible=False, margin=dict(t=20,b=10))
        st.plotly_chart(fig4, use_container_width=True)
