"""
analyze.py — IDX Stock Market Analysis Toolkit
================================================
This is the main cowork file. Run individual functions or use
Claude Code to ask questions about the data.

Quick start:
    python analyze.py                        # daily snapshot (today or latest)
    python analyze.py --date 2025-10-01      # snapshot for a specific date
    python analyze.py --stock BBCA           # price history for one stock
    python analyze.py --top 10               # top 10 movers today
    python analyze.py --foreign              # foreign flow summary
    python analyze.py --sector               # (todo) sector breakdown

The load_master() function returns the full DataFrame — import it
in any ad-hoc script or notebook for custom analysis.
"""

import os
import argparse
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV = os.path.join(BASE_DIR, "master.csv")

# ── Data loading ──────────────────────────────────────────────────────────────

def load_master(date_from: str = None, date_to: str = None) -> pd.DataFrame:
    """
    Load master.csv into a DataFrame.
    Optionally filter by date range (YYYY-MM-DD strings).
    Numeric columns are coerced automatically.
    """
    df = pd.read_csv(MASTER_CSV, low_memory=False)
    df["DATE"] = pd.to_datetime(df["DATE"])

    numeric_cols = [
        "Sebelumnya", "Open Price", "Tertinggi", "Terendah", "Penutupan",
        "Selisih", "Volume", "Nilai", "Frekuensi", "Index Individual",
        "Offer", "Offer Volume", "Bid", "Bid Volume",
        "Listed Shares", "Tradeble Shares", "Weight For Index",
        "Foreign Sell", "Foreign Buy",
        "Non Regular Volume", "Non Regular Value", "Non Regular Frequency",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if date_from:
        df = df[df["DATE"] >= pd.to_datetime(date_from)]
    if date_to:
        df = df[df["DATE"] <= pd.to_datetime(date_to)]

    return df


def latest_date(df: pd.DataFrame = None) -> str:
    if df is None:
        df = load_master()
    return df["DATE"].max().strftime("%Y-%m-%d")


def get_day(df: pd.DataFrame, date: str = None) -> pd.DataFrame:
    """Return all rows for a single trading date (default: latest)."""
    if date is None:
        date = latest_date(df)
    return df[df["DATE"] == pd.to_datetime(date)].copy()


# ── Daily snapshot ────────────────────────────────────────────────────────────

def daily_snapshot(date: str = None, top_n: int = 10):
    """Print a market summary for a given date."""
    df = load_master()
    day = get_day(df, date)
    date_str = day["DATE"].iloc[0].strftime("%Y-%m-%d") if len(day) else date or latest_date(df)

    active = day[day["Volume"] > 0]

    print(f"\n{'='*60}")
    print(f"  IDX Market Snapshot — {date_str}")
    print(f"{'='*60}")
    print(f"  Stocks listed  : {len(day):,}")
    print(f"  Stocks traded  : {len(active):,}")

    if len(active) == 0:
        print("  No trading data available for this date.")
        return

    gainers = (active["Selisih"] > 0).sum()
    losers  = (active["Selisih"] < 0).sum()
    flat    = (active["Selisih"] == 0).sum()
    print(f"  Gainers        : {gainers:,}")
    print(f"  Losers         : {losers:,}")
    print(f"  Unchanged      : {flat:,}")

    total_value  = active["Nilai"].sum()
    total_volume = active["Volume"].sum()
    total_freq   = active["Frekuensi"].sum()
    print(f"  Total value    : Rp {total_value/1e12:.2f}T")
    print(f"  Total volume   : {total_volume/1e9:.2f}B lots")
    print(f"  Total freq     : {total_freq/1e6:.2f}M transactions")

    net_foreign = active["Foreign Buy"].sum() - active["Foreign Sell"].sum()
    print(f"  Net foreign    : Rp {net_foreign/1e9:.2f}B  "
          f"({'NET BUY' if net_foreign > 0 else 'NET SELL'})")

    # Top gainers (% change)
    active = active.copy()
    active["pct_change"] = (active["Selisih"] / active["Sebelumnya"].replace(0, pd.NA)) * 100

    print(f"\n  Top {top_n} Gainers (%):")
    top_g = active.nlargest(top_n, "pct_change")[
        ["Kode Saham", "Nama Perusahaan", "Sebelumnya", "Penutupan", "Selisih", "pct_change", "Volume"]
    ]
    print(top_g.to_string(index=False))

    print(f"\n  Top {top_n} Losers (%):")
    top_l = active.nsmallest(top_n, "pct_change")[
        ["Kode Saham", "Nama Perusahaan", "Sebelumnya", "Penutupan", "Selisih", "pct_change", "Volume"]
    ]
    print(top_l.to_string(index=False))

    print(f"\n  Top {top_n} by Value (Rp):")
    top_v = active.nlargest(top_n, "Nilai")[
        ["Kode Saham", "Nama Perusahaan", "Penutupan", "Selisih", "Nilai", "Frekuensi"]
    ]
    top_v = top_v.copy()
    top_v["Nilai (B)"] = (top_v["Nilai"] / 1e9).round(1)
    print(top_v[["Kode Saham", "Nama Perusahaan", "Penutupan", "Selisih", "Nilai (B)", "Frekuensi"]].to_string(index=False))

    print()


# ── Stock history ─────────────────────────────────────────────────────────────

def stock_history(ticker: str, date_from: str = None, date_to: str = None):
    """Print OHLCV + foreign flow history for a single stock."""
    df = load_master(date_from, date_to)
    s = df[df["Kode Saham"].str.upper() == ticker.upper()].copy()
    s = s.sort_values("DATE")

    if s.empty:
        print(f"No data found for ticker: {ticker.upper()}")
        return

    name = s["Nama Perusahaan"].iloc[0]
    print(f"\n{'='*70}")
    print(f"  {ticker.upper()} — {name}")
    print(f"  Period: {s['DATE'].iloc[0].date()} → {s['DATE'].iloc[-1].date()}  ({len(s)} days)")
    print(f"{'='*70}")

    s["pct"] = (s["Selisih"] / s["Sebelumnya"].replace(0, pd.NA) * 100).round(2)
    s["Net Foreign (M)"] = ((s["Foreign Buy"] - s["Foreign Sell"]) / 1e6).round(1)

    cols = ["DATE", "Sebelumnya", "Open Price", "Tertinggi", "Terendah",
            "Penutupan", "Selisih", "pct", "Volume", "Nilai", "Net Foreign (M)"]
    cols = [c for c in cols if c in s.columns]
    s["DATE"] = s["DATE"].dt.strftime("%Y-%m-%d")
    print(s[cols].to_string(index=False))

    # Summary stats
    print(f"\n  52-week high : {s['Tertinggi'].max()}")
    print(f"  52-week low  : {s['Terendah'].min()}")
    last_close = s["Penutupan"].iloc[-1]
    first_close = s["Penutupan"].iloc[0]
    total_return = ((last_close - first_close) / first_close * 100) if first_close else 0
    print(f"  Return (period): {total_return:.1f}%")
    print(f"  Avg daily vol  : {s['Volume'].mean()/1e6:.1f}M lots")
    net_foreign_total = (s["Foreign Buy"] - s["Foreign Sell"]).sum()
    print(f"  Net foreign (period): Rp {net_foreign_total/1e9:.1f}B")
    print()


# ── Foreign flow ──────────────────────────────────────────────────────────────

def foreign_flow(date: str = None, top_n: int = 20):
    """Show top foreign buy/sell stocks for a date."""
    df = load_master()
    day = get_day(df, date)
    date_str = day["DATE"].iloc[0].strftime("%Y-%m-%d") if len(day) else "latest"
    active = day[day["Volume"] > 0].copy()

    if active.empty:
        print(f"No data for {date_str}")
        return

    active["Net Foreign"] = active["Foreign Buy"] - active["Foreign Sell"]
    active["Net Foreign (B)"] = (active["Net Foreign"] / 1e9).round(2)
    active["Foreign Buy (B)"]  = (active["Foreign Buy"] / 1e9).round(2)
    active["Foreign Sell (B)"] = (active["Foreign Sell"] / 1e9).round(2)

    print(f"\n{'='*60}")
    print(f"  Foreign Flow — {date_str}")
    print(f"{'='*60}")
    total_buy  = active["Foreign Buy"].sum()
    total_sell = active["Foreign Sell"].sum()
    net        = total_buy - total_sell
    print(f"  Total Foreign Buy  : Rp {total_buy/1e9:.2f}B")
    print(f"  Total Foreign Sell : Rp {total_sell/1e9:.2f}B")
    print(f"  Net               : Rp {net/1e9:.2f}B  ({'NET BUY' if net > 0 else 'NET SELL'})")

    print(f"\n  Top {top_n} Foreign NET BUY:")
    fb = active.nlargest(top_n, "Net Foreign")[
        ["Kode Saham", "Nama Perusahaan", "Penutupan", "Selisih",
         "Foreign Buy (B)", "Foreign Sell (B)", "Net Foreign (B)"]
    ]
    print(fb.to_string(index=False))

    print(f"\n  Top {top_n} Foreign NET SELL:")
    fs = active.nsmallest(top_n, "Net Foreign")[
        ["Kode Saham", "Nama Perusahaan", "Penutupan", "Selisih",
         "Foreign Buy (B)", "Foreign Sell (B)", "Net Foreign (B)"]
    ]
    print(fs.to_string(index=False))
    print()


# ── Trend: market-wide breadth over time ─────────────────────────────────────

def market_breadth(date_from: str = None, date_to: str = None):
    """Show daily advance/decline line and net foreign flow over a period."""
    df = load_master(date_from, date_to)

    summary = (
        df.groupby("DATE")
        .apply(lambda g: pd.Series({
            "Gainers":    (g["Selisih"] > 0).sum(),
            "Losers":     (g["Selisih"] < 0).sum(),
            "Flat":       (g["Selisih"] == 0).sum(),
            "Value (T)":  round(g["Nilai"].sum() / 1e12, 2),
            "Net Foreign (B)": round((g["Foreign Buy"] - g["Foreign Sell"]).sum() / 1e9, 2),
        }), include_groups=False)
        .reset_index()
    )
    summary["DATE"] = summary["DATE"].dt.strftime("%Y-%m-%d")
    summary["A/D"] = summary["Gainers"] - summary["Losers"]

    print(f"\n{'='*70}")
    print(f"  Market Breadth Summary")
    if date_from or date_to:
        print(f"  Period: {date_from or 'start'} → {date_to or 'latest'}")
    print(f"{'='*70}")
    print(summary.to_string(index=False))
    print()


# ── Sector-style: top performers over a period ───────────────────────────────

def top_performers(date_from: str, date_to: str, top_n: int = 20):
    """
    Rank all stocks by total return over a date range.
    Uses first available close as base and last available close as end.
    """
    df = load_master(date_from, date_to)

    first_day = df.groupby("Kode Saham").first()["Penutupan"].rename("Start Price")
    last_day  = df.groupby("Kode Saham").last()["Penutupan"].rename("End Price")
    name_map  = df.groupby("Kode Saham")["Nama Perusahaan"].last()

    perf = pd.concat([first_day, last_day, name_map], axis=1).dropna()
    perf = perf[perf["Start Price"] > 0]
    perf["Return (%)"] = ((perf["End Price"] - perf["Start Price"]) / perf["Start Price"] * 100).round(2)

    print(f"\n{'='*60}")
    print(f"  Top {top_n} Performers — {date_from} → {date_to}")
    print(f"{'='*60}")
    print(perf.nlargest(top_n, "Return (%)")[
        ["Nama Perusahaan", "Start Price", "End Price", "Return (%)"]
    ].to_string())

    print(f"\n  Bottom {top_n} Performers — {date_from} → {date_to}")
    print(f"{'='*60}")
    print(perf.nsmallest(top_n, "Return (%)")[
        ["Nama Perusahaan", "Start Price", "End Price", "Return (%)"]
    ].to_string())
    print()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDX Stock Market Analysis")
    parser.add_argument("--date",    type=str, help="Target date YYYY-MM-DD (default: latest)")
    parser.add_argument("--stock",   type=str, help="Show history for a single ticker, e.g. BBCA")
    parser.add_argument("--top",     type=int, default=10, help="Number of top movers to show")
    parser.add_argument("--foreign", action="store_true", help="Show foreign flow for the date")
    parser.add_argument("--breadth", action="store_true", help="Market breadth over date range")
    parser.add_argument("--from",    dest="date_from", type=str, help="Start date for range analysis")
    parser.add_argument("--to",      dest="date_to",   type=str, help="End date for range analysis")
    parser.add_argument("--perf",    action="store_true", help="Top/bottom performers over date range")
    args = parser.parse_args()

    if args.stock:
        stock_history(args.stock, args.date_from, args.date_to)
    elif args.foreign:
        foreign_flow(args.date, args.top)
    elif args.breadth:
        market_breadth(args.date_from, args.date_to)
    elif args.perf and args.date_from and args.date_to:
        top_performers(args.date_from, args.date_to, args.top)
    else:
        daily_snapshot(args.date, args.top)
