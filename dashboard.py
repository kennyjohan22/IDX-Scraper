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

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV  = os.path.join(BASE_DIR, "master.csv")
SCREENS_DIR = os.path.join(BASE_DIR, "screens")

st.set_page_config(
    page_title="IDX Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(MASTER_CSV, low_memory=False)
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
    df["pct_chg"] = (df["Selisih"] / df["Sebelumnya"].replace(0, np.nan) * 100).round(2)
    return df.sort_values(["Kode Saham", "DATE"]).reset_index(drop=True)

@st.cache_data(ttl=300)
def get_trading_days():
    df = load_data()
    return sorted(df["DATE"].unique(), reverse=True)

# ── Screener logic (reused from screener.py) ──────────────────────────────────
def run_screener(df, as_of_date):
    BASELINE = 30; SIGNAL = 10
    MIN_PRICE = 100; MIN_NILAI = 500_000_000; VOL_THRESH = 1.5
    cutoff = pd.to_datetime(as_of_date)
    d = df[df["DATE"] <= cutoff]
    results = []
    for ticker, grp in d.groupby("Kode Saham"):
        g = grp[grp["Penutupan"] > 0].reset_index(drop=True)
        if len(g) < BASELINE + SIGNAL + 5: continue
        base = g.iloc[-(BASELINE+SIGNAL):-SIGNAL].copy()
        sig  = g.iloc[-SIGNAL:].copy()
        lat  = g.iloc[-1]
        if base["Nilai"].mean() < MIN_NILAI: continue
        if lat["Penutupan"] < MIN_PRICE: continue
        if sig["Volume"].sum() == 0: continue
        b_vol  = base["Volume"].mean();  s_vol  = sig["Volume"].mean()
        b_val  = base["Nilai"].mean();   s_val  = sig["Nilai"].mean()
        b_freq = base["Frekuensi"].mean(); s_freq = sig["Frekuensi"].mean()
        vr = s_vol/b_vol if b_vol>0 else 0
        if vr < VOL_THRESH: continue
        valr = s_val/b_val if b_val>0 else 0
        freqr = s_freq/b_freq if b_freq>0 else 0
        base["rng"] = (base["Tertinggi"]-base["Terendah"])/base["Sebelumnya"].replace(0,np.nan)*100
        sig["rng"]  = (sig["Tertinggi"] -sig["Terendah"]) /sig["Sebelumnya"].replace(0,np.nan)*100
        rr = sig["rng"].mean()/base["rng"].mean() if base["rng"].mean()>0 else 1
        if rr < 0: continue
        f = sig.iloc[0]["Sebelumnya"]; l = sig.iloc[-1]["Penutupan"]
        pc = (l-f)/f*100 if f>0 else 0
        if pc < -30: continue
        nf = sig["Net Foreign"].sum()
        sc = 0
        sc += 40 if vr>=2 else 28 if vr>=1.5 else 15
        sc += 35 if rr<0.7 else 22 if rr<0.9 else 10 if rr<1.0 else 0
        sc += 15 if -5<=pc<=5 else 8 if -15<=pc<=15 else -8 if pc>20 else 0
        sc += 10 if freqr>=1.5 else 5 if freqr>=1.2 else 0
        sc += 5 if valr>=1.5 else 0
        sc = max(0, min(100, sc))
        has_range = rr < 0.90; has_flat = -10<=pc<=10
        secondary = sum([has_range, has_flat])
        alert = "🔴 STRONG" if secondary==2 else "🟡 WATCH" if secondary==1 else "⚪ RADAR"
        results.append({
            "Ticker": ticker, "Company": str(lat["Nama Perusahaan"])[:35],
            "Alert": alert, "Score": sc,
            "Close": int(lat["Penutupan"]),
            "Vol Ratio": round(vr,2), "Range Ratio": round(rr,2),
            "10d Chg%": round(pc,1), "Freq Ratio": round(freqr,2),
            "Net Fgn (B)": round(nf/1e9,2),
            "Avg Val/day (B)": round(b_val/1e9,2),
        })
    if not results:
        return pd.DataFrame()
    out = pd.DataFrame(results)
    order = {"🔴 STRONG":0,"🟡 WATCH":1,"⚪ RADAR":2}
    out["_ord"] = out["Alert"].map(order)
    return out.sort_values(["_ord","Score"], ascending=[True,False]).drop(columns="_ord").reset_index(drop=True)


# ── Sidebar nav ───────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/IDX_logo.svg/512px-IDX_logo.svg.png", width=120)
st.sidebar.title("IDX Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Market Overview", "🔍 Screener", "📈 Stock Deep Dive", "💼 Portfolio"],
)

df = load_data()
trading_days = get_trading_days()
latest_date  = trading_days[0]

st.sidebar.markdown("---")
st.sidebar.caption(f"Data: {df['DATE'].min().date()} → {df['DATE'].max().date()}")
st.sidebar.caption(f"Trading days: {len(trading_days)}")
st.sidebar.caption(f"Stocks: {df['Kode Saham'].nunique()}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Market Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Market Overview":
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
    net_fgn  = active["Net Foreign"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Value",    f"Rp {tot_val/1e12:.2f}T")
    c2.metric("Total Volume",   f"{tot_vol/1e9:.1f}B lots")
    c3.metric("Gainers",        f"{gainers:,}", delta=f"+{gainers}")
    c4.metric("Losers",         f"{losers:,}",  delta=f"-{losers}", delta_color="inverse")
    c5.metric("Net Foreign",
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
    active["Net Fgn (B)"] = active["Net Foreign"]/1e9
    fc1, fc2 = st.columns(2)
    with fc1:
        fb = active.nlargest(10,"Net Foreign")[["Kode Saham","Nama Perusahaan","Penutupan","pct_chg","Net Fgn (B)"]].copy()
        st.dataframe(fb.reset_index(drop=True), use_container_width=True)
    with fc2:
        fs = active.nsmallest(10,"Net Foreign")[["Kode Saham","Nama Perusahaan","Penutupan","pct_chg","Net Fgn (B)"]].copy()
        st.dataframe(fs.reset_index(drop=True), use_container_width=True)

    # Market breadth over time
    st.markdown("---")
    st.subheader("Market Breadth Trend (last 60 days)")
    recent = df[df["DATE"] >= df["DATE"].max() - pd.Timedelta(days=90)]
    breadth = recent.groupby("DATE").apply(
        lambda g: pd.Series({
            "Gainers":   (g["pct_chg"]>0).sum(),
            "Losers":    (g["pct_chg"]<0).sum(),
            "Net Fgn (B)": g["Net Foreign"].sum()/1e9,
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
    st.title("🔍 Pre-Rocket Screener")
    st.caption("Signals: Volume >1.5x baseline | Range compression | Price flat/down")

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

    if not strong.empty:
        st.subheader("🔴 STRONG — All 3 signals active")
        st.dataframe(
            strong.drop(columns=["Alert"]).style
                .background_gradient(subset=["Score"], cmap="Reds")
                .background_gradient(subset=["Vol Ratio"], cmap="Oranges"),
            use_container_width=True, hide_index=True,
        )

    if not watch.empty:
        st.subheader("🟡 WATCH — 2 signals active")
        st.dataframe(
            watch.drop(columns=["Alert"]).style
                .background_gradient(subset=["Score"], cmap="YlOrBr"),
            use_container_width=True, hide_index=True,
        )

    if not radar.empty:
        with st.expander(f"⚪ RADAR — Volume spike only ({len(radar)} stocks)"):
            st.dataframe(radar.drop(columns=["Alert"]), use_container_width=True, hide_index=True)

    # Score distribution
    st.markdown("---")
    fig = px.histogram(results, x="Score", color="Alert",
                       color_discrete_map={"🔴 STRONG":"#e53935","🟡 WATCH":"#ffc107","⚪ RADAR":"#90a4ae"},
                       nbins=20, title="Score Distribution",
                       labels={"Score":"Composite Score","count":"# Stocks"})
    fig.update_layout(height=300, margin=dict(t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)


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
    net_fgn_30  = g.tail(30)["Net Foreign"].sum()

    st.subheader(f"{ticker} — {name}")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Close",        f"{int(close):,}",    delta=f"{chg_pct:.2f}%")
    c2.metric("Period High",  f"{int(hi52):,}")
    c3.metric("Period Low",   f"{int(lo52):,}")
    c4.metric("Avg Vol (20d)",f"{avg_vol_20/1e6:.1f}M")
    c5.metric("Net Fgn (30d)",f"Rp {net_fgn_30/1e9:.1f}B",
              delta="NET BUY" if net_fgn_30>0 else "NET SELL",
              delta_color="normal" if net_fgn_30>0 else "inverse")

    # Candlestick + volume
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=g["DATE"], open=g["Open Price"],
        high=g["Tertinggi"], low=g["Terendah"], close=g["Penutupan"],
        name="Price", increasing_line_color="#00c853", decreasing_line_color="#ff1744",
    ))
    fig.update_layout(
        title=f"{ticker} Price", height=420,
        xaxis_rangeslider_visible=False,
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Volume bars
    colors = ["#00c853" if p>=0 else "#ff1744" for p in g["pct_chg"]]
    fig2 = go.Figure(go.Bar(x=g["DATE"], y=g["Volume"]/1e6, marker_color=colors, name="Volume (M)"))
    fig2.update_layout(title="Volume (M lots)", height=200, margin=dict(t=40,b=10), yaxis_title="M lots")
    st.plotly_chart(fig2, use_container_width=True)

    # Foreign flow
    g["nf_b"] = g["Net Foreign"]/1e9
    colors_fgn = ["#1565c0" if v>0 else "#b71c1c" for v in g["nf_b"]]
    fig3 = go.Figure(go.Bar(x=g["DATE"], y=g["nf_b"], marker_color=colors_fgn, name="Net Foreign (B)"))
    fig3.update_layout(title="Net Foreign Flow (B IDR)", height=200, margin=dict(t=40,b=10), yaxis_title="B IDR")
    st.plotly_chart(fig3, use_container_width=True)

    # Raw data table
    with st.expander("Raw data"):
        show = g[["DATE","Sebelumnya","Open Price","Tertinggi","Terendah","Penutupan",
                   "pct_chg","Volume","Nilai","Frekuensi","Net Foreign"]].copy()
        show["DATE"] = show["DATE"].dt.strftime("%Y-%m-%d")
        show["Nilai (B)"] = (show["Nilai"]/1e9).round(1)
        show["Net Fgn (B)"] = (show["Net Foreign"]/1e9).round(2)
        st.dataframe(show.drop(columns=["Nilai","Net Foreign"]).sort_values("DATE",ascending=False),
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
        shares  = c2.number_input("Shares (lots × 100)", value=int(pos["shares"]), min_value=0, key=f"s{i}")
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
        g_stock   = df[df["Kode Saham"]==t].sort_values("DATE")
        hi = g_stock["Tertinggi"].max()
        dd = (cp - hi) / hi * 100 if hi > 0 else 0
        pnl_rows.append({
            "Ticker": t,
            "Company": str(df[df["Kode Saham"]==t].iloc[-1]["Nama Perusahaan"])[:30],
            "Shares (lots)": lots,
            "Entry": int(entry),
            "Current": int(cp),
            "Chg%": round(pnl_pct, 1),
            "P&L (M IDR)": round(pnl_idr/1e6, 2),
            "Mkt Val (M)": round(mkt_val/1e6, 1),
            "DD from ATH%": round(dd, 1),
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

    st.dataframe(
        pnl_df.style
            .applymap(color_pnl, subset=["Chg%","P&L (M IDR)","DD from ATH%"])
            .format({"Chg%":"{:.1f}%", "DD from ATH%":"{:.1f}%",
                     "P&L (M IDR)":"{:+.2f}", "Mkt Val (M)":"{:.1f}"}),
        use_container_width=True, hide_index=True,
    )

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
