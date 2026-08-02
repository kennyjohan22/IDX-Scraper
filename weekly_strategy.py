"""One-week factor and portfolio-risk helpers for the IDX dashboard.

The functions in this module only use information available on or before the
requested cutoff.  Trade plans are therefore suitable for a next-session
decision; they do not assume a fill at the close that produced the signal.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd


MIN_LIQUID_VALUE = 500_000_000
WEEKLY_MIN_LIQUID_VALUE = 5_000_000_000


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def estimate_foreign_value(net_share_volume, close):
    """Estimate foreign net value from share volume using the session close.

    IDX summary fields expose foreign buy/sell as share quantities. Multiplying
    by the close is only an estimate because actual transactions occur at many
    prices during the session.
    """
    return pd.to_numeric(net_share_volume, errors="coerce") * pd.to_numeric(
        close, errors="coerce"
    )


def idx_tick_size(price: float) -> int:
    """Return the IDX equity price fraction for a rupiah price level."""
    value = float(price)
    if value < 200:
        return 1
    if value < 500:
        return 2
    if value < 2_000:
        return 5
    if value < 5_000:
        return 10
    return 25


def round_to_idx_tick(price: float, direction: str = "nearest") -> float:
    """Round a rupiah price to a valid IDX price fraction."""
    if not np.isfinite(price) or price <= 0:
        return 0.0
    tick = idx_tick_size(price)
    scaled = float(price) / tick
    if direction == "down":
        units = math.floor(scaled)
    elif direction == "up":
        units = math.ceil(scaled)
    elif direction == "nearest":
        units = math.floor(scaled + 0.5)
    else:
        raise ValueError("direction must be 'down', 'up', or 'nearest'")
    return float(units * tick)


def _return_pct(closes: pd.Series, periods: int) -> float:
    valid = _numeric(closes).dropna()
    if len(valid) <= periods or valid.iloc[-periods - 1] <= 0:
        return 0.0
    return float((valid.iloc[-1] / valid.iloc[-periods - 1] - 1.0) * 100.0)


def _atr(history: pd.DataFrame, periods: int = 14) -> float:
    if history.empty:
        return 0.0
    high = _numeric(history["Tertinggi"])
    low = _numeric(history["Terendah"])
    previous = _numeric(history["Sebelumnya"])
    valid = (high > 0) & (low > 0) & (previous > 0)
    if not valid.any():
        return 0.0
    true_range = pd.concat(
        [(high - low).abs(), (high - previous).abs(), (low - previous).abs()],
        axis=1,
    ).max(axis=1)
    return float(true_range[valid].tail(periods).mean())


def compute_weekly_factors(
    history: pd.DataFrame,
    as_of_date,
    market_return_20d_pct: float = 0.0,
) -> Dict[str, float]:
    """Compute short-horizon factors for one ticker at a market cutoff."""
    cutoff = pd.Timestamp(as_of_date)
    g = history.copy()
    g["DATE"] = pd.to_datetime(g["DATE"])
    g = g[g["DATE"] <= cutoff].sort_values("DATE").reset_index(drop=True)
    if g.empty:
        raise ValueError("history has no observations on or before as_of_date")

    closes = _numeric(g["Penutupan"])
    last = g.iloc[-1]
    close = float(closes.iloc[-1]) if pd.notna(closes.iloc[-1]) else 0.0
    return_5d = _return_pct(closes, 5)
    return_20d = _return_pct(closes, 20)

    trend_window = closes.dropna().tail(21)
    path = float(trend_window.diff().abs().sum()) if len(trend_window) > 1 else 0.0
    trend_efficiency = (
        float((trend_window.iloc[-1] - trend_window.iloc[0]) / path)
        if path > 0
        else 0.0
    )

    recent = g.tail(5)
    recent_high = _numeric(recent["Tertinggi"])
    recent_low = _numeric(recent["Terendah"])
    recent_close = _numeric(recent["Penutupan"])
    ranges = recent_high - recent_low
    locations = ((recent_close - recent_low) / ranges.replace(0, np.nan)).clip(0, 1)
    close_location = float(locations.dropna().mean()) if locations.notna().any() else 0.5

    atr = _atr(g.tail(40))
    atr_pct = atr / close * 100.0 if close > 0 else 0.0
    avg_value_20d = float(_numeric(g["Nilai"]).tail(20).mean())
    latest_active = bool(
        pd.Timestamp(last["DATE"]) == cutoff
        and float(pd.to_numeric(last["Volume"], errors="coerce") or 0) > 0
        and close > 0
    )

    return {
        "close": close,
        "return_5d_pct": return_5d,
        "return_20d_pct": return_20d,
        "relative_strength_20d_pct": return_20d - float(market_return_20d_pct),
        "trend_efficiency_20d": trend_efficiency,
        "close_location_5d": close_location,
        "atr": atr,
        "atr_pct": atr_pct,
        "avg_value_20d": avg_value_20d,
        "latest_active": latest_active,
    }


def calculate_position_size(
    entry: float,
    stop: float,
    account_size: float,
    risk_pct: float,
    max_allocation_pct: float = 20.0,
) -> Dict[str, float]:
    """Size an IDX position in 100-share lots, capped by risk and capital."""
    if entry <= 0 or stop <= 0 or stop >= entry or account_size <= 0:
        return {"lots": 0, "shares": 0, "capital": 0.0, "risk_idr": 0.0}

    risk_per_share = entry - stop
    risk_budget = account_size * max(risk_pct, 0.0) / 100.0
    allocation_budget = account_size * max(max_allocation_pct, 0.0) / 100.0
    risk_shares = risk_budget / risk_per_share
    capital_shares = allocation_budget / entry
    shares = int(min(risk_shares, capital_shares) // 100) * 100
    shares = max(shares, 0)

    return {
        "lots": shares // 100,
        "shares": shares,
        "capital": float(shares * entry),
        "risk_idr": float(shares * risk_per_share),
    }


def assess_weekly_regime(
    market: pd.DataFrame,
    as_of_date=None,
    min_liquid_value: float = MIN_LIQUID_VALUE,
) -> Dict[str, float]:
    """Assess breadth on names that are liquid and active at the cutoff."""
    df = market.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    cutoff = (
        pd.Timestamp(as_of_date)
        if as_of_date is not None
        else pd.Timestamp(df["DATE"].max())
    )
    df = df[df["DATE"] <= cutoff].sort_values(["Kode Saham", "DATE"])

    rows = []
    for ticker, group in df.groupby("Kode Saham", sort=False):
        g = group[_numeric(group["Penutupan"]) > 0].sort_values("DATE")
        if len(g) < 21:
            continue
        latest = g.iloc[-1]
        if pd.Timestamp(latest["DATE"]) != cutoff:
            continue
        if float(pd.to_numeric(latest["Volume"], errors="coerce") or 0) <= 0:
            continue
        if float(_numeric(g["Nilai"]).tail(20).mean()) < min_liquid_value:
            continue
        closes = _numeric(g["Penutupan"])
        rows.append(
            {
                "ticker": ticker,
                "above_ma20": closes.iloc[-1] > closes.tail(20).mean(),
                "above_ma50": (
                    closes.iloc[-1] > closes.tail(50).mean() if len(closes) >= 50 else np.nan
                ),
                "return_20d_pct": _return_pct(closes, 20),
                "return_5d_pct": _return_pct(closes, 5),
            }
        )

    universe = pd.DataFrame(rows)
    if universe.empty:
        return {
            "label": "UNKNOWN",
            "score": 0,
            "liquid_stocks": 0,
            "pct_above_ma20": 0.0,
            "pct_above_ma50": 0.0,
            "median_return_20d_pct": 0.0,
            "median_return_5d_pct": 0.0,
            "max_exposure_pct": 0.0,
            "risk_cap_pct": 0.0,
        }

    pct_above_ma20 = float(universe["above_ma20"].mean() * 100.0)
    ma50_values = universe["above_ma50"].dropna()
    pct_above_ma50 = float(ma50_values.mean() * 100.0) if not ma50_values.empty else 50.0
    median_20d = float(universe["return_20d_pct"].median())
    median_5d = float(universe["return_5d_pct"].median())

    score = 0
    score += 1 if pct_above_ma20 >= 55 else -1 if pct_above_ma20 <= 45 else 0
    score += 1 if pct_above_ma50 >= 55 else -1 if pct_above_ma50 <= 45 else 0
    score += 1 if median_20d >= 2 else -1 if median_20d <= -2 else 0
    score += 1 if median_5d >= 0 else -1

    # The weekly blend is still a pilot: its 2026 holdout performance was
    # useful for relative ranking but negative after assumed costs.  Regime
    # therefore scales a deliberately small risk budget rather than granting
    # full portfolio exposure.
    if score >= 2:
        label, exposure, risk_cap = "RISK-ON", 25.0, 0.25
    elif score <= -2:
        label, exposure, risk_cap = "DEFENSIVE", 0.0, 0.0
    else:
        label, exposure, risk_cap = "SELECTIVE", 15.0, 0.15

    return {
        "label": label,
        "score": score,
        "liquid_stocks": int(len(universe)),
        "pct_above_ma20": pct_above_ma20,
        "pct_above_ma50": pct_above_ma50,
        "median_return_20d_pct": median_20d,
        "median_return_5d_pct": median_5d,
        "max_exposure_pct": exposure,
        "risk_cap_pct": risk_cap,
    }


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = _numeric(denominator).replace(0, np.nan)
    return _numeric(numerator) / denom


def _weekly_raw_factors(window: pd.DataFrame) -> Dict[str, float]:
    """Return the six preselected factors for one 60-session window."""
    close = _numeric(window["Penutupan"])
    previous = _numeric(window["Sebelumnya"])
    high = _numeric(window["Tertinggi"])
    volume = _numeric(window["Volume"])
    value = _numeric(window["Nilai"])
    frequency = _numeric(window["Frekuensi"])
    open_price = _numeric(window["Open Price"])
    foreign_net = _numeric(window["Foreign Buy"]) - _numeric(window["Foreign Sell"])

    valid_close_count = int(((close > 0) & close.notna()).sum())
    valid_high = high[(high > 0) & high.notna()]
    latest_close = float(close.iloc[-1])

    returns20 = _safe_ratio(close.tail(20), previous.tail(20)) - 1.0
    returns20 = returns20.replace([np.inf, -np.inf], np.nan).dropna()

    volume10 = volume.tail(10)
    volume10 = volume10[(volume10 > 0) & volume10.notna()]
    value10 = value.tail(10)
    frequency10 = frequency.tail(10)
    valid_trade = value10.notna() & (frequency10 > 0)

    recent_open = open_price.tail(5)
    recent_previous = previous.tail(5)
    recent_volume = volume.tail(5)
    valid_gap = (
        (recent_open > 0)
        & (recent_previous > 0)
        & (recent_volume > 0)
        & recent_open.notna()
        & recent_previous.notna()
        & recent_volume.notna()
    )
    gaps5 = (
        recent_open[valid_gap] / recent_previous[valid_gap] - 1.0
    ).replace([np.inf, -np.inf], np.nan).dropna()

    current_net = foreign_net.tail(5)
    current_volume = volume.tail(5)
    prior_net = foreign_net.iloc[-25:-5]
    prior_volume = volume.iloc[-25:-5]
    current_valid = current_net.notna() & (current_volume > 0)
    prior_valid = prior_net.notna() & (prior_volume > 0)

    if valid_close_count < 45 or len(valid_high) < 45:
        raise ValueError("insufficient valid 60-session price history")
    if len(returns20) < 15:
        raise ValueError("insufficient return history")
    if len(volume10) < 8 or float(volume10.mean()) <= 0:
        raise ValueError("insufficient volume history")
    if int(valid_trade.sum()) < 8 or float(frequency10[valid_trade].sum()) <= 0:
        raise ValueError("insufficient trade-frequency history")
    if len(gaps5) < 4:
        raise ValueError("insufficient gap history")
    if int(current_valid.sum()) < 4 or int(prior_valid.sum()) < 15:
        raise ValueError("insufficient foreign-flow history")

    current_flow_share = float(
        current_net[current_valid].sum() / current_volume[current_valid].sum()
    )
    prior_flow_share = float(prior_net[prior_valid].sum() / prior_volume[prior_valid].sum())
    net10 = foreign_net.tail(10)
    total_volume10 = volume.tail(10)
    net10_valid = net10.notna() & (total_volume10 > 0)

    return {
        "break60": latest_close / float(valid_high.max()) - 1.0,
        "rv20": float(returns20.std(ddof=1)),
        "volume_cv10": float(volume10.std(ddof=1) / volume10.mean()),
        "avg_trade_value10": float(
            value10[valid_trade].sum() / frequency10[valid_trade].sum()
        ),
        "abs_gap5": float(gaps5.abs().mean()),
        "foreign_acceleration": current_flow_share - prior_flow_share,
        "foreign_net10_share": float(
            net10[net10_valid].sum() / total_volume10[net10_valid].sum()
        )
        if net10_valid.any()
        else 0.0,
        "foreign_positive_days10": float((net10[net10_valid] > 0).mean())
        if net10_valid.any()
        else 0.0,
        "momentum20": _return_pct(close, 20) / 100.0,
    }


def rank_weekly_candidates(
    market: pd.DataFrame,
    as_of_date=None,
    min_avg_value: float = WEEKLY_MIN_LIQUID_VALUE,
) -> pd.DataFrame:
    """Rank the liquid, active universe with the preselected six-factor blend.

    The score uses same-date cross-sectional percentile ranks and only data
    through ``as_of_date``.  It is intended for a next-session decision.
    """
    df = market.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    cutoff = (
        pd.Timestamp(as_of_date)
        if as_of_date is not None
        else pd.Timestamp(df["DATE"].max())
    )
    df = df[df["DATE"] <= cutoff]
    market_dates = sorted(df["DATE"].dropna().unique())[-60:]
    recent = df[df["DATE"].isin(market_dates)].sort_values(["Kode Saham", "DATE"])

    rows = []
    for ticker, group in recent.groupby("Kode Saham", sort=False):
        g = group.sort_values("DATE").reset_index(drop=True)
        if g.empty or pd.Timestamp(g.iloc[-1]["DATE"]) != cutoff:
            continue
        close = float(pd.to_numeric(g.iloc[-1]["Penutupan"], errors="coerce"))
        latest_volume = float(pd.to_numeric(g.iloc[-1]["Volume"], errors="coerce"))
        avg_value20 = float(_numeric(g["Nilai"]).tail(20).fillna(0).mean())
        if (
            not np.isfinite(close)
            or not np.isfinite(latest_volume)
            or not np.isfinite(avg_value20)
            or close < 100
            or latest_volume <= 0
            or avg_value20 < min_avg_value
        ):
            continue
        try:
            factors = _weekly_raw_factors(g)
        except (ValueError, ZeroDivisionError):
            continue
        rows.append(
            {
                "Ticker": ticker,
                "Company": str(g.iloc[-1].get("Nama Perusahaan", ticker)),
                "Close": close,
                "Avg Value/day (B)": avg_value20 / 1e9,
                "Latest Active": True,
                **factors,
            }
        )

    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked

    percentile_columns = {
        "Breakout Rank": ("break60", True),
        "Low Vol Rank": ("rv20", False),
        "Volume Quality Rank": ("volume_cv10", False),
        "Trade Size Rank": ("avg_trade_value10", True),
        "Gap Discipline Rank": ("abs_gap5", False),
        "Foreign Accel Rank": ("foreign_acceleration", True),
    }
    rank_names = []
    for rank_name, (source, higher_is_better) in percentile_columns.items():
        ranked[rank_name] = ranked[source].rank(
            pct=True, method="average", ascending=higher_is_better
        )
        rank_names.append(rank_name)

    ranked["Weekly Score"] = ranked[rank_names].mean(axis=1) * 100.0
    ranked["Break60 %"] = ranked["break60"] * 100.0
    ranked["RV20 %"] = ranked["rv20"] * 100.0
    ranked["Volume CV10"] = ranked["volume_cv10"]
    ranked["Avg Trade (M)"] = ranked["avg_trade_value10"] / 1e6
    ranked["Abs Gap5 %"] = ranked["abs_gap5"] * 100.0
    ranked["Foreign Accel %Vol"] = ranked["foreign_acceleration"] * 100.0
    ranked["Foreign Net10 %Vol"] = ranked["foreign_net10_share"] * 100.0
    ranked["Foreign +Days %"] = ranked["foreign_positive_days10"] * 100.0
    ranked["Momentum20 %"] = ranked["momentum20"] * 100.0

    return ranked.sort_values("Weekly Score", ascending=False).reset_index(drop=True)


def select_actionable_candidates(
    rankings: pd.DataFrame,
    candidate_pool: int = 30,
    min_avg_value_b: float = 10.0,
    min_close: float = 500.0,
    min_foreign_net10_pct: float = 0.0,
    core_score: float = 75.0,
) -> pd.DataFrame:
    """Apply conservative execution guards to the relative weekly ranking.

    The factor score identifies relative leaders; these additional rules keep
    the actionable book in the most liquid names, avoid sub-Rp500 shares, and
    require non-negative recent foreign share flow.  They are portfolio guards,
    not claims that any selected stock has positive absolute expectancy.
    """
    required = {
        "Ticker",
        "Weekly Score",
        "Avg Value/day (B)",
        "Foreign Net10 %Vol",
        "Close",
    }
    missing = required.difference(rankings.columns)
    if missing:
        raise ValueError(f"rankings missing required columns: {sorted(missing)}")

    pool_size = max(int(candidate_pool), 0)
    candidates = (
        rankings.sort_values("Weekly Score", ascending=False)
        .head(pool_size)
        .copy()
    )
    if "Latest Active" in candidates.columns:
        candidates = candidates[candidates["Latest Active"].fillna(False)]
    candidates = candidates[
        (candidates["Avg Value/day (B)"] >= float(min_avg_value_b))
        & (candidates["Close"] >= float(min_close))
        & (candidates["Foreign Net10 %Vol"] >= float(min_foreign_net10_pct))
    ].copy()
    candidates["Recommendation"] = np.where(
        candidates["Weekly Score"] >= float(core_score),
        "CORE",
        "CONDITIONAL",
    )
    return candidates.reset_index(drop=True)


def build_weekly_trade_plan(
    market: pd.DataFrame,
    rankings: pd.DataFrame,
    regime: Dict[str, float],
    as_of_date,
    account_size: float,
    requested_risk_pct: float,
    max_positions: int = 5,
    max_position_pct: float = 8.0,
    max_pair_correlation: float = 0.65,
    existing_holdings: Dict[str, int] = None,
) -> pd.DataFrame:
    """Create a next-session plan capped against new and existing positions.

    ``existing_holdings`` maps tickers to actual share counts (not lots). Their
    latest market value consumes the regime exposure budget, and their recent
    return streams participate in the correlation guard.
    """
    if rankings.empty or regime.get("max_exposure_pct", 0) <= 0:
        return pd.DataFrame()

    cutoff = pd.Timestamp(as_of_date)
    df = market.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df[df["DATE"] <= cutoff]
    effective_risk = min(
        max(float(requested_risk_pct), 0.0), float(regime.get("risk_cap_pct", 0.0))
    )
    total_capital_limit = account_size * float(regime["max_exposure_pct"]) / 100.0
    holdings = {}
    for ticker, shares in (existing_holdings or {}).items():
        try:
            parsed_shares = int(shares)
        except (TypeError, ValueError, OverflowError):
            return pd.DataFrame()
        if parsed_shares > 0:
            holdings[str(ticker)] = parsed_shares
    capital_used = 0.0
    rows = []
    selected_returns = []

    for ticker, shares in holdings.items():
        g = df[df["Kode Saham"] == ticker].sort_values("DATE").reset_index(drop=True)
        if g.empty:
            # Unknown existing exposure cannot be safely reconciled to the cap.
            return pd.DataFrame()
        close = float(pd.to_numeric(g.iloc[-1]["Penutupan"], errors="coerce"))
        if not np.isfinite(close) or close <= 0:
            return pd.DataFrame()
        capital_used += shares * close
        holding_returns = pd.Series(
            _numeric(g["Penutupan"])
            .pct_change(fill_method=None)
            .tail(60)
            .to_numpy(),
            index=pd.to_datetime(g["DATE"]).tail(60),
            dtype=float,
        ).dropna()
        if not holding_returns.empty:
            selected_returns.append(holding_returns)

    if capital_used >= total_capital_limit:
        return pd.DataFrame()

    for _, candidate in rankings.head(max_positions * 3).iterrows():
        if len(rows) >= max_positions:
            break
        ticker = candidate["Ticker"]
        if ticker in holdings:
            continue
        g = df[df["Kode Saham"] == ticker].sort_values("DATE").reset_index(drop=True)
        if len(g) < 20 or pd.Timestamp(g.iloc[-1]["DATE"]) != cutoff:
            continue

        candidate_returns = pd.Series(
            _numeric(g["Penutupan"])
            .pct_change(fill_method=None)
            .tail(60)
            .to_numpy(),
            index=pd.to_datetime(g["DATE"]).tail(60),
            dtype=float,
        ).dropna()
        too_correlated = False
        for existing in selected_returns:
            aligned = pd.concat([candidate_returns, existing], axis=1).dropna()
            if len(aligned) >= 20 and float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])) > max_pair_correlation:
                too_correlated = True
                break
        if too_correlated:
            continue

        latest = g.iloc[-1]
        close = float(pd.to_numeric(latest["Penutupan"], errors="coerce"))
        last_high = float(pd.to_numeric(latest["Tertinggi"], errors="coerce"))
        atr = _atr(g.tail(40))
        if close <= 0 or atr <= 0:
            continue
        ma20 = float(_numeric(g["Penutupan"]).tail(20).mean())
        stop = max(close - 2.0 * atr, ma20 * 0.98)
        stop = max(stop, close * 0.92)
        stop = min(stop, close * 0.97)
        entry_reference = round_to_idx_tick(close, direction="nearest")
        max_entry = round_to_idx_tick(close + 0.5 * atr, direction="down")
        stop = round_to_idx_tick(stop, direction="up")
        last_high = round_to_idx_tick(last_high, direction="nearest")
        if max_entry < entry_reference or stop >= entry_reference:
            continue
        sizing_entry = max_entry

        remaining = max(total_capital_limit - capital_used, 0.0)
        if remaining <= 0:
            break
        remaining_pct = remaining / account_size * 100.0
        allocation_pct = min(max_position_pct, remaining_pct)
        sizing = calculate_position_size(
            entry=sizing_entry,
            stop=stop,
            account_size=account_size,
            risk_pct=effective_risk,
            max_allocation_pct=allocation_pct,
        )
        if sizing["lots"] <= 0:
            continue
        capital_used += sizing["capital"]
        selected_returns.append(candidate_returns)
        risk_per_share = sizing_entry - stop
        target_1r = round_to_idx_tick(
            sizing_entry + risk_per_share, direction="down"
        )
        target_2r = round_to_idx_tick(
            sizing_entry + 2.0 * risk_per_share, direction="down"
        )

        rows.append(
            {
                "Ticker": ticker,
                "Weekly Score": round(float(candidate["Weekly Score"]), 1),
                "Close Ref": entry_reference,
                "Last High": last_high,
                "Trigger": entry_reference,
                "Max Entry": max_entry,
                "Stop": stop,
                "Target 1R": target_1r,
                "Target 2R": target_2r,
                "ATR %": round(atr / close * 100.0, 1),
                "Lots": int(sizing["lots"]),
                "Capital IDR": float(sizing["capital"]),
                "Risk IDR": float(sizing["risk_idr"]),
                "Effective Risk %": effective_risk,
            }
        )

    return pd.DataFrame(rows)
