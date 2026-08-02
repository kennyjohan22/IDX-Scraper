"""Canonical feature calculations shared by production and research screeners."""

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def obv_slope_is_positive(signal: pd.DataFrame) -> bool:
    """Return whether On-Balance Volume has a positive linear trend."""
    obv = 0.0
    obv_series = []
    for row in signal.itertuples(index=False):
        close = getattr(row, "Penutupan")
        previous = getattr(row, "Sebelumnya")
        volume = getattr(row, "Volume")
        if close > previous:
            obv += volume
        elif close < previous:
            obv -= volume
        obv_series.append(obv)

    if len(obv_series) < 2:
        return False
    return bool(np.polyfit(range(len(obv_series)), obv_series, 1)[0] > 0)


def position_in_trading_range(
    history: pd.DataFrame,
    as_of_pos: int,
    lookback: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Return the latest close's 0-100 position in a valid rolling price range."""
    window = history.iloc[max(0, as_of_pos - lookback) : as_of_pos]
    if window.empty:
        return 50.0

    valid_highs = window.loc[window["Tertinggi"] > 0, "Tertinggi"]
    valid_lows = window.loc[window["Terendah"] > 0, "Terendah"]
    if valid_highs.empty or valid_lows.empty:
        return 50.0

    high = valid_highs.max()
    low = valid_lows.min()
    if high <= low:
        return 50.0

    latest_close = history.iloc[as_of_pos - 1]["Penutupan"]
    position = (latest_close - low) / (high - low) * 100
    return float(max(0.0, min(100.0, position)))
