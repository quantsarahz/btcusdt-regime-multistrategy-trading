from __future__ import annotations

import pandas as pd


AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample_ohlcv(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV bars with right-closed/right-labeled convention."""
    if not isinstance(df_1m.index, pd.DatetimeIndex):
        raise TypeError("Input DataFrame index must be DatetimeIndex")

    bars = df_1m.resample(rule, label="right", closed="right").agg(AGG)
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    return bars


def build_15m_and_1h(df_1m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build 15m and 1h bars from cleaned 1m bars."""
    bars_15m = resample_ohlcv(df_1m, "15min")
    bars_1h = resample_ohlcv(df_1m, "1h")
    return bars_15m, bars_1h
