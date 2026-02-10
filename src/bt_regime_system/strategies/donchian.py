from __future__ import annotations

import numpy as np
import pandas as pd


def generate_signal(bars: pd.DataFrame, window: int = 20) -> pd.Series:
    """Simple Donchian breakout signal in {-1,0,1}."""
    upper = bars["high"].rolling(window).max().shift(1)
    lower = bars["low"].rolling(window).min().shift(1)
    close = bars["close"]

    raw = np.where(close > upper, 1.0, np.where(close < lower, -1.0, np.nan))
    signal = pd.Series(raw, index=bars.index).ffill().fillna(0.0)
    return signal
