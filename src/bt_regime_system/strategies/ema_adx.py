from __future__ import annotations

import numpy as np
import pandas as pd

from bt_regime_system.features.indicators import adx, ema


def generate_signal(
    bars: pd.DataFrame,
    fast: int = 21,
    slow: int = 55,
    adx_window: int = 14,
    adx_threshold: float = 20.0,
) -> pd.Series:
    """EMA trend signal gated by ADX strength."""
    fast_ema = ema(bars["close"], fast)
    slow_ema = ema(bars["close"], slow)
    trend = np.where(fast_ema > slow_ema, 1.0, -1.0)

    strength = adx(bars["high"], bars["low"], bars["close"], window=adx_window)
    gated = np.where(strength >= adx_threshold, trend, 0.0)
    return pd.Series(gated, index=bars.index).fillna(0.0)
