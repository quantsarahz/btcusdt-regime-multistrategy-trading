from __future__ import annotations

import numpy as np
import pandas as pd

from bt_regime_system.features.indicators import adx, ema


UP = "trend_up"
DOWN = "trend_down"
RANGE = "range"


def detect_regime_1h(
    bars_1h: pd.DataFrame,
    fast_ema: int = 24,
    slow_ema: int = 96,
    adx_window: int = 14,
    adx_threshold: float = 20.0,
) -> pd.Series:
    """Regime labels from 1h bars based on EMA direction and ADX trend strength."""
    fast = ema(bars_1h["close"], fast_ema)
    slow = ema(bars_1h["close"], slow_ema)
    strength = adx(bars_1h["high"], bars_1h["low"], bars_1h["close"], adx_window)

    labels = np.where(
        strength < adx_threshold,
        RANGE,
        np.where(fast >= slow, UP, DOWN),
    )
    return pd.Series(labels, index=bars_1h.index, name="regime")
