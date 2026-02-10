from __future__ import annotations

import numpy as np
import pandas as pd


def generate_signal(
    bars: pd.DataFrame,
    window: int = 48,
    entry_z: float = 1.5,
    exit_z: float = 0.3,
) -> pd.Series:
    """Z-score mean reversion signal in {-1,0,1}."""
    close = bars["close"]
    mean = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    z = (close - mean) / std.replace(0.0, np.nan)

    raw = np.where(z <= -entry_z, 1.0, np.where(z >= entry_z, -1.0, np.nan))
    signal = pd.Series(raw, index=bars.index)
    signal = signal.where(z.abs() > exit_z, 0.0)
    signal = signal.ffill().fillna(0.0)
    return signal
