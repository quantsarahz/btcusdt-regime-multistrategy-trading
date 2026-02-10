from __future__ import annotations

import pandas as pd


def to_utc_index(index: pd.Index) -> pd.DatetimeIndex:
    """Return a timezone-aware UTC DatetimeIndex."""
    out = pd.to_datetime(index, utc=True)
    if not isinstance(out, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex-like input")
    return out
