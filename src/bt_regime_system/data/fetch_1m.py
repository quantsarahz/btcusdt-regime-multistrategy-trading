from __future__ import annotations

from pathlib import Path

import pandas as pd


OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def load_1m_csv(path: Path) -> pd.DataFrame:
    """Load 1m OHLCV CSV with a `timestamp` column into canonical format."""
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    if "timestamp" not in df.columns:
        raise ValueError("Input CSV must contain `timestamp` column")

    missing = [c for c in OHLCV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    out = df.set_index("timestamp").sort_index()[OHLCV_COLS].astype(float)
    return out
