from __future__ import annotations

import pandas as pd


OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def clean_1m_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean 1m bars: dedupe, sort, validate OHLC constraints, fill missing minutes."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Input DataFrame index must be DatetimeIndex")

    out = df.copy()
    out.columns = [c.lower().strip() for c in out.columns]
    missing = [c for c in OHLCV_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = out[OHLCV_COLS]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=OHLCV_COLS)

    valid = (
        (out["high"] >= out["low"])
        & (out["high"] >= out["open"])
        & (out["high"] >= out["close"])
        & (out["low"] <= out["open"])
        & (out["low"] <= out["close"])
        & (out["volume"] >= 0)
    )
    out = out.loc[valid]

    full_index = pd.date_range(out.index.min(), out.index.max(), freq="1min", tz="UTC")
    out = out.reindex(full_index)

    out["close"] = out["close"].ffill()
    for col in ["open", "high", "low"]:
        out[col] = out[col].fillna(out["close"])
    out["volume"] = out["volume"].fillna(0.0)

    return out.astype(float)
