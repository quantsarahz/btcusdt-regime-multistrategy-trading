from __future__ import annotations

import pandas as pd

from bt_regime_system.features.indicators import build_ema_adx_features_15m

SIGNAL_COLUMN = "signal_ema_adx"


def _standardize_bars_15m(bars_15m: pd.DataFrame) -> pd.DataFrame:
    out = bars_15m.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    required = ["timestamp", "high", "low", "close"]
    missing = set(required).difference(out.columns)
    if missing:
        raise ValueError(f"bars_15m missing columns: {sorted(missing)}")

    out = out[required].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for col in ["high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["timestamp", "high", "low", "close"])
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return out.reset_index(drop=True)


def generate_signal(
    bars_15m: pd.DataFrame,
    ema_fast: int = 21,
    ema_slow: int = 55,
    adx_window: int = 14,
    adx_threshold: float = 20.0,
    use_adx_filter: bool = True,
) -> pd.DataFrame:
    """Generate EMA+ADX trend signal on 15m bars.

    Interface:
    - Input: bars_15m DataFrame with at least timestamp/high/low/close.
    - Output: DataFrame[timestamp, signal_ema_adx] with signal in {-1, 0, 1}.
    """
    standardized = _standardize_bars_15m(bars_15m)
    if standardized.empty:
        return pd.DataFrame(columns=["timestamp", SIGNAL_COLUMN])

    features = build_ema_adx_features_15m(
        standardized,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        adx_window=adx_window,
    )

    trend_up = features["ema_fast"] > features["ema_slow"]
    trend_down = features["ema_fast"] < features["ema_slow"]
    trend_active = (features["adx"] >= adx_threshold) if use_adx_filter else pd.Series(True, index=features.index)

    signal = pd.Series(0.0, index=features.index, dtype="float64")
    signal.loc[trend_active & trend_up] = 1.0
    signal.loc[trend_active & trend_down] = -1.0

    out = pd.DataFrame(
        {
            "timestamp": features["timestamp"],
            SIGNAL_COLUMN: signal.clip(-1.0, 1.0).astype(float),
        }
    )
    return out
