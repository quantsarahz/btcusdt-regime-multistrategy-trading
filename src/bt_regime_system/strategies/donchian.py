from __future__ import annotations

import pandas as pd

from bt_regime_system.features.indicators import build_donchian_features_15m

SIGNAL_COLUMN = "signal_donchian"


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
    window: int = 20,
    hold_until_opposite: bool = True,
) -> pd.DataFrame:
    """Generate Donchian breakout signal on 15m bars.

    Interface:
    - Input: bars_15m DataFrame with at least timestamp/high/low/close.
    - Output: DataFrame[timestamp, signal_donchian] with signal in {-1, 0, 1}.
    """
    standardized = _standardize_bars_15m(bars_15m)
    if standardized.empty:
        return pd.DataFrame(columns=["timestamp", SIGNAL_COLUMN])

    features = build_donchian_features_15m(standardized, window=window)

    long_entry = features["close"] > features["donchian_high"]
    short_entry = features["close"] < features["donchian_low"]

    signal = pd.Series(float("nan"), index=features.index, dtype="float64")
    signal.loc[long_entry] = 1.0
    signal.loc[short_entry] = -1.0

    if hold_until_opposite:
        signal = signal.ffill().fillna(0.0)
    else:
        signal = signal.fillna(0.0)

    out = pd.DataFrame(
        {
            "timestamp": features["timestamp"],
            SIGNAL_COLUMN: signal.clip(-1.0, 1.0).astype(float),
        }
    )
    return out
