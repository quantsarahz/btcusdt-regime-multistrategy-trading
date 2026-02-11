from __future__ import annotations

import pandas as pd

from bt_regime_system.features.indicators import build_mean_reversion_features_15m

SIGNAL_COLUMN = "signal_mean_reversion"


def _standardize_bars_15m(bars_15m: pd.DataFrame) -> pd.DataFrame:
    out = bars_15m.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    required = ["timestamp", "close"]
    missing = set(required).difference(out.columns)
    if missing:
        raise ValueError(f"bars_15m missing columns: {sorted(missing)}")

    out = out[required].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")

    out = out.dropna(subset=["timestamp", "close"])
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return out.reset_index(drop=True)


def _stateful_mean_reversion_signal(
    zscore: pd.Series,
    entry_z: float,
    exit_z: float,
) -> pd.Series:
    position = 0
    out: list[float] = []

    for value in zscore:
        if pd.isna(value):
            out.append(float(position))
            continue

        if position == 0:
            if value <= -entry_z:
                position = 1
            elif value >= entry_z:
                position = -1
        elif position == 1:
            if value >= -exit_z:
                position = 0
        elif position == -1:
            if value <= exit_z:
                position = 0

        out.append(float(position))

    return pd.Series(out, index=zscore.index, dtype="float64")


def generate_signal(
    bars_15m: pd.DataFrame,
    z_window: int = 48,
    entry_z: float = 1.5,
    exit_z: float = 0.5,
) -> pd.DataFrame:
    """Generate mean-reversion signal on 15m bars using close z-score.

    Interface:
    - Input: bars_15m DataFrame with at least timestamp/close.
    - Output: DataFrame[timestamp, signal_mean_reversion] with signal in {-1, 0, 1}.
    """
    if exit_z > entry_z:
        raise ValueError("exit_z must be <= entry_z")

    standardized = _standardize_bars_15m(bars_15m)
    if standardized.empty:
        return pd.DataFrame(columns=["timestamp", SIGNAL_COLUMN])

    features = build_mean_reversion_features_15m(standardized, window=z_window)
    signal = _stateful_mean_reversion_signal(features["close_zscore"], entry_z=entry_z, exit_z=exit_z)

    out = pd.DataFrame(
        {
            "timestamp": features["timestamp"],
            SIGNAL_COLUMN: signal.clip(-1.0, 1.0).astype(float),
        }
    )
    return out
