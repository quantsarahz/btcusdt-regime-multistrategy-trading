from __future__ import annotations

from typing import Any

import pandas as pd

from bt_regime_system.regime.detect import R1, R2, R3, R4


REGIME_ALLOCATIONS: dict[str, dict[str, float]] = {
    R1: {"donchian": 0.6, "ema_adx": 0.4, "mean_reversion": 0.0},
    R2: {"donchian": 0.4, "ema_adx": 0.4, "mean_reversion": 0.0},
    R3: {"donchian": 0.0, "ema_adx": 0.0, "mean_reversion": 1.0},
    R4: {"donchian": 0.0, "ema_adx": 0.0, "mean_reversion": 0.0},
}


def weights_for_regime(regime: str) -> dict[str, float]:
    if regime not in REGIME_ALLOCATIONS:
        raise KeyError(f"Unknown regime: {regime}")
    return REGIME_ALLOCATIONS[regime].copy()


def build_allocation_table(regime_series: pd.Series) -> pd.DataFrame:
    """Map regime labels to strategy weights row-by-row."""
    if not isinstance(regime_series, pd.Series):
        regime_series = pd.Series(regime_series)

    rows: list[dict[str, Any]] = []
    for ts, regime in regime_series.items():
        if pd.isna(regime):
            alloc = REGIME_ALLOCATIONS[R4]
            resolved_regime = R4
        else:
            resolved_regime = str(regime)
            alloc = weights_for_regime(resolved_regime)

        row = {
            "timestamp": ts,
            "regime": resolved_regime,
            "w_donchian": float(alloc["donchian"]),
            "w_ema_adx": float(alloc["ema_adx"]),
            "w_mean_reversion": float(alloc["mean_reversion"]),
        }
        row["w_total"] = row["w_donchian"] + row["w_ema_adx"] + row["w_mean_reversion"]
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("timestamp").reset_index(drop=True)
    return out
