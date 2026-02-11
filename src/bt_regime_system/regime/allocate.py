from __future__ import annotations

from typing import Any

import pandas as pd

from bt_regime_system.regime.detect import R1, R2, R3, R4

STRATEGY_KEYS = ("donchian", "ema_adx", "mean_reversion")

DEFAULT_REGIME_ALLOCATIONS: dict[str, dict[str, float]] = {
    R1: {"donchian": 0.6, "ema_adx": 0.4, "mean_reversion": 0.0},
    R2: {"donchian": 0.4, "ema_adx": 0.4, "mean_reversion": 0.0},
    R3: {"donchian": 0.0, "ema_adx": 0.0, "mean_reversion": 1.0},
    R4: {"donchian": 0.0, "ema_adx": 0.0, "mean_reversion": 0.0},
}


def _validate_allocations(regime_allocations: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    out = {k: v.copy() for k, v in DEFAULT_REGIME_ALLOCATIONS.items()}
    if regime_allocations is None:
        return out

    if not isinstance(regime_allocations, dict):
        raise ValueError("regime_allocations must be a mapping")

    for regime in [R1, R2, R3, R4]:
        if regime not in regime_allocations:
            continue

        raw_weights = regime_allocations[regime]
        if not isinstance(raw_weights, dict):
            raise ValueError(f"allocation for {regime} must be a mapping")

        row: dict[str, float] = {}
        for key in STRATEGY_KEYS:
            value = raw_weights.get(key, out[regime][key])
            value_f = float(value)
            if value_f < 0:
                raise ValueError(f"allocation weight must be non-negative: {regime}.{key}")
            row[key] = value_f

        total = sum(row.values())
        if total > 1.0 + 1e-9:
            raise ValueError(f"allocation weights for {regime} exceed 1.0: {total:.6f}")

        out[regime] = row

    return out


def weights_for_regime(
    regime: str,
    regime_allocations: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    allocations = _validate_allocations(regime_allocations)
    if regime not in allocations:
        raise KeyError(f"Unknown regime: {regime}")
    return allocations[regime].copy()


def build_allocation_table(
    regime_series: pd.Series,
    regime_allocations: dict[str, dict[str, float]] | None = None,
    default_regime: str = R4,
) -> pd.DataFrame:
    """Map regime labels to strategy weights row-by-row."""
    if not isinstance(regime_series, pd.Series):
        regime_series = pd.Series(regime_series)

    allocations = _validate_allocations(regime_allocations)
    if default_regime not in allocations:
        raise KeyError(f"Unknown default_regime: {default_regime}")

    rows: list[dict[str, Any]] = []
    for ts, regime in regime_series.items():
        if pd.isna(regime):
            resolved_regime = default_regime
        else:
            resolved_regime = str(regime)
            if resolved_regime not in allocations:
                resolved_regime = default_regime

        alloc = allocations[resolved_regime]
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
        out = out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return out
