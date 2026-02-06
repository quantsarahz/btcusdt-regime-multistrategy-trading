"""Regime classification engine for BTCUSDT 1h bars."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from btcusdt_regime_trading.data.loaders import load_klines_1h_processed
from btcusdt_regime_trading.features.bar_features import make_bar_features_1h


def classify_regime_raw(
    features_df: pd.DataFrame,
    trend_threshold: float = 0.0,
    vol_quantile: float = 0.5,
) -> pd.Series:
    """Classify raw regimes based on trend strength and volatility."""
    if "trend_strength" not in features_df.columns or "rv_ewm" not in features_df.columns:
        raise ValueError("features_df must include trend_strength and rv_ewm")

    trend = features_df["trend_strength"]
    rv = features_df["rv_ewm"]

    direction = pd.Series(index=features_df.index, dtype="object")
    direction[trend > trend_threshold] = "Up"
    direction[trend < -trend_threshold] = "Down"
    direction[(trend <= trend_threshold) & (trend >= -trend_threshold)] = "Range"

    vol_cut = rv.rolling(168, min_periods=24).quantile(vol_quantile)
    volatility = pd.Series(index=features_df.index, dtype="object")
    volatility[rv > vol_cut] = "HighVol"
    volatility[rv <= vol_cut] = "LowVol"

    return (direction + "_" + volatility).astype("object")


def apply_stability_filter(regime_raw: pd.Series, k: int = 3) -> pd.Series:
    """Stabilize regime changes by requiring k consecutive bars."""
    if k <= 0:
        raise ValueError("k must be >= 1")

    index = regime_raw.index
    out = pd.Series(index=index, dtype="object")

    current: Optional[str] = None
    candidate: Optional[str] = None
    candidate_count = 0

    for ts, value in regime_raw.items():
        if pd.isna(value):
            out.loc[ts] = current
            continue

        if current is None:
            if candidate is None or value != candidate:
                candidate = value
                candidate_count = 1
            else:
                candidate_count += 1
            if candidate_count >= k:
                current = candidate
            out.loc[ts] = current
            continue

        if value == current:
            candidate = None
            candidate_count = 0
            out.loc[ts] = current
            continue

        if candidate is None or value != candidate:
            candidate = value
            candidate_count = 1
        else:
            candidate_count += 1

        if candidate_count >= k:
            current = candidate
            candidate = None
            candidate_count = 0

        out.loc[ts] = current

    return out


def compute_regime_1h(
    features_df: pd.DataFrame,
    k: int = 3,
    trend_threshold: float = 0.0,
    vol_quantile: float = 0.5,
) -> pd.DataFrame:
    """Compute raw and stabilized regimes for 1h features."""
    regime_raw = classify_regime_raw(features_df, trend_threshold=trend_threshold, vol_quantile=vol_quantile)
    regime = apply_stability_filter(regime_raw, k=k)

    out = pd.DataFrame({"regime_raw": regime_raw, "regime": regime}, index=features_df.index)
    if "is_imputed" in features_df.columns:
        out["is_imputed"] = features_df["is_imputed"]
    return out


def _count_switches(regime: pd.Series) -> int:
    cleaned = regime.dropna()
    if cleaned.empty:
        return 0
    return int((cleaned != cleaned.shift(1)).sum() - 1)


def main() -> None:
    df = load_klines_1h_processed()
    features = make_bar_features_1h(df)
    regimes = compute_regime_1h(features)

    print("Regime counts:")
    print(regimes["regime"].value_counts(dropna=False))
    print("Regime switches:", _count_switches(regimes["regime"]))
    print("Head:")
    print(regimes.head(10))


if __name__ == "__main__":
    main()
