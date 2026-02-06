"""Bar-level feature engineering for regime classification."""

from __future__ import annotations

import math
import pandas as pd

from btcusdt_regime_trading.data.loaders import load_klines_1h_processed


def make_bar_features_1h(df: pd.DataFrame) -> pd.DataFrame:
    """Compute bar-level features for 1h BTCUSDT data."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")

    close = df["close"]
    high = df["high"]
    low = df["low"]

    ratio = close / close.shift(1)
    ratio = ratio.where(ratio > 0)
    log_ret_1 = ratio.apply(math.log)

    rv_ewm = log_ret_1.ewm(span=24, adjust=False).std()

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=48, adjust=False).mean()
    trend_strength = ema_fast - ema_slow

    range_norm = (high - low) / close

    out = pd.DataFrame(
        {
            "log_ret_1": log_ret_1,
            "rv_ewm": rv_ewm,
            "trend_strength": trend_strength,
            "range_norm": range_norm,
        },
        index=df.index,
    )

    if "is_imputed" in df.columns:
        out["is_imputed"] = df["is_imputed"]

    return out


def _print_summary(features: pd.DataFrame) -> None:
    print(f"shape: {features.shape}")
    print("head:")
    print(features.head(5))
    print("tail:")
    print(features.tail(5))
    print("nan_count:")
    print(features.isna().sum())


def main() -> None:
    df = load_klines_1h_processed()
    features = make_bar_features_1h(df)
    _print_summary(features)


if __name__ == "__main__":
    main()
