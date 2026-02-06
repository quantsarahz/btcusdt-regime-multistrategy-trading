"""Processed dataset loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from btcusdt_regime_trading.utils.paths import PROCESSED_DATA_DIR


def _load_partitioned_parquet(base_dir: Path, symbol: str, interval: str) -> pd.DataFrame:
    """Load partitioned parquet files and return a UTC-indexed DataFrame."""
    data_dir = base_dir / "binance_spot" / symbol / interval
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()

    frames = [pd.read_parquet(path) for path in files]
    df = pd.concat(frames, axis=0)

    if "open_time" in df.columns:
        df = df.set_index("open_time", drop=False)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index is not DatetimeIndex")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_klines_1h_processed(symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Load processed 1h klines for a symbol."""
    return _load_partitioned_parquet(PROCESSED_DATA_DIR, symbol, "1h")


def load_klines_1h(source: str = "processed") -> pd.DataFrame:
    """Load 1h klines from the specified source."""
    if source != "processed":
        raise ValueError("Only source='processed' is supported")
    return load_klines_1h_processed()


def _print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("Rows: 0")
        return

    print(f"Rows: {len(df):,}")
    print(f"Start: {df.index.min().isoformat()}")
    print(f"End: {df.index.max().isoformat()}")
    if "is_imputed" in df.columns:
        print(f"Imputed rows: {int(df['is_imputed'].sum())}")


def main() -> None:
    df = load_klines_1h_processed()
    _print_summary(df)


if __name__ == "__main__":
    main()
