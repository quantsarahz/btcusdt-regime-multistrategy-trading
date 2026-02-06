"""Local dataset loader for BTCUSDT 1h klines."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yaml

from btcusdt_regime_trading.utils.paths import PROJECT_ROOT, RAW_DATA_DIR


CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
REQUIRED_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "num_trades",
}


def load_config() -> dict:
    """Load YAML config from the project configs directory."""
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _iter_years(start_dt: datetime, end_dt: datetime) -> Iterable[int]:
    for year in range(start_dt.year, end_dt.year + 1):
        yield year


def get_raw_klines_path(
    symbol: str,
    interval: str,
    year: int,
    exchange_subdir: str = "binance_spot",
) -> Path:
    """Return the raw kline parquet path for a specific year."""
    return RAW_DATA_DIR / exchange_subdir / symbol / interval / f"{year}.parquet"


def _load_year(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "open_time" in df.columns:
        df = df.set_index("open_time", drop=False)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Index is not DatetimeIndex for {path}")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def load_klines_1h(
    symbol: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    exchange_subdir: str = "binance_spot",
) -> pd.DataFrame:
    """Load 1h klines from local yearly parquet files (raw, no imputation)."""
    cfg = load_config()
    venue = cfg.get("venue", {})
    time_range = cfg.get("time_range", {})

    symbol = symbol or venue.get("symbol")
    if not symbol:
        raise ValueError("symbol is required")

    start_val = start or time_range.get("start")
    end_val = end or time_range.get("end")
    if not start_val or not end_val:
        raise ValueError("start and end are required")

    start_dt = _parse_datetime(start_val)
    end_dt = _parse_datetime(end_val)

    frames = []
    for year in _iter_years(start_dt, end_dt):
        path = get_raw_klines_path(symbol, "1h", year, exchange_subdir=exchange_subdir)
        if not path.exists():
            continue
        frames.append(_load_year(path))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=0)

    if "open_time" in df.columns:
        df = df.set_index("open_time", drop=False)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index is not DatetimeIndex")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]

    if not df.index.is_monotonic_increasing:
        raise ValueError("Index is not monotonic increasing after sort")

    return df


def _format_ts(ts: pd.Timestamp) -> str:
    return ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts)


def _print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("Rows: 0")
        return

    print(f"Rows: {len(df):,}")
    print(f"Start: {_format_ts(df.index.min())}")
    print(f"End: {_format_ts(df.index.max())}")
    print("Head:")
    print(df.head(3))
    print("Tail:")
    print(df.tail(3))


def main() -> None:
    df = load_klines_1h()
    _print_summary(df)


if __name__ == "__main__":
    main()
