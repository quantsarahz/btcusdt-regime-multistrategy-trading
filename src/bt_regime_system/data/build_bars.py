from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bt_regime_system.utils.io import ensure_dir

OHLCV = ["open", "high", "low", "close", "volume"]
INPUT_COLUMNS = ["timestamp", *OHLCV]
AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}
# Right-labeled bars at 00:00 belong to the previous minute's month.
PARTITION_OFFSET = pd.Timedelta(minutes=1)


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=INPUT_COLUMNS)


def _standardize_clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    if "timestamp" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={"index": "timestamp"})
        else:
            raise ValueError("Clean frame must contain `timestamp` column or DatetimeIndex")

    missing = [c for c in INPUT_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Clean frame missing columns: {missing}")

    out = out[INPUT_COLUMNS]
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"])
    for col in OHLCV:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=OHLCV)
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    for col in OHLCV:
        out[col] = out[col].astype(float)
    return out.reset_index(drop=True)


def resample_ohlcv(clean_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample clean 1m bars to higher timeframe OHLCV with right-close labeling."""
    standardized = _standardize_clean_frame(clean_1m)
    if standardized.empty:
        return _empty_bars()

    indexed = standardized.set_index("timestamp").sort_index()
    bars = indexed.resample(rule, closed="right", label="right").agg(AGG)
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    bars = bars.reset_index()

    bars = bars[["timestamp", *OHLCV]].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    for col in OHLCV:
        bars[col] = pd.to_numeric(bars[col], errors="coerce").astype(float)
    return bars


def build_15m_and_1h(clean_1m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build 15m and 1h bars from clean 1m data."""
    bars_15m = resample_ohlcv(clean_1m, "15min")
    bars_1h = resample_ohlcv(clean_1m, "1h")
    return bars_15m, bars_1h


def monthly_bar_filename(symbol: str, freq: str, month: str) -> str:
    return f"{symbol.upper()}_{freq}_{month}.parquet"


def write_monthly_bars(
    data: pd.DataFrame,
    out_dir: Path,
    symbol: str,
    freq: str,
) -> list[Path]:
    """Write/merge bars into monthly parquet files."""
    ensure_dir(out_dir)
    if data.empty:
        return []

    frame = _standardize_clean_frame(data)
    partition_ts = frame["timestamp"] - PARTITION_OFFSET
    frame["month"] = partition_ts.dt.strftime("%Y-%m")

    written: list[Path] = []
    for month, chunk in frame.groupby("month", sort=True):
        target = out_dir / monthly_bar_filename(symbol, freq, month)
        new_rows = chunk[INPUT_COLUMNS].sort_values("timestamp")

        if target.exists():
            existing = pd.read_parquet(target)
            existing = _standardize_clean_frame(existing)
            merged = pd.concat([existing, new_rows], ignore_index=True)
            merged = merged.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        else:
            merged = new_rows.drop_duplicates("timestamp", keep="last")

        merged.to_parquet(target, index=False)
        written.append(target)

    return written


def _collect_clean_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*.parquet"))
    raise FileNotFoundError(f"Input path not found: {input_path}")


def run_build_bars(
    input_path: Path,
    output_15m_dir: Path,
    output_1h_dir: Path,
    symbol: str = "BTCUSDT",
) -> dict[str, Any]:
    """Build bars from one file or a folder of clean 1m parquet files."""
    files = _collect_clean_files(input_path)
    if not files:
        return {
            "rows_in": 0,
            "rows_15m": 0,
            "rows_1h": 0,
            "files_15m": [],
            "files_1h": [],
        }

    frames: list[pd.DataFrame] = []
    rows_in = 0
    for file in files:
        df = pd.read_parquet(file)
        rows_in += len(df)
        frames.append(df)

    merged_clean = pd.concat(frames, ignore_index=True)
    bars_15m, bars_1h = build_15m_and_1h(merged_clean)

    files_15m = write_monthly_bars(bars_15m, out_dir=output_15m_dir, symbol=symbol, freq="15m")
    files_1h = write_monthly_bars(bars_1h, out_dir=output_1h_dir, symbol=symbol, freq="1h")

    return {
        "rows_in": rows_in,
        "rows_15m": len(bars_15m),
        "rows_1h": len(bars_1h),
        "files_15m": files_15m,
        "files_1h": files_1h,
    }
