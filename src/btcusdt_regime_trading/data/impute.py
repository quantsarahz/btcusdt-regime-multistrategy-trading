"""Impute missing 1h BTCUSDT klines and write processed files."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from btcusdt_regime_trading.data.dataset import load_config, load_klines_1h
from btcusdt_regime_trading.utils.paths import PROCESSED_DATA_DIR


NUMERIC_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


def _parse_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _write_by_year(df: pd.DataFrame, out_dir: Path, overwrite: bool) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for year, group in df.groupby(df.index.year):
        path = out_dir / f"{year}.parquet"
        if path.exists() and not overwrite:
            continue
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        group.to_parquet(tmp_path, index=True)
        tmp_path.replace(path)
        saved.append(path)
    return saved


def impute_klines_1h(
    symbol: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    exchange_subdir: str = "binance_spot",
    out_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict[str, object]:
    """Impute missing 1h bars, keeping a flag column, and write processed files."""
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

    df = load_klines_1h(symbol=symbol, start=start_val, end=end_val, exchange_subdir=exchange_subdir)
    if df.empty:
        return {
            "summary": {
                "n_rows": 0,
                "imputed_rows": 0,
                "start_ts": None,
                "end_ts": None,
            },
            "saved": [],
        }

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    actual_start = df.index.min()
    actual_end = df.index.max()
    adj_start = max(start_dt, actual_start)
    adj_end = min(end_dt, actual_end)
    if adj_start > adj_end:
        return {
            "summary": {
                "n_rows": 0,
                "imputed_rows": 0,
                "start_ts": None,
                "end_ts": None,
            },
            "saved": [],
        }

    df = df.loc[(df.index >= adj_start) & (df.index <= adj_end)].copy()

    expected_idx = pd.date_range(start=adj_start, end=adj_end, freq="1H", tz="UTC")
    missing_idx = expected_idx.difference(df.index)

    df = df.reindex(expected_idx)
    df["open_time"] = df.index
    df["is_imputed"] = df.index.isin(missing_idx)

    imputed_mask = df["is_imputed"]
    if imputed_mask.any():
        prev_close = df["close"].ffill()
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df.loc[imputed_mask, col] = prev_close.loc[imputed_mask]

        for col in ["volume", "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]:
            if col in df.columns:
                df.loc[imputed_mask, col] = 0.0
        if "num_trades" in df.columns:
            df.loc[imputed_mask, "num_trades"] = 0

        if "close_time" in df.columns:
            df.loc[imputed_mask, "close_time"] = (
                df.loc[imputed_mask].index + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
            )

    if "symbol" in df.columns:
        df.loc[:, "symbol"] = df["symbol"].fillna(symbol)
    else:
        df["symbol"] = symbol

    if "interval" in df.columns:
        df.loc[:, "interval"] = df["interval"].fillna("1h")
    else:
        df["interval"] = "1h"

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_index()

    output_base = Path(out_dir) if out_dir is not None else PROCESSED_DATA_DIR
    out_path = output_base / exchange_subdir / symbol / "1h"
    saved = _write_by_year(df, out_path, overwrite=overwrite)

    summary = {
        "n_rows": int(len(df)),
        "imputed_rows": int(imputed_mask.sum()),
        "start_ts": adj_start.isoformat(),
        "end_ts": adj_end.isoformat(),
        "missing_count": int(len(missing_idx)),
    }

    return {"summary": summary, "saved": saved}


def _print_summary(result: Dict[str, object]) -> None:
    summary = result.get("summary", {})
    print("Imputation Summary")
    print(f"  n_rows: {summary.get('n_rows')}")
    print(f"  imputed_rows: {summary.get('imputed_rows')}")
    print(f"  start_ts: {summary.get('start_ts')}")
    print(f"  end_ts: {summary.get('end_ts')}")
    print(f"  missing_count: {summary.get('missing_count')}")
    saved = result.get("saved", [])
    print("Saved")
    for path in saved:
        print(f"  {path}")


def main() -> None:
    result = impute_klines_1h()
    _print_summary(result)


if __name__ == "__main__":
    main()
