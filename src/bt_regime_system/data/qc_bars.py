from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from bt_regime_system.utils.io import ensure_dir

OHLCV_COLS = ["open", "high", "low", "close", "volume"]
INPUT_COLUMNS = ["timestamp", *OHLCV_COLS]


def _standardize_bars_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    if "timestamp" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={"index": "timestamp"})
        else:
            raise ValueError("Bars frame must contain `timestamp` column or DatetimeIndex")

    missing = [c for c in INPUT_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Bars frame missing columns: {missing}")

    out = out[INPUT_COLUMNS]
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for col in OHLCV_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    return out.reset_index(drop=True)


def _valid_ohlcv_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame[OHLCV_COLS].notna().all(axis=1)
        & (frame["open"] > 0)
        & (frame["high"] > 0)
        & (frame["low"] > 0)
        & (frame["close"] > 0)
        & (frame["volume"] >= 0)
        & (frame["high"] >= frame[["open", "close", "low"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close", "high"]].min(axis=1))
    )


def _freq_to_offset(freq: str) -> str:
    normalized = freq.strip().lower()
    if normalized in {"15m", "15min"}:
        return "15min"
    if normalized in {"1h", "60min"}:
        return "1h"
    raise ValueError(f"Unsupported frequency: {freq}")


def build_bars_qc_report(bars: pd.DataFrame, freq: str) -> dict[str, Any]:
    standardized = _standardize_bars_frame(bars)
    freq_alias = _freq_to_offset(freq)

    duplicate_timestamp_count = int(standardized["timestamp"].duplicated().sum())
    invalid_ohlc_count = int((~_valid_ohlcv_mask(standardized)).sum()) if len(standardized) else 0
    negative_volume_count = int((standardized["volume"] < 0).sum()) if len(standardized) else 0

    if standardized.empty:
        min_timestamp: str | None = None
        max_timestamp: str | None = None
        missing_timestamp_count = 0
    else:
        ts = standardized["timestamp"]
        min_timestamp = ts.min().isoformat()
        max_timestamp = ts.max().isoformat()
        expected = pd.date_range(ts.min(), ts.max(), freq=freq_alias, tz="UTC")
        missing_timestamp_count = int(len(expected) - len(ts.drop_duplicates()))

    nan_count_by_column = {
        col: int(standardized[col].isna().sum()) if col in standardized.columns else 0
        for col in INPUT_COLUMNS
    }

    return {
        "frequency": freq_alias,
        "row_count": int(len(standardized)),
        "min_timestamp": min_timestamp,
        "max_timestamp": max_timestamp,
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "missing_timestamp_count": missing_timestamp_count,
        "nan_count_by_column": nan_count_by_column,
        "invalid_ohlc_count": invalid_ohlc_count,
        "negative_volume_count": negative_volume_count,
    }


def _extract_month(name: str) -> str | None:
    match = re.search(r"(\d{4}-\d{2})", name)
    return match.group(1) if match else None


def _report_filename(bars_file: Path, freq: str) -> str:
    month = _extract_month(bars_file.name) or "unknown"
    freq_alias = _freq_to_offset(freq)
    return f"qc_bars_{freq_alias}_{month}.json"


def _summary_filename(freq: str) -> str:
    freq_alias = _freq_to_offset(freq)
    return f"qc_bars_{freq_alias}_summary.json"


def write_qc_report(report: dict[str, Any], path: Path) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


def run_qc_bars(input_path: Path, freq: str, report_dir: Path) -> dict[str, Any]:
    """Run QC for bars parquet files and write monthly + summary reports."""
    ensure_dir(report_dir)

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob("*.parquet"))
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    freq_alias = _freq_to_offset(freq)
    monthly_reports: list[Path] = []
    combined_ts: list[pd.Series] = []

    rows_total = 0
    invalid_total = 0
    negative_volume_total = 0
    duplicate_in_file_total = 0
    missing_in_file_total = 0
    nan_totals = {col: 0 for col in INPUT_COLUMNS}

    for bars_file in files:
        bars = pd.read_parquet(bars_file)
        report = build_bars_qc_report(bars, freq=freq_alias)

        report_path = report_dir / _report_filename(bars_file, freq_alias)
        write_qc_report(report, report_path)
        monthly_reports.append(report_path)

        rows_total += int(report["row_count"])
        invalid_total += int(report["invalid_ohlc_count"])
        negative_volume_total += int(report["negative_volume_count"])
        duplicate_in_file_total += int(report["duplicate_timestamp_count"])
        missing_in_file_total += int(report["missing_timestamp_count"])
        for col, value in report["nan_count_by_column"].items():
            nan_totals[col] += int(value)

        standardized = _standardize_bars_frame(bars)
        if not standardized.empty:
            combined_ts.append(standardized["timestamp"])

    if combined_ts:
        global_ts = pd.concat(combined_ts, ignore_index=True)
        global_ts = pd.to_datetime(global_ts, utc=True).sort_values().reset_index(drop=True)
        global_duplicate_timestamp_count = int(global_ts.duplicated().sum())

        unique_ts = global_ts.drop_duplicates()
        expected = pd.date_range(unique_ts.min(), unique_ts.max(), freq=freq_alias, tz="UTC")
        global_missing_timestamp_count = int(len(expected) - len(unique_ts))

        min_timestamp = unique_ts.min().isoformat()
        max_timestamp = unique_ts.max().isoformat()
    else:
        global_duplicate_timestamp_count = 0
        global_missing_timestamp_count = 0
        min_timestamp = None
        max_timestamp = None

    summary = {
        "frequency": freq_alias,
        "files_processed": len(files),
        "row_count": rows_total,
        "min_timestamp": min_timestamp,
        "max_timestamp": max_timestamp,
        "duplicate_timestamp_count": duplicate_in_file_total,
        "missing_timestamp_count": missing_in_file_total,
        "nan_count_by_column": nan_totals,
        "invalid_ohlc_count": invalid_total,
        "negative_volume_count": negative_volume_total,
        "global_duplicate_timestamp_count": global_duplicate_timestamp_count,
        "global_missing_timestamp_count": global_missing_timestamp_count,
    }

    summary_path = report_dir / _summary_filename(freq_alias)
    write_qc_report(summary, summary_path)

    return {
        "frequency": freq_alias,
        "files_processed": len(files),
        "monthly_reports": monthly_reports,
        "summary_path": summary_path,
        "summary": summary,
    }
