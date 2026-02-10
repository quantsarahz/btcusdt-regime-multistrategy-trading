from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from bt_regime_system.utils.io import ensure_dir

OHLCV_COLS = ["open", "high", "low", "close", "volume"]
INPUT_COLUMNS = ["timestamp", *OHLCV_COLS]
CLEAN_COLUMNS = [*INPUT_COLUMNS, "is_synthetic"]


def _empty_clean_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CLEAN_COLUMNS)


def _standardize_raw_frame(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    if "timestamp" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={"index": "timestamp"})
        else:
            raise ValueError("Raw frame must contain `timestamp` column or DatetimeIndex")

    missing = [c for c in INPUT_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Raw frame missing columns: {missing}")

    out = out[INPUT_COLUMNS]
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for col in OHLCV_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out.dropna(subset=["timestamp"])


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


def clean_1m_ohlcv(raw: pd.DataFrame, fill_missing: bool = True) -> pd.DataFrame:
    """Clean raw 1m OHLCV bars into a contract-compliant frame."""
    standardized = _standardize_raw_frame(raw)
    if standardized.empty:
        return _empty_clean_frame()

    deduped = standardized.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    valid = deduped.loc[_valid_ohlcv_mask(deduped), INPUT_COLUMNS].copy()
    if valid.empty:
        return _empty_clean_frame()

    valid["is_synthetic"] = 0
    if fill_missing:
        indexed = valid.set_index("timestamp")
        full_index = pd.date_range(
            start=indexed.index.min(),
            end=indexed.index.max(),
            freq="1min",
            tz="UTC",
        )
        reindexed = indexed.reindex(full_index)

        synthetic_mask = reindexed["close"].isna()
        reindexed["close"] = reindexed["close"].ffill()
        for col in ["open", "high", "low"]:
            reindexed[col] = reindexed[col].fillna(reindexed["close"])
        reindexed["volume"] = reindexed["volume"].fillna(0.0)
        reindexed["is_synthetic"] = synthetic_mask.astype("int8")

        reindexed = reindexed.dropna(subset=["open", "high", "low", "close"]).copy()
        cleaned = reindexed.reset_index().rename(columns={"index": "timestamp"})
    else:
        cleaned = valid

    cleaned = cleaned[CLEAN_COLUMNS].sort_values("timestamp").reset_index(drop=True)
    for col in OHLCV_COLS:
        cleaned[col] = cleaned[col].astype(float)
    cleaned["is_synthetic"] = cleaned["is_synthetic"].astype("int8")
    return cleaned


def build_qc_report(raw: pd.DataFrame, clean: pd.DataFrame) -> dict[str, Any]:
    """Build a JSON-serializable quality report for raw->clean conversion."""
    standardized = _standardize_raw_frame(raw)

    duplicate_timestamp_count = int(standardized["timestamp"].duplicated().sum())
    invalid_ohlc_count = int((~_valid_ohlcv_mask(standardized)).sum()) if len(standardized) else 0
    negative_volume_count = int((standardized["volume"] < 0).sum()) if len(standardized) else 0

    if clean.empty:
        min_timestamp: str | None = None
        max_timestamp: str | None = None
        missing_timestamp_count = 0
    else:
        clean_ts = pd.to_datetime(clean["timestamp"], utc=True)
        min_timestamp = clean_ts.min().isoformat()
        max_timestamp = clean_ts.max().isoformat()
        expected = pd.date_range(clean_ts.min(), clean_ts.max(), freq="1min", tz="UTC")
        missing_timestamp_count = int(len(expected) - len(clean))

    nan_count_by_column = {
        col: int(clean[col].isna().sum()) if col in clean.columns else 0 for col in CLEAN_COLUMNS
    }

    return {
        "row_count": int(len(clean)),
        "min_timestamp": min_timestamp,
        "max_timestamp": max_timestamp,
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "missing_timestamp_count": missing_timestamp_count,
        "nan_count_by_column": nan_count_by_column,
        "invalid_ohlc_count": invalid_ohlc_count,
        "negative_volume_count": negative_volume_count,
        "synthetic_row_count": int(clean["is_synthetic"].sum()) if "is_synthetic" in clean.columns else 0,
    }


def _extract_month(name: str) -> str | None:
    match = re.search(r"(\d{4}-\d{2})", name)
    return match.group(1) if match else None


def _clean_filename(raw_file: Path) -> str:
    if "_1m_" in raw_file.name:
        return raw_file.name.replace("_1m_", "_1m_clean_", 1)
    return f"{raw_file.stem}_clean.parquet"


def _report_filename(raw_file: Path, clean: pd.DataFrame) -> str:
    month = _extract_month(raw_file.name)
    if month is None and not clean.empty:
        month = pd.to_datetime(clean["timestamp"], utc=True).iloc[0].strftime("%Y-%m")
    if month is None:
        month = "unknown"
    return f"qc_clean_1m_{month}.json"


def write_qc_report(report: dict[str, Any], path: Path) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


def run_qc_1m(
    input_path: Path,
    output_dir: Path,
    report_dir: Path,
    fill_missing: bool = True,
) -> list[dict[str, Any]]:
    """Run QC for one raw parquet file or all parquet files in a directory."""
    ensure_dir(output_dir)
    ensure_dir(report_dir)

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob("*.parquet"))
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    summaries: list[dict[str, Any]] = []
    for raw_file in files:
        raw = pd.read_parquet(raw_file)
        clean = clean_1m_ohlcv(raw, fill_missing=fill_missing)

        clean_path = output_dir / _clean_filename(raw_file)
        clean.to_parquet(clean_path, index=False)

        report = build_qc_report(raw=raw, clean=clean)
        report_path = report_dir / _report_filename(raw_file, clean)
        write_qc_report(report, report_path)

        summaries.append(
            {
                "raw_file": raw_file,
                "clean_file": clean_path,
                "report_file": report_path,
                "rows_in": int(len(raw)),
                "rows_out": int(len(clean)),
            }
        )

    return summaries
