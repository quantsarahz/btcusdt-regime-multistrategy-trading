from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from bt_regime_system.regime.detect import R1, R2, R3, R4
from bt_regime_system.utils.io import ensure_dir

REGIME_VALUES = (R1, R2, R3, R4)


def _extract_month(name: str) -> str | None:
    match = re.search(r"(\d{4}-\d{2})", name)
    return match.group(1) if match else None


def _collect_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob(pattern))
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _freq_alias(frame: str) -> str:
    normalized = frame.strip().lower()
    if normalized in {"1h", "1hour"}:
        return "1h"
    if normalized in {"15m", "15min"}:
        return "15min"
    raise ValueError(f"Unsupported regime frame: {frame}")


def _report_filename(frame: str, month: str) -> str:
    return f"qc_regime_{frame}_{month}.json"


def _summary_filename(frame: str) -> str:
    return f"qc_regime_{frame}_summary.json"


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _switch_stats(regime: pd.Series) -> tuple[int, float]:
    if regime.empty:
        return 0, 0.0
    s = regime.astype("string")
    if len(s) <= 1:
        return 0, 0.0
    switched = (s != s.shift(1)) & s.shift(1).notna() & s.notna()
    switch_count = int(switched.sum())
    switch_rate = float(switch_count / max(len(s) - 1, 1))
    return switch_count, switch_rate


def _empty_distribution() -> dict[str, int]:
    return {label: 0 for label in REGIME_VALUES}


def _distribution(regime: pd.Series) -> dict[str, int]:
    out = _empty_distribution()
    s = regime.astype("string")
    for label in REGIME_VALUES:
        out[label] = int((s == label).sum())
    return out


def _standardize_regime_1h(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    required = [
        "timestamp",
        "regime",
        "is_trend",
        "is_high_vol",
        "ema_fast",
        "ema_slow",
        "adx",
        "atrp",
        "vol_threshold",
    ]
    missing = set(required).difference(out.columns)
    if missing:
        raise ValueError(f"regime_1h frame missing columns: {sorted(missing)}")

    out = out[required].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["regime"] = out["regime"].astype("string")

    for col in ["is_trend", "is_high_vol"]:
        out[col] = out[col].astype("boolean")

    for col in ["ema_fast", "ema_slow", "adx", "atrp", "vol_threshold"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    return out.reset_index(drop=True)


def _standardize_regime_15m(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    required = {"timestamp", "regime", "regime_timestamp"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"regime_15m frame missing columns: {sorted(missing)}")

    out = out[["timestamp", "regime", "regime_timestamp"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["regime_timestamp"] = pd.to_datetime(out["regime_timestamp"], utc=True, errors="coerce")
    out["regime"] = out["regime"].astype("string")

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    return out.reset_index(drop=True)


def _base_report(
    standardized: pd.DataFrame,
    frame: str,
    freq: str,
    nan_columns: list[str],
) -> dict[str, Any]:
    duplicate_timestamp_count = int(standardized["timestamp"].duplicated().sum())

    if standardized.empty:
        min_timestamp: str | None = None
        max_timestamp: str | None = None
        missing_timestamp_count = 0
        switch_count = 0
        switch_rate = 0.0
        regime_distribution = _empty_distribution()
    else:
        deduped = standardized.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        ts = deduped["timestamp"]

        min_timestamp = ts.min().isoformat()
        max_timestamp = ts.max().isoformat()
        expected = pd.date_range(ts.min(), ts.max(), freq=freq, tz="UTC")
        missing_timestamp_count = int(len(expected) - len(ts))

        switch_count, switch_rate = _switch_stats(deduped["regime"])
        regime_distribution = _distribution(deduped["regime"])

    nan_count_by_column = {
        col: int(standardized[col].isna().sum()) if col in standardized.columns else 0 for col in nan_columns
    }

    invalid_regime_count = int((~standardized["regime"].isin(REGIME_VALUES)).fillna(True).sum()) if "regime" in standardized.columns else 0

    return {
        "frame": frame,
        "frequency": freq,
        "row_count": int(len(standardized)),
        "min_timestamp": min_timestamp,
        "max_timestamp": max_timestamp,
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "missing_timestamp_count": missing_timestamp_count,
        "nan_count_by_column": nan_count_by_column,
        "invalid_regime_count": invalid_regime_count,
        "regime_distribution": regime_distribution,
        "switch_count": switch_count,
        "switch_rate": switch_rate,
    }


def build_regime_1h_qc_report(regime_1h: pd.DataFrame) -> dict[str, Any]:
    standardized = _standardize_regime_1h(regime_1h)
    return _base_report(
        standardized=standardized,
        frame="1h",
        freq=_freq_alias("1h"),
        nan_columns=[
            "timestamp",
            "regime",
            "is_trend",
            "is_high_vol",
            "ema_fast",
            "ema_slow",
            "adx",
            "atrp",
            "vol_threshold",
        ],
    )


def _build_lookahead_report(
    regime_15m: pd.DataFrame,
    regime_lookup: pd.Series,
    first_regime_ts: pd.Timestamp | None,
    default_regime: str,
) -> dict[str, int]:
    if regime_15m.empty:
        return {
            "lookahead_violation_count": 0,
            "missing_regime_timestamp_count": 0,
            "unknown_regime_timestamp_count": 0,
            "regime_mismatch_count": 0,
            "default_before_first_count": 0,
            "non_default_before_first_count": 0,
        }

    deduped = regime_15m.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    lookahead_violation_count = int(
        ((deduped["regime_timestamp"].notna()) & (deduped["regime_timestamp"] > deduped["timestamp"]))
        .sum()
    )
    missing_regime_timestamp_count = int(deduped["regime_timestamp"].isna().sum())

    if regime_lookup.empty:
        unknown_regime_timestamp_count = int(deduped["regime_timestamp"].notna().sum())
        regime_mismatch_count = 0
    else:
        expected_regime = deduped["regime_timestamp"].map(regime_lookup)
        compare_mask = deduped["regime_timestamp"].notna()

        unknown_regime_timestamp_count = int(compare_mask.sum() - expected_regime.notna().sum())

        matched_mask = compare_mask & expected_regime.notna()
        regime_mismatch_count = int(
            (
                deduped.loc[matched_mask, "regime"].astype("string")
                != expected_regime.loc[matched_mask].astype("string")
            ).sum()
        )

    if first_regime_ts is None:
        before_first_mask = pd.Series(True, index=deduped.index)
    else:
        before_first_mask = deduped["timestamp"] < first_regime_ts

    regime_str = deduped["regime"].astype("string")
    default_before_first_count = int((before_first_mask & (regime_str == default_regime)).sum())
    non_default_before_first_count = int((before_first_mask & (regime_str != default_regime)).sum())

    return {
        "lookahead_violation_count": lookahead_violation_count,
        "missing_regime_timestamp_count": missing_regime_timestamp_count,
        "unknown_regime_timestamp_count": unknown_regime_timestamp_count,
        "regime_mismatch_count": regime_mismatch_count,
        "default_before_first_count": default_before_first_count,
        "non_default_before_first_count": non_default_before_first_count,
    }


def build_regime_15m_qc_report(
    regime_15m: pd.DataFrame,
    regime_lookup: pd.Series,
    first_regime_ts: pd.Timestamp | None,
    default_regime: str,
) -> dict[str, Any]:
    standardized = _standardize_regime_15m(regime_15m)
    report = _base_report(
        standardized=standardized,
        frame="15m",
        freq=_freq_alias("15m"),
        nan_columns=["timestamp", "regime", "regime_timestamp"],
    )
    report.update(
        _build_lookahead_report(
            regime_15m=standardized,
            regime_lookup=regime_lookup,
            first_regime_ts=first_regime_ts,
            default_regime=default_regime,
        )
    )
    return report


def _aggregate_nan_counts(reports: list[dict[str, Any]]) -> dict[str, int]:
    keys: set[str] = set()
    for report in reports:
        keys.update(report.get("nan_count_by_column", {}).keys())
    return {
        key: int(sum(int(report.get("nan_count_by_column", {}).get(key, 0)) for report in reports))
        for key in sorted(keys)
    }


def _aggregate_distribution(reports: list[dict[str, Any]]) -> dict[str, int]:
    totals = _empty_distribution()
    for report in reports:
        dist = report.get("regime_distribution", {})
        for label in REGIME_VALUES:
            totals[label] += int(dist.get(label, 0))
    return totals


def _build_summary(
    reports: list[dict[str, Any]],
    files_processed: int,
    global_frame: pd.DataFrame,
    frame: str,
    freq: str,
    regime_lookup: pd.Series,
    first_regime_ts: pd.Timestamp | None,
    default_regime: str,
) -> dict[str, Any]:
    row_count = int(sum(int(r.get("row_count", 0)) for r in reports))
    duplicate_timestamp_count = int(sum(int(r.get("duplicate_timestamp_count", 0)) for r in reports))
    missing_timestamp_count = int(sum(int(r.get("missing_timestamp_count", 0)) for r in reports))
    invalid_regime_count = int(sum(int(r.get("invalid_regime_count", 0)) for r in reports))
    switch_count = int(sum(int(r.get("switch_count", 0)) for r in reports))

    summary = {
        "frame": frame,
        "frequency": freq,
        "files_processed": files_processed,
        "row_count": row_count,
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "missing_timestamp_count": missing_timestamp_count,
        "nan_count_by_column": _aggregate_nan_counts(reports),
        "invalid_regime_count": invalid_regime_count,
        "regime_distribution": _aggregate_distribution(reports),
        "switch_count": switch_count,
        "switch_rate": float(switch_count / max(row_count - 1, 1)) if row_count > 0 else 0.0,
    }

    if global_frame.empty:
        summary.update(
            {
                "min_timestamp": None,
                "max_timestamp": None,
                "global_duplicate_timestamp_count": 0,
                "global_missing_timestamp_count": 0,
                "global_switch_count": 0,
                "global_switch_rate": 0.0,
            }
        )
        if frame == "15m":
            summary.update(
                {
                    "lookahead_violation_count": 0,
                    "missing_regime_timestamp_count": 0,
                    "unknown_regime_timestamp_count": 0,
                    "regime_mismatch_count": 0,
                    "default_before_first_count": 0,
                    "non_default_before_first_count": 0,
                }
            )
        return summary

    ordered = global_frame.sort_values("timestamp").reset_index(drop=True)
    global_duplicate_timestamp_count = int(ordered["timestamp"].duplicated().sum())

    deduped = ordered.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts = deduped["timestamp"]
    expected = pd.date_range(ts.min(), ts.max(), freq=freq, tz="UTC")
    global_missing_timestamp_count = int(len(expected) - len(ts))

    global_switch_count, global_switch_rate = _switch_stats(deduped["regime"])

    summary.update(
        {
            "min_timestamp": ts.min().isoformat(),
            "max_timestamp": ts.max().isoformat(),
            "global_duplicate_timestamp_count": global_duplicate_timestamp_count,
            "global_missing_timestamp_count": global_missing_timestamp_count,
            "global_switch_count": global_switch_count,
            "global_switch_rate": global_switch_rate,
        }
    )

    if frame == "15m":
        summary.update(
            _build_lookahead_report(
                regime_15m=deduped,
                regime_lookup=regime_lookup,
                first_regime_ts=first_regime_ts,
                default_regime=default_regime,
            )
        )

    return summary


def run_qc_regime(
    regime_1h_path: Path,
    regime_15m_path: Path,
    report_dir: Path,
    symbol: str = "BTCUSDT",
    default_regime: str = R4,
) -> dict[str, Any]:
    ensure_dir(report_dir)

    files_1h = _collect_files(regime_1h_path, f"{symbol.upper()}_regime_1h_*.parquet")
    files_15m = _collect_files(regime_15m_path, f"{symbol.upper()}_regime_15m_*.parquet")

    reports_1h: list[dict[str, Any]] = []
    monthly_1h: list[Path] = []
    global_1h_frames: list[pd.DataFrame] = []

    for file in files_1h:
        frame = pd.read_parquet(file)
        report = build_regime_1h_qc_report(frame)

        month = _extract_month(file.name) or "unknown"
        report_path = report_dir / _report_filename("1h", month)
        _write_json(report, report_path)

        reports_1h.append(report)
        monthly_1h.append(report_path)
        global_1h_frames.append(_standardize_regime_1h(frame))

    if global_1h_frames:
        global_1h = pd.concat(global_1h_frames, ignore_index=True)
        lookup_source = global_1h.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        regime_lookup = pd.Series(lookup_source["regime"].values, index=lookup_source["timestamp"]).astype("string")
        first_regime_ts = pd.to_datetime(lookup_source["timestamp"], utc=True).min()
    else:
        global_1h = pd.DataFrame(columns=["timestamp", "regime"])
        regime_lookup = pd.Series(dtype="string")
        first_regime_ts = None

    reports_15m: list[dict[str, Any]] = []
    monthly_15m: list[Path] = []
    global_15m_frames: list[pd.DataFrame] = []

    for file in files_15m:
        frame = pd.read_parquet(file)
        report = build_regime_15m_qc_report(
            regime_15m=frame,
            regime_lookup=regime_lookup,
            first_regime_ts=first_regime_ts,
            default_regime=default_regime,
        )

        month = _extract_month(file.name) or "unknown"
        report_path = report_dir / _report_filename("15m", month)
        _write_json(report, report_path)

        reports_15m.append(report)
        monthly_15m.append(report_path)
        global_15m_frames.append(_standardize_regime_15m(frame))

    global_15m = pd.concat(global_15m_frames, ignore_index=True) if global_15m_frames else pd.DataFrame(columns=["timestamp", "regime", "regime_timestamp"])

    summary_1h = _build_summary(
        reports=reports_1h,
        files_processed=len(files_1h),
        global_frame=global_1h,
        frame="1h",
        freq=_freq_alias("1h"),
        regime_lookup=pd.Series(dtype="string"),
        first_regime_ts=None,
        default_regime=default_regime,
    )

    summary_15m = _build_summary(
        reports=reports_15m,
        files_processed=len(files_15m),
        global_frame=global_15m,
        frame="15m",
        freq=_freq_alias("15m"),
        regime_lookup=regime_lookup,
        first_regime_ts=first_regime_ts,
        default_regime=default_regime,
    )

    summary_path_1h = report_dir / _summary_filename("1h")
    summary_path_15m = report_dir / _summary_filename("15m")
    _write_json(summary_1h, summary_path_1h)
    _write_json(summary_15m, summary_path_15m)

    return {
        "regime_1h": {
            "files_processed": len(files_1h),
            "monthly_reports": monthly_1h,
            "summary_path": summary_path_1h,
            "summary": summary_1h,
        },
        "regime_15m": {
            "files_processed": len(files_15m),
            "monthly_reports": monthly_15m,
            "summary_path": summary_path_15m,
            "summary": summary_15m,
        },
    }
