from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bt_regime_system.regime.detect import R4, monthly_regime_filename
from bt_regime_system.utils.io import ensure_dir

PARTITION_OFFSET = pd.Timedelta(minutes=1)


def _standardize_regime_1h(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    required = {"timestamp", "regime"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"regime_1h frame missing columns: {sorted(missing)}")

    out = out[["timestamp", "regime"]]
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates("timestamp", keep="last")
    out["regime"] = out["regime"].astype("string")
    return out.reset_index(drop=True)


def _standardize_bars_15m(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    required = {"timestamp"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"bars_15m frame missing columns: {sorted(missing)}")

    out = out[["timestamp"]]
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates("timestamp", keep="last")
    return out.reset_index(drop=True)


def align_regime_to_15m(
    regime_1h: pd.DataFrame,
    bars_15m: pd.DataFrame,
    default_regime: str = R4,
) -> pd.DataFrame:
    """Align completed 1h regime labels to 15m timestamps without lookahead."""
    regime_std = _standardize_regime_1h(regime_1h)
    bars_std = _standardize_bars_15m(bars_15m)

    if bars_std.empty:
        return pd.DataFrame(columns=["timestamp", "regime", "regime_timestamp"])

    if regime_std.empty:
        out = bars_std.copy()
        out["regime"] = default_regime
        out["regime_timestamp"] = pd.NaT
        return out[["timestamp", "regime", "regime_timestamp"]]

    left = bars_std.rename(columns={"timestamp": "timestamp"})
    right = regime_std.rename(columns={"timestamp": "regime_timestamp"})

    aligned = pd.merge_asof(
        left.sort_values("timestamp"),
        right.sort_values("regime_timestamp"),
        left_on="timestamp",
        right_on="regime_timestamp",
        direction="backward",
    )

    aligned["regime"] = aligned["regime"].fillna(default_regime)
    return aligned[["timestamp", "regime", "regime_timestamp"]].reset_index(drop=True)


def write_monthly_regime_15m(
    aligned_15m: pd.DataFrame,
    out_dir: Path,
    symbol: str,
) -> list[Path]:
    ensure_dir(out_dir)
    if aligned_15m.empty:
        return []

    required = {"timestamp", "regime", "regime_timestamp"}
    missing = required.difference(aligned_15m.columns)
    if missing:
        raise ValueError(f"aligned_15m missing columns: {sorted(missing)}")

    frame = aligned_15m.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["regime_timestamp"] = pd.to_datetime(frame["regime_timestamp"], utc=True, errors="coerce")

    partition_ts = frame["timestamp"] - PARTITION_OFFSET
    frame["month"] = partition_ts.dt.strftime("%Y-%m")

    written: list[Path] = []
    for month, chunk in frame.groupby("month", sort=True):
        target = out_dir / monthly_regime_filename(symbol=symbol, frame="15m", month=month)
        new_rows = chunk.drop(columns=["month"]).sort_values("timestamp")

        if target.exists():
            existing = pd.read_parquet(target)
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
            existing["regime_timestamp"] = pd.to_datetime(existing["regime_timestamp"], utc=True, errors="coerce")
            merged = pd.concat([existing, new_rows], ignore_index=True)
            merged = merged.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        else:
            merged = new_rows.drop_duplicates("timestamp", keep="last")

        merged.to_parquet(target, index=False)
        written.append(target)

    return written


def _collect_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob(pattern))
    raise FileNotFoundError(f"Input path not found: {input_path}")


def run_align_regime(
    regime_1h_path: Path,
    bars_15m_path: Path,
    output_dir: Path,
    symbol: str = "BTCUSDT",
    default_regime: str = R4,
) -> dict[str, Any]:
    regime_files = _collect_files(regime_1h_path, f"{symbol.upper()}_regime_1h_*.parquet")
    bars_files = _collect_files(bars_15m_path, f"{symbol.upper()}_15m_*.parquet")

    if not bars_files:
        return {
            "rows_regime_1h": 0,
            "rows_bars_15m": 0,
            "rows_aligned": 0,
            "files_written": [],
        }

    regime_frames: list[pd.DataFrame] = []
    rows_regime_1h = 0
    for file in regime_files:
        df = pd.read_parquet(file)
        rows_regime_1h += len(df)
        regime_frames.append(df)
    regime_1h = pd.concat(regime_frames, ignore_index=True) if regime_frames else pd.DataFrame(columns=["timestamp", "regime"])

    bars_frames: list[pd.DataFrame] = []
    rows_bars_15m = 0
    for file in bars_files:
        df = pd.read_parquet(file)
        rows_bars_15m += len(df)
        bars_frames.append(df)
    bars_15m = pd.concat(bars_frames, ignore_index=True)

    aligned = align_regime_to_15m(regime_1h=regime_1h, bars_15m=bars_15m, default_regime=default_regime)
    files_written = write_monthly_regime_15m(aligned_15m=aligned, out_dir=output_dir, symbol=symbol)

    return {
        "rows_regime_1h": rows_regime_1h,
        "rows_bars_15m": rows_bars_15m,
        "rows_aligned": int(len(aligned)),
        "files_written": files_written,
    }
