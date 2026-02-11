from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bt_regime_system.features.indicators import build_regime_features_1h
from bt_regime_system.utils.io import ensure_dir

R1 = "R1"  # Trend + LowVol
R2 = "R2"  # Trend + HighVol
R3 = "R3"  # Range + LowVol
R4 = "R4"  # Range + HighVol
REGIME_VALUES = (R1, R2, R3, R4)
PARTITION_OFFSET = pd.Timedelta(minutes=1)


def _standardize_bars_1h(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"bars_1h missing columns: {sorted(missing)}")

    out = out[["timestamp", "open", "high", "low", "close", "volume"]]
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates("timestamp", keep="last")

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    return out.reset_index(drop=True)


def classify_regime(is_trend: pd.Series, is_high_vol: pd.Series) -> pd.Series:
    trend = is_trend.fillna(False).astype(bool)
    high_vol = is_high_vol.fillna(False).astype(bool)

    regime = pd.Series(R3, index=trend.index, dtype="string")
    regime.loc[trend & ~high_vol] = R1
    regime.loc[trend & high_vol] = R2
    regime.loc[~trend & ~high_vol] = R3
    regime.loc[~trend & high_vol] = R4
    return regime


def apply_min_regime_run_filter(
    regime: pd.Series,
    min_run_bars: int = 1,
    default_regime: str = R4,
) -> pd.Series:
    """Confirm regime switch only after `min_run_bars` consecutive new labels."""
    if min_run_bars < 1:
        raise ValueError("min_run_bars must be >= 1")

    out = regime.fillna(default_regime).astype("string")
    if out.empty or min_run_bars == 1:
        return out

    values = out.astype(str).tolist()
    filtered: list[str] = []

    current = values[0]
    pending: str | None = None
    pending_count = 0

    for value in values:
        if value == current:
            pending = None
            pending_count = 0
        else:
            if pending != value:
                pending = value
                pending_count = 1
            else:
                pending_count += 1

            if pending_count >= min_run_bars:
                current = pending
                pending = None
                pending_count = 0

        filtered.append(current)

    return pd.Series(filtered, index=out.index, dtype="string")


def detect_regime_1h(
    bars_1h: pd.DataFrame,
    ema_fast: int = 24,
    ema_slow: int = 96,
    adx_window: int = 14,
    adx_threshold: float = 20.0,
    ema_gap_threshold: float = 0.0,
    atr_window: int = 14,
    vol_lookback: int = 720,
    high_vol_quantile: float = 0.75,
    min_regime_run_bars: int = 1,
) -> pd.DataFrame:
    standardized = _standardize_bars_1h(bars_1h)
    if standardized.empty:
        return pd.DataFrame(
            columns=[
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
        )

    features = build_regime_features_1h(
        standardized,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        adx_window=adx_window,
        adx_threshold=adx_threshold,
        ema_gap_threshold=ema_gap_threshold,
        atr_window=atr_window,
        vol_lookback=vol_lookback,
        high_vol_quantile=high_vol_quantile,
    )

    regime_raw = classify_regime(features["is_trend"], features["is_high_vol"])
    features["regime"] = apply_min_regime_run_filter(
        regime_raw,
        min_run_bars=min_regime_run_bars,
        default_regime=R4,
    )

    # Keep derived boolean flags consistent with final smoothed regime labels.
    features["is_trend"] = features["regime"].isin([R1, R2])
    features["is_high_vol"] = features["regime"].isin([R2, R4])

    return features[
        [
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
    ].copy()


def monthly_regime_filename(symbol: str, frame: str, month: str) -> str:
    return f"{symbol.upper()}_regime_{frame}_{month}.parquet"


def write_monthly_regime(
    regime_data: pd.DataFrame,
    out_dir: Path,
    symbol: str,
    frame: str,
) -> list[Path]:
    ensure_dir(out_dir)
    if regime_data.empty:
        return []

    required = {
        "timestamp",
        "regime",
        "is_trend",
        "is_high_vol",
        "ema_fast",
        "ema_slow",
        "adx",
        "atrp",
        "vol_threshold",
    }
    missing = required.difference(regime_data.columns)
    if missing:
        raise ValueError(f"regime_data missing columns: {sorted(missing)}")

    frame_data = regime_data.copy()
    frame_data["timestamp"] = pd.to_datetime(frame_data["timestamp"], utc=True)
    partition_ts = frame_data["timestamp"] - PARTITION_OFFSET
    frame_data["month"] = partition_ts.dt.strftime("%Y-%m")

    written: list[Path] = []
    for month, chunk in frame_data.groupby("month", sort=True):
        target = out_dir / monthly_regime_filename(symbol=symbol, frame=frame, month=month)
        new_rows = chunk.drop(columns=["month"]).sort_values("timestamp")

        if target.exists():
            existing = pd.read_parquet(target)
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
            merged = pd.concat([existing, new_rows], ignore_index=True)
            merged = merged.sort_values("timestamp", kind="mergesort").drop_duplicates("timestamp", keep="last")
        else:
            merged = new_rows.drop_duplicates("timestamp", keep="last")

        merged.to_parquet(target, index=False)
        written.append(target)

    return written


def _collect_1h_files(input_path: Path, symbol: str = "BTCUSDT") -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob(f"{symbol.upper()}_1h_*.parquet"))
    raise FileNotFoundError(f"Input path not found: {input_path}")


def run_detect_regime(
    input_path: Path,
    output_dir: Path,
    symbol: str = "BTCUSDT",
    ema_fast: int = 24,
    ema_slow: int = 96,
    adx_window: int = 14,
    adx_threshold: float = 20.0,
    ema_gap_threshold: float = 0.0,
    atr_window: int = 14,
    vol_lookback: int = 720,
    high_vol_quantile: float = 0.75,
    min_regime_run_bars: int = 1,
) -> dict[str, Any]:
    files = _collect_1h_files(input_path, symbol=symbol)
    if not files:
        return {
            "rows_in": 0,
            "rows_out": 0,
            "files_written": [],
            "regime_counts": {R1: 0, R2: 0, R3: 0, R4: 0},
            "min_regime_run_bars": int(min_regime_run_bars),
            "switch_count": 0,
        }

    rows_in = 0
    frames: list[pd.DataFrame] = []
    for file in files:
        df = pd.read_parquet(file)
        rows_in += len(df)
        frames.append(df)

    bars_1h = pd.concat(frames, ignore_index=True)
    regime_1h = detect_regime_1h(
        bars_1h,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        adx_window=adx_window,
        adx_threshold=adx_threshold,
        ema_gap_threshold=ema_gap_threshold,
        atr_window=atr_window,
        vol_lookback=vol_lookback,
        high_vol_quantile=high_vol_quantile,
        min_regime_run_bars=min_regime_run_bars,
    )

    files_written = write_monthly_regime(regime_1h, out_dir=output_dir, symbol=symbol, frame="1h")
    regime_counts = regime_1h["regime"].value_counts().to_dict() if not regime_1h.empty else {}
    switch_count = int((regime_1h["regime"] != regime_1h["regime"].shift(1)).sum() - 1) if len(regime_1h) else 0

    return {
        "rows_in": rows_in,
        "rows_out": int(len(regime_1h)),
        "files_written": files_written,
        "regime_counts": {
            R1: int(regime_counts.get(R1, 0)),
            R2: int(regime_counts.get(R2, 0)),
            R3: int(regime_counts.get(R3, 0)),
            R4: int(regime_counts.get(R4, 0)),
        },
        "min_regime_run_bars": int(min_regime_run_bars),
        "switch_count": switch_count,
    }
