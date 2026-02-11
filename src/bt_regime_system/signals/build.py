from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bt_regime_system.regime.allocate import build_allocation_table
from bt_regime_system.regime.detect import R4
from bt_regime_system.strategies.donchian import SIGNAL_COLUMN as DONCHIAN_COL
from bt_regime_system.strategies.donchian import generate_signal as generate_donchian_signal
from bt_regime_system.strategies.ema_adx import SIGNAL_COLUMN as EMA_ADX_COL
from bt_regime_system.strategies.ema_adx import generate_signal as generate_ema_adx_signal
from bt_regime_system.strategies.mean_reversion import SIGNAL_COLUMN as MEAN_REV_COL
from bt_regime_system.strategies.mean_reversion import generate_signal as generate_mean_reversion_signal
from bt_regime_system.utils.io import ensure_dir

PARTITION_OFFSET = pd.Timedelta(minutes=1)
SIGNAL_COLUMNS = [
    "timestamp",
    DONCHIAN_COL,
    EMA_ADX_COL,
    MEAN_REV_COL,
    "signal_composite",
    "target_position",
]


def _standardize_bars_15m(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    required = ["timestamp", "high", "low", "close"]
    missing = set(required).difference(out.columns)
    if missing:
        raise ValueError(f"bars_15m missing columns: {sorted(missing)}")

    out = out[required].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for col in ["high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["timestamp", "high", "low", "close"])
    out = out.sort_values("timestamp", kind="mergesort").drop_duplicates("timestamp", keep="last")
    return out.reset_index(drop=True)


def _standardize_regime_15m(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    required = ["timestamp", "regime"]
    missing = set(required).difference(out.columns)
    if missing:
        raise ValueError(f"regime_15m missing columns: {sorted(missing)}")

    out = out[required].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["regime"] = out["regime"].astype("string")

    out = out.dropna(subset=["timestamp"])
    out = out.sort_values("timestamp", kind="mergesort").drop_duplicates("timestamp", keep="last")
    return out.reset_index(drop=True)


def _collect_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob(pattern))
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _merge_signals_on_timestamp(base_ts: pd.Series, *signal_frames: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": pd.to_datetime(base_ts, utc=True)})
    for frame in signal_frames:
        out = out.merge(frame, on="timestamp", how="left")
    return out


def _resolve_regime_for_bars(
    bars_15m: pd.DataFrame,
    regime_15m: pd.DataFrame,
    default_regime: str,
) -> pd.Series:
    if regime_15m.empty:
        return pd.Series(default_regime, index=bars_15m.index, dtype="string")

    merged = bars_15m[["timestamp"]].merge(
        regime_15m[["timestamp", "regime"]],
        on="timestamp",
        how="left",
    )
    return merged["regime"].astype("string").fillna(default_regime)


def _apply_execution_constraints(
    target_position: pd.Series,
    long_only: bool = False,
    min_hold_bars: int = 1,
    rebalance_threshold: float = 0.0,
) -> pd.Series:
    if min_hold_bars < 1:
        raise ValueError("min_hold_bars must be >= 1")
    if rebalance_threshold < 0:
        raise ValueError("rebalance_threshold must be >= 0")

    desired = pd.to_numeric(target_position, errors="coerce").fillna(0.0).clip(-1.0, 1.0).astype(float)
    if long_only:
        desired = desired.clip(lower=0.0)

    if desired.empty:
        return desired

    out: list[float] = []
    current = float(desired.iloc[0])
    bars_since_change = min_hold_bars
    out.append(current)

    for value in desired.iloc[1:]:
        proposed = float(value)

        if abs(proposed - current) < rebalance_threshold:
            proposed = current

        if proposed != current and bars_since_change < min_hold_bars:
            proposed = current

        if proposed != current:
            current = proposed
            bars_since_change = 1
        else:
            bars_since_change += 1

        out.append(current)

    return pd.Series(out, index=desired.index, dtype="float64")


def build_signals_15m(
    bars_15m: pd.DataFrame,
    regime_15m: pd.DataFrame,
    default_regime: str = R4,
    donchian_window: int = 20,
    donchian_hold_until_opposite: bool = True,
    ema_fast: int = 21,
    ema_slow: int = 55,
    ema_adx_window: int = 14,
    ema_adx_threshold: float = 20.0,
    ema_use_adx_filter: bool = True,
    mr_z_window: int = 48,
    mr_entry_z: float = 1.5,
    mr_exit_z: float = 0.5,
    regime_allocations: dict[str, dict[str, float]] | None = None,
    long_only: bool = False,
    min_hold_bars: int = 1,
    rebalance_threshold: float = 0.0,
) -> pd.DataFrame:
    bars_std = _standardize_bars_15m(bars_15m)
    if bars_std.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    regime_std = (
        _standardize_regime_15m(regime_15m)
        if not regime_15m.empty
        else pd.DataFrame(columns=["timestamp", "regime"])
    )

    signal_d = generate_donchian_signal(
        bars_std,
        window=donchian_window,
        hold_until_opposite=donchian_hold_until_opposite,
    )
    signal_e = generate_ema_adx_signal(
        bars_std,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        adx_window=ema_adx_window,
        adx_threshold=ema_adx_threshold,
        use_adx_filter=ema_use_adx_filter,
    )
    signal_m = generate_mean_reversion_signal(
        bars_std,
        z_window=mr_z_window,
        entry_z=mr_entry_z,
        exit_z=mr_exit_z,
    )

    merged = _merge_signals_on_timestamp(
        bars_std["timestamp"],
        signal_d,
        signal_e,
        signal_m,
    )

    for col in [DONCHIAN_COL, EMA_ADX_COL, MEAN_REV_COL]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0).clip(-1.0, 1.0)

    regime_series = _resolve_regime_for_bars(
        bars_15m=bars_std,
        regime_15m=regime_std,
        default_regime=default_regime,
    )

    alloc = build_allocation_table(
        pd.Series(regime_series.values, index=bars_std["timestamp"].values),
        regime_allocations=regime_allocations,
        default_regime=default_regime,
    )
    alloc["timestamp"] = pd.to_datetime(alloc["timestamp"], utc=True)

    merged = merged.merge(
        alloc[["timestamp", "w_donchian", "w_ema_adx", "w_mean_reversion"]],
        on="timestamp",
        how="left",
    )
    for col in ["w_donchian", "w_ema_adx", "w_mean_reversion"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    merged["signal_composite"] = (
        merged[DONCHIAN_COL] * merged["w_donchian"]
        + merged[EMA_ADX_COL] * merged["w_ema_adx"]
        + merged[MEAN_REV_COL] * merged["w_mean_reversion"]
    ).clip(-1.0, 1.0)

    merged["target_position"] = _apply_execution_constraints(
        merged["signal_composite"],
        long_only=long_only,
        min_hold_bars=min_hold_bars,
        rebalance_threshold=rebalance_threshold,
    )

    out = merged[SIGNAL_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    for col in [DONCHIAN_COL, EMA_ADX_COL, MEAN_REV_COL, "signal_composite", "target_position"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(float)

    return out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def monthly_signals_filename(symbol: str, month: str) -> str:
    return f"{symbol.upper()}_signals_15m_{month}.parquet"


def _standardize_signals_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    missing = set(SIGNAL_COLUMNS).difference(out.columns)
    if missing:
        raise ValueError(f"signals_15m missing columns: {sorted(missing)}")

    out = out[SIGNAL_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for col in [DONCHIAN_COL, EMA_ADX_COL, MEAN_REV_COL, "signal_composite", "target_position"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["timestamp"])
    out = out.sort_values("timestamp", kind="mergesort").drop_duplicates("timestamp", keep="last")

    for col in [DONCHIAN_COL, EMA_ADX_COL, MEAN_REV_COL, "signal_composite", "target_position"]:
        out[col] = out[col].fillna(0.0).astype(float)
    return out.reset_index(drop=True)


def write_monthly_signals(
    signals_15m: pd.DataFrame,
    out_dir: Path,
    symbol: str,
) -> list[Path]:
    ensure_dir(out_dir)
    if signals_15m.empty:
        return []

    frame = _standardize_signals_frame(signals_15m)
    partition_ts = frame["timestamp"] - PARTITION_OFFSET
    frame["month"] = partition_ts.dt.strftime("%Y-%m")

    written: list[Path] = []
    for month, chunk in frame.groupby("month", sort=True):
        target = out_dir / monthly_signals_filename(symbol=symbol, month=month)
        new_rows = chunk.drop(columns=["month"]).sort_values("timestamp")

        if target.exists():
            existing = pd.read_parquet(target)
            existing = _standardize_signals_frame(existing)
            merged = pd.concat([existing, new_rows], ignore_index=True)
            merged = merged.sort_values("timestamp", kind="mergesort").drop_duplicates("timestamp", keep="last")
        else:
            merged = new_rows.drop_duplicates("timestamp", keep="last")

        merged.to_parquet(target, index=False)
        written.append(target)

    return written


def run_generate_signals(
    bars_15m_path: Path,
    regime_15m_path: Path,
    output_dir: Path,
    symbol: str = "BTCUSDT",
    default_regime: str = R4,
    donchian_window: int = 20,
    donchian_hold_until_opposite: bool = True,
    ema_fast: int = 21,
    ema_slow: int = 55,
    ema_adx_window: int = 14,
    ema_adx_threshold: float = 20.0,
    ema_use_adx_filter: bool = True,
    mr_z_window: int = 48,
    mr_entry_z: float = 1.5,
    mr_exit_z: float = 0.5,
    regime_allocations: dict[str, dict[str, float]] | None = None,
    long_only: bool = False,
    min_hold_bars: int = 1,
    rebalance_threshold: float = 0.0,
) -> dict[str, Any]:
    bars_files = _collect_files(bars_15m_path, f"{symbol.upper()}_15m_*.parquet")
    regime_files = _collect_files(regime_15m_path, f"{symbol.upper()}_regime_15m_*.parquet")

    if not bars_files:
        return {
            "rows_bars_15m": 0,
            "rows_regime_15m": 0,
            "rows_out": 0,
            "files_written": [],
            "long_only": bool(long_only),
            "min_hold_bars": int(min_hold_bars),
            "rebalance_threshold": float(rebalance_threshold),
        }

    bars_frames: list[pd.DataFrame] = []
    rows_bars_15m = 0
    for file in bars_files:
        df = pd.read_parquet(file)
        rows_bars_15m += len(df)
        bars_frames.append(df)
    bars_15m = pd.concat(bars_frames, ignore_index=True)

    regime_frames: list[pd.DataFrame] = []
    rows_regime_15m = 0
    for file in regime_files:
        df = pd.read_parquet(file)
        rows_regime_15m += len(df)
        regime_frames.append(df)
    regime_15m = pd.concat(regime_frames, ignore_index=True) if regime_frames else pd.DataFrame(columns=["timestamp", "regime"])

    signals_15m = build_signals_15m(
        bars_15m=bars_15m,
        regime_15m=regime_15m,
        default_regime=default_regime,
        donchian_window=donchian_window,
        donchian_hold_until_opposite=donchian_hold_until_opposite,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        ema_adx_window=ema_adx_window,
        ema_adx_threshold=ema_adx_threshold,
        ema_use_adx_filter=ema_use_adx_filter,
        mr_z_window=mr_z_window,
        mr_entry_z=mr_entry_z,
        mr_exit_z=mr_exit_z,
        regime_allocations=regime_allocations,
        long_only=long_only,
        min_hold_bars=min_hold_bars,
        rebalance_threshold=rebalance_threshold,
    )

    files_written = write_monthly_signals(
        signals_15m=signals_15m,
        out_dir=output_dir,
        symbol=symbol,
    )

    return {
        "rows_bars_15m": rows_bars_15m,
        "rows_regime_15m": rows_regime_15m,
        "rows_out": int(len(signals_15m)),
        "files_written": files_written,
        "signal_composite_min": float(signals_15m["signal_composite"].min()) if not signals_15m.empty else 0.0,
        "signal_composite_max": float(signals_15m["signal_composite"].max()) if not signals_15m.empty else 0.0,
        "target_position_abs_mean": float(signals_15m["target_position"].abs().mean()) if not signals_15m.empty else 0.0,
        "long_only": bool(long_only),
        "min_hold_bars": int(min_hold_bars),
        "rebalance_threshold": float(rebalance_threshold),
    }
