from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bt_regime_system.backtest.costs import compute_cost_return, compute_turnover
from bt_regime_system.backtest.metrics import compute_metrics, write_metrics
from bt_regime_system.utils.io import ensure_dir

PARTITION_OFFSET = pd.Timedelta(minutes=1)
BACKTEST_COLUMNS = [
    "timestamp",
    "close",
    "target_position",
    "position",
    "close_return",
    "turnover",
    "gross_return",
    "cost_return",
    "net_return",
    "pnl",
    "equity",
    "bh_position",
    "bh_turnover",
    "bh_gross_return",
    "bh_cost_return",
    "bh_net_return",
    "bh_pnl",
    "bh_equity",
]


def _collect_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob(pattern))
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _standardize_bars_15m(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    required = ["timestamp", "close"]
    missing = set(required).difference(out.columns)
    if missing:
        raise ValueError(f"bars_15m missing columns: {sorted(missing)}")

    out = out[required].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")

    out = out.dropna(subset=["timestamp", "close"]) \
        .sort_values("timestamp") \
        .drop_duplicates("timestamp", keep="last")

    out["close"] = out["close"].astype(float)
    return out.reset_index(drop=True)


def _standardize_signals_15m(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    required = ["timestamp", "target_position"]
    missing = set(required).difference(out.columns)
    if missing:
        raise ValueError(f"signals_15m missing columns: {sorted(missing)}")

    out = out[required].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["target_position"] = pd.to_numeric(out["target_position"], errors="coerce")

    out = out.dropna(subset=["timestamp"]) \
        .sort_values("timestamp") \
        .drop_duplicates("timestamp", keep="last")

    out["target_position"] = out["target_position"].fillna(0.0).clip(-1.0, 1.0).astype(float)
    return out.reset_index(drop=True)


def simulate_backtest_15m(
    bars_15m: pd.DataFrame,
    signals_15m: pd.DataFrame,
    initial_equity: float = 100_000.0,
    fee_bps: float = 4.0,
    slippage_bps: float = 1.0,
    position_lag_bars: int = 1,
    long_only: bool = False,
) -> pd.DataFrame:
    """Simulate bar-close to next-bar execution backtest on 15m data.

    Includes a buy-and-hold baseline (`bh_*` columns) for direct comparison.
    """
    if position_lag_bars < 0:
        raise ValueError("position_lag_bars must be >= 0")

    bars = _standardize_bars_15m(bars_15m)
    if bars.empty:
        return pd.DataFrame(columns=BACKTEST_COLUMNS)

    signals = (
        _standardize_signals_15m(signals_15m)
        if not signals_15m.empty
        else pd.DataFrame(columns=["timestamp", "target_position"])
    )

    frame = bars.merge(signals, on="timestamp", how="left")
    frame["target_position"] = pd.to_numeric(frame["target_position"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    if long_only:
        frame["target_position"] = frame["target_position"].clip(lower=0.0)

    if position_lag_bars > 0:
        frame["position"] = frame["target_position"].shift(position_lag_bars).fillna(0.0)
    else:
        frame["position"] = frame["target_position"]

    frame["close_return"] = frame["close"].pct_change().fillna(0.0)

    frame["turnover"] = compute_turnover(frame["position"])
    frame["gross_return"] = frame["position"] * frame["close_return"]
    frame["cost_return"] = compute_cost_return(frame["position"], fee_bps=fee_bps, slippage_bps=slippage_bps)
    frame["net_return"] = frame["gross_return"] - frame["cost_return"]

    frame["equity"] = float(initial_equity) * (1.0 + frame["net_return"]).cumprod()
    frame["pnl"] = frame["equity"].diff().fillna(frame["equity"] - float(initial_equity))

    frame["bh_position"] = 1.0
    frame["bh_turnover"] = compute_turnover(frame["bh_position"])
    frame["bh_gross_return"] = frame["bh_position"] * frame["close_return"]
    frame["bh_cost_return"] = compute_cost_return(frame["bh_position"], fee_bps=fee_bps, slippage_bps=slippage_bps)
    frame["bh_net_return"] = frame["bh_gross_return"] - frame["bh_cost_return"]

    frame["bh_equity"] = float(initial_equity) * (1.0 + frame["bh_net_return"]).cumprod()
    frame["bh_pnl"] = frame["bh_equity"].diff().fillna(frame["bh_equity"] - float(initial_equity))

    out = frame[BACKTEST_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    for col in BACKTEST_COLUMNS:
        if col != "timestamp":
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(float)
    return out.sort_values("timestamp").reset_index(drop=True)


def monthly_backtest_filename(symbol: str, month: str) -> str:
    return f"{symbol.upper()}_backtest_15m_{month}.parquet"


def _standardize_backtest_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    missing = set(BACKTEST_COLUMNS).difference(out.columns)
    if missing:
        raise ValueError(f"backtest frame missing columns: {sorted(missing)}")

    out = out[BACKTEST_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for col in BACKTEST_COLUMNS:
        if col != "timestamp":
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["timestamp"]) \
        .sort_values("timestamp") \
        .drop_duplicates("timestamp", keep="last")

    for col in BACKTEST_COLUMNS:
        if col != "timestamp":
            out[col] = out[col].fillna(0.0).astype(float)
    return out.reset_index(drop=True)


def write_monthly_backtest(
    backtest_frame: pd.DataFrame,
    out_dir: Path,
    symbol: str,
) -> list[Path]:
    ensure_dir(out_dir)
    if backtest_frame.empty:
        return []

    frame = _standardize_backtest_frame(backtest_frame)
    partition_ts = frame["timestamp"] - PARTITION_OFFSET
    frame["month"] = partition_ts.dt.strftime("%Y-%m")

    written: list[Path] = []
    for month, chunk in frame.groupby("month", sort=True):
        target = out_dir / monthly_backtest_filename(symbol=symbol, month=month)
        new_rows = chunk.drop(columns=["month"]).sort_values("timestamp")

        if target.exists():
            try:
                existing = pd.read_parquet(target)
                existing = _standardize_backtest_frame(existing)
                merged = pd.concat([existing, new_rows], ignore_index=True)
                merged = merged.sort_values("timestamp", kind="mergesort").drop_duplicates("timestamp", keep="last")
            except ValueError:
                # Backward compatibility: pre-baseline files are replaced by current schema rows.
                merged = new_rows.drop_duplicates("timestamp", keep="last")
        else:
            merged = new_rows.drop_duplicates("timestamp", keep="last")

        merged.to_parquet(target, index=False)
        written.append(target)

    return written


def run_backtest(
    bars_15m_path: Path,
    signals_15m_path: Path,
    output_dir: Path,
    metrics_dir: Path,
    symbol: str = "BTCUSDT",
    initial_equity: float = 100_000.0,
    fee_bps: float = 4.0,
    slippage_bps: float = 1.0,
    position_lag_bars: int = 1,
    bars_per_year: int = 35_040,
    long_only: bool = False,
) -> dict[str, Any]:
    bars_files = _collect_files(bars_15m_path, f"{symbol.upper()}_15m_*.parquet")
    signal_files = _collect_files(signals_15m_path, f"{symbol.upper()}_signals_15m_*.parquet")

    if not bars_files:
        return {
            "rows_bars_15m": 0,
            "rows_signals_15m": 0,
            "rows_out": 0,
            "files_written": [],
            "metrics": {},
            "metrics_path": None,
        }

    bars_frames: list[pd.DataFrame] = []
    rows_bars_15m = 0
    for file in bars_files:
        df = pd.read_parquet(file)
        rows_bars_15m += len(df)
        bars_frames.append(df)
    bars_15m = pd.concat(bars_frames, ignore_index=True)

    signal_frames: list[pd.DataFrame] = []
    rows_signals_15m = 0
    for file in signal_files:
        df = pd.read_parquet(file)
        rows_signals_15m += len(df)
        signal_frames.append(df)
    signals_15m = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame(columns=["timestamp", "target_position"])

    backtest_frame = simulate_backtest_15m(
        bars_15m=bars_15m,
        signals_15m=signals_15m,
        initial_equity=initial_equity,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        position_lag_bars=position_lag_bars,
        long_only=long_only,
    )

    files_written = write_monthly_backtest(
        backtest_frame=backtest_frame,
        out_dir=output_dir,
        symbol=symbol,
    )

    metrics = compute_metrics(backtest_frame, bars_per_year=bars_per_year)
    metrics_path = write_metrics(metrics, ensure_dir(metrics_dir) / f"{symbol.upper()}_backtest_metrics.json")

    return {
        "rows_bars_15m": rows_bars_15m,
        "rows_signals_15m": rows_signals_15m,
        "rows_out": int(len(backtest_frame)),
        "files_written": files_written,
        "metrics": metrics,
        "metrics_path": metrics_path,
        "long_only": bool(long_only),
    }
