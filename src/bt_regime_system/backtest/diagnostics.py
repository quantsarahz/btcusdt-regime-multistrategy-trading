from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from bt_regime_system.regime.allocate import STRATEGY_KEYS, weights_for_regime
from bt_regime_system.regime.detect import R4
from bt_regime_system.utils.io import ensure_dir

PARTITION_OFFSET = pd.Timedelta(minutes=1)

SIGNAL_DONCHIAN = "signal_donchian"
SIGNAL_EMA_ADX = "signal_ema_adx"
SIGNAL_MEAN_REV = "signal_mean_reversion"

REQUIRED_BACKTEST_COLUMNS = [
    "timestamp",
    "target_position",
    "position",
    "close_return",
    "turnover",
    "gross_return",
    "cost_return",
    "net_return",
    "equity",
]

REQUIRED_SIGNALS_COLUMNS = [
    "timestamp",
    SIGNAL_DONCHIAN,
    SIGNAL_EMA_ADX,
    SIGNAL_MEAN_REV,
    "signal_composite",
    "target_position",
]

REQUIRED_REGIME_COLUMNS = ["timestamp", "regime"]


def _collect_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob(pattern))
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _to_month(timestamp: pd.Series) -> pd.Series:
    ts = pd.to_datetime(timestamp, utc=True, errors="coerce") - PARTITION_OFFSET
    return ts.dt.strftime("%Y-%m")


def _standardize_backtest(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    missing = set(REQUIRED_BACKTEST_COLUMNS).difference(out.columns)
    if missing:
        raise ValueError(f"backtest frame missing columns: {sorted(missing)}")

    out = out[REQUIRED_BACKTEST_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for col in REQUIRED_BACKTEST_COLUMNS:
        if col != "timestamp":
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["timestamp"])
    out = out.sort_values("timestamp", kind="mergesort").drop_duplicates("timestamp", keep="last")

    for col in REQUIRED_BACKTEST_COLUMNS:
        if col != "timestamp":
            out[col] = out[col].fillna(0.0).astype(float)

    return out.reset_index(drop=True)


def _standardize_signals(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    missing = set(REQUIRED_SIGNALS_COLUMNS).difference(out.columns)
    if missing:
        raise ValueError(f"signals frame missing columns: {sorted(missing)}")

    out = out[REQUIRED_SIGNALS_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")

    for col in REQUIRED_SIGNALS_COLUMNS:
        if col != "timestamp":
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["timestamp"])
    out = out.sort_values("timestamp", kind="mergesort").drop_duplicates("timestamp", keep="last")

    for col in REQUIRED_SIGNALS_COLUMNS:
        if col != "timestamp":
            out[col] = out[col].fillna(0.0).astype(float)

    return out.reset_index(drop=True)


def _standardize_regime(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    missing = set(REQUIRED_REGIME_COLUMNS).difference(out.columns)
    if missing:
        raise ValueError(f"regime frame missing columns: {sorted(missing)}")

    out = out[REQUIRED_REGIME_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["regime"] = out["regime"].astype("string")

    out = out.dropna(subset=["timestamp"])
    out = out.sort_values("timestamp", kind="mergesort").drop_duplicates("timestamp", keep="last")
    return out.reset_index(drop=True)


def _concat_standardized(files: list[Path], standardizer: Any, empty_columns: list[str]) -> pd.DataFrame:
    if not files:
        return pd.DataFrame(columns=empty_columns)

    frames: list[pd.DataFrame] = []
    for file in files:
        frames.append(standardizer(pd.read_parquet(file)))

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("timestamp", kind="mergesort").drop_duplicates("timestamp", keep="last")
    return out.reset_index(drop=True)


def _safe_sharpe(series: pd.Series, bars_per_year: int) -> float:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    std = float(s.std(ddof=0))
    if std == 0.0:
        return 0.0
    return float((float(s.mean()) / std) * (bars_per_year**0.5))


def _initial_equity(backtest: pd.DataFrame) -> float:
    if backtest.empty:
        return 0.0
    first_equity = float(backtest["equity"].iloc[0])
    first_net = float(backtest["net_return"].iloc[0])
    denom = 1.0 + first_net
    if denom == 0.0:
        return first_equity
    return float(first_equity / denom)


def _strategy_weight_frame(
    regime: pd.Series,
    regime_allocations: dict[str, dict[str, float]] | None,
    default_regime: str,
) -> pd.DataFrame:
    resolved = regime.astype("string").fillna(default_regime)

    w_d: list[float] = []
    w_e: list[float] = []
    w_m: list[float] = []

    for label in resolved:
        key = str(label)
        try:
            weights = weights_for_regime(key, regime_allocations=regime_allocations)
        except KeyError:
            weights = weights_for_regime(default_regime, regime_allocations=regime_allocations)

        w_d.append(float(weights["donchian"]))
        w_e.append(float(weights["ema_adx"]))
        w_m.append(float(weights["mean_reversion"]))

    return pd.DataFrame(
        {
            "w_donchian": w_d,
            "w_ema_adx": w_e,
            "w_mean_reversion": w_m,
        }
    )


def _build_regime_tables(frame: pd.DataFrame, bars_per_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    regime_monthly = (
        frame.groupby(["month", "regime"], dropna=False)
        .agg(
            bar_count=("timestamp", "size"),
            gross_return_sum=("gross_return", "sum"),
            cost_return_sum=("cost_return", "sum"),
            net_return_sum=("net_return", "sum"),
            gross_return_mean=("gross_return", "mean"),
            cost_return_mean=("cost_return", "mean"),
            net_return_mean=("net_return", "mean"),
            turnover_sum=("turnover", "sum"),
            turnover_mean=("turnover", "mean"),
            exposure_ratio=("position", lambda s: float((pd.to_numeric(s).abs() > 0).mean())),
            win_rate=("net_return", lambda s: float((pd.to_numeric(s) > 0).mean())),
        )
        .reset_index()
    )
    regime_monthly["net_sharpe"] = regime_monthly.apply(
        lambda row: _safe_sharpe(
            frame.loc[(frame["month"] == row["month"]) & (frame["regime"] == row["regime"]), "net_return"],
            bars_per_year=bars_per_year,
        ),
        axis=1,
    )

    regime_summary = (
        frame.groupby("regime", dropna=False)
        .agg(
            bar_count=("timestamp", "size"),
            gross_return_sum=("gross_return", "sum"),
            cost_return_sum=("cost_return", "sum"),
            net_return_sum=("net_return", "sum"),
            gross_return_mean=("gross_return", "mean"),
            cost_return_mean=("cost_return", "mean"),
            net_return_mean=("net_return", "mean"),
            turnover_sum=("turnover", "sum"),
            turnover_mean=("turnover", "mean"),
            exposure_ratio=("position", lambda s: float((pd.to_numeric(s).abs() > 0).mean())),
            win_rate=("net_return", lambda s: float((pd.to_numeric(s) > 0).mean())),
        )
        .reset_index()
    )
    regime_summary["net_sharpe"] = regime_summary["regime"].map(
        lambda rg: _safe_sharpe(frame.loc[frame["regime"] == rg, "net_return"], bars_per_year=bars_per_year)
    )

    total_net = float(frame["net_return"].sum()) if len(frame) else 0.0
    if total_net != 0.0:
        regime_summary["net_contribution_pct"] = regime_summary["net_return_sum"] / total_net
    else:
        regime_summary["net_contribution_pct"] = 0.0

    return regime_monthly.sort_values(["month", "regime"], kind="mergesort"), regime_summary.sort_values(
        "regime", kind="mergesort"
    )


def _build_strategy_tables(
    frame: pd.DataFrame,
    bars_per_year: int,
    position_lag_bars: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    lag = max(0, int(position_lag_bars))

    frame = frame.copy()
    frame["raw_pos_donchian"] = frame[SIGNAL_DONCHIAN] * frame["w_donchian"]
    frame["raw_pos_ema_adx"] = frame[SIGNAL_EMA_ADX] * frame["w_ema_adx"]
    frame["raw_pos_mean_reversion"] = frame[SIGNAL_MEAN_REV] * frame["w_mean_reversion"]

    for key in STRATEGY_KEYS:
        raw_pos_col = f"raw_pos_{key}"
        raw_ret_col = f"raw_ret_{key}"
        frame[raw_ret_col] = frame[raw_pos_col].shift(lag).fillna(0.0) * frame["close_return"]

    frame["raw_ret_composite"] = (
        frame["raw_ret_donchian"] + frame["raw_ret_ema_adx"] + frame["raw_ret_mean_reversion"]
    )
    frame["ret_execution_adjustment"] = frame["gross_return"] - frame["raw_ret_composite"]

    strategy_monthly = (
        frame.groupby("month", dropna=False)
        .agg(
            bar_count=("timestamp", "size"),
            raw_ret_donchian_sum=("raw_ret_donchian", "sum"),
            raw_ret_ema_adx_sum=("raw_ret_ema_adx", "sum"),
            raw_ret_mean_reversion_sum=("raw_ret_mean_reversion", "sum"),
            raw_ret_composite_sum=("raw_ret_composite", "sum"),
            actual_gross_return_sum=("gross_return", "sum"),
            execution_adjustment_return_sum=("ret_execution_adjustment", "sum"),
            cost_return_sum=("cost_return", "sum"),
            net_return_sum=("net_return", "sum"),
        )
        .reset_index()
        .sort_values("month", kind="mergesort")
    )

    strategy_summary: dict[str, Any] = {
        "position_lag_bars": lag,
        "bars": int(len(frame)),
        "raw_ret_donchian_sum": float(frame["raw_ret_donchian"].sum()),
        "raw_ret_ema_adx_sum": float(frame["raw_ret_ema_adx"].sum()),
        "raw_ret_mean_reversion_sum": float(frame["raw_ret_mean_reversion"].sum()),
        "raw_ret_composite_sum": float(frame["raw_ret_composite"].sum()),
        "actual_gross_return_sum": float(frame["gross_return"].sum()),
        "execution_adjustment_return_sum": float(frame["ret_execution_adjustment"].sum()),
        "cost_return_sum": float(frame["cost_return"].sum()),
        "net_return_sum": float(frame["net_return"].sum()),
        "signal_active_ratio": {
            "donchian": float((frame[SIGNAL_DONCHIAN].abs() > 0).mean()),
            "ema_adx": float((frame[SIGNAL_EMA_ADX].abs() > 0).mean()),
            "mean_reversion": float((frame[SIGNAL_MEAN_REV].abs() > 0).mean()),
        },
        "avg_abs_weight": {
            "donchian": float(frame["w_donchian"].abs().mean()),
            "ema_adx": float(frame["w_ema_adx"].abs().mean()),
            "mean_reversion": float(frame["w_mean_reversion"].abs().mean()),
        },
        "raw_return_annualized": {
            "donchian": float(frame["raw_ret_donchian"].mean() * bars_per_year),
            "ema_adx": float(frame["raw_ret_ema_adx"].mean() * bars_per_year),
            "mean_reversion": float(frame["raw_ret_mean_reversion"].mean() * bars_per_year),
            "composite": float(frame["raw_ret_composite"].mean() * bars_per_year),
        },
    }

    return strategy_monthly, strategy_summary


def _build_cost_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    initial = _initial_equity(frame)

    if len(frame) == 0 or initial <= 0:
        cost_summary = {
            "initial_equity": float(initial),
            "gross_total_return": 0.0,
            "net_total_return": 0.0,
            "cost_drag_total_return": 0.0,
            "gross_return_sum": 0.0,
            "cost_return_sum": 0.0,
            "net_return_sum": 0.0,
            "turnover_sum": 0.0,
            "turnover_mean": 0.0,
            "cost_to_gross_abs_ratio": 0.0,
        }
    else:
        gross_equity = float(initial) * (1.0 + frame["gross_return"]).cumprod()
        net_equity = float(initial) * (1.0 + frame["net_return"]).cumprod()

        gross_total_return = float(gross_equity.iloc[-1] / float(initial) - 1.0)
        net_total_return = float(net_equity.iloc[-1] / float(initial) - 1.0)

        gross_return_sum = float(frame["gross_return"].sum())
        cost_return_sum = float(frame["cost_return"].sum())
        net_return_sum = float(frame["net_return"].sum())

        denom = abs(gross_return_sum)
        ratio = float(cost_return_sum / denom) if denom > 0 else 0.0

        cost_summary = {
            "initial_equity": float(initial),
            "gross_total_return": gross_total_return,
            "net_total_return": net_total_return,
            "cost_drag_total_return": float(gross_total_return - net_total_return),
            "gross_return_sum": gross_return_sum,
            "cost_return_sum": cost_return_sum,
            "net_return_sum": net_return_sum,
            "turnover_sum": float(frame["turnover"].sum()),
            "turnover_mean": float(frame["turnover"].mean()),
            "cost_to_gross_abs_ratio": ratio,
        }

    cost_monthly = (
        frame.groupby("month", dropna=False)
        .agg(
            bar_count=("timestamp", "size"),
            gross_return_sum=("gross_return", "sum"),
            cost_return_sum=("cost_return", "sum"),
            net_return_sum=("net_return", "sum"),
            turnover_sum=("turnover", "sum"),
            turnover_mean=("turnover", "mean"),
        )
        .reset_index()
        .sort_values("month", kind="mergesort")
    )

    cost_monthly["cost_to_gross_abs_ratio"] = cost_monthly.apply(
        lambda row: float(row["cost_return_sum"] / abs(row["gross_return_sum"])) if abs(float(row["gross_return_sum"])) > 0 else 0.0,
        axis=1,
    )

    return cost_monthly, cost_summary


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def run_backtest_diagnostics(
    backtest_path: Path,
    signals_path: Path,
    regime_15m_path: Path,
    output_dir: Path,
    symbol: str = "BTCUSDT",
    bars_per_year: int = 35_040,
    position_lag_bars: int = 1,
    default_regime: str = R4,
    regime_allocations: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    backtest_files = _collect_files(backtest_path, f"{symbol.upper()}_backtest_15m_*.parquet")
    signal_files = _collect_files(signals_path, f"{symbol.upper()}_signals_15m_*.parquet")
    regime_files = _collect_files(regime_15m_path, f"{symbol.upper()}_regime_15m_*.parquet")

    backtest = _concat_standardized(backtest_files, _standardize_backtest, REQUIRED_BACKTEST_COLUMNS)
    if backtest.empty:
        return {
            "rows_backtest": 0,
            "rows_signals": 0,
            "rows_regime": 0,
            "rows_joined": 0,
            "files_written": [],
        }

    signals = _concat_standardized(signal_files, _standardize_signals, REQUIRED_SIGNALS_COLUMNS)
    regime = _concat_standardized(regime_files, _standardize_regime, REQUIRED_REGIME_COLUMNS)

    joined = backtest.merge(signals, on="timestamp", how="left", suffixes=("", "_sig"))
    joined = joined.merge(regime, on="timestamp", how="left")

    for col in [SIGNAL_DONCHIAN, SIGNAL_EMA_ADX, SIGNAL_MEAN_REV, "signal_composite"]:
        if col not in joined.columns:
            joined[col] = 0.0
        joined[col] = pd.to_numeric(joined[col], errors="coerce").fillna(0.0).astype(float)

    joined["regime"] = joined.get("regime", pd.Series(default_regime, index=joined.index)).astype("string").fillna(default_regime)
    joined["month"] = _to_month(joined["timestamp"])

    weight_frame = _strategy_weight_frame(
        regime=joined["regime"],
        regime_allocations=regime_allocations,
        default_regime=default_regime,
    )
    joined = pd.concat([joined.reset_index(drop=True), weight_frame.reset_index(drop=True)], axis=1)

    regime_monthly, regime_summary = _build_regime_tables(joined, bars_per_year=bars_per_year)
    strategy_monthly, strategy_summary = _build_strategy_tables(
        joined,
        bars_per_year=bars_per_year,
        position_lag_bars=position_lag_bars,
    )
    cost_monthly, cost_summary = _build_cost_tables(joined)

    out_dir = ensure_dir(output_dir)
    files_written: list[str] = []

    regime_monthly_path = out_dir / f"{symbol.upper()}_backtest_diag_regime_monthly.csv"
    regime_summary_path = out_dir / f"{symbol.upper()}_backtest_diag_regime_summary.csv"
    strategy_monthly_path = out_dir / f"{symbol.upper()}_backtest_diag_strategy_monthly.csv"
    strategy_summary_path = out_dir / f"{symbol.upper()}_backtest_diag_strategy_summary.json"
    cost_monthly_path = out_dir / f"{symbol.upper()}_backtest_diag_cost_monthly.csv"
    cost_summary_path = out_dir / f"{symbol.upper()}_backtest_diag_cost_summary.json"
    summary_path = out_dir / f"{symbol.upper()}_backtest_diag_summary.json"

    regime_monthly.to_csv(regime_monthly_path, index=False)
    files_written.append(str(regime_monthly_path))

    regime_summary.to_csv(regime_summary_path, index=False)
    files_written.append(str(regime_summary_path))

    strategy_monthly.to_csv(strategy_monthly_path, index=False)
    files_written.append(str(strategy_monthly_path))

    _write_json(strategy_summary, strategy_summary_path)
    files_written.append(str(strategy_summary_path))

    cost_monthly.to_csv(cost_monthly_path, index=False)
    files_written.append(str(cost_monthly_path))

    _write_json(cost_summary, cost_summary_path)
    files_written.append(str(cost_summary_path))

    summary_payload = {
        "symbol": symbol.upper(),
        "rows_backtest": int(len(backtest)),
        "rows_signals": int(len(signals)),
        "rows_regime": int(len(regime)),
        "rows_joined": int(len(joined)),
        "bars_per_year": int(bars_per_year),
        "position_lag_bars": int(position_lag_bars),
        "default_regime": str(default_regime),
        "regime_dimension": {
            "regime_count": int(regime_summary["regime"].nunique()) if not regime_summary.empty else 0,
            "top_regime_by_net_return": (
                str(regime_summary.sort_values("net_return_sum", ascending=False).iloc[0]["regime"])
                if not regime_summary.empty
                else None
            ),
        },
        "strategy_contribution": strategy_summary,
        "cost_decomposition": cost_summary,
        "files": {
            "regime_monthly_csv": str(regime_monthly_path),
            "regime_summary_csv": str(regime_summary_path),
            "strategy_monthly_csv": str(strategy_monthly_path),
            "strategy_summary_json": str(strategy_summary_path),
            "cost_monthly_csv": str(cost_monthly_path),
            "cost_summary_json": str(cost_summary_path),
        },
    }

    _write_json(summary_payload, summary_path)
    files_written.append(str(summary_path))

    return {
        "rows_backtest": int(len(backtest)),
        "rows_signals": int(len(signals)),
        "rows_regime": int(len(regime)),
        "rows_joined": int(len(joined)),
        "files_written": files_written,
        "summary_path": str(summary_path),
        "strategy_summary_path": str(strategy_summary_path),
        "cost_summary_path": str(cost_summary_path),
        "regime_summary_path": str(regime_summary_path),
    }
