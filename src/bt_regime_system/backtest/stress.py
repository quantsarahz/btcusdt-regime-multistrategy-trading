from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from bt_regime_system.backtest.engine import simulate_backtest_15m
from bt_regime_system.backtest.metrics import compute_metrics
from bt_regime_system.utils.io import ensure_dir

REQUIRED_BARS_COLUMNS = ["timestamp", "close"]
REQUIRED_SIGNALS_COLUMNS = ["timestamp", "target_position"]


def _collect_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob(pattern))
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _standardize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    missing = set(REQUIRED_BARS_COLUMNS).difference(out.columns)
    if missing:
        raise ValueError(f"bars frame missing columns: {sorted(missing)}")

    out = out[REQUIRED_BARS_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")

    out = out.dropna(subset=["timestamp", "close"])
    out = out.sort_values("timestamp", kind="mergesort").drop_duplicates("timestamp", keep="last")
    return out.reset_index(drop=True)


def _standardize_signals(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    missing = set(REQUIRED_SIGNALS_COLUMNS).difference(out.columns)
    if missing:
        raise ValueError(f"signals frame missing columns: {sorted(missing)}")

    out = out[REQUIRED_SIGNALS_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["target_position"] = pd.to_numeric(out["target_position"], errors="coerce").fillna(0.0)

    out = out.dropna(subset=["timestamp"])
    out = out.sort_values("timestamp", kind="mergesort").drop_duplicates("timestamp", keep="last")
    out["target_position"] = out["target_position"].clip(-1.0, 1.0)
    return out.reset_index(drop=True)


def _concat_standardized(files: list[Path], standardizer: Any, empty_columns: list[str]) -> pd.DataFrame:
    if not files:
        return pd.DataFrame(columns=empty_columns)

    frames = [standardizer(pd.read_parquet(path)) for path in files]
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("timestamp", kind="mergesort").drop_duplicates("timestamp", keep="last")
    return out.reset_index(drop=True)


def _apply_volatility_multiplier(bars: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    m = float(multiplier)
    if m <= 0.0:
        raise ValueError("volatility multiplier must be > 0")
    if m == 1.0:
        return bars.copy()

    out = bars.copy()
    ret = out["close"].pct_change().fillna(0.0)
    stressed_ret = (ret * m).clip(lower=-0.95)
    out["close"] = float(out["close"].iloc[0]) * (1.0 + stressed_ret).cumprod()
    return out


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _scenario_name(kind: str, value: float | int) -> str:
    if isinstance(value, int):
        return f"{kind}_{value}"
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    text = text.replace("-", "m").replace(".", "p")
    return f"{kind}_{text}"


def _run_single_scenario(
    bars_15m: pd.DataFrame,
    signals_15m: pd.DataFrame,
    bars_per_year: int,
    scenario: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    stressed_bars = _apply_volatility_multiplier(bars_15m, float(scenario["volatility_multiplier"]))

    backtest = simulate_backtest_15m(
        bars_15m=stressed_bars,
        signals_15m=signals_15m,
        initial_equity=float(scenario["initial_equity"]),
        fee_bps=float(scenario["fee_bps"]),
        slippage_bps=float(scenario["slippage_bps"]),
        position_lag_bars=int(scenario["position_lag_bars"]),
        long_only=bool(scenario["long_only"]),
    )

    metrics = compute_metrics(backtest, bars_per_year=bars_per_year)
    return backtest, metrics


def run_backtest_stress(
    bars_15m_path: Path,
    signals_15m_path: Path,
    output_dir: Path,
    symbol: str = "BTCUSDT",
    initial_equity: float = 100_000.0,
    fee_bps: float = 4.0,
    slippage_bps: float = 1.0,
    position_lag_bars: int = 1,
    bars_per_year: int = 35_040,
    long_only: bool = True,
    fee_bps_values: list[float] | None = None,
    slippage_bps_values: list[float] | None = None,
    lag_values: list[int] | None = None,
    volatility_multipliers: list[float] | None = None,
    include_combined_worst_case: bool = True,
) -> dict[str, Any]:
    bars_files = _collect_files(bars_15m_path, f"{symbol.upper()}_15m_*.parquet")
    signal_files = _collect_files(signals_15m_path, f"{symbol.upper()}_signals_15m_*.parquet")

    bars = _concat_standardized(bars_files, _standardize_bars, REQUIRED_BARS_COLUMNS)
    signals = _concat_standardized(signal_files, _standardize_signals, REQUIRED_SIGNALS_COLUMNS)

    if bars.empty:
        return {
            "rows_bars_15m": 0,
            "rows_signals_15m": 0,
            "scenario_count": 0,
            "files_written": [],
        }

    if signals.empty:
        signals = pd.DataFrame({
            "timestamp": bars["timestamp"],
            "target_position": 0.0,
        })

    fee_values = sorted({float(fee_bps)}.union({float(v) for v in (fee_bps_values or [fee_bps, fee_bps * 2.0, fee_bps * 3.0])}))
    slip_values = sorted({float(slippage_bps)}.union({float(v) for v in (slippage_bps_values or [slippage_bps, slippage_bps * 2.0, slippage_bps * 4.0])}))
    lag_set = {int(position_lag_bars)}.union({int(v) for v in (lag_values or [position_lag_bars, position_lag_bars + 1, position_lag_bars + 2])})
    lag_values_sorted = sorted(v for v in lag_set if v >= 0)
    vol_values = sorted({1.0}.union({float(v) for v in (volatility_multipliers or [1.0, 1.5, 2.0])}))

    baseline = {
        "scenario": "baseline",
        "dimension": "baseline",
        "initial_equity": float(initial_equity),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "position_lag_bars": int(position_lag_bars),
        "volatility_multiplier": 1.0,
        "long_only": bool(long_only),
    }

    scenarios: list[dict[str, Any]] = [baseline]

    for v in fee_values:
        if v == float(fee_bps):
            continue
        scenarios.append({**baseline, "scenario": _scenario_name("fee", v), "dimension": "fee", "fee_bps": float(v)})

    for v in slip_values:
        if v == float(slippage_bps):
            continue
        scenarios.append({**baseline, "scenario": _scenario_name("slippage", v), "dimension": "slippage", "slippage_bps": float(v)})

    for v in lag_values_sorted:
        if v == int(position_lag_bars):
            continue
        scenarios.append({**baseline, "scenario": _scenario_name("lag", v), "dimension": "lag", "position_lag_bars": int(v)})

    for v in vol_values:
        if v == 1.0:
            continue
        scenarios.append({**baseline, "scenario": _scenario_name("vol", v), "dimension": "extreme_volatility", "volatility_multiplier": float(v)})

    if include_combined_worst_case:
        worst = {
            **baseline,
            "scenario": "combined_worst_case",
            "dimension": "combined",
            "fee_bps": max(fee_values) if fee_values else float(fee_bps),
            "slippage_bps": max(slip_values) if slip_values else float(slippage_bps),
            "position_lag_bars": max(lag_values_sorted) if lag_values_sorted else int(position_lag_bars),
            "volatility_multiplier": max(vol_values) if vol_values else 1.0,
        }
        scenarios.append(worst)

    rows: list[dict[str, Any]] = []
    backtest_rows_total = 0

    for scenario in scenarios:
        backtest, metrics = _run_single_scenario(
            bars_15m=bars,
            signals_15m=signals,
            bars_per_year=bars_per_year,
            scenario=scenario,
        )
        backtest_rows_total += len(backtest)

        row = {
            "scenario": scenario["scenario"],
            "dimension": scenario["dimension"],
            "fee_bps": float(scenario["fee_bps"]),
            "slippage_bps": float(scenario["slippage_bps"]),
            "position_lag_bars": int(scenario["position_lag_bars"]),
            "volatility_multiplier": float(scenario["volatility_multiplier"]),
            "row_count": int(metrics.get("row_count", 0)),
            "total_return": float(metrics.get("total_return", 0.0)),
            "annual_return": float(metrics.get("annual_return", 0.0)),
            "sharpe": float(metrics.get("sharpe", 0.0)),
            "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
            "bh_total_return": float(metrics.get("bh_total_return", 0.0)),
            "bh_sharpe": float(metrics.get("bh_sharpe", 0.0)),
            "excess_total_return": float(metrics.get("excess_total_return", 0.0)),
            "excess_sharpe": float(metrics.get("excess_sharpe", 0.0)),
            "outperform_buy_hold": bool(metrics.get("outperform_buy_hold", False)),
        }
        rows.append(row)

    results = pd.DataFrame(rows).sort_values(["dimension", "scenario"], kind="mergesort").reset_index(drop=True)

    baseline_row = results.loc[results["scenario"] == "baseline"].iloc[0]
    for metric in ["total_return", "sharpe", "max_drawdown", "excess_total_return"]:
        results[f"delta_{metric}_vs_baseline"] = results[metric] - float(baseline_row[metric])

    out_dir = ensure_dir(output_dir)
    results_path = out_dir / f"{symbol.upper()}_backtest_stress_results.csv"
    summary_path = out_dir / f"{symbol.upper()}_backtest_stress_summary.json"

    results.to_csv(results_path, index=False)

    non_baseline = results[results["scenario"] != "baseline"].copy()
    if non_baseline.empty:
        worst_total = None
        worst_drawdown = None
        best_sharpe = None
    else:
        worst_total = non_baseline.sort_values("total_return", ascending=True).iloc[0]["scenario"]
        worst_drawdown = non_baseline.sort_values("max_drawdown", ascending=True).iloc[0]["scenario"]
        best_sharpe = non_baseline.sort_values("sharpe", ascending=False).iloc[0]["scenario"]

    summary_payload = {
        "symbol": symbol.upper(),
        "rows_bars_15m": int(len(bars)),
        "rows_signals_15m": int(len(signals)),
        "rows_backtest_total": int(backtest_rows_total),
        "scenario_count": int(len(results)),
        "baseline": {
            "fee_bps": float(baseline["fee_bps"]),
            "slippage_bps": float(baseline["slippage_bps"]),
            "position_lag_bars": int(baseline["position_lag_bars"]),
            "volatility_multiplier": float(baseline["volatility_multiplier"]),
            "metrics": {
                "total_return": float(baseline_row["total_return"]),
                "annual_return": float(baseline_row["annual_return"]),
                "sharpe": float(baseline_row["sharpe"]),
                "max_drawdown": float(baseline_row["max_drawdown"]),
                "bh_total_return": float(baseline_row["bh_total_return"]),
                "excess_total_return": float(baseline_row["excess_total_return"]),
            },
        },
        "extremes": {
            "worst_total_return_scenario": worst_total,
            "worst_drawdown_scenario": worst_drawdown,
            "best_sharpe_scenario": best_sharpe,
        },
        "files": {
            "results_csv": str(results_path),
        },
    }

    _write_json(summary_payload, summary_path)

    return {
        "rows_bars_15m": int(len(bars)),
        "rows_signals_15m": int(len(signals)),
        "rows_backtest_total": int(backtest_rows_total),
        "scenario_count": int(len(results)),
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "baseline_metrics": summary_payload["baseline"]["metrics"],
        "extremes": summary_payload["extremes"],
        "files_written": [str(results_path), str(summary_path)],
    }
