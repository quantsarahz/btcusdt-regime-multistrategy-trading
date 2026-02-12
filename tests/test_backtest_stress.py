from __future__ import annotations

import json

import pandas as pd
import pytest

from bt_regime_system.backtest.stress import run_backtest_stress


def _make_bars(start: str, closes: list[float]) -> pd.DataFrame:
    ts = pd.date_range(start=start, periods=len(closes), freq="15min", tz="UTC")
    close = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        }
    )


def test_run_backtest_stress_writes_outputs(tmp_path) -> None:
    bars_dir = tmp_path / "bars"
    signals_dir = tmp_path / "signals"
    out_dir = tmp_path / "metrics"
    bars_dir.mkdir(parents=True)
    signals_dir.mkdir(parents=True)

    bars = _make_bars("2024-01-01T00:15:00Z", [100.0, 102.0, 101.0, 103.0, 104.0, 103.0])
    signals = pd.DataFrame(
        {
            "timestamp": bars["timestamp"],
            "target_position": [0.0, 1.0, 1.0, 0.5, 1.0, 0.0],
        }
    )

    bars.to_parquet(bars_dir / "BTCUSDT_15m_2024-01.parquet", index=False)
    signals.to_parquet(signals_dir / "BTCUSDT_signals_15m_2024-01.parquet", index=False)

    summary = run_backtest_stress(
        bars_15m_path=bars_dir,
        signals_15m_path=signals_dir,
        output_dir=out_dir,
        symbol="BTCUSDT",
        initial_equity=1000.0,
        fee_bps=4.0,
        slippage_bps=1.0,
        position_lag_bars=1,
        bars_per_year=35040,
        long_only=True,
        fee_bps_values=[4.0, 8.0],
        slippage_bps_values=[1.0, 2.0],
        lag_values=[1, 2],
        volatility_multipliers=[1.0, 2.0],
        include_combined_worst_case=True,
    )

    assert summary["rows_bars_15m"] == 6
    assert summary["rows_signals_15m"] == 6
    assert summary["scenario_count"] == 6
    assert len(summary["files_written"]) == 2

    results_path = out_dir / "BTCUSDT_backtest_stress_results.csv"
    summary_path = out_dir / "BTCUSDT_backtest_stress_summary.json"
    assert results_path.exists()
    assert summary_path.exists()

    results = pd.read_csv(results_path)
    assert sorted(results["scenario"].tolist()) == sorted(
        [
            "baseline",
            "fee_8",
            "slippage_2",
            "lag_2",
            "vol_2",
            "combined_worst_case",
        ]
    )

    baseline = results.loc[results["scenario"] == "baseline"].iloc[0]
    assert float(baseline["delta_total_return_vs_baseline"]) == 0.0
    assert float(baseline["delta_sharpe_vs_baseline"]) == 0.0
    assert float(baseline["delta_max_drawdown_vs_baseline"]) == 0.0

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["scenario_count"] == 6
    assert payload["baseline"]["metrics"]["total_return"] == pytest.approx(float(baseline["total_return"]))


def test_run_backtest_stress_empty_bars_returns_no_files(tmp_path) -> None:
    bars_dir = tmp_path / "bars"
    signals_dir = tmp_path / "signals"
    out_dir = tmp_path / "metrics"
    bars_dir.mkdir(parents=True)
    signals_dir.mkdir(parents=True)

    summary = run_backtest_stress(
        bars_15m_path=bars_dir,
        signals_15m_path=signals_dir,
        output_dir=out_dir,
        symbol="BTCUSDT",
    )

    assert summary["rows_bars_15m"] == 0
    assert summary["scenario_count"] == 0
    assert summary["files_written"] == []
