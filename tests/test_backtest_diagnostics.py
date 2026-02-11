from __future__ import annotations

import json

import pandas as pd
import pytest

from bt_regime_system.backtest.diagnostics import run_backtest_diagnostics
from bt_regime_system.regime.detect import R1, R2, R3, R4


def _make_timestamp() -> pd.DatetimeIndex:
    return pd.to_datetime(
        [
            "2024-01-31T23:45:00Z",
            "2024-02-01T00:00:00Z",
            "2024-02-01T00:15:00Z",
            "2024-02-01T00:30:00Z",
        ],
        utc=True,
    )


def test_run_backtest_diagnostics_outputs_reports(tmp_path) -> None:
    backtest_dir = tmp_path / "backtest"
    signals_dir = tmp_path / "signals"
    regime_dir = tmp_path / "regime"
    output_dir = tmp_path / "metrics"

    backtest_dir.mkdir(parents=True)
    signals_dir.mkdir(parents=True)
    regime_dir.mkdir(parents=True)

    ts = _make_timestamp()

    net_return = pd.Series([0.0, 0.009, -0.0105, -0.0005], dtype=float)
    equity = 1000.0 * (1.0 + net_return).cumprod()

    backtest = pd.DataFrame(
        {
            "timestamp": ts,
            "target_position": [0.0, 1.0, 0.5, 0.0],
            "position": [0.0, 1.0, 0.5, 0.0],
            "close_return": [0.0, 0.01, -0.02, 0.03],
            "turnover": [0.0, 1.0, 0.5, 0.5],
            "gross_return": [0.0, 0.01, -0.01, 0.0],
            "cost_return": [0.0, 0.001, 0.0005, 0.0005],
            "net_return": net_return,
            "equity": equity,
        }
    )

    signals = pd.DataFrame(
        {
            "timestamp": ts,
            "signal_donchian": [1.0, 1.0, -1.0, 0.0],
            "signal_ema_adx": [0.0, 1.0, 0.0, 0.0],
            "signal_mean_reversion": [0.0, 0.0, 1.0, 0.0],
            "signal_composite": [0.6, 0.8, 1.0, 0.0],
            "target_position": [0.0, 1.0, 0.5, 0.0],
        }
    )

    regime = pd.DataFrame(
        {
            "timestamp": ts,
            "regime": [R1, R2, R3, R4],
        }
    )

    backtest.to_parquet(backtest_dir / "BTCUSDT_backtest_15m_2024-02.parquet", index=False)
    signals.to_parquet(signals_dir / "BTCUSDT_signals_15m_2024-02.parquet", index=False)
    regime.to_parquet(regime_dir / "BTCUSDT_regime_15m_2024-02.parquet", index=False)

    result = run_backtest_diagnostics(
        backtest_path=backtest_dir,
        signals_path=signals_dir,
        regime_15m_path=regime_dir,
        output_dir=output_dir,
        symbol="BTCUSDT",
        bars_per_year=35040,
        position_lag_bars=1,
        default_regime=R4,
    )

    assert result["rows_backtest"] == 4
    assert result["rows_signals"] == 4
    assert result["rows_regime"] == 4
    assert result["rows_joined"] == 4
    assert len(result["files_written"]) == 7

    cost_summary = json.loads((output_dir / "BTCUSDT_backtest_diag_cost_summary.json").read_text(encoding="utf-8"))
    assert cost_summary["gross_return_sum"] == pytest.approx(0.0)
    assert cost_summary["cost_return_sum"] == pytest.approx(0.002)
    assert cost_summary["net_return_sum"] == pytest.approx(-0.002)

    regime_summary = pd.read_csv(output_dir / "BTCUSDT_backtest_diag_regime_summary.csv")
    assert sorted(regime_summary["regime"].tolist()) == [R1, R2, R3, R4]
    assert regime_summary["bar_count"].sum() == 4

    strategy_summary = json.loads((output_dir / "BTCUSDT_backtest_diag_strategy_summary.json").read_text(encoding="utf-8"))
    assert strategy_summary["position_lag_bars"] == 1
    assert strategy_summary["bars"] == 4
    assert strategy_summary["execution_adjustment_return_sum"] == pytest.approx(
        strategy_summary["actual_gross_return_sum"] - strategy_summary["raw_ret_composite_sum"]
    )


def test_run_backtest_diagnostics_empty_backtest_returns_empty_result(tmp_path) -> None:
    backtest_dir = tmp_path / "backtest"
    signals_dir = tmp_path / "signals"
    regime_dir = tmp_path / "regime"
    output_dir = tmp_path / "metrics"

    backtest_dir.mkdir(parents=True)
    signals_dir.mkdir(parents=True)
    regime_dir.mkdir(parents=True)

    result = run_backtest_diagnostics(
        backtest_path=backtest_dir,
        signals_path=signals_dir,
        regime_15m_path=regime_dir,
        output_dir=output_dir,
        symbol="BTCUSDT",
    )

    assert result["rows_backtest"] == 0
    assert result["rows_joined"] == 0
    assert result["files_written"] == []
