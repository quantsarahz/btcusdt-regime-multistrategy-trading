from __future__ import annotations

import json

import pandas as pd

from bt_regime_system.backtest.engine import run_backtest, simulate_backtest_15m


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


def test_simulate_backtest_15m_without_costs() -> None:
    bars = _make_bars("2024-01-01T00:15:00Z", [100.0, 110.0, 121.0])
    signals = pd.DataFrame(
        {
            "timestamp": bars["timestamp"],
            "target_position": [1.0, 1.0, 1.0],
        }
    )

    out = simulate_backtest_15m(
        bars_15m=bars,
        signals_15m=signals,
        initial_equity=1000.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        position_lag_bars=1,
    )

    assert out["position"].tolist() == [0.0, 1.0, 1.0]
    assert out["net_return"].round(10).tolist() == [0.0, 0.1, 0.1]
    assert out["equity"].round(10).tolist() == [1000.0, 1100.0, 1210.0]


def test_simulate_backtest_15m_with_costs_reduces_returns() -> None:
    bars = _make_bars("2024-01-01T00:15:00Z", [100.0, 110.0, 121.0])
    signals = pd.DataFrame(
        {
            "timestamp": bars["timestamp"],
            "target_position": [1.0, 1.0, 1.0],
        }
    )

    out = simulate_backtest_15m(
        bars_15m=bars,
        signals_15m=signals,
        initial_equity=1000.0,
        fee_bps=10.0,
        slippage_bps=5.0,
        position_lag_bars=1,
    )

    assert out["turnover"].tolist() == [0.0, 1.0, 0.0]
    assert out["cost_return"].round(10).tolist() == [0.0, 0.0015, 0.0]
    assert out["net_return"].round(10).tolist() == [0.0, 0.0985, 0.1]
    assert out["equity"].iloc[-1] < 1210.0


def test_simulate_backtest_15m_long_only_blocks_shorts() -> None:
    bars = _make_bars("2024-01-01T00:15:00Z", [100.0, 99.0, 98.0, 97.0])
    signals = pd.DataFrame(
        {
            "timestamp": bars["timestamp"],
            "target_position": [1.0, -1.0, -1.0, 1.0],
        }
    )

    out = simulate_backtest_15m(
        bars_15m=bars,
        signals_15m=signals,
        initial_equity=1000.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        position_lag_bars=1,
        long_only=True,
    )

    assert out["target_position"].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert min(out["position"].tolist()) >= 0.0


def test_run_backtest_writes_outputs_and_metrics(tmp_path) -> None:
    bars_dir = tmp_path / "bars"
    signals_dir = tmp_path / "signals"
    out_dir = tmp_path / "backtest"
    metrics_dir = tmp_path / "metrics"
    bars_dir.mkdir(parents=True)
    signals_dir.mkdir(parents=True)

    bars = _make_bars("2024-01-31T23:45:00Z", [100.0, 101.0, 102.0])
    signals = pd.DataFrame(
        {
            "timestamp": bars["timestamp"],
            "target_position": [1.0, 1.0, 1.0],
        }
    )

    bars.to_parquet(bars_dir / "BTCUSDT_15m_2024-02.parquet", index=False)
    signals.to_parquet(signals_dir / "BTCUSDT_signals_15m_2024-02.parquet", index=False)

    summary = run_backtest(
        bars_15m_path=bars_dir,
        signals_15m_path=signals_dir,
        output_dir=out_dir,
        metrics_dir=metrics_dir,
        symbol="BTCUSDT",
        initial_equity=1000.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        position_lag_bars=1,
        bars_per_year=35040,
        long_only=True,
    )

    assert summary["rows_bars_15m"] == 3
    assert summary["rows_signals_15m"] == 3
    assert summary["rows_out"] == 3
    assert summary["long_only"] is True

    files = sorted(out_dir.glob("BTCUSDT_backtest_15m_2024-*.parquet"))
    assert len(files) == 2
    assert files[0].name == "BTCUSDT_backtest_15m_2024-01.parquet"
    assert files[1].name == "BTCUSDT_backtest_15m_2024-02.parquet"

    merged = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    assert len(merged) == 3
    assert set(merged.columns) == {
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
    }
    assert merged["bh_position"].tolist() == [1.0, 1.0, 1.0]

    metrics_path = metrics_dir / "BTCUSDT_backtest_metrics.json"
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["row_count"] == 3
    assert "sharpe" in metrics
    assert "max_drawdown" in metrics
    assert "bh_total_return" in metrics
    assert "excess_total_return" in metrics
    assert metrics["bh_trade_count"] == 1
