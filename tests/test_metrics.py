"""Unit tests for backtest metrics edge cases."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from btcusdt_regime_trading.backtest.metrics import compute_metrics


def _build_test_result(index: pd.DatetimeIndex, returns_values: list[float]) -> dict[str, Any]:
    """Build a minimal valid backtest result for unit tests."""
    if len(index) != len(returns_values):
        raise ValueError("index length must match returns_values length")

    returns = pd.Series(np.asarray(returns_values, dtype=float), index=index, name="returns")
    equity = pd.Series(100000.0 * np.cumprod(1.0 + returns.to_numpy(dtype=float)), index=index, name="equity_curve")
    pnl = pd.Series(returns.to_numpy(dtype=float) * 100000.0, index=index, name="pnl")
    positions = pd.Series(1.0, index=index, name="positions")
    positions.iloc[0] = 0.0

    trades = pd.DataFrame(
        {
            "timestamp": [index[1]],
            "position_before": [0.0],
            "position_after": [1.0],
            "trade_notional": [100000.0],
            "fee": [40.0],
        }
    )

    return {
        "equity_curve": equity,
        "returns": returns,
        "pnl": pnl,
        "positions": positions,
        "trades": trades,
    }


def _assert_raises_value_error(expected_substring: str, fn: Any, *args: Any, **kwargs: Any) -> None:
    """Assert a callable raises ValueError containing expected_substring."""
    try:
        fn(*args, **kwargs)
    except ValueError as exc:
        assert expected_substring in str(exc), f"Expected '{expected_substring}' in '{exc}'"
        return
    raise AssertionError("Expected ValueError was not raised")


def test_metrics_missing_key_raises() -> None:
    idx = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    result = _build_test_result(idx, [0.0, 0.01, -0.005, 0.007, 0.0, 0.003, -0.002, 0.004])
    result.pop("trades")

    _assert_raises_value_error("missing required keys", compute_metrics, result)


def test_metrics_bad_trade_timestamp_raises() -> None:
    idx = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    result = _build_test_result(idx, [0.0, 0.01, -0.005, 0.007, 0.0, 0.003, -0.002, 0.004])
    result["trades"].loc[0, "timestamp"] = pd.Timestamp("1999-01-01T00:00:00Z")

    _assert_raises_value_error("trade timestamps must align", compute_metrics, result)


def test_metrics_unsorted_index_raises() -> None:
    idx = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    unsorted_idx = pd.DatetimeIndex([idx[0], idx[2], idx[1], idx[3], idx[4], idx[5], idx[6], idx[7]])
    result = _build_test_result(unsorted_idx, [0.0, 0.01, -0.005, 0.007, 0.0, 0.003, -0.002, 0.004])

    _assert_raises_value_error("monotonic increasing", compute_metrics, result)


def test_metrics_duplicate_index_raises() -> None:
    idx = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    duplicate_idx = pd.DatetimeIndex([idx[0], idx[1], idx[2], idx[3], idx[4], idx[4], idx[6], idx[7]])
    result = _build_test_result(duplicate_idx, [0.0, 0.01, -0.005, 0.007, 0.0, 0.003, -0.002, 0.004])

    _assert_raises_value_error("must not contain duplicate", compute_metrics, result)


def test_metrics_zero_volatility_boundaries() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    result = _build_test_result(idx, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    metrics = compute_metrics(result)

    assert np.isnan(metrics["sharpe"])
    assert np.isnan(metrics["sortino"])
    assert np.isnan(metrics["calmar"])
    assert np.isnan(metrics["downside_vol"])


def test_metrics_zero_downside_volatility_boundary() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    result = _build_test_result(idx, [0.0, 0.002, 0.001, 0.003, 0.0015, 0.0025])

    metrics = compute_metrics(result)

    assert np.isnan(metrics["downside_vol"])
    assert np.isnan(metrics["sortino"])
    assert np.isfinite(metrics["sharpe"])
