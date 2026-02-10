import numpy as np
import pandas as pd

from bt_regime_system.backtest.engine import run_backtest


def test_backtest_equity_grows_in_simple_uptrend() -> None:
    idx = pd.date_range("2024-01-01", periods=200, freq="15min", tz="UTC")
    close = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    signal = pd.Series(1.0, index=idx)

    result = run_backtest(close=close, target_signal=signal, fee_bps=0.0, slippage_bps=0.0)

    assert result["equity"].iloc[-1] > 1.0
