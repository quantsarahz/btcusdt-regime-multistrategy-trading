"""Benchmark strategies for backtest comparison."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from btcusdt_regime_trading.backtest.engine import BacktestEngine


def _validate_price_df(df: pd.DataFrame) -> None:
    """Validate price data input for benchmarks."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex")
    if "open" not in df.columns or "close" not in df.columns:
        raise ValueError("price data must contain 'open' and 'close' columns")


def run_buy_and_hold(df: pd.DataFrame, engine: BacktestEngine) -> Dict[str, object]:
    """Buy & Hold benchmark: always long +1, effective from the second bar (next-bar-open execution)."""
    _validate_price_df(df)
    target = pd.Series(1.0, index=df.index)
    result = engine.run(df, target)
    if result["equity_curve"].isna().any():
        raise ValueError("equity_curve contains NaNs")
    return result


def run_equal_weight(
    df: pd.DataFrame, signals: pd.DataFrame, engine: BacktestEngine
) -> Dict[str, object]:
    """Equal-weight benchmark: average strategy signals per bar."""
    _validate_price_df(df)
    if not signals.index.equals(df.index):
        raise ValueError("signals index must match price data index")
    if signals.isna().any().any():
        raise ValueError("signals must not contain NaNs")

    target = signals.mean(axis=1).clip(-1.0, 1.0)
    result = engine.run(df, target)
    if result["equity_curve"].isna().any():
        raise ValueError("equity_curve contains NaNs")
    return result


def run_best_single(
    df: pd.DataFrame, signals_dict: Dict[str, pd.Series], engine: BacktestEngine
) -> Dict[str, object]:
    """Hindsight benchmark: best single strategy by final equity."""
    if not signals_dict:
        raise ValueError("signals_dict cannot be empty")
    _validate_price_df(df)

    best_name = None
    best_result = None
    best_final_equity = -np.inf

    for name, series in signals_dict.items():
        if not series.index.equals(df.index):
            raise ValueError(f"signals index for {name} must match price data index")
        if series.isna().any():
            raise ValueError(f"signals for {name} contain NaNs")

        result = engine.run(df, series)
        equity = result["equity_curve"]
        if equity.isna().any():
            raise ValueError(f"equity_curve contains NaNs for {name}")

        final_equity = equity.iloc[-1]
        if final_equity > best_final_equity:
            best_final_equity = final_equity
            best_name = name
            best_result = result

    return {"name": best_name, "result": best_result, "equity_curve": best_result["equity_curve"]}




def unwrap_benchmark_result(res: Dict[str, object]) -> Dict[str, object]:
    """Normalize benchmark outputs to a backtest result dict."""
    if isinstance(res, dict) and "result" in res:
        return res["result"]  # best_single wrapper
    return res


def main() -> None:
    from btcusdt_regime_trading.data.loaders import load_klines_1h_processed
    from btcusdt_regime_trading.backtest.engine import BacktestConfig

    df = load_klines_1h_processed()
    engine = BacktestEngine(BacktestConfig())

    # Dummy signals
    signals = pd.DataFrame(
        {
            "Strategy_A": 1.0,
            "Strategy_B": -1.0,
            "Strategy_C": [1.0 if i % 2 == 0 else -1.0 for i in range(len(df))],
        },
        index=df.index,
    )

    buy_hold = run_buy_and_hold(df, engine)

    # For equal-weight, use dummy strategy signals
    equal_weight = run_equal_weight(df, signals, engine)

    signals_dict = {name: signals[name] for name in signals.columns}
    best_single = run_best_single(df, signals_dict, engine)

    print("Buy & Hold final equity:", buy_hold["equity_curve"].iloc[-1])
    print("Equal-weight final equity:", equal_weight["equity_curve"].iloc[-1])
    print("Best single strategy:", best_single["name"])
    print("Best single final equity:", best_single["result"]["equity_curve"].iloc[-1])

    print("Buy & Hold equity head:")
    print(buy_hold["equity_curve"].head(3))
    print("Buy & Hold equity tail:")
    print(buy_hold["equity_curve"].tail(3))

    print("Equal-weight equity head:")
    print(equal_weight["equity_curve"].head(3))
    print("Equal-weight equity tail:")
    print(equal_weight["equity_curve"].tail(3))

    print("Best single equity head:")
    print(best_single["result"]["equity_curve"].head(3))
    print("Best single equity tail:")
    print(best_single["result"]["equity_curve"].tail(3))


if __name__ == "__main__":
    main()
