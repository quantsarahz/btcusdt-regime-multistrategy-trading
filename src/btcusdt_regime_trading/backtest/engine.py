"""Backtest execution engine (single-asset, next-bar-open execution)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest configuration."""

    name: str = "default_backtest"
    initial_capital: float = 1_000_000.0
    fee_bps: float = 4.0
    slippage_bps: float = 0.0
    allow_short: bool = True


class BacktestEngine:
    """Execute a simple next-bar-open backtest.

    Positions are determined by target_position[t] and executed at the next bar open.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def run(self, df: pd.DataFrame, target_position: pd.Series) -> Dict[str, object]:
        """Run backtest and return equity curve, positions, pnl, returns, trades."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df.index must be a DatetimeIndex")
        if not isinstance(target_position.index, pd.DatetimeIndex):
            raise ValueError("target_position.index must be a DatetimeIndex")
        if not df.index.equals(target_position.index):
            raise ValueError("df and target_position index must match")
        if "open" not in df.columns or "close" not in df.columns:
            raise ValueError("df must contain 'open' and 'close' columns")

        target = target_position.astype(float).copy()
        if not self.config.allow_short:
            target = target.clip(lower=0.0, upper=1.0)

        n = len(df)
        idx = df.index

        positions = pd.Series(0.0, index=idx)
        equity = pd.Series(np.nan, index=idx)
        pnl = pd.Series(0.0, index=idx)
        returns = pd.Series(0.0, index=idx)

        equity.iloc[0] = self.config.initial_capital

        trades = []

        for t in range(n - 1):
            pos_t = positions.iloc[t]
            next_pos = target.iloc[t]
            positions.iloc[t + 1] = next_pos

            trade_notional = abs(next_pos - pos_t) * equity.iloc[t]
            fee = trade_notional * self.config.fee_bps * 1e-4
            slippage = trade_notional * self.config.slippage_bps * 1e-4

            open_next = df["open"].iloc[t + 1]
            close_next = df["close"].iloc[t + 1]
            pnl_next = next_pos * (close_next / open_next - 1.0) * equity.iloc[t]

            equity.iloc[t + 1] = equity.iloc[t] + pnl_next - fee - slippage
            pnl.iloc[t + 1] = pnl_next - fee - slippage
            returns.iloc[t + 1] = pnl.iloc[t + 1] / equity.iloc[t] if equity.iloc[t] != 0 else 0.0

            if trade_notional > 0:
                trades.append(
                    {
                        "timestamp": idx[t + 1],
                        "position_before": pos_t,
                        "position_after": next_pos,
                        "trade_notional": trade_notional,
                        "fee": fee,
                    }
                )

        if equity.isna().any():
            raise ValueError("equity curve contains NaNs")

        trades_df = pd.DataFrame(trades, columns=["timestamp", "position_before", "position_after", "trade_notional", "fee"])

        return {
            "equity_curve": equity,
            "positions": positions,
            "pnl": pnl,
            "returns": returns,
            "trades": trades_df,
        }


def _print_summary(result: Dict[str, object]) -> None:
    equity = result["equity_curve"]
    trades = result["trades"]
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0

    print(f"final_equity: {equity.iloc[-1]:.2f}")
    print(f"max_equity: {equity.max():.2f}")
    print(f"min_equity: {equity.min():.2f}")
    print(f"total_return: {total_return:.4f}")
    print(f"n_trades: {len(trades)}")
    print("equity_head:")
    print(equity.head(5))
    print("equity_tail:")
    print(equity.tail(5))


def main() -> None:
    from btcusdt_regime_trading.data.loaders import load_klines_1h_processed

    df = load_klines_1h_processed()
    target = pd.Series(0.0, index=df.index)
    if len(target) > 100:
        target.iloc[100:] = 1.0

    engine = BacktestEngine(BacktestConfig())
    result = engine.run(df, target)
    _print_summary(result)


if __name__ == "__main__":
    main()
