from __future__ import annotations

from typing import Dict

import pandas as pd

from bt_regime_system.backtest.costs import transaction_cost_from_turnover


def run_backtest(
    close: pd.Series,
    target_signal: pd.Series,
    fee_bps: float = 2.0,
    slippage_bps: float = 1.0,
) -> Dict[str, pd.Series]:
    """Run a simple vectorized backtest with one-bar execution delay."""
    target_signal = target_signal.reindex(close.index).fillna(0.0).clip(-1.0, 1.0)
    position = target_signal.shift(1).fillna(0.0)

    returns = close.pct_change().fillna(0.0)
    gross_pnl = position * returns

    turnover = position.diff().abs().fillna(position.abs())
    cost = transaction_cost_from_turnover(turnover, fee_bps=fee_bps, slippage_bps=slippage_bps)
    net_pnl = gross_pnl - cost

    equity = (1.0 + net_pnl).cumprod()

    return {
        "close": close,
        "signal": target_signal,
        "position": position,
        "returns": returns,
        "gross_pnl": gross_pnl,
        "cost": cost,
        "net_pnl": net_pnl,
        "equity": equity,
    }
