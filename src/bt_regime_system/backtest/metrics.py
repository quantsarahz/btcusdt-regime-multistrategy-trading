from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def summarize_performance(net_pnl: pd.Series, bars_per_year: int = 35040) -> Dict[str, float]:
    """Compute core performance metrics."""
    net = net_pnl.dropna()
    if net.empty:
        raise ValueError("net_pnl is empty")

    equity = (1.0 + net).cumprod()
    total_return = equity.iloc[-1] - 1.0

    years = max(len(net) / bars_per_year, 1.0 / bars_per_year)
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0

    vol = net.std(ddof=0) * np.sqrt(bars_per_year)
    sharpe = np.nan if vol == 0 else (net.mean() * bars_per_year) / vol

    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1.0
    max_drawdown = drawdown.min()

    win_rate = (net > 0).mean()

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "volatility": float(vol),
        "sharpe": float(sharpe) if not np.isnan(sharpe) else float("nan"),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
    }
