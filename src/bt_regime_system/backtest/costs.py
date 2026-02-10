from __future__ import annotations

import pandas as pd


def transaction_cost_from_turnover(
    turnover: pd.Series,
    fee_bps: float = 2.0,
    slippage_bps: float = 1.0,
) -> pd.Series:
    """Return per-bar linear trading cost from turnover."""
    total_bps = fee_bps + slippage_bps
    return turnover * (total_bps / 10_000.0)
