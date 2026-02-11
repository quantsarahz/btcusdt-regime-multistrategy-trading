from __future__ import annotations

import pandas as pd


def cost_rate_from_bps(fee_bps: float = 4.0, slippage_bps: float = 1.0) -> float:
    """Return per-unit turnover cost rate from bps inputs."""
    fee = float(fee_bps)
    slip = float(slippage_bps)
    if fee < 0 or slip < 0:
        raise ValueError("fee_bps and slippage_bps must be non-negative")
    return (fee + slip) / 10_000.0


def compute_turnover(position: pd.Series) -> pd.Series:
    """Compute per-bar turnover as absolute position change."""
    if not isinstance(position, pd.Series):
        position = pd.Series(position)

    pos = pd.to_numeric(position, errors="coerce").fillna(0.0).astype(float)
    turnover = pos.diff().abs()
    if len(turnover):
        turnover.iloc[0] = abs(pos.iloc[0])
    return turnover.fillna(0.0)


def compute_cost_return(
    position: pd.Series,
    fee_bps: float = 4.0,
    slippage_bps: float = 1.0,
) -> pd.Series:
    """Convert turnover into return-space trading cost per bar."""
    turnover = compute_turnover(position)
    return turnover * cost_rate_from_bps(fee_bps=fee_bps, slippage_bps=slippage_bps)
