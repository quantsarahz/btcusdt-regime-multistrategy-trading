from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Any

import pandas as pd

from bt_regime_system.utils.io import ensure_dir


REQUIRED_COLUMNS = [
    "timestamp",
    "position",
    "turnover",
    "net_return",
    "pnl",
    "equity",
    "bh_position",
    "bh_turnover",
    "bh_net_return",
    "bh_pnl",
    "bh_equity",
]


def _standardize_backtest_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    missing = set(REQUIRED_COLUMNS).difference(out.columns)
    if missing:
        raise ValueError(f"backtest frame missing columns: {sorted(missing)}")

    out = out[REQUIRED_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for col in REQUIRED_COLUMNS:
        if col != "timestamp":
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["timestamp"]) \
        .sort_values("timestamp") \
        .drop_duplicates("timestamp", keep="last")

    for col in REQUIRED_COLUMNS:
        if col != "timestamp":
            out[col] = out[col].fillna(0.0).astype(float)

    return out.reset_index(drop=True)


def _initial_equity_from_series(equity: pd.Series, net_return: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    first_equity = float(equity.iloc[0])
    first_r = float(net_return.iloc[0]) if len(net_return) else 0.0
    denom = 1.0 + first_r
    if denom == 0.0:
        return first_equity
    return float(first_equity / denom)


def _compute_metric_block(
    net: pd.Series,
    equity: pd.Series,
    position: pd.Series,
    turnover: pd.Series,
    bars_per_year: int,
) -> dict[str, float | int]:
    n = len(net)
    if n == 0:
        return {
            "initial_equity": 0.0,
            "final_equity": 0.0,
            "total_return": 0.0,
            "annual_return": 0.0,
            "volatility_annual": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "trade_count": 0,
            "turnover_annual": 0.0,
            "exposure_ratio": 0.0,
            "win_rate_active": 0.0,
        }

    initial_equity = _initial_equity_from_series(equity=equity, net_return=net)
    final_equity = float(equity.iloc[-1])

    total_return = 0.0 if initial_equity == 0 else final_equity / initial_equity - 1.0

    if n > 1 and initial_equity > 0 and final_equity > 0:
        annual_return = (final_equity / initial_equity) ** (bars_per_year / n) - 1.0
    else:
        annual_return = 0.0

    vol_bar = float(net.std(ddof=0))
    volatility_annual = vol_bar * sqrt(bars_per_year)

    mean_bar = float(net.mean())
    sharpe = 0.0 if vol_bar == 0 else (mean_bar / vol_bar) * sqrt(bars_per_year)

    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    calmar = 0.0 if max_drawdown >= 0 else annual_return / abs(max_drawdown)

    trade_count = int((turnover > 0).sum())
    turnover_annual = float(turnover.mean()) * bars_per_year if n else 0.0
    exposure_ratio = float((position.abs() > 0).mean())

    active_mask = position.abs() > 0
    if int(active_mask.sum()) > 0:
        win_rate_active = float((net[active_mask] > 0).mean())
    else:
        win_rate_active = 0.0

    return {
        "initial_equity": float(initial_equity),
        "final_equity": float(final_equity),
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "volatility_annual": float(volatility_annual),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "calmar": float(calmar),
        "trade_count": trade_count,
        "turnover_annual": float(turnover_annual),
        "exposure_ratio": float(exposure_ratio),
        "win_rate_active": float(win_rate_active),
    }


def compute_metrics(backtest_frame: pd.DataFrame, bars_per_year: int = 35_040) -> dict[str, float | int | bool]:
    """Compute strategy and buy&hold baseline metrics plus comparison fields."""
    frame = _standardize_backtest_frame(backtest_frame)
    if frame.empty:
        return {
            "row_count": 0,
            "initial_equity": 0.0,
            "final_equity": 0.0,
            "total_return": 0.0,
            "annual_return": 0.0,
            "volatility_annual": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "trade_count": 0,
            "turnover_annual": 0.0,
            "exposure_ratio": 0.0,
            "win_rate_active": 0.0,
            "bh_initial_equity": 0.0,
            "bh_final_equity": 0.0,
            "bh_total_return": 0.0,
            "bh_annual_return": 0.0,
            "bh_volatility_annual": 0.0,
            "bh_sharpe": 0.0,
            "bh_max_drawdown": 0.0,
            "bh_calmar": 0.0,
            "bh_trade_count": 0,
            "bh_turnover_annual": 0.0,
            "bh_exposure_ratio": 0.0,
            "bh_win_rate_active": 0.0,
            "excess_total_return": 0.0,
            "excess_annual_return": 0.0,
            "excess_sharpe": 0.0,
            "outperform_buy_hold": False,
        }

    strategy = _compute_metric_block(
        net=frame["net_return"],
        equity=frame["equity"],
        position=frame["position"],
        turnover=frame["turnover"],
        bars_per_year=bars_per_year,
    )
    buy_hold = _compute_metric_block(
        net=frame["bh_net_return"],
        equity=frame["bh_equity"],
        position=frame["bh_position"],
        turnover=frame["bh_turnover"],
        bars_per_year=bars_per_year,
    )

    excess_total_return = float(strategy["total_return"] - buy_hold["total_return"])
    excess_annual_return = float(strategy["annual_return"] - buy_hold["annual_return"])
    excess_sharpe = float(strategy["sharpe"] - buy_hold["sharpe"])

    return {
        "row_count": int(len(frame)),
        "initial_equity": float(strategy["initial_equity"]),
        "final_equity": float(strategy["final_equity"]),
        "total_return": float(strategy["total_return"]),
        "annual_return": float(strategy["annual_return"]),
        "volatility_annual": float(strategy["volatility_annual"]),
        "sharpe": float(strategy["sharpe"]),
        "max_drawdown": float(strategy["max_drawdown"]),
        "calmar": float(strategy["calmar"]),
        "trade_count": int(strategy["trade_count"]),
        "turnover_annual": float(strategy["turnover_annual"]),
        "exposure_ratio": float(strategy["exposure_ratio"]),
        "win_rate_active": float(strategy["win_rate_active"]),
        "bh_initial_equity": float(buy_hold["initial_equity"]),
        "bh_final_equity": float(buy_hold["final_equity"]),
        "bh_total_return": float(buy_hold["total_return"]),
        "bh_annual_return": float(buy_hold["annual_return"]),
        "bh_volatility_annual": float(buy_hold["volatility_annual"]),
        "bh_sharpe": float(buy_hold["sharpe"]),
        "bh_max_drawdown": float(buy_hold["max_drawdown"]),
        "bh_calmar": float(buy_hold["calmar"]),
        "bh_trade_count": int(buy_hold["trade_count"]),
        "bh_turnover_annual": float(buy_hold["turnover_annual"]),
        "bh_exposure_ratio": float(buy_hold["exposure_ratio"]),
        "bh_win_rate_active": float(buy_hold["win_rate_active"]),
        "excess_total_return": excess_total_return,
        "excess_annual_return": excess_annual_return,
        "excess_sharpe": excess_sharpe,
        "outperform_buy_hold": bool(strategy["total_return"] > buy_hold["total_return"]),
    }


def write_metrics(metrics: dict[str, Any], path: Path) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return path
