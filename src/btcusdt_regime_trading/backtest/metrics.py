"""Performance and risk metrics for single-asset BTCUSDT backtests."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 365.25 * 24 * 3600
REQUIRED_RESULT_KEYS = ("equity_curve", "returns", "pnl", "positions", "trades")


def safe_series(s: Any, name: str = "series") -> pd.Series:
    """Return a Series and enforce emptiness/NaN checks."""
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    out = s.copy()
    if out.empty:
        raise ValueError(f"{name} is empty")
    if out.isna().any():
        raise ValueError(f"{name} contains NaN values")
    return out


def _validate_datetime_index(index: pd.Index, name: str, min_len: int = 1) -> pd.DatetimeIndex:
    """Validate that index is DatetimeIndex, monotonic, unique, and long enough."""
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{name} must be a DatetimeIndex")
    if len(index) < min_len:
        raise ValueError(f"{name} must contain at least {min_len} timestamps")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} must be monotonic increasing")
    if index.has_duplicates:
        raise ValueError(f"{name} must not contain duplicate timestamps")
    return index


def infer_bars_per_year(index: pd.DatetimeIndex) -> float:
    """Infer bars/year using median timestamp spacing (leap-year aware)."""
    idx = _validate_datetime_index(index, name="index", min_len=2)

    delta_seconds = idx.to_series().diff().dt.total_seconds().dropna()
    delta_seconds = delta_seconds[delta_seconds > 0]
    if delta_seconds.empty:
        raise ValueError("could not infer bar spacing from index")

    median_delta_seconds = float(delta_seconds.median())
    return float(SECONDS_PER_YEAR / median_delta_seconds)


def infer_years(index: pd.DatetimeIndex) -> float:
    """Infer sample length in years using timestamp span (leap-year aware)."""
    idx = _validate_datetime_index(index, name="index", min_len=2)

    span_seconds = (idx[-1] - idx[0]).total_seconds()
    years = float(span_seconds / SECONDS_PER_YEAR)
    if years <= 0:
        raise ValueError("years must be positive")
    return years


def compute_max_drawdown(equity_curve: pd.Series) -> Dict[str, Any]:
    """Compute max drawdown statistics from an equity curve."""
    eq = safe_series(equity_curve, name="equity_curve").astype(float)
    _validate_datetime_index(eq.index, name="equity_curve index", min_len=1)

    running_peak = eq.cummax()
    drawdown = eq / running_peak - 1.0

    mdd_pct = float(drawdown.min())
    trough_time = drawdown.idxmin()
    peak_value = float(running_peak.loc[trough_time])
    peak_candidates = eq.loc[:trough_time]
    peak_time = peak_candidates[peak_candidates == peak_value].index[-1]

    post_trough = eq.loc[trough_time:]
    recovered = post_trough[post_trough >= peak_value]
    recovery_time = recovered.index[0] if len(recovered) > 0 else None

    end_time = recovery_time if recovery_time is not None else eq.index[-1]
    duration_hours = float((end_time - peak_time).total_seconds() / 3600.0)

    return {
        "mdd_pct": mdd_pct,
        "peak_time": peak_time.isoformat(),
        "trough_time": trough_time.isoformat(),
        "recovery_time": recovery_time.isoformat() if recovery_time is not None else "",
        "duration_hours": duration_hours,
    }


def _max_drawdown_duration_hours(equity_curve: pd.Series) -> float:
    """Longest peak-to-recovery drawdown duration in hours."""
    eq = safe_series(equity_curve, name="equity_curve").astype(float)
    _validate_datetime_index(eq.index, name="equity_curve index", min_len=1)

    durations: list[float] = []
    peak_value = -np.inf
    current_peak_time: pd.Timestamp | None = None
    in_drawdown = False
    drawdown_peak_time: pd.Timestamp | None = None

    for ts, value in eq.items():
        if value >= peak_value:
            peak_value = float(value)
            current_peak_time = ts
            if in_drawdown and drawdown_peak_time is not None:
                durations.append((ts - drawdown_peak_time).total_seconds() / 3600.0)
                in_drawdown = False
                drawdown_peak_time = None
        else:
            if not in_drawdown:
                in_drawdown = True
                drawdown_peak_time = current_peak_time

    if in_drawdown and drawdown_peak_time is not None:
        durations.append((eq.index[-1] - drawdown_peak_time).total_seconds() / 3600.0)

    if not durations:
        return 0.0
    return float(max(durations))


def _to_python_scalar(v: Any) -> Any:
    """Convert numpy/pandas scalar values to plain Python scalars."""
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def compute_metrics(result: Dict[str, Any], rf: float = 0.0) -> Dict[str, float | int | str]:
    """Compute performance, risk, and trading metrics from a backtest result dict.

    Conventions:
    - Strict validation: required keys must exist, and equity/returns/pnl/positions
      must have exact index match with no NaNs.
    - All bar-level indexes must be DatetimeIndex, monotonic increasing, and unique.
    - Sharpe/Sortino/Calmar return NaN when denominator is undefined (not forced to 0/inf).
    - turnover_ann is bar-based annualized turnover:
      sum(trade_notional / equity_at_trade) / n_bars * bars_per_year.
    """
    missing_keys = [k for k in REQUIRED_RESULT_KEYS if k not in result]
    if missing_keys:
        raise ValueError(f"result missing required keys: {missing_keys}")

    equity = safe_series(result["equity_curve"], name="equity_curve").astype(float)
    returns = safe_series(result["returns"], name="returns").astype(float)
    pnl = safe_series(result["pnl"], name="pnl").astype(float)
    positions = safe_series(result["positions"], name="positions").astype(float)
    trades = result["trades"]

    if not isinstance(trades, pd.DataFrame):
        raise ValueError("result['trades'] must be a DataFrame")

    _validate_datetime_index(equity.index, name="equity_curve index", min_len=2)
    _validate_datetime_index(returns.index, name="returns index", min_len=2)
    _validate_datetime_index(pnl.index, name="pnl index", min_len=2)
    _validate_datetime_index(positions.index, name="positions index", min_len=2)

    if not returns.index.equals(equity.index):
        raise ValueError("returns index must exactly match equity_curve index")
    if not pnl.index.equals(equity.index):
        raise ValueError("pnl index must exactly match equity_curve index")
    if not positions.index.equals(equity.index):
        raise ValueError("positions index must exactly match equity_curve index")

    bars_per_year = infer_bars_per_year(equity.index)
    years = infer_years(equity.index)

    equity_start = float(equity.iloc[0])
    equity_end = float(equity.iloc[-1])

    if equity_start == 0:
        raise ValueError("equity start value cannot be zero")

    total_return = equity_end / equity_start - 1.0
    cagr = (equity_end / equity_start) ** (1.0 / years) - 1.0
    cumulative_pnl = float(pnl.sum())

    ann_vol = float(returns.std(ddof=0) * np.sqrt(bars_per_year))

    dd = compute_max_drawdown(equity)
    max_drawdown_pct = float(dd["mdd_pct"])
    max_drawdown_duration_hours = _max_drawdown_duration_hours(equity)

    downside = returns[returns < 0]
    downside_vol = float(downside.std(ddof=0) * np.sqrt(bars_per_year)) if len(downside) > 0 else np.nan

    var_95 = float(returns.quantile(0.05))
    var_99 = float(returns.quantile(0.01))
    cvar_tail = returns[returns <= var_95]
    cvar_95 = float(cvar_tail.mean()) if len(cvar_tail) > 0 else np.nan

    rf_per_bar = rf / bars_per_year
    mean_excess_return_per_bar = float(returns.mean() - rf_per_bar)
    mean_excess_return_annual = float(mean_excess_return_per_bar * bars_per_year)

    # Keep undefined denominator cases as NaN. Reporting can map NaN for display if needed.
    sharpe = mean_excess_return_annual / ann_vol if ann_vol > 0 else np.nan
    sortino = mean_excess_return_annual / downside_vol if downside_vol and downside_vol > 0 else np.nan
    calmar = cagr / abs(max_drawdown_pct) if max_drawdown_pct < 0 else np.nan

    avg_exposure = float(positions.abs().mean())
    time_in_market = float((positions != 0).mean())

    turnover_ann = np.nan
    total_fees = np.nan

    if "trade_notional" in trades.columns:
        trade_notional = trades["trade_notional"].fillna(0.0).astype(float).abs()

        if "timestamp" in trades.columns:
            trade_time = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")
        elif isinstance(trades.index, pd.DatetimeIndex):
            trade_time = trades.index
        else:
            raise ValueError(
                "trades must have 'timestamp' column or DatetimeIndex when trade_notional is provided"
            )

        trade_index = pd.DatetimeIndex(trade_time)
        if trade_index.isna().any():
            raise ValueError("trades contain invalid timestamps")

        missing_trade_timestamps = trade_index.difference(equity.index)
        if len(missing_trade_timestamps) > 0:
            raise ValueError("trade timestamps must align with equity_curve index")

        equity_at_trade = equity.reindex(trade_index).astype(float)
        turnover_values = trade_notional.to_numpy(dtype=float) / equity_at_trade.to_numpy(dtype=float)
        if not np.isfinite(turnover_values).all():
            raise ValueError("non-finite turnover values detected")

        # Bar-based annualization: sample turnover / n_bars * bars_per_year.
        turnover_per_bar_mean = float(turnover_values.sum() / len(equity))
        turnover_ann = float(turnover_per_bar_mean * bars_per_year)

        non_zero_trade_mask = trade_notional > 0
        n_trades = int(non_zero_trade_mask.sum())
    else:
        n_trades = int((positions.diff().fillna(0.0) != 0).sum())

    if "fee" in trades.columns:
        total_fees = float(trades["fee"].fillna(0.0).astype(float).sum())

    if np.isnan(total_fees) or cumulative_pnl == 0:
        cost_as_pct_pnl = np.nan
    else:
        cost_as_pct_pnl = float(total_fees / abs(cumulative_pnl))

    in_market_mask = positions != 0
    pnl_in_market = pnl[in_market_mask.reindex(pnl.index).fillna(False)]
    hit_rate_bars = float((pnl_in_market > 0).mean()) if len(pnl_in_market) > 0 else np.nan

    metrics: Dict[str, Any] = {
        "n_bars": int(len(equity)),
        "start_time": equity.index[0].isoformat(),
        "end_time": equity.index[-1].isoformat(),
        "years": float(years),
        "bars_per_year": float(bars_per_year),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "cumulative_pnl": float(cumulative_pnl),
        "ann_vol": float(ann_vol),
        "max_drawdown_pct": float(max_drawdown_pct),
        "max_drawdown_peak_time": dd["peak_time"],
        "max_drawdown_trough_time": dd["trough_time"],
        "max_drawdown_recovery_time": dd["recovery_time"],
        "max_drawdown_duration_hours": float(max_drawdown_duration_hours),
        "downside_vol": float(downside_vol) if not np.isnan(downside_vol) else np.nan,
        "var_95": float(var_95),
        "var_99": float(var_99),
        "cvar_95": float(cvar_95) if not np.isnan(cvar_95) else np.nan,
        "sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan,
        "sortino": float(sortino) if not np.isnan(sortino) else np.nan,
        "calmar": float(calmar) if not np.isnan(calmar) else np.nan,
        "avg_exposure": float(avg_exposure),
        "time_in_market": float(time_in_market),
        "turnover_ann": float(turnover_ann) if not np.isnan(turnover_ann) else np.nan,
        "n_trades": int(n_trades),
        "total_fees": float(total_fees) if not np.isnan(total_fees) else np.nan,
        "cost_as_pct_pnl": float(cost_as_pct_pnl) if not np.isnan(cost_as_pct_pnl) else np.nan,
        "hit_rate_bars": float(hit_rate_bars) if not np.isnan(hit_rate_bars) else np.nan,
    }

    return {k: _to_python_scalar(v) for k, v in metrics.items()}


def main() -> None:
    """Run a simple sanity check for metrics computation."""
    from btcusdt_regime_trading.backtest.engine import BacktestConfig, BacktestEngine
    from btcusdt_regime_trading.data.loaders import load_klines_1h_processed

    df = load_klines_1h_processed()

    target = pd.Series(0.0, index=df.index, name="target_position")
    target.iloc[100:] = 1.0

    engine = BacktestEngine(BacktestConfig())
    result = engine.run(df, target)

    metrics = compute_metrics(result)

    for key in sorted(metrics):
        print(f"{key}: {metrics[key]}")

    print("-- key metrics --")
    for key in [
        "total_return",
        "cagr",
        "sharpe",
        "max_drawdown_pct",
        "turnover_ann",
        "n_trades",
        "total_fees",
    ]:
        print(f"{key}: {metrics[key]}")





if __name__ == "__main__":
    main()
