from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def _as_series(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be pandas.Series")
    return pd.to_numeric(series, errors="coerce")


def sma(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    s = _as_series(series, "series")
    mp = window if min_periods is None else min_periods
    return s.rolling(window=window, min_periods=mp).mean()


def ema(series: pd.Series, span: int, min_periods: int | None = None) -> pd.Series:
    s = _as_series(series, "series")
    mp = span if min_periods is None else min_periods
    return s.ewm(span=span, adjust=False, min_periods=mp).mean()


def rolling_std(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    s = _as_series(series, "series")
    mp = window if min_periods is None else min_periods
    return s.rolling(window=window, min_periods=mp).std(ddof=0)


def returns(series: pd.Series, method: Literal["pct", "log"] = "pct") -> pd.Series:
    s = _as_series(series, "series")
    if method == "pct":
        return s.pct_change()
    if method == "log":
        ratio = s / s.shift(1)
        return np.log(ratio.where(ratio > 0))
    raise ValueError("method must be 'pct' or 'log'")


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    h = _as_series(high, "high")
    l = _as_series(low, "low")
    c = _as_series(close, "close")

    prev_close = c.shift(1)
    tr_components = pd.concat(
        [(h - l).abs(), (h - prev_close).abs(), (l - prev_close).abs()],
        axis=1,
    )
    return tr_components.max(axis=1)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    method: Literal["wilder", "sma"] = "wilder",
) -> pd.Series:
    tr = true_range(high, low, close)
    if method == "sma":
        return tr.rolling(window=window, min_periods=window).mean()
    if method == "wilder":
        return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    raise ValueError("method must be 'wilder' or 'sma'")


def atr_percent(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    method: Literal["wilder", "sma"] = "wilder",
) -> pd.Series:
    c = _as_series(close, "close")
    a = atr(high, low, c, window=window, method=method)
    denom = c.replace(0, pd.NA)
    return a / denom


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    h = _as_series(high, "high")
    l = _as_series(low, "low")
    c = _as_series(close, "close")

    up_move = h.diff()
    down_move = -l.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr_wilder = atr(h, l, c, window=window, method="wilder")
    plus_dm_smooth = plus_dm.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    plus_di = 100.0 * plus_dm_smooth / atr_wilder.replace(0, pd.NA)
    minus_di = 100.0 * minus_dm_smooth / atr_wilder.replace(0, pd.NA)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)
    return dx.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    s = _as_series(series, "series")
    mean = sma(s, window=window, min_periods=min_periods)
    std = rolling_std(s, window=window, min_periods=min_periods)
    return (s - mean) / std.replace(0, pd.NA)


def donchian_high(high: pd.Series, window: int, shift: int = 1) -> pd.Series:
    h = _as_series(high, "high")
    return h.rolling(window=window, min_periods=window).max().shift(shift)


def donchian_low(low: pd.Series, window: int, shift: int = 1) -> pd.Series:
    l = _as_series(low, "low")
    return l.rolling(window=window, min_periods=window).min().shift(shift)


def build_regime_features_1h(
    bars_1h: pd.DataFrame,
    ema_fast: int = 24,
    ema_slow: int = 96,
    adx_window: int = 14,
    adx_threshold: float = 20.0,
    ema_gap_threshold: float = 0.0,
    atr_window: int = 14,
    vol_lookback: int = 720,
    high_vol_quantile: float = 0.75,
) -> pd.DataFrame:
    req = {"timestamp", "high", "low", "close"}
    missing = req.difference(bars_1h.columns)
    if missing:
        raise ValueError(f"bars_1h missing columns: {sorted(missing)}")

    out = bars_1h.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp").reset_index(drop=True)

    out["ema_fast"] = ema(out["close"], span=ema_fast)
    out["ema_slow"] = ema(out["close"], span=ema_slow)
    out["adx"] = adx(out["high"], out["low"], out["close"], window=adx_window)
    out["atrp"] = atr_percent(out["high"], out["low"], out["close"], window=atr_window)

    min_periods = min(max(100, atr_window * 3), vol_lookback)
    out["vol_threshold"] = out["atrp"].rolling(vol_lookback, min_periods=min_periods).quantile(high_vol_quantile)
    fallback_threshold = out["atrp"].expanding(min_periods=min_periods).quantile(high_vol_quantile)
    out["vol_threshold"] = out["vol_threshold"].fillna(fallback_threshold)

    trend_strength = out["adx"] >= adx_threshold
    ema_gap = (out["ema_fast"] - out["ema_slow"]).abs() / out["close"].replace(0, pd.NA)
    trend_gap_ok = ema_gap >= ema_gap_threshold

    out["is_trend"] = (trend_strength & trend_gap_ok).fillna(False)
    out["is_high_vol"] = (out["atrp"] >= out["vol_threshold"]).fillna(False)

    return out


def build_donchian_features_15m(bars_15m: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    req = {"timestamp", "high", "low", "close"}
    missing = req.difference(bars_15m.columns)
    if missing:
        raise ValueError(f"bars_15m missing columns: {sorted(missing)}")

    out = bars_15m.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp").reset_index(drop=True)
    out["donchian_high"] = donchian_high(out["high"], window=window)
    out["donchian_low"] = donchian_low(out["low"], window=window)
    return out


def build_ema_adx_features_15m(
    bars_15m: pd.DataFrame,
    ema_fast: int = 21,
    ema_slow: int = 55,
    adx_window: int = 14,
) -> pd.DataFrame:
    req = {"timestamp", "high", "low", "close"}
    missing = req.difference(bars_15m.columns)
    if missing:
        raise ValueError(f"bars_15m missing columns: {sorted(missing)}")

    out = bars_15m.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp").reset_index(drop=True)
    out["ema_fast"] = ema(out["close"], span=ema_fast)
    out["ema_slow"] = ema(out["close"], span=ema_slow)
    out["adx"] = adx(out["high"], out["low"], out["close"], window=adx_window)
    return out


def build_mean_reversion_features_15m(bars_15m: pd.DataFrame, window: int = 48) -> pd.DataFrame:
    req = {"timestamp", "close"}
    missing = req.difference(bars_15m.columns)
    if missing:
        raise ValueError(f"bars_15m missing columns: {sorted(missing)}")

    out = bars_15m.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp").reset_index(drop=True)

    out["close_mean"] = sma(out["close"], window=window)
    out["close_std"] = rolling_std(out["close"], window=window)
    out["close_zscore"] = zscore(out["close"], window=window)
    return out
