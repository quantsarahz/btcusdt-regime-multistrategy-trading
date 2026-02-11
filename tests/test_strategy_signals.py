from __future__ import annotations

import pandas as pd

from bt_regime_system.strategies import donchian, ema_adx, mean_reversion


def _make_bars_15m(start: str, periods: int) -> pd.DataFrame:
    ts = pd.date_range(start=start, periods=periods, freq="15min", tz="UTC")
    base = pd.Series(range(periods), dtype=float)
    close = 100.0 + base
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.2,
            "high": close + 0.6,
            "low": close - 0.6,
            "close": close,
            "volume": 1.0,
        }
    )


def _assert_signal_frame(frame: pd.DataFrame, signal_col: str, expected_len: int) -> None:
    assert list(frame.columns) == ["timestamp", signal_col]
    assert len(frame) == expected_len
    assert pd.to_datetime(frame["timestamp"], utc=True).duplicated().sum() == 0
    assert frame[signal_col].dropna().isin([-1.0, 0.0, 1.0]).all()


def test_donchian_generate_signal_interface_and_behavior() -> None:
    bars = _make_bars_15m("2024-01-01T00:15:00Z", periods=8)

    out = donchian.generate_signal(bars, window=3, hold_until_opposite=True)

    _assert_signal_frame(out, "signal_donchian", expected_len=len(bars))
    assert out["signal_donchian"].iloc[:3].eq(0.0).all()
    assert out["signal_donchian"].iloc[-1] == 1.0


def test_ema_adx_generate_signal_interface_and_behavior() -> None:
    bars = _make_bars_15m("2024-01-01T00:15:00Z", periods=40)

    out = ema_adx.generate_signal(
        bars,
        ema_fast=3,
        ema_slow=8,
        adx_window=3,
        adx_threshold=0.0,
        use_adx_filter=False,
    )

    _assert_signal_frame(out, "signal_ema_adx", expected_len=len(bars))
    assert (out["signal_ema_adx"] == 1.0).sum() > 0
    assert (out["signal_ema_adx"] == -1.0).sum() == 0


def test_mean_reversion_generate_signal_interface_and_behavior() -> None:
    ts = pd.date_range("2024-01-01T00:15:00Z", periods=12, freq="15min", tz="UTC")
    close = pd.Series([100, 100, 100, 104, 103, 100, 97, 96, 98, 100, 101, 100], dtype=float)
    bars = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 1.0,
        }
    )

    out = mean_reversion.generate_signal(
        bars,
        z_window=3,
        entry_z=1.0,
        exit_z=0.2,
    )

    _assert_signal_frame(out, "signal_mean_reversion", expected_len=len(bars))
    assert (out["signal_mean_reversion"] == -1.0).sum() > 0
    assert (out["signal_mean_reversion"] == 1.0).sum() > 0


def test_mean_reversion_generate_signal_rejects_invalid_thresholds() -> None:
    bars = _make_bars_15m("2024-01-01T00:15:00Z", periods=10)

    try:
        mean_reversion.generate_signal(bars, z_window=3, entry_z=1.0, exit_z=1.2)
    except ValueError as exc:
        assert "exit_z" in str(exc)
    else:
        raise AssertionError("Expected ValueError for exit_z > entry_z")
