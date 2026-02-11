from __future__ import annotations

import pandas as pd
import pytest

from bt_regime_system.regime.detect import (
    R1,
    R2,
    R3,
    R4,
    apply_min_regime_run_filter,
    classify_regime,
    detect_regime_1h,
    run_detect_regime,
)


def _make_1h_bars(start: str, periods: int) -> pd.DataFrame:
    ts = pd.date_range(start=start, periods=periods, freq="1h", tz="UTC")
    base = pd.Series(range(periods), dtype=float)
    close = 100.0 + 0.2 * base + (base % 7) * 0.05
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.1,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": 1.0,
        }
    )


def test_classify_regime_4_quadrants() -> None:
    is_trend = pd.Series([True, True, False, False])
    is_high_vol = pd.Series([False, True, False, True])

    out = classify_regime(is_trend, is_high_vol)

    assert out.tolist() == [R1, R2, R3, R4]


def test_apply_min_regime_run_filter_three_bars() -> None:
    regime = pd.Series([R1, R2, R1, R2, R2, R2, R1, R1, R1], dtype="string")

    out = apply_min_regime_run_filter(regime, min_run_bars=3)

    assert out.tolist() == [R1, R1, R1, R1, R1, R2, R2, R2, R1]


def test_apply_min_regime_run_filter_invalid_min_run() -> None:
    regime = pd.Series([R1, R2], dtype="string")

    with pytest.raises(ValueError):
        _ = apply_min_regime_run_filter(regime, min_run_bars=0)


def test_detect_regime_1h_returns_expected_columns() -> None:
    bars = _make_1h_bars("2024-01-01T01:00:00Z", periods=240)

    out = detect_regime_1h(
        bars,
        ema_fast=12,
        ema_slow=48,
        adx_window=14,
        adx_threshold=15.0,
        atr_window=14,
        vol_lookback=96,
        high_vol_quantile=0.75,
    )

    assert len(out) == len(bars)
    assert list(out.columns) == [
        "timestamp",
        "regime",
        "is_trend",
        "is_high_vol",
        "ema_fast",
        "ema_slow",
        "adx",
        "atrp",
        "vol_threshold",
    ]
    assert pd.to_datetime(out["timestamp"], utc=True).duplicated().sum() == 0
    assert set(out["regime"].dropna().unique()).issubset({R1, R2, R3, R4})


def test_run_detect_regime_writes_monthly_outputs(tmp_path) -> None:
    bars_dir = tmp_path / "bars_1h"
    out_dir = tmp_path / "regime_1h"
    bars_dir.mkdir(parents=True)

    bars = _make_1h_bars("2024-01-31T23:00:00Z", periods=4)
    bars.to_parquet(bars_dir / "BTCUSDT_1h_2024-01.parquet", index=False)

    result = run_detect_regime(
        input_path=bars_dir,
        output_dir=out_dir,
        symbol="BTCUSDT",
        ema_fast=3,
        ema_slow=8,
        adx_window=3,
        adx_threshold=10.0,
        atr_window=3,
        vol_lookback=3,
        high_vol_quantile=0.5,
    )

    assert result["rows_in"] == 4
    assert result["rows_out"] == 4

    files = sorted(out_dir.glob("BTCUSDT_regime_1h_2024-*.parquet"))
    assert len(files) == 2
    assert files[0].name == "BTCUSDT_regime_1h_2024-01.parquet"
    assert files[1].name == "BTCUSDT_regime_1h_2024-02.parquet"
