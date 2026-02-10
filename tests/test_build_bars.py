from __future__ import annotations

import pandas as pd

from bt_regime_system.data.build_bars import build_15m_and_1h, run_build_bars


def _make_clean_frame(start: str, periods: int) -> pd.DataFrame:
    ts = pd.date_range(start=start, periods=periods, freq="1min", tz="UTC")
    base = pd.Series(range(periods), dtype=float)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100.0 + base,
            "high": 101.0 + base,
            "low": 99.0 + base,
            "close": 100.5 + base,
            "volume": 1.0,
            "is_synthetic": 0,
        }
    )


def test_build_15m_and_1h_basic_aggregation() -> None:
    clean = _make_clean_frame("2024-01-01T00:01:00Z", periods=120)

    bars_15m, bars_1h = build_15m_and_1h(clean)

    assert len(bars_15m) == 8
    assert len(bars_1h) == 2

    first_15m = bars_15m.iloc[0]
    assert first_15m["timestamp"] == pd.Timestamp("2024-01-01T00:15:00Z")
    assert first_15m["open"] == 100.0
    assert first_15m["close"] == 114.5
    assert first_15m["high"] == 115.0
    assert first_15m["low"] == 99.0
    assert first_15m["volume"] == 15.0


def test_run_build_bars_writes_monthly_outputs(tmp_path) -> None:
    clean_dir = tmp_path / "clean"
    out_15m = tmp_path / "bars_15m"
    out_1h = tmp_path / "bars_1h"
    clean_dir.mkdir(parents=True)

    jan_tail = _make_clean_frame("2024-01-31T23:56:00Z", periods=5)
    feb_head = _make_clean_frame("2024-02-01T00:01:00Z", periods=15)

    jan_tail.to_parquet(clean_dir / "BTCUSDT_1m_clean_2024-01.parquet", index=False)
    feb_head.to_parquet(clean_dir / "BTCUSDT_1m_clean_2024-02.parquet", index=False)

    summary = run_build_bars(
        input_path=clean_dir,
        output_15m_dir=out_15m,
        output_1h_dir=out_1h,
        symbol="BTCUSDT",
    )

    assert summary["rows_in"] == 20
    assert summary["rows_15m"] > 0
    assert summary["rows_1h"] > 0

    f15 = sorted(out_15m.glob("BTCUSDT_15m_*.parquet"))
    f1h = sorted(out_1h.glob("BTCUSDT_1h_*.parquet"))

    assert len(f15) == 2
    assert len(f1h) == 2
    assert f15[0].name == "BTCUSDT_15m_2024-01.parquet"
    assert f15[1].name == "BTCUSDT_15m_2024-02.parquet"
    assert f1h[0].name == "BTCUSDT_1h_2024-01.parquet"
    assert f1h[1].name == "BTCUSDT_1h_2024-02.parquet"

    bars_15m = pd.concat([pd.read_parquet(p) for p in f15], ignore_index=True)
    ts_15m = pd.to_datetime(bars_15m["timestamp"], utc=True)
    assert ts_15m.duplicated().sum() == 0
