from __future__ import annotations

import json

import pandas as pd

from bt_regime_system.data.qc_bars import build_bars_qc_report, run_qc_bars


def _make_bars(start: str, periods: int, freq: str) -> pd.DataFrame:
    ts = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    base = pd.Series(range(periods), dtype=float)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100.0 + base,
            "high": 101.0 + base,
            "low": 99.0 + base,
            "close": 100.5 + base,
            "volume": 1.0,
        }
    )


def test_build_bars_qc_report_detects_missing_interval() -> None:
    bars = _make_bars("2024-01-01T00:15:00Z", periods=3, freq="15min")
    bars = bars.drop(index=[1]).reset_index(drop=True)  # remove 00:30

    report = build_bars_qc_report(bars, freq="15m")

    assert report["row_count"] == 2
    assert report["missing_timestamp_count"] == 1
    assert report["duplicate_timestamp_count"] == 0
    assert report["invalid_ohlc_count"] == 0


def test_run_qc_bars_writes_monthly_and_summary_reports(tmp_path) -> None:
    bars_dir = tmp_path / "bars_1h"
    report_dir = tmp_path / "reports"
    bars_dir.mkdir(parents=True)

    jan = _make_bars("2024-01-01T01:00:00Z", periods=5, freq="1h")
    feb = _make_bars("2024-02-01T01:00:00Z", periods=5, freq="1h")

    jan.to_parquet(bars_dir / "BTCUSDT_1h_2024-01.parquet", index=False)
    feb.to_parquet(bars_dir / "BTCUSDT_1h_2024-02.parquet", index=False)

    result = run_qc_bars(input_path=bars_dir, freq="1h", report_dir=report_dir)

    assert result["files_processed"] == 2
    assert result["summary"]["row_count"] == 10

    monthly = sorted(report_dir.glob("qc_bars_1h_2024-*.json"))
    assert len(monthly) == 2

    summary_path = report_dir / "qc_bars_1h_summary.json"
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["files_processed"] == 2
    assert summary["global_duplicate_timestamp_count"] == 0
