from __future__ import annotations

import json

import pandas as pd

from bt_regime_system.data.qc_1m import build_qc_report, clean_1m_ohlcv, run_qc_1m


def test_clean_1m_fills_gap_and_marks_synthetic() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01T00:01:00Z",
                "2024-01-01T00:02:00Z",
                "2024-01-01T00:02:00Z",  # duplicate, keep last
                "2024-01-01T00:04:00Z",  # 00:03 missing
            ],
            "open": [100.0, 101.0, 111.0, 103.0],
            "high": [101.0, 102.0, 112.0, 104.0],
            "low": [99.0, 100.0, 110.0, 102.0],
            "close": [100.5, 101.5, 111.5, 103.5],
            "volume": [1.0, 2.0, 9.0, 4.0],
        }
    )

    clean = clean_1m_ohlcv(raw, fill_missing=True)

    assert len(clean) == 4
    assert clean["timestamp"].min() == pd.Timestamp("2024-01-01T00:01:00Z")
    assert clean["timestamp"].max() == pd.Timestamp("2024-01-01T00:04:00Z")

    t_0002 = clean.loc[clean["timestamp"] == pd.Timestamp("2024-01-01T00:02:00Z")].iloc[0]
    assert t_0002["open"] == 111.0

    t_0003 = clean.loc[clean["timestamp"] == pd.Timestamp("2024-01-01T00:03:00Z")].iloc[0]
    assert t_0003["is_synthetic"] == 1
    assert t_0003["volume"] == 0.0
    assert t_0003["open"] == t_0002["close"]


def test_clean_1m_drops_invalid_rows_without_fill() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01T00:01:00Z",
                "2024-01-01T00:02:00Z",
                "2024-01-01T00:03:00Z",
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 100.0, 103.0],  # invalid at 00:02 (high < low)
            "low": [99.0, 100.5, 101.0],
            "close": [100.5, 100.8, 102.5],
            "volume": [1.0, 2.0, -1.0],  # invalid at 00:03
        }
    )

    clean = clean_1m_ohlcv(raw, fill_missing=False)

    assert len(clean) == 1
    assert clean.iloc[0]["timestamp"] == pd.Timestamp("2024-01-01T00:01:00Z")
    assert clean.iloc[0]["is_synthetic"] == 0


def test_run_qc_1m_writes_clean_and_report(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    clean_dir = tmp_path / "clean"
    report_dir = tmp_path / "reports"
    raw_dir.mkdir(parents=True)

    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:01:00Z", "2024-01-01T00:03:00Z"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1.0, 2.0],
        }
    )
    raw_path = raw_dir / "BTCUSDT_1m_2024-01.parquet"
    raw.to_parquet(raw_path, index=False)

    summaries = run_qc_1m(input_path=raw_dir, output_dir=clean_dir, report_dir=report_dir)

    assert len(summaries) == 1
    clean_path = clean_dir / "BTCUSDT_1m_clean_2024-01.parquet"
    report_path = report_dir / "qc_clean_1m_2024-01.json"

    assert clean_path.exists()
    assert report_path.exists()

    cleaned = pd.read_parquet(clean_path)
    assert len(cleaned) == 3

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["row_count"] == 3
    assert report["missing_timestamp_count"] == 0
    assert "nan_count_by_column" in report


def test_build_qc_report_required_keys() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:01:00Z"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1.0],
        }
    )
    clean = clean_1m_ohlcv(raw)

    report = build_qc_report(raw, clean)

    required = {
        "row_count",
        "min_timestamp",
        "max_timestamp",
        "duplicate_timestamp_count",
        "missing_timestamp_count",
        "nan_count_by_column",
        "invalid_ohlc_count",
        "negative_volume_count",
    }
    assert required.issubset(report.keys())
