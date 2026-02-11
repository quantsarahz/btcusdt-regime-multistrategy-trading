from __future__ import annotations

import json

import pandas as pd

from bt_regime_system.regime.detect import R1, R2, R3, R4
from bt_regime_system.regime.qc import build_regime_15m_qc_report, run_qc_regime


def test_build_regime_15m_qc_report_detects_lookahead_and_mismatch() -> None:
    regime_lookup = pd.Series(
        [R1, R2],
        index=pd.to_datetime(["2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z"], utc=True),
        dtype="string",
    )

    regime_15m = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:15:00Z",
                    "2024-01-01T01:00:00Z",
                    "2024-01-01T01:15:00Z",
                    "2024-01-01T01:30:00Z",
                    "2024-01-01T01:45:00Z",
                ],
                utc=True,
            ),
            "regime": [R4, R1, R2, R3, R1],
            "regime_timestamp": pd.to_datetime(
                [
                    None,
                    "2024-01-01T01:00:00Z",
                    "2024-01-01T02:00:00Z",
                    "2024-01-01T01:00:00Z",
                    "2024-01-01T03:00:00Z",
                ],
                utc=True,
            ),
        }
    )

    report = build_regime_15m_qc_report(
        regime_15m=regime_15m,
        regime_lookup=regime_lookup,
        first_regime_ts=pd.Timestamp("2024-01-01T01:00:00Z"),
        default_regime=R4,
    )

    assert report["missing_timestamp_count"] == 2
    assert report["lookahead_violation_count"] == 2
    assert report["missing_regime_timestamp_count"] == 1
    assert report["unknown_regime_timestamp_count"] == 1
    assert report["regime_mismatch_count"] == 1
    assert report["default_before_first_count"] == 1
    assert report["non_default_before_first_count"] == 0


def test_run_qc_regime_writes_monthly_and_summary_reports(tmp_path) -> None:
    regime_1h_dir = tmp_path / "regime_1h"
    regime_15m_dir = tmp_path / "regime_15m"
    report_dir = tmp_path / "reports"
    regime_1h_dir.mkdir(parents=True)
    regime_15m_dir.mkdir(parents=True)

    regime_1h = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T01:00:00Z",
                    "2024-01-01T02:00:00Z",
                    "2024-01-01T03:00:00Z",
                ],
                utc=True,
            ),
            "regime": [R1, R2, R3],
            "is_trend": [True, True, False],
            "is_high_vol": [False, True, False],
            "ema_fast": [100.0, 101.0, 102.0],
            "ema_slow": [99.0, 99.5, 100.0],
            "adx": [22.0, 25.0, 15.0],
            "atrp": [0.01, 0.02, 0.015],
            "vol_threshold": [0.015, 0.015, 0.015],
        }
    )
    regime_1h.to_parquet(regime_1h_dir / "BTCUSDT_regime_1h_2024-01.parquet", index=False)

    regime_15m = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:15:00Z",
                    "2024-01-01T01:00:00Z",
                    "2024-01-01T01:15:00Z",
                    "2024-01-01T01:30:00Z",
                ],
                utc=True,
            ),
            "regime": [R4, R1, R1, R1],
            "regime_timestamp": pd.to_datetime(
                [
                    None,
                    "2024-01-01T01:00:00Z",
                    "2024-01-01T01:00:00Z",
                    "2024-01-01T01:00:00Z",
                ],
                utc=True,
            ),
        }
    )
    regime_15m.to_parquet(regime_15m_dir / "BTCUSDT_regime_15m_2024-01.parquet", index=False)

    result = run_qc_regime(
        regime_1h_path=regime_1h_dir,
        regime_15m_path=regime_15m_dir,
        report_dir=report_dir,
        symbol="BTCUSDT",
        default_regime=R4,
    )

    assert result["regime_1h"]["files_processed"] == 1
    assert result["regime_15m"]["files_processed"] == 1

    monthly_1h = sorted(p for p in report_dir.glob("qc_regime_1h_*.json") if "summary" not in p.name)
    monthly_15m = sorted(p for p in report_dir.glob("qc_regime_15m_*.json") if "summary" not in p.name)
    assert len(monthly_1h) == 1
    assert len(monthly_15m) == 1

    summary_1h_path = report_dir / "qc_regime_1h_summary.json"
    summary_15m_path = report_dir / "qc_regime_15m_summary.json"
    assert summary_1h_path.exists()
    assert summary_15m_path.exists()

    summary_1h = json.loads(summary_1h_path.read_text(encoding="utf-8"))
    summary_15m = json.loads(summary_15m_path.read_text(encoding="utf-8"))

    assert summary_1h["row_count"] == 3
    assert summary_1h["global_missing_timestamp_count"] == 0

    assert summary_15m["row_count"] == 4
    assert summary_15m["lookahead_violation_count"] == 0
    assert summary_15m["regime_mismatch_count"] == 0
