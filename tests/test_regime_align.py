from __future__ import annotations

import pandas as pd

from bt_regime_system.regime.align import align_regime_to_15m, run_align_regime
from bt_regime_system.regime.detect import R1, R2, R4


def test_align_regime_to_15m_no_lookahead() -> None:
    regime_1h = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z"], utc=True),
            "regime": [R1, R2],
        }
    )
    bars_15m = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:15:00Z",
                    "2024-01-01T00:30:00Z",
                    "2024-01-01T00:45:00Z",
                    "2024-01-01T01:00:00Z",
                    "2024-01-01T01:15:00Z",
                    "2024-01-01T01:30:00Z",
                    "2024-01-01T01:45:00Z",
                    "2024-01-01T02:00:00Z",
                    "2024-01-01T02:15:00Z",
                ],
                utc=True,
            )
        }
    )

    out = align_regime_to_15m(regime_1h=regime_1h, bars_15m=bars_15m, default_regime=R4)

    assert out["regime"].tolist() == [R4, R4, R4, R1, R1, R1, R1, R2, R2]
    assert pd.isna(out.loc[2, "regime_timestamp"])
    assert out.loc[4, "regime_timestamp"] == pd.Timestamp("2024-01-01T01:00:00Z")
    assert out.loc[8, "regime_timestamp"] == pd.Timestamp("2024-01-01T02:00:00Z")


def test_run_align_regime_writes_monthly_outputs(tmp_path) -> None:
    regime_dir = tmp_path / "regime_1h"
    bars_dir = tmp_path / "bars_15m"
    out_dir = tmp_path / "regime_15m"
    regime_dir.mkdir(parents=True)
    bars_dir.mkdir(parents=True)

    regime_1h = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-31T23:00:00Z", "2024-02-01T00:00:00Z", "2024-02-01T01:00:00Z"],
                utc=True,
            ),
            "regime": [R1, R1, R2],
        }
    )
    regime_1h.to_parquet(regime_dir / "BTCUSDT_regime_1h_2024-01.parquet", index=False)

    bars_15m = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-31T23:45:00Z",
                    "2024-02-01T00:00:00Z",
                    "2024-02-01T00:15:00Z",
                    "2024-02-01T00:30:00Z",
                    "2024-02-01T00:45:00Z",
                    "2024-02-01T01:00:00Z",
                ],
                utc=True,
            ),
            "open": [1.0] * 6,
            "high": [1.0] * 6,
            "low": [1.0] * 6,
            "close": [1.0] * 6,
            "volume": [1.0] * 6,
        }
    )
    bars_15m.to_parquet(bars_dir / "BTCUSDT_15m_2024-02.parquet", index=False)

    result = run_align_regime(
        regime_1h_path=regime_dir,
        bars_15m_path=bars_dir,
        output_dir=out_dir,
        symbol="BTCUSDT",
        default_regime=R4,
    )

    assert result["rows_regime_1h"] == 3
    assert result["rows_bars_15m"] == 6
    assert result["rows_aligned"] == 6

    files = sorted(out_dir.glob("BTCUSDT_regime_15m_2024-*.parquet"))
    assert len(files) == 2
    assert files[0].name == "BTCUSDT_regime_15m_2024-01.parquet"
    assert files[1].name == "BTCUSDT_regime_15m_2024-02.parquet"
