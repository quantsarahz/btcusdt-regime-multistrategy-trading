from __future__ import annotations

import pandas as pd
import pytest

from bt_regime_system.analysis.plots import run_plot_price_regime_1h
from bt_regime_system.regime.detect import R1, R2, R3


def test_run_plot_price_regime_1h_writes_png(tmp_path) -> None:
    pytest.importorskip("matplotlib")

    bars_dir = tmp_path / "bars_1h"
    regime_dir = tmp_path / "regime_1h"
    out_dir = tmp_path / "plots"
    bars_dir.mkdir(parents=True)
    regime_dir.mkdir(parents=True)

    ts = pd.date_range("2024-01-01T01:00:00Z", periods=6, freq="1h", tz="UTC")
    bars = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100.5, 101.2, 102.4, 103.1, 104.0, 105.2],
            "volume": [1, 1, 1, 1, 1, 1],
        }
    )
    regime = pd.DataFrame(
        {
            "timestamp": ts,
            "regime": [R1, R1, R2, R2, R3, R3],
        }
    )

    bars.to_parquet(bars_dir / "BTCUSDT_1h_2024-01.parquet", index=False)
    regime.to_parquet(regime_dir / "BTCUSDT_regime_1h_2024-01.parquet", index=False)

    summary = run_plot_price_regime_1h(
        bars_1h_path=bars_dir,
        regime_1h_path=regime_dir,
        output_path=out_dir,
        symbol="BTCUSDT",
    )

    output_path = summary["output_path"]
    assert output_path.exists()
    assert output_path.suffix == ".png"
    assert output_path.stat().st_size > 0
    assert summary["rows_plot"] == 6
