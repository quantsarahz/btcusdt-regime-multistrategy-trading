import numpy as np
import pandas as pd

from bt_regime_system.data.build_bars import build_15m_and_1h


def test_build_15m_and_1h_shapes() -> None:
    idx = pd.date_range("2024-01-01", periods=120, freq="1min", tz="UTC")
    base = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    df = pd.DataFrame(
        {
            "open": base,
            "high": base + 1,
            "low": base - 1,
            "close": base,
            "volume": 1.0,
        }
    )

    bars_15m, bars_1h = build_15m_and_1h(df)

    assert len(bars_15m) > 0
    assert len(bars_1h) > 0
    assert all(col in bars_15m.columns for col in ["open", "high", "low", "close", "volume"])
    assert all(col in bars_1h.columns for col in ["open", "high", "low", "close", "volume"])
