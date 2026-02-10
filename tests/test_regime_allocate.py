from __future__ import annotations

import pandas as pd

from bt_regime_system.regime.allocate import build_allocation_table, weights_for_regime
from bt_regime_system.regime.detect import R1, R2, R3, R4


def test_weights_for_regime_matches_mapping() -> None:
    assert weights_for_regime(R1) == {"donchian": 0.6, "ema_adx": 0.4, "mean_reversion": 0.0}
    assert weights_for_regime(R2) == {"donchian": 0.4, "ema_adx": 0.4, "mean_reversion": 0.0}
    assert weights_for_regime(R3) == {"donchian": 0.0, "ema_adx": 0.0, "mean_reversion": 1.0}
    assert weights_for_regime(R4) == {"donchian": 0.0, "ema_adx": 0.0, "mean_reversion": 0.0}


def test_build_allocation_table_defaults_missing_to_r4() -> None:
    idx = pd.to_datetime(["2024-01-01T00:15:00Z", "2024-01-01T00:30:00Z", "2024-01-01T00:45:00Z"], utc=True)
    regime_series = pd.Series([R1, None, R3], index=idx)

    out = build_allocation_table(regime_series)

    assert out["regime"].tolist() == [R1, R4, R3]
    assert out["w_total"].tolist() == [1.0, 0.0, 1.0]
    assert out.loc[0, "w_donchian"] == 0.6
    assert out.loc[1, "w_mean_reversion"] == 0.0
    assert out.loc[2, "w_mean_reversion"] == 1.0
