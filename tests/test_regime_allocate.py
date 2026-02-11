from __future__ import annotations

import pandas as pd
import pytest

from bt_regime_system.regime.allocate import build_allocation_table, weights_for_regime
from bt_regime_system.regime.detect import R1, R2, R3, R4


def test_weights_for_regime_matches_mapping() -> None:
    assert weights_for_regime(R1) == {"donchian": 0.6, "ema_adx": 0.4, "mean_reversion": 0.0}
    assert weights_for_regime(R2) == {"donchian": 0.4, "ema_adx": 0.4, "mean_reversion": 0.0}
    assert weights_for_regime(R3) == {"donchian": 0.0, "ema_adx": 0.0, "mean_reversion": 1.0}
    assert weights_for_regime(R4) == {"donchian": 0.0, "ema_adx": 0.0, "mean_reversion": 0.0}


def test_weights_for_regime_supports_overrides() -> None:
    overrides = {
        R1: {"donchian": 0.5, "ema_adx": 0.3, "mean_reversion": 0.0},
    }
    assert weights_for_regime(R1, regime_allocations=overrides) == {
        "donchian": 0.5,
        "ema_adx": 0.3,
        "mean_reversion": 0.0,
    }


def test_build_allocation_table_defaults_missing_to_r4() -> None:
    idx = pd.to_datetime(["2024-01-01T00:15:00Z", "2024-01-01T00:30:00Z", "2024-01-01T00:45:00Z"], utc=True)
    regime_series = pd.Series([R1, None, R3], index=idx)

    out = build_allocation_table(regime_series)

    assert out["regime"].tolist() == [R1, R4, R3]
    assert out["w_total"].tolist() == [1.0, 0.0, 1.0]
    assert out.loc[0, "w_donchian"] == 0.6
    assert out.loc[1, "w_mean_reversion"] == 0.0
    assert out.loc[2, "w_mean_reversion"] == 1.0


def test_build_allocation_table_unknown_regime_uses_default_regime() -> None:
    idx = pd.to_datetime(["2024-01-01T00:15:00Z"], utc=True)
    regime_series = pd.Series(["UNKNOWN"], index=idx)

    out = build_allocation_table(regime_series, default_regime=R4)
    assert out["regime"].tolist() == [R4]
    assert out["w_total"].tolist() == [0.0]


def test_build_allocation_table_rejects_negative_weight() -> None:
    idx = pd.to_datetime(["2024-01-01T00:15:00Z"], utc=True)
    regime_series = pd.Series([R1], index=idx)

    with pytest.raises(ValueError, match="non-negative"):
        build_allocation_table(
            regime_series,
            regime_allocations={R1: {"donchian": -0.1, "ema_adx": 0.4, "mean_reversion": 0.0}},
        )


def test_build_allocation_table_rejects_total_weight_over_one() -> None:
    idx = pd.to_datetime(["2024-01-01T00:15:00Z"], utc=True)
    regime_series = pd.Series([R1], index=idx)

    with pytest.raises(ValueError, match="exceed 1.0"):
        build_allocation_table(
            regime_series,
            regime_allocations={R1: {"donchian": 0.8, "ema_adx": 0.5, "mean_reversion": 0.0}},
        )
