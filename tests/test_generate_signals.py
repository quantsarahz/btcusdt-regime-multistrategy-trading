from __future__ import annotations

import pandas as pd

from bt_regime_system.regime.detect import R1, R2, R3, R4
from bt_regime_system.signals import build as mod


def _make_bars(start: str, periods: int) -> pd.DataFrame:
    ts = pd.date_range(start=start, periods=periods, freq="15min", tz="UTC")
    base = pd.Series(range(periods), dtype=float)
    close = 100.0 + base
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1.0,
        }
    )


def test_build_signals_15m_weighted_composite(monkeypatch) -> None:
    bars = _make_bars("2024-01-01T00:15:00Z", periods=4)
    regime = pd.DataFrame(
        {
            "timestamp": bars["timestamp"],
            "regime": [R1, R2, R3, R4],
        }
    )

    monkeypatch.setattr(
        mod,
        "generate_donchian_signal",
        lambda bars_15m, **kwargs: pd.DataFrame(
            {"timestamp": bars_15m["timestamp"], mod.DONCHIAN_COL: [1.0, 1.0, -1.0, 1.0]}
        ),
    )
    monkeypatch.setattr(
        mod,
        "generate_ema_adx_signal",
        lambda bars_15m, **kwargs: pd.DataFrame(
            {"timestamp": bars_15m["timestamp"], mod.EMA_ADX_COL: [1.0, -1.0, 1.0, 1.0]}
        ),
    )
    monkeypatch.setattr(
        mod,
        "generate_mean_reversion_signal",
        lambda bars_15m, **kwargs: pd.DataFrame(
            {"timestamp": bars_15m["timestamp"], mod.MEAN_REV_COL: [-1.0, 1.0, 1.0, -1.0]}
        ),
    )

    out = mod.build_signals_15m(bars_15m=bars, regime_15m=regime)

    assert list(out.columns) == [
        "timestamp",
        "signal_donchian",
        "signal_ema_adx",
        "signal_mean_reversion",
        "signal_composite",
        "target_position",
    ]
    assert out["signal_composite"].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert out["target_position"].tolist() == [1.0, 0.0, 1.0, 0.0]


def test_build_signals_15m_missing_regime_defaults_to_r4(monkeypatch) -> None:
    bars = _make_bars("2024-01-01T00:15:00Z", periods=2)
    regime = pd.DataFrame(
        {
            "timestamp": [bars["timestamp"].iloc[1]],
            "regime": [R1],
        }
    )

    monkeypatch.setattr(
        mod,
        "generate_donchian_signal",
        lambda bars_15m, **kwargs: pd.DataFrame({"timestamp": bars_15m["timestamp"], mod.DONCHIAN_COL: [1.0, 1.0]}),
    )
    monkeypatch.setattr(
        mod,
        "generate_ema_adx_signal",
        lambda bars_15m, **kwargs: pd.DataFrame({"timestamp": bars_15m["timestamp"], mod.EMA_ADX_COL: [0.0, 1.0]}),
    )
    monkeypatch.setattr(
        mod,
        "generate_mean_reversion_signal",
        lambda bars_15m, **kwargs: pd.DataFrame({"timestamp": bars_15m["timestamp"], mod.MEAN_REV_COL: [1.0, 0.0]}),
    )

    out = mod.build_signals_15m(bars_15m=bars, regime_15m=regime, default_regime=R4)

    assert out["target_position"].tolist() == [0.0, 1.0]


def test_build_signals_15m_execution_constraints_applied(monkeypatch) -> None:
    bars = _make_bars("2024-01-01T00:15:00Z", periods=6)
    regime = pd.DataFrame(
        {
            "timestamp": bars["timestamp"],
            "regime": [R1] * 6,
        }
    )

    monkeypatch.setattr(
        mod,
        "generate_donchian_signal",
        lambda bars_15m, **kwargs: pd.DataFrame(
            {"timestamp": bars_15m["timestamp"], mod.DONCHIAN_COL: [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]}
        ),
    )
    monkeypatch.setattr(
        mod,
        "generate_ema_adx_signal",
        lambda bars_15m, **kwargs: pd.DataFrame(
            {"timestamp": bars_15m["timestamp"], mod.EMA_ADX_COL: [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]}
        ),
    )
    monkeypatch.setattr(
        mod,
        "generate_mean_reversion_signal",
        lambda bars_15m, **kwargs: pd.DataFrame(
            {"timestamp": bars_15m["timestamp"], mod.MEAN_REV_COL: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        ),
    )

    out = mod.build_signals_15m(
        bars_15m=bars,
        regime_15m=regime,
        long_only=True,
        min_hold_bars=3,
        rebalance_threshold=0.25,
    )

    assert out["signal_composite"].tolist() == [0.0, 0.6, 1.0, 1.0, 0.0, 0.0]
    assert out["target_position"].tolist() == [0.0, 0.6, 0.6, 0.6, 0.0, 0.0]


def test_build_signals_15m_long_only_clips_negative_targets(monkeypatch) -> None:
    bars = _make_bars("2024-01-01T00:15:00Z", periods=3)
    regime = pd.DataFrame(
        {
            "timestamp": bars["timestamp"],
            "regime": [R3, R3, R3],
        }
    )

    monkeypatch.setattr(
        mod,
        "generate_donchian_signal",
        lambda bars_15m, **kwargs: pd.DataFrame({"timestamp": bars_15m["timestamp"], mod.DONCHIAN_COL: [0.0, 0.0, 0.0]}),
    )
    monkeypatch.setattr(
        mod,
        "generate_ema_adx_signal",
        lambda bars_15m, **kwargs: pd.DataFrame({"timestamp": bars_15m["timestamp"], mod.EMA_ADX_COL: [0.0, 0.0, 0.0]}),
    )
    monkeypatch.setattr(
        mod,
        "generate_mean_reversion_signal",
        lambda bars_15m, **kwargs: pd.DataFrame({"timestamp": bars_15m["timestamp"], mod.MEAN_REV_COL: [-1.0, -1.0, -1.0]}),
    )

    out = mod.build_signals_15m(
        bars_15m=bars,
        regime_15m=regime,
        long_only=True,
    )
    assert out["signal_composite"].tolist() == [-1.0, -1.0, -1.0]
    assert out["target_position"].tolist() == [0.0, 0.0, 0.0]


def test_run_generate_signals_writes_monthly_outputs(tmp_path, monkeypatch) -> None:
    bars_dir = tmp_path / "bars"
    regime_dir = tmp_path / "regime"
    out_dir = tmp_path / "signals"
    bars_dir.mkdir(parents=True)
    regime_dir.mkdir(parents=True)

    bars = _make_bars("2024-01-31T23:45:00Z", periods=3)
    regime = pd.DataFrame(
        {
            "timestamp": bars["timestamp"],
            "regime": [R1, R1, R1],
        }
    )

    bars.to_parquet(bars_dir / "BTCUSDT_15m_2024-02.parquet", index=False)
    regime.to_parquet(regime_dir / "BTCUSDT_regime_15m_2024-02.parquet", index=False)

    monkeypatch.setattr(
        mod,
        "generate_donchian_signal",
        lambda bars_15m, **kwargs: pd.DataFrame({"timestamp": bars_15m["timestamp"], mod.DONCHIAN_COL: [1.0, 1.0, 1.0]}),
    )
    monkeypatch.setattr(
        mod,
        "generate_ema_adx_signal",
        lambda bars_15m, **kwargs: pd.DataFrame({"timestamp": bars_15m["timestamp"], mod.EMA_ADX_COL: [0.0, 0.0, 0.0]}),
    )
    monkeypatch.setattr(
        mod,
        "generate_mean_reversion_signal",
        lambda bars_15m, **kwargs: pd.DataFrame({"timestamp": bars_15m["timestamp"], mod.MEAN_REV_COL: [0.0, 0.0, 0.0]}),
    )

    result = mod.run_generate_signals(
        bars_15m_path=bars_dir,
        regime_15m_path=regime_dir,
        output_dir=out_dir,
        symbol="BTCUSDT",
    )

    assert result["rows_bars_15m"] == 3
    assert result["rows_regime_15m"] == 3
    assert result["rows_out"] == 3
    assert result["long_only"] is False
    assert result["min_hold_bars"] == 1
    assert result["rebalance_threshold"] == 0.0

    files = sorted(out_dir.glob("BTCUSDT_signals_15m_2024-*.parquet"))
    assert len(files) == 2
    assert files[0].name == "BTCUSDT_signals_15m_2024-01.parquet"
    assert files[1].name == "BTCUSDT_signals_15m_2024-02.parquet"

    merged = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    assert len(merged) == 3
    assert set(merged.columns) == {
        "timestamp",
        "signal_donchian",
        "signal_ema_adx",
        "signal_mean_reversion",
        "signal_composite",
        "target_position",
    }
