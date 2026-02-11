from __future__ import annotations

import json

import pandas as pd
import pytest

from bt_regime_system.backtest import walk_forward as wf
from bt_regime_system.regime.detect import R1, R4


def _make_bars() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01T00:15:00Z", periods=8, freq="15min", tz="UTC")
    close = pd.Series([100, 101, 102, 103, 104, 105, 106, 107], dtype=float)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        }
    )


def _make_bars_train_down_test_up() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01T00:15:00Z", periods=8, freq="15min", tz="UTC")
    close = pd.Series([100, 99, 98, 97, 98, 99, 100, 101], dtype=float)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        }
    )


def _base_alloc() -> dict[str, dict[str, float]]:
    return {
        R1: {"donchian": 0.6, "ema_adx": 0.4, "mean_reversion": 0.0},
        "R2": {"donchian": 0.4, "ema_adx": 0.4, "mean_reversion": 0.0},
        "R3": {"donchian": 0.0, "ema_adx": 0.0, "mean_reversion": 1.0},
        R4: {"donchian": 0.0, "ema_adx": 0.0, "mean_reversion": 0.0},
    }


def test_run_walk_forward_selects_best_candidate_and_writes_outputs(tmp_path, monkeypatch) -> None:
    bars_dir = tmp_path / "bars"
    regime_dir = tmp_path / "regime"
    out_dir = tmp_path / "metrics"
    bars_dir.mkdir(parents=True)
    regime_dir.mkdir(parents=True)

    bars = _make_bars()
    regime = pd.DataFrame(
        {
            "timestamp": bars["timestamp"],
            "regime": [R1] * len(bars),
        }
    )

    bars.to_parquet(bars_dir / "BTCUSDT_15m_2024-01.parquet", index=False)
    regime.to_parquet(regime_dir / "BTCUSDT_regime_15m_2024-01.parquet", index=False)

    def fake_build_signals_15m(
        bars_15m: pd.DataFrame,
        regime_15m: pd.DataFrame,
        regime_allocations: dict | None = None,
        **_: object,
    ) -> pd.DataFrame:
        alloc = regime_allocations or {}
        r1 = alloc.get("R1", {})
        donchian_w = float(r1.get("donchian", 0.0))

        # Candidate with higher R1 donchian weight gets long exposure; other stays flat.
        target = [1.0] * len(bars_15m) if donchian_w > 0.8 else [0.0] * len(bars_15m)
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(bars_15m["timestamp"], utc=True),
                "target_position": target,
            }
        )

    monkeypatch.setattr(wf, "build_signals_15m", fake_build_signals_15m)

    folds = [
        {
            "name": "wf_small",
            "train_start": "2024-01-01T00:15:00Z",
            "train_end": "2024-01-01T01:00:00Z",
            "test_start": "2024-01-01T01:15:00Z",
            "test_end": "2024-01-01T02:00:00Z",
        }
    ]

    candidates = {
        "good": {"R1": {"donchian": 0.9, "ema_adx": 0.0, "mean_reversion": 0.0}},
        "bad": {"R1": {"donchian": 0.1, "ema_adx": 0.0, "mean_reversion": 0.0}},
    }

    summary = wf.run_walk_forward(
        bars_15m_path=bars_dir,
        regime_15m_path=regime_dir,
        output_dir=out_dir,
        symbol="BTCUSDT",
        base_regime_allocations=_base_alloc(),
        candidate_allocations=candidates,
        folds=folds,
        selection_metric="sharpe",
        initial_equity=1000.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        position_lag_bars=0,
        long_only=True,
    )

    assert summary["rows_bars"] == 8
    assert summary["rows_regime"] == 8
    assert summary["candidate_count"] == 3  # current + good + bad
    assert summary["fold_count"] == 1
    assert summary["selected_candidate_count"].get("good", 0) == 1

    folds_df = pd.read_csv(out_dir / "BTCUSDT_walk_forward_folds.csv")
    assert folds_df.loc[0, "selected_candidate"] == "good"
    assert folds_df.loc[0, "selection_scope"] == "train"
    assert pd.isna(folds_df.loc[0, "valid_score"])
    assert float(folds_df.loc[0, "test_total_return"]) > 0.0

    summary_payload = json.loads((out_dir / "BTCUSDT_walk_forward_summary.json").read_text(encoding="utf-8"))
    assert summary_payload["oos_metrics"]["total_return"] > 0.0


def test_run_walk_forward_without_valid_uses_train_for_selection(tmp_path, monkeypatch) -> None:
    bars_dir = tmp_path / "bars"
    regime_dir = tmp_path / "regime"
    out_dir = tmp_path / "metrics"
    bars_dir.mkdir(parents=True)
    regime_dir.mkdir(parents=True)

    bars = _make_bars_train_down_test_up()
    regime = pd.DataFrame(
        {
            "timestamp": bars["timestamp"],
            "regime": [R1] * len(bars),
        }
    )

    bars.to_parquet(bars_dir / "BTCUSDT_15m_2024-01.parquet", index=False)
    regime.to_parquet(regime_dir / "BTCUSDT_regime_15m_2024-01.parquet", index=False)

    def fake_build_signals_15m(
        bars_15m: pd.DataFrame,
        regime_15m: pd.DataFrame,
        regime_allocations: dict | None = None,
        **_: object,
    ) -> pd.DataFrame:
        alloc = regime_allocations or {}
        r1 = alloc.get("R1", {})
        donchian_w = float(r1.get("donchian", 0.0))

        # High weight candidate is short-all-bars (better in train), others long-all-bars (better in test).
        target = [-1.0] * len(bars_15m) if donchian_w > 0.8 else [1.0] * len(bars_15m)
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(bars_15m["timestamp"], utc=True),
                "target_position": target,
            }
        )

    monkeypatch.setattr(wf, "build_signals_15m", fake_build_signals_15m)

    folds = [
        {
            "name": "wf_train_only_selection",
            "train_start": "2024-01-01T00:15:00Z",
            "train_end": "2024-01-01T01:00:00Z",
            "test_start": "2024-01-01T01:15:00Z",
            "test_end": "2024-01-01T02:00:00Z",
        }
    ]

    candidates = {
        "train_edge": {"R1": {"donchian": 0.9, "ema_adx": 0.0, "mean_reversion": 0.0}},
        "leak_edge": {"R1": {"donchian": 0.1, "ema_adx": 0.0, "mean_reversion": 0.0}},
    }

    summary = wf.run_walk_forward(
        bars_15m_path=bars_dir,
        regime_15m_path=regime_dir,
        output_dir=out_dir,
        symbol="BTCUSDT",
        base_regime_allocations=_base_alloc(),
        candidate_allocations=candidates,
        folds=folds,
        selection_metric="total_return",
        initial_equity=1000.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        position_lag_bars=0,
        long_only=False,
    )

    assert summary["selected_candidate_count"].get("train_edge", 0) == 1

    folds_df = pd.read_csv(out_dir / "BTCUSDT_walk_forward_folds.csv")
    assert folds_df.loc[0, "selection_scope"] == "train"
    assert folds_df.loc[0, "selected_candidate"] == "train_edge"

    scores_df = pd.read_csv(out_dir / "BTCUSDT_walk_forward_scores.csv").set_index("candidate")
    assert float(scores_df.loc["train_edge", "selection_score"]) > float(scores_df.loc["leak_edge", "selection_score"])
    assert float(scores_df.loc["train_edge", "test_total_return"]) < float(scores_df.loc["leak_edge", "test_total_return"])


def test_run_walk_forward_empty_bars_returns_no_files(tmp_path) -> None:
    bars_dir = tmp_path / "bars"
    regime_dir = tmp_path / "regime"
    out_dir = tmp_path / "metrics"
    bars_dir.mkdir(parents=True)
    regime_dir.mkdir(parents=True)

    summary = wf.run_walk_forward(
        bars_15m_path=bars_dir,
        regime_15m_path=regime_dir,
        output_dir=out_dir,
        symbol="BTCUSDT",
    )

    assert summary["rows_bars"] == 0
    assert summary["fold_count"] == 0
    assert summary["files_written"] == []



def test_parse_folds_rejects_overlapping_test_windows() -> None:
    folds = [
        {
            "name": "f1",
            "train_start": "2024-01-01T00:00:00Z",
            "train_end": "2024-01-31T23:59:59Z",
            "test_start": "2024-02-01T00:00:00Z",
            "test_end": "2024-02-29T23:59:59Z",
        },
        {
            "name": "f2",
            "train_start": "2024-01-01T00:00:00Z",
            "train_end": "2024-01-31T23:59:59Z",
            "test_start": "2024-02-15T00:00:00Z",
            "test_end": "2024-03-15T23:59:59Z",
        },
    ]

    with pytest.raises(ValueError, match="test windows overlap"):
        wf._parse_folds(
            folds_cfg=folds,
            start_ts=pd.Timestamp("2024-01-01T00:00:00Z"),
            end_ts=pd.Timestamp("2024-12-31T23:59:59Z"),
        )


def test_parse_folds_rejects_non_chronological_split_order() -> None:
    folds = [
        {
            "name": "bad_order",
            "train_start": "2024-01-01T00:00:00Z",
            "train_end": "2024-06-30T23:59:59Z",
            "valid_start": "2024-06-15T00:00:00Z",
            "valid_end": "2024-07-31T23:59:59Z",
            "test_start": "2024-08-01T00:00:00Z",
            "test_end": "2024-12-31T23:59:59Z",
        }
    ]

    with pytest.raises(ValueError, match="train_end < valid_start"):
        wf._parse_folds(
            folds_cfg=folds,
            start_ts=pd.Timestamp("2024-01-01T00:00:00Z"),
            end_ts=pd.Timestamp("2024-12-31T23:59:59Z"),
        )
