from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bt_regime_system.backtest.engine import simulate_backtest_15m
from bt_regime_system.backtest.metrics import compute_metrics
from bt_regime_system.regime.allocate import DEFAULT_REGIME_ALLOCATIONS
from bt_regime_system.regime.detect import R4
from bt_regime_system.signals.build import build_signals_15m
from bt_regime_system.utils.io import ensure_dir

REQUIRED_BARS_COLUMNS = ["timestamp", "high", "low", "close"]
REQUIRED_REGIME_COLUMNS = ["timestamp", "regime"]

SELECTION_METRICS = {
    "sharpe": "sharpe",
    "total_return": "total_return",
    "annual_return": "annual_return",
    "calmar": "calmar",
}


@dataclass
class Fold:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp | None
    valid_end: pd.Timestamp | None
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _collect_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob(pattern))
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _to_timestamp(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {value}")
    return pd.Timestamp(ts)


def _to_month(timestamp: pd.Series) -> pd.Series:
    ts = pd.to_datetime(timestamp, utc=True, errors="coerce") - pd.Timedelta(minutes=1)
    return ts.dt.strftime("%Y-%m")


def _standardize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    missing = set(REQUIRED_BARS_COLUMNS).difference(out.columns)
    if missing:
        raise ValueError(f"bars frame missing columns: {sorted(missing)}")

    out = out[REQUIRED_BARS_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for col in ["high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["timestamp", "high", "low", "close"])
    out = out.sort_values("timestamp", kind="mergesort").drop_duplicates("timestamp", keep="last")
    return out.reset_index(drop=True)


def _standardize_regime(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    missing = set(REQUIRED_REGIME_COLUMNS).difference(out.columns)
    if missing:
        raise ValueError(f"regime frame missing columns: {sorted(missing)}")

    out = out[REQUIRED_REGIME_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["regime"] = out["regime"].astype("string")

    out = out.dropna(subset=["timestamp"])
    out = out.sort_values("timestamp", kind="mergesort").drop_duplicates("timestamp", keep="last")
    return out.reset_index(drop=True)


def _concat_standardized(files: list[Path], standardizer: Any, empty_columns: list[str]) -> pd.DataFrame:
    if not files:
        return pd.DataFrame(columns=empty_columns)

    frames = [standardizer(pd.read_parquet(path)) for path in files]
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("timestamp", kind="mergesort").drop_duplicates("timestamp", keep="last")
    return out.reset_index(drop=True)


def _default_candidates(base_alloc: dict[str, dict[str, float]]) -> dict[str, dict[str, dict[str, float]]]:
    base = {k: v.copy() for k, v in base_alloc.items()}

    out = {
        "current": {k: v.copy() for k, v in base.items()},
        "r1_55_45": {k: v.copy() for k, v in base.items()},
        "r2_50_30": {k: v.copy() for k, v in base.items()},
        "r2_30_50": {k: v.copy() for k, v in base.items()},
        "r3_80pct": {k: v.copy() for k, v in base.items()},
    }

    out["r1_55_45"]["R1"] = {"donchian": 0.55, "ema_adx": 0.45, "mean_reversion": 0.0}
    out["r2_50_30"]["R2"] = {"donchian": 0.50, "ema_adx": 0.30, "mean_reversion": 0.0}
    out["r2_30_50"]["R2"] = {"donchian": 0.30, "ema_adx": 0.50, "mean_reversion": 0.0}
    out["r3_80pct"]["R3"] = {"donchian": 0.0, "ema_adx": 0.0, "mean_reversion": 0.8}

    return out


def _merge_with_base(
    base_alloc: dict[str, dict[str, float]],
    candidate_allocations: dict[str, Any] | None,
) -> dict[str, dict[str, dict[str, float]]]:
    if candidate_allocations is None:
        return _default_candidates(base_alloc)

    if not isinstance(candidate_allocations, dict) or not candidate_allocations:
        raise ValueError("candidate_allocations must be a non-empty mapping")

    out: dict[str, dict[str, dict[str, float]]] = {}
    for name, raw in candidate_allocations.items():
        if not isinstance(raw, dict):
            raise ValueError(f"candidate '{name}' must be a mapping")

        merged = {k: v.copy() for k, v in base_alloc.items()}
        for regime, weights in raw.items():
            if not isinstance(weights, dict):
                raise ValueError(f"candidate '{name}' regime '{regime}' must be a mapping")
            row = merged.get(regime, {}).copy()
            for key in ["donchian", "ema_adx", "mean_reversion"]:
                if key in weights:
                    row[key] = float(weights[key])
            merged[regime] = row

        out[str(name)] = merged

    if "current" not in out:
        out = {"current": {k: v.copy() for k, v in base_alloc.items()}, **out}

    return out


def _default_folds(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[Fold]:
    years = sorted(pd.date_range(start_ts, end_ts, freq="YS", tz="UTC").year.unique().tolist())
    if len(years) < 2:
        return [
            Fold(
                name="fold_single",
                train_start=start_ts,
                train_end=end_ts,
                valid_start=None,
                valid_end=None,
                test_start=start_ts,
                test_end=end_ts,
            )
        ]

    folds: list[Fold] = []
    for i in range(1, len(years)):
        train_start = pd.Timestamp(f"{years[0]}-01-01T00:00:00Z")
        train_end = pd.Timestamp(f"{years[i] - 1}-12-31T23:59:59Z")
        test_start = pd.Timestamp(f"{years[i]}-01-01T00:00:00Z")
        test_end = pd.Timestamp(f"{years[i]}-12-31T23:59:59Z")

        train_start = max(train_start, start_ts)
        train_end = min(train_end, end_ts)
        test_start = max(test_start, start_ts)
        test_end = min(test_end, end_ts)

        if train_start <= train_end and test_start <= test_end:
            folds.append(
                Fold(
                    name=f"train_{train_start.year}_{train_end.year}_test_{test_start.year}",
                    train_start=train_start,
                    train_end=train_end,
                    valid_start=None,
                    valid_end=None,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

    if not folds:
        folds.append(
            Fold(
                name="fold_single",
                train_start=start_ts,
                train_end=end_ts,
                valid_start=None,
                valid_end=None,
                test_start=start_ts,
                test_end=end_ts,
            )
        )

    return folds


def _parse_folds(folds_cfg: list[dict[str, Any]] | None, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[Fold]:
    if folds_cfg is None:
        return _default_folds(start_ts, end_ts)

    if not isinstance(folds_cfg, list) or not folds_cfg:
        raise ValueError("walk_forward.folds must be a non-empty list")

    folds: list[Fold] = []
    for i, item in enumerate(folds_cfg):
        if not isinstance(item, dict):
            raise ValueError(f"fold at index {i} must be a mapping")

        name = str(item.get("name", f"fold_{i+1}"))
        train_start = _to_timestamp(str(item["train_start"]))
        train_end = _to_timestamp(str(item["train_end"]))

        valid_start_raw = item.get("valid_start")
        valid_end_raw = item.get("valid_end")
        if (valid_start_raw is None) ^ (valid_end_raw is None):
            raise ValueError(f"fold '{name}' must set both valid_start and valid_end or neither")

        valid_start = _to_timestamp(str(valid_start_raw)) if valid_start_raw is not None else None
        valid_end = _to_timestamp(str(valid_end_raw)) if valid_end_raw is not None else None

        test_start = _to_timestamp(str(item["test_start"]))
        test_end = _to_timestamp(str(item["test_end"]))

        if train_start > train_end:
            raise ValueError(f"fold '{name}' has train_start > train_end")
        if valid_start is not None and valid_end is not None and valid_start > valid_end:
            raise ValueError(f"fold '{name}' has valid_start > valid_end")
        if test_start > test_end:
            raise ValueError(f"fold '{name}' has test_start > test_end")

        # Enforce chronological split order to avoid overlap/leakage inside each fold.
        if valid_start is not None and valid_end is not None:
            if train_end >= valid_start:
                raise ValueError(f"fold '{name}' requires train_end < valid_start")
            if valid_end >= test_start:
                raise ValueError(f"fold '{name}' requires valid_end < test_start")
        else:
            if train_end >= test_start:
                raise ValueError(f"fold '{name}' requires train_end < test_start when no valid split")

        clipped_train_start = max(train_start, start_ts)
        clipped_train_end = min(train_end, end_ts)
        clipped_valid_start = max(valid_start, start_ts) if valid_start is not None else None
        clipped_valid_end = min(valid_end, end_ts) if valid_end is not None else None
        clipped_test_start = max(test_start, start_ts)
        clipped_test_end = min(test_end, end_ts)

        folds.append(
            Fold(
                name=name,
                train_start=clipped_train_start,
                train_end=clipped_train_end,
                valid_start=clipped_valid_start,
                valid_end=clipped_valid_end,
                test_start=clipped_test_start,
                test_end=clipped_test_end,
            )
        )

    valid_folds: list[Fold] = []
    for fold in folds:
        if fold.train_start > fold.train_end or fold.test_start > fold.test_end:
            continue
        if fold.valid_start is not None and fold.valid_end is not None and fold.valid_start > fold.valid_end:
            continue
        valid_folds.append(fold)

    if not valid_folds:
        return []

    ordered = sorted(valid_folds, key=lambda x: (x.test_start, x.test_end, x.name))
    for prev, curr in zip(ordered, ordered[1:]):
        if prev.test_end >= curr.test_start:
            raise ValueError(
                f"test windows overlap between folds '{prev.name}' and '{curr.name}'"
            )

    return ordered


def _mask_between(ts: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    out = pd.to_datetime(ts, utc=True, errors="coerce")
    return (out >= start) & (out <= end)


def _score_from_metrics(metrics: dict[str, Any], metric_name: str) -> float:
    key = SELECTION_METRICS[metric_name]
    value = metrics.get(key, 0.0)
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    if pd.isna(score):
        return float("-inf")
    return score


def _metric_value(metrics: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = metrics.get(key, default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = default
    return out


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def run_walk_forward(
    bars_15m_path: Path,
    regime_15m_path: Path,
    output_dir: Path,
    symbol: str = "BTCUSDT",
    base_regime_allocations: dict[str, dict[str, float]] | None = None,
    candidate_allocations: dict[str, Any] | None = None,
    folds: list[dict[str, Any]] | None = None,
    selection_metric: str = "sharpe",
    default_regime: str = R4,
    bars_per_year: int = 35_040,
    initial_equity: float = 100_000.0,
    fee_bps: float = 4.0,
    slippage_bps: float = 1.0,
    position_lag_bars: int = 1,
    long_only: bool = True,
    min_hold_bars: int = 1,
    rebalance_threshold: float = 0.0,
    donchian_window: int = 20,
    donchian_hold_until_opposite: bool = True,
    ema_fast: int = 21,
    ema_slow: int = 55,
    ema_adx_window: int = 14,
    ema_adx_threshold: float = 20.0,
    ema_use_adx_filter: bool = True,
    mr_z_window: int = 48,
    mr_entry_z: float = 1.5,
    mr_exit_z: float = 0.5,
) -> dict[str, Any]:
    metric_key = str(selection_metric).strip().lower()
    if metric_key not in SELECTION_METRICS:
        raise ValueError(f"Unsupported selection_metric: {selection_metric}")

    base_alloc = (
        {k: {kk: float(vv) for kk, vv in v.items()} for k, v in base_regime_allocations.items()}
        if base_regime_allocations
        else {k: v.copy() for k, v in DEFAULT_REGIME_ALLOCATIONS.items()}
    )

    candidate_map = _merge_with_base(base_alloc, candidate_allocations)

    bars_files = _collect_files(bars_15m_path, f"{symbol.upper()}_15m_*.parquet")
    regime_files = _collect_files(regime_15m_path, f"{symbol.upper()}_regime_15m_*.parquet")

    bars = _concat_standardized(bars_files, _standardize_bars, REQUIRED_BARS_COLUMNS)
    regime = _concat_standardized(regime_files, _standardize_regime, REQUIRED_REGIME_COLUMNS)

    if bars.empty:
        return {
            "rows_bars": 0,
            "rows_regime": 0,
            "rows_backtest_total": 0,
            "candidate_count": len(candidate_map),
            "fold_count": 0,
            "files_written": [],
        }

    start_ts = pd.Timestamp(bars["timestamp"].min())
    end_ts = pd.Timestamp(bars["timestamp"].max())
    folds_resolved = _parse_folds(folds_cfg=folds, start_ts=start_ts, end_ts=end_ts)

    candidate_frames: dict[str, pd.DataFrame] = {}
    for name, alloc in candidate_map.items():
        signals = build_signals_15m(
            bars_15m=bars,
            regime_15m=regime,
            default_regime=default_regime,
            donchian_window=donchian_window,
            donchian_hold_until_opposite=donchian_hold_until_opposite,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            ema_adx_window=ema_adx_window,
            ema_adx_threshold=ema_adx_threshold,
            ema_use_adx_filter=ema_use_adx_filter,
            mr_z_window=mr_z_window,
            mr_entry_z=mr_entry_z,
            mr_exit_z=mr_exit_z,
            regime_allocations=alloc,
            long_only=long_only,
            min_hold_bars=min_hold_bars,
            rebalance_threshold=rebalance_threshold,
        )

        backtest = simulate_backtest_15m(
            bars_15m=bars,
            signals_15m=signals,
            initial_equity=initial_equity,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            position_lag_bars=position_lag_bars,
            long_only=long_only,
        )
        backtest["timestamp"] = pd.to_datetime(backtest["timestamp"], utc=True)
        candidate_frames[name] = backtest

    fold_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    selected_oos_frames: list[pd.DataFrame] = []

    for fold_idx, fold in enumerate(folds_resolved):
        has_valid_window = fold.valid_start is not None and fold.valid_end is not None
        selection_scope = "valid" if has_valid_window else "train"

        best_name: str | None = None
        best_score = float("-inf")
        best_selection_scope = selection_scope
        best_train_metrics: dict[str, Any] = {}
        best_valid_metrics: dict[str, Any] = {}
        best_test_metrics: dict[str, Any] = {}

        for candidate_name, backtest in candidate_frames.items():
            train_mask = _mask_between(backtest["timestamp"], fold.train_start, fold.train_end)
            test_mask = _mask_between(backtest["timestamp"], fold.test_start, fold.test_end)

            train_metrics = compute_metrics(backtest.loc[train_mask].copy(), bars_per_year=bars_per_year)
            if has_valid_window:
                valid_mask = _mask_between(backtest["timestamp"], fold.valid_start, fold.valid_end)
                valid_metrics = compute_metrics(backtest.loc[valid_mask].copy(), bars_per_year=bars_per_year)
            else:
                valid_metrics = {}
            test_metrics = compute_metrics(backtest.loc[test_mask].copy(), bars_per_year=bars_per_year)

            selected_metrics = valid_metrics if has_valid_window else train_metrics
            score = _score_from_metrics(selected_metrics, metric_key)
            valid_score = _score_from_metrics(valid_metrics, metric_key) if has_valid_window else float("nan")

            score_rows.append(
                {
                    "fold": fold.name,
                    "candidate": candidate_name,
                    "selection_scope": selection_scope,
                    "selection_metric": metric_key,
                    "selection_score": score,
                    "train_score": _score_from_metrics(train_metrics, metric_key),
                    "valid_score": valid_score,
                    "train_total_return": _metric_value(train_metrics, "total_return", default=0.0),
                    "train_sharpe": _metric_value(train_metrics, "sharpe", default=0.0),
                    "train_max_drawdown": _metric_value(train_metrics, "max_drawdown", default=0.0),
                    "valid_total_return": _metric_value(valid_metrics, "total_return"),
                    "valid_sharpe": _metric_value(valid_metrics, "sharpe"),
                    "valid_max_drawdown": _metric_value(valid_metrics, "max_drawdown"),
                    "test_total_return": _metric_value(test_metrics, "total_return", default=0.0),
                    "test_sharpe": _metric_value(test_metrics, "sharpe", default=0.0),
                    "test_max_drawdown": _metric_value(test_metrics, "max_drawdown", default=0.0),
                }
            )

            if score > best_score:
                best_score = score
                best_name = candidate_name
                best_selection_scope = selection_scope
                best_train_metrics = train_metrics
                best_valid_metrics = valid_metrics
                best_test_metrics = test_metrics

        if best_name is None:
            continue

        fold_rows.append(
            {
                "fold": fold.name,
                "selection_scope": best_selection_scope,
                "selection_metric": metric_key,
                "selection_score": float(best_score),
                "selected_candidate": best_name,
                "train_start": fold.train_start.isoformat(),
                "train_end": fold.train_end.isoformat(),
                "valid_start": fold.valid_start.isoformat() if fold.valid_start is not None else "",
                "valid_end": fold.valid_end.isoformat() if fold.valid_end is not None else "",
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
                "train_score": _score_from_metrics(best_train_metrics, metric_key),
                "valid_score": _score_from_metrics(best_valid_metrics, metric_key) if has_valid_window else float("nan"),
                "train_total_return": _metric_value(best_train_metrics, "total_return", default=0.0),
                "train_sharpe": _metric_value(best_train_metrics, "sharpe", default=0.0),
                "train_max_drawdown": _metric_value(best_train_metrics, "max_drawdown", default=0.0),
                "valid_total_return": _metric_value(best_valid_metrics, "total_return"),
                "valid_sharpe": _metric_value(best_valid_metrics, "sharpe"),
                "valid_max_drawdown": _metric_value(best_valid_metrics, "max_drawdown"),
                "test_total_return": _metric_value(best_test_metrics, "total_return", default=0.0),
                "test_sharpe": _metric_value(best_test_metrics, "sharpe", default=0.0),
                "test_max_drawdown": _metric_value(best_test_metrics, "max_drawdown", default=0.0),
                "test_bh_total_return": _metric_value(best_test_metrics, "bh_total_return", default=0.0),
                "test_excess_total_return": _metric_value(best_test_metrics, "excess_total_return", default=0.0),
            }
        )

        chosen = candidate_frames[best_name]
        test_mask = _mask_between(chosen["timestamp"], fold.test_start, fold.test_end)
        oos_chunk = chosen.loc[test_mask].copy()
        oos_chunk["_fold_order"] = fold_idx
        oos_chunk["fold"] = fold.name
        oos_chunk["candidate"] = best_name
        selected_oos_frames.append(oos_chunk)

    score_df = pd.DataFrame(score_rows)
    if not score_df.empty:
        score_df = score_df.sort_values(["fold", "candidate"], kind="mergesort").reset_index(drop=True)

    fold_df = pd.DataFrame(fold_rows)
    if not fold_df.empty:
        fold_df = fold_df.sort_values("fold", kind="mergesort").reset_index(drop=True)

    if selected_oos_frames:
        oos_frame = pd.concat(selected_oos_frames, ignore_index=True)
        oos_frame = oos_frame.sort_values(["timestamp", "_fold_order"], kind="mergesort")
        oos_frame = oos_frame.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        oos_frame = oos_frame.drop(columns=["_fold_order"], errors="ignore")

        oos_metrics = compute_metrics(oos_frame.copy(), bars_per_year=bars_per_year)
        oos_monthly = (
            oos_frame.assign(month=_to_month(oos_frame["timestamp"]))
            .groupby(["month", "fold", "candidate"], dropna=False)
            .agg(
                bar_count=("timestamp", "size"),
                total_return_sum=("net_return", "sum"),
                gross_return_sum=("gross_return", "sum"),
                cost_return_sum=("cost_return", "sum"),
                turnover_sum=("turnover", "sum"),
                turnover_mean=("turnover", "mean"),
            )
            .reset_index()
            .sort_values(["month", "fold"], kind="mergesort")
        )
    else:
        oos_frame = pd.DataFrame()
        oos_metrics = {
            "row_count": 0,
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "bh_total_return": 0.0,
            "bh_sharpe": 0.0,
            "bh_max_drawdown": 0.0,
            "excess_total_return": 0.0,
            "excess_sharpe": 0.0,
            "outperform_buy_hold": False,
        }
        oos_monthly = pd.DataFrame(
            columns=[
                "month",
                "fold",
                "candidate",
                "bar_count",
                "total_return_sum",
                "gross_return_sum",
                "cost_return_sum",
                "turnover_sum",
                "turnover_mean",
            ]
        )

    selected_counts = (
        fold_df["selected_candidate"].value_counts().sort_index().to_dict() if not fold_df.empty else {}
    )

    out_dir = ensure_dir(output_dir)
    scores_path = out_dir / f"{symbol.upper()}_walk_forward_scores.csv"
    folds_path = out_dir / f"{symbol.upper()}_walk_forward_folds.csv"
    oos_monthly_path = out_dir / f"{symbol.upper()}_walk_forward_oos_monthly.csv"
    summary_path = out_dir / f"{symbol.upper()}_walk_forward_summary.json"

    score_df.to_csv(scores_path, index=False)
    fold_df.to_csv(folds_path, index=False)
    oos_monthly.to_csv(oos_monthly_path, index=False)

    summary_payload = {
        "symbol": symbol.upper(),
        "selection_metric": metric_key,
        "rows_bars": int(len(bars)),
        "rows_regime": int(len(regime)),
        "rows_backtest_total": int(sum(len(v) for v in candidate_frames.values())),
        "candidate_count": int(len(candidate_frames)),
        "candidates": sorted(candidate_frames.keys()),
        "fold_count": int(len(fold_df)),
        "selected_candidate_count": selected_counts,
        "execution": {
            "long_only": bool(long_only),
            "min_hold_bars": int(min_hold_bars),
            "rebalance_threshold": float(rebalance_threshold),
            "position_lag_bars": int(position_lag_bars),
            "fee_bps": float(fee_bps),
            "slippage_bps": float(slippage_bps),
        },
        "oos_metrics": {
            "row_count": int(oos_metrics.get("row_count", 0)),
            "total_return": float(oos_metrics.get("total_return", 0.0)),
            "annual_return": float(oos_metrics.get("annual_return", 0.0)),
            "sharpe": float(oos_metrics.get("sharpe", 0.0)),
            "max_drawdown": float(oos_metrics.get("max_drawdown", 0.0)),
            "bh_total_return": float(oos_metrics.get("bh_total_return", 0.0)),
            "bh_sharpe": float(oos_metrics.get("bh_sharpe", 0.0)),
            "bh_max_drawdown": float(oos_metrics.get("bh_max_drawdown", 0.0)),
            "excess_total_return": float(oos_metrics.get("excess_total_return", 0.0)),
            "excess_sharpe": float(oos_metrics.get("excess_sharpe", 0.0)),
            "outperform_buy_hold": bool(oos_metrics.get("outperform_buy_hold", False)),
        },
        "files": {
            "scores_csv": str(scores_path),
            "folds_csv": str(folds_path),
            "oos_monthly_csv": str(oos_monthly_path),
        },
    }

    _write_json(summary_payload, summary_path)

    return {
        "rows_bars": int(len(bars)),
        "rows_regime": int(len(regime)),
        "rows_backtest_total": int(sum(len(v) for v in candidate_frames.values())),
        "candidate_count": int(len(candidate_frames)),
        "fold_count": int(len(fold_df)),
        "selected_candidate_count": selected_counts,
        "oos_metrics": summary_payload["oos_metrics"],
        "scores_path": str(scores_path),
        "folds_path": str(folds_path),
        "oos_monthly_path": str(oos_monthly_path),
        "summary_path": str(summary_path),
        "files_written": [str(scores_path), str(folds_path), str(oos_monthly_path), str(summary_path)],
    }
