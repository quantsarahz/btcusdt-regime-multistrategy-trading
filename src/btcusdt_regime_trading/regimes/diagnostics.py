"""Baseline regime diagnostics for BTCUSDT 1h."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from btcusdt_regime_trading.data.loaders import load_klines_1h_processed
from btcusdt_regime_trading.features.bar_features import make_bar_features_1h
from btcusdt_regime_trading.regimes.regime_engine import compute_regime_1h
from btcusdt_regime_trading.utils.paths import REPORTS_DIR


def _regime_segments(regime: pd.Series) -> pd.DataFrame:
    """Build continuous regime segments from a regime series."""
    segments: List[dict] = []
    current = None
    start_ts = None
    length = 0

    for ts, value in regime.items():
        if pd.isna(value):
            if current is not None:
                segments.append(
                    {
                        "start_ts": start_ts,
                        "end_ts": ts - pd.Timedelta(hours=1),
                        "regime": current,
                        "length_bars": length,
                    }
                )
            current = None
            start_ts = None
            length = 0
            continue

        if current is None:
            current = value
            start_ts = ts
            length = 1
            continue

        if value == current:
            length += 1
        else:
            segments.append(
                {
                    "start_ts": start_ts,
                    "end_ts": ts - pd.Timedelta(hours=1),
                    "regime": current,
                    "length_bars": length,
                }
            )
            current = value
            start_ts = ts
            length = 1

    if current is not None:
        segments.append(
            {
                "start_ts": start_ts,
                "end_ts": regime.index[-1],
                "regime": current,
                "length_bars": length,
            }
        )

    return pd.DataFrame(segments, columns=["start_ts", "end_ts", "regime", "length_bars"])


def _regime_distribution(regime: pd.Series) -> pd.DataFrame:
    valid = regime.dropna()
    counts = valid.value_counts()
    total = counts.sum()
    pct = counts / total if total > 0 else counts
    return pd.DataFrame({"count": counts, "pct": pct}).reset_index().rename(columns={"index": "regime"})


def _duration_summary(segments: pd.DataFrame) -> pd.DataFrame:
    if segments.empty:
        return pd.DataFrame(columns=["regime", "n_segments", "mean_length", "median_length", "p25", "p75", "max_length"])

    grouped = segments.groupby("regime")["length_bars"]
    summary = grouped.agg(
        n_segments="count",
        mean_length="mean",
        median_length="median",
        p25=lambda s: s.quantile(0.25),
        p75=lambda s: s.quantile(0.75),
        max_length="max",
    )
    summary = summary.reset_index()
    return summary


def _transition_matrix(regime: pd.Series) -> pd.DataFrame:
    valid = regime.dropna()
    if valid.empty:
        return pd.DataFrame()

    prev = valid.shift(1)
    transitions = pd.DataFrame({"from": prev, "to": valid}).dropna()
    matrix = pd.crosstab(transitions["from"], transitions["to"])
    return matrix


def _switch_summary(regime: pd.Series) -> pd.DataFrame:
    total_bars = int(len(regime))
    valid = regime.dropna()
    n_valid = int(len(valid))
    if n_valid <= 1:
        n_switches = 0
        switch_rate = 0.0
    else:
        n_switches = int((valid != valid.shift(1)).sum() - 1)
        switch_rate = n_switches / (n_valid - 1)

    return pd.DataFrame(
        [
            {
                "total_bars": total_bars,
                "n_valid_bars": n_valid,
                "n_switches": n_switches,
                "switch_rate": switch_rate,
            }
        ]
    )


def _market_stats_by_regime(features: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"log_ret_1": features["log_ret_1"], "regime": regime})
    df = df.dropna(subset=["regime"])

    if df.empty:
        return pd.DataFrame(columns=["regime", "n_bars", "mean_return", "vol_return", "sharpe_like"])

    grouped = df.groupby("regime")
    mean_ret = grouped["log_ret_1"].mean()
    vol_ret = grouped["log_ret_1"].std()
    n_bars = grouped.size()

    sharpe_like = mean_ret / vol_ret.replace(0, np.nan)

    out = pd.DataFrame(
        {
            "regime": mean_ret.index,
            "n_bars": n_bars.values,
            "mean_return": mean_ret.values,
            "vol_return": vol_ret.values,
            "sharpe_like": sharpe_like.values,
        }
    )
    return out


def run_regime_diagnostics_1h() -> Dict[str, object]:
    """Run baseline regime diagnostics and write CSV outputs."""
    tables_dir = REPORTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = load_klines_1h_processed()
    features = make_bar_features_1h(df)
    regimes = compute_regime_1h(features)

    regime_series = regimes["regime"]

    segments = _regime_segments(regime_series)
    distribution = _regime_distribution(regime_series)
    duration_summary = _duration_summary(segments)
    transition_matrix = _transition_matrix(regime_series)
    switch_summary = _switch_summary(regime_series)
    market_stats = _market_stats_by_regime(features, regime_series)

    paths = {
        "distribution": tables_dir / "regime_distribution_1h.csv",
        "duration_summary": tables_dir / "regime_duration_summary_1h.csv",
        "transition_matrix": tables_dir / "regime_transition_matrix_1h.csv",
        "switch_summary": tables_dir / "regime_switch_summary_1h.csv",
        "market_stats": tables_dir / "regime_market_stats_1h.csv",
    }

    distribution.to_csv(paths["distribution"], index=False)
    duration_summary.to_csv(paths["duration_summary"], index=False)

    if transition_matrix.empty:
        transition_matrix = pd.DataFrame(columns=[])
    transition_matrix.to_csv(paths["transition_matrix"], index=True)

    switch_summary.to_csv(paths["switch_summary"], index=False)
    market_stats.to_csv(paths["market_stats"], index=False)

    return {
        "paths": paths,
        "switches": switch_summary.iloc[0].to_dict(),
        "distribution_head": distribution.head(5),
    }


def main() -> None:
    result = run_regime_diagnostics_1h()
    print("Saved tables to:")
    for key, path in result["paths"].items():
        print(f"  {key}: {path}")
    print("Distribution head:")
    print(result["distribution_head"])
    switches = result["switches"]
    print(f"Switches: {switches.get('n_switches')} (rate={switches.get('switch_rate')})")


if __name__ == "__main__":
    main()
