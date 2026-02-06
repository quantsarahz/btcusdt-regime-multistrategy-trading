"""Regime visualization plots for BTCUSDT 1h data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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


def _regime_colors(regimes: List[str]) -> Dict[str, str]:
    base = {
        "Up_LowVol": "#2ca02c",
        "Up_HighVol": "#98df8a",
        "Down_LowVol": "#d62728",
        "Down_HighVol": "#ff9896",
        "Range_LowVol": "#1f77b4",
        "Range_HighVol": "#aec7e8",
    }
    colors = {}
    for i, reg in enumerate(regimes):
        colors[reg] = base.get(reg, plt.cm.tab20(i % 20))
    return colors


def plot_regime_timeline(close: pd.Series, regime: pd.Series, out_path: Path) -> None:
    """Plot close price with regime-shaded background."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(close.index, close.values, color="black", linewidth=0.8, label="Close")

    segments = _regime_segments(regime)
    regimes = sorted(segments["regime"].unique().tolist()) if not segments.empty else []
    colors = _regime_colors(regimes)

    for _, row in segments.iterrows():
        ax.axvspan(row["start_ts"], row["end_ts"], color=colors.get(row["regime"], "#cccccc"), alpha=0.15)

    legend_patches = [Patch(facecolor=colors[r], label=r, alpha=0.4) for r in regimes]
    if legend_patches:
        ax.legend(handles=legend_patches, loc="upper left", ncol=3, fontsize=8)

    ax.set_title("BTCUSDT 1h Regime Timeline")
    ax.set_xlabel("Time")
    ax.set_ylabel("Close")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_regime_distribution(regime: pd.Series, out_path: Path) -> None:
    """Plot regime distribution as a percentage bar chart."""
    valid = regime.dropna()
    counts = valid.value_counts()
    pct = (counts / counts.sum()) * 100 if counts.sum() > 0 else counts

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(pct.index.astype(str), pct.values)
    ax.set_title("Regime Distribution (Percent)")
    ax.set_xlabel("Regime")
    ax.set_ylabel("Percent")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_regime_duration_boxplot(regime: pd.Series, out_path: Path) -> None:
    """Plot boxplots of regime segment durations in hours."""
    segments = _regime_segments(regime)
    if segments.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Regime Duration Boxplot")
        ax.set_xlabel("Regime")
        ax.set_ylabel("Duration (hours)")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return

    regimes = sorted(segments["regime"].unique().tolist())
    data = [segments.loc[segments["regime"] == r, "length_bars"].values for r in regimes]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, labels=regimes, showfliers=False)
    ax.set_title("Regime Duration Boxplot")
    ax.set_xlabel("Regime")
    ax.set_ylabel("Duration (hours)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    df = load_klines_1h_processed()
    features = make_bar_features_1h(df)
    regimes = compute_regime_1h(features)

    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    timeline_path = figures_dir / "regime_timeline_1h.png"
    dist_path = figures_dir / "regime_distribution_1h.png"
    duration_path = figures_dir / "regime_duration_boxplot_1h.png"

    plot_regime_timeline(df["close"], regimes["regime"], timeline_path)
    plot_regime_distribution(regimes["regime"], dist_path)
    plot_regime_duration_boxplot(regimes["regime"], duration_path)

    print("Saved figures:")
    print(f"  {timeline_path}")
    print(f"  {dist_path}")
    print(f"  {duration_path}")


if __name__ == "__main__":
    main()
