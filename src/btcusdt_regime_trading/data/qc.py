"""Data quality checks for BTCUSDT 1h klines."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from btcusdt_regime_trading.data.dataset import load_klines_1h
from btcusdt_regime_trading.utils.paths import REPORTS_DIR


REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume", "close_time", "num_trades"]


def _ensure_dirs(base_dir: Path) -> tuple[Path, Path]:
    tables_dir = base_dir / "tables"
    figures_dir = base_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir


def _build_missing_gaps(missing_idx: pd.DatetimeIndex) -> pd.DataFrame:
    columns = ["gap_start", "gap_end", "n_missing"]
    if missing_idx.empty:
        return pd.DataFrame(columns=columns)

    missing_idx = missing_idx.sort_values()
    gaps: List[dict] = []
    start = missing_idx[0]
    prev = missing_idx[0]

    for ts in missing_idx[1:]:
        if ts == prev + pd.Timedelta(hours=1):
            prev = ts
            continue
        n_missing = int(((prev - start) / pd.Timedelta(hours=1)) + 1)
        gaps.append({"gap_start": start, "gap_end": prev, "n_missing": n_missing})
        start = ts
        prev = ts

    n_missing = int(((prev - start) / pd.Timedelta(hours=1)) + 1)
    gaps.append({"gap_start": start, "gap_end": prev, "n_missing": n_missing})

    gaps_df = pd.DataFrame(gaps)
    return gaps_df.sort_values("gap_start").head(50)


def _plot_close(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if df.empty:
        ax.set_title("Close Price (no data)")
    else:
        ax.plot(df.index, df["close"], linewidth=0.8)
        ax.set_title("Close Price")
    ax.set_xlabel("Time")
    ax.set_ylabel("Close")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_missing_by_month(missing_idx: pd.DatetimeIndex, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if missing_idx.empty:
        ax.set_title("Missing Bars by Month (no missing)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Missing Count")
    else:
        month_counts = (
            pd.Series(1, index=missing_idx)
            .groupby(missing_idx.to_period("M"))
            .sum()
            .sort_index()
        )
        ax.bar(month_counts.index.astype(str), month_counts.values)
        ax.set_title("Missing Bars by Month")
        ax.set_xlabel("Month")
        ax.set_ylabel("Missing Count")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_returns_hist(log_returns: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if log_returns.dropna().empty:
        ax.set_title("Log Returns Histogram (no data)")
    else:
        ax.hist(log_returns.dropna(), bins=80)
        ax.set_title("Log Returns Histogram")
    ax.set_xlabel("Log Return")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_qc_1h(symbol: Optional[str] = None, out_dir: Optional[Path] = None) -> Dict[str, object]:
    """Run QC checks on local 1h klines and write artifacts."""
    base_dir = Path(out_dir) if out_dir is not None else REPORTS_DIR
    tables_dir, figures_dir = _ensure_dirs(base_dir)

    df = load_klines_1h(symbol=symbol)

    if df.empty:
        start_ts = pd.NaT
        end_ts = pd.NaT
        expected_idx = pd.DatetimeIndex([], tz="UTC")
    else:
        start_ts = df.index.min()
        end_ts = df.index.max()
        expected_idx = pd.date_range(start=start_ts, end=end_ts, freq="1H", tz="UTC")

    actual_idx = df.index
    if not isinstance(actual_idx, pd.DatetimeIndex):
        actual_idx = pd.DatetimeIndex(actual_idx)
    if actual_idx.tz is None:
        actual_idx = actual_idx.tz_localize("UTC")
    else:
        actual_idx = actual_idx.tz_convert("UTC")

    missing_idx = expected_idx.difference(actual_idx)
    missing_count = int(len(missing_idx))
    missing_gaps = _build_missing_gaps(missing_idx)

    duplicate_count = int(actual_idx.duplicated().sum())

    required_missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if required_missing:
        raise ValueError(f"Missing required columns: {required_missing}")

    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    ohlc = df[["open", "high", "low", "close"]]
    high_violation = int((ohlc["high"] < ohlc[["open", "close"]].max(axis=1)).sum())
    low_violation = int((ohlc["low"] > ohlc[["open", "close"]].min(axis=1)).sum())
    high_low_violation = int((ohlc["high"] < ohlc["low"]).sum())

    non_positive_ohlc = int((ohlc <= 0).any(axis=1).sum())
    negative_volume = int((df["volume"] < 0).sum())

    log_returns = np.log(df["close"]).diff()
    outlier_mask = log_returns.abs() > 0.15
    outlier_count = int(outlier_mask.sum())
    top_outliers = (
        log_returns.abs()
        .dropna()
        .sort_values(ascending=False)
        .head(20)
    )

    outliers_df = pd.DataFrame(
        {
            "open_time": top_outliers.index,
            "log_return": log_returns.loc[top_outliers.index].values,
            "abs_log_return": top_outliers.values,
        }
    )

    summary = {
        "n_rows": int(len(df)),
        "start_ts": None if pd.isna(start_ts) else start_ts.isoformat(),
        "end_ts": None if pd.isna(end_ts) else end_ts.isoformat(),
        "missing_count": missing_count,
        "missing_gaps_count": int(len(missing_gaps)),
        "duplicate_count": duplicate_count,
        "ohlc_high_violation": high_violation,
        "ohlc_low_violation": low_violation,
        "ohlc_high_low_violation": high_low_violation,
        "non_positive_ohlc_count": non_positive_ohlc,
        "negative_volume_count": negative_volume,
        "return_outlier_count": outlier_count,
    }

    summary_path = tables_dir / "qc_summary_1h.csv"
    missing_gaps_path = tables_dir / "missing_gaps_1h.csv"
    outliers_path = tables_dir / "return_outliers_1h.csv"

    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    missing_gaps.to_csv(missing_gaps_path, index=False)
    outliers_df.to_csv(outliers_path, index=False)

    close_path = figures_dir / "close_price_1h.png"
    missing_month_path = figures_dir / "missing_by_month_1h.png"
    returns_hist_path = figures_dir / "returns_hist_1h.png"

    _plot_close(df, close_path)
    _plot_missing_by_month(missing_idx, missing_month_path)
    _plot_returns_hist(log_returns, returns_hist_path)

    return {
        "summary": summary,
        "tables": {
            "qc_summary": summary_path,
            "missing_gaps": missing_gaps_path,
            "return_outliers": outliers_path,
        },
        "figures": {
            "close_price": close_path,
            "missing_by_month": missing_month_path,
            "returns_hist": returns_hist_path,
        },
    }


def _print_summary(result: Dict[str, object]) -> None:
    summary = result.get("summary", {})
    print("QC Summary")
    for key in [
        "n_rows",
        "start_ts",
        "end_ts",
        "missing_count",
        "missing_gaps_count",
        "duplicate_count",
        "ohlc_high_violation",
        "ohlc_low_violation",
        "ohlc_high_low_violation",
        "non_positive_ohlc_count",
        "negative_volume_count",
        "return_outlier_count",
    ]:
        print(f"  {key}: {summary.get(key)}")

    tables = result.get("tables", {})
    figures = result.get("figures", {})
    print("Outputs")
    print(f"  tables: {tables}")
    print(f"  figures: {figures}")


def main() -> None:
    result = run_qc_1h()
    _print_summary(result)


if __name__ == "__main__":
    main()
