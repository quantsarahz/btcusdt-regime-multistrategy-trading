from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bt_regime_system.regime.detect import R1, R2, R3, R4
from bt_regime_system.utils.io import ensure_dir

REGIME_COLORS = {
    R1: "#2ca25f",  # green
    R2: "#fb8c00",  # orange
    R3: "#3182bd",  # blue
    R4: "#bdbdbd",  # gray
}


def _collect_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob(pattern))
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _standardize_bars_1h(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    required = {"timestamp", "close"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"bars_1h missing columns: {sorted(missing)}")

    out = out[["timestamp", "close"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")

    out = out.dropna(subset=["timestamp", "close"]) \
        .sort_values("timestamp") \
        .drop_duplicates("timestamp", keep="last")

    out["close"] = out["close"].astype(float)
    return out.reset_index(drop=True)


def _standardize_regime_1h(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    required = {"timestamp", "regime"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"regime_1h missing columns: {sorted(missing)}")

    out = out[["timestamp", "regime"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["regime"] = out["regime"].astype("string")

    out = out.dropna(subset=["timestamp"]) \
        .sort_values("timestamp") \
        .drop_duplicates("timestamp", keep="last")

    return out.reset_index(drop=True)


def _resolve_plot_path(output_path: Path, symbol: str) -> Path:
    if output_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
        return output_path
    return output_path / f"{symbol.upper()}_price_regime_1h.png"


def _segment_bounds(timestamps: pd.Series) -> list[pd.Timestamp]:
    if timestamps.empty:
        return []

    ts = pd.to_datetime(timestamps, utc=True)
    if len(ts) == 1:
        return [ts.iloc[0], ts.iloc[0] + pd.Timedelta(hours=1)]

    delta = ts.diff().median()
    if pd.isna(delta) or delta <= pd.Timedelta(0):
        delta = pd.Timedelta(hours=1)

    bounds = ts.tolist()
    bounds.append(ts.iloc[-1] + delta)
    return bounds


def run_plot_price_regime_1h(
    bars_1h_path: Path,
    regime_1h_path: Path,
    output_path: Path,
    symbol: str = "BTCUSDT",
    default_regime: str = R4,
) -> dict[str, Any]:
    """Render and save price line with regime background bands on 1h timeline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    bars_files = _collect_files(bars_1h_path, f"{symbol.upper()}_1h_*.parquet")
    regime_files = _collect_files(regime_1h_path, f"{symbol.upper()}_regime_1h_*.parquet")

    if not bars_files:
        raise FileNotFoundError(f"No 1h bars files found for {symbol}: {bars_1h_path}")

    bars = pd.concat([pd.read_parquet(p) for p in bars_files], ignore_index=True)
    bars_std = _standardize_bars_1h(bars)

    if regime_files:
        regime = pd.concat([pd.read_parquet(p) for p in regime_files], ignore_index=True)
        regime_std = _standardize_regime_1h(regime).rename(columns={"timestamp": "regime_timestamp"})

        aligned = pd.merge_asof(
            bars_std[["timestamp", "close"]].sort_values("timestamp"),
            regime_std.sort_values("regime_timestamp"),
            left_on="timestamp",
            right_on="regime_timestamp",
            direction="backward",
        )
        aligned["regime"] = aligned["regime"].astype("string").fillna(default_regime)
    else:
        aligned = bars_std[["timestamp", "close"]].copy()
        aligned["regime"] = default_regime

    aligned = aligned.sort_values("timestamp").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(aligned["timestamp"], aligned["close"], color="#1f2937", linewidth=1.0, label=f"{symbol.upper()} close")

    bounds = _segment_bounds(aligned["timestamp"])
    regime_values = aligned["regime"].astype(str).tolist()
    if bounds:
        start_idx = 0
        for i in range(1, len(regime_values) + 1):
            is_new_segment = i == len(regime_values) or regime_values[i] != regime_values[i - 1]
            if not is_new_segment:
                continue

            regime_label = regime_values[start_idx]
            color = REGIME_COLORS.get(regime_label, REGIME_COLORS[R4])
            ax.axvspan(bounds[start_idx], bounds[i], color=color, alpha=0.15, linewidth=0)
            start_idx = i

    ax.set_title(f"{symbol.upper()} 1h Close with Regime Bands")
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("Close Price")
    ax.grid(alpha=0.25)

    patches = [Patch(facecolor=REGIME_COLORS[k], alpha=0.25, label=k) for k in [R1, R2, R3, R4]]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + patches, labels + [R1, R2, R3, R4], loc="upper left", ncol=5)

    target = _resolve_plot_path(output_path, symbol=symbol)
    ensure_dir(target.parent)
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)

    return {
        "rows_bars_1h": int(len(bars_std)),
        "rows_regime_1h": int(len(regime_files) and len(regime_std) or 0),
        "rows_plot": int(len(aligned)),
        "output_path": target,
        "start_timestamp": aligned["timestamp"].min(),
        "end_timestamp": aligned["timestamp"].max(),
    }
