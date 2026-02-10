from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from bt_regime_system.data.build_bars import build_15m_and_1h
from bt_regime_system.data.fetch_1m import load_1m_csv
from bt_regime_system.data.qc_1m import clean_1m_ohlcv
from bt_regime_system.regime.detect import detect_regime_1h
from bt_regime_system.utils.logging import get_logger

app = typer.Typer(help="Regime + multi-strategy CLI")
logger = get_logger(__name__)


@app.command("fetch-1m")
def fetch_1m_cmd(input_csv: Path, output_parquet: Path) -> None:
    """Load 1m CSV and persist as parquet."""
    df = load_1m_csv(input_csv)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet)
    logger.info("Saved 1m parquet: %s (%d rows)", output_parquet, len(df))


@app.command("qc-1m")
def qc_1m_cmd(input_parquet: Path, output_parquet: Path) -> None:
    """Clean 1m bars and persist cleaned parquet."""
    df = pd.read_parquet(input_parquet)
    clean = clean_1m_ohlcv(df)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(output_parquet)
    logger.info("Saved cleaned 1m parquet: %s (%d rows)", output_parquet, len(clean))


@app.command("build-bars")
def build_bars_cmd(input_parquet: Path, output_15m: Path, output_1h: Path) -> None:
    """Build 15m and 1h bars from cleaned 1m bars."""
    df = pd.read_parquet(input_parquet)
    bars_15m, bars_1h = build_15m_and_1h(df)
    output_15m.parent.mkdir(parents=True, exist_ok=True)
    output_1h.parent.mkdir(parents=True, exist_ok=True)
    bars_15m.to_parquet(output_15m)
    bars_1h.to_parquet(output_1h)
    logger.info("Saved 15m bars: %s (%d rows)", output_15m, len(bars_15m))
    logger.info("Saved 1h bars: %s (%d rows)", output_1h, len(bars_1h))


@app.command("detect-regime")
def detect_regime_cmd(input_1h: Path, output_regime: Path) -> None:
    """Detect regime from 1h bars and persist labels."""
    bars_1h = pd.read_parquet(input_1h)
    regime = detect_regime_1h(bars_1h)
    output_regime.parent.mkdir(parents=True, exist_ok=True)
    regime.to_frame("regime").to_parquet(output_regime)
    logger.info("Saved regime series: %s (%d rows)", output_regime, len(regime))


if __name__ == "__main__":
    app()
