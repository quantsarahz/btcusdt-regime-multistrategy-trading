from __future__ import annotations

from pathlib import Path

import typer
import yaml

from bt_regime_system.data.fetch_1m import fetch_1m_klines, write_monthly_raw_1m
from bt_regime_system.utils.logging import get_logger

app = typer.Typer(help="BTCUSDT regime trading system CLI")
logger = get_logger(__name__)


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain mapping object: {path}")
    return data


@app.command("fetch-1m")
def fetch_1m_cmd(
    start: str = typer.Option(..., help="UTC start close-time, e.g. 2024-01-01T00:01:00Z"),
    end: str = typer.Option(..., help="UTC end close-time, e.g. 2024-01-31T23:59:00Z"),
    symbol: str | None = typer.Option(None, help="Trading symbol, defaults to config value"),
    out_dir: Path | None = typer.Option(None, help="Output directory for raw monthly parquet"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="Default config path"),
) -> None:
    """Download 1m bars and write monthly raw parquet files."""
    cfg = _read_yaml(config)
    cfg_data = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    cfg_paths = cfg_data.get("paths", {}) if isinstance(cfg_data.get("paths"), dict) else {}

    resolved_symbol = symbol or cfg_data.get("symbol") or "BTCUSDT"
    resolved_out_dir = out_dir or Path(cfg_paths.get("raw_1m", "data/raw_1m"))

    logger.info("Fetching 1m bars: symbol=%s start=%s end=%s", resolved_symbol, start, end)
    frame = fetch_1m_klines(symbol=resolved_symbol, start=start, end=end)
    written = write_monthly_raw_1m(frame, out_dir=resolved_out_dir, symbol=resolved_symbol)

    logger.info("Fetched rows: %d", len(frame))
    logger.info("Written files: %d", len(written))
    for path in written:
        logger.info("Wrote: %s", path)


if __name__ == "__main__":
    app()
