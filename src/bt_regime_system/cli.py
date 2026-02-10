from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml

from bt_regime_system.data.build_bars import run_build_bars
from bt_regime_system.data.fetch_1m import fetch_1m_klines, write_monthly_raw_1m
from bt_regime_system.data.qc_1m import run_qc_1m
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


def _config_data(cfg: dict) -> tuple[dict, dict]:
    cfg_data = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    cfg_paths = cfg_data.get("paths", {}) if isinstance(cfg_data.get("paths"), dict) else {}
    return cfg_data, cfg_paths


@app.command("fetch-1m")
def fetch_1m_cmd(
    start: str = typer.Option("2022-01-01T00:01:00Z", help="UTC start close-time, default 2022-01-01T00:01:00Z"),
    end: str = typer.Option("2025-12-31T23:59:00Z", help="UTC end close-time, default 2025-12-31T23:59:00Z"),
    symbol: Optional[str] = typer.Option(None, help="Trading symbol, defaults to config value"),
    out_dir: Optional[Path] = typer.Option(None, help="Output directory for raw monthly parquet"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="Default config path"),
) -> None:
    """Download 1m bars and write monthly raw parquet files."""
    cfg = _read_yaml(config)
    cfg_data, cfg_paths = _config_data(cfg)

    resolved_symbol = symbol or cfg_data.get("symbol") or "BTCUSDT"
    resolved_out_dir = out_dir or Path(cfg_paths.get("raw_1m", "data/raw_1m"))

    logger.info("Fetching 1m bars: symbol=%s start=%s end=%s", resolved_symbol, start, end)
    frame = fetch_1m_klines(symbol=resolved_symbol, start=start, end=end)
    written = write_monthly_raw_1m(frame, out_dir=resolved_out_dir, symbol=resolved_symbol)

    logger.info("Fetched rows: %d", len(frame))
    logger.info("Written files: %d", len(written))
    for path in written:
        logger.info("Wrote: %s", path)


@app.command("qc-1m")
def qc_1m_cmd(
    input_path: Path = typer.Option(Path("data/raw_1m"), help="Raw 1m parquet file or folder"),
    output_dir: Optional[Path] = typer.Option(None, help="Clean 1m output folder"),
    report_dir: Optional[Path] = typer.Option(None, help="QC report output folder"),
    fill_missing: bool = typer.Option(True, help="Fill missing 1m rows with previous close policy"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="Default config path"),
) -> None:
    """Clean raw 1m bars and generate QC reports."""
    cfg = _read_yaml(config)
    _, cfg_paths = _config_data(cfg)

    resolved_output_dir = output_dir or Path(cfg_paths.get("clean_1m", "data/clean_1m"))
    resolved_report_dir = report_dir or Path(cfg_paths.get("reports", "data/reports"))

    summaries = run_qc_1m(
        input_path=input_path,
        output_dir=resolved_output_dir,
        report_dir=resolved_report_dir,
        fill_missing=fill_missing,
    )

    logger.info("QC files processed: %d", len(summaries))
    for item in summaries:
        logger.info(
            "QC %s -> %s (%d -> %d rows)",
            item["raw_file"],
            item["clean_file"],
            item["rows_in"],
            item["rows_out"],
        )
        logger.info("Report: %s", item["report_file"])


@app.command("build-bars")
def build_bars_cmd(
    input_path: Path = typer.Option(Path("data/clean_1m"), help="Clean 1m parquet file or folder"),
    output_15m_dir: Optional[Path] = typer.Option(None, help="15m bars output folder"),
    output_1h_dir: Optional[Path] = typer.Option(None, help="1h bars output folder"),
    symbol: Optional[str] = typer.Option(None, help="Trading symbol, defaults to config value"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="Default config path"),
) -> None:
    """Build 15m and 1h bars from clean 1m parquet data."""
    cfg = _read_yaml(config)
    cfg_data, cfg_paths = _config_data(cfg)

    resolved_symbol = symbol or cfg_data.get("symbol") or "BTCUSDT"
    resolved_15m_dir = output_15m_dir or Path(cfg_paths.get("bars_15m", "data/bars_15m"))
    resolved_1h_dir = output_1h_dir or Path(cfg_paths.get("bars_1h", "data/bars_1h"))

    summary = run_build_bars(
        input_path=input_path,
        output_15m_dir=resolved_15m_dir,
        output_1h_dir=resolved_1h_dir,
        symbol=resolved_symbol,
    )

    logger.info("Build bars rows in: %d", summary["rows_in"])
    logger.info("Built 15m rows: %d", summary["rows_15m"])
    logger.info("Built 1h rows: %d", summary["rows_1h"])
    logger.info("15m files written: %d", len(summary["files_15m"]))
    logger.info("1h files written: %d", len(summary["files_1h"]))


if __name__ == "__main__":
    app()
