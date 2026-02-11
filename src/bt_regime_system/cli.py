from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml

from bt_regime_system.data.build_bars import run_build_bars
from bt_regime_system.data.fetch_1m import fetch_1m_klines, write_monthly_raw_1m
from bt_regime_system.data.qc_1m import run_qc_1m
from bt_regime_system.data.qc_bars import run_qc_bars
from bt_regime_system.regime.align import run_align_regime
from bt_regime_system.regime.detect import run_detect_regime
from bt_regime_system.regime.qc import run_qc_regime
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


def _config_regime(cfg: dict) -> tuple[dict, dict, dict]:
    regime = cfg.get("regime", {}) if isinstance(cfg.get("regime"), dict) else {}
    trend = regime.get("trend", {}) if isinstance(regime.get("trend"), dict) else {}
    volatility = regime.get("volatility", {}) if isinstance(regime.get("volatility"), dict) else {}
    output = regime.get("output", {}) if isinstance(regime.get("output"), dict) else {}
    return trend, volatility, output


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


@app.command("qc-bars")
def qc_bars_cmd(
    bars_15m_path: Optional[Path] = typer.Option(None, help="15m bars parquet folder or file"),
    bars_1h_path: Optional[Path] = typer.Option(None, help="1h bars parquet folder or file"),
    report_dir: Optional[Path] = typer.Option(None, help="QC report output folder"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="Default config path"),
) -> None:
    """Run QC checks for 15m and 1h bars and write reports."""
    cfg = _read_yaml(config)
    _, cfg_paths = _config_data(cfg)

    resolved_15m_path = bars_15m_path or Path(cfg_paths.get("bars_15m", "data/bars_15m"))
    resolved_1h_path = bars_1h_path or Path(cfg_paths.get("bars_1h", "data/bars_1h"))
    resolved_report_dir = report_dir or Path(cfg_paths.get("reports", "data/reports"))

    result_15m = run_qc_bars(input_path=resolved_15m_path, freq="15m", report_dir=resolved_report_dir)
    result_1h = run_qc_bars(input_path=resolved_1h_path, freq="1h", report_dir=resolved_report_dir)

    logger.info(
        "QC bars %s: files=%d rows=%d global_dup=%d global_missing=%d",
        result_15m["frequency"],
        result_15m["summary"]["files_processed"],
        result_15m["summary"]["row_count"],
        result_15m["summary"]["global_duplicate_timestamp_count"],
        result_15m["summary"]["global_missing_timestamp_count"],
    )
    logger.info("Summary report: %s", result_15m["summary_path"])

    logger.info(
        "QC bars %s: files=%d rows=%d global_dup=%d global_missing=%d",
        result_1h["frequency"],
        result_1h["summary"]["files_processed"],
        result_1h["summary"]["row_count"],
        result_1h["summary"]["global_duplicate_timestamp_count"],
        result_1h["summary"]["global_missing_timestamp_count"],
    )
    logger.info("Summary report: %s", result_1h["summary_path"])


@app.command("detect-regime")
def detect_regime_cmd(
    input_path: Optional[Path] = typer.Option(None, help="1h bars parquet folder or file"),
    output_dir: Optional[Path] = typer.Option(None, help="Regime 1h output folder"),
    symbol: Optional[str] = typer.Option(None, help="Trading symbol, defaults to config value"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="Default config path"),
) -> None:
    """Detect 1h regime labels (R1-R4) from 1h bars and write monthly outputs."""
    cfg = _read_yaml(config)
    cfg_data, cfg_paths = _config_data(cfg)
    trend_cfg, vol_cfg, out_cfg = _config_regime(cfg)

    resolved_symbol = symbol or cfg_data.get("symbol") or "BTCUSDT"
    resolved_input_path = input_path or Path(cfg_paths.get("bars_1h", "data/bars_1h"))
    resolved_output_dir = output_dir or Path(out_cfg.get("dir_1h", "results/regime"))

    summary = run_detect_regime(
        input_path=resolved_input_path,
        output_dir=resolved_output_dir,
        symbol=resolved_symbol,
        ema_fast=int(trend_cfg.get("ema_fast", 24)),
        ema_slow=int(trend_cfg.get("ema_slow", 96)),
        adx_window=int(trend_cfg.get("adx_window", 14)),
        adx_threshold=float(trend_cfg.get("adx_threshold", 20.0)),
        ema_gap_threshold=float(trend_cfg.get("ema_gap_threshold", 0.0)),
        atr_window=int(vol_cfg.get("atr_window", 14)),
        vol_lookback=int(vol_cfg.get("quantile_lookback", 720)),
        high_vol_quantile=float(vol_cfg.get("high_vol_quantile", 0.75)),
    )

    logger.info("Regime detect rows in: %d", summary["rows_in"])
    logger.info("Regime detect rows out: %d", summary["rows_out"])
    logger.info("Regime files written: %d", len(summary["files_written"]))
    logger.info("Regime counts: %s", summary["regime_counts"])


@app.command("align-regime")
def align_regime_cmd(
    regime_1h_path: Optional[Path] = typer.Option(None, help="Regime 1h parquet folder or file"),
    bars_15m_path: Optional[Path] = typer.Option(None, help="15m bars parquet folder or file"),
    output_dir: Optional[Path] = typer.Option(None, help="Aligned 15m regime output folder"),
    symbol: Optional[str] = typer.Option(None, help="Trading symbol, defaults to config value"),
    default_regime: Optional[str] = typer.Option(None, help="Fallback regime before first available 1h label"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="Default config path"),
) -> None:
    """Align completed 1h regime labels to 15m timeline without lookahead."""
    cfg = _read_yaml(config)
    cfg_data, cfg_paths = _config_data(cfg)
    _, _, out_cfg = _config_regime(cfg)

    resolved_symbol = symbol or cfg_data.get("symbol") or "BTCUSDT"
    resolved_regime_1h_path = regime_1h_path or Path(out_cfg.get("dir_1h", "results/regime"))
    resolved_bars_15m_path = bars_15m_path or Path(cfg_paths.get("bars_15m", "data/bars_15m"))
    resolved_output_dir = output_dir or Path(out_cfg.get("dir_15m", "results/regime"))
    resolved_default_regime = default_regime or str(out_cfg.get("default_regime_15m", "R4"))

    summary = run_align_regime(
        regime_1h_path=resolved_regime_1h_path,
        bars_15m_path=resolved_bars_15m_path,
        output_dir=resolved_output_dir,
        symbol=resolved_symbol,
        default_regime=resolved_default_regime,
    )

    logger.info("Align regime rows 1h: %d", summary["rows_regime_1h"])
    logger.info("Align regime rows 15m bars: %d", summary["rows_bars_15m"])
    logger.info("Align regime rows out: %d", summary["rows_aligned"])
    logger.info("Aligned regime files written: %d", len(summary["files_written"]))


@app.command("qc-regime")
def qc_regime_cmd(
    regime_1h_path: Optional[Path] = typer.Option(None, help="Regime 1h parquet folder or file"),
    regime_15m_path: Optional[Path] = typer.Option(None, help="Regime 15m parquet folder or file"),
    report_dir: Optional[Path] = typer.Option(None, help="Regime QC report output folder"),
    symbol: Optional[str] = typer.Option(None, help="Trading symbol, defaults to config value"),
    default_regime: Optional[str] = typer.Option(None, help="Fallback regime expected before first 1h label"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="Default config path"),
) -> None:
    """Run regime QC checks and write monthly + summary reports."""
    cfg = _read_yaml(config)
    cfg_data, cfg_paths = _config_data(cfg)
    _, _, out_cfg = _config_regime(cfg)

    resolved_symbol = symbol or cfg_data.get("symbol") or "BTCUSDT"
    resolved_regime_1h_path = regime_1h_path or Path(out_cfg.get("dir_1h", "results/regime"))
    resolved_regime_15m_path = regime_15m_path or Path(out_cfg.get("dir_15m", "results/regime"))
    resolved_report_dir = report_dir or Path(cfg_paths.get("reports", "data/reports"))
    resolved_default_regime = default_regime or str(out_cfg.get("default_regime_15m", "R4"))

    result = run_qc_regime(
        regime_1h_path=resolved_regime_1h_path,
        regime_15m_path=resolved_regime_15m_path,
        report_dir=resolved_report_dir,
        symbol=resolved_symbol,
        default_regime=resolved_default_regime,
    )

    r1 = result["regime_1h"]
    r15 = result["regime_15m"]

    logger.info(
        "QC regime 1h: files=%d rows=%d global_dup=%d global_missing=%d",
        r1["summary"]["files_processed"],
        r1["summary"]["row_count"],
        r1["summary"]["global_duplicate_timestamp_count"],
        r1["summary"]["global_missing_timestamp_count"],
    )
    logger.info("Summary report: %s", r1["summary_path"])

    logger.info(
        "QC regime 15m: files=%d rows=%d global_dup=%d global_missing=%d lookahead=%d mismatch=%d",
        r15["summary"]["files_processed"],
        r15["summary"]["row_count"],
        r15["summary"]["global_duplicate_timestamp_count"],
        r15["summary"]["global_missing_timestamp_count"],
        r15["summary"]["lookahead_violation_count"],
        r15["summary"]["regime_mismatch_count"],
    )
    logger.info("Summary report: %s", r15["summary_path"])


if __name__ == "__main__":
    app()
