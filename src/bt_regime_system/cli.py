from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml

from bt_regime_system.backtest.diagnostics import run_backtest_diagnostics
from bt_regime_system.backtest.engine import run_backtest
from bt_regime_system.backtest.walk_forward import run_walk_forward
from bt_regime_system.analysis.plots import run_plot_price_regime_1h
from bt_regime_system.data.build_bars import run_build_bars
from bt_regime_system.data.fetch_1m import fetch_1m_klines, write_monthly_raw_1m
from bt_regime_system.data.qc_1m import run_qc_1m
from bt_regime_system.data.qc_bars import run_qc_bars
from bt_regime_system.regime.align import run_align_regime
from bt_regime_system.regime.detect import run_detect_regime
from bt_regime_system.regime.qc import run_qc_regime
from bt_regime_system.signals.build import run_generate_signals
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


def _config_regime_smoothing(cfg: dict) -> dict:
    regime = cfg.get("regime", {}) if isinstance(cfg.get("regime"), dict) else {}
    smoothing = regime.get("smoothing", {}) if isinstance(regime.get("smoothing"), dict) else {}
    return smoothing

def _config_signals(cfg: dict) -> tuple[dict, dict, dict, dict, dict]:
    signals = cfg.get("signals", {}) if isinstance(cfg.get("signals"), dict) else {}
    output = signals.get("output", {}) if isinstance(signals.get("output"), dict) else {}
    donchian = signals.get("donchian", {}) if isinstance(signals.get("donchian"), dict) else {}
    ema_adx = signals.get("ema_adx", {}) if isinstance(signals.get("ema_adx"), dict) else {}
    mean_reversion = signals.get("mean_reversion", {}) if isinstance(signals.get("mean_reversion"), dict) else {}
    execution = signals.get("execution", {}) if isinstance(signals.get("execution"), dict) else {}
    return output, donchian, ema_adx, mean_reversion, execution


def _config_allocator(cfg: dict) -> dict:
    allocator = cfg.get("allocator", {}) if isinstance(cfg.get("allocator"), dict) else {}
    strategy_weights = allocator.get("strategy_weights", {}) if isinstance(allocator.get("strategy_weights"), dict) else {}
    return strategy_weights


def _coerce_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _config_backtest(cfg: dict) -> tuple[dict, dict, dict]:
    backtest = cfg.get("backtest", {}) if isinstance(cfg.get("backtest"), dict) else {}
    output = backtest.get("output", {}) if isinstance(backtest.get("output"), dict) else {}
    execution = backtest.get("execution", {}) if isinstance(backtest.get("execution"), dict) else {}
    assumptions = backtest.get("assumptions", {}) if isinstance(backtest.get("assumptions"), dict) else {}
    return output, execution, assumptions


def _config_walk_forward(cfg: dict) -> dict:
    backtest = cfg.get("backtest", {}) if isinstance(cfg.get("backtest"), dict) else {}
    wf = backtest.get("walk_forward", {}) if isinstance(backtest.get("walk_forward"), dict) else {}
    return wf


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
    min_regime_run_bars: Optional[int] = typer.Option(
        None,
        help="Minimum consecutive 1h bars required to confirm a regime switch",
    ),
    config: Path = typer.Option(Path("configs/default.yaml"), help="Default config path"),
) -> None:
    """Detect 1h regime labels (R1-R4) from 1h bars and write monthly outputs."""
    cfg = _read_yaml(config)
    cfg_data, cfg_paths = _config_data(cfg)
    trend_cfg, vol_cfg, out_cfg = _config_regime(cfg)
    smoothing_cfg = _config_regime_smoothing(cfg)

    resolved_symbol = symbol or cfg_data.get("symbol") or "BTCUSDT"
    resolved_input_path = input_path or Path(cfg_paths.get("bars_1h", "data/bars_1h"))
    resolved_output_dir = output_dir or Path(out_cfg.get("dir_1h", "results/regime"))
    resolved_min_regime_run_bars = int(
        min_regime_run_bars
        if min_regime_run_bars is not None
        else smoothing_cfg.get("min_regime_run_bars", 1)
    )

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
        min_regime_run_bars=resolved_min_regime_run_bars,
    )

    logger.info("Regime detect rows in: %d", summary["rows_in"])
    logger.info("Regime detect rows out: %d", summary["rows_out"])
    logger.info("Regime files written: %d", len(summary["files_written"]))
    logger.info("Regime counts: %s", summary["regime_counts"])
    logger.info("Regime min_run_bars: %d", summary["min_regime_run_bars"])
    logger.info("Regime switch_count: %d", summary["switch_count"])


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


@app.command("plot-regime")
def plot_regime_cmd(
    bars_1h_path: Optional[Path] = typer.Option(None, help="1h bars parquet folder or file"),
    regime_1h_path: Optional[Path] = typer.Option(None, help="Regime 1h parquet folder or file"),
    output_path: Optional[Path] = typer.Option(None, help="Plot output file or folder"),
    symbol: Optional[str] = typer.Option(None, help="Trading symbol, defaults to config value"),
    default_regime: Optional[str] = typer.Option(None, help="Fallback regime when missing"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="Default config path"),
) -> None:
    """Plot 1h close price with regime background bands and save image."""
    cfg = _read_yaml(config)
    cfg_data, cfg_paths = _config_data(cfg)
    _, _, out_cfg = _config_regime(cfg)

    resolved_symbol = symbol or cfg_data.get("symbol") or "BTCUSDT"
    resolved_bars_1h_path = bars_1h_path or Path(cfg_paths.get("bars_1h", "data/bars_1h"))
    resolved_regime_1h_path = regime_1h_path or Path(out_cfg.get("dir_1h", "results/regime"))
    resolved_output_path = output_path or Path(out_cfg.get("dir_plots", "results/regime/plots"))
    resolved_default_regime = default_regime or str(out_cfg.get("default_regime_15m", "R4"))

    summary = run_plot_price_regime_1h(
        bars_1h_path=resolved_bars_1h_path,
        regime_1h_path=resolved_regime_1h_path,
        output_path=resolved_output_path,
        symbol=resolved_symbol,
        default_regime=resolved_default_regime,
    )

    logger.info("Plot rows bars_1h: %d", summary["rows_bars_1h"])
    logger.info("Plot rows regime_1h: %d", summary["rows_regime_1h"])
    logger.info("Plot rows rendered: %d", summary["rows_plot"])
    logger.info("Plot output: %s", summary["output_path"])
    logger.info("Plot range: %s -> %s", summary["start_timestamp"], summary["end_timestamp"])


@app.command("generate-signals")
def generate_signals_cmd(
    bars_15m_path: Optional[Path] = typer.Option(None, help="15m bars parquet folder or file"),
    regime_15m_path: Optional[Path] = typer.Option(None, help="15m regime parquet folder or file"),
    output_dir: Optional[Path] = typer.Option(None, help="Signals 15m output folder"),
    symbol: Optional[str] = typer.Option(None, help="Trading symbol, defaults to config value"),
    default_regime: Optional[str] = typer.Option(None, help="Fallback regime for missing 15m labels"),
    long_only: Optional[bool] = typer.Option(None, help="Enforce long-only target positions"),
    min_hold_bars: Optional[int] = typer.Option(None, help="Minimum hold bars before changing position"),
    rebalance_threshold: Optional[float] = typer.Option(None, help="Ignore small target_position changes below this threshold"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="Default config path"),
) -> None:
    """Generate 15m strategy signals and composite target positions."""
    cfg = _read_yaml(config)
    cfg_data, cfg_paths = _config_data(cfg)
    _, _, regime_out_cfg = _config_regime(cfg)
    signals_out_cfg, don_cfg, ema_cfg, mr_cfg, signals_exec_cfg = _config_signals(cfg)
    allocator_cfg = _config_allocator(cfg)

    resolved_symbol = symbol or cfg_data.get("symbol") or "BTCUSDT"
    resolved_bars_15m_path = bars_15m_path or Path(cfg_paths.get("bars_15m", "data/bars_15m"))
    resolved_regime_15m_path = regime_15m_path or Path(regime_out_cfg.get("dir_15m", "results/regime"))
    resolved_output_dir = output_dir or Path(signals_out_cfg.get("dir_15m", cfg_paths.get("signals", "results/signals")))
    resolved_default_regime = default_regime or str(regime_out_cfg.get("default_regime_15m", "R4"))
    venue = str(cfg_data.get("venue", "")).lower()
    spot_default_long_only = venue.endswith("spot")
    resolved_long_only = _coerce_bool(
        long_only if long_only is not None else signals_exec_cfg.get("long_only"),
        default=spot_default_long_only,
    )
    resolved_min_hold_bars = int(min_hold_bars if min_hold_bars is not None else signals_exec_cfg.get("min_hold_bars", 1))
    resolved_rebalance_threshold = float(
        rebalance_threshold if rebalance_threshold is not None else signals_exec_cfg.get("rebalance_threshold", 0.0)
    )

    summary = run_generate_signals(
        bars_15m_path=resolved_bars_15m_path,
        regime_15m_path=resolved_regime_15m_path,
        output_dir=resolved_output_dir,
        symbol=resolved_symbol,
        default_regime=resolved_default_regime,
        donchian_window=int(don_cfg.get("window", 20)),
        donchian_hold_until_opposite=bool(don_cfg.get("hold_until_opposite", True)),
        ema_fast=int(ema_cfg.get("ema_fast", 21)),
        ema_slow=int(ema_cfg.get("ema_slow", 55)),
        ema_adx_window=int(ema_cfg.get("adx_window", 14)),
        ema_adx_threshold=float(ema_cfg.get("adx_threshold", 20.0)),
        ema_use_adx_filter=bool(ema_cfg.get("use_adx_filter", True)),
        mr_z_window=int(mr_cfg.get("z_window", 48)),
        mr_entry_z=float(mr_cfg.get("entry_z", 1.5)),
        mr_exit_z=float(mr_cfg.get("exit_z", 0.5)),
        regime_allocations=allocator_cfg,
        long_only=resolved_long_only,
        min_hold_bars=resolved_min_hold_bars,
        rebalance_threshold=resolved_rebalance_threshold,
    )

    logger.info("Generate signals rows bars_15m: %d", summary["rows_bars_15m"])
    logger.info("Generate signals rows regime_15m: %d", summary["rows_regime_15m"])
    logger.info("Generate signals rows out: %d", summary["rows_out"])
    logger.info("Generate signals files written: %d", len(summary["files_written"]))
    logger.info(
        "Composite signal range: [%.4f, %.4f]",
        summary["signal_composite_min"],
        summary["signal_composite_max"],
    )
    logger.info(
        "Execution constraints: long_only=%s min_hold_bars=%d rebalance_threshold=%.4f",
        summary["long_only"],
        summary["min_hold_bars"],
        summary["rebalance_threshold"],
    )


@app.command("run-backtest")
def run_backtest_cmd(
    bars_15m_path: Optional[Path] = typer.Option(None, help="15m bars parquet folder or file"),
    signals_15m_path: Optional[Path] = typer.Option(None, help="15m signals parquet folder or file"),
    output_dir: Optional[Path] = typer.Option(None, help="Backtest output folder"),
    metrics_dir: Optional[Path] = typer.Option(None, help="Backtest metrics output folder"),
    symbol: Optional[str] = typer.Option(None, help="Trading symbol, defaults to config value"),
    initial_equity: Optional[float] = typer.Option(None, help="Initial equity"),
    fee_bps: Optional[float] = typer.Option(None, help="Per-turnover fee in bps"),
    slippage_bps: Optional[float] = typer.Option(None, help="Per-turnover slippage in bps"),
    position_lag_bars: Optional[int] = typer.Option(None, help="Bars to lag target_position for execution"),
    bars_per_year: Optional[int] = typer.Option(None, help="Bars per year used for annualization"),
    long_only: Optional[bool] = typer.Option(None, help="Enforce long-only positions in simulation"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="Default config path"),
) -> None:
    """Run 15m backtest from generated signals and write outputs + metrics."""
    cfg = _read_yaml(config)
    cfg_data, cfg_paths = _config_data(cfg)
    _, _, _, _, signals_exec_cfg = _config_signals(cfg)
    backtest_out_cfg, backtest_exec_cfg, backtest_assump_cfg = _config_backtest(cfg)

    resolved_symbol = symbol or cfg_data.get("symbol") or "BTCUSDT"
    resolved_bars_15m_path = bars_15m_path or Path(cfg_paths.get("bars_15m", "data/bars_15m"))
    resolved_signals_15m_path = signals_15m_path or Path(cfg_paths.get("signals", "results/signals"))
    resolved_output_dir = output_dir or Path(backtest_out_cfg.get("dir_backtest", cfg_paths.get("backtest", "results/backtest")))
    resolved_metrics_dir = metrics_dir or Path(backtest_out_cfg.get("dir_metrics", cfg_paths.get("metrics", "results/metrics")))

    resolved_initial_equity = float(initial_equity if initial_equity is not None else backtest_exec_cfg.get("initial_equity", 100000.0))
    resolved_fee_bps = float(fee_bps if fee_bps is not None else backtest_exec_cfg.get("fee_bps", 4.0))
    resolved_slippage_bps = float(slippage_bps if slippage_bps is not None else backtest_exec_cfg.get("slippage_bps", 1.0))
    resolved_position_lag_bars = int(position_lag_bars if position_lag_bars is not None else backtest_assump_cfg.get("position_lag_bars", 1))
    resolved_bars_per_year = int(bars_per_year if bars_per_year is not None else backtest_assump_cfg.get("bars_per_year_15m", 35040))
    venue = str(cfg_data.get("venue", "")).lower()
    spot_default_long_only = venue.endswith("spot")
    cfg_long_only = backtest_exec_cfg.get("long_only", signals_exec_cfg.get("long_only"))
    resolved_long_only = _coerce_bool(long_only if long_only is not None else cfg_long_only, default=spot_default_long_only)

    summary = run_backtest(
        bars_15m_path=resolved_bars_15m_path,
        signals_15m_path=resolved_signals_15m_path,
        output_dir=resolved_output_dir,
        metrics_dir=resolved_metrics_dir,
        symbol=resolved_symbol,
        initial_equity=resolved_initial_equity,
        fee_bps=resolved_fee_bps,
        slippage_bps=resolved_slippage_bps,
        position_lag_bars=resolved_position_lag_bars,
        bars_per_year=resolved_bars_per_year,
        long_only=resolved_long_only,
    )

    logger.info("Backtest rows bars_15m: %d", summary["rows_bars_15m"])
    logger.info("Backtest rows signals_15m: %d", summary["rows_signals_15m"])
    logger.info("Backtest rows out: %d", summary["rows_out"])
    logger.info("Backtest files written: %d", len(summary["files_written"]))
    logger.info("Metrics path: %s", summary["metrics_path"])
    logger.info("Execution constraints: long_only=%s", summary.get("long_only"))

    metrics = summary.get("metrics", {})
    if metrics:
        logger.info(
            "Strategy total_return=%.6f sharpe=%.4f max_drawdown=%.6f",
            float(metrics.get("total_return", 0.0)),
            float(metrics.get("sharpe", 0.0)),
            float(metrics.get("max_drawdown", 0.0)),
        )
        logger.info(
            "BuyHold total_return=%.6f sharpe=%.4f max_drawdown=%.6f",
            float(metrics.get("bh_total_return", 0.0)),
            float(metrics.get("bh_sharpe", 0.0)),
            float(metrics.get("bh_max_drawdown", 0.0)),
        )
        logger.info(
            "Excess vs BuyHold total_return=%.6f sharpe=%.4f outperform=%s",
            float(metrics.get("excess_total_return", 0.0)),
            float(metrics.get("excess_sharpe", 0.0)),
            bool(metrics.get("outperform_buy_hold", False)),
        )


@app.command("diagnose-backtest")
def diagnose_backtest_cmd(
    backtest_path: Optional[Path] = typer.Option(None, help="Backtest parquet folder or file"),
    signals_15m_path: Optional[Path] = typer.Option(None, help="15m signals parquet folder or file"),
    regime_15m_path: Optional[Path] = typer.Option(None, help="15m regime parquet folder or file"),
    output_dir: Optional[Path] = typer.Option(None, help="Diagnostics output folder"),
    symbol: Optional[str] = typer.Option(None, help="Trading symbol, defaults to config value"),
    default_regime: Optional[str] = typer.Option(None, help="Fallback regime for missing labels"),
    bars_per_year: Optional[int] = typer.Option(None, help="Bars per year used for annualization"),
    position_lag_bars: Optional[int] = typer.Option(None, help="Execution lag used for contribution alignment"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="Default config path"),
) -> None:
    """Run backtest attribution diagnostics (regime, strategy contribution, cost)."""
    cfg = _read_yaml(config)
    cfg_data, cfg_paths = _config_data(cfg)
    _, _, regime_out_cfg = _config_regime(cfg)
    allocator_cfg = _config_allocator(cfg)
    backtest_out_cfg, _, backtest_assump_cfg = _config_backtest(cfg)

    resolved_symbol = symbol or cfg_data.get("symbol") or "BTCUSDT"
    resolved_backtest_path = backtest_path or Path(backtest_out_cfg.get("dir_backtest", cfg_paths.get("backtest", "results/backtest")))
    resolved_signals_path = signals_15m_path or Path(cfg_paths.get("signals", "results/signals"))
    resolved_regime_path = regime_15m_path or Path(regime_out_cfg.get("dir_15m", "results/regime"))
    resolved_output_dir = output_dir or Path(backtest_out_cfg.get("dir_metrics", cfg_paths.get("metrics", "results/metrics")))
    resolved_default_regime = default_regime or str(regime_out_cfg.get("default_regime_15m", "R4"))
    resolved_bars_per_year = int(bars_per_year if bars_per_year is not None else backtest_assump_cfg.get("bars_per_year_15m", 35040))
    resolved_position_lag_bars = int(position_lag_bars if position_lag_bars is not None else backtest_assump_cfg.get("position_lag_bars", 1))

    summary = run_backtest_diagnostics(
        backtest_path=resolved_backtest_path,
        signals_path=resolved_signals_path,
        regime_15m_path=resolved_regime_path,
        output_dir=resolved_output_dir,
        symbol=resolved_symbol,
        bars_per_year=resolved_bars_per_year,
        position_lag_bars=resolved_position_lag_bars,
        default_regime=resolved_default_regime,
        regime_allocations=allocator_cfg,
    )

    logger.info("Diagnostics rows backtest: %d", summary["rows_backtest"])
    logger.info("Diagnostics rows signals: %d", summary["rows_signals"])
    logger.info("Diagnostics rows regime: %d", summary["rows_regime"])
    logger.info("Diagnostics rows joined: %d", summary["rows_joined"])
    logger.info("Diagnostics files written: %d", len(summary["files_written"]))
    logger.info("Diagnostics summary path: %s", summary["summary_path"])


@app.command("walk-forward")
def walk_forward_cmd(
    bars_15m_path: Optional[Path] = typer.Option(None, help="15m bars parquet folder or file"),
    regime_15m_path: Optional[Path] = typer.Option(None, help="15m regime parquet folder or file"),
    output_dir: Optional[Path] = typer.Option(None, help="Walk-forward output folder"),
    symbol: Optional[str] = typer.Option(None, help="Trading symbol, defaults to config value"),
    selection_metric: Optional[str] = typer.Option(None, help="Selection metric: sharpe/total_return/annual_return/calmar"),
    default_regime: Optional[str] = typer.Option(None, help="Fallback regime for missing labels"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="Default config path"),
) -> None:
    """Run rolling walk-forward validation with valid-if-provided, else train selection."""
    cfg = _read_yaml(config)
    cfg_data, cfg_paths = _config_data(cfg)
    _, _, regime_out_cfg = _config_regime(cfg)
    signals_out_cfg, don_cfg, ema_cfg, mr_cfg, signals_exec_cfg = _config_signals(cfg)
    allocator_cfg = _config_allocator(cfg)
    backtest_out_cfg, backtest_exec_cfg, backtest_assump_cfg = _config_backtest(cfg)
    walk_cfg = _config_walk_forward(cfg)

    resolved_symbol = symbol or cfg_data.get("symbol") or "BTCUSDT"
    resolved_bars_15m_path = bars_15m_path or Path(cfg_paths.get("bars_15m", "data/bars_15m"))
    resolved_regime_15m_path = regime_15m_path or Path(regime_out_cfg.get("dir_15m", "results/regime"))
    resolved_output_dir = output_dir or Path(walk_cfg.get("output_dir", backtest_out_cfg.get("dir_metrics", cfg_paths.get("metrics", "results/metrics"))))
    resolved_default_regime = default_regime or str(regime_out_cfg.get("default_regime_15m", "R4"))
    resolved_selection_metric = str(selection_metric or walk_cfg.get("selection_metric", "sharpe"))

    venue = str(cfg_data.get("venue", "")).lower()
    spot_default_long_only = venue.endswith("spot")
    cfg_long_only = backtest_exec_cfg.get("long_only", signals_exec_cfg.get("long_only"))
    resolved_long_only = _coerce_bool(cfg_long_only, default=spot_default_long_only)

    resolved_initial_equity = float(backtest_exec_cfg.get("initial_equity", 100000.0))
    resolved_fee_bps = float(backtest_exec_cfg.get("fee_bps", 4.0))
    resolved_slippage_bps = float(backtest_exec_cfg.get("slippage_bps", 1.0))
    resolved_position_lag_bars = int(backtest_assump_cfg.get("position_lag_bars", 1))
    resolved_bars_per_year = int(backtest_assump_cfg.get("bars_per_year_15m", 35040))

    resolved_min_hold_bars = int(signals_exec_cfg.get("min_hold_bars", 1))
    resolved_rebalance_threshold = float(signals_exec_cfg.get("rebalance_threshold", 0.0))

    summary = run_walk_forward(
        bars_15m_path=resolved_bars_15m_path,
        regime_15m_path=resolved_regime_15m_path,
        output_dir=resolved_output_dir,
        symbol=resolved_symbol,
        base_regime_allocations=allocator_cfg,
        candidate_allocations=walk_cfg.get("candidates"),
        folds=walk_cfg.get("folds"),
        selection_metric=resolved_selection_metric,
        default_regime=resolved_default_regime,
        bars_per_year=resolved_bars_per_year,
        initial_equity=resolved_initial_equity,
        fee_bps=resolved_fee_bps,
        slippage_bps=resolved_slippage_bps,
        position_lag_bars=resolved_position_lag_bars,
        long_only=resolved_long_only,
        min_hold_bars=resolved_min_hold_bars,
        rebalance_threshold=resolved_rebalance_threshold,
        donchian_window=int(don_cfg.get("window", 20)),
        donchian_hold_until_opposite=bool(don_cfg.get("hold_until_opposite", True)),
        ema_fast=int(ema_cfg.get("ema_fast", 21)),
        ema_slow=int(ema_cfg.get("ema_slow", 55)),
        ema_adx_window=int(ema_cfg.get("adx_window", 14)),
        ema_adx_threshold=float(ema_cfg.get("adx_threshold", 20.0)),
        ema_use_adx_filter=bool(ema_cfg.get("use_adx_filter", True)),
        mr_z_window=int(mr_cfg.get("z_window", 48)),
        mr_entry_z=float(mr_cfg.get("entry_z", 1.5)),
        mr_exit_z=float(mr_cfg.get("exit_z", 0.5)),
    )

    logger.info("Walk-forward rows bars: %d", summary["rows_bars"])
    logger.info("Walk-forward rows regime: %d", summary["rows_regime"])
    logger.info("Walk-forward candidate count: %d", summary["candidate_count"])
    logger.info("Walk-forward fold count: %d", summary["fold_count"])
    logger.info("Walk-forward selected candidates: %s", summary["selected_candidate_count"])
    logger.info(
        "Walk-forward OOS total_return=%.6f sharpe=%.4f vs BH total_return=%.6f",
        float(summary["oos_metrics"].get("total_return", 0.0)),
        float(summary["oos_metrics"].get("sharpe", 0.0)),
        float(summary["oos_metrics"].get("bh_total_return", 0.0)),
    )
    logger.info("Walk-forward summary path: %s", summary["summary_path"])


if __name__ == "__main__":
    app()
