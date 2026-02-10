# BTCUSDT Regime Multi-Strategy System (Rebuilt)

A clean-slate project that follows this pipeline:
1. Download only 1-minute market data.
2. Run 1-minute quality control and cleaning.
3. Build 15-minute and 1-hour bars from cleaned 1-minute bars.
4. Detect market regime on 1-hour bars.
5. Run three strategies on 15-minute bars.
6. Backtest and export metrics.

## Project Structure

The repository follows the exact architecture requested in this rebuild.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## Minimal CLI Flow

```bash
bt-regime fetch-1m --input-csv data/raw_1m/sample.csv --output-parquet data/raw_1m/sample.parquet
bt-regime qc-1m --input-parquet data/raw_1m/sample.parquet --output-parquet data/clean_1m/sample_clean.parquet
bt-regime build-bars --input-parquet data/clean_1m/sample_clean.parquet --output-15m data/bars_15m/sample_15m.parquet --output-1h data/bars_1h/sample_1h.parquet
bt-regime detect-regime --input-1h data/bars_1h/sample_1h.parquet --output-regime results/regime/sample_regime.parquet
```
