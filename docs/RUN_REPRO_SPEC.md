# Run & Reproducibility Specification

## 1. Scope

This document defines the standard reproducible workflow for this project:
- Symbol: `BTCUSDT`
- Venue: `binance_spot`
- Timezone: `UTC`
- Source frequency: `1m`
- Derived frequencies: `15m`, `1h`
- Regime classes: `R1`, `R2`, `R3`, `R4`

Primary config and contract files:
- `configs/default.yaml`
- `configs/data_contract.yaml`

## 2. Environment

Required:
- Python `>=3.10`
- Install from project metadata (`pyproject.toml`)

Setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev]'
```

Optional sanity check:

```bash
PYTHONPATH=src pytest -q
```

## 3. Canonical Execution Order

Run the following commands in order (all from repo root):

```bash
PYTHONPATH=src python -m bt_regime_system.cli fetch-1m --config configs/default.yaml
PYTHONPATH=src python -m bt_regime_system.cli qc-1m --input-path data/raw_1m --config configs/default.yaml
PYTHONPATH=src python -m bt_regime_system.cli build-bars --input-path data/clean_1m --config configs/default.yaml
PYTHONPATH=src python -m bt_regime_system.cli qc-bars --config configs/default.yaml
PYTHONPATH=src python -m bt_regime_system.cli detect-regime --config configs/default.yaml
PYTHONPATH=src python -m bt_regime_system.cli align-regime --config configs/default.yaml
PYTHONPATH=src python -m bt_regime_system.cli qc-regime --config configs/default.yaml
PYTHONPATH=src python -m bt_regime_system.cli plot-regime --config configs/default.yaml
PYTHONPATH=src python -m bt_regime_system.cli generate-signals --config configs/default.yaml
PYTHONPATH=src python -m bt_regime_system.cli run-backtest --config configs/default.yaml
PYTHONPATH=src python -m bt_regime_system.cli diagnose-backtest --config configs/default.yaml
PYTHONPATH=src python -m bt_regime_system.cli walk-forward --config configs/default.yaml
PYTHONPATH=src python -m bt_regime_system.cli stress-backtest --config configs/default.yaml
```

## 4. Step Inputs/Outputs

### 4.1 fetch-1m
Input:
- Binance klines API
- Default time range in CLI:
  - `2022-01-01T00:01:00Z` to `2025-12-31T23:59:00Z`

Output:
- `data/raw_1m/BTCUSDT_1m_YYYY-MM.parquet`

Behavior:
- Monthly partitioned write.
- Existing monthly file is merged by timestamp, deduplicated, keep last.

### 4.2 qc-1m
Input:
- `data/raw_1m/*.parquet`

Output:
- Clean data: `data/clean_1m/BTCUSDT_1m_clean_YYYY-MM.parquet`
- Monthly QC report: `data/reports/qc_clean_1m_YYYY-MM.json`

Behavior:
- Enforces timestamp/price/volume validity.
- Optional missing minute fill policy from config.

### 4.3 build-bars
Input:
- `data/clean_1m/*.parquet`

Output:
- `data/bars_15m/BTCUSDT_15m_YYYY-MM.parquet`
- `data/bars_1h/BTCUSDT_1h_YYYY-MM.parquet`

Behavior:
- Resample right-closed/right-labeled.
- Monthly partition uses timestamp-1min for month boundary assignment.

### 4.4 qc-bars
Input:
- 15m and 1h bar folders

Output:
- Monthly reports:
  - `data/reports/qc_bars_15min_YYYY-MM.json`
  - `data/reports/qc_bars_1h_YYYY-MM.json`
- Summary reports:
  - `data/reports/qc_bars_15min_summary.json`
  - `data/reports/qc_bars_1h_summary.json`

### 4.5 detect-regime
Input:
- `data/bars_1h/*.parquet`

Output:
- `results/regime/BTCUSDT_regime_1h_YYYY-MM.parquet`

Behavior:
- Regime from trend x volatility.
- Smoothing active: `min_regime_run_bars=3` (default from config).

### 4.6 align-regime
Input:
- 1h regime files
- 15m bars

Output:
- `results/regime/BTCUSDT_regime_15m_YYYY-MM.parquet`

Behavior:
- Forward-fill last completed 1h regime to 15m timeline.
- No lookahead: uses backward as-of alignment.

### 4.7 qc-regime
Input:
- 1h and 15m regime outputs

Output:
- Monthly reports:
  - `data/reports/qc_regime_1h_YYYY-MM.json`
  - `data/reports/qc_regime_15m_YYYY-MM.json`
- Summary reports:
  - `data/reports/qc_regime_1h_summary.json`
  - `data/reports/qc_regime_15m_summary.json`

Checks include:
- Continuity and duplicates
- NaN and invalid labels
- Switch count/rate
- Lookahead violations and mismatch checks

### 4.8 plot-regime
Output:
- `results/regime/plots/BTCUSDT_price_regime_1h.png`

### 4.9 generate-signals
Input:
- 15m bars
- 15m aligned regime

Output:
- `results/signals/BTCUSDT_signals_15m_YYYY-MM.parquet`

Signal schema contract:
- `timestamp`
- `signal_donchian`
- `signal_ema_adx`
- `signal_mean_reversion`
- `signal_composite`
- `target_position`

### 4.10 run-backtest
Input:
- 15m bars
- 15m signals

Output:
- Monthly backtest:
  - `results/backtest/BTCUSDT_backtest_15m_YYYY-MM.parquet`
- Metrics:
  - `results/metrics/BTCUSDT_backtest_metrics.json`

Includes:
- Strategy curve
- Buy-and-hold baseline metrics (`bh_*`)

### 4.11 diagnose-backtest
Output (under metrics directory):
- `BTCUSDT_backtest_diag_regime_monthly.csv`
- `BTCUSDT_backtest_diag_regime_summary.csv`
- `BTCUSDT_backtest_diag_strategy_monthly.csv`
- `BTCUSDT_backtest_diag_strategy_summary.json`
- `BTCUSDT_backtest_diag_cost_monthly.csv`
- `BTCUSDT_backtest_diag_cost_summary.json`
- `BTCUSDT_backtest_diag_summary.json`

### 4.12 walk-forward
Output:
- `results/metrics/BTCUSDT_walk_forward_scores.csv`
- `results/metrics/BTCUSDT_walk_forward_folds.csv`
- `results/metrics/BTCUSDT_walk_forward_oos_monthly.csv`
- `results/metrics/BTCUSDT_walk_forward_summary.json`

Behavior:
- Uses non-overlapping test windows (enforced in fold parser).

### 4.13 stress-backtest
Output:
- `results/metrics/BTCUSDT_backtest_stress_results.csv`
- `results/metrics/BTCUSDT_backtest_stress_summary.json`

Scenarios:
- Fee stress
- Slippage stress
- Execution lag stress
- Volatility multiplier stress
- Optional combined worst case

## 5. Reproducibility Checklist

Before run:
- Record commit hash:

```bash
git rev-parse HEAD
```

- Confirm config files unchanged for the run.

After run:
- Check expected monthly file counts (for 2022-2025 full range):

```bash
ls data/raw_1m/BTCUSDT_1m_*.parquet | wc -l
ls data/clean_1m/BTCUSDT_1m_clean_*.parquet | wc -l
ls data/bars_15m/BTCUSDT_15m_*.parquet | wc -l
ls data/bars_1h/BTCUSDT_1h_*.parquet | wc -l
ls results/regime/BTCUSDT_regime_1h_*.parquet | wc -l
ls results/regime/BTCUSDT_regime_15m_*.parquet | wc -l
ls results/signals/BTCUSDT_signals_15m_*.parquet | wc -l
ls results/backtest/BTCUSDT_backtest_15m_*.parquet | wc -l
```

Expected count for full 2022-01 to 2025-12 range: `48` each monthly set.

- Confirm core summary files exist:
  - `data/reports/qc_bars_15min_summary.json`
  - `data/reports/qc_bars_1h_summary.json`
  - `data/reports/qc_regime_1h_summary.json`
  - `data/reports/qc_regime_15m_summary.json`
  - `results/metrics/BTCUSDT_backtest_metrics.json`
  - `results/metrics/BTCUSDT_walk_forward_summary.json`
  - `results/metrics/BTCUSDT_backtest_stress_summary.json`

## 6. Drift Sources (if results differ)

Most common causes:
- Config changed (`default.yaml`)
- Different run date with updated upstream exchange data corrections
- Partial rerun scope (not full chain)
- Different Python/package versions
- Manual edits in intermediate parquet outputs

To minimize drift:
- Use same commit, same config, same full execution order.
- Keep all timestamps in UTC.
- Prefer full rerun for production-grade reproduction.
