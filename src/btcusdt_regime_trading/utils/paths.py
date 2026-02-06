"""Project path utilities."""

from pathlib import Path


# Locate project root based on this file's location.
# File: <root>/src/btcusdt_regime_trading/utils/paths.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Common project directories.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Ensure essential directories exist.
for _dir in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, ARTIFACTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
