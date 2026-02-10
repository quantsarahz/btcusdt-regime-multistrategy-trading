from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Create directory if missing and return the same path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
