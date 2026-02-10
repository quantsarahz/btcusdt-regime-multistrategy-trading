from __future__ import annotations

from pathlib import Path


def ensure_parent(path: Path) -> None:
    """Create parent directories for a file path if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
