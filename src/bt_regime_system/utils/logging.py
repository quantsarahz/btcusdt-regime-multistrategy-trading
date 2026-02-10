from __future__ import annotations

import logging


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Build a standard project logger."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    return logging.getLogger(name)
