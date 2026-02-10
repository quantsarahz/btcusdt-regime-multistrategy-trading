from __future__ import annotations

import pandas as pd


def parse_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    """Parse a timestamp-like value and normalize it to UTC."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def to_milliseconds(ts: pd.Timestamp) -> int:
    """Convert a UTC timestamp to Unix milliseconds."""
    utc_ts = parse_utc_timestamp(ts)
    return int(utc_ts.timestamp() * 1000)
