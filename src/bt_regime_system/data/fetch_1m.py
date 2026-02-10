from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from bt_regime_system.utils.io import ensure_dir
from bt_regime_system.utils.time_utils import parse_utc_timestamp, to_milliseconds

BINANCE_BASE_URL = "https://api.binance.com"
KLINES_PATH = "/api/v3/klines"
INTERVAL = "1m"
MS_PER_MINUTE = 60_000
OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


class BinanceAPIError(RuntimeError):
    """Raised when Binance kline API repeatedly fails."""


def _request_klines(
    session: requests.Session,
    base_url: str,
    params: dict[str, str | int],
    max_retries: int = 3,
    timeout_sec: int = 30,
) -> list[list]:
    url = f"{base_url.rstrip('/')}{KLINES_PATH}"
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout_sec)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list):
                raise BinanceAPIError("Unexpected kline payload type")
            return data
        except Exception as exc:  # pragma: no cover - defensive retry path
            last_error = exc
            if attempt < max_retries:
                time.sleep(0.5 * attempt)

    raise BinanceAPIError(f"Failed to request klines after {max_retries} attempts: {last_error}")


def _rows_to_frame(rows: list[list]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    raw = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    out = pd.DataFrame(
        {
            # Binance returns open_time; contract uses bar_close_time.
            "timestamp": pd.to_datetime(raw["open_time"], unit="ms", utc=True)
            + pd.Timedelta(minutes=1),
            "open": raw["open"].astype(float),
            "high": raw["high"].astype(float),
            "low": raw["low"].astype(float),
            "close": raw["close"].astype(float),
            "volume": raw["volume"].astype(float),
        }
    )
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return out.reset_index(drop=True)


def fetch_1m_klines(
    symbol: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    base_url: str = BINANCE_BASE_URL,
    limit: int = 1000,
    pause_sec: float = 0.1,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch 1m klines from Binance for [start, end] using UTC close-time timestamps."""
    start_utc = parse_utc_timestamp(start)
    end_utc = parse_utc_timestamp(end)
    if end_utc <= start_utc:
        raise ValueError("`end` must be later than `start`")

    # API filters by open_time; contract timestamps are close_time at minute boundary.
    api_start_ms = to_milliseconds(start_utc - pd.Timedelta(minutes=1))
    api_end_ms = to_milliseconds(end_utc - pd.Timedelta(minutes=1))

    client = session or requests.Session()
    cursor = api_start_ms
    all_rows: list[list] = []

    while cursor <= api_end_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": INTERVAL,
            "startTime": cursor,
            "endTime": api_end_ms,
            "limit": min(limit, 1000),
        }
        page = _request_klines(client, base_url=base_url, params=params)
        if not page:
            break

        all_rows.extend(page)
        last_open_ms = int(page[-1][0])
        next_cursor = last_open_ms + MS_PER_MINUTE
        if next_cursor <= cursor:
            break
        cursor = next_cursor

        if pause_sec > 0:
            time.sleep(pause_sec)

    data = _rows_to_frame(all_rows)
    if data.empty:
        return data

    mask = (data["timestamp"] >= start_utc) & (data["timestamp"] <= end_utc)
    return data.loc[mask, OUTPUT_COLUMNS].reset_index(drop=True)


def monthly_raw_filename(symbol: str, month: str) -> str:
    return f"{symbol.upper()}_1m_{month}.parquet"


def write_monthly_raw_1m(data: pd.DataFrame, out_dir: Path, symbol: str) -> list[Path]:
    """Write/merge raw 1m bars into monthly parquet files."""
    ensure_dir(out_dir)
    if data.empty:
        return []

    required = set(OUTPUT_COLUMNS)
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns for write: {sorted(missing)}")

    frame = data[OUTPUT_COLUMNS].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["month"] = frame["timestamp"].dt.strftime("%Y-%m")

    written: list[Path] = []
    for month, chunk in frame.groupby("month", sort=True):
        target = out_dir / monthly_raw_filename(symbol, month)
        new_rows = chunk[OUTPUT_COLUMNS].sort_values("timestamp")

        if target.exists():
            existing = pd.read_parquet(target)
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
            merged = pd.concat([existing, new_rows], ignore_index=True)
            merged = merged.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        else:
            merged = new_rows.drop_duplicates("timestamp", keep="last")

        merged.to_parquet(target, index=False)
        written.append(target)

    return written
