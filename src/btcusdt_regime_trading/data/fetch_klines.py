"""Binance spot kline downloader."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import time

import pandas as pd
import requests
import yaml

from btcusdt_regime_trading.utils.paths import PROJECT_ROOT, RAW_DATA_DIR


CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
RETRYABLE_STATUSES = {418, 429, 500, 502, 503, 504}


def load_config() -> dict:
    """Load YAML config from the project configs directory."""
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _has_time_component(value: str) -> bool:
    return "T" in value or ":" in value


def _parse_datetime(value: str, *, is_end: bool) -> datetime:
    if not isinstance(value, str):
        raise ValueError("time_range values must be strings")

    has_time = _has_time_component(value)
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        if not has_time:
            if is_end:
                dt = dt.replace(hour=23, minute=59, second=59, microsecond=999_000)
            else:
                dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _year_slices(start_dt: datetime, end_dt: datetime) -> Iterable[Tuple[int, datetime, datetime]]:
    year = start_dt.year
    while year <= end_dt.year:
        year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
        year_end = datetime(year, 12, 31, 23, 59, 59, 999_000, tzinfo=timezone.utc)
        chunk_start = max(start_dt, year_start)
        chunk_end = min(end_dt, year_end)
        if chunk_start <= chunk_end:
            yield year, chunk_start, chunk_end
        year += 1


def _empty_df(symbol: str, interval: str) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "open_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "close_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
            "quote_volume": pd.Series(dtype="float64"),
            "num_trades": pd.Series(dtype="int64"),
            "taker_buy_base_volume": pd.Series(dtype="float64"),
            "taker_buy_quote_volume": pd.Series(dtype="float64"),
            "symbol": pd.Series(dtype="object"),
            "interval": pd.Series(dtype="object"),
        }
    )
    df["symbol"] = symbol
    df["interval"] = interval
    return df.set_index("open_time", drop=False)


def _klines_to_df(rows: List[list], symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    if not rows:
        return _empty_df(symbol, interval)

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "num_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df = df.drop(columns=["ignore"])

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    float_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["num_trades"] = pd.to_numeric(df["num_trades"], errors="coerce").astype("int64")

    df["symbol"] = symbol
    df["interval"] = interval

    df = df[(df["open_time"] >= start_dt) & (df["open_time"] <= end_dt)]
    df = df.sort_values("open_time")
    df = df.drop_duplicates(subset=["open_time"], keep="first")
    return df.set_index("open_time", drop=False)


class _RetryableError(RuntimeError):
    pass


def _request_klines(
    session: requests.Session,
    url: str,
    params: dict,
    max_retries: int,
    sleep_seconds: float,
) -> List[list]:
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise RuntimeError("Invalid JSON response") from exc
            if resp.status_code in RETRYABLE_STATUSES:
                raise _RetryableError(f"HTTP {resp.status_code}: {resp.text}")
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
        except (requests.RequestException, _RetryableError) as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Request failed after {max_retries} attempts: {exc}") from exc
            time.sleep(max(sleep_seconds, 0.1) * attempt)
    return []


def fetch_klines_range(
    symbol: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
    base_url: str,
    limit: int,
    sleep: float,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Fetch klines for a time range using paginated requests."""
    url = f"{base_url.rstrip('/')}/api/v3/klines"
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    owns_session = session is None
    if session is None:
        session = requests.Session()

    rows: List[list] = []
    try:
        while start_ms <= end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": limit,
            }
            data = _request_klines(session, url, params, max_retries=5, sleep_seconds=sleep)
            if not data:
                break

            rows.extend(data)
            last_close = data[-1][6]
            next_start = int(last_close) + 1
            if next_start <= start_ms:
                raise RuntimeError("Pagination stalled: non-advancing startTime")
            start_ms = next_start
            if sleep > 0:
                time.sleep(sleep)
    finally:
        if owns_session:
            session.close()

    return _klines_to_df(rows, symbol, interval, start_dt, end_dt)


_API_SETTINGS: Optional[dict] = None


def _get_api_settings() -> dict:
    global _API_SETTINGS
    if _API_SETTINGS is None:
        cfg = load_config()
        _API_SETTINGS = cfg.get("binance_api", {})
    return _API_SETTINGS


def download_klines_by_year(
    symbol: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
    overwrite: bool = False,
) -> List[Path]:
    """Download klines by calendar year and save to parquet files."""
    api_cfg = _get_api_settings()
    base_url = api_cfg.get("base_url")
    limit = int(api_cfg.get("klines_limit", 1000))
    sleep = float(api_cfg.get("rate_limit_sleep_sec", 0.2))

    if not base_url:
        raise ValueError("binance_api.base_url is required")

    out_dir = RAW_DATA_DIR / "binance_spot" / symbol / interval
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    with requests.Session() as session:
        for year, year_start, year_end in _year_slices(start_dt, end_dt):
            out_path = out_dir / f"{year}.parquet"
            if out_path.exists() and not overwrite:
                continue

            df = fetch_klines_range(
                symbol=symbol,
                interval=interval,
                start_dt=year_start,
                end_dt=year_end,
                base_url=base_url,
                limit=limit,
                sleep=sleep,
                session=session,
            )

            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            df.to_parquet(tmp_path, index=True)
            tmp_path.replace(out_path)
            saved.append(out_path)

    return saved


def main() -> None:
    cfg = load_config()

    venue = cfg.get("venue", {})
    if venue.get("exchange") != "binance" or venue.get("market_type") != "spot":
        raise ValueError("Only Binance spot is supported by this downloader")

    symbol = venue.get("symbol")
    if not symbol:
        raise ValueError("venue.symbol is required")

    time_range = cfg.get("time_range", {})
    timezone_name = time_range.get("timezone")
    if timezone_name != "UTC":
        raise ValueError("time_range.timezone must be 'UTC'")

    start_dt = _parse_datetime(time_range.get("start"), is_end=False)
    end_dt = _parse_datetime(time_range.get("end"), is_end=True)

    freqs = cfg.get("frequencies") or []
    if not freqs:
        raise ValueError("frequencies must include at least one interval")

    for interval in freqs:
        print(f"Downloading {symbol} {interval}...")
        saved = download_klines_by_year(symbol, interval, start_dt, end_dt, overwrite=False)
        for path in saved:
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
