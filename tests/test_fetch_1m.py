from __future__ import annotations

import pandas as pd

from bt_regime_system.data import fetch_1m as mod


def _kline(open_ms: int, o: str, h: str, l: str, c: str, v: str) -> list:
    return [open_ms, o, h, l, c, v, open_ms + 59_999, "0", 1, "0", "0", "0"]


def test_rows_to_frame_uses_close_time_timestamp() -> None:
    rows = [
        _kline(1_704_067_200_000, "100", "101", "99", "100.5", "12"),
        _kline(1_704_067_260_000, "100.5", "102", "100", "101", "10"),
    ]

    out = mod._rows_to_frame(rows)

    assert list(out.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert out.loc[0, "timestamp"] == pd.Timestamp("2024-01-01 00:01:00+00:00")
    assert out.loc[1, "timestamp"] == pd.Timestamp("2024-01-01 00:02:00+00:00")


def test_fetch_1m_klines_paginates_and_filters(monkeypatch) -> None:
    start = pd.Timestamp("2024-01-01 00:01:00+00:00")
    end = pd.Timestamp("2024-01-01 00:03:00+00:00")
    api_start = int((start - pd.Timedelta(minutes=1)).timestamp() * 1000)

    page1 = [
        _kline(api_start, "100", "101", "99", "100.5", "10"),
        _kline(api_start + 60_000, "100.5", "102", "100", "101.5", "11"),
    ]
    page2 = [
        _kline(api_start + 120_000, "101.5", "103", "101", "102.5", "12"),
    ]

    def fake_request(session, base_url, params, max_retries=3, timeout_sec=30):
        start_ms = int(params["startTime"])
        if start_ms == api_start:
            return page1
        if start_ms == api_start + 120_000:
            return page2
        return []

    monkeypatch.setattr(mod, "_request_klines", fake_request)

    out = mod.fetch_1m_klines("BTCUSDT", start=start, end=end, pause_sec=0.0)

    assert len(out) == 3
    assert out["timestamp"].min() == start
    assert out["timestamp"].max() == end


def test_write_monthly_raw_1m_merges_on_timestamp(tmp_path) -> None:
    jan = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01 00:01:00+00:00", "2024-01-01 00:02:00+00:00"], utc=True
            ),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [10.0, 11.0],
        }
    )
    mod.write_monthly_raw_1m(jan, tmp_path, "BTCUSDT")

    jan_update = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01 00:02:00+00:00", "2024-01-01 00:03:00+00:00"], utc=True
            ),
            "open": [999.0, 102.0],
            "high": [999.0, 103.0],
            "low": [999.0, 101.0],
            "close": [999.0, 102.5],
            "volume": [99.0, 12.0],
        }
    )
    written = mod.write_monthly_raw_1m(jan_update, tmp_path, "BTCUSDT")

    assert len(written) == 1
    merged = pd.read_parquet(written[0]).sort_values("timestamp").reset_index(drop=True)
    assert len(merged) == 3
    assert merged.loc[1, "open"] == 999.0
