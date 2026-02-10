import pandas as pd

from bt_regime_system.data.qc_1m import clean_1m_ohlcv


def test_clean_1m_removes_invalid_and_fills_missing_minutes() -> None:
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00+00:00",
            "2024-01-01 00:01:00+00:00",
            "2024-01-01 00:01:00+00:00",  # duplicate
            "2024-01-01 00:03:00+00:00",  # missing 00:02
        ]
    )
    df = pd.DataFrame(
        {
            "open": [100, 101, 102, 103],
            "high": [101, 102, 103, 102],
            "low": [99, 100, 101, 104],   # invalid last row (low > high)
            "close": [100.5, 101.5, 102.5, 101.0],
            "volume": [1.0, 2.0, 3.0, 4.0],
        },
        index=idx,
    )

    clean = clean_1m_ohlcv(df)

    assert clean.index.min() == pd.Timestamp("2024-01-01 00:00:00+00:00")
    assert clean.index.max() == pd.Timestamp("2024-01-01 00:01:00+00:00")
    assert len(clean) == 2
    assert (clean["volume"] >= 0).all()
