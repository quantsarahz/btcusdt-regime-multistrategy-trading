"""Strategy base interface and utilities.

No-lookahead rule: at time t, strategies may only use information up to and
including t. The backtest will execute orders at the next bar open.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    """Strategy metadata and capabilities."""

    name: str
    version: str = "0.1"
    bar_freq: str = "1h"
    supports_short: bool = True
    uses_regime: bool = False
    description: str = ""


class BaseStrategy(ABC):
    """Abstract strategy interface.

    Strategies must return a position series aligned with df.index. Positions
    represent target exposure in [-1, 1] by default. Execution occurs at the
    next bar open.
    """

    spec: StrategySpec

    def __init__(self, spec: StrategySpec) -> None:
        self.spec = spec

    def fit(self, df: pd.DataFrame) -> "BaseStrategy":
        """Optional fit step for ML strategies (default no-op)."""
        return self

    @abstractmethod
    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        """Generate target positions aligned to df.index without look-ahead."""
        raise NotImplementedError

    def validate_input(self, df: pd.DataFrame, required_cols: List[str]) -> None:
        """Validate input DataFrame schema and index."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df.index must be a DatetimeIndex")
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"df missing required columns: {missing}")

    def clip_positions(self, pos: pd.Series, low: float = -1.0, high: float = 1.0) -> pd.Series:
        """Clip positions to [low, high] and ensure float dtype and alignment."""
        if not isinstance(pos, pd.Series):
            raise ValueError("pos must be a pandas Series")
        if not isinstance(pos.index, pd.DatetimeIndex):
            raise ValueError("pos.index must be a DatetimeIndex")
        pos = pos.astype(float)
        pos = pos.clip(lower=low, upper=high)
        return pos


class BuyAndHoldStrategy(BaseStrategy):
    """Simple buy-and-hold strategy for sanity checks."""

    def __init__(self) -> None:
        super().__init__(StrategySpec(name="buy_and_hold"))

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        self.validate_input(df, required_cols=["close"])
        pos = pd.Series(1.0, index=df.index, name="position_buy_and_hold")
        return self.clip_positions(pos)


def _print_summary(pos: pd.Series) -> None:
    print(f"shape: {pos.shape}")
    print("head:")
    print(pos.head(5))
    print("tail:")
    print(pos.tail(5))
    print("unique_values:")
    print(sorted(pos.dropna().unique()))
    print(f"min: {pos.min()}, max: {pos.max()}")


def main() -> None:
    from btcusdt_regime_trading.data.loaders import load_klines_1h_processed

    df = load_klines_1h_processed()
    strat = BuyAndHoldStrategy()
    pos = strat.generate_positions(df)
    _print_summary(pos)


if __name__ == "__main__":
    main()
