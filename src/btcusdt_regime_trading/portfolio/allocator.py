"""Portfolio allocator combining strategy signals with router weights."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AllocatorConfig:
    """Allocator configuration."""

    name: str = "default_allocator"
    max_leverage: float = 1.0
    normalize: bool = True
    description: str = "Baseline signal-weight allocator"


class PortfolioAllocator:
    """Combine strategy-level signals into a portfolio-level target position.

    Allocation uses elementwise signal * weight, then aggregates across
    strategies. Optional normalization scales by sum(|signal * weight|).
    """

    def __init__(self, config: AllocatorConfig) -> None:
        self.config = config

    def allocate(self, signals: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
        """Allocate portfolio position from strategy signals and weights."""
        if not isinstance(signals.index, pd.DatetimeIndex) or not isinstance(weights.index, pd.DatetimeIndex):
            raise ValueError("signals and weights must use DatetimeIndex")
        if not signals.index.equals(weights.index):
            raise ValueError("signals and weights index must match")
        if list(signals.columns) != list(weights.columns):
            raise ValueError("signals and weights columns must match and be in the same order")

        sig = signals.copy()
        wgt = weights.copy()

        if sig.isna().any().any() or wgt.isna().any().any():
            raise ValueError("signals and weights must not contain NaNs")

        weighted = sig * wgt
        raw_position = weighted.sum(axis=1)

        if self.config.normalize:
            denom = weighted.abs().sum(axis=1)
            non_zero = denom > 0
            raw_position.loc[non_zero] = raw_position.loc[non_zero] / denom.loc[non_zero]

        raw_position = raw_position.clip(-self.config.max_leverage, self.config.max_leverage)
        raw_position.name = "target_position"

        if raw_position.isna().any():
            raise ValueError("allocator produced NaN positions")

        return raw_position


def _print_summary(pos: pd.Series, max_leverage: float) -> None:
    print(f"shape: {pos.shape}")
    print("head:")
    print(pos.head(5))
    print("tail:")
    print(pos.tail(5))
    print(f"min={pos.min():.4f}, max={pos.max():.4f}, mean={pos.mean():.4f}")
    zero_count = int((pos == 0).sum())
    print(f"zero_positions: {zero_count}")
    print(f"within_bounds: {pos.min() >= -max_leverage and pos.max() <= max_leverage}")


def main() -> None:
    from btcusdt_regime_trading.data.loaders import load_klines_1h_processed
    from btcusdt_regime_trading.features.bar_features import make_bar_features_1h
    from btcusdt_regime_trading.regimes.regime_engine import compute_regime_1h
    from btcusdt_regime_trading.router.strategy_router import RoutingConfig, StrategyRouter

    df = load_klines_1h_processed()

    if "regime" not in df.columns:
        feats = make_bar_features_1h(df)
        reg = compute_regime_1h(feats)
        df = df.join(reg[["regime"]], how="left")

    strategy_names = [
        "ema_adx_trend",
        "donchian_breakout",
        "kalman_mean_reversion",
        "transformer_encoder",
    ]

    signals = pd.DataFrame(
        {
            "ema_adx_trend": 1.0,
            "donchian_breakout": 0.5,
            "kalman_mean_reversion": -0.3,
            "transformer_encoder": 0.2,
        },
        index=df.index,
    )

    router = StrategyRouter(RoutingConfig(mode="hard", normalize_weights=False), strategy_names)
    weights = router.route(df)

    allocator = PortfolioAllocator(AllocatorConfig(max_leverage=1.0, normalize=True))
    pos = allocator.allocate(signals, weights)
    _print_summary(pos, allocator.config.max_leverage)


if __name__ == "__main__":
    main()
