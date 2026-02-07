"""Strategy router for regime-aware multi-strategy allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

import pandas as pd


@dataclass(frozen=True)
class RoutingConfig:
    """Routing configuration for regime-to-strategy mapping."""

    name: str = "default_router"
    mode: Literal["hard", "soft"] = "hard"
    normalize_weights: bool = True
    description: str = "Baseline regime-to-strategy mapping"


class StrategyRouter:
    """Route market regimes to strategy allocation weights.

    This router decides which strategies are active under which regime and
    produces per-bar weights for each strategy.
    """

    _mapping: Dict[str, Dict[str, float]] = {
        "Up_LowVol": {
            "ema_adx_trend": 1.0,
            "donchian_breakout": 0.5,
            "kalman_mean_reversion": 0.3,
            "transformer_encoder": 0.5,
        },
        "Up_HighVol": {
            "ema_adx_trend": 0.5,
            "donchian_breakout": 1.0,
            "kalman_mean_reversion": 0.0,
            "transformer_encoder": 0.5,
        },
        "Down_LowVol": {
            "ema_adx_trend": 1.0,
            "donchian_breakout": 0.5,
            "kalman_mean_reversion": 0.3,
            "transformer_encoder": 0.5,
        },
        "Down_HighVol": {
            "ema_adx_trend": 0.5,
            "donchian_breakout": 1.0,
            "kalman_mean_reversion": 0.0,
            "transformer_encoder": 0.5,
        },
    }

    def __init__(self, config: RoutingConfig, strategy_names: List[str]) -> None:
        self.config = config
        self.strategy_names = list(strategy_names)
        self._validate_mapping()

    def _validate_mapping(self) -> None:
        mapping_strats = set()
        for weights in self._mapping.values():
            mapping_strats.update(weights.keys())
        unknown = mapping_strats - set(self.strategy_names)
        if unknown:
            raise ValueError(f"Mapping contains unknown strategies: {sorted(unknown)}")

    def route(self, df: pd.DataFrame) -> pd.DataFrame:
        """Route regimes to strategy weights.

        Returns a DataFrame with index == df.index and columns == strategy_names.
        """
        if "regime" not in df.columns:
            raise ValueError("df must contain a 'regime' column")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df.index must be a DatetimeIndex")

        regimes = df["regime"]
        unknown_regimes = sorted(set(regimes.dropna()) - set(self._mapping.keys()))
        if unknown_regimes:
            raise ValueError(f"Unknown regime values: {unknown_regimes}")

        map_table = pd.DataFrame.from_dict(self._mapping, orient="index")
        map_table = map_table.reindex(columns=self.strategy_names)

        # Initialize with zeros so missing regimes map to zero weights.
        weights = pd.DataFrame(0.0, index=df.index, columns=self.strategy_names)

        valid_mask = regimes.notna()
        if valid_mask.any():
            mapped = map_table.loc[regimes[valid_mask]].copy()
            mapped.index = df.index[valid_mask]
            weights.loc[valid_mask] = mapped.values

        if self.config.mode == "soft" and self.config.normalize_weights:
            row_sum = weights.sum(axis=1)
            non_zero = row_sum > 0
            weights.loc[non_zero] = weights.loc[non_zero].div(row_sum[non_zero], axis=0)

        if weights.isna().any().any():
            raise ValueError("Routing produced NaN weights")
        if not weights.index.equals(df.index):
            raise ValueError("Output index does not match input index")

        return weights


def _print_stats(weights: pd.DataFrame) -> None:
    row_sum = weights.sum(axis=1)
    print(f"shape: {weights.shape}")
    print("head:")
    print(weights.head(5))
    print("tail:")
    print(weights.tail(5))
    print("row_sum stats:")
    print(f"  min={row_sum.min():.4f}, max={row_sum.max():.4f}, mean={row_sum.mean():.4f}")
    zero_rows = int((row_sum == 0).sum())
    print(f"zero_weight_rows: {zero_rows}")


def main() -> None:
    from btcusdt_regime_trading.data.loaders import load_klines_1h_processed
    from btcusdt_regime_trading.features.bar_features import make_bar_features_1h
    from btcusdt_regime_trading.regimes.regime_engine import compute_regime_1h

    df = load_klines_1h_processed()
    features = make_bar_features_1h(df)
    regimes = compute_regime_1h(features)
    df = df.copy()
    df["regime"] = regimes["regime"]

    strategy_names = [
        "ema_adx_trend",
        "donchian_breakout",
        "kalman_mean_reversion",
        "transformer_encoder",
    ]

    router_hard = StrategyRouter(RoutingConfig(mode="hard"), strategy_names)
    weights_hard = router_hard.route(df)
    print("HARD routing")
    _print_stats(weights_hard)

    router_soft = StrategyRouter(RoutingConfig(mode="soft", normalize_weights=True), strategy_names)
    weights_soft = router_soft.route(df)
    print("SOFT routing")
    _print_stats(weights_soft)


if __name__ == "__main__":
    main()
