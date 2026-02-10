from __future__ import annotations

from typing import Dict


DEFAULT_ALLOCATIONS: Dict[str, Dict[str, float]] = {
    "trend_up": {
        "donchian": 0.50,
        "ema_adx": 0.40,
        "mean_reversion": 0.10,
    },
    "trend_down": {
        "donchian": 0.45,
        "ema_adx": 0.45,
        "mean_reversion": 0.10,
    },
    "range": {
        "donchian": 0.10,
        "ema_adx": 0.20,
        "mean_reversion": 0.70,
    },
}


def weights_for_regime(regime: str) -> Dict[str, float]:
    """Return strategy allocation by regime label."""
    if regime not in DEFAULT_ALLOCATIONS:
        raise KeyError(f"Unknown regime: {regime}")
    return DEFAULT_ALLOCATIONS[regime]
