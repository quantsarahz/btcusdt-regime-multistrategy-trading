from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity(equity: pd.Series) -> plt.Figure:
    """Return an equity curve figure."""
    fig, ax = plt.subplots(figsize=(10, 4))
    equity.plot(ax=ax, lw=1.2, title="Equity Curve")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
