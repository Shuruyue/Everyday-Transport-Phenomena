"""Shared figure style. Figures made from synthetic data carry a watermark so
they can never be mistaken for measurements."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "font.size": 10, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})

COLORS = {"data": "#222222", "model": "#1f77b4", "alt": "#d62728", "band": "#1f77b4",
          "aux": "#2ca02c", "aux2": "#9467bd"}


def watermark(fig, synthetic):
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC DATA\npipeline test only", ha="center", va="center",
                 fontsize=26, color="red", alpha=0.18, rotation=25, weight="bold")


def save(fig, path, synthetic):
    watermark(fig, synthetic)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
