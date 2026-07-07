"""
plotting.py — publication-grade matplotlib style shared across EDA steps.

Every EDA step imports `apply_style` and `save_figure` so the resulting
figure panel (write-up, step 7) is stylistically uniform without repeating
rcParams tweaks in every script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

_STYLE_APPLIED = False


def apply_style() -> None:
    """Set matplotlib rcParams for print-quality figures.

    Idempotent — safe to call from multiple steps in one process.
    """
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    _STYLE_APPLIED = True


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    """Save a figure to disk, creating the parent dir if needed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
