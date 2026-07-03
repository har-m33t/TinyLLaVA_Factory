"""Matplotlib/seaborn plot generation.

Every plot function returns the on-disk :class:`Path` it wrote. Callers
collect those into the audit log and hand them to the LLM interpretation
step. Matplotlib figures are always closed before the function returns —
long EDA runs otherwise leak memory on the login node.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


LOG = logging.getLogger(__name__)


def _mpl():
    """Import matplotlib lazily so ``import cvd_eda.eda`` stays cheap."""
    import matplotlib

    matplotlib.use("Agg")  # headless — Task 6 runs on the HPC
    import matplotlib.pyplot as plt

    return matplotlib, plt


def _save(fig, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    _, plt = _mpl()
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Cohort composition
# --------------------------------------------------------------------------- #


def plot_label_bar(per_label: Dict[str, int], out: Path, dpi: int) -> Path:
    _, plt = _mpl()
    labels = list(per_label.keys())
    counts = [per_label[label] for label in labels]
    fig, ax = plt.subplots(figsize=(max(4, 0.4 * len(labels) + 3), 4))
    ax.bar(labels, counts, color="steelblue")
    ax.set_ylabel("samples")
    ax.set_title("Reviewed label counts")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, out, dpi)


def plot_per_series_bar(per_series: Dict[str, int], out: Path, dpi: int) -> Path:
    _, plt = _mpl()
    series = list(per_series.keys())
    counts = [per_series[s] for s in series]
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(series) + 3), 4))
    ax.bar(series, counts, color="darkorange")
    ax.set_ylabel("samples")
    ax.set_title(f"Samples per series (top {len(series)})")
    ax.tick_params(axis="x", rotation=75)
    return _save(fig, out, dpi)


# --------------------------------------------------------------------------- #
# Per-sample QC
# --------------------------------------------------------------------------- #


def plot_library_size_hist(per_sample: pd.DataFrame, out: Path, dpi: int) -> Path:
    _, plt = _mpl()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(per_sample["library_size"].dropna(), bins=40, color="steelblue")
    ax.set_xlabel("per-sample summed expression (post-normalization)")
    ax.set_ylabel("samples")
    ax.set_title("Library size distribution")
    return _save(fig, out, dpi)


def plot_genes_detected_hist(per_sample: pd.DataFrame, out: Path, dpi: int) -> Path:
    _, plt = _mpl()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(per_sample["genes_detected"].dropna(), bins=40, color="seagreen")
    ax.set_xlabel("genes detected (>0)")
    ax.set_ylabel("samples")
    ax.set_title("Genes detected per sample")
    return _save(fig, out, dpi)


def plot_biotype_share_box(biotype_share: pd.DataFrame, out: Path, dpi: int) -> Path:
    _, plt = _mpl()
    top = biotype_share.mean().sort_values(ascending=False).head(10).index
    data = [biotype_share[bt].dropna() for bt in top]
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(top) + 3), 4))
    ax.boxplot(data, labels=list(top))
    ax.set_ylabel("share of total expression")
    ax.set_title("Biotype composition (top 10)")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, out, dpi)


# --------------------------------------------------------------------------- #
# Sample relationships
# --------------------------------------------------------------------------- #


def plot_pca_scatter(
    scores: pd.DataFrame,
    color_by: pd.Series,
    explained_variance_ratio,
    out: Path,
    dpi: int,
    *,
    color_name: str,
) -> Path:
    _, plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 5))
    categories = color_by.astype(str).fillna("(missing)")
    unique = list(pd.Index(categories.unique()))
    cmap = plt.get_cmap("tab20" if len(unique) > 10 else "tab10")
    for i, cat in enumerate(unique):
        mask = (categories == cat).values
        ax.scatter(
            scores.loc[mask, scores.columns[0]],
            scores.loc[mask, scores.columns[1]],
            s=18,
            alpha=0.8,
            label=cat if len(unique) <= 20 else None,
            color=cmap(i % cmap.N),
        )
    v1 = explained_variance_ratio[0] * 100 if explained_variance_ratio else 0.0
    v2 = explained_variance_ratio[1] * 100 if len(explained_variance_ratio) > 1 else 0.0
    ax.set_xlabel(f"PC1 ({v1:.1f}%)")
    ax.set_ylabel(f"PC2 ({v2:.1f}%)")
    ax.set_title(f"PCA — colored by {color_name}")
    if len(unique) <= 20:
        ax.legend(loc="best", fontsize=8, frameon=True)
    return _save(fig, out, dpi)


def plot_tsne_scatter(
    tsne: pd.DataFrame,
    color_by: pd.Series,
    out: Path,
    dpi: int,
    *,
    color_name: str,
) -> Path:
    _, plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 5))
    categories = color_by.astype(str).fillna("(missing)")
    unique = list(pd.Index(categories.unique()))
    cmap = plt.get_cmap("tab20" if len(unique) > 10 else "tab10")
    for i, cat in enumerate(unique):
        mask = (categories == cat).values
        ax.scatter(
            tsne.loc[mask, "tSNE1"],
            tsne.loc[mask, "tSNE2"],
            s=18,
            alpha=0.8,
            label=cat if len(unique) <= 20 else None,
            color=cmap(i % cmap.N),
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE — colored by {color_name}")
    if len(unique) <= 20:
        ax.legend(loc="best", fontsize=8, frameon=True)
    return _save(fig, out, dpi)


def plot_sample_corr_heatmap(
    corr: pd.DataFrame,
    linkage_order,
    out: Path,
    dpi: int,
    max_samples: int = 200,
) -> Path:
    _, plt = _mpl()
    order = [s for s in linkage_order if s in corr.index]
    if len(order) > max_samples:
        idx = np.linspace(0, len(order) - 1, num=max_samples).round().astype(int)
        order = [order[i] for i in idx]
        LOG.info("Downsampled correlation heatmap to %d samples for rendering.", max_samples)
    reordered = corr.loc[order, order]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(reordered.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Sample-sample correlation (n={len(order)}, hierarchical order)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    return _save(fig, out, dpi)


# --------------------------------------------------------------------------- #
# Confounder screen
# --------------------------------------------------------------------------- #


def plot_confounder_heatmap(per_pc: pd.DataFrame, out: Path, dpi: int) -> Path:
    _, plt = _mpl()
    if per_pc.empty:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "no covariates available", ha="center", va="center")
        ax.axis("off")
        return _save(fig, out, dpi)
    fig, ax = plt.subplots(
        figsize=(max(5, 0.6 * len(per_pc.columns) + 2), max(3, 0.4 * len(per_pc.index) + 2))
    )
    im = ax.imshow(per_pc.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(per_pc.columns)))
    ax.set_xticklabels(per_pc.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(per_pc.index)))
    ax.set_yticklabels(per_pc.index)
    ax.set_title("Top PCs vs covariates (eta² / r²)")
    for i in range(per_pc.shape[0]):
        for j in range(per_pc.shape[1]):
            v = per_pc.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7, label="association")
    return _save(fig, out, dpi)
