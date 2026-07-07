"""
clustering.py — Task 5: sample-sample correlation heatmap + hierarchical clustering.

Method
------
Correlation heatmap over the full ARCHS4 corpus would be an
`n_samples × n_samples` dense matrix — at ~700k samples that's a >5TB
float32 matrix, and is neither computable nor useful. As permitted by the
TODO, we compute the correlation matrix on a documented subsample; the
subsample is the same one produced by step 3 (normalize), so this step's
figure is directly comparable with the sample-centric t-SNE (step 4).

Even the subsample matrix (default 20k × 20k) is 1.6GB float32 in memory —
tractable but heavy — so we downsample once more to `HEATMAP_N` (default
2000) samples for the plotted heatmap itself. The hierarchical clustering
(linkage) is computed on the same 2000 × 2000 matrix.

Cross-check with step 4
-----------------------
The `clustering.csv` output carries the leaf order from the hierarchical
clustering; downstream, this can be joined against the t-SNE coordinates
(step 4) on `sample_idx` to check that samples close in linkage are also
close in t-SNE. The write-up (step 7) records the outcome of that check.

Outputs (under `<outdir>/clustering/`):
    sample_correlation_heatmap.png
    linkage.npy                     scipy linkage matrix
    heatmap_sample_indices.npy      indices of the samples that were plotted
    clustering.csv                  per-sample leaf order (for cross-check)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from ..plotting import apply_style, save_figure

logger = logging.getLogger(__name__)

# Named constant, not a bare literal: this is the sub-sub-sample size drawn
# *nested* from step 3's N=20 000 downstream subsample (see run() below). Do
# not raise this above ~5 000 without checking that the resulting corr matrix
# still fits both memory and legibility budgets.
HEATMAP_N = 2000
HEATMAP_SEED = 20260705
LINKAGE_METHOD = "average"  # UPGMA — standard for correlation-based clustering


def _pearson_corr_samples(mat: np.ndarray) -> np.ndarray:
    """Return the (n_samples, n_samples) Pearson correlation matrix.

    `mat` is expected shape (n_genes, n_samples), so correlation is between
    columns. Uses `np.corrcoef` on the transposed sample-major matrix.
    """
    # np.corrcoef treats rows as variables; give it samples-as-rows.
    return np.corrcoef(mat.T)


def run(outdir: Path) -> Path:
    apply_style()
    out = outdir / "clustering"
    out.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()
    normalized_dir = outdir / "normalized"
    mat_path = normalized_dir / "subsample_matrix.npy"
    idx_path = normalized_dir / "subsample_indices.npy"
    if not mat_path.exists():
        raise FileNotFoundError(f"expected {mat_path}; run step 3 (normalize) first")

    mat = np.load(mat_path)  # (n_genes, n_downstream)
    ds_idx = np.load(idx_path)

    n_downstream = mat.shape[1]
    # Nested draw: `pick` indexes into columns of the step-3 subsample matrix
    # (positions 0..n_downstream-1), so the heatmap samples are guaranteed to
    # be a subset of the N=20 000 t-SNE pool used by dimred.py — not an
    # independent draw from the full corpus. This gives the heatmap and
    # sample-centric t-SNE a common set of sample IDs for cross-figure joins.
    if HEATMAP_N < n_downstream:
        rng = np.random.default_rng(HEATMAP_SEED)
        pick = np.sort(rng.choice(n_downstream, size=HEATMAP_N, replace=False))
    else:
        pick = np.arange(n_downstream)
    logger.info(
        "computing %d x %d correlation matrix (nested draw from step-3 N=%d pool)",
        len(pick), len(pick), n_downstream,
    )
    sub_mat = mat[:, pick]
    corr = _pearson_corr_samples(sub_mat)

    # Convert correlation to a valid distance matrix. `1 - r` isn't a proper
    # metric but is the standard bulk-RNA-seq convention (Bioconductor
    # workflows). `squareform` requires a symmetric zero-diagonal matrix.
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # numerical symmetry guard
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=LINKAGE_METHOD)
    order = leaves_list(Z)

    np.save(out / "linkage.npy", Z)
    np.save(out / "heatmap_sample_indices.npy", ds_idx[pick])

    pd.DataFrame({
        "sample_idx": ds_idx[pick],
        "leaf_order": np.argsort(order),
    }).to_csv(out / "clustering.csv", index=False)

    _plot_heatmap(corr[np.ix_(order, order)], out / "sample_correlation_heatmap.png")

    manifest = {
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "n_downstream_input": int(n_downstream),
        "heatmap_n": int(len(pick)),
        "heatmap_seed": int(HEATMAP_SEED),
        "linkage_method": LINKAGE_METHOD,
        "distance": "1 - Pearson r (symmetrized)",
        "subsample_selection": (
            "uniform random without replacement, NESTED within the step-3 "
            "N={} subsample (i.e. drawn from the same sample-ID pool that "
            "dimred.py used for the primary sample-centric t-SNE)".format(n_downstream)
        ),
        "note": (
            "Full-corpus sample-sample correlation would be an "
            "n_samples x n_samples matrix (>5TB float32 at ~700k samples); "
            "we sub-sub-sample from the step-3 subsample to make the heatmap "
            "storable and legible. The nested draw guarantees cross-figure "
            "consistency with dimred.py's primary sample-centric t-SNE."
        ),
    }
    with open(out / "clustering_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("clustering manifest: %s", out / "clustering_manifest.json")
    return out


def _plot_heatmap(corr_ordered: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(corr_ordered, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"sample-sample Pearson correlation\n(n={corr_ordered.shape[0]}, leaves reordered by {LINKAGE_METHOD} linkage)"
    )
    fig.colorbar(im, ax=ax, label="Pearson r")
    save_figure(fig, out_path)
