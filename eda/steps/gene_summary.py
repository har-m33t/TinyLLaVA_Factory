"""
gene_summary.py — Task 6: per-gene summary on the full ARCHS4 human corpus.

Outputs (under `<outdir>/gene_summary/`):
    gene_summary_full.csv
        One row per gene: gene_symbol, ensembl_gene_id (if present),
        detection_rate (fraction of samples with counts > 0), mean_count,
        gene_biotype (if the release ships it).
    gene_detection_rate_hist.png
        Histogram of detection rate across all genes.
    gene_biotype_bar.png (only written if biotype metadata is available)
        Sample counts per biotype.

Method
------
Computed by streaming gene-chunks through the H5 file (not sample-chunks:
detection rate needs per-gene aggregation over all samples, so gene-major
chunks amortise best). Because the H5 layout is sample-major, gene-chunks
are slower per byte than sample-chunks; we keep the chunk small (default
`GENE_CHUNK = 512`) to stay under memory budget.

Biotype composition is best-effort. Not all ARCHS4 releases ship a
`gene_biotype` field in `/meta/genes`; if it's absent, we skip the biotype
outputs and log a warning — this is documented in the write-up.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset import io as archs4_io
from ..plotting import apply_style, save_figure

logger = logging.getLogger(__name__)

GENE_CHUNK = 512


def run(h5_path: Path, outdir: Path) -> Path:
    apply_style()
    out = outdir / "gene_summary"
    out.mkdir(parents=True, exist_ok=True)

    with archs4_io.open_h5(h5_path) as h5:
        shape = archs4_io.get_shape(h5)
        symbols = archs4_io.gene_symbols(h5)
        ensembl = archs4_io.read_gene_field_any(h5, "ensembl_gene_id", "ensembl_gene")
        biotype = archs4_io.read_gene_field_any(h5, "gene_biotype", "biotype")

        detection = np.zeros(shape.n_genes, dtype=np.int64)
        total_counts = np.zeros(shape.n_genes, dtype=np.int64)

        for sl, chunk in archs4_io.iter_gene_chunks(h5, chunk_size=GENE_CHUNK):
            # chunk shape: (chunk_size, n_samples)
            detection[sl] = (chunk > 0).sum(axis=1)
            total_counts[sl] = chunk.sum(axis=1)

    detection_rate = detection / shape.n_samples
    mean_count = total_counts / shape.n_samples

    df = pd.DataFrame({
        "gene_symbol": symbols,
        "detection_rate": detection_rate,
        "mean_count": mean_count,
    })
    if ensembl is not None:
        df["ensembl_gene_id"] = ensembl
    if biotype is not None:
        df["gene_biotype"] = biotype
    else:
        logger.warning("gene_biotype not present in this ARCHS4 release; skipping biotype figure")

    csv_path = out / "gene_summary_full.csv"
    df.to_csv(csv_path, index=False)
    logger.info("gene summary written to %s (%d genes)", csv_path, len(df))

    _plot_detection_rate(df, out / "gene_detection_rate_hist.png")
    if biotype is not None:
        _plot_biotype(df, out / "gene_biotype_bar.png")
    return csv_path


def _plot_detection_rate(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.hist(df["detection_rate"].to_numpy(), bins=50, color="#4C78A8")
    ax.set_xlabel("fraction of samples with counts > 0")
    ax.set_ylabel("genes")
    ax.set_title("ARCHS4 per-gene detection rate")
    save_figure(fig, out_path)


def _plot_biotype(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    counts = df["gene_biotype"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.barh(counts.index[::-1], counts.values[::-1], color="#4C78A8")
    ax.set_xlabel("genes")
    ax.set_title("ARCHS4 gene biotype composition (top 15)")
    save_figure(fig, out_path)
