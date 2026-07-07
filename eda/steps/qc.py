"""
qc.py — Task 2: per-sample QC on the full ARCHS4 human corpus.

Outputs (under `<outdir>/qc/`):
    qc_full_dataset.csv
        One row per sample: library_size, genes_detected, singlecellprobability
        (if present), outlier_flag columns.
    qc_library_size_hist.png
        log10 library-size distribution across all samples.
    qc_genes_detected_hist.png
        Genes-detected distribution across all samples.

Method:
    Two per-sample summaries — library size (sum of raw counts) and genes
    detected (count of genes with counts > 0) — computed by streaming through
    the H5 file in sample-chunks. This is the same QC pair used in the
    Bioconductor/recount3 quickstart (Love/Huber-style) as pre-normalization
    diagnostics.

Outlier flagging:
    We do NOT drop samples at this stage — pre-CVD-selection EDA must be
    non-destructive. We report:
      - `outlier_lib_size_lo`: log10(lib) < median(log10 lib) - 3*MAD
      - `outlier_lib_size_hi`: log10(lib) > median(log10 lib) + 3*MAD
      - `outlier_low_detection`: genes_detected < median - 3*MAD
    MAD-based thresholds are the recount3 default; 3*MAD is intentionally
    permissive so we surface extreme samples without pre-committing to a
    filter policy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset import io as archs4_io
from ..plotting import apply_style, save_figure

logger = logging.getLogger(__name__)

MAD_K = 3.0  # threshold multiplier; 3*MAD is the recount3 convention
CHUNK_SIZE = 2048


def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _compute_per_sample(h5_path: Path) -> pd.DataFrame:
    with archs4_io.open_h5(h5_path) as h5:
        shape = archs4_io.get_shape(h5)
        lib_size = np.zeros(shape.n_samples, dtype=np.int64)
        genes_detected = np.zeros(shape.n_samples, dtype=np.int32)

        for sl, chunk in archs4_io.iter_sample_chunks(h5, chunk_size=CHUNK_SIZE):
            # chunk shape: (n_genes, chunk_size)
            lib_size[sl] = chunk.sum(axis=0)
            genes_detected[sl] = (chunk > 0).sum(axis=0)

        gsm = archs4_io.read_sample_field(h5, "geo_accession")
        sc_prob = archs4_io.read_sample_field(h5, "singlecellprobability")

    df = pd.DataFrame({
        "geo_accession": gsm if gsm is not None else np.arange(shape.n_samples).astype(str),
        "library_size": lib_size,
        "genes_detected": genes_detected,
    })
    if sc_prob is not None:
        df["singlecellprobability"] = np.asarray(sc_prob, dtype=float)
    return df


def _flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Log-transform lib size before MAD to symmetrize the heavy right tail.
    log_lib = np.log10(np.maximum(df["library_size"].to_numpy(), 1))
    med_lib, mad_lib = np.median(log_lib), _mad(log_lib)
    if mad_lib == 0:
        mad_lib = 1e-9  # degenerate corpus guard; documented in write-up
    df["outlier_lib_size_lo"] = log_lib < med_lib - MAD_K * 1.4826 * mad_lib
    df["outlier_lib_size_hi"] = log_lib > med_lib + MAD_K * 1.4826 * mad_lib

    det = df["genes_detected"].to_numpy().astype(float)
    med_det, mad_det = np.median(det), _mad(det)
    if mad_det == 0:
        mad_det = 1e-9
    df["outlier_low_detection"] = det < med_det - MAD_K * 1.4826 * mad_det
    return df


def run(h5_path: Path, outdir: Path) -> Path:
    apply_style()
    out = outdir / "qc"
    out.mkdir(parents=True, exist_ok=True)

    df = _compute_per_sample(h5_path)
    df = _flag_outliers(df)

    csv_path = out / "qc_full_dataset.csv"
    df.to_csv(csv_path, index=False)
    logger.info("per-sample QC written to %s (%d samples)", csv_path, len(df))

    n_lo = int(df["outlier_lib_size_lo"].sum())
    n_hi = int(df["outlier_lib_size_hi"].sum())
    n_low_det = int(df["outlier_low_detection"].sum())
    logger.info(
        "outlier summary (reported only, NOT dropped): lib_lo=%d, lib_hi=%d, low_detection=%d",
        n_lo, n_hi, n_low_det,
    )

    _plot_lib_size(df, out / "qc_library_size_hist.png")
    _plot_genes_detected(df, out / "qc_genes_detected_hist.png")
    return csv_path


def _plot_lib_size(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    log_lib = np.log10(np.maximum(df["library_size"].to_numpy(), 1))
    ax.hist(log_lib, bins=80, color="#4C78A8")
    ax.set_xlabel("log10(library size)")
    ax.set_ylabel("samples")
    ax.set_title("ARCHS4 library size distribution")
    save_figure(fig, out_path)


def _plot_genes_detected(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.hist(df["genes_detected"].to_numpy(), bins=80, color="#4C78A8")
    ax.set_xlabel("genes detected (counts > 0)")
    ax.set_ylabel("samples")
    ax.set_title("ARCHS4 genes-detected distribution")
    save_figure(fig, out_path)
