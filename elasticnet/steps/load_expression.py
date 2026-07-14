"""
load_expression.py — Task 3: materialise the training-pool count matrix.

Reads raw counts for *only* the sample indices in the training pool from
the ARCHS4 H5. Applies a low-count gene filter computed on this pool
(not on the whole corpus — CVD-heavy pools have different gene detection
rates than the multi-tissue whole corpus, and pre-filtering by a whole-
corpus mask would silently attenuate any CVD-specific gene signal).
Then log2(x + 1) so the matrix is on a scale sklearn is happy with.

Chosen filter (locked)
----------------------
A gene is kept if at least `min_detection_frac` of the training-pool
samples have a raw count > `count_threshold`. Defaults: detection_frac=0.10,
count_threshold=0 (i.e. "detected at all in ≥10% of the pool"). This is the
same conservative floor used in the whole-corpus EDA's gene-summary step
for downstream reporting.

Outputs
-------
X.npy          float32, shape (n_pool_samples, n_kept_genes) — sklearn-shaped
gene_symbols.npy   str,  shape (n_kept_genes,)  — the kept gene ID vector
kept_gene_mask.npy bool, shape (n_genes,)       — mask into the H5's gene axis
load_manifest.json log
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from eda.dataset import io as archs4_io

logger = logging.getLogger(__name__)


DEFAULT_MIN_DETECTION_FRAC = 0.10
DEFAULT_COUNT_THRESHOLD = 0
DEFAULT_CHUNK_SIZE = 2048  # samples per H5 read — matches EDA's stream helper
LOG2_PSEUDOCOUNT = 1.0


def _streaming_detection_counts(
    h5,
    pool_indices: np.ndarray,
    count_threshold: int,
    chunk_size: int,
) -> np.ndarray:
    """Pass 1: count how many pool samples have each gene detected.

    Streams the pool in `chunk_size`-sized batches so we never allocate
    the full (n_genes, n_pool) matrix. Returns a (n_genes,) int32 vector.
    """
    n_genes = archs4_io.get_shape(h5).n_genes
    detected_counts = np.zeros(n_genes, dtype=np.int64)
    n_pool = pool_indices.shape[0]
    for start in range(0, n_pool, chunk_size):
        stop = min(start + chunk_size, n_pool)
        chunk_idx = pool_indices[start:stop]
        chunk_counts = archs4_io.read_samples_by_index(h5, chunk_idx)
        # `> count_threshold` on int counts avoids the float32 allocation
        # we'd get from casting first.
        detected_counts += (chunk_counts > count_threshold).sum(axis=1)
        logger.info("pass 1 (detection): %d / %d pool samples processed", stop, n_pool)
    return detected_counts


def _streaming_load_and_log2(
    h5,
    pool_indices: np.ndarray,
    keep_mask: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """Pass 2: build the (n_pool, n_kept_genes) log2-transformed matrix.

    Preallocates the output at final size; each chunk is masked to
    kept-genes only, so intermediate buffers stay small.
    """
    n_pool = pool_indices.shape[0]
    n_kept = int(keep_mask.sum())
    x = np.empty((n_pool, n_kept), dtype=np.float32)
    for start in range(0, n_pool, chunk_size):
        stop = min(start + chunk_size, n_pool)
        chunk_idx = pool_indices[start:stop]
        chunk_counts = archs4_io.read_samples_by_index(h5, chunk_idx)
        chunk_kept = chunk_counts[keep_mask, :].astype(np.float32, copy=False)
        # In-place log2(count + 1): halves the peak buffer of naive log2.
        chunk_kept += LOG2_PSEUDOCOUNT
        np.log2(chunk_kept, out=chunk_kept)
        # Write in (sample, gene) orientation directly — no post-hoc transpose.
        x[start:stop, :] = chunk_kept.T
        logger.info("pass 2 (log2 load): %d / %d pool samples written", stop, n_pool)
    return x


def run(
    h5_path: Path,
    subsample_dir: Path,
    outdir: Path,
    min_detection_frac: float = DEFAULT_MIN_DETECTION_FRAC,
    count_threshold: int = DEFAULT_COUNT_THRESHOLD,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Path:
    out = outdir / "expression"
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    pool = pd.read_parquet(subsample_dir / "training_pool.parquet")
    pool_indices = pool["sample_index"].to_numpy()
    n_pool = pool_indices.shape[0]

    with archs4_io.open_h5(h5_path) as h5:
        symbols = archs4_io.gene_symbols(h5)
        detected_counts = _streaming_detection_counts(
            h5, pool_indices, count_threshold, chunk_size
        )
        detected_frac = detected_counts / max(n_pool, 1)
        keep_mask = detected_frac >= min_detection_frac
        n_kept = int(keep_mask.sum())
        logger.info(
            "low-count gene filter: kept %d / %d genes (detected in >= %.2f%% of pool at count > %d)",
            n_kept, keep_mask.size, 100.0 * min_detection_frac, count_threshold,
        )

        symbols_kept = symbols[keep_mask]
        x = _streaming_load_and_log2(h5, pool_indices, keep_mask, chunk_size)

    logger.info("materialised training-pool matrix: shape %s (%.1f GB)",
                x.shape, x.nbytes / 1e9)

    np.save(out / "X.npy", x)
    np.save(out / "gene_symbols.npy", symbols_kept)
    np.save(out / "kept_gene_mask.npy", keep_mask)

    manifest = {
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(x.shape[0]),
        "n_genes_before_filter": int(keep_mask.size),
        "n_genes_after_filter": n_kept,
        "min_detection_frac": float(min_detection_frac),
        "count_threshold": int(count_threshold),
        "log2_pseudocount": LOG2_PSEUDOCOUNT,
        "shape_X": list(x.shape),
        "chunk_size": int(chunk_size),
        "note": (
            "Low-count filter computed on the training pool, not the whole "
            "corpus — see docstring. Loaded via streaming two-pass to keep "
            "peak memory bounded by the final matrix size."
        ),
    }
    with open(out / "load_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return out
