"""
normalize.py — Task 3: quantile normalization + log2 transform.

Replicates the ARCHS4 paper's preprocessing (Lachmann et al. 2018, Methods):
quantile normalization applied per-organism, followed by a log2 transform,
before any downstream visualization.

Scale problem and the chunked strategy
--------------------------------------
Classical quantile normalization (Bolstad et al. 2003) requires the full
`n_genes × n_samples` matrix in memory to compute the reference quantile
distribution (row-wise mean of sorted columns). At ~700k samples × ~35k
genes, that's tens of terabytes of intermediate storage — impossible on
any single workstation, and impractical to persist as a normalized matrix
even in float32 (would exceed ~100GB).

We therefore use the standard **reference-distribution** variant, which is
mathematically equivalent to full quantile normalization when the reference
subset is large enough to approximate the true rank-mean distribution:

  1. Draw a uniform random subsample of `n_ref` samples (default 10,000).
  2. Load only those columns into memory.
  3. Compute the reference quantile vector = row-wise mean of column-sorted
     counts. Persist it — this is the *only* artifact needed by downstream
     steps to normalize any new sample.
  4. Materialize a normalized+log2 subsample matrix on a *separate* random
     subsample of `n_downstream` samples (default 20,000) for use by t-SNE
     (step 4) and the correlation heatmap (step 5). Persist as float32 npy.

This matches the approach used at TCGA/GTEx scale (e.g. tximport reference
distribution) and is explicitly documented as a limitation in the write-up
(step 7).

Outputs (under `<outdir>/normalized/`):
    reference_distribution.npy
        Float64 vector of length n_genes — the reference quantile distribution.
    subsample_indices.npy
        Int vector of the sample indices used to build the downstream matrix.
    subsample_matrix.npy
        Float32 matrix, shape (n_genes, n_downstream) — quantile-normalized
        and log2-transformed. Feeds steps 4, 5.
    normalize_manifest.json
        Records n_ref, n_downstream, seed, timestamps.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..dataset import io as archs4_io

logger = logging.getLogger(__name__)

DEFAULT_N_REFERENCE = 10_000
DEFAULT_N_DOWNSTREAM = 5_000
DEFAULT_SEED = 20260705  # date the pipeline was fixed; documented in write-up
LOG2_PSEUDOCOUNT = 1.0


def compute_reference_distribution(
    h5_path: Path, n_ref: int, seed: int, pool: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (reference_vector, reference_indices).

    reference_vector has shape (n_genes,); its i-th entry is the mean value
    at rank i across the `n_ref` reference samples. The reference is drawn
    from `pool` — the singlecell-filtered set of eligible sample indices
    computed once by `filter_bulk_indices`.
    """
    with archs4_io.open_h5(h5_path) as h5:
        ref_idx = archs4_io.subsample_from_pool(pool, n_ref, seed=seed)
        logger.info("loading %d reference samples (of %d in bulk pool) for quantile reference distribution",
                    len(ref_idx), len(pool))
        ref_counts = archs4_io.read_samples_by_index(h5, ref_idx).astype(np.float64)

    # Sort each column ascending (in-place), then row-mean.
    ref_counts.sort(axis=0)
    reference = ref_counts.mean(axis=1)
    return reference, ref_idx


def apply_quantile_norm(
    counts: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Map each sample-column's counts onto the reference quantile distribution.

    Ties are broken by average-rank (the Bolstad convention), matching R's
    `preprocessCore::normalize.quantiles`. Returns a float64 matrix of the
    same shape as `counts`.

    Parameters
    ----------
    counts : (n_genes, n_samples) raw counts
    reference : (n_genes,) reference quantile vector
    """
    n_genes, n_samples = counts.shape
    if reference.shape[0] != n_genes:
        raise ValueError(f"reference length {reference.shape[0]} != n_genes {n_genes}")
    normalized = np.empty_like(counts, dtype=np.float64)
    for j in range(n_samples):
        col = counts[:, j]
        order = np.argsort(col, kind="stable")
        # Bolstad tie handling: within any run of equal input values, replace
        # the reference values assigned to that run with the run's mean.
        sorted_col = col[order]
        sorted_ref = reference.copy()
        change_points = np.flatnonzero(np.diff(sorted_col) != 0) + 1
        starts = np.concatenate(([0], change_points))
        ends = np.concatenate((change_points, [n_genes]))
        for s, e in zip(starts, ends):
            if e - s > 1:
                sorted_ref[s:e] = sorted_ref[s:e].mean()
        normalized[order, j] = sorted_ref
    return normalized


def log2_transform(mat: np.ndarray, pseudocount: float = LOG2_PSEUDOCOUNT) -> np.ndarray:
    return np.log2(mat + pseudocount)


def run(
    h5_path: Path,
    outdir: Path,
    n_ref: int = DEFAULT_N_REFERENCE,
    n_downstream: int = DEFAULT_N_DOWNSTREAM,
    seed: int = DEFAULT_SEED,
) -> Path:
    out = outdir / "normalized"
    out.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()

    # Compute the singlecell-filtered pool once, upstream of every random
    # subsampling step in the pipeline. Reference and downstream draws below,
    # and dimred's stability draw + clustering's heatmap draw downstream, all
    # sample from this pool (or subsets of it), so the sc-prob > threshold
    # exclusion propagates automatically.
    with archs4_io.open_h5(h5_path) as h5:
        pool, filter_stats = archs4_io.filter_bulk_indices(h5)
    logger.info(
        "single-cell filter: kept %d / %d samples (excluded %d = %.3f%%; threshold %.2f)",
        filter_stats["kept"], filter_stats["total"],
        filter_stats["excluded"], filter_stats["excluded_pct"], filter_stats["threshold"],
    )

    reference, ref_idx = compute_reference_distribution(
        h5_path, n_ref=n_ref, seed=seed, pool=pool
    )
    np.save(out / "reference_distribution.npy", reference)
    logger.info("reference distribution saved to %s", out / "reference_distribution.npy")

    # Downstream subsample: a *fresh* draw from the same pool, independent of
    # the reference draw. Same seed source but a different offset so they're
    # reproducible without being identical.
    with archs4_io.open_h5(h5_path) as h5:
        ds_idx = archs4_io.subsample_from_pool(pool, n_downstream, seed=seed + 1)
        logger.info("materialising %d downstream samples for quantile norm + log2", len(ds_idx))
        ds_counts = archs4_io.read_samples_by_index(h5, ds_idx).astype(np.float64)

    normalized = apply_quantile_norm(ds_counts, reference)
    logged = log2_transform(normalized).astype(np.float32)

    np.save(out / "subsample_indices.npy", ds_idx)
    np.save(out / "subsample_matrix.npy", logged)

    manifest = {
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "n_reference_samples": int(n_ref),
        "n_downstream_samples": int(n_downstream),
        "seed": int(seed),
        "log2_pseudocount": LOG2_PSEUDOCOUNT,
        "singlecell_filter": filter_stats,
        "note": (
            "Reference-distribution quantile norm (Bolstad-style tie handling), "
            "computed on a random uniform subsample of the singlecell-filtered "
            "pool. The full corpus is never materialized as a normalized matrix "
            "— see normalize.py docstring."
        ),
    }
    with open(out / "normalize_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("normalization manifest: %s", out / "normalize_manifest.json")
    return out
