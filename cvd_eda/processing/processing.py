"""Sample filtering, deduplication, gene filtering, normalization.

Each step is a pure function that takes/returns DataFrames plus a small
dataclass "report" so :class:`~cvd_eda.task4_processing.logging_utils.ProcessingLog`
can persist a full audit trail.

Normalization default: **CPM + log2**.
    * Deterministic and dependency-free (no R, no pydeseq2 at import time).
    * Adequate for downstream EDA — PCA, sample-sample correlation, clustering.
    * If the elastic-net stage later needs library-composition normalization
      (RNA-composition bias, high-count-gene bias), rerun this step with
      ``--norm-method deseq2``. The processing log records which method was
      used, so downstream stages can gate on it.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Step 1: CVD-relevance subset
# --------------------------------------------------------------------------- #
@dataclass
class SampleFilterReport:
    n_relevance_rows: int
    n_relevance_pass: int
    n_missing_from_counts: int
    n_output: int
    min_confidence: float


def subset_to_cvd_relevant(
    counts: pd.DataFrame,
    sample_meta: pd.DataFrame,
    relevance_df: pd.DataFrame,
    min_confidence: float,
    accepted_labels: Tuple[str, ...],
) -> Tuple[pd.DataFrame, pd.DataFrame, SampleFilterReport]:
    """Restrict to samples marked CVD-relevant with confidence ≥ threshold.

    ``relevance_df`` is Task 3's ``cvd_relevance_{dataset}.csv``. Required
    columns: ``sample_id``, ``llm_relevance``, ``confidence``. Optional
    columns (attached onto ``sample_meta`` as ``rel_*`` for provenance):
    ``matched_keyword``, ``source_series_id``.
    """
    required = {"sample_id", "llm_relevance", "confidence"}
    missing = required - set(relevance_df.columns)
    if missing:
        raise ValueError(f"Relevance CSV missing required columns: {sorted(missing)}")

    keep = relevance_df[
        relevance_df["llm_relevance"].isin(accepted_labels)
        & (relevance_df["confidence"] >= min_confidence)
    ]
    wanted_ids = set(keep["sample_id"].astype(str))
    have_ids = set(counts.columns.astype(str))
    keep_ids = wanted_ids & have_ids
    n_missing = len(wanted_ids - have_ids)
    if n_missing:
        log.warning(
            "%d CVD-relevant sample_ids in relevance CSV are absent from counts; dropped.",
            n_missing,
        )

    keep_ids_ordered = [s for s in counts.columns.astype(str) if s in keep_ids]
    counts_out = counts.loc[:, keep_ids_ordered]
    sample_meta_out = sample_meta.loc[keep_ids_ordered].copy()

    rel_indexed = keep.set_index("sample_id")
    for col in ("llm_relevance", "confidence", "matched_keyword", "source_series_id"):
        if col in rel_indexed.columns:
            sample_meta_out[f"rel_{col}"] = rel_indexed[col].reindex(sample_meta_out.index)

    report = SampleFilterReport(
        n_relevance_rows=len(relevance_df),
        n_relevance_pass=len(keep),
        n_missing_from_counts=n_missing,
        n_output=len(keep_ids_ordered),
        min_confidence=min_confidence,
    )
    return counts_out, sample_meta_out, report


# --------------------------------------------------------------------------- #
# Step 2: sample deduplication
# --------------------------------------------------------------------------- #
@dataclass
class DedupReport:
    n_input: int
    n_repeated_sample_id_removed: int
    n_identical_vector_removed: int
    n_output: int
    dropped_sample_ids: List[str] = field(default_factory=list)


def deduplicate_samples(
    counts: pd.DataFrame,
    sample_meta: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, DedupReport]:
    """Drop samples that are duplicates of another sample.

    Two failure modes we care about (both empirically observed in ARCHS4):
      1. Same GEO accession (sample_id) appearing in two columns — usually
         a reprocessing artifact. Keep the first.
      2. Different sample_ids with byte-identical count vectors — the same
         underlying FASTQ resubmitted under a new GSM. Keep the first.

    We deliberately do **not** dedup across ARCHS4↔RECOUNT3, even if the
    same GSM appears in both — the two pipelines align differently and
    surface pipeline-driven variance we may want to see in EDA.
    """
    n_input = counts.shape[1]

    # (1) collapse repeated sample_ids
    dup_id_mask = counts.columns.duplicated(keep="first")
    n_id_dup = int(dup_id_mask.sum())
    counts = counts.loc[:, ~dup_id_mask]
    sample_meta = sample_meta.loc[~sample_meta.index.duplicated(keep="first")]

    # (2) hash each column, drop later duplicates
    hashes = []
    for col in counts.columns:
        col_bytes = np.ascontiguousarray(counts[col].to_numpy()).tobytes()
        hashes.append(hashlib.sha1(col_bytes).hexdigest())
    hash_series = pd.Series(hashes, index=counts.columns)
    exact_dup_mask = hash_series.duplicated(keep="first")
    dropped = list(counts.columns[exact_dup_mask].astype(str))
    n_exact_dup = int(exact_dup_mask.sum())
    if n_exact_dup:
        log.info(
            "Dropping %d samples whose count vector is identical to an earlier sample.",
            n_exact_dup,
        )

    counts = counts.loc[:, ~exact_dup_mask]
    sample_meta = sample_meta.loc[counts.columns]

    return (
        counts,
        sample_meta,
        DedupReport(
            n_input=n_input,
            n_repeated_sample_id_removed=n_id_dup,
            n_identical_vector_removed=n_exact_dup,
            n_output=counts.shape[1],
            dropped_sample_ids=dropped,
        ),
    )


# --------------------------------------------------------------------------- #
# Step 3: low-count gene filter
# --------------------------------------------------------------------------- #
@dataclass
class GeneFilterReport:
    n_input_genes: int
    n_kept_genes: int
    cpm_threshold: float
    min_samples_required: int


def filter_low_count_genes(
    counts: pd.DataFrame,
    cpm_threshold: float,
    min_samples_frac: float,
    min_samples_abs: int,
) -> Tuple[pd.DataFrame, GeneFilterReport]:
    """Keep genes expressed above ``cpm_threshold`` CPM in a minimum number of samples.

    ``min_samples_required = max(ceil(min_samples_frac * N), min_samples_abs)``
    so tiny cohorts don't set an absurdly low bar.
    """
    n_samples = counts.shape[1]
    min_samples_required = max(int(np.ceil(min_samples_frac * n_samples)), min_samples_abs)

    lib_size = counts.sum(axis=0).replace(0, np.nan)
    cpm = counts.divide(lib_size, axis=1) * 1e6
    cpm = cpm.fillna(0.0)

    passes = (cpm > cpm_threshold).sum(axis=1)
    keep_mask = passes >= min_samples_required
    counts_out = counts.loc[keep_mask]

    return counts_out, GeneFilterReport(
        n_input_genes=counts.shape[0],
        n_kept_genes=int(keep_mask.sum()),
        cpm_threshold=cpm_threshold,
        min_samples_required=min_samples_required,
    )


# --------------------------------------------------------------------------- #
# Step 4: normalization
# --------------------------------------------------------------------------- #
@dataclass
class NormalizationReport:
    method: str
    notes: str


def normalize(
    counts: pd.DataFrame,
    method: str,
    log_pseudocount: float,
) -> Tuple[pd.DataFrame, NormalizationReport]:
    """Return normalized expression matrix.

    Supported methods:
      * ``cpm_log2``: ``log2(CPM + pseudocount)`` — default.
      * ``deseq2``:   DESeq2 median-of-ratios via optional ``pydeseq2`` dep.
      * ``tmm``:      edgeR TMM via rpy2 — not wired up; raises so caller
                      can choose either to install the dep or fall back.
    """
    if method == "cpm_log2":
        lib_size = counts.sum(axis=0).replace(0, np.nan)
        cpm = counts.divide(lib_size, axis=1) * 1e6
        cpm = cpm.fillna(0.0)
        norm = np.log2(cpm + log_pseudocount)
        return norm, NormalizationReport(
            method="cpm_log2",
            notes=(
                f"log2(CPM + {log_pseudocount}); library size computed per sample; "
                "zeros preserved as log2(pseudocount)."
            ),
        )

    if method == "deseq2":
        try:
            from pydeseq2.dds import DeseqDataSet
        except ImportError as e:  # pragma: no cover — depends on optional dep
            raise ImportError(
                "norm_method='deseq2' requires `pip install pydeseq2`."
            ) from e

        # DeseqDataSet expects (samples × genes) integer counts.
        counts_int = counts.round().astype(np.int64).T
        dds = DeseqDataSet(
            counts=counts_int,
            metadata=pd.DataFrame(index=counts_int.index),
            design="~1",
        )
        dds.fit_size_factors()
        size_factors = pd.Series(dds.obsm["size_factors"], index=counts_int.index)
        norm = counts.divide(size_factors, axis=1)
        norm = np.log2(norm + 1.0)
        return norm, NormalizationReport(
            method="deseq2",
            notes=(
                "DESeq2 median-of-ratios size factors via pydeseq2; log2(normalized + 1)."
            ),
        )

    if method == "tmm":
        raise NotImplementedError(
            "TMM requires edgeR via rpy2 and is not wired up in this module. "
            "Use --norm-method cpm_log2 (default) or --norm-method deseq2."
        )

    raise ValueError(f"Unknown norm method: {method!r}")
