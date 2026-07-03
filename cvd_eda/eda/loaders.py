"""Load Task 4's normalized matrix, sample metadata, and Task 5's
human-reviewed labels into one :class:`LabeledDataset`.

The rest of the EDA pipeline reads that struct — nothing else knows about
Parquet, CSV, or the review-file convention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


LOG = logging.getLogger(__name__)


REQUIRED_LABEL_COLUMNS = ("sample_id", "proposed_label", "confidence")


@dataclass
class LabeledDataset:
    """Analysis-ready view of one dataset.

    Shape contract:
        * ``expression``: ``(n_genes, n_samples)`` normalized matrix.
          Rows indexed by canonical Ensembl ID (versionless), columns by
          ``sample_id``.
        * ``sample_meta``: ``sample_id``-indexed frame. Guaranteed to have a
          ``label`` column (from the reviewed labels) and a ``series_id``
          column (either passed through from Task 4 or the ``rel_source_series_id``
          fallback).
    """

    name: str
    expression: pd.DataFrame
    sample_meta: pd.DataFrame
    n_samples_matrix: int
    n_samples_labeled: int
    n_samples_dropped_unlabeled: int


def load_labels(
    labels_csv: Path,
    *,
    label_column: str = "proposed_label",
    confidence_column: str = "confidence",
    min_confidence: float = 0.0,
) -> pd.DataFrame:
    """Load the reviewed labels CSV.

    Any extra columns are preserved; only the three required ones are
    validated. Rows with confidence below ``min_confidence`` are dropped
    (default 0.0 keeps everything — the reviewer already vetted the file).
    """
    df = pd.read_csv(labels_csv)
    missing = [c for c in REQUIRED_LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Reviewed labels CSV {labels_csv} is missing required columns: "
            f"{missing}. Expected at least {REQUIRED_LABEL_COLUMNS}."
        )

    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["confidence"] = pd.to_numeric(df[confidence_column], errors="coerce").fillna(0.0)
    if min_confidence > 0:
        n_before = len(df)
        df = df.loc[df["confidence"] >= min_confidence].copy()
        LOG.info(
            "Dropped %d/%d label rows below confidence %.2f",
            n_before - len(df),
            n_before,
            min_confidence,
        )

    df = df.rename(columns={label_column: "label"})
    df = df.drop_duplicates(subset=["sample_id"], keep="first")
    return df.set_index("sample_id")


def load_dataset(
    dataset_name: str,
    matrix_parquet: Path,
    sample_meta_parquet: Path,
    labels_df: pd.DataFrame,
) -> LabeledDataset:
    """Load one Task 4 output trio and join it against the reviewed labels.

    Samples with no reviewed label are dropped — Task 6 exists to look at
    labeled samples only. The count of dropped samples is preserved in the
    returned struct so the audit log can record it.
    """
    matrix = pd.read_parquet(matrix_parquet)
    sample_meta = pd.read_parquet(sample_meta_parquet)

    matrix.columns = matrix.columns.astype(str)
    sample_meta.index = sample_meta.index.astype(str)

    n_matrix = matrix.shape[1]

    # Attach labels; drop unlabeled samples.
    have_labels = labels_df.index.intersection(matrix.columns)
    n_dropped = n_matrix - len(have_labels)
    if n_dropped:
        LOG.warning(
            "Dropping %d/%d samples from %s: no reviewed label present.",
            n_dropped,
            n_matrix,
            dataset_name,
        )
    if len(have_labels) == 0:
        raise ValueError(
            f"No samples in {matrix_parquet.name} have a matching row in the "
            f"reviewed labels CSV. Sample IDs may not line up."
        )

    ordered = [s for s in matrix.columns if s in have_labels]
    expression = matrix.loc[:, ordered].copy()
    sample_meta = sample_meta.reindex(ordered).copy()

    # Merge label columns into sample_meta so plotting code only reads one frame.
    label_cols = ("label", "confidence", "evidence_quote", "uncertain_reason",
                  "source_series_id")
    for col in label_cols:
        if col in labels_df.columns:
            sample_meta[col] = labels_df.loc[ordered, col].values

    # Task 4 stores the source series either as ``series_id`` or as the
    # provenance column ``rel_source_series_id``. Normalize to ``series_id``
    # so downstream code doesn't have to know.
    if "series_id" not in sample_meta.columns:
        if "rel_source_series_id" in sample_meta.columns:
            sample_meta["series_id"] = sample_meta["rel_source_series_id"]
        elif "source_series_id" in sample_meta.columns:
            sample_meta["series_id"] = sample_meta["source_series_id"]

    return LabeledDataset(
        name=dataset_name,
        expression=expression,
        sample_meta=sample_meta,
        n_samples_matrix=n_matrix,
        n_samples_labeled=len(ordered),
        n_samples_dropped_unlabeled=n_dropped,
    )


def load_gene_biotype_map(tsv_path: Optional[Path]) -> Optional[pd.Series]:
    """Optional Ensembl-ID → biotype lookup for the biotype-composition QC.

    File is a TSV/CSV with at least ``ensembl_id`` and ``biotype`` columns.
    Version suffixes on the ensembl_id are stripped so the mapping matches
    the canonical (versionless) IDs Task 4 emits.
    """
    if tsv_path is None:
        return None
    path = Path(tsv_path)
    sep = "," if path.suffix.lower() == ".csv" else "\t"
    df = pd.read_csv(path, sep=sep)
    if "ensembl_id" not in df.columns or "biotype" not in df.columns:
        raise ValueError(
            f"Gene biotype map {path} must contain 'ensembl_id' and 'biotype' columns."
        )
    df["ensembl_id"] = df["ensembl_id"].astype(str).str.split(".").str[0]
    return df.set_index("ensembl_id")["biotype"]
