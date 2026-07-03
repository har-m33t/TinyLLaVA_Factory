"""Harmonize gene identifiers into a single canonical space.

Canonical choice: **Ensembl gene ID, versionless** (e.g. ``ENSG00000141510``).

Why Ensembl versionless?
  * Ensembl IDs are stable across HGNC symbol changes; symbols rot every
    release. ``TP53`` → OK today, but historically shared with ``p53``,
    ``LFS1``, etc.; using symbols means silently merging or dropping genes
    depending on which release each dataset was annotated against.
  * RECOUNT3 already exports Ensembl IDs (with ``.N`` version suffixes).
  * ARCHS4 v2.x exports ``ensembl_gene_id`` alongside symbols. Older ARCHS4
    releases were symbol-only, which is why a ``symbol -> ensembl_id``
    mapping table is still supported as a fallback.
  * Stripping the version suffix collapses gene-model revisions of the same
    locus (e.g. ``ENSG00000141510.15`` and ``.16``) rather than treating them
    as distinct features. This costs a tiny amount of paralog-level
    resolution and the drop is logged.

When two raw rows map to the same canonical Ensembl ID (e.g. two symbols
that share an Ensembl gene, or an ID appearing twice in ARCHS4), we
**sum** their counts. That's the DESeq2/edgeR convention for gene-level
aggregation and preserves the "counts of reads assigned to gene G" semantic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

_ENSG_RE = re.compile(r"^(ENSG\d+)(?:\.\d+)?$")


def strip_ensembl_version(gene_id: str) -> Optional[str]:
    """``ENSG00000141510.15`` → ``ENSG00000141510``. Non-matches return ``None``."""
    if not isinstance(gene_id, str):
        return None
    m = _ENSG_RE.match(gene_id)
    return m.group(1) if m else None


@dataclass
class HarmonizationResult:
    counts: pd.DataFrame  # index = canonical ensembl_id, columns = sample_id
    n_mapped: int
    n_unmapped: int
    n_duplicate_canonical: int


def harmonize_to_ensembl(
    counts: pd.DataFrame,
    gene_meta: pd.DataFrame,
    source_scheme: str,
    symbol_to_ensembl: Optional[pd.Series] = None,
) -> HarmonizationResult:
    """Return counts with index remapped to canonical Ensembl gene IDs.

    Rows whose raw ID cannot be mapped are dropped (and counted). When
    multiple raw rows collapse to the same canonical ID, their counts are
    summed.
    """
    raw_ids = counts.index.astype(str)

    if source_scheme == "ensembl":
        canonical = pd.Series([strip_ensembl_version(x) for x in raw_ids], index=counts.index)
    elif source_scheme == "symbol":
        # Prefer an already-present ensembl_id column in gene_meta (ARCHS4 v2.x case)
        if (
            "ensembl_id" in gene_meta.columns
            and gene_meta["ensembl_id"].astype(str).str.startswith("ENSG").any()
        ):
            canonical = gene_meta["ensembl_id"].astype(str).map(strip_ensembl_version)
            canonical.index = counts.index
        else:
            if symbol_to_ensembl is None:
                raise ValueError(
                    "source_scheme='symbol' and no ensembl_id column in gene_meta; "
                    "pass --gene-id-map with a TSV of columns [symbol, ensembl_id]."
                )
            m = symbol_to_ensembl.to_dict()
            canonical = pd.Series([m.get(str(s)) for s in raw_ids], index=counts.index)
    else:
        raise ValueError(f"Unknown source_scheme: {source_scheme!r}")

    unmapped_mask = canonical.isna() | (canonical.astype(str) == "")
    n_unmapped = int(unmapped_mask.sum())
    if n_unmapped:
        log.info("Dropping %d/%d rows with no Ensembl mapping.", n_unmapped, len(canonical))

    counts_mapped = counts.loc[~unmapped_mask].copy()
    counts_mapped.index = pd.Index(canonical.loc[~unmapped_mask].values, name="ensembl_id")

    n_before_collapse = len(counts_mapped)
    counts_collapsed = counts_mapped.groupby(level=0).sum()
    n_dup = n_before_collapse - len(counts_collapsed)
    if n_dup:
        log.info("Sum-collapsed %d duplicate canonical IDs.", n_dup)

    return HarmonizationResult(
        counts=counts_collapsed,
        n_mapped=len(counts_collapsed),
        n_unmapped=n_unmapped,
        n_duplicate_canonical=n_dup,
    )


def load_symbol_to_ensembl_map(path: Path) -> pd.Series:
    """Load a TSV of columns ``[symbol, ensembl_id]`` into a ``Series`` indexed by symbol.

    Ambiguous symbols (multiple ``ensembl_id`` rows for the same symbol) keep
    the first mapping; the number dropped is logged. Ensembl versions on the
    map's ``ensembl_id`` column are stripped so the map's output matches the
    canonical space.
    """
    path = Path(path)
    df = pd.read_csv(path, sep="\t")
    required = {"symbol", "ensembl_id"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Gene ID map at {path} must have columns 'symbol' and 'ensembl_id'; "
            f"got {list(df.columns)}."
        )
    n_dup = int(df["symbol"].duplicated().sum())
    if n_dup:
        log.warning(
            "Gene ID map has %d duplicate symbols; keeping first mapping for each.", n_dup
        )
    df = df.drop_duplicates(subset="symbol", keep="first")
    return (
        df.set_index("symbol")["ensembl_id"]
        .astype(str)
        .map(strip_ensembl_version)
    )
