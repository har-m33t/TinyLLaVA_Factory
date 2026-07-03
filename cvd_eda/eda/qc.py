"""Per-sample QC on the Task 4 normalized matrix.

Two metrics we can always compute (library size and genes-detected-per-sample)
plus one optional (biotype composition) that only runs when the caller
supplies an Ensembl→biotype map. All three are reported as per-sample tables
so plotting code can render histograms + boxplots without recomputing.

Note on library size: Task 4 emits `log2(CPM + 1)`. Per-sample summed
expression on that scale is a proxy for library complexity, not raw read
count. For raw library size a caller can pass the pre-normalization matrix
here — the code only cares about the shape, not the units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .loaders import LabeledDataset


@dataclass
class QCReport:
    per_sample: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    biotype_share: Optional[pd.DataFrame] = None
    biotype_share_summary: Optional[Dict[str, Dict[str, float]]] = None


def _five_number(values: pd.Series) -> Dict[str, float]:
    v = values.dropna().astype(float)
    if v.empty:
        return {}
    return {
        "n": int(v.size),
        "min": float(v.min()),
        "q1": float(v.quantile(0.25)),
        "median": float(v.median()),
        "q3": float(v.quantile(0.75)),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=0)),
    }


def compute(
    ds: LabeledDataset,
    biotype_map: Optional[pd.Series] = None,
    *,
    detected_threshold: float = 0.0,
) -> QCReport:
    """Per-sample library size, genes-detected, and (optional) biotype share.

    ``detected_threshold`` is compared against the values in ``ds.expression``.
    For a `log2(CPM + 1)` matrix, 0.0 corresponds to genes with truly zero
    input counts (since `log2(0 + 1) == 0`), so it's a sensible default.
    """
    expr = ds.expression

    lib_size = expr.sum(axis=0)
    detected = (expr > detected_threshold).sum(axis=0)

    per_sample = pd.DataFrame(
        {
            "library_size": lib_size.astype(float),
            "genes_detected": detected.astype(int),
            "label": ds.sample_meta["label"].astype(str).reindex(expr.columns).values,
        },
        index=expr.columns,
    )

    report = QCReport(
        per_sample=per_sample,
        summary={
            "library_size": _five_number(per_sample["library_size"]),
            "genes_detected": _five_number(per_sample["genes_detected"].astype(float)),
        },
    )

    if biotype_map is not None:
        # Fraction of each sample's total expression coming from each biotype.
        # Genes with no biotype mapping get bucketed as ``unknown`` rather than
        # silently dropped, so the shares sum to 1.
        gene_biotype = pd.Series(
            "unknown", index=expr.index, name="biotype"
        ).astype(str)
        overlap = expr.index.intersection(biotype_map.index)
        gene_biotype.loc[overlap] = biotype_map.loc[overlap].astype(str).values

        # Group genes by biotype, sum, divide by per-sample total.
        totals = expr.sum(axis=0).replace(0, np.nan)
        grouped = expr.groupby(gene_biotype).sum()
        biotype_share = grouped.divide(totals, axis=1).fillna(0.0).T
        biotype_share.index.name = "sample_id"

        report.biotype_share = biotype_share
        report.biotype_share_summary = {
            bt: _five_number(biotype_share[bt]) for bt in biotype_share.columns
        }

    return report
