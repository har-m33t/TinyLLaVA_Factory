"""Cohort composition summary.

Sample counts, series counts, per-label counts, and — when present in the
sample metadata — sex / race / age / tissue breakdowns. Everything here is
pure pandas; nothing plots or writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd

from .loaders import LabeledDataset


# Columns we opportunistically summarize if the sample metadata carries
# them. RECOUNT3 typically has some subset; ARCHS4's sample-level fields are
# sparser, so we tolerate any of these being missing.
_DEMOGRAPHIC_COLUMNS = (
    "sex", "gender",
    "race", "ethnicity",
    "age", "age_at_diagnosis", "age_at_index",
    "tissue", "tissue_type", "body_site",
)


@dataclass
class CohortReport:
    n_samples: int
    n_series: int
    per_label: Dict[str, int] = field(default_factory=dict)
    per_series: Dict[str, int] = field(default_factory=dict)
    demographics: Dict[str, Dict[str, int]] = field(default_factory=dict)
    numeric_demographics: Dict[str, Dict[str, float]] = field(default_factory=dict)


def _series_of(meta: pd.DataFrame) -> pd.Series:
    if "series_id" in meta.columns:
        return meta["series_id"].astype(str).fillna("(unknown)")
    return pd.Series(["(unknown)"] * len(meta), index=meta.index)


def summarize(ds: LabeledDataset) -> CohortReport:
    """Produce a CohortReport for one dataset.

    Numeric demographics (things that look like age) get a five-number
    summary rather than a value_counts breakdown so the report stays legible
    on samples where every age is unique.
    """
    meta = ds.sample_meta
    series = _series_of(meta)

    report = CohortReport(
        n_samples=len(meta),
        n_series=int(series.nunique()),
        per_label=meta["label"].astype(str).value_counts().to_dict(),
        per_series=series.value_counts().head(20).to_dict(),
    )

    seen = set()
    for col in _DEMOGRAPHIC_COLUMNS:
        if col not in meta.columns or col in seen:
            continue
        seen.add(col)
        series_data = meta[col]
        if pd.api.types.is_numeric_dtype(series_data):
            valid = series_data.dropna().astype(float)
            if valid.empty:
                continue
            report.numeric_demographics[col] = {
                "n": int(valid.size),
                "min": float(valid.min()),
                "median": float(valid.median()),
                "max": float(valid.max()),
                "mean": float(valid.mean()),
                "std": float(valid.std(ddof=0)),
            }
        else:
            counts = series_data.astype(str).fillna("(missing)").value_counts().to_dict()
            report.demographics[col] = counts

    return report
