"""
cohort.py — Task 1: cohort composition on the full ARCHS4 human corpus.

Outputs (written under `<outdir>/cohort/`):
    cohort_composition_full.csv
        One row per grouping variable × level, with sample counts.
    cohort_samples_by_year.png
        Sample count histogram by GEO submission year.
    cohort_single_cell_flag.png
        Sample count bar by singlecellprobability threshold (bulk vs. likely
        single-cell, per ARCHS4's own flag).

Design notes:
    - Reads only the metadata columns; does NOT touch the expression matrix,
      so it's cheap enough to run standalone.
    - Does NOT drop any samples. The ARCHS4 paper's single-cell flag is
      *reported* here, and the decision to exclude single-cell samples is
      documented in the write-up (step 7) and applied as a downstream filter,
      not silently here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset import io as archs4_io
from ..plotting import apply_style, save_figure

logger = logging.getLogger(__name__)

# Threshold above which ARCHS4 flags a sample as likely single-cell. The paper
# recommends 0.5 as a working cutoff; we report both counts and let the
# downstream filter decide.
SC_PROB_THRESHOLD = 0.5


def _extract_year(submission_dates: np.ndarray) -> np.ndarray:
    """Return an int year vector; NaN-sentinel (-1) where parse fails.

    ARCHS4 stores submission dates in ISO-ish strings; some releases use
    'YYYY-MM-DD', older ones use 'MMM DD YYYY'. Parse via pandas to absorb
    both without hardcoding format.
    """
    parsed = pd.to_datetime(pd.Series(submission_dates), errors="coerce", utc=False)
    years = parsed.dt.year.to_numpy()
    years = np.where(pd.isna(years), -1, years).astype(int)
    return years


def run(h5_path: Path, outdir: Path) -> Path:
    """Compute cohort composition and write CSV + figures. Returns the CSV path."""
    apply_style()
    out = outdir / "cohort"
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    with archs4_io.open_h5(h5_path) as h5:
        shape = archs4_io.get_shape(h5)
        rows.append({"grouping": "total", "level": "samples", "count": shape.n_samples})
        rows.append({"grouping": "total", "level": "genes", "count": shape.n_genes})

        # Unique GEO series count
        series = archs4_io.read_sample_field(h5, "series_id")
        if series is not None:
            rows.append({"grouping": "total", "level": "geo_series", "count": int(pd.Series(series).nunique())})

        # Single-cell flag distribution
        sc_prob = archs4_io.read_sample_field(h5, "singlecellprobability")
        if sc_prob is not None:
            sc_prob = np.asarray(sc_prob, dtype=float)
            n_sc = int((sc_prob >= SC_PROB_THRESHOLD).sum())
            n_bulk = int((sc_prob < SC_PROB_THRESHOLD).sum())
            rows.append({"grouping": "single_cell_flag", "level": f"likely_sc(>={SC_PROB_THRESHOLD})", "count": n_sc})
            rows.append({"grouping": "single_cell_flag", "level": f"likely_bulk(<{SC_PROB_THRESHOLD})", "count": n_bulk})

            fig, ax = _bar_singlecell(n_bulk, n_sc)
            save_figure(fig, out / "cohort_single_cell_flag.png")

        # Auto-detect label fields for the subset-EDA use case. These fields
        # are absent in whole-corpus ARCHS4 releases, so this block is a no-op
        # there; when the CVD-subset pipeline writes its subset H5 with
        # `case_control_label` and `etiology_label` embedded (see
        # `cvd_subset/subset_h5.py`), the composition CSV surfaces them
        # without any coupling to the subset code.
        for label_field in ("case_control_label", "etiology_label"):
            arr = archs4_io.read_sample_field(h5, label_field)
            if arr is None:
                continue
            counts = pd.Series(arr).value_counts().sort_index()
            for level, ct in counts.items():
                rows.append({"grouping": label_field, "level": str(level), "count": int(ct)})

        # Submission-year distribution
        sub_dates = archs4_io.read_sample_field(h5, "submission_date")
        if sub_dates is not None:
            years = _extract_year(sub_dates)
            valid = years[years > 0]
            year_counts = pd.Series(valid).value_counts().sort_index()
            for yr, ct in year_counts.items():
                rows.append({"grouping": "submission_year", "level": str(int(yr)), "count": int(ct)})

            fig, ax = _hist_year(year_counts)
            save_figure(fig, out / "cohort_samples_by_year.png")

    df = pd.DataFrame(rows, columns=["grouping", "level", "count"])
    csv_path = out / "cohort_composition_full.csv"
    df.to_csv(csv_path, index=False)
    logger.info("cohort composition written to %s", csv_path)
    return csv_path


def _bar_singlecell(n_bulk: int, n_sc: int):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.bar(["bulk", "likely single-cell"], [n_bulk, n_sc], color=["#4C78A8", "#F58518"])
    ax.set_ylabel("samples")
    ax.set_title(f"ARCHS4 samples by single-cell flag (threshold {SC_PROB_THRESHOLD})")
    for i, v in enumerate([n_bulk, n_sc]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    return fig, ax


def _hist_year(year_counts: pd.Series):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.bar(year_counts.index.astype(int).astype(str), year_counts.values, color="#4C78A8")
    ax.set_xlabel("GEO submission year")
    ax.set_ylabel("samples")
    ax.set_title("ARCHS4 sample count by GEO submission year")
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    return fig, ax
