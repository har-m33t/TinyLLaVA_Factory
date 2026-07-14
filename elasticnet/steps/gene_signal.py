"""
gene_signal.py — Task 10: aggregate coefficients across outer folds
                          and cross-check against ClinGen HCVD.

Aggregation
-----------
For each gene, compute:
- mean_coef      — mean β across outer folds
- median_coef    — robust variant, useful when one fold is anomalous
- nonzero_frac   — fraction of folds where |β| > 0. This is the "signal
                   concentration" metric the elastic net was chosen for:
                   genes with mean_coef large *and* nonzero_frac near 1
                   are the confident signal; large-|mean_coef| /
                   low-nonzero-frac genes are single-fold quirks.

The full ranked list is written (task 10 explicitly says "not just top N"),
sorted by |mean_coef| descending. Ties broken by nonzero_frac descending.

ClinGen cross-check
-------------------
Applied *after* ranking: a boolean `in_clingen_hcvd` column, plus a
count of ClinGen recoveries in the top-N. See `elasticnet/clingen.py`
for the reference set and its provenance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..clingen import CLINGEN_HCVD_STARTER

logger = logging.getLogger(__name__)


def run(
    folds_dir: Path,
    expression_dir: Path,
    outdir: Path,
    top_n_clingen_report: int = 100,
) -> Path:
    out = outdir / "gene_signal"
    out.mkdir(parents=True, exist_ok=True)

    gene_symbols = np.load(expression_dir / "gene_symbols.npy", allow_pickle=False)

    fold_dirs = sorted([d for d in folds_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    if not fold_dirs:
        raise RuntimeError(f"No fold_* directories found under {folds_dir}.")

    coefs = np.stack([np.load(d / "coefficients.npy") for d in fold_dirs], axis=0)
    if coefs.shape[1] != gene_symbols.shape[0]:
        raise ValueError(
            f"coefficient width {coefs.shape[1]} != n_genes {gene_symbols.shape[0]}. "
            "Fold artifacts were built against a different gene mask than the "
            "expression matrix. Rerun train + gene_signal on matched artifacts."
        )

    mean_coef = coefs.mean(axis=0)
    median_coef = np.median(coefs, axis=0)
    nonzero_frac = (coefs != 0).mean(axis=0)

    clingen_set = set(CLINGEN_HCVD_STARTER)
    in_clingen = np.array([str(g).upper() in {c.upper() for c in clingen_set} for g in gene_symbols])

    df = pd.DataFrame({
        "gene_symbol": gene_symbols,
        "mean_coef": mean_coef.astype(np.float32),
        "median_coef": median_coef.astype(np.float32),
        "abs_mean_coef": np.abs(mean_coef).astype(np.float32),
        "nonzero_frac": nonzero_frac.astype(np.float32),
        "in_clingen_hcvd": in_clingen,
    })
    df = df.sort_values(
        by=["abs_mean_coef", "nonzero_frac"],
        ascending=[False, False],
    ).reset_index(drop=True)
    df.to_csv(out / "gene_signal_ranking.csv", index=False)

    top = df.head(top_n_clingen_report)
    summary = {
        "n_genes": int(len(df)),
        "n_folds": int(coefs.shape[0]),
        "clingen_starter_size": int(len(clingen_set)),
        "clingen_present_in_kept_genes": int(in_clingen.sum()),
        "clingen_recovered_in_top_n": {
            f"top_{top_n_clingen_report}": int(top["in_clingen_hcvd"].sum()),
        },
        "top_10_by_abs_mean_coef": df.head(10)[
            ["gene_symbol", "mean_coef", "nonzero_frac", "in_clingen_hcvd"]
        ].to_dict(orient="records"),
    }
    with open(out / "gene_signal_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "gene signal: %d ClinGen HCVD genes present in kept-gene set; %d recovered in top-%d",
        int(in_clingen.sum()),
        int(top["in_clingen_hcvd"].sum()),
        top_n_clingen_report,
    )
    return out
