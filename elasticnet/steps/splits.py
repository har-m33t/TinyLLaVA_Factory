"""
splits.py — Task 4: StratifiedGroupKFold over the training pool.

Grouping is by `source_series_id`. Same-series leakage is the single most
important thing to prevent here: with a whole-corpus weak label, the model
is one shortcut away from learning study-specific batch signatures instead
of biology. Grouping the folds so no series appears in both train and test
kills that shortcut.

Stratification is by label so each fold gets a representative positive
rate (still 1:10-ish after subsampling — StratifiedGroupKFold enforces
this per-fold, subject to the group constraint).

Outputs
-------
fold_assignments.npy   int, shape (n_pool,) — 0..n_outer-1 fold ID per sample
splits_manifest.json   fold sizes, per-fold pos/neg counts, series overlap check
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)


DEFAULT_N_OUTER_FOLDS = 5
DEFAULT_SEED = 20260708


def run(
    subsample_dir: Path,
    outdir: Path,
    n_outer_folds: int = DEFAULT_N_OUTER_FOLDS,
    seed: int = DEFAULT_SEED,
) -> Path:
    out = outdir / "splits"
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    pool = pd.read_parquet(subsample_dir / "training_pool.parquet")
    y = pool["label"].to_numpy()
    groups = pool["source_series_id"].to_numpy()
    n = y.shape[0]

    n_unique_series = int(pd.unique(groups).size)
    if n_unique_series < n_outer_folds:
        raise RuntimeError(
            f"Only {n_unique_series} unique series in training pool — cannot "
            f"build {n_outer_folds} grouped folds. Reduce n_outer_folds or "
            "check that series_id was loaded correctly."
        )

    sgkf = StratifiedGroupKFold(n_splits=n_outer_folds, shuffle=True, random_state=seed)
    fold_assignments = np.full(n, -1, dtype=np.int8)
    for fold_id, (_train_idx, test_idx) in enumerate(sgkf.split(np.zeros(n), y, groups)):
        fold_assignments[test_idx] = fold_id

    if (fold_assignments < 0).any():
        raise RuntimeError("Some samples were not assigned to any fold — bug in split logic.")

    np.save(out / "fold_assignments.npy", fold_assignments)

    per_fold = []
    for fold_id in range(n_outer_folds):
        mask = fold_assignments == fold_id
        per_fold.append({
            "fold_id": fold_id,
            "n_samples": int(mask.sum()),
            "n_positive": int((y[mask] == 1).sum()),
            "n_negative": int((y[mask] == 0).sum()),
            "n_unique_series": int(pd.unique(groups[mask]).size),
        })

    # Explicit no-series-leakage check: every series appears in exactly one fold.
    per_series_folds = pd.DataFrame({"series": groups, "fold": fold_assignments}) \
        .groupby("series")["fold"].nunique()
    leaked_series = per_series_folds[per_series_folds > 1]
    if len(leaked_series) > 0:
        raise RuntimeError(
            f"Series leakage detected across folds: {len(leaked_series)} series "
            "appear in more than one fold. StratifiedGroupKFold should prevent this."
        )

    manifest = {
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "n_outer_folds": int(n_outer_folds),
        "seed": int(seed),
        "n_samples": int(n),
        "n_unique_series": n_unique_series,
        "per_fold": per_fold,
        "series_leakage_check": "passed",
    }
    with open(out / "splits_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return out
