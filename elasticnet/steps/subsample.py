"""
subsample.py — Task 2: build the training pool by subsampling negatives.

Class-imbalance strategy (locked in `.claude/elastic_net_todo.md`)
-----------------------------------------------------------------
Keep every CVD-matched (positive) sample. Draw a fixed ratio of negatives
uniformly at random from the remaining corpus (default 10:1 neg:pos).
Combine with `class_weight="balanced"` on the model side — belt-and-suspenders
on a noisy weak label.

The single-cell filter from `dataset.io.filter_bulk_indices` is applied
first: single-cell samples are excluded from *both* the positive and
negative pools before subsampling, matching the same "exclude sc,
subsample from bulk pool" convention used everywhere else in the EDA.

Outputs
-------
training_pool.parquet
    Columns: sample_index, sample_id (GSM), label, source_series_id.
    This is the population that steps 3+ actually train/test on.
subsample_manifest.json
    Records n_positive, n_negative, ratio, seed, filter stats, timestamps.
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


DEFAULT_NEGATIVE_RATIO = 10
DEFAULT_SEED = 20260707  # elastic net stage epoch


def run(
    h5_path: Path,
    label_dir: Path,
    outdir: Path,
    negative_ratio: int = DEFAULT_NEGATIVE_RATIO,
    seed: int = DEFAULT_SEED,
) -> Path:
    out = outdir / "subsample"
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    labels = np.load(label_dir / "labels.npy")
    n_samples = labels.shape[0]

    # Apply the single-cell filter *before* the pos/neg split so the pool
    # semantics match every other subsampling step in the codebase.
    with archs4_io.open_h5(h5_path) as h5:
        bulk_pool, filter_stats = archs4_io.filter_bulk_indices(h5)
        sample_ids = archs4_io.read_sample_field(h5, "geo_accession")
        series_ids = archs4_io.read_sample_field(h5, "series_id")
    if sample_ids is None or series_ids is None:
        raise KeyError("H5 missing geo_accession or series_id — required for training pool.")

    in_bulk = np.zeros(n_samples, dtype=bool)
    in_bulk[bulk_pool] = True

    pos_mask = (labels == 1) & in_bulk
    neg_mask = (labels == 0) & in_bulk

    pos_idx = np.flatnonzero(pos_mask)
    neg_idx_pool = np.flatnonzero(neg_mask)

    n_pos = int(pos_idx.size)
    if n_pos == 0:
        raise RuntimeError(
            "No positive samples after single-cell filter — keyword list may "
            "have failed to match, or the label file was built against a "
            "different H5. Nothing to train on."
        )

    n_neg_target = min(n_pos * negative_ratio, int(neg_idx_pool.size))
    rng = np.random.default_rng(seed)
    neg_idx = np.sort(neg_idx_pool[rng.choice(neg_idx_pool.size, size=n_neg_target, replace=False)])

    logger.info(
        "training pool: %d positives (all kept) + %d negatives (of %d bulk negatives available; ratio %d:1)",
        n_pos, n_neg_target, int(neg_idx_pool.size), negative_ratio,
    )

    pool_idx = np.concatenate([pos_idx, neg_idx])
    pool_labels = np.concatenate([
        np.ones(n_pos, dtype=np.int8),
        np.zeros(n_neg_target, dtype=np.int8),
    ])

    df = pd.DataFrame({
        "sample_index": pool_idx.astype(np.int64),
        "sample_id": sample_ids[pool_idx],
        "label": pool_labels,
        "source_series_id": series_ids[pool_idx],
    })
    df.to_parquet(out / "training_pool.parquet", index=False)

    manifest = {
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "n_positive": n_pos,
        "n_negative": n_neg_target,
        "negative_ratio_requested": negative_ratio,
        "negative_ratio_actual": round(n_neg_target / n_pos, 3) if n_pos else 0.0,
        "seed": int(seed),
        "singlecell_filter": filter_stats,
        "n_unique_series_positives": int(pd.unique(series_ids[pos_idx]).size),
        "n_unique_series_negatives": int(pd.unique(series_ids[neg_idx]).size),
    }
    with open(out / "subsample_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return out
