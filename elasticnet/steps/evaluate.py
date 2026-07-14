"""
evaluate.py — Task 8: per-fold + aggregate performance metrics.

Reads each fold's `test_predictions.parquet` from the training-orchestrator
output, computes the metrics locked in `.claude/elastic_net_todo.md`, and
writes `performance_by_fold.csv` plus a summary manifest with mean ± std
across folds.

Metrics
-------
- ROC-AUC          — imbalance-insensitive baseline
- PR-AUC (average_precision)  — **primary metric** given the 10:1 imbalance
- accuracy         — reported but not lead-with, because of imbalance
- sensitivity      — recall on positives = TP / (TP + FN)
- specificity      — recall on negatives = TN / (TN + FP)
- F1               — harmonic mean of precision/recall
- n_positive       — sanity check on stratification

Threshold for hard metrics (accuracy, sens, spec, F1) is 0.5 — noted in the
writeup as a default calibration choice.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


PREDICT_THRESHOLD = 0.5


def _fold_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute one fold's metrics; robust to a fold with only one class present."""
    y_pred = (y_score >= PREDICT_THRESHOLD).astype(np.int8)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    metrics = {
        "n_samples": int(y_true.shape[0]),
        "n_positive": n_pos,
        "n_negative": n_neg,
    }
    # AUCs require both classes present; return NaN if not.
    both_classes = n_pos > 0 and n_neg > 0
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_score)) if both_classes else float("nan")
    metrics["pr_auc"] = float(average_precision_score(y_true, y_score)) if both_classes else float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["accuracy"] = float((tp + tn) / max(tp + tn + fp + fn, 1))
    metrics["sensitivity"] = float(tp / max(tp + fn, 1)) if n_pos > 0 else float("nan")
    metrics["specificity"] = float(tn / max(tn + fp, 1)) if n_neg > 0 else float("nan")
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"] = int(tp), int(tn), int(fp), int(fn)
    return metrics


def run(folds_dir: Path, outdir: Path) -> Path:
    out = outdir / "performance"
    out.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted([d for d in folds_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    if not fold_dirs:
        raise RuntimeError(f"No fold_* directories found under {folds_dir}.")

    rows = []
    for fold_dir in fold_dirs:
        pred = pd.read_parquet(fold_dir / "test_predictions.parquet")
        m = _fold_metrics(pred["label"].to_numpy(), pred["y_score"].to_numpy())
        m["fold_id"] = int(fold_dir.name.split("_")[-1])
        rows.append(m)
    df = pd.DataFrame(rows).sort_values("fold_id").reset_index(drop=True)
    df.to_csv(out / "performance_by_fold.csv", index=False)

    numeric_cols = ["roc_auc", "pr_auc", "accuracy", "sensitivity", "specificity", "f1"]
    summary = {
        "n_folds": int(len(df)),
        "predict_threshold": PREDICT_THRESHOLD,
        "per_metric": {
            col: {
                "mean": float(np.nanmean(df[col])),
                "std": float(np.nanstd(df[col], ddof=1)) if len(df) > 1 else 0.0,
                "min": float(np.nanmin(df[col])),
                "max": float(np.nanmax(df[col])),
            }
            for col in numeric_cols
        },
        "note": (
            "Metrics computed on the matched 10:1 training pool, not the true "
            "corpus-wide imbalance. Real-world deployment would require "
            "separate calibration — see writeup."
        ),
    }
    with open(out / "performance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "performance summary: PR-AUC %.3f ± %.3f, ROC-AUC %.3f ± %.3f (n_folds=%d)",
        summary["per_metric"]["pr_auc"]["mean"], summary["per_metric"]["pr_auc"]["std"],
        summary["per_metric"]["roc_auc"]["mean"], summary["per_metric"]["roc_auc"]["std"],
        len(df),
    )
    return out
