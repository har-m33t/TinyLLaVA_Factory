"""probe.py — steps 4 and 5 of the linear-probe stage.

Reusable linear-probe CV harness that runs on top of the cached embeddings
from `extract.py`. Same one implementation is invoked once per (variant,
negative-pool) pair — the TODO's explicit rule "one implementation, not five
copies".

Task setup
----------
Binary task: positive class = disease-confirmed CVD samples (per step 2);
negative class = one of the two pools written into the embedding frames:
    neg_whole_corpus   — matches the elastic-net stage's negative pool logic
    neg_hard           — tissue_only_disease_unconfirmed, harder negative

CV protocol
-----------
- StratifiedGroupKFold(n_splits=5, groups=series_id, y=positive)
- Inside each fold: `Pipeline([StandardScaler, LogisticRegression])`
  Standardization is fit on train fold only (never on val) — a leak here
  would double-count the val samples' means into their own predictions.
- Fixed seed, folds logged as a manifest for auditability.
- The linear (logistic) layer is the only trainable component — encoder
  weights are baked into the embedding parquet upstream.

Metrics (per fold, then mean±std across folds)
----------------------------------------------
ROC-AUC, PR-AUC, accuracy, sensitivity, specificity, F1, Brier score for
calibration. Same metric set as the elastic-net stage for downstream direct
comparability.

Skipping a fold
---------------
If a fold's train or val set has fewer than 25 positives OR 25 negatives,
that fold is recorded but not fit — the count is written to the manifest and
the metrics come back NaN. The step-2 25-per-fold-class floor is a downstream
guard; this is the runtime-side symmetric guard.

CLI
---
    # single (variant, pool) combination — one probe.py invocation
    python -m linear_probe.probe \\
        --embeddings linear_probe/embeddings/embeddings_BulkFormer-37M.parquet \\
        --pool neg_whole_corpus \\
        --outdir linear_probe/results/BulkFormer-37M/neg_whole_corpus/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score,
    precision_recall_curve, roc_auc_score, auc as sk_auc,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent

FLOOR_PER_FOLD_PER_CLASS = 25


@dataclass(frozen=True)
class FoldResult:
    fold: int
    n_train: int
    n_val: int
    n_train_pos: int
    n_val_pos: int
    n_train_neg: int
    n_val_neg: int
    train_series_count: int
    val_series_count: int
    fit_seconds: float | None
    skipped: bool
    reason: str | None
    roc_auc: float | None
    pr_auc: float | None
    accuracy: float | None
    sensitivity: float | None
    specificity: float | None
    f1: float | None
    brier: float | None


def _log() -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    return logging.getLogger("linear_probe.probe")


def _load_frame(embeddings_path: Path, negative_pool: str,
                logger: logging.Logger) -> pd.DataFrame:
    """Load the extracted embeddings and cut to the requested (positive ∪ neg_pool) subset."""
    df = pd.read_parquet(embeddings_path)
    if "pool" not in df.columns:
        raise KeyError(f"'pool' column missing in {embeddings_path}")

    keep = df["pool"].isin(["positive", negative_pool])
    df = df.loc[keep].reset_index(drop=True)

    n_pos = int(df["is_positive"].sum())
    n_neg = int(len(df) - n_pos)
    logger.info(f"pool='{negative_pool}': loaded {len(df)} samples "
                f"({n_pos} positives, {n_neg} negatives)")
    return df


def _feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, y (binary is_positive), groups (series_id)."""
    emb_cols = [c for c in df.columns if c.startswith("e") and c[1:].isdigit()]
    X = df[emb_cols].to_numpy(dtype=np.float32)
    y = df["is_positive"].astype(int).to_numpy()
    groups = df["series_id"].astype(str).to_numpy()
    return X, y, groups


def _fold_metrics(y_val: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    tp = int(((y_pred == 1) & (y_val == 1)).sum())
    tn = int(((y_pred == 0) & (y_val == 0)).sum())
    fp = int(((y_pred == 1) & (y_val == 0)).sum())
    fn = int(((y_pred == 0) & (y_val == 1)).sum())
    prec, rec, _ = precision_recall_curve(y_val, y_prob)
    return {
        "roc_auc":    float(roc_auc_score(y_val, y_prob)),
        "pr_auc":     float(sk_auc(rec, prec)),
        "accuracy":   float(accuracy_score(y_val, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "f1":         float(f1_score(y_val, y_pred)),
        "brier":      float(brier_score_loss(y_val, y_prob)),
    }


def run_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, k_folds: int,
           seed: int, logger: logging.Logger) -> list[FoldResult]:
    skf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_results: list[FoldResult] = []

    for i, (train_idx, val_idx) in enumerate(skf.split(X, y, groups)):
        y_train, y_val = y[train_idx], y[val_idx]
        n_train_pos, n_val_pos = int(y_train.sum()), int(y_val.sum())
        n_train_neg, n_val_neg = int((1 - y_train).sum()), int((1 - y_val).sum())
        common = dict(
            fold=i,
            n_train=int(len(train_idx)), n_val=int(len(val_idx)),
            n_train_pos=n_train_pos, n_val_pos=n_val_pos,
            n_train_neg=n_train_neg, n_val_neg=n_val_neg,
            train_series_count=int(np.unique(groups[train_idx]).size),
            val_series_count=int(np.unique(groups[val_idx]).size),
        )

        below_floor = min(n_train_pos, n_val_pos, n_train_neg, n_val_neg) < FLOOR_PER_FOLD_PER_CLASS
        if below_floor:
            reason = (f"below {FLOOR_PER_FOLD_PER_CLASS}/fold/class floor "
                      f"(train pos/neg = {n_train_pos}/{n_train_neg}, "
                      f"val pos/neg = {n_val_pos}/{n_val_neg})")
            logger.warning(f"  fold {i}: SKIP — {reason}")
            fold_results.append(FoldResult(
                **common, fit_seconds=None, skipped=True, reason=reason,
                roc_auc=None, pr_auc=None, accuracy=None, sensitivity=None,
                specificity=None, f1=None, brier=None,
            ))
            continue

        # Pipeline: standardization is fit on train fold only.
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf",   LogisticRegression(max_iter=2000, solver="lbfgs",
                                         class_weight="balanced", random_state=seed)),
        ])

        t0 = time.perf_counter()
        pipe.fit(X[train_idx], y_train)
        fit_seconds = round(time.perf_counter() - t0, 3)
        y_prob = pipe.predict_proba(X[val_idx])[:, 1]

        metrics = _fold_metrics(y_val, y_prob)
        logger.info(f"  fold {i}: ROC-AUC={metrics['roc_auc']:.3f}, "
                    f"PR-AUC={metrics['pr_auc']:.3f}, F1={metrics['f1']:.3f}, "
                    f"acc={metrics['accuracy']:.3f} "
                    f"(n_train={len(train_idx)}, n_val={len(val_idx)}, fit={fit_seconds}s)")
        fold_results.append(FoldResult(
            **common, fit_seconds=fit_seconds, skipped=False, reason=None,
            **{k: v for k, v in metrics.items()},
        ))
    return fold_results


def summarize(fold_results: list[FoldResult]) -> dict:
    ran = [r for r in fold_results if not r.skipped]
    keys = ("roc_auc", "pr_auc", "accuracy", "sensitivity", "specificity", "f1", "brier")
    summary: dict = {"n_folds_ran": len(ran), "n_folds_skipped": len(fold_results) - len(ran)}
    for k in keys:
        vals = np.asarray([getattr(r, k) for r in ran if getattr(r, k) is not None], dtype=float)
        summary[f"{k}_mean"] = float(np.mean(vals)) if vals.size else None
        summary[f"{k}_std"]  = float(np.std(vals))  if vals.size else None
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Linear probe over cached embeddings (steps 4/5).")
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="Path to embeddings_{variant}.parquet.")
    parser.add_argument("--pool", required=True, choices=["neg_whole_corpus", "neg_hard"],
                        help="Which negative pool to run against.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260707)
    args = parser.parse_args(argv)

    logger = _log()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = _load_frame(args.embeddings, args.pool, logger)
    X, y, groups = _feature_matrix(df)
    logger.info(f"X.shape={X.shape}, n_positive={int(y.sum())}, "
                f"n_series={np.unique(groups).size}")

    fold_results = run_cv(X, y, groups, args.k_folds, args.seed, logger)
    summary = summarize(fold_results)

    variant_name = args.embeddings.stem.replace("embeddings_", "")
    payload = {
        "variant": variant_name,
        "negative_pool": args.pool,
        "n_samples": int(len(df)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(df) - y.sum()),
        "n_series": int(np.unique(groups).size),
        "seed": args.seed,
        "k_folds": args.k_folds,
        "summary": summary,
        "folds": [asdict(r) for r in fold_results],
    }
    (args.outdir / "probe_results.json").write_text(json.dumps(payload, indent=2))
    logger.info(f"wrote {args.outdir / 'probe_results.json'}")

    # Also flatten a one-row CSV for aggregation across (variant, pool) later.
    row = {"variant": variant_name, "negative_pool": args.pool, **summary}
    pd.DataFrame([row]).to_csv(args.outdir / "probe_results.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
