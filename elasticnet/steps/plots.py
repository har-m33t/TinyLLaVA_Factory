"""
plots.py — Task 9: graphical performance outputs.

Produces the six plots called for in `.claude/elastic_net_todo.md`:
    roc_curve.png              per-fold + mean±std ROC
    pr_curve.png               per-fold + mean PR (primary plot given imbalance)
    confusion_matrix.png       counts aggregated across folds
    coefficient_path.png       |β| vs log(C) across folds — a stability read
    top_genes_coefficients.png bar chart, colored by ClinGen HCVD membership
    calibration_curve.png      reliability diagram

All plots read only from artifacts on disk (fold predictions, aggregated
coefficient table). No fresh model fitting happens here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


TOP_N_GENES_PLOT = 30


def _fold_dirs(folds_dir: Path) -> list[Path]:
    return sorted([d for d in folds_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")])


def _read_fold_preds(fold_dirs: list[Path]) -> list[pd.DataFrame]:
    return [pd.read_parquet(d / "test_predictions.parquet") for d in fold_dirs]


def plot_roc(fold_preds: list[pd.DataFrame], out_path: Path) -> None:
    fpr_grid = np.linspace(0, 1, 200)
    interp_tprs = []
    aucs = []
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, pred in enumerate(fold_preds):
        y, s = pred["label"].to_numpy(), pred["y_score"].to_numpy()
        if y.sum() == 0 or y.sum() == len(y):
            continue
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        aucs.append(auc)
        ax.plot(fpr, tpr, alpha=0.35, lw=1.0, label=f"fold {i}: AUC={auc:.3f}")
        interp_tprs.append(np.interp(fpr_grid, fpr, tpr, left=0, right=1))
    if interp_tprs:
        mean_tpr = np.mean(interp_tprs, axis=0)
        std_tpr = np.std(interp_tprs, axis=0)
        ax.plot(fpr_grid, mean_tpr, color="black", lw=2.0,
                label=f"mean: AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}")
        ax.fill_between(fpr_grid, mean_tpr - std_tpr, mean_tpr + std_tpr, color="black", alpha=0.15)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — per-fold + mean±std")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr(fold_preds: list[pd.DataFrame], out_path: Path) -> None:
    recall_grid = np.linspace(0, 1, 200)
    interp_precs = []
    aps = []
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, pred in enumerate(fold_preds):
        y, s = pred["label"].to_numpy(), pred["y_score"].to_numpy()
        if y.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        aps.append(ap)
        # PR is not monotone in recall — sort by recall for a clean interp.
        order = np.argsort(recall)
        recall_s, precision_s = recall[order], precision[order]
        ax.plot(recall_s, precision_s, alpha=0.35, lw=1.0, label=f"fold {i}: AP={ap:.3f}")
        interp_precs.append(np.interp(recall_grid, recall_s, precision_s, left=1.0, right=0.0))
    if interp_precs:
        mean_prec = np.mean(interp_precs, axis=0)
        std_prec = np.std(interp_precs, axis=0)
        ax.plot(recall_grid, mean_prec, color="black", lw=2.0,
                label=f"mean: AP={np.mean(aps):.3f}±{np.std(aps):.3f}")
        ax.fill_between(recall_grid, mean_prec - std_prec, mean_prec + std_prec, color="black", alpha=0.15)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — primary metric given class imbalance")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(fold_preds: list[pd.DataFrame], out_path: Path) -> None:
    y_all = np.concatenate([p["label"].to_numpy() for p in fold_preds])
    p_all = np.concatenate([p["y_pred"].to_numpy() for p in fold_preds])
    cm = confusion_matrix(y_all, p_all, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["not CVD", "CVD"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix — aggregated over {len(fold_preds)} folds")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_coef_path(fold_dirs: list[Path], out_path: Path) -> None:
    """|β| distribution across folds vs the chosen C — a stability snapshot.

    We don't have a full coef-path over the LogisticRegressionCV.Cs grid
    at the fold-artifact level (that would require saving the internal
    coefs_paths_ per fold), so this plot shows the final β magnitudes
    per fold — enough for a "did every fold pick similar features?"
    read. If the paper needs the full regularization path later, save
    `coefs_paths_` from LogisticRegressionCV and plot here.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, d in enumerate(fold_dirs):
        coef = np.load(d / "coefficients.npy")
        abs_sorted = np.sort(np.abs(coef))[::-1]
        ax.plot(abs_sorted, alpha=0.6, lw=1.0, label=f"fold {i}: {int((coef != 0).sum())} nonzero")
    ax.set_yscale("log")
    ax.set_xlabel("Gene rank (by |β|)")
    ax.set_ylabel("|β| (log scale)")
    ax.set_title("Fitted coefficient magnitudes per outer fold")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top_genes(ranking_csv: Path, out_path: Path, top_n: int = TOP_N_GENES_PLOT) -> None:
    df = pd.read_csv(ranking_csv).head(top_n).iloc[::-1]  # smallest at bottom
    colors = ["#d62728" if in_c else "#4c72b0" for in_c in df["in_clingen_hcvd"]]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.25 * top_n)))
    ax.barh(df["gene_symbol"], df["mean_coef"], color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Mean coefficient across outer folds")
    ax.set_title(f"Top {top_n} genes by |mean β| — red = ClinGen HCVD member")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_calibration(fold_preds: list[pd.DataFrame], out_path: Path) -> None:
    y_all = np.concatenate([p["label"].to_numpy() for p in fold_preds])
    s_all = np.concatenate([p["y_score"].to_numpy() for p in fold_preds])
    frac_pos, mean_pred = calibration_curve(y_all, s_all, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.8, label="perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability diagram (pooled across folds)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(folds_dir: Path, ranking_csv: Path, outdir: Path) -> Path:
    out = outdir / "plots"
    out.mkdir(parents=True, exist_ok=True)

    fdirs = _fold_dirs(folds_dir)
    fold_preds = _read_fold_preds(fdirs)

    plot_roc(fold_preds, out / "roc_curve.png")
    plot_pr(fold_preds, out / "pr_curve.png")
    plot_confusion(fold_preds, out / "confusion_matrix.png")
    plot_coef_path(fdirs, out / "coefficient_path.png")
    plot_top_genes(ranking_csv, out / "top_genes_coefficients.png")
    plot_calibration(fold_preds, out / "calibration_curve.png")
    logger.info("wrote 6 plots to %s", out)
    return out
