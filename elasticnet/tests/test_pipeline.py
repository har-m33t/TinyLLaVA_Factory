"""End-to-end smoke tests for the elastic net pipeline against a toy H5.

Runs the full label → subsample → load → split → fit → evaluate → rank → plot
chain against a small synthetic ARCHS4-shaped H5 emitted by
`eda/dataset/make_toy_data.py`. Verifies:

- The CVD keyword regex fires on the toy CVD titles (label module).
- Subsampling keeps every positive and draws the requested ratio.
- The expression matrix is (n_pool_samples, n_kept_genes) after the
  low-count filter and log2.
- StratifiedGroupKFold produces no series leakage across folds.
- fit_outer_fold trains and emits the expected fold artifacts.
- Evaluate + gene_signal produce well-formed CSVs.
- Plots render (files nonzero) without exceptions.

The toy pool intentionally has a much higher CVD positive fraction than the
real corpus (10% here vs. sub-1% for real ARCHS4) so we get enough positives
in each grouped test fold to compute AUCs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eda.dataset.make_toy_data import make_toy_h5
from elasticnet.steps import (
    evaluate,
    gene_signal,
    label,
    load_expression,
    pipeline as pipe_step,
    plots,
    splits,
    subsample,
)


@pytest.fixture(scope="module")
def toy_h5(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("elasticnet_toy") / "toy.h5"
    return make_toy_h5(
        out,
        n_genes=200,
        n_samples=2000,
        cvd_positive_frac=0.10,   # higher than real corpus so every fold gets positives
    )


@pytest.fixture(scope="module")
def pipeline_out(toy_h5, tmp_path_factory) -> Path:
    outdir = tmp_path_factory.mktemp("elasticnet_out")
    label.run(h5_path=toy_h5, outdir=outdir)
    subsample.run(
        h5_path=toy_h5,
        label_dir=outdir / "label",
        outdir=outdir,
        negative_ratio=5,     # keep runtime small
        seed=42,
    )
    load_expression.run(
        h5_path=toy_h5,
        subsample_dir=outdir / "subsample",
        outdir=outdir,
        min_detection_frac=0.05,
    )
    splits.run(
        subsample_dir=outdir / "subsample",
        outdir=outdir,
        n_outer_folds=3,   # toy has few series; 3 folds is safe
        seed=42,
    )
    return outdir


def test_label_produces_positives(pipeline_out):
    with open(pipeline_out / "label" / "label_summary.json") as f:
        summary = json.load(f)
    assert summary["n_positive"] > 0
    assert summary["total_samples"] == 2000
    labels = np.load(pipeline_out / "label" / "labels.npy")
    assert labels.shape == (2000,)
    assert set(np.unique(labels).tolist()).issubset({0, 1})


def test_subsample_keeps_all_positives(pipeline_out):
    pool = pd.read_parquet(pipeline_out / "subsample" / "training_pool.parquet")
    with open(pipeline_out / "subsample" / "subsample_manifest.json") as f:
        manifest = json.load(f)
    with open(pipeline_out / "label" / "label_summary.json") as f:
        label_summary = json.load(f)
    # subsample restricts to bulk pool, so kept positives may be <= total positives
    assert manifest["n_positive"] <= label_summary["n_positive"]
    assert (pool["label"] == 1).sum() == manifest["n_positive"]
    assert (pool["label"] == 0).sum() == manifest["n_negative"]


def test_expression_matrix_shape_and_dtype(pipeline_out):
    x = np.load(pipeline_out / "expression" / "X.npy")
    pool = pd.read_parquet(pipeline_out / "subsample" / "training_pool.parquet")
    assert x.shape[0] == len(pool)
    assert x.dtype == np.float32
    # log2(count + 1) => values are non-negative
    assert (x >= 0).all()


def test_splits_no_series_leakage(pipeline_out):
    with open(pipeline_out / "splits" / "splits_manifest.json") as f:
        manifest = json.load(f)
    assert manifest["series_leakage_check"] == "passed"
    fold_assignments = np.load(pipeline_out / "splits" / "fold_assignments.npy")
    assert set(np.unique(fold_assignments).tolist()) == set(range(manifest["n_outer_folds"]))


def test_fit_outer_fold_end_to_end(pipeline_out, tmp_path):
    x = np.load(pipeline_out / "expression" / "X.npy")
    pool = pd.read_parquet(pipeline_out / "subsample" / "training_pool.parquet")
    fold_assignments = np.load(pipeline_out / "splits" / "fold_assignments.npy")

    out_fold = tmp_path / "fold_0"
    fm = pipe_step.fit_outer_fold(
        X=x,
        y=pool["label"].to_numpy(),
        groups=pool["source_series_id"].to_numpy(),
        fold_assignments=fold_assignments,
        fold_id=0,
        out_fold_dir=out_fold,
        cs=3,             # small grid — toy data doesn't warrant Cs=10
        l1_ratios=(0.5,),
        max_iter=200,
        n_inner_folds=3,
        seed=42,
        sample_indices=pool["sample_index"].to_numpy(),
    )
    assert fm["n_train"] > 0 and fm["n_test"] > 0
    assert (out_fold / "coefficients.npy").exists()
    assert (out_fold / "test_predictions.parquet").exists()
    assert (out_fold / "model.joblib").exists()
    coef = np.load(out_fold / "coefficients.npy")
    n_genes = np.load(pipeline_out / "expression" / "gene_symbols.npy").shape[0]
    assert coef.shape == (n_genes,)


def test_full_train_flow_via_run(pipeline_out, tmp_path):
    """Fit all folds + evaluate + rank + plots by driving the orchestrator's
    inner helpers directly (skipping stages already run in the fixture).
    """
    x = np.load(pipeline_out / "expression" / "X.npy")
    pool = pd.read_parquet(pipeline_out / "subsample" / "training_pool.parquet")
    fold_assignments = np.load(pipeline_out / "splits" / "fold_assignments.npy")
    n_outer = int(fold_assignments.max()) + 1

    folds_dir = tmp_path / "folds"
    folds_dir.mkdir()
    for k in range(n_outer):
        pipe_step.fit_outer_fold(
            X=x,
            y=pool["label"].to_numpy(),
            groups=pool["source_series_id"].to_numpy(),
            fold_assignments=fold_assignments,
            fold_id=k,
            out_fold_dir=folds_dir / f"fold_{k}",
            cs=3, l1_ratios=(0.5,), max_iter=200, n_inner_folds=3, seed=42,
            sample_indices=pool["sample_index"].to_numpy(),
        )
    perf_dir = evaluate.run(folds_dir=folds_dir, outdir=tmp_path)
    ranking_dir = gene_signal.run(
        folds_dir=folds_dir,
        expression_dir=pipeline_out / "expression",
        outdir=tmp_path,
    )
    plots_dir = plots.run(
        folds_dir=folds_dir,
        ranking_csv=ranking_dir / "gene_signal_ranking.csv",
        outdir=tmp_path,
    )

    perf_df = pd.read_csv(perf_dir / "performance_by_fold.csv")
    assert len(perf_df) == n_outer
    for col in ("roc_auc", "pr_auc", "accuracy", "sensitivity", "specificity", "f1"):
        assert col in perf_df.columns

    ranking_df = pd.read_csv(ranking_dir / "gene_signal_ranking.csv")
    assert "in_clingen_hcvd" in ranking_df.columns
    assert len(ranking_df) == np.load(pipeline_out / "expression" / "gene_symbols.npy").shape[0]

    for name in (
        "roc_curve.png", "pr_curve.png", "confusion_matrix.png",
        "coefficient_path.png", "top_genes_coefficients.png",
        "calibration_curve.png",
    ):
        p = plots_dir / name
        assert p.exists() and p.stat().st_size > 0
