"""Offline smoke test — no ARCHS4/RECOUNT3 files, no Anthropic API.

Fabricates a synthetic (genes × samples) matrix with a planted batch
effect and a two-arm case/control split, then walks it through every EDA
step. Also drives the CLI end-to-end via ``--disable-llm-interpretation``
and a fabricated ``.reviewed.csv`` labels file so the review-file gate
gets exercised too.

Run::

    python -m cvd_eda.eda.smoke_test

Exit code 0 = every check passed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import cohort, confounders, qc, relationships
from .config import EDAConfig
from .loaders import LabeledDataset, load_dataset, load_labels
from . import run as cli_run


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _fake_expression(n_genes: int = 200, n_samples: int = 24) -> pd.DataFrame:
    """Synthetic log-CPM-ish matrix with two planted signals:

      * ``label``  : first half = case, second half = control. Adds +/- 0.5
                     to a chunk of genes so PCA separates them.
      * ``batch``  : alternating series1/series2. Adds +/- 1.0 to a *different*
                     chunk of genes so batch dominates PC1 by design
                     (this is the confounder-screen positive case).
    """
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=1.0, size=(n_genes, n_samples))

    # Batch effect on genes 0..60 — flip sign for odd-indexed samples.
    for j in range(n_samples):
        X[0:60, j] += 1.0 if (j % 2 == 0) else -1.0
    # Label effect on genes 60..100 — case (first half) vs. control.
    for j in range(n_samples):
        X[60:100, j] += 0.5 if j < n_samples // 2 else -0.5

    gene_ids = [f"ENSG{100000 + i:08d}" for i in range(n_genes)]
    sample_ids = [f"GSM{1000 + i}" for i in range(n_samples)]
    return pd.DataFrame(
        X,
        index=pd.Index(gene_ids, name="gene_id"),
        columns=pd.Index(sample_ids, name="sample_id"),
    )


def _fake_sample_meta(sample_ids: list[str]) -> pd.DataFrame:
    n = len(sample_ids)
    return pd.DataFrame(
        {
            "series_id": ["GSE_batch1" if i % 2 == 0 else "GSE_batch2" for i in range(n)],
            "title": [f"synthetic sample {i}" for i in range(n)],
            "sex": ["F" if i % 3 == 0 else "M" for i in range(n)],
            "tissue": ["heart_LV"] * n,
            "rel_matched_keyword": ["cardiomyopathy"] * n,
            "rel_source_series_id": [
                "GSE_batch1" if i % 2 == 0 else "GSE_batch2" for i in range(n)
            ],
        },
        index=pd.Index(sample_ids, name="sample_id"),
    )


def _fake_labels(sample_ids: list[str]) -> pd.DataFrame:
    n = len(sample_ids)
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "proposed_label": ["case"] * (n // 2) + ["control"] * (n - n // 2),
            "confidence": [0.9] * n,
            "evidence_quote": ["quote"] * n,
            "uncertain_reason": [""] * n,
            "source_series_id": [
                "GSE_batch1" if i % 2 == 0 else "GSE_batch2" for i in range(n)
            ],
            "model": ["synthetic"] * n,
            "cached": [False] * n,
        }
    )


def _make_dataset() -> LabeledDataset:
    expr = _fake_expression()
    meta = _fake_sample_meta(list(expr.columns))
    labels_df = _fake_labels(list(expr.columns))
    ds = LabeledDataset(
        name="synthetic",
        expression=expr,
        sample_meta=meta,
        n_samples_matrix=expr.shape[1],
        n_samples_labeled=expr.shape[1],
        n_samples_dropped_unlabeled=0,
    )
    # Attach labels the same way load_dataset would.
    ds.sample_meta["label"] = labels_df.set_index("sample_id")["proposed_label"].reindex(
        expr.columns
    ).values
    ds.sample_meta["confidence"] = 0.9
    return ds


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def _check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        raise AssertionError(f"smoke test check failed: {name} — {detail}")


def test_pipeline_in_memory() -> None:
    print("== in-memory pipeline ==")
    ds = _make_dataset()

    coh = cohort.summarize(ds)
    _check("cohort counts both labels", set(coh.per_label) == {"case", "control"})
    _check("cohort counts both series",
           len(coh.per_series) == 2, f"got {coh.per_series}")

    qc_report = qc.compute(ds)
    _check("qc reports per-sample library size",
           qc_report.per_sample.shape[0] == ds.expression.shape[1])

    rel = relationships.analyze(
        ds.expression,
        top_variable_genes=100,
        n_pca_components=5,
        run_tsne_flag=True,
        tsne_perplexity=5.0,
        tsne_random_state=0,
    )
    _check("PCA scores rows == n_samples",
           rel.pca.scores.shape[0] == ds.expression.shape[1])
    _check("PCA fitted at least 2 components", rel.pca.n_components >= 2)
    _check("t-SNE produced 2D embedding", rel.tsne is not None and rel.tsne.shape[1] == 2)
    _check("corr matrix is symmetric and unit-diagonal",
           bool(np.allclose(np.diag(rel.sample_corr.to_numpy()), 1.0))
           and rel.sample_corr.shape[0] == rel.sample_corr.shape[1])

    conf = confounders.screen(
        rel.pca.scores, ds.sample_meta,
        top_pcs=5, flag_threshold=0.30,
    )
    _check("confounder screen returns per-PC table",
           conf.per_pc.shape[0] == 5)
    # We planted the batch effect to dominate; series_id should show up as flagged.
    flagged_covs = {row["covariate"] for row in conf.flagged}
    _check("planted batch effect surfaced in flags",
           "series_id" in flagged_covs, f"flagged = {conf.flagged}")


def test_cli_end_to_end(tmp: Path) -> None:
    print("== CLI end-to-end (LLM disabled) ==")
    ds = _make_dataset()
    matrix_path = tmp / "cvd_matrix_synthetic_normalized.parquet"
    meta_path = tmp / "cvd_sample_meta_synthetic.parquet"
    ds.expression.to_parquet(matrix_path)
    ds.sample_meta.drop(columns=["label", "confidence"], errors="ignore").to_parquet(meta_path)

    labels = _fake_labels(list(ds.expression.columns))
    labels_path = tmp / "label_proposals_synthetic.reviewed.csv"
    labels.to_csv(labels_path, index=False)

    out_dir = tmp / "out"
    rc = cli_run.main([
        "--dataset", "synthetic",
        "--matrix", str(matrix_path),
        "--sample-meta", str(meta_path),
        "--labels", str(labels_path),
        "--output-dir", str(out_dir),
        "--disable-llm-interpretation",
        "--top-variable-genes", "100",
        "--n-pca-components", "5",
        "--tsne-perplexity", "5",
        "--confounder-flag-threshold", "0.30",
    ])
    _check("CLI exit code 0", rc == 0, f"got {rc}")

    ds_out = out_dir / "synthetic"
    log_path = ds_out / "eda_run_log_synthetic.json"
    stats_path = ds_out / "eda_summary_stats_synthetic.csv"
    plots_dir = ds_out / "eda_plots"
    _check("log written", log_path.exists())
    _check("summary stats CSV written", stats_path.exists())
    _check("plots dir created", plots_dir.is_dir())

    log = json.loads(log_path.read_text())
    _check("log records config", "config" in log and log["config"])
    _check("log records every step",
           set(log["steps"]) == {"cohort", "qc", "relationships", "confounders"},
           f"got {set(log['steps'])}")
    _check("log records outputs", "plots_dir" in log["outputs"])
    _check("log records flagged confounders (planted batch effect)",
           any(row["covariate"] == "series_id" for row in log["flagged_confounders"]))

    # The plot inventory should include one PCA per coloring column.
    plot_keys = set(log["plots"].keys())
    _check("PCA colored by label rendered", "pca_by_label" in plot_keys)
    _check("PCA colored by series_id rendered", "pca_by_series_id" in plot_keys)
    _check("confounder heatmap rendered", "confounder_screen" in plot_keys)
    _check("sample-sample correlation rendered", "sample_correlation" in plot_keys)

    # Every declared plot path should actually exist on disk.
    for name, p in log["plots"].items():
        _check(f"plot on disk: {name}", Path(p).exists(), p)

    stats = pd.read_csv(stats_path)
    _check("stats CSV has cohort rows",
           bool((stats["metric"] == "cohort").any()))
    _check("stats CSV has PCA variance rows",
           bool((stats["metric"] == "pca.explained_variance_ratio").any()))


def test_review_gate(tmp: Path) -> None:
    print("== review-file gate ==")
    # Build the fixtures once — the CLI should reject the non-reviewed path
    # *before* reading anything, but constructing the args needs valid paths.
    ds = _make_dataset()
    matrix_path = tmp / "gate_matrix.parquet"
    meta_path = tmp / "gate_meta.parquet"
    ds.expression.to_parquet(matrix_path)
    ds.sample_meta.drop(columns=["label", "confidence"], errors="ignore").to_parquet(meta_path)
    labels = _fake_labels(list(ds.expression.columns))
    bad_path = tmp / "label_proposals_synthetic.csv"      # missing .reviewed.
    good_path = tmp / "label_proposals_synthetic.reviewed.csv"
    labels.to_csv(bad_path, index=False)
    labels.to_csv(good_path, index=False)

    out_dir = tmp / "gate_out"
    try:
        cli_run.main([
            "--dataset", "gate",
            "--matrix", str(matrix_path),
            "--sample-meta", str(meta_path),
            "--labels", str(bad_path),
            "--output-dir", str(out_dir),
            "--disable-llm-interpretation",
        ])
    except SystemExit as exc:
        _check("gate rejects non-reviewed labels", exc.code == 2, f"got exit {exc.code}")
    else:
        raise AssertionError("gate did not raise SystemExit for unreviewed labels file")

    rc = cli_run.main([
        "--dataset", "gate_ok",
        "--matrix", str(matrix_path),
        "--sample-meta", str(meta_path),
        "--labels", str(good_path),
        "--output-dir", str(out_dir),
        "--disable-llm-interpretation",
        "--top-variable-genes", "100",
        "--n-pca-components", "5",
        "--tsne-perplexity", "5",
    ])
    _check("gate accepts .reviewed. labels", rc == 0)


def test_loader_and_labels(tmp: Path) -> None:
    print("== loaders + reviewed labels ==")
    ds = _make_dataset()
    matrix_path = tmp / "load_matrix.parquet"
    meta_path = tmp / "load_meta.parquet"
    ds.expression.to_parquet(matrix_path)
    ds.sample_meta.drop(columns=["label", "confidence"], errors="ignore").to_parquet(meta_path)

    labels = _fake_labels(list(ds.expression.columns)[:-2])  # drop last 2 → unlabeled
    labels_path = tmp / "load_labels.reviewed.csv"
    labels.to_csv(labels_path, index=False)

    labels_df = load_labels(labels_path)
    loaded = load_dataset("load_test", matrix_path, meta_path, labels_df)
    _check("loader dropped unlabeled samples",
           loaded.n_samples_dropped_unlabeled == 2,
           f"got {loaded.n_samples_dropped_unlabeled}")
    _check("loaded frame carries label column",
           "label" in loaded.sample_meta.columns)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_pipeline_in_memory()
        test_loader_and_labels(tmp)
        test_cli_end_to_end(tmp)
        test_review_gate(tmp)
    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
