"""Offline smoke test — no real ARCHS4/RECOUNT3 files required.

Fabricates a tiny dataset in memory and walks it through every stage of
:mod:`cvd_eda.processing`, then also drives the CLI (:mod:`run`) via
synthetic Parquet inputs written to a temp directory.

Run::

    python -m cvd_eda.processing.smoke_test
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ProcessingConfig
from .gene_ids import harmonize_to_ensembl, strip_ensembl_version
from .loaders import RawDataset, load_recount3_project
from .processing import (
    deduplicate_samples,
    filter_low_count_genes,
    normalize,
    subset_to_cvd_relevant,
)
from . import run as cli_run


def _fake_dataset(n_genes: int = 20, n_samples: int = 12) -> RawDataset:
    rng = np.random.default_rng(42)

    # Ensembl IDs with version suffixes, plus two entries that collapse to the same
    # canonical ID (versionless) so we exercise the sum-collapse path.
    canonical = [f"ENSG{100000 + i:08d}" for i in range(n_genes)]
    versioned = [f"{g}.{rng.integers(1, 20)}" for g in canonical]
    versioned[3] = versioned[2].split(".")[0] + ".7"  # force collision on canonical
    versioned[-1] = "NOT_AN_ENSEMBL_ID"  # will be dropped as unmapped

    sample_ids = [f"GSM{1000 + i}" for i in range(n_samples)]

    counts = rng.integers(0, 500, size=(n_genes, n_samples))
    # Force one gene to be low-expression across all samples so the CPM filter drops it.
    counts[0, :] = 0
    counts_df = pd.DataFrame(
        counts,
        index=pd.Index(versioned, name="gene_id_raw"),
        columns=pd.Index(sample_ids, name="sample_id"),
    )

    # Introduce a duplicate: sample 5 == sample 4 exactly (should be caught by dedup).
    counts_df.iloc[:, 5] = counts_df.iloc[:, 4].values

    sample_meta = pd.DataFrame(
        {
            "series_id": ["GSE1"] * (n_samples // 2) + ["GSE2"] * (n_samples - n_samples // 2),
            "title": [f"sample {i}" for i in range(n_samples)],
        },
        index=pd.Index(sample_ids, name="sample_id"),
    )
    gene_meta = pd.DataFrame(
        {"ensembl_id": versioned, "symbol": [f"SYM{i}" for i in range(n_genes)]},
        index=pd.Index(versioned, name="gene_id_raw"),
    )
    return RawDataset(
        name="synthetic",
        counts=counts_df,
        sample_meta=sample_meta,
        gene_meta=gene_meta,
        gene_id_scheme="ensembl",
    )


def _fake_relevance(sample_ids: list[str]) -> pd.DataFrame:
    """First 8 samples are yes/high-conf; next 2 uncertain; last 2 yes/low-conf."""
    rows = []
    for i, sid in enumerate(sample_ids):
        if i < 8:
            rel, conf = "yes", 0.9
        elif i < 10:
            rel, conf = "uncertain", 0.5
        else:
            rel, conf = "yes", 0.4
        rows.append(
            {
                "sample_id": sid,
                "llm_relevance": rel,
                "confidence": conf,
                "matched_keyword": "cardiomyopathy",
                "source_series_id": "GSE_test",
            }
        )
    return pd.DataFrame(rows)


def _check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        raise AssertionError(f"smoke test check failed: {name} — {detail}")


def test_pipeline_in_memory() -> None:
    print("== in-memory pipeline ==")
    raw = _fake_dataset()
    relevance = _fake_relevance(list(raw.counts.columns))
    cfg = ProcessingConfig(
        min_relevance_confidence=0.7,
        cpm_threshold=1.0,
        min_samples_per_gene_frac=0.2,
        min_samples_per_gene_abs=2,
        norm_method="cpm_log2",
    )

    counts, sample_meta, sub_report = subset_to_cvd_relevant(
        raw.counts, raw.sample_meta, relevance,
        min_confidence=cfg.min_relevance_confidence,
        accepted_labels=cfg.accepted_relevance_labels,
    )
    _check("subset kept 8 high-confidence yes samples",
           sub_report.n_output == 8, f"got {sub_report.n_output}")
    _check("subset attached rel_confidence to sample_meta",
           "rel_confidence" in sample_meta.columns)

    counts, sample_meta, dedup_report = deduplicate_samples(counts, sample_meta)
    _check("dedup caught the identical-vector duplicate",
           dedup_report.n_identical_vector_removed == 1,
           f"got {dedup_report.n_identical_vector_removed}")
    _check("dedup output has 7 samples", counts.shape[1] == 7)

    harm = harmonize_to_ensembl(counts, raw.gene_meta.loc[counts.index], "ensembl")
    _check("harmonize dropped the non-Ensembl row",
           harm.n_unmapped == 1, f"got {harm.n_unmapped}")
    _check("harmonize sum-collapsed the version-collision row",
           harm.n_duplicate_canonical == 1, f"got {harm.n_duplicate_canonical}")
    _check("harmonized index has no version suffix",
           all(strip_ensembl_version(g) == g for g in harm.counts.index))

    counts, gene_report = filter_low_count_genes(
        harm.counts,
        cpm_threshold=cfg.cpm_threshold,
        min_samples_frac=cfg.min_samples_per_gene_frac,
        min_samples_abs=cfg.min_samples_per_gene_abs,
    )
    _check("low-count filter dropped the all-zero gene",
           gene_report.n_kept_genes < gene_report.n_input_genes,
           f"kept {gene_report.n_kept_genes}/{gene_report.n_input_genes}")

    norm, norm_report = normalize(counts, method=cfg.norm_method, log_pseudocount=cfg.log_pseudocount)
    _check("normalize returned finite values", np.isfinite(norm.to_numpy()).all())
    _check("normalize reports cpm_log2", norm_report.method == "cpm_log2")


def test_cli_recount3(tmp: Path) -> None:
    print("== CLI recount3 end-to-end ==")
    raw = _fake_dataset()
    raw.counts.to_parquet(tmp / "SRP_TEST_counts.parquet")
    raw.sample_meta.to_parquet(tmp / "SRP_TEST_coldata.parquet")

    rel = _fake_relevance(list(raw.counts.columns))
    rel_path = tmp / "cvd_relevance_recount3.csv"
    rel.to_csv(rel_path, index=False)

    out_dir = tmp / "out"
    rc = cli_run.main([
        "--dataset", "recount3",
        "--recount3-counts-dir", str(tmp),
        "--relevance-csv", str(rel_path),
        "--output-dir", str(out_dir),
        "--min-samples-per-gene-abs", "2",
    ])
    _check("CLI exit code 0", rc == 0, f"got {rc}")

    log_path = out_dir / "processing_log_recount3_SRP_TEST.json"
    matrix_path = out_dir / "cvd_matrix_recount3_SRP_TEST_normalized.parquet"
    _check("processing log written", log_path.exists())
    _check("normalized matrix written", matrix_path.exists())

    log_data = json.loads(log_path.read_text())
    _check("log records config", "config" in log_data and log_data["config"])
    _check("log records every step",
           set(log_data["steps"]) == {
               "subset_cvd_relevant", "deduplicate", "harmonize_gene_ids",
               "filter_low_count_genes", "normalize",
           })
    _check("log outputs point at the parquet",
           log_data["outputs"]["normalized_matrix"].endswith(".parquet"))

    matrix = pd.read_parquet(matrix_path)
    _check("matrix has samples as columns", matrix.shape[1] == log_data["outputs"]["n_samples_final"])
    _check("matrix index looks like Ensembl", all(g.startswith("ENSG") for g in matrix.index))


def test_recount3_loader_transpose_recovery(tmp: Path) -> None:
    print("== RECOUNT3 loader: transposed input ==")
    raw = _fake_dataset()
    # Save transposed on purpose — loader should detect and flip back.
    raw.counts.T.to_parquet(tmp / "TR_TEST_counts.parquet")
    raw.sample_meta.to_parquet(tmp / "TR_TEST_coldata.parquet")
    loaded = load_recount3_project(
        tmp / "TR_TEST_counts.parquet", tmp / "TR_TEST_coldata.parquet"
    )
    _check("loader recovered orientation",
           list(loaded.counts.columns) == list(raw.counts.columns))
    _check("loader marks ensembl scheme", loaded.gene_id_scheme == "ensembl")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_pipeline_in_memory()
        test_cli_recount3(tmp)
        test_recount3_loader_transpose_recovery(tmp)
    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
