"""
test_steps.py — smoke-level tests for the whole-corpus EDA pipeline.

Each step module in `eda/steps/` is exercised end-to-end against the toy
ARCHS4-shaped H5 produced by `eda.dataset.make_toy_data`. The tests do NOT
assert scientific correctness of results (that's what the ARCHS4-paper
methodology check in the write-up is for) — they assert:

  * every declared output file is created,
  * shapes/ranges of the outputs match what the step promises,
  * the biotype-absent path in step 6 works without crashing,
  * QC "flag, don't drop" is preserved: the QC CSV has one row per sample,
    with boolean flag columns.

Run:
    pytest eda/tests -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eda.dataset import io as archs4_io
from eda.dataset.make_toy_data import make_toy_h5
from eda.steps import clustering, cohort, dimred, gene_summary, normalize, qc

TOY_N_GENES = 500
TOY_N_SAMPLES = 2000
N_REF = 200
N_DOWNSTREAM = 400
HEATMAP_N_TEST = 150
STABILITY_N_TEST = 120


@pytest.fixture(scope="module")
def toy_h5(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("toy_h5")
    return make_toy_h5(d / "toy.h5", n_genes=TOY_N_GENES, n_samples=TOY_N_SAMPLES)


@pytest.fixture(scope="module")
def toy_h5_no_biotype(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("toy_h5_nb")
    return make_toy_h5(d / "toy_nb.h5", n_genes=TOY_N_GENES, n_samples=TOY_N_SAMPLES,
                       include_biotype=False)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("eda_out")


def test_cohort(toy_h5: Path, run_dir: Path) -> None:
    csv_path = cohort.run(toy_h5, run_dir)
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert set(df.columns) == {"grouping", "level", "count"}
    total = df[(df.grouping == "total") & (df.level == "samples")]["count"].iloc[0]
    assert int(total) == TOY_N_SAMPLES
    assert (run_dir / "cohort" / "cohort_single_cell_flag.png").exists()
    assert (run_dir / "cohort" / "cohort_samples_by_year.png").exists()


def test_qc_flags_not_drops(toy_h5: Path, run_dir: Path) -> None:
    csv_path = qc.run(toy_h5, run_dir)
    df = pd.read_csv(csv_path)
    # "flag, don't drop": row count matches sample count exactly.
    assert len(df) == TOY_N_SAMPLES
    for col in ("outlier_lib_size_lo", "outlier_lib_size_hi", "outlier_low_detection"):
        assert col in df.columns, f"missing flag column {col}"
        assert df[col].dtype == bool
    assert (df["library_size"] > 0).all()
    assert (df["genes_detected"] >= 0).all()
    assert (df["genes_detected"] <= TOY_N_GENES).all()


def test_normalize(toy_h5: Path, run_dir: Path) -> None:
    normalize.run(toy_h5, run_dir, n_ref=N_REF, n_downstream=N_DOWNSTREAM)
    nd = run_dir / "normalized"
    ref = np.load(nd / "reference_distribution.npy")
    mat = np.load(nd / "subsample_matrix.npy")
    idx = np.load(nd / "subsample_indices.npy")

    assert ref.shape == (TOY_N_GENES,)
    # Reference vector must be non-decreasing (it's a mean of sorted columns).
    assert np.all(np.diff(ref) >= -1e-9)
    assert mat.shape == (TOY_N_GENES, N_DOWNSTREAM)
    assert mat.dtype == np.float32
    # log2 output should sit in a sensible range for count data.
    assert np.isfinite(mat).all()
    assert mat.min() >= 0.0
    assert idx.shape == (N_DOWNSTREAM,)
    assert idx.min() >= 0 and idx.max() < TOY_N_SAMPLES

    manifest = json.loads((nd / "normalize_manifest.json").read_text())
    assert manifest["n_reference_samples"] == N_REF
    assert manifest["n_downstream_samples"] == N_DOWNSTREAM
    assert "seed" in manifest

    # Single-cell filter must be recorded and internally consistent.
    filt = manifest["singlecell_filter"]
    assert filt["total"] == TOY_N_SAMPLES
    assert filt["kept"] + filt["excluded"] == filt["total"]
    assert filt["threshold"] == 0.5
    # Toy generator draws sc_prob from beta(1.2, 8) so a small fraction
    # should be excluded — sanity-check the counts are non-degenerate.
    assert 0 < filt["excluded"] < TOY_N_SAMPLES

    # Every downstream sample index must come from the filtered pool.
    with archs4_io.open_h5(toy_h5) as h5:
        pool, _ = archs4_io.filter_bulk_indices(h5)
    assert np.isin(idx, pool).all()


def test_dimred(toy_h5: Path, run_dir: Path, monkeypatch) -> None:
    # Toy runs at N_DOWNSTREAM=400; shrink the stability target so the nested
    # draw is exercised (must be strictly smaller than primary N).
    monkeypatch.setattr(dimred, "N_TSNE_STABILITY", STABILITY_N_TEST)
    dimred.run(toy_h5, run_dir)
    out = run_dir / "dimred"
    for name in (
        f"tsne_sample_centric_n{N_DOWNSTREAM}.png",
        f"tsne_sample_centric_n{STABILITY_N_TEST}.png",
        "tsne_gene_centric.png",
        "pca_full.png",
        f"tsne_scores_n{N_DOWNSTREAM}.csv",
        f"tsne_scores_n{STABILITY_N_TEST}.csv",
        "tsne_gene_scores.csv",
        "pca_scores.csv",
        "pca_explained_variance_ratio.npy",
        "dimred_manifest.json",
    ):
        assert (out / name).exists(), f"missing {name}"

    tsne = pd.read_csv(out / f"tsne_scores_n{N_DOWNSTREAM}.csv")
    assert {"tsne_1", "tsne_2"}.issubset(tsne.columns)
    assert len(tsne) == N_DOWNSTREAM

    stab = pd.read_csv(out / f"tsne_scores_n{STABILITY_N_TEST}.csv")
    assert len(stab) == STABILITY_N_TEST
    # Nested draw: every stability sample_idx must appear in the primary run.
    assert set(stab["sample_idx"]).issubset(set(tsne["sample_idx"]))

    gene_tsne = pd.read_csv(out / "tsne_gene_scores.csv")
    assert len(gene_tsne) == TOY_N_GENES

    var_ratio = np.load(out / "pca_explained_variance_ratio.npy")
    assert var_ratio.shape[0] > 0
    assert (var_ratio >= 0).all()
    assert var_ratio.sum() <= 1.0 + 1e-6

    manifest = json.loads((out / "dimred_manifest.json").read_text())
    assert manifest["perplexity_sample_centric"] == 50
    assert manifest["perplexity_gene_centric"] == 30
    assert manifest["tsne_initial_dims"] == 50
    assert manifest["n_points_sample_centric_primary"] == N_DOWNSTREAM
    stab_info = manifest["n_points_sample_centric_stability"]
    assert stab_info is not None and stab_info["n_points"] == STABILITY_N_TEST


def test_clustering(run_dir: Path, monkeypatch) -> None:
    # Shrink HEATMAP_N so the test runs against the toy-scale subsample.
    monkeypatch.setattr(clustering, "HEATMAP_N", HEATMAP_N_TEST)
    clustering.run(run_dir)
    out = run_dir / "clustering"
    assert (out / "sample_correlation_heatmap.png").exists()
    Z = np.load(out / "linkage.npy")
    # scipy linkage returns (n-1, 4).
    assert Z.shape == (HEATMAP_N_TEST - 1, 4)
    heat_idx = np.load(out / "heatmap_sample_indices.npy")
    assert heat_idx.shape == (HEATMAP_N_TEST,)
    df = pd.read_csv(out / "clustering.csv")
    assert len(df) == HEATMAP_N_TEST
    assert {"sample_idx", "leaf_order"}.issubset(df.columns)

    manifest = json.loads((out / "clustering_manifest.json").read_text())
    assert manifest["heatmap_n"] == HEATMAP_N_TEST
    assert manifest["linkage_method"] == "average"

    # Nested draw invariant: every heatmap sample_idx must appear in the
    # step-3 downstream subsample (the primary N=20 000 pool at real scale).
    ds_idx = np.load(run_dir / "normalized" / "subsample_indices.npy")
    assert set(heat_idx).issubset(set(ds_idx))


def test_gene_summary_with_biotype(toy_h5: Path, run_dir: Path) -> None:
    csv_path = gene_summary.run(toy_h5, run_dir)
    df = pd.read_csv(csv_path)
    assert len(df) == TOY_N_GENES
    assert (df["detection_rate"] >= 0).all() and (df["detection_rate"] <= 1).all()
    assert "gene_biotype" in df.columns
    assert (run_dir / "gene_summary" / "gene_detection_rate_hist.png").exists()
    assert (run_dir / "gene_summary" / "gene_biotype_bar.png").exists()


def test_gene_summary_biotype_absent(toy_h5_no_biotype: Path, tmp_path: Path) -> None:
    csv_path = gene_summary.run(toy_h5_no_biotype, tmp_path)
    df = pd.read_csv(csv_path)
    assert "gene_biotype" not in df.columns
    assert (tmp_path / "gene_summary" / "gene_detection_rate_hist.png").exists()
    # No biotype figure when biotype metadata is absent.
    assert not (tmp_path / "gene_summary" / "gene_biotype_bar.png").exists()


def test_filter_bulk_indices_and_pool_draw(toy_h5: Path, tmp_path: Path) -> None:
    import h5py

    # With sc_prob present, some samples are excluded.
    with archs4_io.open_h5(toy_h5) as h5:
        pool, stats = archs4_io.filter_bulk_indices(h5)
    assert stats["total"] == TOY_N_SAMPLES
    assert stats["threshold"] == 0.5
    assert stats["kept"] == len(pool)
    assert stats["kept"] + stats["excluded"] == TOY_N_SAMPLES
    assert 0 < stats["excluded"] < TOY_N_SAMPLES

    # `subsample_from_pool` returns k unique indices from the pool, seeded.
    a = archs4_io.subsample_from_pool(pool, 100, seed=1)
    b = archs4_io.subsample_from_pool(pool, 100, seed=1)
    c = archs4_io.subsample_from_pool(pool, 100, seed=2)
    assert len(a) == 100 and len(np.unique(a)) == 100
    assert np.array_equal(a, b)          # same seed → same draw
    assert not np.array_equal(a, c)      # different seed → different draw
    assert np.isin(a, pool).all()        # subset invariant

    # Missing sc_prob field ⇒ no filter applied, note recorded. Build a
    # copy of the toy H5 and drop the field before opening read-only.
    stripped = tmp_path / "toy_no_sc.h5"
    with h5py.File(toy_h5, "r") as src, h5py.File(stripped, "w") as dst:
        def _copy(name, obj):
            if name == "meta/samples/singlecellprobability":
                return
            if isinstance(obj, h5py.Group):
                dst.create_group(name)
            else:
                dst.create_dataset(name, data=obj[()])
        src.visititems(_copy)

    with archs4_io.open_h5(stripped) as h5:
        pool_nf, stats_nf = archs4_io.filter_bulk_indices(h5)
    assert stats_nf["excluded"] == 0
    assert stats_nf["kept"] == TOY_N_SAMPLES
    assert "note" in stats_nf
    assert len(pool_nf) == TOY_N_SAMPLES


def test_quantile_norm_algebra() -> None:
    """`apply_quantile_norm` maps ranks onto the reference vector, Bolstad ties.

    Reference is [1, 2, 3, 4]; column [5, 7, 7, 9] gives ranks (0, 1, 2, 3)
    with values 7 tied at ranks 1 and 2, so both map to avg(ref[1], ref[2]) = 2.5.
    Expected: 5→ref[0]=1, 7→2.5, 7→2.5, 9→ref[3]=4.
    """
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    col = np.array([[7.0], [7.0], [5.0], [9.0]])
    out = normalize.apply_quantile_norm(col, ref)
    assert np.allclose(out.ravel(), [2.5, 2.5, 1.0, 4.0])

    # No ties — pure rank map, no averaging.
    col2 = np.array([[3.0], [1.0], [4.0], [2.0]])
    out2 = normalize.apply_quantile_norm(col2, ref)
    assert np.allclose(out2.ravel(), [3.0, 1.0, 4.0, 2.0])
