"""
dimred.py — Task 4: dimensionality reduction, replicating ARCHS4's own paper.

Deliverables (under `<outdir>/dimred/`):
    tsne_sample_centric_n20000.png  primary sample t-SNE (perplexity=50, N=20 000)
    tsne_sample_centric_n5000.png   stability check t-SNE at N=5 000 (nested draw)
    tsne_gene_centric.png           gene t-SNE (perplexity=30)
    pca_full.png                    scree + PC1/PC2 scatter
    tsne_scores_n20000.csv          per-sample t-SNE 2D coords (primary run)
    tsne_scores_n5000.csv           per-sample t-SNE 2D coords (stability run)
    tsne_gene_scores.csv            per-gene t-SNE 2D coords (gene-centric)
    pca_scores.csv                  per-sample PCA scores + explained variance

Parameter fidelity
------------------
The ARCHS4 paper (Lachmann et al. 2018, Methods) uses `Rtsne` with:
    - sample-centric embedding: perplexity = 50
    - gene-centric embedding:   perplexity = 30
Both are computed on the quantile-normalized log2 matrix. We match those
perplexities and, for reproducibility, seed both t-SNE runs and PCA. We
substitute scikit-learn's Barnes-Hut t-SNE for `Rtsne`; both implement the
van der Maaten Barnes-Hut algorithm.

Scale caveat
------------
scikit-learn's t-SNE is O(N log N) and practical up to ~50k points. The
ARCHS4 corpus has ~700k+ samples; running t-SNE on all of them exceeds this
budget on a single workstation, so we run t-SNE on the subsample matrix
produced by step 3 (default 20k samples). This mirrors what the ARCHS4
paper's Fig. 2 did at 187k samples (also a subsample-for-visualization
choice, though at that scale the full corpus was tractable in Rtsne).
The write-up (step 7) states this substitution explicitly. PCA is
computed on the same subsample.

Stability check
---------------
Alongside the primary N=20 000 run we compute a second sample-centric
t-SNE at N=5 000 drawn as a nested random subset of the primary 20 000
sample-index pool. This is a stability check — confirming cluster
structure isn't an artifact of the specific 20K draw — not a full sweep.
Only these two sizes are run; do not add more without updating the
write-up.

Rtsne parity note
-----------------
Rtsne defaults to `initial_dims=50` — it PCA-projects the input to 50
dimensions before running Barnes-Hut t-SNE. scikit-learn's `TSNE` does NOT
do this pre-projection (its `init="pca"` only affects the *2D embedding*
init, not the input dimensionality). To match Rtsne's behaviour, we
explicitly PCA-project both the sample-centric and gene-centric inputs to
`TSNE_INITIAL_DIMS=50` before calling sklearn's `TSNE`. This is documented
in the write-up alongside the other substitutions.

Colouring the sample-centric embedding
--------------------------------------
Per the TODO, we colour by whatever grouping variables are structurally
available at whole-dataset scale: single-cell flag and submission year.
Tissue labels are unstructured free text in ARCHS4 and are handled by the
downstream (CVD-subset) EDA, not here.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ..dataset import io as archs4_io
from ..plotting import apply_style, save_figure

logger = logging.getLogger(__name__)

# ARCHS4 paper params — do NOT change without updating the write-up.
PERPLEXITY_SAMPLE = 50
PERPLEXITY_GENE = 30
TSNE_SEED = 20260705
PCA_SEED = 20260705
N_PCS = 50
# Rtsne's `initial_dims` default: PCA-project inputs to this many dims before
# Barnes-Hut t-SNE. sklearn does not do this automatically; we do it manually.
TSNE_INITIAL_DIMS = 50

# Stability-check subsample size. Drawn as a *nested* random subset of the
# primary N=20 000 pool from step 3 — same seed convention, different
# offset. Do not add more sizes without updating the Task 7 write-up.
N_TSNE_STABILITY = 5000
TSNE_STABILITY_SEED = TSNE_SEED + 1


def _load_subsample(normalized_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    mat_path = normalized_dir / "subsample_matrix.npy"
    idx_path = normalized_dir / "subsample_indices.npy"
    if not mat_path.exists() or not idx_path.exists():
        raise FileNotFoundError(
            f"expected normalized subsample at {mat_path}; run step 3 (normalize) first"
        )
    mat = np.load(mat_path)  # (n_genes, n_downstream), float32
    idx = np.load(idx_path)
    return mat, idx


def _pca_project(x: np.ndarray, n_components: int, label: str) -> np.ndarray:
    """PCA-project `x` to `n_components` dims to match Rtsne's `initial_dims`.

    If x already has fewer features than n_components, return it unchanged.
    """
    if x.shape[1] <= n_components:
        return x
    logger.info("PCA-projecting %s from %d -> %d dims (Rtsne initial_dims parity)",
                label, x.shape[1], n_components)
    pca = PCA(n_components=n_components, random_state=PCA_SEED, svd_solver="randomized")
    return pca.fit_transform(x)


def _tsne(x: np.ndarray, perplexity: int, label: str) -> np.ndarray:
    x_reduced = _pca_project(x, TSNE_INITIAL_DIMS, label)
    logger.info("running t-SNE (%s): perplexity=%d, n_points=%d, n_features=%d",
                label, perplexity, x_reduced.shape[0], x_reduced.shape[1])
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=TSNE_SEED,
        n_jobs=-1,
    )
    return tsne.fit_transform(x_reduced)


def _pca(x: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    n_components = min(n_components, min(x.shape) - 1)
    pca = PCA(n_components=n_components, random_state=PCA_SEED, svd_solver="randomized")
    scores = pca.fit_transform(x)
    return scores, pca.explained_variance_ratio_


def _sample_metadata(h5_path: Path, sample_indices: np.ndarray) -> pd.DataFrame:
    """Return the metadata needed to colour the sample-centric embedding."""
    with archs4_io.open_h5(h5_path) as h5:
        gsm = archs4_io.read_sample_field(h5, "geo_accession")
        sc = archs4_io.read_sample_field(h5, "singlecellprobability")
        sub = archs4_io.read_sample_field(h5, "submission_date")

    meta = pd.DataFrame({"sample_idx": sample_indices})
    if gsm is not None:
        meta["geo_accession"] = gsm[sample_indices]
    if sc is not None:
        meta["singlecellprobability"] = np.asarray(sc, dtype=float)[sample_indices]
    if sub is not None:
        parsed = pd.to_datetime(pd.Series(sub[sample_indices]), errors="coerce")
        meta["submission_year"] = parsed.dt.year.to_numpy()
    return meta


def run(
    h5_path: Path,
    outdir: Path,
    *,
    perplexity_sample: int = PERPLEXITY_SAMPLE,
    perplexity_gene: int = PERPLEXITY_GENE,
) -> Path:
    """Run PCA + sample/gene t-SNE.

    Perplexity overrides are exposed so the CVD-subset orchestrator can
    scale down at subset scale (a few hundred samples don't tolerate the
    whole-corpus perplexity=50 well). Defaults reproduce the ARCHS4-paper
    values for the whole-corpus run.
    """
    apply_style()
    out = outdir / "dimred"
    out.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()
    mat, ds_idx = _load_subsample(outdir / "normalized")
    logger.info("loaded normalized subsample: shape=%s", mat.shape)

    # Sample-centric primary run (N = ds_idx size, typically 20 000).
    # Matrix on disk is (n_genes, n_samples); transpose for sklearn's
    # row-per-observation convention.
    x_sample = mat.T  # (n_downstream, n_genes)
    n_primary = x_sample.shape[0]

    tsne_sample = _tsne(x_sample, perplexity_sample, f"sample-centric N={n_primary}")
    meta = _sample_metadata(h5_path, ds_idx)
    tsne_df = meta.copy()
    tsne_df["tsne_1"] = tsne_sample[:, 0]
    tsne_df["tsne_2"] = tsne_sample[:, 1]
    tsne_df.to_csv(out / f"tsne_scores_n{n_primary}.csv", index=False)
    _plot_tsne_sample(tsne_df, out / f"tsne_sample_centric_n{n_primary}.png", n_primary,
                      perplexity=perplexity_sample)

    # Sample-centric stability run: nested draw of N_TSNE_STABILITY sample
    # positions out of the primary pool. Same seed convention, different
    # offset. If the primary pool is already <= N_TSNE_STABILITY (toy runs),
    # skip the stability figure — there's no smaller subset to compare to.
    stability_info = None
    if n_primary > N_TSNE_STABILITY:
        rng = np.random.default_rng(TSNE_STABILITY_SEED)
        pick = np.sort(rng.choice(n_primary, size=N_TSNE_STABILITY, replace=False))
        x_stab = x_sample[pick]
        tsne_stab = _tsne(x_stab, perplexity_sample, f"sample-centric N={N_TSNE_STABILITY} (stability)")
        stab_df = meta.iloc[pick].copy().reset_index(drop=True)
        stab_df["tsne_1"] = tsne_stab[:, 0]
        stab_df["tsne_2"] = tsne_stab[:, 1]
        stab_df.to_csv(out / f"tsne_scores_n{N_TSNE_STABILITY}.csv", index=False)
        _plot_tsne_sample(
            stab_df, out / f"tsne_sample_centric_n{N_TSNE_STABILITY}.png", N_TSNE_STABILITY,
            perplexity=perplexity_sample,
        )
        stability_info = {
            "n_points": int(N_TSNE_STABILITY),
            "seed": int(TSNE_STABILITY_SEED),
            "drawn_from": "primary N={} sample-centric pool (nested)".format(n_primary),
        }
    else:
        logger.info(
            "primary N=%d <= stability N=%d; skipping stability run", n_primary, N_TSNE_STABILITY
        )

    # PCA on the primary sample matrix.
    pca_scores, var_ratio = _pca(x_sample, n_components=N_PCS)
    pca_df = pd.DataFrame(pca_scores, columns=[f"PC{i+1}" for i in range(pca_scores.shape[1])])
    for col in meta.columns:
        pca_df[col] = meta[col].to_numpy()
    pca_df.to_csv(out / "pca_scores.csv", index=False)
    np.save(out / "pca_explained_variance_ratio.npy", var_ratio)
    _plot_pca(pca_df, var_ratio, out / "pca_full.png")

    # Gene-centric: rows = genes. Same matrix, no transpose.
    # At ~35k genes with perplexity 30, sklearn Barnes-Hut is tractable
    # (~few minutes on a workstation).
    x_gene = mat  # (n_genes, n_downstream)
    tsne_gene = _tsne(x_gene, perplexity_gene, "gene-centric")
    _write_gene_tsne(h5_path, tsne_gene, out)
    _plot_tsne_gene(tsne_gene, out / "tsne_gene_centric.png", perplexity=perplexity_gene)

    manifest = {
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "perplexity_sample_centric": perplexity_sample,
        "perplexity_gene_centric": perplexity_gene,
        "tsne_initial_dims": TSNE_INITIAL_DIMS,
        "tsne_seed": TSNE_SEED,
        "pca_seed": PCA_SEED,
        "n_pcs": N_PCS,
        "n_points_sample_centric_primary": int(n_primary),
        "n_points_sample_centric_stability": stability_info,
        "n_points_gene_centric": int(x_gene.shape[0]),
        "note": (
            "sklearn Barnes-Hut TSNE with PCA pre-projection to "
            f"{TSNE_INITIAL_DIMS} dims to match Rtsne's initial_dims default. "
            "Run on the step-3 subsample, not the full corpus — documented "
            "deviation from Lachmann et al. 2018. The stability run is a "
            "nested subset of the primary pool, drawn with a distinct seed."
        ),
    }
    with open(out / "dimred_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("dimred manifest: %s", out / "dimred_manifest.json")
    return out


def _write_gene_tsne(h5_path: Path, coords: np.ndarray, out: Path) -> None:
    with archs4_io.open_h5(h5_path) as h5:
        symbols = archs4_io.gene_symbols(h5)
    df = pd.DataFrame({
        "gene_symbol": symbols,
        "tsne_1": coords[:, 0],
        "tsne_2": coords[:, 1],
    })
    df.to_csv(out / "tsne_gene_scores.csv", index=False)


def _plot_tsne_sample(
    df: pd.DataFrame, out_path: Path, n_points: int,
    perplexity: int = PERPLEXITY_SAMPLE,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    header = f"sample-centric t-SNE (perp={perplexity}, N={n_points})"

    # Panel A: colour by single-cell probability (continuous), if available.
    if "singlecellprobability" in df.columns:
        sc = df["singlecellprobability"].to_numpy()
        sc0 = axes[0].scatter(df["tsne_1"], df["tsne_2"], c=sc, s=2, cmap="viridis", alpha=0.7)
        fig.colorbar(sc0, ax=axes[0], label="single-cell prob")
        axes[0].set_title(f"{header}\ncolour: single-cell prob")
    else:
        axes[0].scatter(df["tsne_1"], df["tsne_2"], s=2, color="#4C78A8", alpha=0.7)
        axes[0].set_title(header)

    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")

    # Panel B: colour by submission year (categorical → viridis via mapping).
    if "submission_year" in df.columns and df["submission_year"].notna().any():
        years = df["submission_year"].to_numpy(dtype=float)
        sc1 = axes[1].scatter(df["tsne_1"], df["tsne_2"], c=years, s=2, cmap="plasma", alpha=0.7)
        fig.colorbar(sc1, ax=axes[1], label="submission year")
        axes[1].set_title(f"{header}\ncolour: submission year")
    else:
        axes[1].scatter(df["tsne_1"], df["tsne_2"], s=2, color="#4C78A8", alpha=0.7)
        axes[1].set_title(header)
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")

    fig.tight_layout()
    save_figure(fig, out_path)


def _plot_tsne_gene(
    coords: np.ndarray, out_path: Path, perplexity: int = PERPLEXITY_GENE
) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(coords[:, 0], coords[:, 1], s=1, color="#4C78A8", alpha=0.5)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"gene-centric t-SNE (perp={perplexity})")
    save_figure(fig, out_path)


def _plot_pca(df: pd.DataFrame, var_ratio: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    axes[0].bar(np.arange(1, len(var_ratio) + 1), var_ratio * 100, color="#4C78A8")
    axes[0].set_xlabel("PC")
    axes[0].set_ylabel("% variance explained")
    axes[0].set_title(f"PCA scree (first {len(var_ratio)} PCs)")

    if "singlecellprobability" in df.columns:
        sc = df["singlecellprobability"].to_numpy()
        p = axes[1].scatter(df["PC1"], df["PC2"], c=sc, s=2, cmap="viridis", alpha=0.7)
        fig.colorbar(p, ax=axes[1], label="single-cell prob")
    else:
        axes[1].scatter(df["PC1"], df["PC2"], s=2, color="#4C78A8", alpha=0.7)
    axes[1].set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    axes[1].set_title("PCA — PC1 vs PC2")

    fig.tight_layout()
    save_figure(fig, out_path)
