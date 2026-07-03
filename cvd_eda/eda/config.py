"""Typed configuration for Task 6 EDA runs.

Every threshold, cap, or feature toggle the EDA pipeline treats as tunable
lives here as a field on :class:`EDAConfig`. CLI flags in ``run.py`` map
1:1 onto these fields, matching the pattern established by
:mod:`cvd_eda.processing.config`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EDAConfig:
    # ---- Label consumption ----
    # Only proposals reviewed by a human are eligible. The CLI additionally
    # enforces the ``.reviewed.`` filename convention documented in Task 5.
    min_label_confidence: float = 0.0
    accepted_label_review_status: str = "reviewed"

    # ---- Sample-relationship analyses ----
    # Cap on how many genes feed PCA / t-SNE / sample-sample correlation.
    # We take the top-variance genes on the normalized matrix, which keeps
    # runtime bounded (~10^4 genes on ~10^4 samples is manageable) without
    # meaningfully changing which components dominate variance.
    top_variable_genes: int = 5000

    n_pca_components: int = 10
    run_tsne: bool = True
    tsne_perplexity: float = 30.0
    tsne_random_state: int = 0

    # ---- Confounder screening ----
    # A PC "dominates" a covariate when its association exceeds this cutoff.
    # For categorical covariates we compute eta^2 (one-way ANOVA); for
    # continuous covariates, Pearson r^2. Threshold intentionally lenient —
    # the goal is to *flag* for the reviewer, not to gate downstream code.
    confounder_association_flag: float = 0.30
    top_pcs_for_confounder_screen: int = 5

    # ---- Sample-sample correlation heatmap ----
    # Above this sample count we downsample for readability. The full matrix
    # is still computed for the confounder screen; the plot is what shrinks.
    heatmap_sample_cap: int = 200

    # ---- Plot rendering ----
    plot_dpi: int = 150
    plot_format: str = "png"

    def as_dict(self) -> dict:
        return asdict(self)
