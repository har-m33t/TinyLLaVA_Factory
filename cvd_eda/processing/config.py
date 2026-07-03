"""Default thresholds and typed config for Task 4 processing.

Everything the pipeline treats as a knob lives here as a field on
:class:`ProcessingConfig`. CLI flags in ``run.py`` map 1:1 onto these fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Tuple

NormMethod = Literal["cpm_log2", "deseq2", "tmm"]


@dataclass(frozen=True)
class ProcessingConfig:
    # ---- CVD relevance subsetting (Task 3 → Task 4 handoff) ----
    min_relevance_confidence: float = 0.7
    accepted_relevance_labels: Tuple[str, ...] = ("yes",)

    # ---- Low-count gene filter ----
    # Keep gene g iff CPM_g(s) > cpm_threshold in at least
    # max(min_samples_per_gene_frac * N, min_samples_per_gene_abs) samples.
    cpm_threshold: float = 1.0
    min_samples_per_gene_frac: float = 0.2
    min_samples_per_gene_abs: int = 10

    # ---- Normalization ----
    norm_method: NormMethod = "cpm_log2"
    log_pseudocount: float = 1.0

    # ---- Canonical gene ID space ----
    # Only one value supported today; declared explicitly so it shows up in the
    # processing log and any future switch is obvious.
    canonical_gene_id: Literal["ensembl_versionless"] = "ensembl_versionless"

    def as_dict(self) -> dict:
        return asdict(self)
