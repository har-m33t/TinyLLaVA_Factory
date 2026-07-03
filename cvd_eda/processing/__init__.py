"""Task 4: Data Processing & Cleaning Agent.

Public API — everything else is an implementation detail.
"""

from .config import ProcessingConfig
from .loaders import RawDataset, load_archs4, load_recount3_project
from .processing import (
    deduplicate_samples,
    filter_low_count_genes,
    normalize,
    subset_to_cvd_relevant,
)
from .gene_ids import harmonize_to_ensembl, load_symbol_to_ensembl_map

__all__ = [
    "ProcessingConfig",
    "RawDataset",
    "load_archs4",
    "load_recount3_project",
    "subset_to_cvd_relevant",
    "deduplicate_samples",
    "filter_low_count_genes",
    "normalize",
    "harmonize_to_ensembl",
    "load_symbol_to_ensembl_map",
]
