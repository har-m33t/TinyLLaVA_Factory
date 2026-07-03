"""CLI entrypoint for Task 4: Data Processing & Cleaning Agent.

Usage
-----
ARCHS4::

    python -m cvd_eda.task4_processing.run \
        --dataset archs4 \
        --archs4-h5 /path/to/archs4_raw.h5 \
        --relevance-csv /path/to/cvd_relevance_archs4.csv \
        --output-dir /path/to/task4_out/ \
        [--gene-id-map /path/to/symbol_to_ensembl.tsv]

RECOUNT3 (one CLI invocation processes every project in the directory)::

    python -m cvd_eda.task4_processing.run \
        --dataset recount3 \
        --recount3-counts-dir /path/to/recount3_raw/ \
        --relevance-csv /path/to/cvd_relevance_recount3.csv \
        --output-dir /path/to/task4_out/ \
        [--recount3-projects SRP123456 GTEX_HEART ...]

Config overrides (all optional; :class:`ProcessingConfig` supplies defaults)::

    --min-confidence 0.7
    --cpm-threshold 1.0
    --min-samples-per-gene-frac 0.2
    --min-samples-per-gene-abs 10
    --norm-method cpm_log2   # or deseq2 (requires pydeseq2) / tmm (not wired up)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import ProcessingConfig
from .gene_ids import harmonize_to_ensembl, load_symbol_to_ensembl_map
from .loaders import RawDataset, load_archs4, load_recount3_project
from .logging_utils import ProcessingLog
from .processing import (
    deduplicate_samples,
    filter_low_count_genes,
    normalize,
    subset_to_cvd_relevant,
)

log = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def _safe(name: str) -> str:
    """``recount3:SRP123`` → ``recount3_SRP123`` for filesystem paths."""
    return name.replace(":", "_").replace("/", "_")


def _process_one_dataset(
    raw: RawDataset,
    relevance_df: pd.DataFrame,
    cfg: ProcessingConfig,
    symbol_map: Optional[pd.Series],
    output_dir: Path,
) -> Path:
    plog = ProcessingLog(dataset=raw.name, config=cfg.as_dict())
    plog.inputs = {
        "raw_n_samples": raw.counts.shape[1],
        "raw_n_genes": raw.counts.shape[0],
        "gene_id_scheme": raw.gene_id_scheme,
        "relevance_rows": len(relevance_df),
    }
    safe = _safe(raw.name)
    log_path = output_dir / f"processing_log_{safe}.json"

    # 1. CVD-relevance subset
    counts, sample_meta, sample_report = subset_to_cvd_relevant(
        counts=raw.counts,
        sample_meta=raw.sample_meta,
        relevance_df=relevance_df,
        min_confidence=cfg.min_relevance_confidence,
        accepted_labels=cfg.accepted_relevance_labels,
    )
    plog.add_step("subset_cvd_relevant", sample_report)
    if counts.shape[1] == 0:
        plog.add_warning(
            "No samples survived CVD-relevance subsetting; skipping remaining steps."
        )
        plog.finalize(log_path)
        return log_path

    # 2. Dedup
    counts, sample_meta, dedup_report = deduplicate_samples(counts, sample_meta)
    plog.add_step("deduplicate", dedup_report)

    # 3. Gene ID harmonization → canonical Ensembl (versionless)
    harm = harmonize_to_ensembl(
        counts=counts,
        gene_meta=raw.gene_meta.loc[counts.index],
        source_scheme=raw.gene_id_scheme,
        symbol_to_ensembl=symbol_map,
    )
    counts = harm.counts
    plog.add_step(
        "harmonize_gene_ids",
        {
            "source_scheme": raw.gene_id_scheme,
            "canonical": cfg.canonical_gene_id,
            "n_mapped": harm.n_mapped,
            "n_unmapped": harm.n_unmapped,
            "n_duplicate_canonical_collapsed": harm.n_duplicate_canonical,
        },
    )

    # 4. Low-count gene filter
    counts, gene_report = filter_low_count_genes(
        counts,
        cpm_threshold=cfg.cpm_threshold,
        min_samples_frac=cfg.min_samples_per_gene_frac,
        min_samples_abs=cfg.min_samples_per_gene_abs,
    )
    plog.add_step("filter_low_count_genes", gene_report)

    # 5. Normalize
    norm_counts, norm_report = normalize(
        counts,
        method=cfg.norm_method,
        log_pseudocount=cfg.log_pseudocount,
    )
    plog.add_step("normalize", norm_report)

    # 6. Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / f"cvd_matrix_{safe}_normalized.parquet"
    sample_meta_path = output_dir / f"cvd_sample_meta_{safe}.parquet"
    norm_counts.to_parquet(matrix_path)
    sample_meta.to_parquet(sample_meta_path)

    plog.outputs = {
        "normalized_matrix": str(matrix_path),
        "sample_metadata": str(sample_meta_path),
        "n_genes_final": norm_counts.shape[0],
        "n_samples_final": norm_counts.shape[1],
    }
    plog.finalize(log_path)
    return log_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cvd_eda.task4_processing.run",
        description="Task 4: Data Processing & Cleaning Agent for the CVD EDA workflow.",
    )
    p.add_argument("--dataset", choices=["archs4", "recount3"], required=True)
    p.add_argument(
        "--relevance-csv",
        required=True,
        type=Path,
        help="Task 3 output: cvd_relevance_{dataset}.csv",
    )
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--gene-id-map",
        type=Path,
        default=None,
        help="TSV of columns [symbol, ensembl_id]; required when the source uses symbols.",
    )

    p.add_argument("--archs4-h5", type=Path, default=None, help="Path to ARCHS4 .h5 file.")

    p.add_argument(
        "--recount3-counts-dir",
        type=Path,
        default=None,
        help="Directory of Task 2 exports: {project}_counts.parquet + {project}_coldata.parquet.",
    )
    p.add_argument(
        "--recount3-projects",
        nargs="*",
        default=None,
        help="Optional list of project IDs to process (default: everything in the directory).",
    )

    p.add_argument("--min-confidence", type=float, default=None)
    p.add_argument("--cpm-threshold", type=float, default=None)
    p.add_argument("--min-samples-per-gene-frac", type=float, default=None)
    p.add_argument("--min-samples-per-gene-abs", type=int, default=None)
    p.add_argument(
        "--norm-method", choices=["cpm_log2", "deseq2", "tmm"], default=None
    )

    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _cfg_from_args(args: argparse.Namespace) -> ProcessingConfig:
    overrides: dict = {}
    if args.min_confidence is not None:
        overrides["min_relevance_confidence"] = args.min_confidence
    if args.cpm_threshold is not None:
        overrides["cpm_threshold"] = args.cpm_threshold
    if args.min_samples_per_gene_frac is not None:
        overrides["min_samples_per_gene_frac"] = args.min_samples_per_gene_frac
    if args.min_samples_per_gene_abs is not None:
        overrides["min_samples_per_gene_abs"] = args.min_samples_per_gene_abs
    if args.norm_method is not None:
        overrides["norm_method"] = args.norm_method
    return ProcessingConfig(**overrides)


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    cfg = _cfg_from_args(args)
    symbol_map = load_symbol_to_ensembl_map(args.gene_id_map) if args.gene_id_map else None
    relevance_df = pd.read_csv(args.relevance_csv)

    if args.dataset == "archs4":
        if args.archs4_h5 is None:
            print("--archs4-h5 is required for --dataset archs4", file=sys.stderr)
            return 2
        # Pull only the candidate sample columns off disk — ARCHS4 has ~1M samples.
        candidate_ids = relevance_df["sample_id"].astype(str).tolist()
        raw = load_archs4(args.archs4_h5, sample_ids=candidate_ids)
        _process_one_dataset(raw, relevance_df, cfg, symbol_map, args.output_dir)
        return 0

    if args.dataset == "recount3":
        if args.recount3_counts_dir is None:
            print(
                "--recount3-counts-dir is required for --dataset recount3", file=sys.stderr
            )
            return 2
        counts_dir: Path = args.recount3_counts_dir
        counts_files = sorted(counts_dir.glob("*_counts.parquet"))
        projects = args.recount3_projects or [
            p.stem.replace("_counts", "") for p in counts_files
        ]
        if not projects:
            print(f"No projects found under {counts_dir}", file=sys.stderr)
            return 2
        for project_id in projects:
            counts_pq = counts_dir / f"{project_id}_counts.parquet"
            coldata_pq = counts_dir / f"{project_id}_coldata.parquet"
            if not counts_pq.exists() or not coldata_pq.exists():
                log.warning(
                    "Skipping project %s: expected files missing (%s, %s).",
                    project_id,
                    counts_pq.exists(),
                    coldata_pq.exists(),
                )
                continue
            raw = load_recount3_project(counts_pq, coldata_pq)
            _process_one_dataset(raw, relevance_df, cfg, symbol_map, args.output_dir)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
