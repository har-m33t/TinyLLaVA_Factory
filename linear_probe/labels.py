"""labels.py — step 2 of the linear-probe stage.

Builds the sample-level label frames the downstream steps depend on, and runs
the mortality-keyword search over ARCHS4 metadata so we know whether task 2
(mortality prediction) is runnable on this corpus.

Positive pool (disease classification), locked by the TODO:
    cvd_subtype ∈ {
        disease_matched_subtype_unresolved,
        coronary_artery_disease,
        heart_failure,
        hypertension,
        cardiomyopathy_other,
        arrhythmia_afib,
    }

Negative pools (per the TODO's option (c), reported separately downstream):
    (a) whole_corpus_non_cvd  — samples with is_cvd_pool=False AND
        n_disease_categories_matched=0 AND singlecellprobability<0.5.
        Matches the elastic-net stage's negative pool logic for direct
        comparability with that baseline.
    (b) tissue_only_hard_neg  — cvd_subtype == "tissue_only_disease_unconfirmed"
        (~27.5K samples). CVD-tissue but no confirmed disease keyword hit —
        tests whether the encoder picks up disease signal beyond tissue
        signal.

Mortality keyword search:
    Terms: "deceased", "death", "survival", "vital status", "follow-up",
           "outcome", "mortality"
    Fields: characteristics_ch1, title, source_name_ch1
    Scope: CVD pool only (is_cvd_pool == True)

Nothing here needs the encoders or a GPU — pure I/O + string matching.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_H5   = REPO / "eda" / "dataset" / "cvd_data" / "archs4" / "human_gene_v2.latest.h5"
EDA_LABELS   = REPO / "eda" / "dataset" / "cvd_data" / "extended_eda_out" / "labels" / "sample_labels.parquet"
DEFAULT_OUT  = HERE

POSITIVE_SUBTYPES = (
    "disease_matched_subtype_unresolved",
    "coronary_artery_disease",
    "heart_failure",
    "hypertension",
    "cardiomyopathy_other",
    "arrhythmia_afib",
)

# Field names in the ARCHS4 H5, per its meta/samples/ layout.
SC_PROB_MAX = 0.5  # same threshold used by the extended EDA + elastic-net stages

MORTALITY_TERMS = (
    "deceased", "death", "survival", "vital status", "follow-up",
    "outcome", "mortality",
)
MORTALITY_FIELDS = ("characteristics_ch1", "title", "source_name_ch1")


@dataclass(frozen=True)
class Row:
    field: str
    term: str
    n_hits: int


def _log() -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    return logging.getLogger("linear_probe.labels")


def _decode_h5_bytes(arr: np.ndarray) -> np.ndarray:
    """ARCHS4 stores strings as fixed-width bytes — decode to str, tolerating
    already-decoded arrays as well."""
    if arr.dtype == object:
        return np.asarray([x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else str(x)
                           for x in arr])
    return arr.astype(str)


def _read_sc_probability(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        return f["meta/samples/singlecellprobability"][:]


def build_probe_labels(h5_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Assemble the master probe-labels frame from the extended EDA labels
    + a bulk-only filter derived from `singlecellprobability`.

    Returns a frame indexed by `sample_index` (H5 column position, 0..N-1),
    with columns `geo_accession`, `series_id`, `cvd_subtype`, `is_cvd_pool`,
    `is_positive`, `is_neg_whole_corpus`, `is_neg_hard`, `is_bulk`.
    """
    logger.info(f"loading extended-EDA sample_labels from {EDA_LABELS}")
    labels = pd.read_parquet(EDA_LABELS)
    logger.info(f"  {len(labels):,} rows, columns={list(labels.columns)}")

    logger.info("loading singlecellprobability from ARCHS4 H5")
    sc_prob = _read_sc_probability(h5_path)
    labels = labels.copy()
    labels["singlecellprobability"] = sc_prob
    labels["is_bulk"] = labels["singlecellprobability"] < SC_PROB_MAX

    labels["is_positive"] = labels["cvd_subtype"].isin(POSITIVE_SUBTYPES) & labels["is_bulk"]
    labels["is_neg_whole_corpus"] = (
        (~labels["is_cvd_pool"])
        & (labels["n_disease_categories_matched"] == 0)
        & labels["is_bulk"]
    )
    labels["is_neg_hard"] = (labels["cvd_subtype"] == "tissue_only_disease_unconfirmed") & labels["is_bulk"]

    keep_cols = ["sample_index", "geo_accession", "series_id", "cvd_subtype", "is_cvd_pool",
                 "is_positive", "is_neg_whole_corpus", "is_neg_hard", "is_bulk",
                 "singlecellprobability"]
    return labels[keep_cols]


def _search_terms_in_field(values: np.ndarray, terms: Iterable[str]) -> dict[str, np.ndarray]:
    """Case-insensitive substring search for each term against `values`.

    Returns {term: bool_mask over values}.
    """
    lowered = np.asarray([v.lower() for v in values])
    out: dict[str, np.ndarray] = {}
    for term in terms:
        out[term] = np.asarray([term in v for v in lowered])
    return out


def mortality_search(labels: pd.DataFrame, h5_path: Path, logger: logging.Logger) -> dict:
    """Scan the CVD-pool samples' free-text metadata for mortality keywords.

    Reports per-field, per-term hit counts + a per-sample any-term-any-field
    hit mask so downstream steps can decide whether the mortality task is
    runnable.
    """
    cvd_mask = labels["is_cvd_pool"].to_numpy()
    cvd_idx = labels.loc[cvd_mask, "sample_index"].to_numpy()
    logger.info(f"scanning {cvd_idx.size:,} CVD-pool samples for mortality keywords")

    per_field_hits: dict[str, dict[str, int]] = {}
    any_term_any_field = np.zeros(cvd_idx.size, dtype=bool)
    per_term_across_fields = {t: np.zeros(cvd_idx.size, dtype=bool) for t in MORTALITY_TERMS}

    with h5py.File(h5_path, "r") as f:
        for field in MORTALITY_FIELDS:
            logger.info(f"  scanning field '{field}'")
            arr = _decode_h5_bytes(f[f"meta/samples/{field}"][:])
            arr = arr[cvd_idx]
            hits = _search_terms_in_field(arr, MORTALITY_TERMS)
            per_field_hits[field] = {term: int(mask.sum()) for term, mask in hits.items()}
            for term, mask in hits.items():
                any_term_any_field |= mask
                per_term_across_fields[term] |= mask

    n_samples_with_any_hit = int(any_term_any_field.sum())
    hit_frame = pd.DataFrame({
        "sample_index":  cvd_idx,
        "any_mortality_term_hit": any_term_any_field,
        **{f"hit_{t.replace(' ', '_').replace('-', '_')}": per_term_across_fields[t] for t in MORTALITY_TERMS},
    })
    return {
        "n_cvd_pool_samples": int(cvd_idx.size),
        "n_samples_with_any_hit": n_samples_with_any_hit,
        "pct_of_pool_with_any_hit": round(100.0 * n_samples_with_any_hit / max(cvd_idx.size, 1), 3),
        "per_field_hits": per_field_hits,
        "per_term_hits_across_fields": {t: int(per_term_across_fields[t].sum()) for t in MORTALITY_TERMS},
        "fields_searched": list(MORTALITY_FIELDS),
        "keyword_list": list(MORTALITY_TERMS),
    }, hit_frame


def _twenty_five_per_fold_class_floor(pos_counts: dict[str, int], k: int = 5) -> dict[str, bool]:
    """The step-2 guardrail: is there room for at least 25 positives per fold
    per class? A yes here is necessary (not sufficient) for the label to be
    "runnable" downstream."""
    return {label: (n // k) >= 25 for label, n in pos_counts.items()}


def write_label_definitions_md(labels: pd.DataFrame, mortality: dict,
                                pos_counts: dict[str, int], k_folds: int, out: Path,
                                logger: logging.Logger) -> None:
    n_pos = int(labels["is_positive"].sum())
    n_neg_whole = int(labels["is_neg_whole_corpus"].sum())
    n_neg_hard  = int(labels["is_neg_hard"].sum())
    n_cvd_pool  = int(labels["is_cvd_pool"].sum())
    n_bulk      = int(labels["is_bulk"].sum())
    n_pos_series = labels.loc[labels["is_positive"], "series_id"].nunique()

    lines = [
        "# Linear-probe stage — label definitions (§ step 2)",
        "",
        "This file records the label decisions locked in for the frozen-encoder",
        "linear-probe evaluation. See `.claude/linear_probe_todo.md` for the",
        "prescriptive tasks and the rationale.",
        "",
        "## Bulk-only filter",
        "",
        f"Applied uniformly to positives + both negative pools: `singlecellprobability < {SC_PROB_MAX}`.",
        f"Total samples in corpus: **{len(labels):,}**.",
        f"Retained after bulk-only filter: **{n_bulk:,}**.",
        f"CVD pool (`is_cvd_pool`, pre-bulk filter): **{n_cvd_pool:,}**.",
        "",
        "## Positive pool — disease classification",
        "",
        f"6 disease-confirmed CVD subtypes, per TODO's locked decision:",
        "",
        "| Subtype | N (bulk-only) |",
        "|---|---:|",
    ]
    for s in POSITIVE_SUBTYPES:
        lines.append(f"| `{s}` | {pos_counts[s]:,} |")
    lines += [
        f"| **Total positives** | **{n_pos:,}** |",
        "",
        f"Distinct positive `series_id`s: **{n_pos_series}** — this is the pool that",
        "gets grouped by `source_series_id` in every downstream StratifiedGroupKFold.",
        "",
        "### 25/fold/class floor check (k=5)",
        "",
        "TODO §2 requires that a labelled class have enough samples for at",
        f"least 25/fold. At k={k_folds} that means N ≥ 125 per class. Result:",
        "",
        "| Subtype | N | ≥ 125? |",
        "|---|---:|:-:|",
    ]
    for s in POSITIVE_SUBTYPES:
        floor_ok = pos_counts[s] >= 25 * k_folds
        lines.append(f"| `{s}` | {pos_counts[s]:,} | {'✅' if floor_ok else '⚠️'} |")
    lines += [
        "",
        "The task runs on the aggregated 6-subtype positive pool (binary label:",
        "confirmed CVD vs. negative). Per-subtype breakouts are reported downstream",
        "as slices, not separate CV runs.",
        "",
        "## Negative pools — reported separately",
        "",
        "TODO §2 option (c): run both negative pools, report separately.",
        "",
        "### (a) whole-corpus non-CVD",
        "",
        f"`~is_cvd_pool AND n_disease_categories_matched==0 AND is_bulk`. N = **{n_neg_whole:,}**.",
        "",
        "Matches the elastic-net stage's negative pool logic — direct comparability",
        "with that baseline. Not manually curated; label noise is expected (same",
        "limitation as the elastic-net stage).",
        "",
        "### (b) tissue-only hard negatives",
        "",
        f"`cvd_subtype == \"tissue_only_disease_unconfirmed\" AND is_bulk`. N = **{n_neg_hard:,}**.",
        "",
        "CVD-relevant tissue but no confirmed disease keyword hit. These are the",
        "\"hard\" negatives — samples that share tissue-of-origin with positives but",
        "lack the disease signal. The extended-EDA review explicitly said this",
        "bucket is NOT a positive label; here it's used as a hard-negative comparison,",
        "which is the second option the TODO offers.",
        "",
        "### Down-sampling ratios",
        "",
        f"Both pools are far larger than the positive pool ({n_pos:,}). Downstream",
        "step 3 caps each negative pool at `--neg-ratio × n_positives` (default 3×,",
        "matching the elastic-net stage's ratio) with a fixed seed, grouped-by-",
        "series_id to preserve fold integrity.",
        "",
        "## Mortality label — search result",
        "",
        f"Ran keyword search over CVD-pool samples ({mortality['n_cvd_pool_samples']:,}).",
        f"Terms: `{', '.join(MORTALITY_TERMS)}`. Fields: `{', '.join(MORTALITY_FIELDS)}`.",
        "",
        f"**Samples with any-term-any-field hit: {mortality['n_samples_with_any_hit']:,}** "
        f"({mortality['pct_of_pool_with_any_hit']}%).",
        "",
        "Per-term hit counts across fields:",
        "",
        "| Term | Samples with a hit |",
        "|---|---:|",
    ]
    for t, n in mortality["per_term_hits_across_fields"].items():
        lines.append(f"| `{t}` | {n:,} |")
    lines += [
        "",
        "See `mortality_label_search_result.json` for the per-field breakdown and",
        "`mortality_hits_by_sample.parquet` for the per-sample mask. Whether the",
        "task is runnable (25/fold/class floor) is decided by step 6 after the",
        "extraction of actual outcome values, not just the existence of a term hit.",
        "",
        "## Reproducibility",
        "",
        f"- Bulk filter threshold: `singlecellprobability < {SC_PROB_MAX}` (unchanged from EDA/elasticnet).",
        f"- All positive/negative definitions derive from `{EDA_LABELS.relative_to(REPO)}` + the H5.",
        "- No manual curation, no negation rules — same standard as the elastic-net stage.",
    ]
    out.write_text("\n".join(lines) + "\n")
    logger.info(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build linear-probe labels (step 2).")
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5, help="Path to ARCHS4 H5.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--k-folds", type=int, default=5)
    args = parser.parse_args(argv)

    logger = _log()
    args.outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    labels = build_probe_labels(args.h5, logger)
    logger.info(f"built probe-labels frame in {time.perf_counter() - t0:.1f}s "
                f"({len(labels):,} rows)")

    labels_path = args.outdir / "probe_sample_labels.parquet"
    labels.to_parquet(labels_path, index=False)
    logger.info(f"wrote {labels_path}")

    pos_counts_all = labels.loc[labels["is_positive"]] \
        .groupby("cvd_subtype").size().to_dict()
    pos_counts = {s: int(pos_counts_all.get(s, 0)) for s in POSITIVE_SUBTYPES}
    logger.info(f"positive counts by subtype: {pos_counts}")
    logger.info(f"total positive samples (bulk-only): {int(labels['is_positive'].sum()):,}")
    logger.info(f"negative pool (a) whole-corpus non-CVD (bulk-only): "
                f"{int(labels['is_neg_whole_corpus'].sum()):,}")
    logger.info(f"negative pool (b) tissue-only hard negs (bulk-only):  "
                f"{int(labels['is_neg_hard'].sum()):,}")

    t0 = time.perf_counter()
    mortality, mortality_hits = mortality_search(labels, args.h5, logger)
    logger.info(f"mortality keyword search took {time.perf_counter() - t0:.1f}s")

    mortality_json = args.outdir / "mortality_label_search_result.json"
    mortality_json.write_text(json.dumps(mortality, indent=2))
    logger.info(f"wrote {mortality_json}")

    mortality_hits_path = args.outdir / "mortality_hits_by_sample.parquet"
    mortality_hits.to_parquet(mortality_hits_path, index=False)
    logger.info(f"wrote {mortality_hits_path}")

    write_label_definitions_md(labels, mortality, pos_counts, args.k_folds,
                               args.outdir / "label_definitions.md", logger)

    floor_check = _twenty_five_per_fold_class_floor(pos_counts, args.k_folds)
    logger.info(f"25/fold/class floor by subtype: {floor_check}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
