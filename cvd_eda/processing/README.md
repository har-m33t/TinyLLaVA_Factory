# Task 4 — Data Processing & Cleaning Agent

Turns raw ARCHS4/RECOUNT3 counts into an analysis-ready normalized matrix,
per dataset, so Task 6 (EDA) can run against a consistent shape.

Scope is bounded by `.claude/EDA_CLAUDE_TASKS.md` Task 4: subset to
CVD-relevant samples (Task 3 output), deduplicate, harmonize gene IDs to a
single canonical space, filter low-count genes, and normalize.

## Inputs

From Task 1 (ARCHS4 ingestion):
* `archs4_raw.h5` — HDF5 file with `/data/expression`, `/meta/samples/*`, `/meta/genes/*`.

From Task 2 (RECOUNT3 ingestion), for each project:
* `{project}_counts.parquet` — genes × samples integer counts, gene index = versioned Ensembl IDs.
* `{project}_coldata.parquet` — sample-indexed metadata.

From Task 3 (metadata curation):
* `cvd_relevance_{dataset}.csv` with required columns
  `sample_id, llm_relevance, confidence` and optional
  `matched_keyword, source_series_id`.

## Outputs

Written to `--output-dir`, one file trio per dataset:

* `cvd_matrix_{dataset}_normalized.parquet` — normalized expression, canonical
  Ensembl gene ID (versionless) as the row index, `sample_id` as columns.
* `cvd_sample_meta_{dataset}.parquet` — per-sample metadata (source series,
  title, plus `rel_*` columns propagated from the Task 3 CSV).
* `processing_log_{dataset}.json` — structured audit trail. Records the config
  used, per-step reports (with counts of samples/genes kept vs. dropped),
  outputs, and environment. Task 7 (reporting) consumes this.

`{dataset}` is `archs4` or `recount3_{project_id}` (colons are sanitized to
underscores in filenames).

## CLI

```bash
# ARCHS4
python -m cvd_eda.processing.run \
    --dataset archs4 \
    --archs4-h5 "$CVD_EDA_DATA_DIR/archs4_raw.h5" \
    --relevance-csv cvd_eda/logs/cvd_relevance_archs4.csv \
    --output-dir cvd_eda/logs/task4_out/

# RECOUNT3 — one invocation walks every project in the directory
python -m cvd_eda.processing.run \
    --dataset recount3 \
    --recount3-counts-dir cvd_eda/data/recount3_raw/ \
    --relevance-csv cvd_eda/logs/cvd_relevance_recount3.csv \
    --output-dir cvd_eda/logs/task4_out/
```

Optional overrides (defaults from `ProcessingConfig`):

| Flag | Default | Purpose |
|---|---|---|
| `--min-confidence` | `0.7` | Minimum Task 3 confidence to keep a sample. |
| `--cpm-threshold` | `1.0` | CPM cutoff for the low-count gene filter. |
| `--min-samples-per-gene-frac` | `0.2` | Fraction of samples that must clear the CPM cutoff. |
| `--min-samples-per-gene-abs` | `10` | Absolute lower bound on that count (dominates tiny cohorts). |
| `--norm-method` | `cpm_log2` | Also accepts `deseq2` (requires `pydeseq2`). `tmm` is intentionally unwired. |
| `--gene-id-map` | none | TSV `[symbol, ensembl_id]`; only required if ARCHS4 exposes symbols instead of Ensembl. |
| `--recount3-projects` | all found | Subset the RECOUNT3 sweep. |
| `-v` | off | Debug logging. |

## Pipeline

1. **CVD subset** — keep samples where `llm_relevance ∈ {"yes"}` and
   `confidence ≥ min_confidence`. Attaches `rel_*` columns onto sample_meta
   for provenance so downstream EDA/report can trace decisions.
2. **Deduplicate** — collapse repeated `sample_id`s, then drop samples whose
   full count vector matches an earlier sample byte-for-byte (SHA-1 over the
   contiguous column bytes). Cross-dataset dedup is deliberately **not** done —
   ARCHS4 and RECOUNT3 align differently and that variance is EDA signal.
3. **Harmonize gene IDs** → canonical Ensembl versionless (see below).
4. **Low-count gene filter** — keep gene `g` iff `CPM_g(s) > cpm_threshold`
   in at least `max(⌈frac·N⌉, abs)` samples.
5. **Normalize** — default `log2(CPM + 1)`; DESeq2 median-of-ratios available
   via `--norm-method deseq2`.

Every step returns a dataclass report which the `ProcessingLog` persists to
`processing_log_{dataset}.json`.

## Design decisions

### Canonical gene ID: Ensembl gene ID, versionless

Ensembl IDs are stable across HGNC symbol churn; symbols aren't. RECOUNT3
already exports Ensembl (with `.N` version suffixes). ARCHS4 v2.x exports
`ensembl_gene_id` alongside symbols. Version suffixes are stripped so
`ENSG00000141510.15` and `.16` collapse to the same locus; when two raw rows
land on the same canonical ID (paralog symbols, duplicate ARCHS4 rows) their
counts are **summed** — the DESeq2/edgeR gene-level aggregation convention.

Legacy ARCHS4 builds that are symbol-only fall back to a
`symbol → ensembl_id` TSV passed via `--gene-id-map`. No `mygene` runtime
lookup — deterministic and offline.

### Normalization: `log2(CPM + 1)` as default

* Deterministic, no R, no `pydeseq2` at import time.
* Good enough for EDA: PCA, sample-sample correlations, hierarchical
  clustering all behave sensibly on log-CPM.
* Library-composition biases (a handful of high-count genes distorting
  ratios) matter for the elastic-net stage but not for QC-level EDA. When
  that stage lands, rerun with `--norm-method deseq2` — the log records
  which method was used, so downstream stages can gate on it.
* `tmm` (edgeR via rpy2) is intentionally not wired up: it would drag R into
  what is otherwise a pure-Python step. If TMM becomes the required method,
  wrap it as a separate `--norm-method tmm` implementation calling `Rscript`.

### Deduplication semantics

Exact-vector dedup only. It catches the two failure modes we've actually
seen (repeated GSMs, reprocessed FASTQs resubmitted under new IDs). Near-
duplicate detection (correlation-threshold clustering across samples) is
EDA territory and shouldn't preempt what Task 6 is meant to surface.

## Smoke test

Fully offline — no real files needed:

```bash
python -m cvd_eda.processing.smoke_test
```

Fabricates a tiny synthetic dataset, walks it through every step, and drives
the CLI end-to-end via a temp Parquet trio. Exit code 0 = all 20 checks
passed. Use it as a change-safety net after edits.

## Module layout

```
cvd_eda/processing/
├── __init__.py           # public re-exports
├── config.py             # ProcessingConfig dataclass (all thresholds)
├── loaders.py            # ARCHS4 h5 + RECOUNT3 parquet -> RawDataset
├── gene_ids.py           # Ensembl versioning, symbol map, sum-collapse
├── processing.py         # subset / dedup / gene filter / normalize
├── logging_utils.py      # ProcessingLog JSON accumulator
├── run.py                # argparse CLI (this is the entrypoint)
├── smoke_test.py         # offline pipeline smoke test
└── README.md             # this file
```

## Non-goals

* No batch correction (Task 6 gets to look at PCA-vs-series_id before we
  decide whether ComBat / harmony are called for).
* No label assignment (that's Task 5, gated by human review).
* No dataset merging — ARCHS4 and RECOUNT3 are processed independently and
  merged (if ever) downstream, where alignment differences can be inspected.
