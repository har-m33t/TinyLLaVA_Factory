# Linear-probe stage — label definitions (§ step 2)

This file records the label decisions locked in for the frozen-encoder
linear-probe evaluation. See `.claude/linear_probe_todo.md` for the
prescriptive tasks and the rationale.

## Bulk-only filter

Applied uniformly to positives + both negative pools: `singlecellprobability < 0.5`.
Total samples in corpus: **1,098,771**.
Retained after bulk-only filter: **779,009**.
CVD pool (`is_cvd_pool`, pre-bulk filter): **38,072**.

## Positive pool — disease classification

6 disease-confirmed CVD subtypes, per TODO's locked decision:

| Subtype | N (bulk-only) |
|---|---:|
| `disease_matched_subtype_unresolved` | 5,915 |
| `coronary_artery_disease` | 945 |
| `heart_failure` | 921 |
| `hypertension` | 679 |
| `cardiomyopathy_other` | 143 |
| `arrhythmia_afib` | 122 |
| **Total positives** | **8,725** |

Distinct positive `series_id`s: **480** — this is the pool that
gets grouped by `source_series_id` in every downstream StratifiedGroupKFold.

### 25/fold/class floor check (k=5)

TODO §2 requires that a labelled class have enough samples for at
least 25/fold. At k=5 that means N ≥ 125 per class. Result:

| Subtype | N | ≥ 125? |
|---|---:|:-:|
| `disease_matched_subtype_unresolved` | 5,915 | ✅ |
| `coronary_artery_disease` | 945 | ✅ |
| `heart_failure` | 921 | ✅ |
| `hypertension` | 679 | ✅ |
| `cardiomyopathy_other` | 143 | ✅ |
| `arrhythmia_afib` | 122 | ⚠️ |

The task runs on the aggregated 6-subtype positive pool (binary label:
confirmed CVD vs. negative). Per-subtype breakouts are reported downstream
as slices, not separate CV runs.

## Negative pools — reported separately

TODO §2 option (c): run both negative pools, report separately.

### (a) whole-corpus non-CVD

`~is_cvd_pool AND n_disease_categories_matched==0 AND is_bulk`. N = **487,226**.

Matches the elastic-net stage's negative pool logic — direct comparability
with that baseline. Not manually curated; label noise is expected (same
limitation as the elastic-net stage).

### (b) tissue-only hard negatives

`cvd_subtype == "tissue_only_disease_unconfirmed" AND is_bulk`. N = **22,307**.

CVD-relevant tissue but no confirmed disease keyword hit. These are the
"hard" negatives — samples that share tissue-of-origin with positives but
lack the disease signal. The extended-EDA review explicitly said this
bucket is NOT a positive label; here it's used as a hard-negative comparison,
which is the second option the TODO offers.

### Down-sampling ratios

Both pools are far larger than the positive pool (8,725). Downstream
step 3 caps each negative pool at `--neg-ratio × n_positives` (default 3×,
matching the elastic-net stage's ratio) with a fixed seed, grouped-by-
series_id to preserve fold integrity.

## Mortality label — search result

Ran keyword search over CVD-pool samples (38,072).
Terms: `deceased, death, survival, vital status, follow-up, outcome, mortality`. Fields: `characteristics_ch1, title, source_name_ch1`.

**Samples with any-term-any-field hit: 366** (0.961%).

Per-term hit counts across fields:

| Term | Samples with a hit |
|---|---:|
| `deceased` | 0 |
| `death` | 316 |
| `survival` | 0 |
| `vital status` | 0 |
| `follow-up` | 0 |
| `outcome` | 79 |
| `mortality` | 0 |

See `mortality_label_search_result.json` for the per-field breakdown and
`mortality_hits_by_sample.parquet` for the per-sample mask. Whether the
task is runnable (25/fold/class floor) is decided by step 6 after the
extraction of actual outcome values, not just the existence of a term hit.

## Reproducibility

- Bulk filter threshold: `singlecellprobability < 0.5` (unchanged from EDA/elasticnet).
- All positive/negative definitions derive from `eda/dataset/cvd_data/extended_eda_out/labels/sample_labels.parquet` + the H5.
- No manual curation, no negation rules — same standard as the elastic-net stage.
