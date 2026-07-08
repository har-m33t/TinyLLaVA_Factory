# Mortality prediction — status: **not runnable on this corpus**

**Step:** linear-probe stage, § 6 (per `.claude/linear_probe_todo.md`).

## What was attempted

Per TODO § 2, ran a keyword search across the 38,072-sample CVD pool
(`is_cvd_pool == True`) using the pre-scoped term list and metadata fields:

- **Terms:** `deceased`, `death`, `survival`, `vital status`, `follow-up`,
  `outcome`, `mortality`
- **Fields:** `characteristics_ch1`, `title`, `source_name_ch1`

The search was case-insensitive substring matching, same method used by the
extended-EDA labels stage (see `eda/extended_eda/labels.py`).

Full per-field, per-term counts are in `mortality_label_search_result.json`
and the per-sample hit mask is in `mortality_hits_by_sample.parquet`.

## Findings

**366 samples** (0.96% of the CVD pool) hit any term in any field. The
distribution is heavily concentrated in two terms in one field:

| Term | Any-field hits |
|---|---:|
| `death` | 316 |
| `outcome` | 79 |
| `deceased` | 0 |
| `survival` | 0 |
| `vital status` | 0 |
| `follow-up` | 0 |
| `mortality` | 0 |

The `death` and `outcome` hits both come almost entirely from
`characteristics_ch1`; `title` and `source_name_ch1` are effectively empty
for this task.

## Why this is not runnable

A term hit is not the same as a usable outcome label. To train a mortality
probe I would need per-sample `alive` / `deceased` — not just "the word
`death` appears somewhere in the characteristics blob". Actually parsing
that would require:

1. A hand-written value-parse rule per study (each series formats its
   metadata differently: `vital_status: deceased`, `outcome: dead`, `event:
   1`, etc.).
2. In many series the "death" hit is actually a reference to *cause* of
   death (e.g. `died of infection`) unrelated to a mortality outcome
   column at all.

Even after such parsing, the 25/fold/class floor from § 2 requires ≥125
samples per class per fold — i.e. ≥125 deceased AND ≥125 alive per fold.
Against 316 total `death` hits (of which many are not real vital-status
labels), that floor is essentially unreachable on this corpus without doing
the manual-curation work the TODO explicitly excludes from this stage.

## Decision recorded here

Following TODO § 6's explicit instruction: **task is marked
`not_runnable`.** No proxy label was invented. No mortality
`_by_variant.csv` is produced. Disease classification (§ 5) proceeds
independently — mortality's absence does not gate it, per the resequencing
decision at the top of the TODO.

This is a real, reportable finding about what ARCHS4's public metadata
supports — not a gap being hidden. Any future work that wants mortality
labels needs a curated outcome column upstream, not a keyword scan of
free-text.
