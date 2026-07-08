# Section 1 — Whole-dataset disease-level breakdown

**Patient resolution.** 17.49% of the 1,098,771
samples in this ARCHS4 human release have a resolvable patient/subject/donor
key in `characteristics_ch1`; the remaining samples fall back to
sample-as-own-patient (see `definitions.md`). All per-patient numbers below
should be read with that caveat in mind.

**Category assignment.** Each sample gets exactly one disease category by
first-match against the MeSH-style keyword lists in `definitions.md`.
Samples that hit no keyword are surfaced explicitly in the
`Unclear / unlabeled` row (they are not dropped).

**Priority-order caveat (limitation).** Categories are assigned by the fixed
priority order documented in `definitions.md` (Cardiovascular first, then
Cancer/neoplasm, then Neurological, …). A sample whose metadata matches
keywords from multiple categories is always assigned to whichever category
appears higher in this order, which inflates that category's row and
correspondingly undercounts every later category's count of comorbid
samples. Downstream consumers that need a comorbidity-aware view should join
back to the label table's `n_disease_categories_matched` column.

**Genes-detected definition.** Count of genes with non-zero counts in a
given sample. This column is joined in from
`eda_out/qc/qc_full_dataset.csv`, which the whole-corpus QC step already
computed using the same definition.

## Table

| Disease category | N patients (post-fallback) | N samples | Samples/patient (mean, median, IQR) | Genes detected/sample (mean, median, IQR) | N series | Resolution % | N patients (truly resolved) |
|---|---:|---:|---|---|---:|---:|---:|
| Cardiovascular | 10,080 | 10,557 | mean 1.0, median 1.0, IQR [1.0, 1.0] | mean 27814.4, median 29655.0, IQR [21683.0, 34133.0] | 549 | 6.87% | 248 |
| Cancer / neoplasm | 227,398 | 244,221 | mean 1.1, median 1.0, IQR [1.0, 1.0] | mean 25492.5, median 29013.0, IQR [15331.0, 34372.0] | 12,980 | 9.39% | 6,102 |
| Neurological | 4,411 | 4,711 | mean 1.1, median 1.0, IQR [1.0, 1.0] | mean 33807.7, median 35511.0, IQR [30176.0, 41363.5] | 313 | 13.39% | 331 |
| Infectious | 25,377 | 38,988 | mean 1.5, median 1.0, IQR [1.0, 1.0] | mean 19274.4, median 19617.5, IQR [4018.0, 33274.2] | 782 | 41.33% | 2,501 |
| Metabolic / endocrine | 23,063 | 24,810 | mean 1.1, median 1.0, IQR [1.0, 1.0] | mean 11491.9, median 6801.5, IQR [4389.2, 11365.5] | 289 | 9.59% | 632 |
| Autoimmune | 5,994 | 15,139 | mean 2.5, median 1.0, IQR [1.0, 1.0] | mean 17755.3, median 12355.0, IQR [3511.0, 31644.0] | 245 | 66.97% | 994 |
| Respiratory | 10,516 | 11,712 | mean 1.1, median 1.0, IQR [1.0, 1.0] | mean 29673.6, median 30151.5, IQR [24431.0, 36824.2] | 387 | 18.42% | 961 |
| Renal | 4,132 | 5,608 | mean 1.4, median 1.0, IQR [1.0, 1.0] | mean 15383.4, median 6762.5, IQR [3270.8, 29992.5] | 215 | 27.78% | 82 |
| Musculoskeletal | 970 | 1,026 | mean 1.1, median 1.0, IQR [1.0, 1.0] | mean 32175.1, median 32582.5, IQR [28133.5, 38317.8] | 79 | 7.80% | 24 |
| Unclear / unlabeled | 627,082 | 741,999 | mean 1.2, median 1.0, IQR [1.0, 1.0] | mean 22455.2, median 25557.0, IQR [9231.0, 33143.0] | 25,780 | 18.26% | 20,539 |
| Total | 938,264 | 1,098,771 | mean 1.2, median 1.0, IQR [1.0, 1.0] | mean 22855.2, median 26390.0, IQR [9264.0, 33452.0] | — | 17.49% | 31,655 |


CSV: `section1_whole_breakdown/whole_dataset_disease_breakdown.csv`.
