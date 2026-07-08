# Extended EDA write-up (§4)

Both tables and the methods behind them, in one place. Full keyword lists and
patient-resolution details live in `definitions.md`.

## Methods, briefly

* **Patient resolution.** 17.49% of 1,098,771
  samples have a resolvable patient key in `characteristics_ch1`; the rest
  fall back to sample-as-own-patient. Same-`series_id` scoping is required to
  count two samples as the same patient. Read every "N patients" number
  alongside its "samples/patient" distribution.
* **Disease taxonomy (§1).** MeSH-style broad categories, first-match wins
  against the fixed priority order in `definitions.md` (Cardiovascular first,
  then Cancer/neoplasm, then Neurological, then Infectious, then
  Metabolic/endocrine, then Autoimmune, then Respiratory, then Renal, then
  Musculoskeletal, then the `Unclear / unlabeled` sentinel). Explicit
  `Unclear / unlabeled` bucket for samples that hit no keyword.
  **Limitation:** comorbid samples matching multiple categories are always
  counted under whichever category appears higher in this order — inflating
  earlier categories and undercounting later ones for those samples.
* **CVD scope (§2).** Union of CVD disease keyword hit OR CVD tissue keyword
  hit. Pool composition on this release: disease-only 4,207;
  tissue-only 27,515; both 6,350;
  total 38,072. Residual "other/unspecified" is split
  into `Disease-matched, subtype unresolved` and
  `Tissue-only, disease status unconfirmed` — the latter is NOT
  disease-positive and must be excluded from any downstream positive-label
  cohort.
* **Genes detected.** Reused from `eda_out/qc/qc_full_dataset.csv` (non-zero
  count definition).
* **Cross-check (§3).** Verdict: consistent. Section-1 Cardiovascular row
  N samples = 10,557 equals the
  label table's `is_cvd_disease` count = 10,557;
  section-2 Total CVD N samples = 38,072
  equals the label table's `is_cvd_pool` count = 38,072,
  which is the disease count plus 27,515
  tissue-only additions.

## Section 1 — Whole-dataset breakdown

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

## Section 2 — CVD-only breakdown

| CVD subtype | N patients (post-fallback) | N samples | Samples/patient (mean, median, IQR) | Genes detected/sample (mean, median, IQR) | N series | Resolution % | N patients (truly resolved) |
|---|---:|---:|---|---|---:|---:|---:|
| Heart failure (DCM + ICM) | 971 | 971 | mean 1.0, median 1.0, IQR [1.0, 1.0] | mean 27188.8, median 31844.0, IQR [23195.5, 36421.0] | 60 | 1.03% | 10 |
| Arrhythmia / AFib | 127 | 127 | mean 1.0, median 1.0, IQR [1.0, 1.0] | mean 35028.9, median 33014.0, IQR [30913.5, 39950.0] | 13 | 0.79% | 1 |
| Coronary artery disease | 885 | 980 | mean 1.1, median 1.0, IQR [1.0, 1.0] | mean 32612.8, median 30845.5, IQR [26774.0, 40198.2] | 82 | 13.57% | 38 |
| Cardiomyopathy (other) | 137 | 148 | mean 1.1, median 1.0, IQR [1.0, 1.0] | mean 33023.0, median 33403.5, IQR [30519.8, 37364.8] | 21 | 10.81% | 5 |
| Hypertension | 620 | 716 | mean 1.2, median 1.0, IQR [1.0, 1.0] | mean 33263.2, median 33524.5, IQR [29580.8, 37034.8] | 36 | 23.18% | 70 |
| Disease-matched, subtype unresolved | 7,340 | 7,615 | mean 1.0, median 1.0, IQR [1.0, 1.0] | mean 26542.8, median 28827.0, IQR [19712.0, 32804.0] | 392 | 5.24% | 124 |
| Tissue-only, disease status unconfirmed | 26,565 | 27,515 | mean 1.0, median 1.0, IQR [1.0, 1.0] | mean 21026.8, median 25063.0, IQR [6823.0, 31884.0] | 1,434 | 5.19% | 477 |
| Total CVD | 36,638 | 38,072 | mean 1.0, median 1.0, IQR [1.0, 1.0] | mean 22908.9, median 26796.5, IQR [9110.2, 32543.2] | — | 5.65% | 718 |


CSV: `section2_cvd_breakdown/cvd_disease_breakdown.csv`.

## Notable flags for downstream scoping

**Subtypes too small for 5-fold stratified CV** (< 25 patients):

* **Heart failure (DCM + ICM)** — clears the 25-patient floor after fallback (971 patients over 971 samples), but only 10 patients are truly resolved (1.03% coverage). A stratified 5-fold CV over the resolved-only subset would fall under the floor; treat the post-fallback N as an upper bound only.
* **Arrhythmia / AFib** — clears the 25-patient floor after fallback (127 patients over 127 samples), but only 1 patients are truly resolved (0.79% coverage). A stratified 5-fold CV over the resolved-only subset would fall under the floor; treat the post-fallback N as an upper bound only.
* **Cardiomyopathy (other)** — clears the 25-patient floor after fallback (137 patients over 148 samples), but only 5 patients are truly resolved (10.81% coverage). A stratified 5-fold CV over the resolved-only subset would fall under the floor; treat the post-fallback N as an upper bound only.


See `definitions.md` for the full disease taxonomy, patient-key patterns,
and CVD subtype keyword lists (Appendices A-D).
