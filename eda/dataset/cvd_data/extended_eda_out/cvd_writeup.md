# Section 2 — CVD-only breakdown by subtype

**Pool definition.** A sample enters the CVD pool if it matched either the
cardiovascular disease keyword net **or** a cardiovascular tissue term
(heart, aorta, coronary artery, etc.). This union captures the PI's phrasing:
CVD includes samples that "come from cardiovascular tissue, or cardiovascular
disease". See `definitions.md` Appendices A, B for the exact lists.

**Pool composition on this release**

| Route into the CVD pool | N samples |
|---|---:|
| CVD disease keyword only | 4,207 |
| CVD tissue keyword only | 27,515 |
| Both keyword and tissue | 6,350 |
| **Total pool** | **38,072** |

**Two fallback buckets — do NOT collapse them.** After Issue 1, the residual
"other/unspecified" bucket is split into two, based on whether the sample
matched a cardiovascular disease keyword at all:

* **Disease-matched, subtype unresolved** — `is_cvd_disease` is TRUE but no
  specific subtype keyword net (heart failure, arrhythmia+AFib, CAD,
  cardiomyopathy-other, hypertension) matched. These are a real
  disease-positive subset with an ambiguous subtype label.
* **Tissue-only, disease status unconfirmed** — the sample entered the CVD
  pool ONLY through a cardiovascular tissue keyword; `is_cvd_disease` is
  FALSE. **These samples must NOT be treated as CVD-disease-positive by any
  downstream consumer** (in particular, the linear probe stage that reads
  this write-up when deciding what counts as a positive label). They belong
  in negative / unlabeled / control cohorts, not in the disease-positive
  cohort.

**Patient resolution.** Whole corpus: 17.49% of the
1,098,771 samples have a resolvable patient key. Within the CVD
pool specifically: 5.65% (2,152 of 38,072 samples;
718 truly-resolved patients). Per-patient numbers within the CVD pool
follow the same convention as section 1 — read alongside samples-per-patient
and the truly-resolved column.

**Category-priority caveat.** The whole-dataset Section 1 table pins
Cardiovascular first in the priority order (see `definitions.md`). Any
CVD-disease-matched sample that ALSO matched a non-CVD category's keyword net
is counted under `Cardiovascular` in Section 1, suppressing that non-CVD
category's count for those comorbid samples. See
`cvd_pool_composition.json → comorbidity_with_non_cvd_categories` for the
per-release count.

## Table

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



**Small-cohort watch (truly-resolved patient counts).** The following subtypes fall below the 25-patient floor when counting only samples with a resolvable `characteristics_ch1` patient key (rather than the post-fallback N patients reported in the table). Read this alongside the table:

* **Heart failure (DCM + ICM)** — 10 truly-resolved patients (out of 971 samples, 971 post-fallback patients).
* **Arrhythmia / AFib** — 1 truly-resolved patients (out of 127 samples, 127 post-fallback patients).
* **Cardiomyopathy (other)** — 5 truly-resolved patients (out of 148 samples, 137 post-fallback patients).

CSV: `section2_cvd_breakdown/cvd_disease_breakdown.csv`.
