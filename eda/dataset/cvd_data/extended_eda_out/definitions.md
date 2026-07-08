# Extended EDA — definitions (§0)

This file states the patient-resolution method and its coverage %, and the
disease taxonomy used, **before** any numbers are generated in sections 1-3.

## Patient vs. sample

GEO / ARCHS4 organises metadata around **samples** (GSM accessions), not
patients. A single patient may contribute multiple samples (replicates,
timepoints, multiple tissues). There is no structured patient/subject ID field
in ARCHS4; patient identifiers, when present, live inside the free-text
`characteristics_ch1` column as `key: value` pairs.

### Resolution method

For each sample we scan `characteristics_ch1` for the first of these keys
(case-insensitive; longer variants preferred): `patient id, patientid, patient_id, subject id, subjectid, subject_id, individual id, individualid, individual_id, donor id, donorid, donor_id, patient, subject, individual, donor`.
The value after the `:` (or `=`), stripped and lowercased, is the raw patient
id. Placeholder values (`NA`, `N/A`, `none`, `unknown`, `-`, `?`, `not available`)
are treated as absent — a placeholder is worse than nothing because it would
collapse all placeholder-having samples in a study into one fake patient.

The **canonical patient key** used by every "N patients" and "samples per
patient" aggregation is:

    patient_key = f"{series_id}::{patient_id}"

Requiring same-`series_id` scoping is the important guard: patient id "1" in
GSE A has nothing to do with patient id "1" in GSE B. Cross-series patient
identity would require a linking process (name/DOB/etc.) that GEO does not
expose.

### Fallback

For samples where **no** patient key is resolvable, the fallback is
sample-as-own-patient — each such sample counts as its own patient
(`patient_key = "__unresolved__::<geo_accession>"`). This upper-bounds
patient counts and preserves per-disease sample totals. **This means every
"N patients" number reported below is dominated by the sample-as-own-patient
fallback whenever coverage % is low, and should be read alongside "samples
per patient" (mean/median/IQR) — which surfaces the true replicate structure
in the fraction of samples that do have a resolvable key.**

### Coverage on this release

Whole corpus:

| Field | Value |
|---|---:|
| Total samples | 1,098,771 |
| Samples with a resolvable patient key | 192,162 |
| Coverage % | 17.49% |
| Unique patients (resolved) | 31,655 |
| Unique patients (fallback = own sample) | 906,609 |
| Unique patients (total) | 938,264 |


Within the CVD pool (§2 pool = is_cvd_disease OR is_cvd_tissue):

| Field | Value |
|---|---:|
| CVD-pool samples | 38,072 |
| Samples with a resolvable patient key | 2,152 |
| Resolution % | 5.65% |
| Unique patients (truly resolved) | 718 |
| Unique patients (after fallback) | 36,638 |

Per subtype (post-Issue-1 bucket split):

| Subtype | N samples | N resolved samples | Resolution % | Truly-resolved patients |
|---|---:|---:|---:|---:|
| Heart failure (DCM + ICM) | 971 | 10 | 1.03% | 10 |
| Arrhythmia / AFib | 127 | 1 | 0.79% | 1 |
| Coronary artery disease | 980 | 133 | 13.57% | 38 |
| Cardiomyopathy (other) | 148 | 16 | 10.81% | 5 |
| Hypertension | 716 | 166 | 23.18% | 70 |
| Disease-matched, subtype unresolved | 7,615 | 399 | 5.24% | 124 |
| Tissue-only, disease status unconfirmed | 27,515 | 1,427 | 5.19% | 477 |


## Disease taxonomy

We use MeSH-style broad categories rather than resolving every specific
named condition — this is intentionally a landscape view, not a clinical
registry. Each sample gets one category (first match wins in a fixed
priority order); anything that fails every keyword lands in an explicit
`Unclear / unlabeled` bucket rather than silently disappearing.

Categories (priority-ordered):
  1. Cardiovascular
  2. Cancer / neoplasm
  3. Neurological
  4. Infectious
  5. Metabolic / endocrine
  6. Autoimmune
  7. Respiratory
  8. Renal
  9. Musculoskeletal
  10. Unclear / unlabeled

**First-match consequence.** Category assignment is strictly first-match in
the ordered list above. A sample whose metadata hits keywords from multiple
categories (a comorbid study — e.g. "atherosclerosis in breast cancer
patients") is always assigned to whichever category appears higher in this
order, which inflates the earlier category's count and correspondingly
undercounts every later category's count of comorbid samples.


**Comorbid overlap (this release).** 799 of 10,557 CVD-disease-matched pool samples (7.57%) ALSO matched at least one non-CVD category's keyword net; those samples are reported under `Cardiovascular` in the Section 1 table and are therefore missing from whichever later-priority category also matched.


### Genes-captured definition

`genes_detected` = count of genes with **non-zero** expression in a given
sample. Same definition as `eda/steps/qc.py` produced for the whole-corpus
QC step; we reuse that CSV rather than reintroduce a different threshold
(CPM etc.) here.

### CVD scope (used by section 2)

A sample enters the CVD pool if **either** condition holds:

* **CVD disease keyword hit** — the cardiovascular keyword list above matched
  in title, source_name_ch1, or characteristics_ch1.
* **CVD tissue keyword hit** — anatomical CVD terms (see Appendix B) matched
  in the same three fields.

This corresponds to the PI's phrasing: a sample counts as CVD if it
"comes from cardiovascular tissue, or cardiovascular disease".

## Appendix A — disease-category keyword lists

**Cardiovascular** (16 keywords):
`cardiovasc`, `cardiac`, `heart failure`, `myocardial infarct`, `coronary artery`, `atherosclerosis`, `cardiomyopathy`, `arrhythmia`, `atrial fibrillation`, `hypertension`, `ischemic heart`, `aortic`, `vascular disease`, `congestive heart`, `cardiac hypertrophy`, `cardiac fibrosis`

**Cancer / neoplasm** (18 keywords):
`cancer`, `carcinoma`, `adenocarcinoma`, `sarcoma`, `leukemia`, `lymphoma`, `melanoma`, `glioma`, `glioblastoma`, `tumor`, `tumour`, `neoplasm`, `metastasis`, `metastatic`, `malignant`, `myeloma`, `myelodysplastic`, `hepatocellular`

**Neurological** (16 keywords):
`alzheimer`, `parkinson`, `huntington`, `amyotrophic lateral sclerosis`, `als disease`, `multiple sclerosis`, `epilep`, `schizophrenia`, `autism`, `asd disorder`, `dementia`, `neurodegener`, `stroke`, `cerebral palsy`, `migraine`, `spinal cord injury`

**Infectious** (17 keywords):
`hiv`, `aids`, `hepatitis`, `tuberculosis`, `influenza`, `covid`, `sars-cov`, `sars cov`, `coronavirus`, `malaria`, `sepsis`, `bacterial infection`, `viral infection`, `pneumonia`, `ebola`, `zika`, `dengue`

**Metabolic / endocrine** (14 keywords):
`diabetes`, `diabetic`, `obesity`, `obese`, `metabolic syndrome`, `insulin resistance`, `thyroid`, `hypothyroid`, `hyperthyroid`, `polycystic ovary`, `pcos`, `cushing`, `adrenal insufficiency`, `hyperlipidemia`

**Autoimmune** (15 keywords):
`lupus`, `systemic lupus`, `rheumatoid arthritis`, `psoriasis`, `crohn`, `ulcerative colitis`, `inflammatory bowel disease`, `ibd disease`, `sjogren`, `scleroderma`, `vasculitis`, `autoimmun`, `type 1 diabetes`, `graves disease`, `hashimoto`

**Respiratory** (10 keywords):
`asthma`, `copd`, `chronic obstructive pulmonary`, `cystic fibrosis`, `pulmonary fibrosis`, `idiopathic pulmonary`, `bronchi`, `emphysema`, `sleep apnea`, `ards`

**Renal** (7 keywords):
`renal`, `kidney disease`, `chronic kidney`, `nephr`, `glomerulonephritis`, `dialysis`, `polycystic kidney`

**Musculoskeletal** (8 keywords):
`osteoarthritis`, `osteoporosis`, `arthritis`, `musculoskeletal`, `muscular dystrophy`, `sarcopenia`, `fibromyalgia`, `ankylosing spondylitis`

**Unclear / unlabeled** — sentinel: any sample that hit none of the category regexes above lands here.

## Appendix B — CVD anatomical (tissue) patterns

`heart`, `cardiac muscle`, `myocardium`, `myocardial`, `left ventricle`, `right ventricle`, `ventricular`, `atrium`, `atrial appendage`, `aorta`, `aortic`, `coronary artery`, `vascular smooth muscle`, `endothelial`, `cardiomyocyte`, `cardiovascular`

## Appendix C — CVD subtype keyword lists

**Heart failure (DCM + ICM)** (10 keywords):
`heart failure`, `congestive heart`, `hfref`, `hfpef`, `dilated cardiomyopathy`, ` dcm `, `(dcm)`, `ischemic cardiomyopathy`, ` icm `, `(icm)`

**Arrhythmia / AFib** (7 keywords):
`arrhythmia`, `atrial fibrillation`, `afib`, `a-fib`, `ventricular tachycardia`, `long qt`, `brugada`

**Coronary artery disease** (7 keywords):
`coronary artery disease`, ` cad `, `(cad)`, `coronary artery`, `myocardial infarct`, `ischemic heart`, `atherosclerosis`

**Cardiomyopathy (other)** (7 keywords):
`cardiomyopathy`, `hypertrophic cardiomyopathy`, `restrictive cardiomyopathy`, `arrhythmogenic right ventricular cardiomyopathy`, `arvc`, `cardiac hypertrophy`, `cardiac fibrosis`

**Hypertension** (3 keywords):
`hypertension`, `hypertensive`, `high blood pressure`

**Disease-matched, subtype unresolved** — Non-keyword fallback for CVD-pool samples where `is_cvd_disease` is TRUE (a broad CVD disease keyword matched) but none of the specific subtype nets (heart failure, arrhythmia+AFib, CAD, cardiomyopathy-other, hypertension) matched. These samples ARE real disease-positive; only their subtype label is ambiguous.

**Tissue-only, disease status unconfirmed** — Non-keyword fallback for CVD-pool samples where `is_cvd_disease` is FALSE — the sample entered the pool ONLY via a CVD tissue keyword. Disease status is unconfirmed; these samples MUST NOT be treated as CVD-disease-positive by any downstream consumer (e.g. the linear probe stage). Includes tissue-only haystacks that happened to contain a subtype acronym (e.g. "hfref", "dcm", "cad") — the broad CVD disease net's failure means we have insufficient evidence to trust that acronym as a confirmed disease label.

## Appendix D — patient identifier key patterns

Any of these keys, case-insensitive, in `characteristics_ch1` is treated as a patient identifier:
`patient id`, `patientid`, `patient_id`, `subject id`, `subjectid`, `subject_id`, `individual id`, `individualid`, `individual_id`, `donor id`, `donorid`, `donor_id`, `patient`, `subject`, `individual`, `donor`

