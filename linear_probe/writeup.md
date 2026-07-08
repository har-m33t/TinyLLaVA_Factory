# Linear-probe stage — evaluation floor

This stage produces the "evaluation floor" number for CVD disease
classification: **a frozen BulkFormer encoder with a trainable linear probe
on top, evaluated with 5-fold `StratifiedGroupKFold` on the CVD pool from
the extended EDA.** The finding is the baseline the eventual
encoder→connector→LLM pipeline should be expected to *exceed*, not just
match.

The five BulkFormer parameter scales (37M / 50M / 93M / 127M / 147M) are
being run as separate probes to answer the scale-vs-performance question.
This document reports the **BulkFormer-37M and BulkFormer-50M** runs — the
two smallest variants — which are the only variants this CPU-only Mac can
complete in a reasonable wallclock. The remaining three variants (93M,
127M, 147M) are queued behind either a machine with GPU access or an
MPS-compatible rewrite of BulkFormer's GCNConv layer.

Deliverables per the TODO, all under `linear_probe/`:

| Step | Deliverable | State |
|---|---|---|
| 1 | `checkpoint_verification.json` | ✅ all 5 pass |
| 2 | `label_definitions.md`, `mortality_label_search_result.json` | ✅ |
| 3 | `embeddings/embeddings_BulkFormer-{37M,50M}.parquet`, `extraction_manifest.json` | ⏳ 37M+50M done |
| 4 | `probe.py` | ✅ |
| 5 | `results/disease_classification_by_variant.csv` | ⏳ 37M+50M rows |
| 6 | `results/mortality_prediction_status.md` | ✅ not-runnable |
| 7 | `results/variant_comparison.png`, `variant_comparison_table.csv` | ⏳ 37M+50M points |
| 8 | this file | ✅ |

## 1. Setup

**Sample pool.** Positives are the six disease-confirmed CVD subtypes from
the extended-EDA taxonomy (`disease_matched_subtype_unresolved` +
`coronary_artery_disease` + `heart_failure` + `hypertension` +
`cardiomyopathy_other` + `arrhythmia_afib`), filtered to
`singlecellprobability < 0.5` to drop single-cell samples. Result:
**8,725 positive samples across 480 series**.

Following TODO § 2 option (c), two negative pools were run and are reported
separately:

- **(a) whole-corpus non-CVD** — samples with `is_cvd_pool = False`,
  `n_disease_categories_matched = 0`, and the same bulk-only filter.
  Sub-sampled at 3× the positive pool (26,175 samples across 10,196
  additional series). This is the elastic-net-comparable negative pool.
- **(b) tissue-only hard negatives** — `cvd_subtype ==
  "tissue_only_disease_unconfirmed"`, bulk-only. 22,307 samples across
  1,174 series. CVD-relevant tissue but no disease-keyword confirmation —
  tests whether the encoder picks up signal beyond tissue-of-origin.

The three pools are disjoint by construction, so the union embedding
extraction covers all of them once with pool tags per sample. The
extraction manifest confirms the H5 provides all 20,010 BulkFormer vocab
genes (`mask_prob=0.0` across all batches), so no `-10` mask tokens were
needed.

Full label decisions and counts are in `label_definitions.md`.

**Grouping.** All CV splits use `StratifiedGroupKFold(n_splits=5,
groups=series_id)`, seed 20260707 — same non-negotiable grouping as the
elastic-net stage's outer CV. Without it, the probe can trivially learn
study-specific batch signatures instead of the biology we care about
(same failure mode called out in the elastic-net writeup, same fix).

**Preprocessing.** Raw ARCHS4 counts → gene-length TPM → `log1p` → align
to BulkFormer's 20,010-gene vocab. Standardization inside the probe is fit
on the train fold only per fold, matching the elastic-net stage.

**Encoder.** Frozen throughout — no gradients propagate into BulkFormer.
Sample embedding = mean-pool across the 20,010 gene tokens of the
`gene_emb_output` tensor (`[batch, 20010, dim+3]`), following the notebook's
`aggregate_type='mean'` sample-level extraction. For 37M this is a
**131-dim** sample vector.

## 2. Checkpoint verification (step 1 gate)

All five checkpoints pass the load + forward-pass check on synthetic input.
Details in `checkpoint_verification.json`. Notable calibration point: the
BulkFormer `README` names the variants "37M/50M/93M/127M/147M" including
the shared 25.6M-parameter ESM2 gene-embedding buffer that is loaded as a
constant, not trained; `sum(p.numel())` on the model alone is
consistently ~25M below the naming. All variants land within ±10% of
advertised size once the ESM2 buffer is added back.

## 3. Mortality prediction — status

**Not runnable on this corpus.** The keyword search (§ 2) hit only 366
samples (0.96% of CVD pool), dominated by the word `death` (316) and
`outcome` (79); no `deceased`, `mortality`, `vital status`, `survival`, or
`follow-up` hits at all. Even before parsing what the hits actually mean,
this is well below the 25-per-fold-per-class floor (125/class at k=5) and
would take manual per-study curation that is explicitly out of scope for
this stage. Full reasoning in
`results/mortality_prediction_status.md`.

## 4. Disease classification — BulkFormer-37M and BulkFormer-50M

5-fold grouped CV, per-fold metrics in
`results/{variant}/{pool}/probe_results.json`. Aggregate:

### vs. whole-corpus non-CVD (pool a)

| Variant | ROC-AUC | PR-AUC | Accuracy | F1 | Brier |
|---|---:|---:|---:|---:|---:|
| BulkFormer-37M | 0.925 ± 0.037 | 0.833 ± 0.075 | 0.878 ± 0.016 | 0.769 ± 0.046 | 0.091 ± 0.015 |
| BulkFormer-50M | 0.928 ± 0.036 | 0.847 ± 0.072 | 0.897 ± 0.014 | 0.801 ± 0.038 | 0.077 ± 0.012 |

All 5 folds ran in both variants (n_train ≈ 27–29K, n_val ≈ 6.6–7.3K,
n_train_pos ≈ 6,500–7,300 per fold). No fold hit the 25/class floor.

### vs. tissue-only hard negatives (pool b)

| Variant | ROC-AUC | PR-AUC | Accuracy | F1 | Brier |
|---|---:|---:|---:|---:|---:|
| BulkFormer-37M | 0.781 ± 0.105 | 0.610 ± 0.134 | 0.715 ± 0.092 | 0.583 ± 0.078 | 0.192 ± 0.061 |
| BulkFormer-50M | 0.724 ± 0.110 | 0.534 ± 0.127 | 0.679 ± 0.111 | 0.550 ± 0.063 | 0.210 ± 0.043 |

All 5 folds ran, but variance is materially higher than on pool (a) — some
folds land ROC-AUC 0.6 (37M fold 4, 50M folds 2–3), others 0.91–0.97 (both
variants' fold 1). The tissue-only hard-negative pool concentrates on
~1.6K series, so fold composition is dominated by which specific studies
land on which side, and per-series signal heterogeneity leaks straight
into the CV variance. The wide std bars are more informative here than
the means alone.

### What the two pools tell us together

The ~15 ROC-AUC-point gap between (a) 0.925 and (b) 0.781 on 37M is the
story. Against the easy negative pool, most of the discriminative signal
is almost certainly *tissue*-level (positives are cardiac tissue,
negatives are anything else) — the encoder does not need to know anything
about disease to score highly. Against the hard negatives (also cardiac
tissue), performance drops but stays clearly above chance, which is the
first evidence that the frozen embedding carries some disease-specific
signal beyond tissue-of-origin. Same story, slightly smaller gap on 50M
(0.928 vs 0.724).

The 25-per-fold-per-class floor from § 2 was cleared for both pools — the
`arrhythmia_afib` sub-class flag noted in `label_definitions.md` only
affects per-subtype breakouts, which we're not running here (the binary
task collapses all six positive subtypes).

## 5. Scale-vs-performance (2/5 variants complete)

Two data points, one direction each. Direct comparison, holding
everything else constant (same manifest, same folds, same seed, same
probe hyperparameters):

- **Pool (a) whole-corpus non-CVD:** essentially flat. 37M→50M shifts
  ROC-AUC by +0.003 and PR-AUC by +0.014, both well within a single
  fold's std. Accuracy and F1 improve by ~2–3 percentage points, which
  is the most robust signal here — suggests the 50M embedding is
  *slightly* better calibrated near the 0.5 decision threshold, without
  materially moving the ranking metrics.
- **Pool (b) tissue-only hard negatives:** point estimate goes the wrong
  way. 37M→50M drops ROC-AUC by 5.7 pts (0.781 → 0.724) and PR-AUC by
  7.6 pts (0.610 → 0.534). Std bars overlap heavily (±0.11 and ±0.13
  respectively) so this is not a statistically clean regression at
  n=5 folds, but the point estimate is worth naming rather than
  averaging away.

Neither variant clearly wins on the hard task — the natural read at
this stage is **"flat-to-slightly-worse-with-scale in the 37M→50M
range, on the hard-negative task"**, with a repeat needed to know
whether the 50M drop on pool (b) is real. If the 93M/127M/147M
variants continue the flat-or-worse pattern, that's a meaningful
finding about BulkFormer's frozen embedding for this specific task
(tissue-matched cardiac disease classification) — the extra parameters
may be capturing signal the pretraining objective doesn't align with a
disease vs. tissue distinction. If they reverse the trend, this is
fold noise at n=5.

The comparison plot at `results/variant_comparison.png` re-renders each
time `run_probes.py` picks up a new embedding parquet.

Compute reality on this Mac (CPU-only, batch=16):

| Variant | s/sample | Full pool wallclock |
|---|---:|---:|
| 37M ✅ | 0.085 | 81 min (measured) |
| 50M ✅ | 0.290 | 276 min (measured) |
| 93M | ~1.5 | ~24 h |
| 127M | ~3.0 | ~48 h |
| 147M | ~5.2 | ~83 h |

MPS is not currently a path — BulkFormer's GCNConv uses `torch_sparse` ops
that have no MPS kernel, and native `torch.sparse_coo_tensor`
construction is unimplemented in the MPS backend. Two paths open the
remaining variants: (a) run on a CUDA machine, or (b) rewrite the GCNConv
message-passing to use `scatter_add` (MPS-compatible), with correctness
verified against the CPU forward pass. Neither was done for this report.

## 6. Elastic-net baseline reference (pending)

The comparison plot supports overlaying the elastic-net stage's outer-CV
PR-AUC as a reference line. The elastic-net stage has produced the label
+ subsample + splits + expression manifests
(`eda/dataset/cvd_data/elasticnet_out/`) but no `cv_summary.json` yet, so
the reference line is currently omitted. The plotter's key hunt
(`pr_auc_mean` → `pr_auc`) will pick it up automatically once elastic-net's
`evaluate` step lands its summary file.

## 7. Reproducibility

- Seed: `20260707`, applied to StratifiedGroupKFold + LogisticRegression's
  `random_state` and to negative-pool sub-sampling.
- Encoder frozen — sample embeddings cached to parquet under
  `embeddings/`; probe never sees the raw ARCHS4 H5.
- Standardization is fit-on-train-fold-only inside the sklearn `Pipeline`.
- Full artifact set is under `linear_probe/` and referenced from this file.
- Sklearn's L-BFGS emitted `RuntimeWarning: overflow encountered in matmul`
  during a small number of iterations — the optimizer converged and the
  fold metrics are numerically sensible (finite mean/std, ROC-AUC well
  above chance). This is a nuisance warning from an intermediate
  gradient step, not a correctness signal — worth revisiting if we
  reproduce it on more variants.

## 8. Framing

Per the TODO's closing framing, these numbers are the **evaluation
floor**. Best BulkFormer variant so far (of the two we've run):

- Pool (a) whole-corpus non-CVD — **50M**: ROC-AUC 0.928, PR-AUC 0.847
- Pool (b) tissue-only hard negs — **37M**: ROC-AUC 0.781, PR-AUC 0.610

The full multimodal pipeline (encoder → connector → LLM) that comes
later should be expected to beat those on both pools — matching them
would suggest the connector + LLM aren't adding anything the linear
probe couldn't already extract from the frozen embedding. Beating
both, and by more on the hard-negative pool, is the target. Once the
remaining three variants land, the "best BulkFormer" numbers here get
updated to the actual best-of-5.
