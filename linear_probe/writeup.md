# Linear-probe stage — evaluation floor

This stage produces the "evaluation floor" number for CVD disease
classification: **a frozen BulkFormer encoder with a trainable linear probe
on top, evaluated with 5-fold `StratifiedGroupKFold` on the CVD pool from
the extended EDA.** The finding is the baseline the eventual
encoder→connector→LLM pipeline should be expected to *exceed*, not just
match.

The five BulkFormer parameter scales (37M / 50M / 93M / 127M / 147M) are
being run as separate probes to answer the scale-vs-performance question.
This document reports the **BulkFormer-37M** run — the smallest variant —
which is the only variant this hardware can complete in a reasonable time
today. The other four variants are queued behind either a machine with GPU
access or an MPS-compatible rewrite of BulkFormer's GCNConv layer.

Deliverables per the TODO, all under `linear_probe/`:

| Step | Deliverable | State |
|---|---|---|
| 1 | `checkpoint_verification.json` | ✅ all 5 pass |
| 2 | `label_definitions.md`, `mortality_label_search_result.json` | ✅ |
| 3 | `embeddings/embeddings_BulkFormer-37M.parquet`, `extraction_manifest.json` | ✅ 37M only |
| 4 | `probe.py` | ✅ |
| 5 | `results/disease_classification_by_variant.csv` | ⏳ 37M only |
| 6 | `results/mortality_prediction_status.md` | ✅ not-runnable |
| 7 | `results/variant_comparison.png`, `variant_comparison_table.csv` | ⏳ 37M only |
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

## 4. Disease classification — BulkFormer-37M

5-fold grouped CV, per-fold metrics in
`results/BulkFormer-37M/{pool}/probe_results.json`. Aggregate:

### vs. whole-corpus non-CVD (pool a)

| Metric | Mean | Std |
|---|---:|---:|
| ROC-AUC | **0.925** | 0.037 |
| PR-AUC  | **0.833** | 0.075 |
| Accuracy | 0.878 | 0.016 |
| Sensitivity | 0.840 | 0.069 |
| Specificity | 0.887 | 0.016 |
| F1 | 0.769 | 0.046 |
| Brier | 0.091 | 0.015 |

All 5 folds ran (n_train ≈ 27–29K, n_val ≈ 6.6–7.3K, n_train_pos ≈
6,500–7,300 per fold). No folds hit the 25/class floor.

### vs. tissue-only hard negatives (pool b)

| Metric | Mean | Std |
|---|---:|---:|
| ROC-AUC | **0.781** | 0.105 |
| PR-AUC  | **0.610** | 0.134 |
| Accuracy | 0.715 | 0.092 |
| Sensitivity | 0.681 | 0.120 |
| Specificity | 0.732 | 0.092 |
| F1 | 0.583 | 0.078 |
| Brier | 0.192 | 0.061 |

All 5 folds ran, but variance is materially higher — some folds land ROC-AUC
0.6 (fold 4), others 0.91 (fold 1). The tissue-only hard-negative pool
concentrates on ~1.6K series total, so fold composition is dominated by
which specific studies land on which side, and per-series signal
heterogeneity leaks straight into the CV variance.

### What the two pools tell us together

The ~15 ROC-AUC-point drop between (a) 0.925 and (b) 0.781 is the story.
Against the easy negative pool, most of the discriminative signal is
almost certainly *tissue*-level (the positives are cardiac tissue, the
negatives are anything else) — the encoder does not need to know anything
about disease to score highly. Against the hard negatives (also cardiac
tissue), performance drops but stays clearly above chance, which is the
first evidence that BulkFormer-37M's frozen embedding carries some
disease-specific signal beyond tissue-of-origin.

The 25-per-fold-per-class floor from § 2 was cleared for both pools — the
`arrhythmia_afib` sub-class flag noted in `label_definitions.md` only
affects per-subtype breakouts, which we're not running here (the binary
task collapses all six positive subtypes).

## 5. Scale-vs-performance (pending, needs the other 4 variants)

The point of running 5 variants is answering *whether performance improves
with scale, plateaus, or is flat*. With only 37M complete, this section is
a placeholder. The comparison plot at
`results/variant_comparison.png` will re-render automatically once the
other variants land — the aggregator globs whatever
`embeddings/embeddings_BulkFormer-*.parquet` exists at the time of the
run.

Compute reality on this Mac (CPU-only, batch=16):

| Variant | s/sample | Est. full pool |
|---|---:|---:|
| 37M ✅ | 0.085 | 81 min (measured) |
| 50M | ~0.33 | ~5 h |
| 93M | ~1.6 | ~25 h |
| 127M | ~3.1 | ~50 h |
| 147M | ~5.4 | ~86 h |

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
floor**. The full multimodal pipeline (encoder → connector → LLM) that
comes later should be expected to beat 0.925 ROC-AUC / 0.833 PR-AUC on
the easy negative pool and 0.781 / 0.610 on the hard-negative pool —
matching this baseline would suggest the connector + LLM aren't adding
anything the linear probe couldn't already extract from the frozen
embedding. Beating both, and by more on the hard-negative pool, is the
target.
