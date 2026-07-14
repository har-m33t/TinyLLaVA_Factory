"""
clingen.py — the ClinGen HCVD starter gene set used for post-fit validation.

Purpose
-------
After the elastic net is fit, we ask: "of the top-scoring genes, how many
are already known-CVD from an independent, curated source?" The ClinGen
Hereditary CVD (HCVD) working group's high-confidence starter set is the
reference. This is a *validation cross-check*, applied strictly after
fitting — it must never be used as a pre-filter, otherwise recovery of
"known" genes is circular.

Source
------
ClinGen Hereditary Cardiovascular Disease clinical validity summary
(https://clinicalgenome.org/). The list below is a static snapshot of the
high-confidence CVD gene panel commonly cited in the cardiogenomics
literature (e.g. Ingles et al. 2019, Hershberger et al. 2018). It is a
*starter* set — not a substitute for the live ClinGen curation — but is
sufficient for the "does the model recover known biology?" check called
for in `.claude/elastic_net_todo.md` task 10.

If a newer/more authoritative list becomes available, replace this constant
and re-run `gene_signal.py` — no other code depends on this shape.
"""

from __future__ import annotations

# High-confidence HCVD genes: hypertrophic cardiomyopathy, dilated
# cardiomyopathy, arrhythmogenic cardiomyopathy, long-QT syndrome, etc.
CLINGEN_HCVD_STARTER: tuple[str, ...] = (
    # HCM
    "MYH7", "MYBPC3", "TNNT2", "TNNI3", "TPM1", "ACTC1", "MYL2", "MYL3",
    # DCM
    "TTN", "LMNA", "RBM20", "DES", "SCN5A", "BAG3", "TNNC1", "PLN",
    # ACM / ARVC
    "PKP2", "DSP", "DSG2", "DSC2", "JUP", "TMEM43",
    # Long-QT / channelopathies
    "KCNQ1", "KCNH2", "KCNE1", "KCNE2", "CACNA1C", "CALM1", "CALM2", "CALM3",
    # Familial hypercholesterolemia (CAD risk)
    "LDLR", "APOB", "PCSK9",
    # Aortopathy
    "FBN1", "TGFBR1", "TGFBR2", "ACTA2", "MYH11", "COL3A1", "SMAD3",
    # Additional HCVD-associated
    "NKX2-5", "GATA4", "TBX5", "MEF2A",
)
