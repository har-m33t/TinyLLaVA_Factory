"""Task 5 — Labeling Agent (case/control/subtype). ⚠️ Human-review checkpoint.

Consumes Task 3's ``cvd_relevance_{dataset}.csv`` (only rows the LLM classifier
called ``yes`` with high confidence) plus, when useful, the parent GEO series
description, and proposes an outcome label per sample.

The output — ``label_proposals.csv`` — is a *proposal* file. Task 6 must not
consume it until a human reviewer has walked the flagged rows.

See :mod:`cvd_eda.labeling.README` and :file:`.claude/EDA_CLAUDE_TASKS.md`.
"""

from cvd_eda.labeling.schema import LabelProposal, LABEL_VOCAB, CSV_COLUMNS

__all__ = ["LabelProposal", "LABEL_VOCAB", "CSV_COLUMNS"]
