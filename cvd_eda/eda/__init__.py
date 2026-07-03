"""Task 6 — EDA Agent.

Consumes Task 4's normalized matrix + sample metadata *and* Task 5's
human-reviewed label file, and produces the QC/EDA plot suite, a
summary-stats CSV, per-plot LLM interpretations, and a JSON audit log
that Task 7 (Reporting) will consume.

See :file:`cvd_eda/eda/README.md` for the workflow and
:file:`.claude/EDA_CLAUDE_TASKS.md` for the task brief.
"""

from cvd_eda.eda.config import EDAConfig
from cvd_eda.eda.loaders import LabeledDataset, load_dataset, load_labels

__all__ = ["EDAConfig", "LabeledDataset", "load_dataset", "load_labels"]
