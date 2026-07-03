"""Task 7 — Reporting Agent.

Aggregates the logs and output tables emitted by Tasks 1-6 into a single
Markdown report (``cvd_eda_report.md``) with a go/no-go recommendation for
the elastic-net stage.

The heavy lifting is deterministic — numbers, tables, and rule-based
diagnostics — so the report is reproducible even without an LLM. An
optional LLM synthesis pass adds a short executive summary at the top when
``ANTHROPIC_API_KEY`` is set; the deterministic body is unchanged either way.

See :mod:`cvd_eda.reporting.README` and :file:`.claude/EDA_CLAUDE_TASKS.md`.
"""

from cvd_eda.reporting.schema import (
    DecisionVerdict,
    ReportInputs,
    ReportPayload,
)

__all__ = ["DecisionVerdict", "ReportInputs", "ReportPayload"]
