"""Report payload dataclasses.

Every upstream artifact Task 7 consumes is loaded into a small typed record
so the aggregator, decision logic, and Markdown renderer can share a single
shape. Fields default to a "missing" state so the report tolerates any
artifact being absent — Task 7 must always produce *something* even if
Task 6 hasn't been run yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DecisionVerdict:
    GO = "go"
    CAUTION = "caution"
    NO_GO = "no-go"


# ---------------------------------------------------------------------------
# Per-artifact records
# ---------------------------------------------------------------------------


@dataclass
class Artifact:
    """Base fields every artifact record carries.

    ``available`` is False when the file could not be found; ``error`` is
    populated when it was found but failed to parse. Both are surfaced in
    the report's "Missing / broken inputs" section.
    """

    path: str = ""
    available: bool = False
    error: str | None = None


@dataclass
class IngestionArchs4(Artifact):
    release_version: str = ""
    release_url: str = ""
    checksum_ok: bool | None = None
    n_samples: int | None = None
    n_genes: int | None = None
    file_size_bytes: int | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class IngestionRecount3(Artifact):
    n_projects_ok: int = 0
    n_projects_failed: int = 0
    project_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CurationSummary(Artifact):
    dataset: str = ""
    model: str | None = None
    total_samples: int = 0
    keyword_strong: int = 0
    keyword_ambiguous: int = 0
    keyword_none: int = 0
    llm_yes: int = 0
    llm_no: int = 0
    llm_uncertain: int = 0
    flagged_below_threshold: int = 0
    confidence_threshold: float = 0.0
    # Populated when the accompanying CSV is also available.
    csv_path: str = ""
    csv_available: bool = False
    csv_yes_high_conf: int = 0
    csv_yes_low_conf: int = 0


@dataclass
class ProcessingSummary(Artifact):
    dataset: str = ""
    n_samples_final: int | None = None
    n_genes_final: int | None = None
    norm_method: str | None = None
    steps: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class LabelsSummary(Artifact):
    """Task 5 output. ``reviewed`` distinguishes the raw proposal file from
    the human-approved one — Task 6 is only supposed to consume the reviewed
    version, and the go/no-go decision hinges on this bit."""

    reviewed: bool = False
    n_rows: int = 0
    per_label: dict[str, int] = field(default_factory=dict)
    n_uncertain: int = 0
    mean_confidence: float | None = None
    # Populated from the run log if available.
    log_path: str = ""
    log_available: bool = False
    model: str | None = None


@dataclass
class EdaSummary(Artifact):
    """Task 6 output. Schema is intentionally loose because Task 6 hasn't
    been implemented yet — we surface whatever key/value pairs it emits so
    the report is still useful once it lands."""

    stats: dict[str, Any] = field(default_factory=dict)
    plot_dir: str = ""
    plot_dir_available: bool = False
    plot_files: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Aggregated payload
# ---------------------------------------------------------------------------


@dataclass
class ReportInputs:
    """All upstream artifacts, loaded and normalized. One instance per report."""

    ingestion_archs4: IngestionArchs4 = field(default_factory=IngestionArchs4)
    ingestion_recount3: IngestionRecount3 = field(default_factory=IngestionRecount3)
    curation: list[CurationSummary] = field(default_factory=list)
    processing: list[ProcessingSummary] = field(default_factory=list)
    labels: LabelsSummary = field(default_factory=LabelsSummary)
    eda: EdaSummary = field(default_factory=EdaSummary)


@dataclass
class DecisionRule:
    """One entry in the go/no-go rubric."""

    name: str
    verdict: str          # DecisionVerdict.*
    detail: str


@dataclass
class Decision:
    verdict: str = DecisionVerdict.NO_GO
    reasons: list[DecisionRule] = field(default_factory=list)


@dataclass
class ReportPayload:
    generated_at: str
    inputs_dir: str
    inputs: ReportInputs
    decision: Decision
    narrative: str = ""       # Optional LLM-written executive summary.
    narrative_source: str = ""  # "" | "llm" | "disabled" | "error:<msg>"
