"""Loaders for the artifacts Task 7 aggregates.

Each loader takes a directory (usually ``cvd_eda/logs/``) and returns a
populated schema record. Missing files are reported (not raised) so the
report can still be generated when the pipeline is incomplete — that's the
whole point of Task 7's "here is where we are" mode.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Iterable

from cvd_eda.reporting.schema import (
    CurationSummary,
    EdaSummary,
    IngestionArchs4,
    IngestionRecount3,
    LabelsSummary,
    ProcessingSummary,
    ReportInputs,
)


LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filename conventions (kept in one place so a rename doesn't scatter changes)
# ---------------------------------------------------------------------------

INGESTION_ARCHS4_FILE = "ingestion_log_archs4.json"
INGESTION_RECOUNT3_FILE = "ingestion_log_recount3.json"

CURATION_LOG_GLOB = "curation_log_*.json"
CURATION_CSV_GLOB = "cvd_relevance_*.csv"

PROCESSING_LOG_GLOB = "processing_log_*.json"

# Task 5 writes label_proposals.csv; the human review workflow produces a
# label_proposals.reviewed.csv sibling. We prefer the reviewed file — but
# fall back to the raw one so we can flag it as a no-go blocker.
LABEL_REVIEWED_GLOB = "label_proposals*.reviewed.csv"
LABEL_PROPOSAL_GLOB = "label_proposals*.csv"
LABEL_LOG_GLOB = "task5_run_log_*.json"

EDA_STATS_FILE = "eda_summary_stats.csv"
EDA_PLOT_DIR_CANDIDATES = ("eda_plots", "eda_plots/")


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def load_all(inputs_dir: Path) -> ReportInputs:
    """Load every upstream artifact that can be found under ``inputs_dir``."""
    return ReportInputs(
        ingestion_archs4=load_ingestion_archs4(inputs_dir),
        ingestion_recount3=load_ingestion_recount3(inputs_dir),
        curation=load_curation(inputs_dir),
        processing=load_processing(inputs_dir),
        labels=load_labels(inputs_dir),
        eda=load_eda(inputs_dir),
    )


# ---------------------------------------------------------------------------
# Task 1 — ARCHS4 ingestion
# ---------------------------------------------------------------------------


def load_ingestion_archs4(inputs_dir: Path) -> IngestionArchs4:
    path = inputs_dir / INGESTION_ARCHS4_FILE
    rec = IngestionArchs4(path=str(path))
    doc = _read_json(path, rec)
    if doc is None:
        return rec
    rec.available = True
    rec.release_version = str(doc.get("release_version", ""))
    rec.release_url = str(doc.get("release_url", ""))
    rec.checksum_ok = doc.get("checksum_ok")
    rec.n_samples = _as_int(doc.get("n_samples"))
    rec.n_genes = _as_int(doc.get("n_genes"))
    rec.file_size_bytes = _as_int(doc.get("file_size_bytes"))
    rec.notes = list(doc.get("notes") or [])
    return rec


# ---------------------------------------------------------------------------
# Task 2 — RECOUNT3 ingestion
# ---------------------------------------------------------------------------


def load_ingestion_recount3(inputs_dir: Path) -> IngestionRecount3:
    path = inputs_dir / INGESTION_RECOUNT3_FILE
    rec = IngestionRecount3(path=str(path))
    doc = _read_json(path, rec)
    if doc is None:
        return rec
    rec.available = True
    summary = doc.get("summary") or {}
    rec.n_projects_ok = _as_int(summary.get("ok_or_skipped")) or 0
    rec.n_projects_failed = _as_int(summary.get("failed")) or 0
    projects = doc.get("projects") or []
    rec.project_rows = [
        {
            "group": p.get("group"),
            "project": p.get("project"),
            "status": p.get("status"),
            "error": p.get("error"),
        }
        for p in projects
    ]
    return rec


# ---------------------------------------------------------------------------
# Task 3 — Curation
# ---------------------------------------------------------------------------


def load_curation(inputs_dir: Path) -> list[CurationSummary]:
    out: list[CurationSummary] = []
    for log_path in sorted(inputs_dir.glob(CURATION_LOG_GLOB)):
        summary = CurationSummary(path=str(log_path))
        doc = _read_json(log_path, summary)
        if doc is not None:
            summary.available = True
            summary.dataset = str(doc.get("dataset", ""))
            summary.model = doc.get("model")
            stats = doc.get("stats") or {}
            summary.total_samples = _as_int(stats.get("total")) or 0
            summary.keyword_strong = _as_int(stats.get("keyword_strong")) or 0
            summary.keyword_ambiguous = _as_int(stats.get("keyword_ambiguous")) or 0
            summary.keyword_none = _as_int(stats.get("keyword_none")) or 0
            summary.llm_yes = _as_int(stats.get("llm_yes")) or 0
            summary.llm_no = _as_int(stats.get("llm_no")) or 0
            summary.llm_uncertain = _as_int(stats.get("llm_uncertain")) or 0
            summary.flagged_below_threshold = (
                _as_int(stats.get("flagged_below_threshold")) or 0
            )
            summary.confidence_threshold = float(
                doc.get("confidence_threshold") or 0.0
            )

        # Cross-reference the CSV sibling so we can report the actual
        # "yes / high-confidence" subset that Task 4/5 will consume.
        dataset = summary.dataset or _dataset_from_filename(log_path)
        if dataset:
            csv_path = inputs_dir / f"cvd_relevance_{dataset}.csv"
            summary.csv_path = str(csv_path)
            if csv_path.exists():
                summary.csv_available = True
                yes_high, yes_low = _count_yes_by_confidence(
                    csv_path, summary.confidence_threshold or 0.7
                )
                summary.csv_yes_high_conf = yes_high
                summary.csv_yes_low_conf = yes_low
        out.append(summary)
    return out


# ---------------------------------------------------------------------------
# Task 4 — Processing
# ---------------------------------------------------------------------------


def load_processing(inputs_dir: Path) -> list[ProcessingSummary]:
    out: list[ProcessingSummary] = []
    for log_path in sorted(inputs_dir.glob(PROCESSING_LOG_GLOB)):
        summary = ProcessingSummary(path=str(log_path))
        doc = _read_json(log_path, summary)
        if doc is not None:
            summary.available = True
            summary.dataset = str(doc.get("dataset") or _dataset_from_filename(log_path))
            outputs = doc.get("outputs") or {}
            summary.n_samples_final = _as_int(outputs.get("n_samples_final"))
            summary.n_genes_final = _as_int(outputs.get("n_genes_final"))
            cfg = doc.get("config") or {}
            summary.norm_method = cfg.get("norm_method")
            summary.steps = doc.get("steps") or {}
            summary.warnings = list(doc.get("warnings") or [])
            summary.errors = list(doc.get("errors") or [])
        out.append(summary)
    return out


# ---------------------------------------------------------------------------
# Task 5 — Labels
# ---------------------------------------------------------------------------


def load_labels(inputs_dir: Path) -> LabelsSummary:
    rec = LabelsSummary()

    reviewed_matches = sorted(inputs_dir.glob(LABEL_REVIEWED_GLOB))
    if reviewed_matches:
        rec.path = str(reviewed_matches[0])
        rec.reviewed = True
    else:
        # Fall back to the raw proposals — flagged as no-go by the decision
        # rules below because the human checkpoint hasn't been cleared.
        raw_matches = [
            p for p in sorted(inputs_dir.glob(LABEL_PROPOSAL_GLOB))
            if ".reviewed." not in p.name
        ]
        if raw_matches:
            rec.path = str(raw_matches[0])
            rec.reviewed = False

    if rec.path:
        label_counts, uncertain, mean_conf, n_rows, error = _read_label_csv(Path(rec.path))
        if error is None:
            rec.available = True
            rec.per_label = label_counts
            rec.n_uncertain = uncertain
            rec.mean_confidence = mean_conf
            rec.n_rows = n_rows
        else:
            rec.error = error

    # Task 5 run log — one per dataset, but we only need one for the
    # narrative (model / call counts). Pick the newest.
    log_matches = sorted(inputs_dir.glob(LABEL_LOG_GLOB))
    if log_matches:
        log_path = log_matches[-1]
        rec.log_path = str(log_path)
        doc = _read_json(log_path, None)
        if doc is not None:
            rec.log_available = True
            rec.model = doc.get("model")
    return rec


# ---------------------------------------------------------------------------
# Task 6 — EDA
# ---------------------------------------------------------------------------


def load_eda(inputs_dir: Path) -> EdaSummary:
    rec = EdaSummary()
    stats_path = inputs_dir / EDA_STATS_FILE
    rec.path = str(stats_path)
    if stats_path.exists():
        try:
            with stats_path.open(newline="") as fh:
                reader = csv.reader(fh)
                rows = list(reader)
            # Two shapes supported without opinion:
            # (a) 2 columns: metric,value  → dict
            # (b) tabular: keep first data row as flat dict
            if not rows:
                rec.error = "eda_summary_stats.csv is empty"
            elif len(rows[0]) == 2:
                rec.available = True
                rec.stats = {r[0]: r[1] for r in rows[1:] if len(r) == 2}
            else:
                rec.available = True
                headers = rows[0]
                data_row = rows[1] if len(rows) > 1 else []
                rec.stats = dict(zip(headers, data_row))
        except Exception as exc:  # noqa: BLE001 - keep the report generatable
            rec.error = f"{type(exc).__name__}: {exc}"

    for candidate in EDA_PLOT_DIR_CANDIDATES:
        pdir = inputs_dir / candidate
        if pdir.is_dir():
            rec.plot_dir = str(pdir)
            rec.plot_dir_available = True
            rec.plot_files = sorted(p.name for p in pdir.iterdir() if p.is_file())
            break
    return rec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path, rec: Artifact | None):  # type: ignore[name-defined]  # forward ref
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - report the parse failure via the record
        LOG.warning("failed to parse %s: %s", path, exc)
        if rec is not None:
            rec.error = f"{type(exc).__name__}: {exc}"
        return None


def _as_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dataset_from_filename(path: Path) -> str:
    """``curation_log_archs4.json`` → ``archs4``.

    Falls back to an empty string when the pattern doesn't match, so the
    caller can decide how to handle it.
    """
    m = re.match(r"(?:curation_log|processing_log)_(.+)\.json$", path.name)
    return m.group(1) if m else ""


def _count_yes_by_confidence(
    csv_path: Path, threshold: float
) -> tuple[int, int]:
    """Return (yes_at_or_above_threshold, yes_below_threshold) from Task 3 CSV."""
    yes_high = 0
    yes_low = 0
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if (row.get("llm_relevance") or "").strip().lower() != "yes":
                continue
            try:
                conf = float(row.get("confidence") or 0.0)
            except ValueError:
                conf = 0.0
            if conf >= threshold:
                yes_high += 1
            else:
                yes_low += 1
    return yes_high, yes_low


def _read_label_csv(
    csv_path: Path,
) -> tuple[dict[str, int], int, float | None, int, str | None]:
    """Return (per_label_counts, n_uncertain, mean_confidence, n_rows, error)."""
    counts: dict[str, int] = {}
    n_uncertain = 0
    conf_sum = 0.0
    conf_n = 0
    n_rows = 0
    try:
        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                n_rows += 1
                label = (row.get("proposed_label") or "").strip() or "(missing)"
                counts[label] = counts.get(label, 0) + 1
                if label == "uncertain":
                    n_uncertain += 1
                try:
                    conf_sum += float(row.get("confidence") or 0.0)
                    conf_n += 1
                except ValueError:
                    pass
    except Exception as exc:  # noqa: BLE001
        return {}, 0, None, 0, f"{type(exc).__name__}: {exc}"
    mean_conf = round(conf_sum / conf_n, 3) if conf_n else None
    return counts, n_uncertain, mean_conf, n_rows, None


# Re-export ``Artifact`` for the type hint on ``_read_json`` so the module
# stays self-contained.
from cvd_eda.reporting.schema import Artifact  # noqa: E402
