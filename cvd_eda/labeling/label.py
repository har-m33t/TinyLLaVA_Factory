"""Task 5 orchestrator.

Reads Task 3's ``cvd_relevance_{dataset}.csv``, filters to high-confidence CVD
samples, optionally fetches the GEO series description for each unique
series_id, calls :class:`cvd_eda.labeling.llm.LabelProposer` per sample, and
writes ``label_proposals.csv`` plus a JSON run log.

Exits with a hard STOP banner: the output must not be consumed by Task 6
until a human reviewer has walked the ``uncertain`` rows and spot-checked
the confident ones.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from cvd_eda.labeling.llm import DEFAULT_MODEL, LabelLLMResult, LabelProposer
from cvd_eda.labeling.schema import (
    CSV_COLUMNS,
    LABEL_VOCAB,
    LabelProposal,
    UNCERTAIN_LABEL,
)


LOG = logging.getLogger(__name__)


# Task 3 CSV column names we rely on. Anything else in the row is folded into
# the sample-text blob passed to the LLM.
_TASK3_REQUIRED = ("sample_id", "llm_relevance", "confidence", "source_series_id")


@dataclass
class RunConfig:
    input_csv: Path
    output_csv: Path
    log_path: Path
    llm_cache_dir: Path
    geo_cache_dir: Path | None = None
    model: str = DEFAULT_MODEL
    min_relevance_confidence: float = 0.7
    max_samples: int | None = None
    use_geo_fetch: bool = True
    ncbi_email: str | None = None
    ncbi_api_key: str | None = None


@dataclass
class RunStats:
    total_input_rows: int = 0
    labeled: int = 0
    uncertain: int = 0
    skipped_low_confidence: int = 0
    skipped_not_yes: int = 0
    skipped_no_series: int = 0
    llm_errors: int = 0
    series_fetched: int = 0
    series_missing: int = 0
    per_label: dict[str, int] = field(default_factory=dict)


def load_relevant_samples(
    input_csv: Path, min_confidence: float
) -> tuple[list[dict[str, str]], int, int, int]:
    """Return (rows to label, total rows seen, skipped-not-yes, skipped-low-confidence)."""
    kept: list[dict[str, str]] = []
    total = 0
    skipped_not_yes = 0
    skipped_low = 0

    with input_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in _TASK3_REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"Task 3 CSV {input_csv} is missing required columns: {missing}. "
                f"Expected at least {_TASK3_REQUIRED}."
            )
        for row in reader:
            total += 1
            relevance = (row.get("llm_relevance") or "").strip().lower()
            if relevance != "yes":
                skipped_not_yes += 1
                continue
            try:
                conf = float(row.get("confidence") or 0.0)
            except ValueError:
                conf = 0.0
            if conf < min_confidence:
                skipped_low += 1
                continue
            kept.append(row)

    return kept, total, skipped_not_yes, skipped_low


def build_sample_text(row: dict[str, str]) -> str:
    """Reconstruct a metadata blob for the LLM from a Task 3 row.

    Task 3 always carries ``text`` (the concatenated raw metadata). We also
    surface ``matched_keyword`` and ``reasoning`` because those are the
    upstream justification for why this sample got here — useful signal
    for the labeling model without polluting the evidence-quote requirement
    (the quote has to appear verbatim in the input; we include everything).
    """
    parts: list[str] = []
    if row.get("text"):
        parts.append(row["text"].strip())
    if row.get("matched_keyword"):
        parts.append(f"matched keyword: {row['matched_keyword'].strip()}")
    if row.get("reasoning"):
        parts.append(f"upstream reasoning: {row['reasoning'].strip()}")
    return "\n".join(p for p in parts if p)


def _build_geo_fetcher(config: RunConfig):
    """Reuse the fetcher shipped alongside Task 3.

    ``cvd_eda/task3_curation/geo.py`` holds the GEO client used by Task 3 in
    its ``--use-geo-fetch`` mode. We reuse it verbatim so both tasks share a
    single on-disk cache and one throttling policy.
    """
    if not config.use_geo_fetch or config.geo_cache_dir is None:
        return None
    from cvd_eda.task3_curation.geo import GEOSeriesFetcher

    return GEOSeriesFetcher(
        cache_dir=config.geo_cache_dir,
        email=config.ncbi_email,
        api_key=config.ncbi_api_key,
    )


def _print_stop_banner(config: RunConfig, stats: RunStats) -> None:
    lines = [
        "",
        "================================================================",
        "  STOP — Task 5 output requires human review.",
        "================================================================",
        f"  Proposals written to: {config.output_csv}",
        f"  Run log:              {config.log_path}",
        f"  Labeled:              {stats.labeled}",
        f"    of which uncertain: {stats.uncertain}",
        f"  Skipped (not 'yes'):  {stats.skipped_not_yes}",
        f"  Skipped (low conf):   {stats.skipped_low_confidence}",
        f"  Skipped (no series):  {stats.skipped_no_series}",
        f"  LLM errors:           {stats.llm_errors}",
        "",
        "  Per-label counts:",
    ]
    for label in LABEL_VOCAB:
        count = stats.per_label.get(label, 0)
        lines.append(f"    {label:<12} {count}")
    lines += [
        "",
        "  Do NOT let Task 6 consume label_proposals.csv until a reviewer",
        "  has corrected the 'uncertain' rows and spot-checked the rest.",
        "  See cvd_eda/labeling/README.md for the review workflow.",
        "================================================================",
        "",
    ]
    print("\n".join(lines))


def run(config: RunConfig) -> RunStats:
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    config.llm_cache_dir.mkdir(parents=True, exist_ok=True)

    rows, total, skipped_not_yes, skipped_low = load_relevant_samples(
        config.input_csv, config.min_relevance_confidence
    )
    if config.max_samples is not None:
        rows = rows[: config.max_samples]

    stats = RunStats(
        total_input_rows=total,
        skipped_not_yes=skipped_not_yes,
        skipped_low_confidence=skipped_low,
    )

    proposer = LabelProposer(model=config.model, cache_dir=config.llm_cache_dir)
    geo_fetcher = _build_geo_fetcher(config)

    started = datetime.now(timezone.utc)
    with config.output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for row in rows:
            sample_id = (row.get("sample_id") or "").strip()
            series_id = (row.get("source_series_id") or "").strip()

            if not sample_id:
                LOG.warning("skipping row with empty sample_id: %r", row)
                continue

            series_text = ""
            if geo_fetcher is not None and series_id:
                try:
                    series_text = geo_fetcher.fetch(series_id)
                except Exception as exc:  # noqa: BLE001 - fetcher already soft-fails
                    LOG.warning("GEO fetch failed for %s: %s", series_id, exc)
                    series_text = ""
                if series_text:
                    stats.series_fetched += 1
                else:
                    stats.series_missing += 1
            elif not series_id:
                stats.skipped_no_series += 1

            try:
                result: LabelLLMResult = proposer.propose(
                    sample_id=sample_id,
                    series_id=series_id,
                    sample_text=build_sample_text(row),
                    series_text=series_text,
                )
            except Exception as exc:  # noqa: BLE001 - LLM errors continue the batch
                stats.llm_errors += 1
                LOG.error("Label LLM error for %s: %s", sample_id, exc)
                continue

            proposal = LabelProposal(
                sample_id=sample_id,
                proposed_label=result.proposed_label,
                confidence=result.confidence,
                evidence_quote=result.evidence_quote,
                uncertain_reason=result.uncertain_reason,
                source_series_id=series_id,
                model=result.model,
                cached=result.cached,
            )
            writer.writerow(proposal.to_row())
            stats.labeled += 1
            stats.per_label[proposal.proposed_label] = (
                stats.per_label.get(proposal.proposed_label, 0) + 1
            )
            if proposal.proposed_label == UNCERTAIN_LABEL:
                stats.uncertain += 1

    finished = datetime.now(timezone.utc)
    log_payload = {
        "task": "task5_labeling",
        "model": config.model,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": round((finished - started).total_seconds(), 2),
        "input_csv": str(config.input_csv),
        "output_csv": str(config.output_csv),
        "min_relevance_confidence": config.min_relevance_confidence,
        "max_samples": config.max_samples,
        "use_geo_fetch": config.use_geo_fetch,
        "llm_call_count": proposer.call_count,
        "llm_cache_hit_count": proposer.cache_hit_count,
        "stats": {
            "total_input_rows": stats.total_input_rows,
            "labeled": stats.labeled,
            "uncertain": stats.uncertain,
            "skipped_not_yes": stats.skipped_not_yes,
            "skipped_low_confidence": stats.skipped_low_confidence,
            "skipped_no_series": stats.skipped_no_series,
            "llm_errors": stats.llm_errors,
            "series_fetched": stats.series_fetched,
            "series_missing": stats.series_missing,
            "per_label": stats.per_label,
        },
    }
    config.log_path.write_text(json.dumps(log_payload, indent=2))

    _print_stop_banner(config, stats)
    return stats
