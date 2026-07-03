"""Optional LLM synthesis for Task 7.

The deterministic sections of ``cvd_eda_report.md`` are always produced by
:mod:`cvd_eda.reporting.report`. This module adds a short executive-summary
paragraph at the top when ``ANTHROPIC_API_KEY`` is set, using the same
Anthropic SDK conventions as :mod:`cvd_eda.labeling.llm`.

Kept intentionally narrow: the LLM never invents numbers — it only rewrites
the deterministic decision + rule list into a couple of readable paragraphs,
so any hallucination is bounded to reordering / paraphrasing content that
already appears elsewhere in the report.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict

from cvd_eda.reporting.schema import (
    DecisionVerdict,
    ReportPayload,
)


LOG = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-8"


SYSTEM_PROMPT = """You are the Reporting Agent for a multi-agent CVD RNA-seq EDA pipeline. You have been given the JSON payload the deterministic aggregator produced. Write a 2-3 paragraph executive summary at the top of a Markdown report for a bioinformatician who will decide whether to fit an elastic net on the current data.

Ground rules:
- Do NOT invent numbers. Every quantitative claim must be traceable to a field in the payload.
- Lead with the go/no-go verdict and the single most important reason for it.
- Second paragraph: name the biggest caveat(s), if any, and what should happen next.
- Third paragraph (optional): brief note on cohort / label composition.
- Neutral, terse tone. Do not use marketing language.
- Return plain Markdown. No headings, no bullet lists — just paragraphs.
"""


class NarrativeError(RuntimeError):
    pass


def synthesize_narrative(
    payload: ReportPayload,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 700,
    max_retries: int = 3,
) -> str:
    """Return an LLM-written executive summary. Raises NarrativeError on failure."""
    try:
        import anthropic  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise NarrativeError(
            "The `anthropic` package is required for LLM narrative synthesis. "
            "Install with: uv pip install anthropic, or pass --disable-llm."
        ) from exc

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise NarrativeError(
            "ANTHROPIC_API_KEY is not set. Pass --disable-llm to skip the "
            "narrative section, or set the key and rerun."
        )

    from anthropic import Anthropic

    client = Anthropic()
    prompt = _build_prompt(payload)

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            for block in msg.content:
                if getattr(block, "type", None) == "text":
                    return block.text.strip()
            raise NarrativeError(f"No text block in Anthropic response: {msg}")
        except Exception as exc:  # noqa: BLE001 — SDK + parse errors both retry
            last_exc = exc
            if attempt < max_retries:
                delay = 2 ** (attempt - 1)
                LOG.warning(
                    "reporting narrative LLM call failed (attempt %s/%s): %s; "
                    "retrying in %ss",
                    attempt,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

    raise NarrativeError(
        f"reporting narrative failed after {max_retries} attempts: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_prompt(payload: ReportPayload) -> str:
    """Serialize the aggregator payload for the model.

    We hand the model the *decision* (verdict + reasons) plus the per-task
    numeric summaries — nothing else. That keeps the input under ~1 KB and
    means the model can't drift onto information the deterministic report
    doesn't already contain.
    """
    body = {
        "verdict": payload.decision.verdict,
        "reasons": [asdict(r) for r in payload.decision.reasons],
        "inputs": {
            "ingestion_archs4": {
                "available": payload.inputs.ingestion_archs4.available,
                "release_version": payload.inputs.ingestion_archs4.release_version,
                "checksum_ok": payload.inputs.ingestion_archs4.checksum_ok,
                "n_samples": payload.inputs.ingestion_archs4.n_samples,
                "n_genes": payload.inputs.ingestion_archs4.n_genes,
            },
            "ingestion_recount3": {
                "available": payload.inputs.ingestion_recount3.available,
                "n_projects_ok": payload.inputs.ingestion_recount3.n_projects_ok,
                "n_projects_failed": payload.inputs.ingestion_recount3.n_projects_failed,
            },
            "curation": [
                {
                    "dataset": c.dataset,
                    "total": c.total_samples,
                    "yes_high_conf": c.csv_yes_high_conf,
                    "yes_low_conf": c.csv_yes_low_conf,
                }
                for c in payload.inputs.curation
            ],
            "processing": [
                {
                    "dataset": p.dataset,
                    "n_samples_final": p.n_samples_final,
                    "n_genes_final": p.n_genes_final,
                    "warnings": len(p.warnings),
                    "errors": len(p.errors),
                }
                for p in payload.inputs.processing
            ],
            "labels": {
                "reviewed": payload.inputs.labels.reviewed,
                "n_rows": payload.inputs.labels.n_rows,
                "n_uncertain": payload.inputs.labels.n_uncertain,
                "per_label": payload.inputs.labels.per_label,
                "mean_confidence": payload.inputs.labels.mean_confidence,
            },
            "eda": {
                "available": payload.inputs.eda.available,
                "stats_keys": sorted((payload.inputs.eda.stats or {}).keys()),
                "n_plot_files": len(payload.inputs.eda.plot_files),
            },
        },
    }
    return (
        "Here is the deterministic aggregation. Write the executive summary now.\n\n"
        "```json\n"
        + json.dumps(body, indent=2)
        + "\n```"
    )
