"""Task 3 orchestrator: load metadata → keyword filter → LLM triage → write outputs.

Three passes:

1. **Keyword net** (:mod:`cvd_eda.curation.keywords`) tags each sample as
   ``strong`` / ``ambiguous`` / ``none``.
2. **LLM triage** runs only on ``ambiguous`` samples. Strong hits are called
   ``yes`` on the strength of the phrase alone; no-hit samples are called
   ``no``. This is exactly the split the task spec calls for — don't burn
   API budget on the confident buckets.
3. **Output** — one CSV row per input sample with the exact schema promised
   in ``.claude/EDA_CLAUDE_TASKS.md`` (``sample_id, matched_keyword,
   llm_relevance, confidence, reasoning, source_series_id``) plus a JSON log
   for Task 7.

CLI::

    python -m cvd_eda.curation --dataset archs4 \\
        --input path/to/archs4_raw.h5

    python -m cvd_eda.curation --dataset recount3 \\
        --input path/to/HEART_coldata.parquet \\
        --input path/to/SRP123456_coldata.parquet

Defaults put the CSV at ``cvd_eda/logs/cvd_relevance_{dataset}.csv`` and the
JSON log alongside it — that's where Task 4 (``--relevance-csv``) and Task 5
(``--input``) already look.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from cvd_eda.curation import keywords
from cvd_eda.curation.metadata import load_archs4, load_recount3


LOG = logging.getLogger("cvd_eda.curation")
LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"

OUTPUT_COLUMNS = (
    "sample_id",
    "matched_keyword",
    "llm_relevance",
    "confidence",
    "reasoning",
    "source_series_id",
)

# Text under this length is judged too sparse to classify on its own — with
# ``--use-geo-fetch`` on, we'll pull the series description for extra context.
SPARSE_TEXT_CHARS = 60

# Confidence assigned to keyword-only calls. Kept below the LLM's 0-1 native
# range so a downstream ``confidence >= 0.9`` filter still picks these up but
# operators can tell them apart from LLM-scored rows in the log stats.
KEYWORD_NO_CONFIDENCE = 0.95
KEYWORD_YES_CONFIDENCE = 0.9

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # cvd_eda/


# ---------------------------------------------------------------------------
# Log record
# ---------------------------------------------------------------------------


@dataclass
class RunStats:
    total: int = 0
    keyword_strong: int = 0
    keyword_ambiguous: int = 0
    keyword_none: int = 0
    llm_calls: int = 0
    llm_cache_hits: int = 0
    llm_yes: int = 0
    llm_no: int = 0
    llm_uncertain: int = 0
    flagged_below_threshold: int = 0
    elapsed_sec: float = 0.0


@dataclass
class CurationLog:
    task: str
    dataset: str
    inputs: list[str]
    output_csv: str
    model: Optional[str]
    confidence_threshold: float
    use_geo_fetch: bool
    disable_llm: bool
    run_started_utc: str
    run_finished_utc: str
    stats: dict
    keyword_net: dict
    notes: list[str] = field(default_factory=list)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_output_dir() -> Path:
    """Where the CSV and JSON log land by default.

    Task 5's README and Task 4's ``--relevance-csv`` docs both point at
    ``cvd_eda/logs/``, so that's the default. Override with ``--output-dir``.
    """
    env = os.environ.get("CVD_EDA_LOG_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (_PACKAGE_ROOT / "logs").resolve()


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run(
    *,
    dataset: str,
    inputs: list[str],
    output_dir: str | Path,
    confidence_threshold: float = 0.7,
    use_geo_fetch: bool = False,
    geo_cache: str | Path | None = None,
    llm_cache: str | Path | None = None,
    model: str | None = None,
    disable_llm: bool = False,
) -> Path:
    """Run Task 3 end-to-end for one dataset. Returns the CSV path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = RunStats()
    notes: list[str] = []
    run_started = _utcnow_iso()
    t0 = time.monotonic()

    LOG.info("Loading metadata for %s from %s", dataset, inputs)
    if dataset == "archs4":
        if len(inputs) != 1:
            raise SystemExit(
                "--dataset archs4 expects exactly one --input (the H5 file)"
            )
        df = load_archs4(inputs[0])
    elif dataset == "recount3":
        df = load_recount3(inputs)
    else:
        raise SystemExit(f"Unknown --dataset {dataset!r}; expected archs4|recount3")
    stats.total = len(df)
    LOG.info("Loaded %d samples", stats.total)

    # ---- keyword pass ----------------------------------------------------
    matches = df["text"].map(keywords.match)
    df["_strength"] = matches.map(lambda m: m.strength)
    df["_matched"] = matches.map(lambda m: m.matched)
    df["_matched_primary"] = matches.map(lambda m: m.primary)
    stats.keyword_strong = int((df["_strength"] == "strong").sum())
    stats.keyword_ambiguous = int((df["_strength"] == "ambiguous").sum())
    stats.keyword_none = int((df["_strength"] == "none").sum())
    LOG.info(
        "Keyword pass: strong=%d ambiguous=%d none=%d",
        stats.keyword_strong,
        stats.keyword_ambiguous,
        stats.keyword_none,
    )

    # ---- LLM pass on ambiguous rows -------------------------------------
    classifier = None
    if stats.keyword_ambiguous > 0 and not disable_llm:
        from cvd_eda.curation.llm import LLMClassifier, DEFAULT_MODEL

        classifier = LLMClassifier(
            model=model or DEFAULT_MODEL,
            cache_dir=llm_cache,
        )
    elif stats.keyword_ambiguous > 0 and disable_llm:
        notes.append(
            f"--disable-llm: {stats.keyword_ambiguous} ambiguous samples marked "
            "uncertain (confidence=0) and flagged for human review."
        )

    fetcher = None
    if use_geo_fetch:
        if not geo_cache:
            raise SystemExit("--use-geo-fetch requires --geo-cache DIR")
        from cvd_eda.curation.geo import GEOSeriesFetcher

        fetcher = GEOSeriesFetcher(cache_dir=geo_cache)

    rows: list[dict] = []
    for row in df.itertuples(index=False):
        strength: str = row._strength
        matched: tuple[str, ...] = row._matched
        matched_primary: str = row._matched_primary

        if strength == "none":
            rows.append(
                _out_row(
                    row,
                    matched_keyword="",
                    llm_relevance="no",
                    confidence=KEYWORD_NO_CONFIDENCE,
                    reasoning="No CVD keyword match in sample metadata.",
                )
            )
        elif strength == "strong":
            rows.append(
                _out_row(
                    row,
                    matched_keyword=matched_primary,
                    llm_relevance="yes",
                    confidence=KEYWORD_YES_CONFIDENCE,
                    reasoning=f"Strong CVD keyword(s): {', '.join(matched)}",
                )
            )
        else:  # ambiguous
            if classifier is None:
                rows.append(
                    _out_row(
                        row,
                        matched_keyword=matched_primary,
                        llm_relevance="uncertain",
                        confidence=0.0,
                        reasoning=(
                            "LLM stage disabled; ambiguous keyword(s) "
                            f"{', '.join(matched)} flagged for human review."
                        ),
                    )
                )
                stats.flagged_below_threshold += 1
                continue

            series_text = ""
            if fetcher is not None and len(row.text) < SPARSE_TEXT_CHARS:
                series_text = fetcher.fetch(row.source_series_id)

            result = classifier.classify(
                sample_id=row.sample_id,
                series_id=row.source_series_id,
                matched=matched,
                text=row.text,
                series_text=series_text,
            )
            rows.append(
                _out_row(
                    row,
                    matched_keyword=matched_primary,
                    llm_relevance=result.relevance,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                )
            )
            if result.relevance == "yes":
                stats.llm_yes += 1
            elif result.relevance == "no":
                stats.llm_no += 1
            else:
                stats.llm_uncertain += 1
            if result.confidence < confidence_threshold:
                stats.flagged_below_threshold += 1

    if classifier is not None:
        stats.llm_calls = classifier.call_count
        stats.llm_cache_hits = classifier.cache_hit_count

    out_df = pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))
    csv_path = output_dir / f"cvd_relevance_{dataset}.csv"
    out_df.to_csv(csv_path, index=False)
    LOG.info("Wrote %s (%d rows)", csv_path, len(out_df))

    stats.elapsed_sec = round(time.monotonic() - t0, 2)

    record = CurationLog(
        task="3-metadata-curation",
        dataset=dataset,
        inputs=[str(p) for p in inputs],
        output_csv=str(csv_path),
        model=(model or (classifier.model if classifier else None)),
        confidence_threshold=confidence_threshold,
        use_geo_fetch=use_geo_fetch,
        disable_llm=disable_llm,
        run_started_utc=run_started,
        run_finished_utc=_utcnow_iso(),
        stats=asdict(stats),
        keyword_net={
            "strong": list(keywords.STRONG_KEYWORDS),
            "ambiguous": list(keywords.AMBIGUOUS_KEYWORDS),
        },
        notes=notes,
    )
    log_path = output_dir / f"curation_log_{dataset}.json"
    log_path.write_text(json.dumps(asdict(record), indent=2, sort_keys=True))
    LOG.info("Wrote %s", log_path)
    return csv_path


def _out_row(row, *, matched_keyword, llm_relevance, confidence, reasoning) -> dict:
    return {
        "sample_id": row.sample_id,
        "matched_keyword": matched_keyword,
        "llm_relevance": llm_relevance,
        "confidence": round(float(confidence), 4),
        "reasoning": reasoning,
        "source_series_id": row.source_series_id,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cvd_eda.curation",
        description="Task 3 — CVD relevance curation for ARCHS4 or RECOUNT3 metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, choices=("archs4", "recount3"))
    p.add_argument(
        "--input",
        dest="inputs",
        required=True,
        action="append",
        help=(
            "Input path. Repeatable. "
            "archs4: exactly one H5 file (Task 1 output). "
            "recount3: one or more {project}_coldata.parquet files (Task 2 output)."
        ),
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Where cvd_relevance_{dataset}.csv and curation_log_{dataset}.json "
            "land. Defaults to $CVD_EDA_LOG_DIR if set, else cvd_eda/logs/ "
            "(where Task 4/5 already look)."
        ),
    )
    p.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help=(
            "LLM-confidence below which a sample is flagged for human review in "
            "the log stats. Task 4 defaults to the same value."
        ),
    )
    p.add_argument(
        "--use-geo-fetch",
        action="store_true",
        help="Fetch GSE title/summary from NCBI Entrez when sample text is sparse.",
    )
    p.add_argument("--geo-cache", default=None, help="Directory for GEO series-description cache.")
    p.add_argument(
        "--llm-cache",
        default=None,
        help="Directory for the on-disk LLM response cache. Highly recommended for reruns.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Anthropic model id; defaults to claude-haiku-4-5-20251001.",
    )
    p.add_argument(
        "--disable-llm",
        action="store_true",
        help=(
            "Skip the LLM stage entirely (ambiguous samples become 'uncertain' with "
            "confidence=0). Useful for smoke tests without ANTHROPIC_API_KEY."
        ),
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FMT, stream=sys.stdout)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_output_dir()
    )

    run(
        dataset=args.dataset,
        inputs=args.inputs,
        output_dir=output_dir,
        confidence_threshold=args.confidence_threshold,
        use_geo_fetch=args.use_geo_fetch,
        geo_cache=args.geo_cache,
        llm_cache=args.llm_cache,
        model=args.model,
        disable_llm=args.disable_llm,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
