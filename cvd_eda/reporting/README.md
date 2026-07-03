# Task 7 — Reporting Agent

Implements **Task 7** from `.claude/EDA_CLAUDE_TASKS.md`: read the logs and
tables emitted by Tasks 1-6 and produce a single Markdown report
(`cvd_eda_report.md`) that ends in a go/no-go recommendation for the
elastic-net stage.

## What it does

1. **Load** every upstream artifact under `--inputs-dir` (default
   `cvd_eda/logs/`) into a typed record. Missing or malformed files are
   surfaced in the report, not raised — Task 7 must always emit *something*,
   even when the pipeline is incomplete.
2. **Aggregate** the numbers into a `ReportPayload` — cohort sizes, per-task
   counts, label distribution, batch-effect signals from EDA.
3. **Decide** via a rubric (see below) whether the current subset is ready
   for the elastic-net. The verdict is one of `go` / `caution` / `no-go`.
4. **Render** the payload as Markdown: verdict up top, blockers → caveats →
   passing checks, one section per upstream task, an audit-trail `Sources`
   table at the bottom.
5. **Optionally** ask an LLM to write a 2-3 paragraph executive summary. The
   deterministic body is unchanged whether the LLM is on or off.

## Inputs

Everything is read from a single directory (`--inputs-dir`):

| Task | Artifact | Notes |
|---|---|---|
| 1 | `ingestion_log_archs4.json` | Task 1 output (`cvd_eda.ingestion.archs4_ingest`). |
| 2 | `ingestion_log_recount3.json` | Task 2 output (`cvd_eda.recount3.python.orchestrate`). |
| 3 | `curation_log_{dataset}.json` | One per dataset (`archs4`, `recount3`). |
| 3 | `cvd_relevance_{dataset}.csv` | Sibling CSV — used to count the "yes ≥ threshold" bucket. |
| 4 | `processing_log_{dataset}.json` | One per dataset (or per-project for RECOUNT3). |
| 5 | `label_proposals*.reviewed.csv` | **Required** for a `go`. If only the raw `label_proposals*.csv` is present, the report emits a `no-go` blocker. |
| 5 | `task5_run_log_*.json` | Model + call-count info; optional. |
| 6 | `eda_summary_stats.csv` | Two shapes accepted: `metric,value` rows, or a single-data-row table. |
| 6 | `eda_plots/` | Directory of plot files — enumerated as a bullet list. |

Any artifact can be missing — the report says so in the `Sources` table and
folds the fact into the decision rubric.

## Output

* `cvd_eda_report.md` — the deliverable Task 7 owes.
* Optional `--json-payload PATH` — the same aggregated numbers as JSON, so a
  downstream dashboard can render an alternate view without re-parsing every
  upstream log.

## Decision rubric

A single `no-go` reason downgrades the whole verdict. Every check is
listed in the report, so a reviewer can see *why* the roll-up landed where
it did.

| Signal | Verdict | Trigger |
|---|---|---|
| ARCHS4 checksum mismatch | **no-go** | `checksum_ok == False` in Task 1 log. |
| ARCHS4 log missing | caution | Cannot verify H5 integrity. |
| RECOUNT3 project failures | caution | `summary.failed > 0` in Task 2 log. |
| Curation log missing | **no-go** | Task 3 hasn't run — nothing to filter on. |
| Curation → zero "yes" | **no-go** | Downstream stages have nothing to consume. |
| Processing log missing | **no-go** | No normalized matrix produced. |
| Processing errors present | **no-go** | Non-empty `errors` array in Task 4 log. |
| Processing kept < 40 samples | caution | Below the suggested statistical floor. |
| Labels file missing | **no-go** | Task 5 hasn't run. |
| Only raw `label_proposals.csv` | **no-go** | Human review checkpoint uncleared. |
| Reviewed labels: > 50% uncertain | **no-go** | Reviewer needs to resolve. |
| Reviewed labels: > 20% uncertain | caution | Reduced power. |
| Smaller class < 20 samples | **no-go** | Not enough data for a reliable case/control fit. |
| Class imbalance ≥ 5:1 | caution | Consider class weighting. |
| EDA output missing | **no-go** | Cannot audit confounders. |
| EDA: PC1/batch/series `|corr| ≥ 0.7` | caution | Recommend batch correction first. |

Constants live at the top of `report.py`
(`MIN_CASE_SAMPLES_FOR_GO`, `UNCERTAIN_FRACTION_NO_GO`, …) — tune there.

## Install

Uses only the standard library for the deterministic sections. The optional
LLM synthesis needs the Anthropic SDK, already in the `eda` extra:

```bash
source .venv/bin/activate
uv pip install anthropic     # only if you want --model / --disable-llm off
```

## Credentials

* `ANTHROPIC_API_KEY` — required for the executive-summary paragraph. Pass
  `--disable-llm` to skip the LLM entirely; the deterministic Markdown body
  is identical either way.

## Run

Assuming the earlier tasks have written into `cvd_eda/logs/`:

```bash
source .venv/bin/activate

python -m cvd_eda.reporting.run \
    --inputs-dir cvd_eda/logs \
    --output     cvd_eda/logs/cvd_eda_report.md
```

Without an LLM (deterministic sections only):

```bash
python -m cvd_eda.reporting.run --inputs-dir cvd_eda/logs --disable-llm
```

With a machine-readable payload alongside:

```bash
python -m cvd_eda.reporting.run \
    --inputs-dir     cvd_eda/logs \
    --output         cvd_eda/logs/cvd_eda_report.md \
    --json-payload   cvd_eda/logs/cvd_eda_report_payload.json
```

To make a `no-go` verdict block a CI pipeline:

```bash
python -m cvd_eda.reporting.run --inputs-dir cvd_eda/logs --exit-code-on-no-go 3
```

Smoke test (no network, no Anthropic key required):

```bash
python -m cvd_eda.reporting.smoke_test
```

Fabricates three synthetic pipeline states (full pipeline with reviewed
labels; raw-labels-only; empty inputs dir) and asserts the report renders
with the expected verdict for each.

## Design notes

* **Deterministic body, optional narrative.** The LLM never invents numbers
  — its prompt only receives the aggregated payload and is instructed to
  paraphrase. Any hallucination is bounded to reordering content that
  already appears in the deterministic sections. The narrative is skipped
  quietly if the SDK / key aren't available; the report still ships.
* **Missing files are not errors.** Task 7 has to run at every stage of the
  workflow — including "we just ingested ARCHS4 and want to know where we
  are." The `Sources` table records what was found; the decision rubric
  distinguishes "you haven't run this yet" (no-go blocker on required
  stages) from "this stage optionally is not there" (caveat).
* **Reviewed vs raw labels is a hard gate.** The presence of
  `label_proposals.reviewed.csv` is the sentinel that the Task 5 human
  checkpoint was actually cleared. If only the raw file is present, the
  verdict is unconditionally `no-go` — matching the STOP banner Task 5
  prints on exit.
* **Rubric constants at the top of `report.py`.** So an operator changing
  the "40 sample floor" or "5:1 imbalance ratio" isn't hunting through
  branches to find them.
* **Same Anthropic SDK conventions as Tasks 3 and 5.** The LLM call in
  `llm.py` is intentionally minimal — no on-disk cache, because Task 7 is
  the last stage and a report is rebuilt rarely.

## Files

```
cvd_eda/reporting/
├── README.md          (this file)
├── __init__.py
├── __main__.py        (`python -m cvd_eda.reporting`)
├── schema.py          (dataclasses for the payload + decision records)
├── inputs.py          (loaders for every upstream artifact)
├── report.py          (aggregation, decision rubric, Markdown renderer)
├── llm.py             (optional Anthropic-backed executive summary)
├── run.py             (CLI entrypoint)
└── smoke_test.py      (offline end-to-end fixtures test)
```
