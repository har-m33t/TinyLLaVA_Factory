"""Aggregator + Markdown renderer for Task 7.

Turns :class:`ReportInputs` into a deterministic :class:`ReportPayload`, then
renders that payload into a single Markdown document. The go/no-go verdict
follows a rubric defined in :func:`_evaluate_decision` — see the README for
the full list of blockers / warnings.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from cvd_eda.reporting.schema import (
    CurationSummary,
    Decision,
    DecisionRule,
    DecisionVerdict,
    LabelsSummary,
    ProcessingSummary,
    ReportInputs,
    ReportPayload,
)

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rubric constants — kept in one place so operators can tune them without
# hunting through the decision function.
# ---------------------------------------------------------------------------

MIN_CASE_SAMPLES_FOR_GO = 20
MAX_UNCERTAIN_FRACTION_FOR_GO = 0.20      # >20% uncertain → caution
UNCERTAIN_FRACTION_NO_GO = 0.50           # >50% uncertain → hard no-go
CLASS_IMBALANCE_CAUTION_RATIO = 5.0       # majority / minority >= 5:1 → caution
MIN_FINAL_SAMPLES_FOR_GO = 40             # per-dataset processed sample floor


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def build_payload(inputs: ReportInputs, inputs_dir: Path) -> ReportPayload:
    """Aggregate the loaded artifacts into a report payload.

    LLM narrative (``payload.narrative``) is left empty here — the CLI wires
    it in through :mod:`cvd_eda.reporting.llm` when the user opts in.
    """
    decision = _evaluate_decision(inputs)
    return ReportPayload(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        inputs_dir=str(inputs_dir),
        inputs=inputs,
        decision=decision,
    )


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------


def _evaluate_decision(inputs: ReportInputs) -> Decision:
    """Roll every rubric check up into a single verdict.

    Precedence: any ``no-go`` reason wins over any number of ``caution`` or
    ``go`` reasons. Absent inputs are treated as blockers only when the
    downstream stage cannot proceed without them — Task 6 output missing is
    the primary example.
    """
    rules: list[DecisionRule] = []

    # ----- Task 1: ARCHS4 ingestion ---------------------------------------
    a4 = inputs.ingestion_archs4
    if not a4.available:
        rules.append(DecisionRule(
            name="archs4_ingestion",
            verdict=DecisionVerdict.CAUTION,
            detail=f"ARCHS4 ingestion log not found at {a4.path}; cannot verify H5 integrity.",
        ))
    elif a4.checksum_ok is False:
        rules.append(DecisionRule(
            name="archs4_checksum",
            verdict=DecisionVerdict.NO_GO,
            detail="ARCHS4 SHA1 checksum mismatch — the H5 on disk does not match the published release.",
        ))
    else:
        rules.append(DecisionRule(
            name="archs4_ingestion",
            verdict=DecisionVerdict.GO,
            detail=(
                f"ARCHS4 {a4.release_version} ingested, "
                f"{a4.n_samples or '?'} samples × {a4.n_genes or '?'} genes."
            ),
        ))

    # ----- Task 2: RECOUNT3 ingestion -------------------------------------
    r3 = inputs.ingestion_recount3
    if not r3.available:
        rules.append(DecisionRule(
            name="recount3_ingestion",
            verdict=DecisionVerdict.CAUTION,
            detail="RECOUNT3 ingestion log not found; RECOUNT3 cohort will be missing from the analysis.",
        ))
    elif r3.n_projects_failed > 0:
        rules.append(DecisionRule(
            name="recount3_ingestion",
            verdict=DecisionVerdict.CAUTION,
            detail=(
                f"{r3.n_projects_failed} RECOUNT3 project(s) failed to ingest; "
                "see ingestion_log_recount3.json for the per-project error field."
            ),
        ))
    else:
        rules.append(DecisionRule(
            name="recount3_ingestion",
            verdict=DecisionVerdict.GO,
            detail=f"RECOUNT3: {r3.n_projects_ok} project(s) ingested successfully.",
        ))

    # ----- Task 3: Curation -----------------------------------------------
    if not inputs.curation:
        rules.append(DecisionRule(
            name="curation",
            verdict=DecisionVerdict.NO_GO,
            detail="No curation_log_*.json found — Task 3 (CVD relevance) has not run.",
        ))
    else:
        for cur in inputs.curation:
            if not cur.available:
                rules.append(DecisionRule(
                    name=f"curation_{cur.dataset or 'unknown'}",
                    verdict=DecisionVerdict.CAUTION,
                    detail=f"Curation log at {cur.path} could not be parsed ({cur.error}).",
                ))
                continue
            total_yes = cur.csv_yes_high_conf + cur.csv_yes_low_conf
            if cur.csv_available and total_yes == 0:
                rules.append(DecisionRule(
                    name=f"curation_{cur.dataset}",
                    verdict=DecisionVerdict.NO_GO,
                    detail=(
                        f"Curation ({cur.dataset}) surfaced zero CVD-relevant samples; "
                        "downstream stages have nothing to consume."
                    ),
                ))
            else:
                rules.append(DecisionRule(
                    name=f"curation_{cur.dataset}",
                    verdict=DecisionVerdict.GO,
                    detail=(
                        f"Curation ({cur.dataset}): "
                        f"{cur.csv_yes_high_conf} high-confidence + "
                        f"{cur.csv_yes_low_conf} low-confidence 'yes'."
                    ),
                ))

    # ----- Task 4: Processing ---------------------------------------------
    if not inputs.processing:
        rules.append(DecisionRule(
            name="processing",
            verdict=DecisionVerdict.NO_GO,
            detail="No processing_log_*.json found — Task 4 has not produced a normalized matrix.",
        ))
    else:
        for proc in inputs.processing:
            if not proc.available:
                rules.append(DecisionRule(
                    name=f"processing_{proc.dataset or 'unknown'}",
                    verdict=DecisionVerdict.CAUTION,
                    detail=f"Processing log at {proc.path} could not be parsed ({proc.error}).",
                ))
                continue
            if proc.errors:
                rules.append(DecisionRule(
                    name=f"processing_{proc.dataset}",
                    verdict=DecisionVerdict.NO_GO,
                    detail=(
                        f"Processing ({proc.dataset}) recorded errors: "
                        + "; ".join(proc.errors[:3])
                        + (" …" if len(proc.errors) > 3 else "")
                    ),
                ))
            elif (proc.n_samples_final or 0) < MIN_FINAL_SAMPLES_FOR_GO:
                rules.append(DecisionRule(
                    name=f"processing_{proc.dataset}",
                    verdict=DecisionVerdict.CAUTION,
                    detail=(
                        f"Processing ({proc.dataset}) retained only "
                        f"{proc.n_samples_final or 0} samples — below the "
                        f"suggested floor of {MIN_FINAL_SAMPLES_FOR_GO}."
                    ),
                ))
            else:
                rules.append(DecisionRule(
                    name=f"processing_{proc.dataset}",
                    verdict=DecisionVerdict.GO,
                    detail=(
                        f"Processing ({proc.dataset}): "
                        f"{proc.n_samples_final} samples × "
                        f"{proc.n_genes_final} genes retained, "
                        f"norm={proc.norm_method or '?'}."
                    ),
                ))

    # ----- Task 5: Labels -------------------------------------------------
    lab = inputs.labels
    if not lab.path:
        rules.append(DecisionRule(
            name="labels",
            verdict=DecisionVerdict.NO_GO,
            detail="No label_proposals*.csv found — Task 5 has not been run.",
        ))
    elif not lab.reviewed:
        rules.append(DecisionRule(
            name="labels_reviewed",
            verdict=DecisionVerdict.NO_GO,
            detail=(
                "Only the raw label_proposals.csv is present; the human review "
                "checkpoint has not been cleared. Rename or copy the reviewed "
                "file to label_proposals.reviewed.csv before proceeding."
            ),
        ))
    elif not lab.available:
        rules.append(DecisionRule(
            name="labels",
            verdict=DecisionVerdict.NO_GO,
            detail=f"Reviewed labels file at {lab.path} could not be parsed ({lab.error}).",
        ))
    else:
        rules.extend(_label_quality_rules(lab))

    # ----- Task 6: EDA ----------------------------------------------------
    eda = inputs.eda
    if not eda.available and not eda.plot_dir_available:
        rules.append(DecisionRule(
            name="eda",
            verdict=DecisionVerdict.NO_GO,
            detail="No eda_summary_stats.csv or eda_plots/ found — Task 6 has not been run.",
        ))
    else:
        # Batch effect / confounder heuristic. Task 6's schema isn't pinned,
        # so we look at any key that mentions PC1 / batch / series in the
        # stats dict and surface the numeric value.
        confound_hits = []
        for k, v in (eda.stats or {}).items():
            k_lower = str(k).lower()
            if ("pc1" in k_lower or "batch" in k_lower or "series" in k_lower):
                confound_hits.append((k, v))
        if confound_hits:
            worst_val = _max_abs_numeric([v for _, v in confound_hits])
            if worst_val is not None and worst_val >= 0.7:
                rules.append(DecisionRule(
                    name="eda_confounder",
                    verdict=DecisionVerdict.CAUTION,
                    detail=(
                        "EDA flagged a probable batch / series confounder "
                        f"(worst |corr|≈{worst_val:.2f}); recommend batch "
                        "correction before elastic-net fitting."
                    ),
                ))
            else:
                rules.append(DecisionRule(
                    name="eda",
                    verdict=DecisionVerdict.GO,
                    detail=f"EDA completed; no dominant confounder detected in stats "
                           f"({', '.join(f'{k}={v}' for k, v in confound_hits[:3])}).",
                ))
        else:
            rules.append(DecisionRule(
                name="eda",
                verdict=DecisionVerdict.GO,
                detail=(
                    f"EDA outputs present "
                    f"({'stats' if eda.available else 'plots-only'}); "
                    f"no confounder metric surfaced."
                ),
            ))

    verdict = _roll_up(rules)
    return Decision(verdict=verdict, reasons=rules)


def _label_quality_rules(lab: LabelsSummary) -> list[DecisionRule]:
    """Rules that depend on the reviewed label distribution."""
    rules: list[DecisionRule] = []

    # Uncertain fraction.
    if lab.n_rows > 0:
        frac_uncertain = lab.n_uncertain / lab.n_rows
        if frac_uncertain > UNCERTAIN_FRACTION_NO_GO:
            rules.append(DecisionRule(
                name="labels_uncertain",
                verdict=DecisionVerdict.NO_GO,
                detail=(
                    f"{lab.n_uncertain}/{lab.n_rows} labels remain 'uncertain' "
                    f"({frac_uncertain:.0%}); reviewer needs to resolve or drop them."
                ),
            ))
        elif frac_uncertain > MAX_UNCERTAIN_FRACTION_FOR_GO:
            rules.append(DecisionRule(
                name="labels_uncertain",
                verdict=DecisionVerdict.CAUTION,
                detail=(
                    f"{lab.n_uncertain}/{lab.n_rows} labels are still 'uncertain' "
                    f"({frac_uncertain:.0%}); may reduce statistical power."
                ),
            ))

    # Minimum per-class sample count. "case" and "control" are the primary
    # analysis arms; specific subtypes are audited but not blockers on their
    # own (a study might legitimately only have HCM samples).
    case_n = lab.per_label.get("case", 0)
    control_n = lab.per_label.get("control", 0)
    if case_n and control_n:
        smaller = min(case_n, control_n)
        larger = max(case_n, control_n)
        if smaller < MIN_CASE_SAMPLES_FOR_GO:
            rules.append(DecisionRule(
                name="labels_class_size",
                verdict=DecisionVerdict.NO_GO,
                detail=(
                    f"Smaller class has only {smaller} samples "
                    f"(case={case_n}, control={control_n}); below the "
                    f"floor of {MIN_CASE_SAMPLES_FOR_GO} for a reliable fit."
                ),
            ))
        elif larger / smaller >= CLASS_IMBALANCE_CAUTION_RATIO:
            rules.append(DecisionRule(
                name="labels_class_balance",
                verdict=DecisionVerdict.CAUTION,
                detail=(
                    f"Class imbalance {larger}:{smaller} exceeds "
                    f"{CLASS_IMBALANCE_CAUTION_RATIO:.0f}:1; consider stratified "
                    "sampling or class weighting."
                ),
            ))

    if not rules:
        rules.append(DecisionRule(
            name="labels",
            verdict=DecisionVerdict.GO,
            detail=(
                f"Reviewed labels look usable: {lab.n_rows} rows, "
                f"{lab.n_uncertain} still uncertain, mean confidence "
                f"{lab.mean_confidence if lab.mean_confidence is not None else 'n/a'}."
            ),
        ))
    return rules


def _roll_up(rules: Iterable[DecisionRule]) -> str:
    verdicts = [r.verdict for r in rules]
    if DecisionVerdict.NO_GO in verdicts:
        return DecisionVerdict.NO_GO
    if DecisionVerdict.CAUTION in verdicts:
        return DecisionVerdict.CAUTION
    return DecisionVerdict.GO


def _max_abs_numeric(values: Iterable) -> float | None:
    best: float | None = None
    for v in values:
        try:
            f = abs(float(v))
        except (TypeError, ValueError):
            continue
        if best is None or f > best:
            best = f
    return best


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


VERDICT_EMOJI = {
    DecisionVerdict.GO: "✅",
    DecisionVerdict.CAUTION: "⚠️",
    DecisionVerdict.NO_GO: "⛔",
}


def render_markdown(payload: ReportPayload) -> str:
    """Turn a payload into a `cvd_eda_report.md`-ready string."""
    inputs = payload.inputs
    lines: list[str] = []

    lines.append("# CVD EDA — Task 7 report")
    lines.append("")
    lines.append(f"Generated: `{payload.generated_at}`  ")
    lines.append(f"Inputs directory: `{payload.inputs_dir}`")
    lines.append("")

    # ---- Verdict up top -------------------------------------------------
    verdict = payload.decision.verdict
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        f"**{VERDICT_EMOJI.get(verdict, '?')} {verdict.upper()}** — "
        + _verdict_headline(payload.decision)
    )
    lines.append("")

    # Blockers first, then cautions, then the OKs (compact).
    blockers = [r for r in payload.decision.reasons if r.verdict == DecisionVerdict.NO_GO]
    cautions = [r for r in payload.decision.reasons if r.verdict == DecisionVerdict.CAUTION]
    goods = [r for r in payload.decision.reasons if r.verdict == DecisionVerdict.GO]

    if blockers:
        lines.append("### Blockers")
        for r in blockers:
            lines.append(f"- **{r.name}** — {r.detail}")
        lines.append("")
    if cautions:
        lines.append("### Caveats")
        for r in cautions:
            lines.append(f"- **{r.name}** — {r.detail}")
        lines.append("")
    if goods:
        lines.append("### Passing checks")
        for r in goods:
            lines.append(f"- {r.name}: {r.detail}")
        lines.append("")

    if payload.narrative:
        lines.append("## Executive summary (LLM synthesis)")
        lines.append("")
        lines.append(payload.narrative.strip())
        lines.append("")

    # ---- Section per upstream task --------------------------------------
    lines.extend(_render_ingestion(inputs))
    lines.extend(_render_curation(inputs.curation))
    lines.extend(_render_processing(inputs.processing))
    lines.extend(_render_labels(inputs.labels))
    lines.extend(_render_eda(inputs.eda))

    # ---- Audit trail — the files we actually read ------------------------
    lines.append("## Sources")
    lines.append("")
    lines.append("| Task | Artifact | Status |")
    lines.append("|---|---|---|")
    lines.extend(_source_rows(inputs))
    lines.append("")

    lines.append("---")
    lines.append("_Generated by `cvd_eda.reporting.run` — see "
                 "`cvd_eda/reporting/README.md` for the decision rubric._")
    lines.append("")
    return "\n".join(lines)


def _verdict_headline(decision: Decision) -> str:
    if decision.verdict == DecisionVerdict.GO:
        return "the current subset is ready for the elastic-net stage."
    if decision.verdict == DecisionVerdict.CAUTION:
        return (
            "the current subset can move forward, but the caveats below "
            "should be addressed or explicitly acknowledged first."
        )
    return (
        "the current subset is not ready. Resolve the blockers below "
        "before feeding the elastic-net."
    )


def _render_ingestion(inputs: ReportInputs) -> list[str]:
    lines = ["## Ingestion", ""]

    a4 = inputs.ingestion_archs4
    lines.append("### Task 1 — ARCHS4")
    if not a4.available:
        lines.append(f"_Missing:_ `{a4.path}` — {a4.error or 'file not found.'}")
    else:
        lines.append(f"- Release: **{a4.release_version or '?'}** ({a4.release_url or 'no url'})")
        lines.append(f"- Shape: {a4.n_samples or '?'} samples × {a4.n_genes or '?'} genes")
        size_gb = (a4.file_size_bytes / (1024**3)) if a4.file_size_bytes else None
        if size_gb is not None:
            lines.append(f"- File size: {size_gb:.1f} GiB")
        lines.append(
            f"- Checksum: {'OK' if a4.checksum_ok else 'MISMATCH' if a4.checksum_ok is False else 'skipped/unknown'}"
        )
        if a4.notes:
            lines.append("- Notes:")
            for note in a4.notes:
                lines.append(f"  - {note}")
    lines.append("")

    r3 = inputs.ingestion_recount3
    lines.append("### Task 2 — RECOUNT3")
    if not r3.available:
        lines.append(f"_Missing:_ `{r3.path}` — {r3.error or 'file not found.'}")
    else:
        lines.append(f"- Projects OK/skipped: **{r3.n_projects_ok}**")
        lines.append(f"- Projects failed: **{r3.n_projects_failed}**")
        if r3.project_rows:
            lines.append("- Per-project status:")
            lines.append("")
            lines.append("| group | project | status | error |")
            lines.append("|---|---|---|---|")
            for row in r3.project_rows:
                err = (row.get("error") or "").replace("|", "\\|")
                lines.append(
                    f"| {row.get('group') or ''} | {row.get('project') or ''} | "
                    f"{row.get('status') or ''} | {err} |"
                )
    lines.append("")
    return lines


def _render_curation(curations: list[CurationSummary]) -> list[str]:
    lines = ["## Task 3 — CVD relevance curation", ""]
    if not curations:
        lines.append("_No curation logs found._")
        lines.append("")
        return lines
    lines.append("| dataset | model | total | strong | ambiguous | none | yes (≥thr) | yes (<thr) | threshold |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cur in curations:
        if not cur.available:
            lines.append(f"| {cur.dataset or 'unknown'} | (parse error) | | | | | | | |")
            continue
        lines.append(
            f"| {cur.dataset} | {cur.model or '-'} | {cur.total_samples} | "
            f"{cur.keyword_strong} | {cur.keyword_ambiguous} | {cur.keyword_none} | "
            f"{cur.csv_yes_high_conf if cur.csv_available else '?'} | "
            f"{cur.csv_yes_low_conf if cur.csv_available else '?'} | "
            f"{cur.confidence_threshold:g} |"
        )
    lines.append("")
    return lines


def _render_processing(procs: list[ProcessingSummary]) -> list[str]:
    lines = ["## Task 4 — Processing", ""]
    if not procs:
        lines.append("_No processing logs found._")
        lines.append("")
        return lines
    lines.append("| dataset | samples | genes | norm | warnings | errors |")
    lines.append("|---|---|---|---|---|---|")
    for proc in procs:
        if not proc.available:
            lines.append(f"| {proc.dataset or 'unknown'} | (parse error) | | | | |")
            continue
        lines.append(
            f"| {proc.dataset} | {proc.n_samples_final or '?'} | "
            f"{proc.n_genes_final or '?'} | {proc.norm_method or '-'} | "
            f"{len(proc.warnings)} | {len(proc.errors)} |"
        )
    lines.append("")

    # Surface warnings / errors verbatim — they're the reason a dataset
    # might be dropped from the elastic-net.
    for proc in procs:
        if not proc.available or (not proc.warnings and not proc.errors):
            continue
        lines.append(f"### {proc.dataset}")
        if proc.errors:
            lines.append("**Errors:**")
            for e in proc.errors:
                lines.append(f"- {e}")
        if proc.warnings:
            lines.append("**Warnings:**")
            for w in proc.warnings:
                lines.append(f"- {w}")
        lines.append("")
    return lines


def _render_labels(lab: LabelsSummary) -> list[str]:
    lines = ["## Task 5 — Labels", ""]
    if not lab.path:
        lines.append("_No label_proposals*.csv found — Task 5 has not been run._")
        lines.append("")
        return lines

    reviewed_note = "reviewed" if lab.reviewed else "**RAW (human review not cleared)**"
    lines.append(f"- File: `{lab.path}` ({reviewed_note})")
    if lab.model:
        lines.append(f"- Proposer model: `{lab.model}`")
    if lab.n_rows:
        lines.append(f"- Rows: {lab.n_rows}")
        lines.append(f"- Uncertain: {lab.n_uncertain} "
                     f"({(lab.n_uncertain / lab.n_rows):.0%})")
        if lab.mean_confidence is not None:
            lines.append(f"- Mean confidence: {lab.mean_confidence}")
    if lab.per_label:
        lines.append("")
        lines.append("| label | count |")
        lines.append("|---|---|")
        for label, n in sorted(lab.per_label.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {label} | {n} |")
    lines.append("")
    return lines


def _render_eda(eda) -> list[str]:
    lines = ["## Task 6 — EDA", ""]
    if not eda.available and not eda.plot_dir_available:
        lines.append("_No `eda_summary_stats.csv` or `eda_plots/` found._")
        lines.append("")
        return lines
    if eda.available and eda.stats:
        lines.append("### Summary statistics")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for k, v in eda.stats.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
    if eda.plot_dir_available:
        lines.append(f"### Plots (`{eda.plot_dir}`)")
        lines.append("")
        for name in eda.plot_files:
            lines.append(f"- `{name}`")
        lines.append("")
    return lines


def _source_rows(inputs: ReportInputs) -> list[str]:
    rows: list[str] = []
    rows.append(f"| 1 | `{inputs.ingestion_archs4.path}` | "
                f"{_status(inputs.ingestion_archs4)} |")
    rows.append(f"| 2 | `{inputs.ingestion_recount3.path}` | "
                f"{_status(inputs.ingestion_recount3)} |")
    for cur in inputs.curation:
        rows.append(f"| 3 | `{cur.path}` | {_status(cur)} |")
        if cur.csv_path:
            rows.append(
                f"| 3 | `{cur.csv_path}` | "
                f"{'found' if cur.csv_available else 'missing'} |"
            )
    for proc in inputs.processing:
        rows.append(f"| 4 | `{proc.path}` | {_status(proc)} |")
    if inputs.labels.path:
        rows.append(f"| 5 | `{inputs.labels.path}` | "
                    f"{'reviewed' if inputs.labels.reviewed else 'raw (needs review)'} |")
    if inputs.labels.log_path:
        rows.append(f"| 5 | `{inputs.labels.log_path}` | "
                    f"{'found' if inputs.labels.log_available else 'missing'} |")
    if inputs.eda.path:
        rows.append(f"| 6 | `{inputs.eda.path}` | {_status(inputs.eda)} |")
    if inputs.eda.plot_dir:
        rows.append(f"| 6 | `{inputs.eda.plot_dir}/` | "
                    f"{'found' if inputs.eda.plot_dir_available else 'missing'} |")
    return rows


def _status(rec) -> str:
    if rec.available:
        return "ok"
    if rec.error:
        return f"parse error: {rec.error}"
    return "missing"
