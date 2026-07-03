"""Offline smoke test for Task 7.

Fabricates a full ``logs/`` directory holding synthetic versions of every
artifact Tasks 1-6 emit, runs :mod:`cvd_eda.reporting.run` against it with
``--disable-llm``, and verifies the Markdown output contains the sections
we expect. Uses only the standard library so it stays runnable without the
Anthropic SDK or a network round-trip.

Run::

    python -m cvd_eda.reporting.smoke_test

Exits 0 on success, non-zero on any assertion failure.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

from cvd_eda.reporting import inputs as inputs_mod
from cvd_eda.reporting.report import build_payload, render_markdown
from cvd_eda.reporting.schema import DecisionVerdict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _fabricate(root: Path, *, reviewed_labels: bool = True) -> None:
    """Write one synthetic instance of every upstream artifact under ``root``."""
    root.mkdir(parents=True, exist_ok=True)

    # Task 1 — ARCHS4 ingestion (checksum OK, sensible shape).
    _write_json(root / "ingestion_log_archs4.json", {
        "task": "1-ingestion-archs4",
        "dataset": "ARCHS4 human_gene",
        "release_version": "v2.5",
        "release_url": "https://example.com/human_gene_v2.5.h5",
        "release_date": "2024-08-24",
        "published_sha1": "ae96de0519b9f008b0dc3a9f944ee9007daf2f6a",
        "local_path": "/scratch/archs4/human_gene_v2.5.h5",
        "stable_symlink": "/scratch/archs4/archs4_raw.h5",
        "file_size_bytes": 45_000_000_000,
        "computed_sha1": "ae96de0519b9f008b0dc3a9f944ee9007daf2f6a",
        "checksum_ok": True,
        "n_genes": 40_000,
        "n_samples": 1_200_000,
        "expression_dtype": "int32",
        "top_level_groups": ["data", "meta"],
        "meta_subgroups": ["genes", "info", "samples"],
        "archs4py_smoke": {"ok": True, "api": "archs4py.meta.field"},
        "download_started_utc": "2026-07-02T09:00:00+00:00",
        "download_finished_utc": "2026-07-02T09:45:00+00:00",
        "verification_finished_utc": "2026-07-02T09:55:00+00:00",
        "notes": [],
    })

    # Task 2 — RECOUNT3 ingestion (one project OK).
    _write_json(root / "ingestion_log_recount3.json", {
        "task": "task2_recount3",
        "run_started": "2026-07-02T10:00:00Z",
        "run_finished": "2026-07-02T10:15:00Z",
        "config_path": "cvd_eda/recount3/config/candidate_projects.yaml",
        "output_dir": "cvd_eda/data/recount3_raw",
        "n_candidates": 1,
        "catalog": None,
        "projects": [
            {
                "group": "gtex",
                "project": "HEART",
                "project_home": "data_sources/gtex",
                "organism": "human",
                "status": "ok",
                "counts_path": "cvd_eda/data/recount3_raw/HEART_counts.parquet",
            },
        ],
        "summary": {"total": 1, "ok_or_skipped": 1, "failed": 0},
    })

    # Task 3 — curation (archs4).
    _write_json(root / "curation_log_archs4.json", {
        "task": "3-metadata-curation",
        "dataset": "archs4",
        "inputs": ["cvd_eda/data/archs4_raw.h5"],
        "output_csv": str(root / "cvd_relevance_archs4.csv"),
        "model": "claude-haiku-4-5-20251001",
        "confidence_threshold": 0.7,
        "use_geo_fetch": False,
        "disable_llm": False,
        "run_started_utc": "2026-07-02T11:00:00+00:00",
        "run_finished_utc": "2026-07-02T11:20:00+00:00",
        "stats": {
            "total": 1000,
            "keyword_strong": 220,
            "keyword_ambiguous": 60,
            "keyword_none": 720,
            "llm_calls": 60,
            "llm_cache_hits": 0,
            "llm_yes": 25,
            "llm_no": 25,
            "llm_uncertain": 10,
            "flagged_below_threshold": 12,
            "elapsed_sec": 42.7,
        },
        "keyword_net": {"strong": [], "ambiguous": []},
        "notes": [],
    })

    # Sibling CSV so ``load_curation`` can compute the "yes at threshold" bucket.
    with (root / "cvd_relevance_archs4.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("sample_id", "matched_keyword", "llm_relevance",
                    "confidence", "reasoning", "source_series_id"))
        for i in range(200):
            w.writerow((f"GSM{i:07d}", "myocardial infarct", "yes",
                        0.95 if i < 180 else 0.5,
                        "strong keyword", f"GSE{1000 + (i // 40):06d}"))

    # Task 4 — processing.
    _write_json(root / "processing_log_archs4.json", {
        "dataset": "archs4",
        "started_at": "2026-07-02T12:00:00+00:00",
        "finished_at": "2026-07-02T12:30:00+00:00",
        "config": {
            "min_relevance_confidence": 0.7,
            "cpm_threshold": 1.0,
            "min_samples_per_gene_frac": 0.2,
            "min_samples_per_gene_abs": 10,
            "norm_method": "cpm_log2",
            "log_pseudocount": 1.0,
        },
        "inputs": {"raw_n_samples": 200, "raw_n_genes": 40_000},
        "outputs": {
            "normalized_matrix": "cvd_eda/logs/task4_out/cvd_matrix_archs4_normalized.parquet",
            "sample_metadata": "cvd_eda/logs/task4_out/cvd_sample_meta_archs4.parquet",
            "n_samples_final": 180,
            "n_genes_final": 18_500,
        },
        "steps": {},
        "environment": {},
        "warnings": [],
        "errors": [],
    })

    # Task 5 — reviewed labels.
    csv_name = "label_proposals.reviewed.csv" if reviewed_labels else "label_proposals.csv"
    with (root / csv_name).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("sample_id", "proposed_label", "confidence", "evidence_quote",
                    "uncertain_reason", "source_series_id", "model", "cached"))
        for i in range(180):
            if i < 80:
                label, conf = "case", 0.92
            elif i < 160:
                label, conf = "control", 0.9
            else:
                label, conf = "uncertain", 0.3
            w.writerow((f"GSM{i:07d}", label, conf,
                        "evidence quote here",
                        "" if label != "uncertain" else "no explicit control arm",
                        f"GSE{1000 + (i // 40):06d}",
                        "claude-opus-4-8", False))

    _write_json(root / "task5_run_log_archs4.json", {
        "task": "task5_labeling",
        "model": "claude-opus-4-8",
        "started_at": "2026-07-02T13:00:00+00:00",
        "finished_at": "2026-07-02T13:15:00+00:00",
        "elapsed_seconds": 900,
        "input_csv": str(root / "cvd_relevance_archs4.csv"),
        "output_csv": str(root / "label_proposals.csv"),
        "min_relevance_confidence": 0.7,
        "max_samples": None,
        "use_geo_fetch": True,
        "llm_call_count": 180,
        "llm_cache_hit_count": 0,
        "stats": {},
    })

    # Task 6 — one simple summary CSV with a batch-effect signal.
    with (root / "eda_summary_stats.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("metric", "value"))
        w.writerow(("n_samples", "180"))
        w.writerow(("pc1_series_id_corr", "0.42"))
        w.writerow(("pc1_disease_corr", "0.65"))
    plot_dir = root / "eda_plots"
    plot_dir.mkdir(exist_ok=True)
    for f in ("pca_disease.png", "sample_corr_heatmap.png"):
        (plot_dir / f).write_bytes(b"\x89PNG\r\n\x1a\nplaceholder")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _assert_report(md: str, verdict: str) -> None:
    _check("# CVD EDA — Task 7 report" in md, "report is missing the H1 title")
    _check("## Recommendation" in md, "report is missing the Recommendation section")
    _check(verdict.upper() in md, f"report does not surface the {verdict!r} verdict")
    _check("## Ingestion" in md, "report is missing the ingestion section")
    _check("## Task 3 — CVD relevance curation" in md, "report is missing curation section")
    _check("## Task 4 — Processing" in md, "report is missing processing section")
    _check("## Task 5 — Labels" in md, "report is missing labels section")
    _check("## Task 6 — EDA" in md, "report is missing EDA section")
    _check("## Sources" in md, "report is missing sources table")


def scenario_full_pipeline_reviewed(root: Path) -> None:
    _fabricate(root, reviewed_labels=True)
    loaded = inputs_mod.load_all(root)
    payload = build_payload(loaded, root)
    md = render_markdown(payload)

    _assert_report(md, verdict=payload.decision.verdict)
    _check(
        payload.decision.verdict in {DecisionVerdict.GO, DecisionVerdict.CAUTION},
        f"expected go/caution for reviewed pipeline, got {payload.decision.verdict}. "
        f"Reasons: {payload.decision.reasons}",
    )
    _check(payload.inputs.labels.reviewed, "reviewed label file was not detected")
    _check(payload.inputs.labels.n_rows == 180, "expected 180 labels rows")
    _check(payload.inputs.labels.per_label.get("case") == 80, "expected 80 cases")
    _check(payload.inputs.labels.per_label.get("control") == 80, "expected 80 controls")


def scenario_raw_labels_block(root: Path) -> None:
    _fabricate(root, reviewed_labels=False)
    loaded = inputs_mod.load_all(root)
    payload = build_payload(loaded, root)
    _check(
        payload.decision.verdict == DecisionVerdict.NO_GO,
        "raw (unreviewed) labels must produce a no-go verdict",
    )
    _check(
        any(r.name == "labels_reviewed" for r in payload.decision.reasons),
        "expected a 'labels_reviewed' blocker",
    )


def scenario_missing_everything(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    loaded = inputs_mod.load_all(root)
    payload = build_payload(loaded, root)
    md = render_markdown(payload)
    _assert_report(md, verdict=DecisionVerdict.NO_GO)
    _check(
        payload.decision.verdict == DecisionVerdict.NO_GO,
        "empty inputs dir should produce a no-go verdict",
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    scenarios = [
        ("full pipeline (reviewed labels)", scenario_full_pipeline_reviewed),
        ("labels raw only", scenario_raw_labels_block),
        ("empty inputs dir", scenario_missing_everything),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for i, (name, fn) in enumerate(scenarios):
            root = base / f"scenario_{i}"
            root.mkdir()
            try:
                fn(root)
            except AssertionError as exc:
                print(f"FAIL: {name}: {exc}", file=sys.stderr)
                return 1
            print(f"ok  : {name}")
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
