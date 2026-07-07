"""
eda.py — orchestrator for the whole-corpus ARCHS4 EDA (Tasks 1-6).

Pipeline order (see .claude/eda_todo.md for provenance and rationale):
    1. cohort         cohort_composition_full.csv + composition plots
    2. qc             qc_full_dataset.csv + QC distribution plots
    3. normalize      quantile-normalized log2 reference + subsample matrix
    4. dimred         PCA + sample- and gene-centric t-SNE (perp 50/30)
    5. clustering     sample-sample correlation heatmap + linkage
    6. gene_summary   per-gene detection rate + biotype composition

Steps 4 and 5 depend on step 3 having written the normalized subsample.
Steps 1, 2, 6 read only the raw H5 and can run in any order.

Usage
-----
    # Full run:
    python -m eda.eda --h5 /path/to/human_gene_v2.7.h5 --outdir /path/to/eda_out

    # Just one step (or several):
    python -m eda.eda --h5 ... --outdir ... --only qc,gene_summary

    # Auto-locate H5 under a data root (uses the same glob as dataset/data.py):
    python -m eda.eda --data-root ~/cvd_data --outdir ~/cvd_data/eda_out

Each step is a self-contained module under `eda/steps/`; if any step fails,
the remaining steps still run and the failure is recorded in the manifest.
This matches the "write scripts, don't run them yet" mandate — the code is
robust enough for the eventual real run without pretending a partial result
is a success.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .steps import clustering, cohort, dimred, gene_summary, normalize, qc

ALL_STEPS = ("cohort", "qc", "normalize", "dimred", "clustering", "gene_summary")


def setup_logging(outdir: Path) -> logging.Logger:
    log_dir = outdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"eda_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.log"

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(sh)
    return logging.getLogger("eda")


def resolve_h5(h5_arg: str | None, data_root: str | None) -> Path:
    if h5_arg:
        p = Path(h5_arg).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"H5 file not found: {p}")
        return p
    if not data_root:
        raise ValueError("either --h5 or --data-root must be provided")
    root = Path(data_root).expanduser() / "archs4"
    matches = sorted(root.glob("human_gene_*.h5"))
    if not matches:
        raise FileNotFoundError(f"no ARCHS4 human gene H5 file found under {root}")
    return matches[0]


def _run_step(name: str, fn, logger: logging.Logger) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    logger.info("=== step %s: start ===", name)
    try:
        result = fn()
        finished = datetime.now(timezone.utc).isoformat()
        logger.info("=== step %s: done -> %s ===", name, result)
        return {"step": name, "status": "ok", "started": started, "finished": finished, "output": str(result)}
    except Exception as e:
        finished = datetime.now(timezone.utc).isoformat()
        logger.error("=== step %s: FAILED (%s) ===\n%s", name, e, traceback.format_exc())
        return {"step": name, "status": "failed", "started": started, "finished": finished, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Whole-corpus ARCHS4 EDA pipeline (Tasks 1-6).")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--h5", help="Path to the ARCHS4 human gene H5 file.")
    src.add_argument("--data-root", help="Base directory containing archs4/human_gene_*.h5 (as populated by dataset/data.py).")
    parser.add_argument("--outdir", required=True, help="Base output directory for EDA artifacts.")
    parser.add_argument("--only", default=None,
                        help=f"Comma-separated subset of steps to run. Default: all ({','.join(ALL_STEPS)}).")
    args = parser.parse_args()

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(outdir)

    h5_path = resolve_h5(args.h5, args.data_root)
    logger.info("using H5: %s", h5_path)
    logger.info("outdir:   %s", outdir)

    steps_to_run = ALL_STEPS if args.only is None else tuple(s.strip() for s in args.only.split(","))
    unknown = [s for s in steps_to_run if s not in ALL_STEPS]
    if unknown:
        raise SystemExit(f"unknown step(s): {unknown}. valid: {ALL_STEPS}")

    step_runners = {
        "cohort":       lambda: cohort.run(h5_path, outdir),
        "qc":           lambda: qc.run(h5_path, outdir),
        "normalize":    lambda: normalize.run(h5_path, outdir),
        "dimred":       lambda: dimred.run(h5_path, outdir),
        "clustering":   lambda: clustering.run(outdir),
        "gene_summary": lambda: gene_summary.run(h5_path, outdir),
    }

    manifest = {
        "h5_path": str(h5_path),
        "outdir": str(outdir),
        "run_started": datetime.now(timezone.utc).isoformat(),
        "steps": [],
    }
    for name in steps_to_run:
        manifest["steps"].append(_run_step(name, step_runners[name], logger))
    manifest["run_finished"] = datetime.now(timezone.utc).isoformat()

    manifest_path = outdir / "logs" / "eda_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
