"""
train.py — the elastic net stage entry point.

Runs the full pipeline end to end:
    label -> subsample -> load_expression -> splits -> nested-CV fit
    -> evaluate -> gene_signal -> plots

CLI
---
python -m elasticnet.train \\
    --archs4-h5-path /path/to/human_gene_v2.latest.h5 \\
    --outdir eda/dataset/cvd_data/elasticnet_out \\
    --negative-ratio 10 \\
    --n-outer-folds 5 \\
    --n-inner-folds 5 \\
    --seed 20260707

--only lets you run a subset of the stages, same convention as `eda/eda.py`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .steps import (
    evaluate,
    gene_signal,
    label,
    load_expression,
    pipeline as pipe_step,
    plots,
    splits,
    subsample,
)

STAGE_ORDER = (
    "label",
    "subsample",
    "load_expression",
    "splits",
    "fit",
    "evaluate",
    "gene_signal",
    "plots",
)


def _setup_logging(outdir: Path) -> logging.Logger:
    log_dir = outdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.log"
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
    return logging.getLogger("elasticnet.train")


def _fit_all_folds(
    expression_dir: Path,
    subsample_dir: Path,
    splits_dir: Path,
    outdir: Path,
    n_inner_folds: int,
    seed: int,
    n_jobs: int,
    max_iter: int,
    tol: float,
    top_k_genes: int | None,
    logger: logging.Logger,
) -> Path:
    """Fit every outer fold under `outdir/folds/fold_{k}/` and return that dir."""
    folds_out = outdir / "folds"
    folds_out.mkdir(parents=True, exist_ok=True)

    # mmap the full matrix: each fold materialises only its train/test slice,
    # so peak RSS stays bounded by one fold instead of full-matrix + slice.
    X = np.load(expression_dir / "X.npy", mmap_mode="r")
    pool = pd.read_parquet(subsample_dir / "training_pool.parquet")
    y = pool["label"].to_numpy()
    groups = pool["source_series_id"].to_numpy()
    sample_indices = pool["sample_index"].to_numpy()

    fold_assignments = np.load(splits_dir / "fold_assignments.npy")
    n_outer = int(fold_assignments.max()) + 1

    fold_manifests = []
    for k in range(n_outer):
        logger.info("=== outer fold %d / %d ===", k, n_outer - 1)
        t0 = time.time()
        fm = pipe_step.fit_outer_fold(
            X=X, y=y, groups=groups,
            fold_assignments=fold_assignments, fold_id=k,
            out_fold_dir=folds_out / f"fold_{k}",
            max_iter=max_iter, tol=tol, top_k_genes=top_k_genes,
            n_inner_folds=n_inner_folds, seed=seed, n_jobs=n_jobs,
            sample_indices=sample_indices,
        )
        fm["wall_seconds"] = round(time.time() - t0, 2)
        fold_manifests.append(fm)
        logger.info(
            "fold %d done: C=%.4g, l1_ratio=%.2f, nonzero=%d, wall=%.1fs",
            k, fm["chosen_C"], fm["chosen_l1_ratio"], fm["n_nonzero_coefs"], fm["wall_seconds"],
        )

    with open(folds_out / "folds_summary.json", "w") as f:
        json.dump({"n_outer_folds": n_outer, "folds": fold_manifests}, f, indent=2)
    return folds_out


def run(
    h5_path: Path,
    outdir: Path,
    negative_ratio: int,
    n_outer_folds: int,
    n_inner_folds: int,
    seed: int,
    n_jobs: int,
    max_iter: int,
    tol: float,
    top_k_genes: int | None,
    only: list[str] | None,
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logging(outdir)

    stages = list(only) if only else list(STAGE_ORDER)
    unknown = [s for s in stages if s not in STAGE_ORDER]
    if unknown:
        raise SystemExit(f"unknown --only stage(s): {unknown}. Valid: {STAGE_ORDER}")

    manifest = {
        "started": datetime.now(timezone.utc).isoformat(),
        "h5_path": str(h5_path),
        "outdir": str(outdir),
        "config": {
            "negative_ratio": negative_ratio,
            "n_outer_folds": n_outer_folds,
            "n_inner_folds": n_inner_folds,
            "seed": seed,
            "n_jobs": n_jobs,
            "max_iter": max_iter,
            "tol": tol,
            "top_k_genes": top_k_genes,
        },
        "stages_requested": stages,
        "stage_status": {},
    }

    def _stage(name: str, fn):
        if name not in stages:
            manifest["stage_status"][name] = "skipped"
            return None
        t0 = time.time()
        logger.info("--- stage: %s ---", name)
        try:
            out = fn()
            manifest["stage_status"][name] = {"status": "ok", "wall_seconds": round(time.time() - t0, 2)}
            return out
        except Exception as e:
            manifest["stage_status"][name] = {"status": "error", "error": repr(e)}
            _write_manifest(manifest, outdir)
            raise

    label_dir = _stage("label", lambda: label.run(h5_path=h5_path, outdir=outdir)) or (outdir / "label")
    subsample_dir = _stage(
        "subsample",
        lambda: subsample.run(
            h5_path=h5_path, label_dir=label_dir, outdir=outdir,
            negative_ratio=negative_ratio, seed=seed,
        ),
    ) or (outdir / "subsample")
    expression_dir = _stage(
        "load_expression",
        lambda: load_expression.run(h5_path=h5_path, subsample_dir=subsample_dir, outdir=outdir),
    ) or (outdir / "expression")
    splits_dir = _stage(
        "splits",
        lambda: splits.run(
            subsample_dir=subsample_dir, outdir=outdir,
            n_outer_folds=n_outer_folds, seed=seed,
        ),
    ) or (outdir / "splits")
    folds_dir = _stage(
        "fit",
        lambda: _fit_all_folds(
            expression_dir=expression_dir, subsample_dir=subsample_dir,
            splits_dir=splits_dir, outdir=outdir, n_inner_folds=n_inner_folds,
            seed=seed, n_jobs=n_jobs, max_iter=max_iter, tol=tol,
            top_k_genes=top_k_genes, logger=logger,
        ),
    ) or (outdir / "folds")
    _stage("evaluate", lambda: evaluate.run(folds_dir=folds_dir, outdir=outdir))
    gene_signal_dir = _stage(
        "gene_signal",
        lambda: gene_signal.run(
            folds_dir=folds_dir, expression_dir=expression_dir, outdir=outdir
        ),
    ) or (outdir / "gene_signal")
    _stage(
        "plots",
        lambda: plots.run(
            folds_dir=folds_dir,
            ranking_csv=gene_signal_dir / "gene_signal_ranking.csv",
            outdir=outdir,
        ),
    )

    manifest["finished"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest, outdir)
    logger.info("elastic net pipeline complete → %s", outdir)


def _write_manifest(manifest: dict, outdir: Path) -> None:
    (outdir / "logs").mkdir(parents=True, exist_ok=True)
    with open(outdir / "logs" / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Elastic net whole-corpus CVD-classifier pipeline.")
    p.add_argument("--archs4-h5-path", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--negative-ratio", type=int, default=10)
    p.add_argument("--n-outer-folds", type=int, default=5)
    p.add_argument("--n-inner-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260707)
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Parallelism for LogisticRegressionCV. Leave at 1 for reproducibility.")
    p.add_argument("--max-iter", type=int, default=200,
                   help="saga max iterations per fit (converges well before this at the default tol).")
    p.add_argument("--tol", type=float, default=1e-2,
                   help="saga convergence tolerance. 1e-2 converges in ~50 iters on the reduced "
                        "feature set; tighter values slow fits ~10x for negligible metric gain.")
    p.add_argument("--top-k-genes", type=int, default=1500,
                   help="Keep the top-K most-variable genes per training fold (leakage-safe). "
                        "0 disables reduction and fits on all genes (intractable on ~49k features).")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated stage names to run. Valid: " + ",".join(STAGE_ORDER))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    top_k_genes = args.top_k_genes if args.top_k_genes and args.top_k_genes > 0 else None
    run(
        h5_path=args.archs4_h5_path, outdir=args.outdir,
        negative_ratio=args.negative_ratio, n_outer_folds=args.n_outer_folds,
        n_inner_folds=args.n_inner_folds, seed=args.seed, n_jobs=args.n_jobs,
        max_iter=args.max_iter, tol=args.tol, top_k_genes=top_k_genes,
        only=only,
    )


if __name__ == "__main__":
    main()
