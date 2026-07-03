#!/usr/bin/env python3
"""Orchestrate RECOUNT3 ingestion for the CVD EDA pipeline (Task 2).

Reads a YAML list of candidate projects, invokes the R scripts via `Rscript`
to pull each project through `create_rse()` + `transform_counts()` and export
Parquet, then aggregates one `ingestion_log_recount3.json` file across all
projects. Task 7 (Reporting Agent) consumes that log.

The R scripts emit a single JSON status line prefixed with
``RECOUNT3_STATUS_JSON:`` so this driver can parse per-project status even
when R's exit code is non-zero.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

STATUS_PREFIX = "RECOUNT3_STATUS_JSON:"


def _utcnow() -> str:
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def run_rscript(script: Path, args: list[str]) -> tuple[int, dict | None, str]:
    """Invoke Rscript, stream stderr through, and extract the status JSON.

    Returns (returncode, parsed_status_dict_or_None, combined_stdout).
    """
    if shutil.which("Rscript") is None:
        raise RuntimeError(
            "Rscript not found on PATH. Install R and re-run cvd_eda/task2_recount3/setup.sh, "
            "or `module load R/<version>` on the HPC node before invoking this."
        )

    proc = subprocess.run(
        ["Rscript", "--vanilla", str(script), *args],
        capture_output=True,
        text=True,
    )
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.stderr and not proc.stderr.endswith("\n"):
        sys.stderr.write("\n")

    status: dict | None = None
    for line in proc.stdout.splitlines():
        if line.startswith(STATUS_PREFIX):
            payload = line[len(STATUS_PREFIX):].strip()
            try:
                status = json.loads(payload)
            except json.JSONDecodeError as e:
                sys.stderr.write(
                    f"[orchestrate] Failed to parse status JSON: {e}\n"
                    f"[orchestrate] Payload was: {payload}\n"
                )
    return proc.returncode, status, proc.stdout


def _flatten_candidates(cfg: dict) -> list[dict]:
    """Turn the grouped YAML config into a flat list of candidate dicts."""
    flat: list[dict] = []
    for group_name, entries in (cfg or {}).items():
        for entry in entries or []:
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Candidate entry under '{group_name}' must be a mapping, "
                    f"got {type(entry).__name__}: {entry!r}"
                )
            missing = {"project", "project_home"} - entry.keys()
            if missing:
                raise ValueError(
                    f"Candidate under '{group_name}' is missing keys "
                    f"{sorted(missing)}: {entry!r}"
                )
            flat.append({
                "group": group_name,
                "project": entry["project"],
                "project_home": entry["project_home"],
                "organism": entry.get("organism", "human"),
                "notes": entry.get("notes", ""),
            })
    return flat


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True,
                   help="Path to candidate_projects.yaml")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where {project}_counts.parquet etc. are written "
                        "(recount3_raw/ per the task spec).")
    p.add_argument("--log-path", type=Path, required=True,
                   help="Where ingestion_log_recount3.json is written.")
    p.add_argument("--catalog-path", type=Path, default=None,
                   help="Optional: also enumerate the full available_projects() "
                        "catalog to this Parquet path (snapshot for audit).")
    p.add_argument("--r-scripts-dir", type=Path,
                   default=Path(__file__).resolve().parents[1] / "R",
                   help="Directory containing enumerate_projects.R and pull_and_export.R.")
    p.add_argument("--force", action="store_true",
                   help="Re-ingest a project even if its *_counts.parquet already exists.")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    candidates = _flatten_candidates(cfg)
    if not candidates:
        sys.stderr.write(
            "No candidate projects configured. Edit "
            f"{args.config} — GTEx HEART is the deterministic baseline; "
            "SRA candidates typically come from Task 3.\n"
        )
        return 1

    log: dict = {
        "task": "task2_recount3",
        "run_started": _utcnow(),
        "config_path": str(args.config),
        "output_dir": str(args.output_dir),
        "n_candidates": len(candidates),
        "catalog": None,
        "projects": [],
    }

    if args.catalog_path is not None:
        args.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[catalog] enumerating available_projects() -> {args.catalog_path}")
        rc, _, _ = run_rscript(
            args.r_scripts_dir / "enumerate_projects.R",
            [str(args.catalog_path)],
        )
        log["catalog"] = {
            "path": str(args.catalog_path),
            "returncode": rc,
            "status": "ok" if rc == 0 else "error",
        }

    for cand in candidates:
        counts_out = args.output_dir / f"{cand['project']}_counts.parquet"
        if counts_out.exists() and not args.force:
            print(f"[skip] {cand['group']}::{cand['project']} — already at {counts_out}")
            log["projects"].append({
                **cand,
                "status": "skipped_existing",
                "counts_path": str(counts_out),
            })
            continue

        print(f"[pull] {cand['group']}::{cand['project']} ({cand['project_home']})")
        rc, status, _ = run_rscript(
            args.r_scripts_dir / "pull_and_export.R",
            [cand["project"], cand["project_home"], cand["organism"], str(args.output_dir)],
        )
        entry = {**cand, "returncode": rc}
        if status is not None:
            entry.update(status)
        else:
            entry.update({
                "status": "error",
                "error": "R emitted no RECOUNT3_STATUS_JSON line; check stderr for a stack.",
            })
        log["projects"].append(entry)

    log["run_finished"] = _utcnow()
    ok_states = {"ok", "skipped_existing"}
    n_ok = sum(1 for e in log["projects"] if e.get("status") in ok_states)
    log["summary"] = {
        "total": len(log["projects"]),
        "ok_or_skipped": n_ok,
        "failed": len(log["projects"]) - n_ok,
    }

    with open(args.log_path, "w") as fh:
        json.dump(log, fh, indent=2)
    print(f"[log] wrote {args.log_path}")

    if log["summary"]["failed"]:
        print(
            f"[warn] {log['summary']['failed']} project(s) failed; "
            "see the per-project 'error' fields in the log.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
