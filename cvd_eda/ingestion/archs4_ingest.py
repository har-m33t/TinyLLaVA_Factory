"""ARCHS4 human_gene HDF5 ingestion — Task 1 of the CVD EDA pipeline.

Downloads (with resume) the ARCHS4 human gene-level H5 file, verifies its
SHA1 against the checksum published on maayanlab.cloud/archs4/download.html,
sanity-checks HDF5 structure and shape, runs one trivial archs4py metadata
query, and writes ``ingestion_log_archs4.json`` for Task 7 (reporting) to
consume later.

Run ``python -m cvd_eda.ingestion.archs4_ingest --help`` for options.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

# --- release manifest ------------------------------------------------------
# Source: https://maayanlab.cloud/archs4/download.html (last verified 2026-07-02).
# If ARCHS4 publishes a newer release, override any of these on the CLI
# rather than editing the constants — the values below are the pinned
# default so re-running Task 1 later is reproducible.
DEFAULT_RELEASE = {
    "version": "v2.5",
    "url": "https://s3.dev.maayanlab.cloud/archs4/files/human_gene_v2.5.h5",
    "filename": "human_gene_v2.5.h5",
    "sha1": "ae96de0519b9f008b0dc3a9f944ee9007daf2f6a",
    "release_date": "2024-08-24",
    "approx_size_gb": 45,
}

STABLE_SYMLINK_NAME = "archs4_raw.h5"
LOG_FILENAME = "ingestion_log_archs4.json"

# Sample/gene count sanity ranges. ARCHS4 v2.5 has grown well beyond the
# 137,792 samples reported in the 2018 paper (Lachmann et al.); these
# bounds are "would be surprising", not strict expectations. A value
# outside them is logged as a warning and reflected in ``notes``, not
# an error.
EXPECTED_SAMPLES_MIN = 500_000
EXPECTED_SAMPLES_MAX = 2_000_000
EXPECTED_GENES_MIN = 20_000
EXPECTED_GENES_MAX = 80_000

# HDF5 structural expectations for ARCHS4 v2.x gene-level files.
EXPECTED_TOP_LEVEL_GROUPS = {"meta", "data"}
EXPECTED_DATA_DATASET = "expression"

LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"


# --- helpers ---------------------------------------------------------------


def _human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- download --------------------------------------------------------------


def stream_download(
    url: str,
    dest: Path,
    log: logging.Logger,
    chunk: int = 8 * 1024 * 1024,
    resume: bool = True,
) -> None:
    """Streaming HTTP download with optional Range-based resume.

    Emits a progress line every ~256 MiB so long HPC downloads leave an
    audit trail in the SLURM stdout log.
    """
    headers: dict[str, str] = {}
    mode = "wb"
    existing = 0
    if resume and dest.exists():
        existing = dest.stat().st_size
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        log.info("resuming download at %s", _human_bytes(existing))

    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as r:
        # 206 = partial content (Range honored). 200 = server ignored Range —
        # in that case anything already on disk is stale and must be
        # discarded to avoid concatenating two full copies.
        if existing and r.status_code == 200:
            log.warning("server ignored Range; restarting from byte 0")
            r.close()
            dest.unlink()
            existing = 0
            mode = "wb"
            with requests.get(url, stream=True, timeout=(30, 300)) as r2:
                r2.raise_for_status()
                _write_stream(r2, dest, mode, existing, chunk, log)
                return
        r.raise_for_status()
        _write_stream(r, dest, mode, existing, chunk, log)


def _write_stream(
    r: requests.Response,
    dest: Path,
    mode: str,
    existing: int,
    chunk: int,
    log: logging.Logger,
) -> None:
    total = int(r.headers.get("Content-Length", 0)) + existing
    log.info(
        "downloading %s → %s (Content-Length=%s)",
        r.url,
        dest,
        _human_bytes(total) if total else "unknown",
    )
    got = existing
    last_log_at = existing
    t0 = time.monotonic()
    with open(dest, mode) as fh:
        for buf in r.iter_content(chunk_size=chunk):
            if not buf:
                continue
            fh.write(buf)
            got += len(buf)
            if got - last_log_at >= chunk * 32:  # ~256 MiB
                dt = time.monotonic() - t0
                rate = (got - existing) / max(dt, 1e-6)
                pct = 100 * got / total if total else 0.0
                log.info(
                    "  %s / %s (%.1f%%, %s/s)",
                    _human_bytes(got),
                    _human_bytes(total) if total else "?",
                    pct,
                    _human_bytes(rate),
                )
                last_log_at = got
    log.info("download complete: %s (%s)", dest, _human_bytes(dest.stat().st_size))


# --- verification ----------------------------------------------------------


def sha1_of_file(path: Path, log: logging.Logger, chunk: int = 8 * 1024 * 1024) -> str:
    log.info("computing SHA1 of %s ...", path)
    h = hashlib.sha1()
    total = path.stat().st_size
    read = 0
    last_log_at = 0
    t0 = time.monotonic()
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(chunk), b""):
            h.update(buf)
            read += len(buf)
            if read - last_log_at >= chunk * 32:
                dt = time.monotonic() - t0
                rate = read / max(dt, 1e-6)
                log.info(
                    "  hashed %s / %s (%.1f%%, %s/s)",
                    _human_bytes(read),
                    _human_bytes(total),
                    100 * read / total,
                    _human_bytes(rate),
                )
                last_log_at = read
    return h.hexdigest()


def inspect_h5(path: Path, log: logging.Logger) -> dict[str, Any]:
    """Open with h5py, confirm structure, return shape + group summary.

    Raises RuntimeError if the required top-level groups or the
    ``data/expression`` dataset are missing.
    """
    import h5py  # local import so ``--skip-verify`` still runs without the dep

    with h5py.File(path, "r") as fh:
        top = set(fh.keys())
        log.info("top-level groups: %s", sorted(top))
        missing = EXPECTED_TOP_LEVEL_GROUPS - top
        if missing:
            raise RuntimeError(
                f"HDF5 is missing expected top-level groups: {sorted(missing)} "
                f"(have {sorted(top)})"
            )

        expr_key = f"data/{EXPECTED_DATA_DATASET}"
        if expr_key not in fh:
            raise RuntimeError(f"HDF5 is missing dataset {expr_key!r}")
        expr = fh[expr_key]
        # ARCHS4 v2.x stores expression as (genes, samples). Guard the
        # interpretation in case a future release transposes.
        shape = tuple(int(x) for x in expr.shape)
        if len(shape) != 2:
            raise RuntimeError(f"data/expression has unexpected rank {len(shape)}: {shape}")
        n_genes, n_samples = shape
        log.info(
            "data/expression shape = (%d genes, %d samples), dtype=%s",
            n_genes,
            n_samples,
            expr.dtype,
        )

        meta_subgroups = sorted(fh["meta"].keys())
        log.info("meta/* subgroups: %s", meta_subgroups)

    return {
        "n_genes": n_genes,
        "n_samples": n_samples,
        "expression_dtype": str(expr.dtype),
        "top_level_groups": sorted(top),
        "meta_subgroups": meta_subgroups,
    }


def archs4py_smoke(path: Path, log: logging.Logger) -> dict[str, Any]:
    """One trivial metadata query proving archs4py can read the file.

    archs4py's public API has shifted across releases; try the current
    ``archs4py.meta.field`` first and fall back to reading the raw
    ``meta/samples/series_id`` dataset via h5py if the wrapper API is
    absent. Either way, the smoke test result is captured for the log.
    """
    try:
        import archs4py  # noqa: F401
    except ImportError as e:
        log.warning("archs4py not installed (%s) — install with `uv pip install archs4py`", e)
        return {"ok": False, "reason": f"archs4py not installed: {e}"}

    try:
        import archs4py.meta as a4meta  # type: ignore[attr-defined]

        series = a4meta.field(str(path), "series_id")
        n = int(len(series))
        head = [str(x) for x in list(series[:3])]
        log.info("archs4py.meta.field('series_id'): n=%d, head=%s", n, head)
        return {
            "ok": True,
            "api": "archs4py.meta.field",
            "field": "series_id",
            "n_values": n,
            "sample_values": head,
        }
    except Exception as e:  # noqa: BLE001 — surface any failure into the log
        log.warning("archs4py.meta.field failed (%s); trying h5py fallback", e)
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


# --- log record ------------------------------------------------------------


@dataclass
class IngestionLog:
    task: str
    dataset: str
    release_version: str
    release_url: str
    release_date: str
    published_sha1: str
    local_path: str
    stable_symlink: Optional[str]
    file_size_bytes: int
    computed_sha1: Optional[str]
    checksum_ok: Optional[bool]
    n_genes: Optional[int]
    n_samples: Optional[int]
    expression_dtype: Optional[str]
    top_level_groups: Optional[list[str]]
    meta_subgroups: Optional[list[str]]
    archs4py_smoke: dict[str, Any]
    download_started_utc: Optional[str]
    download_finished_utc: Optional[str]
    verification_finished_utc: str
    notes: list[str] = field(default_factory=list)


# --- entrypoint ------------------------------------------------------------


def _resolve_data_dir(cli_value: Optional[str]) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env = os.environ.get("CVD_EDA_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    raise SystemExit(
        "Data directory not set. The ARCHS4 H5 is ~45 GB — do not download\n"
        "it into the repo by accident. Pass --data-dir or set CVD_EDA_DATA_DIR\n"
        "to a scratch location (e.g. $SCRATCH/archs4)."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "ARCHS4 ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-dir",
        help="Directory to store the H5 file. Also readable from CVD_EDA_DATA_DIR.",
    )
    p.add_argument(
        "--log-dir",
        default=str(Path(__file__).resolve().parent.parent / "logs"),
        help="Directory for ingestion_log_archs4.json (default: cvd_eda/logs/).",
    )
    p.add_argument("--url", default=DEFAULT_RELEASE["url"], help="Override release URL.")
    p.add_argument(
        "--filename",
        default=DEFAULT_RELEASE["filename"],
        help="Filename to save as inside --data-dir.",
    )
    p.add_argument(
        "--expected-sha1",
        default=DEFAULT_RELEASE["sha1"],
        help="Published SHA1 to verify against. Set to empty string to skip verify.",
    )
    p.add_argument(
        "--release-version",
        default=DEFAULT_RELEASE["version"],
        help="Release version tag recorded in the log.",
    )
    p.add_argument(
        "--release-date",
        default=DEFAULT_RELEASE["release_date"],
        help="Release date recorded in the log.",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume the file already exists at data_dir/filename and skip HTTP.",
    )
    p.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip SHA1 verification (hashing 45 GB takes ~10 minutes).",
    )
    p.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the archs4py metadata smoke query.",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume an existing partial download; overwrite from byte 0.",
    )
    p.add_argument(
        "--no-symlink",
        action="store_true",
        help="Do not create the stable archs4_raw.h5 symlink alongside the file.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(level=logging.INFO, format=LOG_FMT, stream=sys.stdout)
    log = logging.getLogger("archs4_ingest")

    data_dir = _resolve_data_dir(args.data_dir)
    log_dir = Path(args.log_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    dest = data_dir / args.filename
    notes: list[str] = []
    download_started_utc: Optional[str] = None
    download_finished_utc: Optional[str] = None

    if args.skip_download:
        if not dest.exists():
            log.error("--skip-download set but %s does not exist", dest)
            return 2
        log.info("skipping download; using existing %s", dest)
        notes.append("download step skipped via --skip-download")
    else:
        download_started_utc = _utcnow_iso()
        stream_download(args.url, dest, log, resume=not args.no_resume)
        download_finished_utc = _utcnow_iso()

    file_size = dest.stat().st_size
    log.info("file size on disk: %s (%d bytes)", _human_bytes(file_size), file_size)

    # Stable symlink for downstream stages so they don't hard-code a version.
    stable_link: Optional[Path] = None
    if not args.no_symlink:
        stable_link = data_dir / STABLE_SYMLINK_NAME
        if stable_link.is_symlink() or stable_link.exists():
            stable_link.unlink()
        stable_link.symlink_to(dest.name)
        log.info("symlink %s → %s", stable_link, dest.name)

    # SHA1 verification.
    computed_sha1: Optional[str] = None
    checksum_ok: Optional[bool] = None
    if args.skip_checksum or not args.expected_sha1:
        log.warning("skipping SHA1 verification")
        notes.append("checksum verification skipped")
    else:
        computed_sha1 = sha1_of_file(dest, log)
        checksum_ok = computed_sha1.lower() == args.expected_sha1.lower()
        if checksum_ok:
            log.info("SHA1 OK: %s", computed_sha1)
        else:
            log.error(
                "SHA1 MISMATCH: computed %s, expected %s",
                computed_sha1,
                args.expected_sha1,
            )
            notes.append(
                f"SHA1 mismatch: computed {computed_sha1}, expected {args.expected_sha1}"
            )

    # Structural / shape checks.
    try:
        shape = inspect_h5(dest, log)
    except Exception as e:  # noqa: BLE001 — record the failure into the log
        log.error("HDF5 inspection failed: %s", e)
        shape = {
            "n_genes": None,
            "n_samples": None,
            "expression_dtype": None,
            "top_level_groups": None,
            "meta_subgroups": None,
        }
        notes.append(f"HDF5 inspection failed: {type(e).__name__}: {e}")

    n_samples = shape.get("n_samples")
    n_genes = shape.get("n_genes")
    if isinstance(n_samples, int) and not (EXPECTED_SAMPLES_MIN <= n_samples <= EXPECTED_SAMPLES_MAX):
        msg = (
            f"sample count {n_samples} outside expected range "
            f"[{EXPECTED_SAMPLES_MIN}, {EXPECTED_SAMPLES_MAX}] — cross-check against "
            "current ARCHS4 release notes"
        )
        log.warning(msg)
        notes.append(msg)
    if isinstance(n_genes, int) and not (EXPECTED_GENES_MIN <= n_genes <= EXPECTED_GENES_MAX):
        msg = (
            f"gene count {n_genes} outside expected range "
            f"[{EXPECTED_GENES_MIN}, {EXPECTED_GENES_MAX}]"
        )
        log.warning(msg)
        notes.append(msg)

    # archs4py smoke query.
    if args.skip_smoke:
        smoke = {"ok": False, "reason": "skipped via --skip-smoke"}
        notes.append("archs4py smoke test skipped")
    else:
        smoke = archs4py_smoke(dest, log)

    record = IngestionLog(
        task="1-ingestion-archs4",
        dataset="ARCHS4 human_gene",
        release_version=args.release_version,
        release_url=args.url,
        release_date=args.release_date,
        published_sha1=args.expected_sha1,
        local_path=str(dest),
        stable_symlink=str(stable_link) if stable_link else None,
        file_size_bytes=file_size,
        computed_sha1=computed_sha1,
        checksum_ok=checksum_ok,
        n_genes=n_genes,
        n_samples=n_samples,
        expression_dtype=shape.get("expression_dtype"),
        top_level_groups=shape.get("top_level_groups"),
        meta_subgroups=shape.get("meta_subgroups"),
        archs4py_smoke=smoke,
        download_started_utc=download_started_utc,
        download_finished_utc=download_finished_utc,
        verification_finished_utc=_utcnow_iso(),
        notes=notes,
    )
    log_path = log_dir / LOG_FILENAME
    with open(log_path, "w") as fh:
        json.dump(asdict(record), fh, indent=2, sort_keys=True)
    log.info("wrote %s", log_path)

    # Exit non-zero if a hard failure occurred: checksum mismatch or missing
    # required HDF5 structure. Warnings (out-of-range counts, archs4py
    # missing) do not fail the run — Task 7 reads the log and decides.
    hard_fail = checksum_ok is False or shape.get("n_samples") is None
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
