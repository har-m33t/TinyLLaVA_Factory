# Task 1 — ARCHS4 ingestion

Owner: Claude Code · Script: `archs4_ingest.py` · Output: `cvd_eda/logs/ingestion_log_archs4.json`

Implements the ARCHS4 half of the ingestion stage of `.claude/EDA_CLAUDE_TASKS.md`.
Downloads the human gene-level ARCHS4 HDF5 file, verifies it against the
SHA1 published on `maayanlab.cloud/archs4/download.html`, sanity-checks
the HDF5 structure and shape, runs one trivial `archs4py` metadata query
to confirm the file is usable, and writes a JSON audit log that Task 7
(reporting) consumes at the end of the pipeline.

## Release pinned

The defaults are pinned to the release verified on **2026-07-02**:

| Field         | Value                                                                   |
|---------------|-------------------------------------------------------------------------|
| Version       | `v2.5`                                                                  |
| URL           | `https://s3.dev.maayanlab.cloud/archs4/files/human_gene_v2.5.h5`        |
| Filename      | `human_gene_v2.5.h5`                                                    |
| Size          | ~45 GB                                                                  |
| SHA1          | `ae96de0519b9f008b0dc3a9f944ee9007daf2f6a`                              |
| Release date  | 2024-08-24                                                              |

If a newer release is out, override on the CLI (`--url`, `--filename`,
`--expected-sha1`, `--release-version`, `--release-date`) rather than
editing constants — the pinned defaults are what makes re-running Task 1
later reproducible.

## Installing runtime deps

Neither `h5py` nor `archs4py` is in `pyproject.toml` (they are only used
by this side pipeline, not TinyLLaVA). Install into the project venv:

```bash
source .venv/bin/activate
uv pip install h5py archs4py requests
```

`requests` is already a `tinyllava` dep; the `uv pip install` line is
idempotent.

## Running

```bash
# 45 GB — put data on scratch, not the repo volume
export CVD_EDA_DATA_DIR=$SCRATCH/archs4

# Foreground (~10-30 min depending on network + local disk)
python -m cvd_eda.ingestion.archs4_ingest

# Or as a SLURM job
sbatch cvd_eda/ingestion/archs4_ingest.slurm   # (not yet provided)
```

Progress is printed to stdout every ~256 MiB, so SLURM stdout gives a
running download / hashing trace.

### Useful flags

| Flag                       | Purpose                                                        |
|----------------------------|----------------------------------------------------------------|
| `--data-dir PATH`          | Where the H5 lands. Overrides `CVD_EDA_DATA_DIR`.              |
| `--log-dir PATH`           | Where `ingestion_log_archs4.json` lands (default `cvd_eda/logs/`). |
| `--skip-download`          | Assume the H5 is already at `data-dir/filename` and just verify. |
| `--skip-checksum`          | Skip SHA1. Hashing 45 GB is ~10 min on decent local SSD.       |
| `--skip-smoke`             | Skip the `archs4py` metadata query.                            |
| `--no-resume`              | Discard any partial H5 on disk and download from byte 0.       |
| `--url` / `--filename` / `--expected-sha1` / `--release-version` / `--release-date` | Override the pinned v2.5 defaults for a newer release. |

## What the script does

1. **Download** — streaming GET with 8 MiB chunks. If a partial file
   already exists at the destination, resume via HTTP `Range` header;
   if the server ignores the Range, discard the partial and restart
   cleanly.
2. **Stable symlink** — `data-dir/archs4_raw.h5 → human_gene_v2.5.h5`
   so downstream stages don't hard-code the version in a path.
3. **SHA1 verification** — hash the file, compare against
   `--expected-sha1`. Mismatch is recorded in the log and exits non-zero.
4. **HDF5 structural check** — open with `h5py`, confirm top-level
   groups `meta/` and `data/` are present and that `data/expression`
   exists with rank 2. Missing groups → hard failure.
5. **Shape sanity check** — extract `(n_genes, n_samples)` and warn if
   outside `[20 000, 80 000]` genes or `[500 000, 2 000 000]` samples.
   These are "would be surprising" bounds, not strict expectations —
   ARCHS4 has grown well past the 137,792 samples of the 2018 paper and
   will keep growing. Warnings are surfaced into `notes` in the log,
   not treated as failures.
6. **archs4py smoke test** — call `archs4py.meta.field(path,
   "series_id")` and record its length + first three values. If
   `archs4py` isn't installed or the wrapper API has moved, the failure
   is captured in the log rather than crashing the run.
7. **Write log** — `cvd_eda/logs/ingestion_log_archs4.json` with the
   fields below. Task 7 will read this verbatim.

## Log schema

`ingestion_log_archs4.json` fields:

| Field                            | Meaning                                             |
|----------------------------------|-----------------------------------------------------|
| `task`                           | Always `"1-ingestion-archs4"`                       |
| `dataset`                        | `"ARCHS4 human_gene"`                               |
| `release_version` / `release_url` / `release_date` / `published_sha1` | Release the log was produced against |
| `local_path`                     | Absolute path to the H5 that was verified           |
| `stable_symlink`                 | Path to `archs4_raw.h5` (or `null` if disabled)     |
| `file_size_bytes`                | Actual file size on disk                            |
| `computed_sha1` / `checksum_ok`  | Verifier output (both `null` if `--skip-checksum`)  |
| `n_genes` / `n_samples`          | Shape of `data/expression`                          |
| `expression_dtype`               | e.g. `uint32`                                       |
| `top_level_groups`               | e.g. `["data", "info", "meta"]`                     |
| `meta_subgroups`                 | e.g. `["genes", "samples"]`                         |
| `archs4py_smoke`                 | `{ok, api, field, n_values, sample_values}` (or `{ok: False, reason}`) |
| `download_started_utc` / `download_finished_utc` | ISO-8601, `null` if `--skip-download` |
| `verification_finished_utc`      | ISO-8601 timestamp of the log write                 |
| `notes`                          | Free-text warnings surfaced during the run          |

## Exit codes

- `0` — all hard checks passed. Warnings may still be in `notes`.
- `1` — hard failure: SHA1 mismatch, or `data/expression` missing/malformed.
- `2` — bad CLI (e.g. `--skip-download` but the file isn't there).

## Not implemented here

- Task 2 (RECOUNT3 ingestion) is a sibling script and lives in this
  same folder once someone writes it.
- Verifying that a specific sample of interest is present. That is
  Task 3's job (metadata curation).
- Any actual expression-data reading. Task 4 owns the counts matrix;
  Task 1 stops at "file downloaded, verified, and openable".
