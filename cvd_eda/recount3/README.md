# Task 2 — RECOUNT3 Ingestion Agent

Owns the "get the RECOUNT3 side of the CVD EDA into a Python-readable form"
step of the pipeline described in `.claude/EDA_CLAUDE_TASKS.md`. Runs entirely
offline once inputs are configured; the only network I/O is the recount3
S3-backed HTTP pulls performed by R.

## What it does

1. **Enumerate** — `Rscript R/enumerate_projects.R` snapshots
   `recount3::available_projects(organism="human")` to a Parquet catalog. This
   gives every downstream step a stable, auditable view of what was available
   at run time.
2. **Pull each candidate** — for every entry in
   `config/candidate_projects.yaml`, `Rscript R/pull_and_export.R`:
   - resolves the row in the catalog by `(project, project_home)`,
   - builds the RangedSummarizedExperiment with `create_rse()`,
   - applies `transform_counts()` to convert recount3's per-base coverage
     into read counts (the input DESeq2 / edgeR expect),
   - dumps the counts matrix, `colData`, and `rowData` to Parquet.
3. **Aggregate a log** — `python/orchestrate.py` collects per-project status
   into a single `ingestion_log_recount3.json` that Task 7 (Reporting)
   consumes.

The R side emits a single-line JSON status record prefixed with
`RECOUNT3_STATUS_JSON:` on stdout, so the Python orchestrator can capture
per-project status regardless of R's exit code.

## Layout

```
cvd_eda/task2_recount3/
├── R/
│   ├── enumerate_projects.R    # available_projects() -> Parquet snapshot
│   └── pull_and_export.R       # create_rse + transform_counts + Parquet export
├── config/
│   └── candidate_projects.yaml # candidate list (GTEx HEART seeded; SRA from Task 3)
├── python/
│   └── orchestrate.py          # drives Rscript, aggregates ingestion log
├── run.sh                      # one-shot entrypoint
├── setup.sh                    # installs recount3 / arrow / jsonlite in R
└── README.md                   # this file
```

## Running it

The user's login node does not have R installed. Do this on the compute node
(or wherever the R module is available):

```bash
# 1) Load R and install dependencies (one-time)
module load R/4.4.0        # or whatever the site provides
./setup.sh                 # BiocManager + recount3 + SummarizedExperiment + arrow + jsonlite

# 2) Make sure the Python side has PyYAML
source .venv/bin/activate  # the uv venv from INSTALL-ENV-CLAUDE.md
pip install pyyaml

# 3) Run the ingest
./run.sh                                   # -> cvd_eda/data/recount3_raw/
./run.sh /scratch/$USER/cvd_eda/recount3   # override output dir
./run.sh /scratch/$USER/cvd_eda/recount3 --force  # re-ingest existing projects
```

`run.sh` is idempotent: projects whose `{project}_counts.parquet` already
exists are skipped unless `--force` is passed.

## Outputs

Per the task spec, under `<output_dir>/`:

```
recount3_raw/
├── available_projects_catalog.parquet   # snapshot of every human project
├── HEART_counts.parquet                 # gene_id x sample columns + gene_id col
├── HEART_coldata.parquet                # sample metadata + sample_id col
├── HEART_rowdata.parquet                # gene metadata (Ensembl id, symbol, biotype, ...)
├── <SRP...>_counts.parquet              # (once Task 3 populates sra:)
├── <SRP...>_coldata.parquet
├── <SRP...>_rowdata.parquet
└── ingestion_log_recount3.json
```

### `ingestion_log_recount3.json` schema

```jsonc
{
  "task": "task2_recount3",
  "run_started":  "2026-07-02T18:04:11Z",   // UTC ISO-8601
  "run_finished": "2026-07-02T18:22:47Z",
  "config_path":  ".../candidate_projects.yaml",
  "output_dir":   ".../recount3_raw",
  "n_candidates": 1,
  "catalog": {
    "path":       ".../available_projects_catalog.parquet",
    "returncode": 0,
    "status":     "ok"
  },
  "projects": [
    {
      "group":        "gtex",                // section of the YAML it came from
      "project":      "HEART",
      "project_home": "data_sources/gtex",
      "organism":     "human",
      "notes":        "GTEx heart tissue ...",
      "returncode":   0,
      "status":       "ok",                  // ok | skipped_existing | error
      "n_samples":    861,                   // set when status=ok
      "n_genes":      63856,
      "counts_path":  ".../HEART_counts.parquet",
      "coldata_path": ".../HEART_coldata.parquet",
      "rowdata_path": ".../HEART_rowdata.parquet",
      "transform":    "recount3::transform_counts (scaled coverage -> counts)",
      "coldata_cols": [ "gtex.smtsd", "gtex.sex", "gtex.age", "... " ],
      "elapsed_sec":  742.31,
      "error":        null                   // string on status=error
    }
  ],
  "summary": { "total": 1, "ok_or_skipped": 1, "failed": 0 }
}
```

The exit code of `run.sh` is:

- `0` — all candidates ok or skipped_existing
- `1` — misconfiguration (empty candidate list, missing keys)
- `2` — at least one project failed to ingest (details in the log's per-project `error`)

## Design decisions

- **Rscript subprocess over `rpy2`.** The task spec allows either. `rpy2`
  requires a compiled bridge tied to the local R + Python ABI, which is
  brittle on HPC. A one-shot `Rscript --vanilla` call per project keeps the
  Python side portable and lets R crashes stay isolated.
- **Catalog is snapshotted, not re-fetched per project.** `available_projects()`
  is cheap but not free (hits S3). More importantly, ingesting a small batch of
  projects against a moving catalog risks the version-drift that Task 7 has to
  reason about. The Parquet snapshot is the single source of truth for what
  was available at ingestion time.
- **`gene_id` and `sample_id` are promoted to real columns.** Parquet has no
  concept of R rownames, so writing them as columns avoids a silent identity
  loss when the Python side reads the files back.
- **CVD keyword filtering is deferred to Task 3.** The spec draws a hard line:
  keyword-based filtering of noisy free text belongs to the Metadata-Curation
  Agent, not the ingester. Task 2's job is deterministic pull-and-export.
  Task 3 writes SRA project IDs into `config/candidate_projects.yaml` under
  `sra:`, and Task 2 is re-run.
- **GTEx heart-tissue subsetting is not done here.** GTEx has one `HEART`
  project spanning both `Heart - Atrial Appendage` and `Heart - Left Ventricle`.
  Task 4 (Cleaning) filters `colData$gtex.smtsd`; that keeps sample selection
  in one place instead of splitting it between two agents.

## Hand-off to Task 3 and Task 4

- **Task 3 (Metadata Curation)** reads every `{project}_coldata.parquet` plus
  the top-level `available_projects_catalog.parquet`. Its output — a list of
  CVD-relevant SRA project IDs — gets appended to `config/candidate_projects.yaml`
  under `sra:`, and Task 2 is re-run to pull them.
- **Task 4 (Cleaning)** reads `{project}_counts.parquet` + `{project}_coldata.parquet`
  and applies the sample-level filters (GTEx tissue, CVD subset from Task 3's
  labels, low-count gene filter, normalization).

## Troubleshooting

- `Rscript not found on PATH` — load an R module first (`module load R/…`).
  `setup.sh` and `orchestrate.py` both fail loudly with this message.
- `Project 'X' with project_home 'Y' not found in available_projects()` —
  the candidate accession is not in the current recount3 catalog. Check
  `available_projects_catalog.parquet` for spelling / project_home value
  (`data_sources/sra` vs `data_sources/gtex`).
- Arrow install fails on the HPC — usually a libarrow C++ mismatch. Set
  `ARROW_R_DEV=TRUE` before `install.packages("arrow")` for a verbose log.
- Slow pulls — GTEx HEART is ~800 samples × ~63k genes and takes several
  minutes end-to-end; SRA projects are usually smaller. If nothing prints for
  more than 10 minutes on a single project, the S3 fetch is probably stuck;
  cancel and re-run.
