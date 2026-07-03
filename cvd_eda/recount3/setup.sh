#!/usr/bin/env bash
# Install the R packages Task 2 needs: recount3 (Bioconductor),
# SummarizedExperiment, arrow (for Parquet), jsonlite (for the status line).
# Assumes R >= 4.2 is on PATH — on the HPC this typically means:
#     module load R/4.4.0        # or whatever the site provides
# Idempotent; safe to re-run.
set -euo pipefail

if ! command -v Rscript >/dev/null 2>&1; then
  echo "error: Rscript not found on PATH." >&2
  echo "  On the HPC, load an R module first, e.g. 'module load R/4.4.0'." >&2
  exit 127
fi

Rscript --vanilla - <<'RSCRIPT'
options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

# ask=FALSE, update=FALSE keeps this non-interactive and avoids surprise upgrades.
BiocManager::install(
  c("recount3", "SummarizedExperiment"),
  ask = FALSE, update = FALSE
)

# arrow: prefer source build so we don't get a wheel mismatched to the R ABI
# on unusual HPC toolchains. If a system libarrow is present it will link to it;
# otherwise arrow bundles its own via a pre-build.
if (!requireNamespace("arrow", quietly = TRUE)) {
  install.packages("arrow")
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite")
}

cat("recount3:              ", as.character(packageVersion("recount3")),              "\n")
cat("SummarizedExperiment:  ", as.character(packageVersion("SummarizedExperiment")),  "\n")
cat("arrow:                 ", as.character(packageVersion("arrow")),                 "\n")
cat("jsonlite:              ", as.character(packageVersion("jsonlite")),              "\n")
RSCRIPT

echo
echo "[ok] R packages installed."
echo "     Python side needs PyYAML: pip install pyyaml"
