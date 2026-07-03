#!/usr/bin/env Rscript
# Enumerate every human project in recount3 and dump the catalog to Parquet.
# Downstream steps (filtering, per-project pulls) read this catalog rather than
# calling available_projects() themselves, so the catalog snapshot is stable
# across the run.

suppressPackageStartupMessages({
  library(recount3)
  library(arrow)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: enumerate_projects.R <output_parquet_path> [organism]")
}
out_path <- args[[1]]
organism <- if (length(args) >= 2) args[[2]] else "human"

message(sprintf("Fetching available_projects(organism=%s) from recount3...", organism))
projects <- available_projects(organism = organism)
projects_df <- as.data.frame(projects)

message(sprintf("Retrieved %d projects across %d project_home buckets.",
                nrow(projects_df),
                length(unique(projects_df$project_home))))

dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
arrow::write_parquet(projects_df, out_path)
message(sprintf("Wrote catalog -> %s", out_path))
