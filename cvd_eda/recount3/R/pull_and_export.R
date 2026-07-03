#!/usr/bin/env Rscript
# Pull one recount3 project, run transform_counts(), and export the counts
# matrix + colData + rowData to Parquet. Emits a single-line JSON status record
# prefixed with "RECOUNT3_STATUS_JSON:" on stdout so the Python orchestrator
# can parse it without depending on R's exit code alone.
#
# Usage:
#   pull_and_export.R <project> <project_home> <organism> <output_dir>
#
# Outputs (under <output_dir>):
#   <project>_counts.parquet   gene_id x sample columns, plus gene_id column
#   <project>_coldata.parquet  sample metadata, plus sample_id column
#   <project>_rowdata.parquet  gene metadata (Ensembl id, symbol, biotype, ...)

suppressPackageStartupMessages({
  library(recount3)
  library(SummarizedExperiment)
  library(arrow)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: pull_and_export.R <project> <project_home> <organism> <output_dir>")
}
project      <- args[[1]]
project_home <- args[[2]]
organism     <- args[[3]]
output_dir   <- args[[4]]

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

t0 <- Sys.time()
base_status <- list(
  project      = project,
  project_home = project_home,
  organism     = organism
)

result <- tryCatch({
  proj_df <- available_projects(organism = organism)
  hit <- proj_df$project == project & proj_df$project_home == project_home
  if (sum(hit) == 0) {
    stop(sprintf("Project '%s' with project_home '%s' not found in available_projects()",
                 project, project_home))
  }
  if (sum(hit) > 1) {
    stop(sprintf("Ambiguous project row: %d matches for %s / %s",
                 sum(hit), project, project_home))
  }

  message(sprintf("[%s] create_rse()...", project))
  rse <- create_rse(proj_df[hit, ])

  message(sprintf("[%s] transform_counts()...", project))
  # transform_counts converts recount3's per-base coverage into approximate
  # read counts (scaled by average read length). This is the canonical entry
  # point that DESeq2/edgeR expect.
  assay(rse, "counts") <- transform_counts(rse)

  counts_df  <- as.data.frame(assay(rse, "counts"), stringsAsFactors = FALSE)
  coldata_df <- as.data.frame(colData(rse),          stringsAsFactors = FALSE)
  rowdata_df <- as.data.frame(rowData(rse),          stringsAsFactors = FALSE)

  # Parquet doesn't preserve rownames — promote them to explicit columns so
  # downstream Python code doesn't have to reconstruct them.
  counts_df$gene_id    <- rownames(counts_df)
  coldata_df$sample_id <- rownames(coldata_df)
  rowdata_df$gene_id   <- rownames(rowdata_df)

  counts_path  <- file.path(output_dir, sprintf("%s_counts.parquet",  project))
  coldata_path <- file.path(output_dir, sprintf("%s_coldata.parquet", project))
  rowdata_path <- file.path(output_dir, sprintf("%s_rowdata.parquet", project))

  arrow::write_parquet(counts_df,  counts_path)
  arrow::write_parquet(coldata_df, coldata_path)
  arrow::write_parquet(rowdata_df, rowdata_path)

  list(
    status        = "ok",
    n_samples     = ncol(rse),
    n_genes       = nrow(rse),
    counts_path   = counts_path,
    coldata_path  = coldata_path,
    rowdata_path  = rowdata_path,
    transform     = "recount3::transform_counts (scaled coverage -> counts)",
    coldata_cols  = colnames(coldata_df),
    error         = NA
  )
}, error = function(e) {
  list(
    status = "error",
    error  = conditionMessage(e)
  )
})

status <- c(
  base_status,
  result,
  list(elapsed_sec = as.numeric(difftime(Sys.time(), t0, units = "secs")))
)

cat(sprintf(
  "RECOUNT3_STATUS_JSON: %s\n",
  toJSON(status, auto_unbox = TRUE, null = "null", na = "null")
))

if (identical(result$status, "error")) quit(status = 1)
