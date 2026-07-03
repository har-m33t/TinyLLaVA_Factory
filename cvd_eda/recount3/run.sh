#!/usr/bin/env bash
# Task 2 entrypoint: ingest every project listed in config/candidate_projects.yaml.
#
# Usage:
#   ./run.sh                       # writes to ../data/recount3_raw/
#   ./run.sh /path/to/output_dir   # override output dir
#   ./run.sh /path/to/output_dir --force   # re-ingest even if parquet exists
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${1:-${HERE}/../data/recount3_raw}"
shift || true

CATALOG_PATH="${OUT_ROOT}/available_projects_catalog.parquet"
LOG_PATH="${OUT_ROOT}/ingestion_log_recount3.json"

mkdir -p "${OUT_ROOT}"

python3 "${HERE}/python/orchestrate.py" \
  --config       "${HERE}/config/candidate_projects.yaml" \
  --output-dir   "${OUT_ROOT}" \
  --catalog-path "${CATALOG_PATH}" \
  --log-path     "${LOG_PATH}" \
  "$@"
