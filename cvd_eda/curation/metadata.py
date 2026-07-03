"""Loaders that flatten ARCHS4 and RECOUNT3 metadata into one shape.

Task 3 doesn't care about dataset-specific field names — the keyword net and
LLM both operate on a single concatenated text blob per sample. Each loader is
responsible for picking the right free-text columns for its dataset and
concatenating them, then returning a DataFrame with a fixed schema:

    sample_id           str  (matches ARCHS4 ``meta/samples/geo_accession`` or
                              RECOUNT3 ``sample_id`` column so Task 4 can join)
    source_series_id    str  (GSE / SRP / GTEx / TCGA project accession; "" if unavailable)
    text                str  (concatenated free-text used for classification)

The rest of the pipeline is dataset-agnostic from that point on.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCHEMA = ("sample_id", "source_series_id", "text")


# ---------------------------------------------------------------------------
# ARCHS4
# ---------------------------------------------------------------------------

# Free-text fields that live under meta/samples/ in the ARCHS4 H5 file. Not
# all releases include every field; we tolerate missing ones. Numeric / QC
# fields (e.g. singlecellprobability) are intentionally excluded.
ARCHS4_TEXT_FIELDS: tuple[str, ...] = (
    "title",
    "source_name_ch1",
    "characteristics_ch1",
    "molecule_ch1",
    "extract_protocol_ch1",
    "description",
    "data_processing",
    "series_title",
    "series_summary",
    "library_source",
    "library_strategy",
    "library_selection",
)

# ``geo_accession`` first: Task 4's ARCHS4 loader keys on it, so Task 3 must
# emit the same identifier or the join will silently drop rows.
ARCHS4_ID_FIELD_CANDIDATES: tuple[str, ...] = (
    "geo_accession",
    "Sample_geo_accession",
    "sample_id",
)
ARCHS4_SERIES_FIELD_CANDIDATES: tuple[str, ...] = (
    "series_id",
    "Sample_series_id",
    "series",
)


def load_archs4(h5_path: str | Path) -> pd.DataFrame:
    """Read ARCHS4 per-sample metadata into the shared schema.

    Uses ``h5py`` directly rather than ``archs4py`` so we don't couple Task 3
    to that package's version-specific accessor names. Byte strings are
    decoded; missing fields are treated as empty.
    """
    import h5py

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        if "meta/samples" not in f:
            raise KeyError(
                f"{h5_path}: expected group 'meta/samples' (ARCHS4 layout). "
                f"Got top-level keys: {list(f.keys())}"
            )
        group = f["meta/samples"]
        available = set(group.keys())

        sample_ids = _pick_and_read(group, ARCHS4_ID_FIELD_CANDIDATES, available)
        if sample_ids is None:
            raise KeyError(
                f"{h5_path}: could not find a sample-id field. Tried "
                f"{ARCHS4_ID_FIELD_CANDIDATES}; available: {sorted(available)}"
            )
        series_ids = _pick_and_read(group, ARCHS4_SERIES_FIELD_CANDIDATES, available)
        if series_ids is None:
            series_ids = [""] * len(sample_ids)

        text_columns: list[list[str]] = []
        for field in ARCHS4_TEXT_FIELDS:
            if field in available:
                text_columns.append(_read_str_dataset(group[field]))
            else:
                text_columns.append(["" for _ in sample_ids])

    n = len(sample_ids)
    for col in text_columns:
        if len(col) != n:
            raise ValueError(
                f"{h5_path}: metadata columns disagree on length ({n} vs {len(col)})"
            )

    text = [" | ".join(v for v in row if v).strip() for row in zip(*text_columns)]
    df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "source_series_id": series_ids,
            "text": text,
        }
    )
    return df.astype({"sample_id": str, "source_series_id": str, "text": str})


def _pick_and_read(group, candidates, available):
    for name in candidates:
        if name in available:
            return _read_str_dataset(group[name])
    return None


def _read_str_dataset(dataset) -> list[str]:
    """Decode an h5py dataset into a list[str], tolerating bytes or object dtypes."""
    arr = dataset[:]
    out: list[str] = []
    for v in arr:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="replace"))
        elif v is None:
            out.append("")
        else:
            out.append(str(v))
    return out


# ---------------------------------------------------------------------------
# RECOUNT3
# ---------------------------------------------------------------------------

# RECOUNT3 coldata columns depend on the source project family (SRA / GTEx /
# TCGA). We take a union of the free-text-y fields likely to describe the
# sample's phenotype. Anything missing in a given parquet is skipped.
RECOUNT3_TEXT_FIELDS: tuple[str, ...] = (
    # SRA-style
    "sra.experiment_title",
    "sra.sample_title",
    "sra.sample_attributes",
    "sra.study_title",
    "sra.study_abstract",
    "sra.library_strategy",
    "sra.library_source",
    "sra.library_selection",
    "study_title",
    "study_abstract",
    # GTEx-style
    "gtex.smtsd",
    "gtex.smts",
    "gtex.smpthnts",
    "gtex.smnabtcht",
    # TCGA-style
    "tcga.gdc_cases.diagnoses.diagnosis",
    "tcga.gdc_cases.project.name",
    "tcga.cgc_case_primary_site",
    "tcga.cgc_case_histological_diagnosis",
    "tcga.cgc_case_clinical_stage",
    # generic
    "characteristics",
)

# Task 2's R export promotes rownames to a real ``sample_id`` column
# (pull_and_export.R:68), so that's the primary candidate.
RECOUNT3_ID_FIELD_CANDIDATES: tuple[str, ...] = (
    "sample_id",
    "external_id",
    "run_acc",
    "rail_id",
    "sra.run_acc",
    "gtex.sampid",
    "tcga.tcga_barcode",
)

RECOUNT3_SERIES_FIELD_CANDIDATES: tuple[str, ...] = (
    "study",
    "project",
    "sra.study_acc",
    "gtex.smstyp",
    "tcga.gdc_cases.project.project_id",
)


def load_recount3(coldata_paths: list[str | Path]) -> pd.DataFrame:
    """Read one or more RECOUNT3 coldata parquets into the shared schema.

    Multiple projects can be merged into one CVD-relevance run — pass all of
    their coldata parquets and they'll be concatenated. sample_id uniqueness
    is enforced across the merged frame (a duplicate is a data bug worth
    surfacing loudly).
    """
    if not coldata_paths:
        raise ValueError("load_recount3: at least one coldata parquet is required")

    frames = [_load_one_recount3(Path(p)) for p in coldata_paths]
    df = pd.concat(frames, ignore_index=True)
    dupes = df["sample_id"][df["sample_id"].duplicated()].unique()
    if len(dupes):
        raise ValueError(
            f"load_recount3: duplicate sample_id across inputs ({len(dupes)} ids); "
            f"first few: {list(dupes[:5])}"
        )
    return df


def _load_one_recount3(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)

    sample_ids = _first_present_column(raw, RECOUNT3_ID_FIELD_CANDIDATES)
    if sample_ids is None:
        raise KeyError(
            f"{path}: could not find a sample-id column. Tried "
            f"{RECOUNT3_ID_FIELD_CANDIDATES}; available: {list(raw.columns)}"
        )
    series_ids = _first_present_column(raw, RECOUNT3_SERIES_FIELD_CANDIDATES)
    if series_ids is None:
        series_ids = pd.Series([""] * len(raw), index=raw.index)

    text_cols = [
        raw[c].astype(str).fillna("") for c in RECOUNT3_TEXT_FIELDS if c in raw.columns
    ]
    if not text_cols:
        raise KeyError(
            f"{path}: no known free-text columns present. Available: {list(raw.columns)}"
        )
    text = text_cols[0].copy()
    for col in text_cols[1:]:
        text = text.str.cat(col, sep=" | ", na_rep="")

    return pd.DataFrame(
        {
            "sample_id": sample_ids.astype(str).values,
            "source_series_id": series_ids.astype(str).values,
            "text": text.str.strip().values,
        }
    )


def _first_present_column(df: pd.DataFrame, candidates: tuple[str, ...]):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None
