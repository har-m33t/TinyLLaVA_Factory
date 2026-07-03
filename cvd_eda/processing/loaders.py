"""Read raw counts + metadata from each upstream dataset into a common shape.

Every loader returns a :class:`RawDataset`. The rest of the pipeline is
loader-agnostic and only reads that struct.

Shape contract
--------------
counts:     ``(n_genes, n_samples)`` DataFrame of integer read counts.
            ``index.name == "gene_id_raw"``, ``columns.name == "sample_id"``.
sample_meta: DataFrame indexed by ``sample_id`` (matches counts columns).
gene_meta:   DataFrame indexed by ``gene_id_raw`` (matches counts index),
            with at least one of ``ensembl_id`` / ``symbol`` columns.
gene_id_scheme: ``"ensembl"`` or ``"symbol"`` — drives harmonization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class RawDataset:
    name: str
    counts: pd.DataFrame
    sample_meta: pd.DataFrame
    gene_meta: pd.DataFrame
    gene_id_scheme: str


def _decode_bytes(arr) -> np.ndarray:
    return np.array(
        [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]
    )


def load_archs4(
    h5_path: Path,
    sample_ids: Optional[list[str]] = None,
) -> RawDataset:
    """Load an ARCHS4 human/mouse H5 file.

    ARCHS4 v2.x layout (as of ``human_gene_v2.5.h5``):
        ``/data/expression``                uint32 ``(n_genes, n_samples)``
        ``/meta/genes/symbol``              str ``(n_genes,)``
        ``/meta/genes/ensembl_gene_id``     str ``(n_genes,)``  (v2.x+)
        ``/meta/samples/geo_accession``     str ``(n_samples,)``
        ``/meta/samples/series_id``         str ``(n_samples,)``
        ``/meta/samples/title``             str ``(n_samples,)``

    If ``sample_ids`` is given, only those columns are pulled off disk. ARCHS4
    ships ~1M samples; we typically want a few thousand CVD-relevant ones.
    """
    import h5py  # local import: ingestion-agent deps not required just to import this module

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        geo = _decode_bytes(f["meta/samples/geo_accession"][:])

        def _optional(key: str) -> np.ndarray:
            return _decode_bytes(f[key][:]) if key in f else np.array([""] * len(geo))

        series = _optional("meta/samples/series_id")
        title = _optional("meta/samples/title")

        symbol = _decode_bytes(f["meta/genes/symbol"][:])
        if "meta/genes/ensembl_gene_id" in f:
            ensembl = _decode_bytes(f["meta/genes/ensembl_gene_id"][:])
            gene_id_scheme = "ensembl"
            gene_id_raw = ensembl
        elif "meta/genes/ensembl_id" in f:
            ensembl = _decode_bytes(f["meta/genes/ensembl_id"][:])
            gene_id_scheme = "ensembl"
            gene_id_raw = ensembl
        else:
            ensembl = np.array([""] * len(symbol))
            gene_id_scheme = "symbol"
            gene_id_raw = symbol

        if sample_ids is not None:
            wanted = set(str(s) for s in sample_ids)
            keep_mask = np.isin(geo, list(wanted))
            keep_idx = np.where(keep_mask)[0]
            if len(keep_idx) == 0:
                raise ValueError(
                    f"None of the {len(sample_ids)} requested sample_ids were "
                    f"found in ARCHS4 file {h5_path}."
                )
            # h5py fancy indexing along a single axis requires sorted, unique indices
            keep_idx_sorted = np.sort(keep_idx)
            expr = f["data/expression"][:, keep_idx_sorted]
            sample_ids_final = geo[keep_idx_sorted]
            series_final = series[keep_idx_sorted]
            title_final = title[keep_idx_sorted]
        else:
            expr = f["data/expression"][:]
            sample_ids_final = geo
            series_final = series
            title_final = title

    counts = pd.DataFrame(
        expr,
        index=pd.Index(gene_id_raw, name="gene_id_raw"),
        columns=pd.Index(sample_ids_final, name="sample_id"),
    )
    sample_meta = pd.DataFrame(
        {"series_id": series_final, "title": title_final},
        index=pd.Index(sample_ids_final, name="sample_id"),
    )
    gene_meta = pd.DataFrame(
        {"symbol": symbol, "ensembl_id": ensembl},
        index=pd.Index(gene_id_raw, name="gene_id_raw"),
    )
    return RawDataset(
        name="archs4",
        counts=counts,
        sample_meta=sample_meta,
        gene_meta=gene_meta,
        gene_id_scheme=gene_id_scheme,
    )


def load_recount3_project(
    counts_parquet: Path,
    coldata_parquet: Path,
) -> RawDataset:
    """Load one RECOUNT3 project export produced by Task 2.

    Task 2 writes ``assay(rse, "counts")`` as a Parquet with genes as the
    outer axis and samples as the inner axis, plus ``colData(rse)`` as a
    sample-indexed Parquet. Orientation is detected by matching one of the
    two axes against ``coldata`` — if a user's R export transposed the
    matrix, we transpose it back rather than fail.
    """
    counts_parquet = Path(counts_parquet)
    coldata_parquet = Path(coldata_parquet)

    counts = pd.read_parquet(counts_parquet)
    coldata = pd.read_parquet(coldata_parquet)

    coldata_ids = set(coldata.index.astype(str))
    col_ids = set(counts.columns.astype(str))
    idx_ids = set(counts.index.astype(str))

    if col_ids & coldata_ids and not (idx_ids & coldata_ids):
        pass  # already (genes × samples)
    elif idx_ids & coldata_ids and not (col_ids & coldata_ids):
        counts = counts.T
    else:
        raise ValueError(
            f"Cannot orient {counts_parquet.name} against {coldata_parquet.name}: "
            f"neither axis matches coldata's sample index. "
            f"counts.shape={counts.shape}, len(coldata)={len(coldata)}."
        )

    counts.index = counts.index.astype(str)
    counts.columns = counts.columns.astype(str)
    counts.index.name = "gene_id_raw"
    counts.columns.name = "sample_id"

    project_id = counts_parquet.stem.replace("_counts", "")
    gene_meta = pd.DataFrame(
        {"ensembl_id": counts.index},
        index=pd.Index(counts.index, name="gene_id_raw"),
    )
    return RawDataset(
        name=f"recount3:{project_id}",
        counts=counts,
        sample_meta=coldata,
        gene_meta=gene_meta,
        gene_id_scheme="ensembl",
    )
