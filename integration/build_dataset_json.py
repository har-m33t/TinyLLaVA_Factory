"""build_dataset_json.py — convert the CVD-pool samples into a TinyLLaVA
training set for the BulkFormer tower.

For each selected sample this:
  1. reads raw ARCHS4 counts,
  2. runs the same normalize→TPM→log1p→20,010-gene-vocab alignment as the
     linear-probe stage (reusing `linear_probe/extract.py`), and
  3. writes the resulting `[20010]` float32 vector to `<image_folder>/<id>.npy`.

It then emits two JSON files in TinyLLaVA's confirmed schema (see
`integration/repo_findings.md` §1) — one for stage-1 connector alignment
(`pretrain`) and one for stage-2 instruction finetune (`finetune`) — that
reference those `.npy` files via the literal `"image"` field. The data loader's
`.npy` branch (`tinyllava/data/dataset.py`) loads the vector directly and the
BulkFormer tower consumes it.

CLI
---
    # small balanced set: 200 positives + 200 negatives
    python -m integration.build_dataset_json --n-per-class 200

    # tiny synthetic-free real set for a quick end-to-end check
    python -m integration.build_dataset_json --n-per-class 8 --outdir integration/data_tiny
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from linear_probe.extract import (  # noqa: E402  (path injected above)
    ArchS4CountReader,
    DEFAULT_H5,
    DEFAULT_LABELS,
    load_bulkformer_vocab,
    normalize_and_align,
)

QUESTION = ("<image>\nDoes this transcriptome sample show evidence of "
            "cardiovascular disease? Answer with 'Yes' or 'No'.")


def _log() -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    return logging.getLogger("integration.build_dataset_json")


def select_samples(labels_path: Path, n_per_class: int, neg_pool: str,
                   seed: int, logger: logging.Logger) -> pd.DataFrame:
    labels = pd.read_parquet(labels_path)
    pos = labels.loc[labels["is_positive"]]
    neg_col = "is_neg_hard" if neg_pool == "hard" else "is_neg_whole_corpus"
    neg = labels.loc[labels[neg_col]]

    rng = np.random.default_rng(seed)
    pos = pos.iloc[rng.permutation(len(pos))[:n_per_class]].copy()
    neg = neg.iloc[rng.permutation(len(neg))[:n_per_class]].copy()
    pos["label"] = True
    neg["label"] = False
    sel = pd.concat([pos, neg], ignore_index=True)
    sel = sel.sort_values("sample_index").reset_index(drop=True)
    logger.info(f"selected {len(pos)} positives + {len(neg)} negatives "
                f"(neg pool: {neg_col})")
    return sel


def write_vectors(sel: pd.DataFrame, h5_path: Path, image_dir: Path,
                  batch_size: int, logger: logging.Logger) -> list[str]:
    vocab, length_dict = load_bulkformer_vocab(logger)
    reader = ArchS4CountReader(h5_path, sel["sample_index"].to_numpy(), logger)
    image_dir.mkdir(parents=True, exist_ok=True)

    positions = sel["sample_index"].to_numpy()
    ids = sel["geo_accession"].astype(str).to_numpy()
    filenames: list[str] = []
    for start in range(0, len(sel), batch_size):
        end = min(start + batch_size, len(sel))
        counts = reader.read_batch(positions[start:end])            # [b, N_h5]
        aligned, _ = normalize_and_align(counts, reader.h5_gene_symbols,
                                         vocab, length_dict, logger)  # [b, 20010]
        for row, sample_id in zip(aligned, ids[start:end]):
            fname = f"{sample_id}.npy"
            np.save(image_dir / fname, row.astype(np.float32))
            filenames.append(fname)
        logger.info(f"wrote {end}/{len(sel)} expression vectors")
    return filenames


def build_json(sel: pd.DataFrame, filenames: list[str]) -> tuple[list, list]:
    pretrain, finetune = [], []
    for (_, rec), fname in zip(sel.iterrows(), filenames):
        answer = "Yes" if rec["label"] else "No"
        caption = ("Cardiovascular-disease transcriptome." if rec["label"]
                   else "Non-cardiovascular transcriptome.")
        sample_id = str(rec["geo_accession"])
        pretrain.append({
            "id": sample_id,
            "image": fname,
            "conversations": [
                {"from": "human", "value": "<image>"},
                {"from": "gpt", "value": caption},
            ],
        })
        finetune.append({
            "id": sample_id,
            "image": fname,
            "conversations": [
                {"from": "human", "value": QUESTION},
                {"from": "gpt", "value": answer},
            ],
        })
    return pretrain, finetune


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build TinyLLaVA JSON + .npy vectors for BulkFormer.")
    p.add_argument("--h5", type=Path, default=DEFAULT_H5)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--outdir", type=Path, default=HERE / "data")
    p.add_argument("--n-per-class", type=int, default=200)
    p.add_argument("--neg-pool", choices=["whole", "hard"], default="whole")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260707)
    args = p.parse_args(argv)

    logger = _log()
    args.outdir.mkdir(parents=True, exist_ok=True)
    image_dir = args.outdir / "images"

    sel = select_samples(args.labels, args.n_per_class, args.neg_pool, args.seed, logger)
    filenames = write_vectors(sel, args.h5, image_dir, args.batch_size, logger)
    pretrain, finetune = build_json(sel, filenames)

    (args.outdir / "pretrain.json").write_text(json.dumps(pretrain, indent=2))
    (args.outdir / "finetune.json").write_text(json.dumps(finetune, indent=2))
    logger.info(f"wrote {len(pretrain)} samples → {args.outdir}/pretrain.json + finetune.json; "
                f"vectors in {image_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
