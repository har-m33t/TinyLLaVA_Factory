"""verify_checkpoints.py — step 1 of the linear-probe stage.

Load each of the 5 BulkFormer checkpoints we downloaded, confirm each loads
cleanly with the correct parameter count, and run one forward pass on
synthetic input to confirm the output tensor shape.

This is the gate that keeps a bad-checkpoint problem cheap: fail here rather
than after we've spent hours extracting embeddings across ~40k samples.

Writes `linear_probe/checkpoint_verification.json` with per-variant results
and exits non-zero if any variant fails. Downstream steps must not run if this
step reports any failure.

Runs CPU-only by default (macOS), which is enough for a handful of forward
passes; pass --device mps or --device cuda if available.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BULKFORMER_REPO = REPO / "bulkencoders" / "BulkFormer"
CHECKPOINTS_ROOT = REPO / "bulkencoders" / "checkpoints" / "bulkformer"

# BulkFormer.py imports as `from utils.BulkFormer_block import ...`, i.e. a
# top-level `utils` package — we have to put the repo root on sys.path so those
# imports resolve, otherwise the class won't load.
sys.path.insert(0, str(BULKFORMER_REPO))


@dataclass(frozen=True)
class Variant:
    name: str          # e.g. "BulkFormer-37M"
    ckpt_filename: str
    dim: int
    p_repeat: int


# From bulkencoders/BulkFormer/model/config.py — the five parameter scales.
# Fixed across all variants: bins=0, gb_repeat=1, bin_head=12, full_head=8,
# gene_length=20010.
VARIANTS: tuple[Variant, ...] = (
    Variant("BulkFormer-37M",  "BulkFormer-37M.pt",  dim=128, p_repeat=1),
    Variant("BulkFormer-50M",  "BulkFormer-50M.pt",  dim=256, p_repeat=2),
    Variant("BulkFormer-93M",  "BulkFormer-93M.pt",  dim=512, p_repeat=6),
    Variant("BulkFormer-127M", "BulkFormer-127M.pt", dim=640, p_repeat=8),
    Variant("BulkFormer-147M", "BulkFormer-147M.pt", dim=640, p_repeat=12),
)

FIXED_PARAMS = {"bins": 0, "gb_repeat": 1, "bin_head": 12, "full_head": 8, "gene_length": 20010}

# Nominal parameter counts as advertised by BulkFormer's model/README.md, in
# millions. Empirically these counts include the shared ~25.6M-parameter ESM2
# gene-embedding buffer (`esm2_feature_concat.pt`, 102 MB float32), which is
# loaded as a constant input rather than a trainable model parameter — so
# `sum(p.numel())` on the module alone is systematically ~25M below every
# nominal. We compare `(trainable + ESM2) vs nominal` with a ±10% tolerance;
# expect a ~5–10% residual gap driven by the sparse gene graph buffer and
# performer-attention specifics that also aren't counted in `sum(p.numel())`.
NOMINAL_PARAMS_M = {
    "BulkFormer-37M": 37, "BulkFormer-50M": 50, "BulkFormer-93M": 93,
    "BulkFormer-127M": 127, "BulkFormer-147M": 147,
}
PARAM_COUNT_TOL = 0.10


def _log() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("linear_probe.verify_checkpoints")


def _load_graph_and_gene_emb(device: torch.device, logger: logging.Logger):
    """Load the shared support tensors: gene-gene graph (SparseTensor) + ESM2 gene embeddings.

    All five variants share these — build once, reuse.
    """
    from torch_geometric.typing import SparseTensor

    support = CHECKPOINTS_ROOT / "support"
    logger.info(f"loading support tensors from {support}")

    graph_rc = torch.load(support / "G_tcga.pt", map_location="cpu", weights_only=False)
    graph_w  = torch.load(support / "G_tcga_weight.pt", map_location="cpu", weights_only=False)
    # The notebook transposes: SparseTensor(row=graph[1], col=graph[0], value=weights).t()
    graph = SparseTensor(row=graph_rc[1], col=graph_rc[0], value=graph_w).t().to(device)

    gene_emb = torch.load(support / "esm2_feature_concat.pt", map_location="cpu", weights_only=False)
    logger.info(f"  ESM2 gene embedding: shape={list(gene_emb.shape)}, "
                f"numel={gene_emb.numel():,} ({gene_emb.numel() / 1e6:.2f}M)")
    return graph, gene_emb


def _build_model(variant: Variant, graph, gene_emb, device: torch.device):
    """Instantiate one BulkFormer variant with its size-specific hyperparameters."""
    from utils.BulkFormer import BulkFormer  # only importable after sys.path hack above
    params = {"dim": variant.dim, "p_repeat": variant.p_repeat,
              "graph": graph, "gene_emb": gene_emb, **FIXED_PARAMS}
    return BulkFormer(**params).to(device)


def _load_state_dict(model: torch.nn.Module, ckpt_path: Path):
    """Load a checkpoint, stripping the `module.` prefix left over from DDP training."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    fixed = OrderedDict()
    for k, v in raw.items():
        fixed[k[7:] if k.startswith("module.") else k] = v
    return model.load_state_dict(fixed, strict=True)


def _verify_one(variant: Variant, graph, gene_emb, device: torch.device,
                batch_size: int, logger: logging.Logger) -> dict:
    """Run the full verification protocol for one variant. Any exception is captured."""
    result: dict = {"variant": variant.name, "dim": variant.dim, "p_repeat": variant.p_repeat}
    ckpt_path = CHECKPOINTS_ROOT / "models" / variant.ckpt_filename
    logger.info(f"[{variant.name}] verifying {ckpt_path.name}")

    try:
        t0 = time.perf_counter()
        model = _build_model(variant, graph, gene_emb, device)
        result["build_seconds"] = round(time.perf_counter() - t0, 2)

        t0 = time.perf_counter()
        load_result = _load_state_dict(model, ckpt_path)
        result["load_seconds"] = round(time.perf_counter() - t0, 2)
        result["missing_keys"]    = list(load_result.missing_keys)
        result["unexpected_keys"] = list(load_result.unexpected_keys)

        trainable_count = sum(p.numel() for p in model.parameters())
        esm2_count = gene_emb.numel()
        nominal_check_count = trainable_count + esm2_count
        result["trainable_param_count"] = trainable_count
        result["trainable_param_count_millions"] = round(trainable_count / 1e6, 2)
        result["esm2_buffer_millions"] = round(esm2_count / 1e6, 2)
        result["nominal_check_millions"] = round(nominal_check_count / 1e6, 2)
        nominal = NOMINAL_PARAMS_M[variant.name]
        result["nominal_advertised_millions"] = nominal
        rel_err = abs(nominal_check_count / 1e6 - nominal) / nominal
        result["nominal_check_relative_error"] = round(rel_err, 4)
        param_count_ok = rel_err <= PARAM_COUNT_TOL

        model.eval()
        gene_length = FIXED_PARAMS["gene_length"]
        # Synthetic input in the "log(TPM+1)"-like range — positive floats, sparse-ish.
        rng = torch.Generator(device="cpu").manual_seed(20260707)
        x = torch.rand((batch_size, gene_length), generator=rng) * 8.0  # ~[0, 8)
        x = x.to(device)

        with torch.no_grad():
            t0 = time.perf_counter()
            emb = model(x, mask_prob=0.0, output_expr=False)
            result["forward_seconds"] = round(time.perf_counter() - t0, 2)

        result["output_shape"] = list(emb.shape)
        expected_shape = [batch_size, gene_length, variant.dim + 3]  # +3 for aux feats concat
        result["expected_shape"] = expected_shape
        shape_ok = list(emb.shape) == expected_shape

        # Also confirm the model isn't producing degenerate output (all-nan, all-zero,
        # or all-same). Loading with a shape-matching but wrong state_dict can
        # sometimes pass the shape check while yielding garbage.
        emb_cpu = emb.detach().float().cpu()
        result["output_has_nan"]  = bool(torch.isnan(emb_cpu).any().item())
        result["output_has_inf"]  = bool(torch.isinf(emb_cpu).any().item())
        result["output_std"]      = float(emb_cpu.std().item())
        result["output_mean"]     = float(emb_cpu.mean().item())
        output_healthy = (not result["output_has_nan"] and not result["output_has_inf"]
                         and result["output_std"] > 1e-4)

        result["load_ok"]         = not result["missing_keys"] and not result["unexpected_keys"]
        result["param_count_ok"]  = param_count_ok
        result["shape_ok"]        = shape_ok
        result["output_healthy"]  = output_healthy
        result["pass"] = all([result["load_ok"], param_count_ok, shape_ok, output_healthy])

        # Free the model before moving to the next variant — 147M is ~500MB, and we
        # don't want them all resident simultaneously.
        del model, emb, emb_cpu
        return result

    except Exception as e:  # noqa: BLE001 — we do want to capture anything the load can throw
        logger.exception(f"[{variant.name}] verification raised")
        result["pass"] = False
        result["exception_type"] = type(e).__name__
        result["exception_message"] = str(e)
        result["traceback"] = traceback.format_exc()
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify BulkFormer checkpoints (step 1).")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Synthetic-input batch size for the forward pass. Small by default — "
                             "this is a load/shape check, not a benchmark.")
    parser.add_argument("--out", type=Path, default=HERE / "checkpoint_verification.json")
    args = parser.parse_args(argv)

    logger = _log()
    device = torch.device(args.device)
    logger.info(f"device: {device}, batch_size: {args.batch_size}")

    graph, gene_emb = _load_graph_and_gene_emb(device, logger)

    results = []
    for v in VARIANTS:
        r = _verify_one(v, graph, gene_emb, device, args.batch_size, logger)
        summary = "PASS" if r.get("pass") else "FAIL"
        logger.info(f"[{v.name}] {summary} (trainable={r.get('trainable_param_count_millions', '?')}M, "
                    f"nominal_check={r.get('nominal_check_millions', '?')}M vs advertised "
                    f"{r.get('nominal_advertised_millions', '?')}M, "
                    f"shape={r.get('output_shape', '?')})")
        results.append(r)

    all_pass = all(r["pass"] for r in results)
    manifest = {
        "device": str(device),
        "batch_size": args.batch_size,
        "torch_version": torch.__version__,
        "param_count_tolerance": PARAM_COUNT_TOL,
        "all_pass": all_pass,
        "results": results,
    }
    args.out.write_text(json.dumps(manifest, indent=2))
    logger.info(f"wrote {args.out}")

    if not all_pass:
        logger.error("one or more variants failed — do NOT proceed to steps 2+")
        return 1
    logger.info("all 5 checkpoints verified — step 1 gate cleared")
    return 0


if __name__ == "__main__":
    sys.exit(main())
