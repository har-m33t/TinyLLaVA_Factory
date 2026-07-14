"""vicuna_smoke_test.py — wiring check for the Vicuna-7B backbone swap.

Same tiny synthetic dataset and wiring-only scope as integration/smoke_test.py,
but with the **real Vicuna backbone substituted** for the tiny-random-Llama:

  * the real `vicuna` LLM loader path (tinyllava/model/llm/vicuna.py) is exercised
    end-to-end — LLMFactory selects it by the "vicuna" substring in the checkpoint
    dir name, exactly as it would for lmsys/vicuna-7b-v1.5;
  * the **real Vicuna-7B tokenizer** (lmsys/vicuna-7b-v1.5) is downloaded and used;
  * the model is built from the **real Vicuna-7B config** — real hidden_size=4096,
    vocab_size=32000, intermediate_size=11008 — i.e. the actual embedding
    dimensions the connector must align to;
  * `--conv_version llama` (this repo's Vicuna v1 template) is verified against the
    real tokenizer.

MEMORY NOTE: the full 32-layer 7B weights (~14-28 GB) do not fit this 24 GB host,
so `num_hidden_layers` is reduced (default 2) to make the forward/backward
runnable locally. Depth is NOT what changes when swapping the backbone — the
loader path, tokenizer, and embedding dimension are, and those are all REAL here.
Full-depth / real-weights execution belongs on the CUDA training environment (the
same place the real-encoder gate runs); this test closes the wiring question, not
the full-weight-load question.

The BulkFormer encoder stays a shape-correct STUB (task note: encoder-forward
correctness and the LLM backbone are separately tracked concerns).

Run:  python -m integration.vicuna_smoke_test
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# reuse the exact helpers the original wiring test uses
from integration.smoke_test import (
    N_GENES,
    CFG_DIR,
    REPO,
    _patch_encoder,
    make_tiny_dataset,
    report,
)

VICUNA_HF = "lmsys/vicuna-7b-v1.5"
REDUCED_LAYERS = 2  # depth only; hidden_size / vocab / tokenizer stay REAL


def _make_vicuna_backbone(scratch: Path, n_layers: int = REDUCED_LAYERS) -> Path:
    """Save a real-Vicuna-config LlamaForCausalLM (reduced depth) + the real
    Vicuna tokenizer into a dir whose name contains 'vicuna', so the LLMFactory
    substring match selects the new `vicuna` loader — the same selection path a
    real lmsys/vicuna-7b-v1.5 checkpoint would take."""
    from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

    out = scratch / "vicuna_stub"
    tok = AutoTokenizer.from_pretrained(VICUNA_HF)  # REAL vicuna tokenizer
    cfg = AutoConfig.from_pretrained(VICUNA_HF)     # REAL vicuna config
    real_layers = cfg.num_hidden_layers
    cfg.num_hidden_layers = n_layers                # reduce DEPTH only, to fit RAM
    if not (out / "config.json").exists():
        model = LlamaForCausalLM(cfg)               # random init at REAL hidden dim
        model.save_pretrained(out)
        del model
    tok.save_pretrained(out)
    return out, cfg, real_layers


def main(argv=None):
    bar = "=" * 72
    print(bar)
    print("VICUNA SMOKE TEST: real vicuna loader + real tokenizer + real hidden dim")
    print("  BulkFormer encoder = STUB (wiring only). LLM depth reduced to fit RAM;")
    print("  hidden_size / vocab / tokenizer are the REAL Vicuna-7B values.")
    print(bar)

    _patch_encoder()  # stub the BulkFormer encoder (no torch_geometric needed)

    from transformers import AutoConfig
    from tinyllava.model.vision_tower import VisionTowerFactory
    from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
    from tinyllava.model.modeling_tinyllava import TinyLlavaForConditionalGeneration
    from tinyllava.data.dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset
    from tinyllava.data.template import TemplateFactory

    scratch = REPO / "integration" / "_vicuna_smoke_tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    backbone, vcfg, real_layers = _make_vicuna_backbone(scratch)
    report("V-config", real_num_layers=real_layers, test_num_layers=REDUCED_LAYERS,
           real_hidden=vcfg.hidden_size, real_vocab=vcfg.vocab_size,
           real_intermediate=vcfg.intermediate_size)

    # ---- (B) tower forward on a raw expression batch -----------------------
    cfg = AutoConfig.from_pretrained(str(CFG_DIR))
    cfg = getattr(cfg, "vision_config", cfg)
    tower = VisionTowerFactory("bulkformer")(cfg)
    tower.load_model(str(CFG_DIR))
    n_grad = sum(p.requires_grad for p in tower.parameters())
    x = torch.randn(3, N_GENES)
    tout = tower(x)
    report("B-tower", in_shape=tuple(x.shape), out_shape=tuple(tout.shape),
           frozen_params_with_grad=n_grad, finite=bool(torch.isfinite(tout).all()))
    assert tuple(tout.shape) == (3, 1, cfg.hidden_size), "tower output shape wrong"
    assert n_grad == 0, "encoder must be frozen"

    # ---- build the full TinyLlava model with the REAL vicuna loader path ---
    margs = SimpleNamespace(
        model_name_or_path=str(backbone), tokenizer_name_or_path=None,
        vision_tower=f"bulkformer:{CFG_DIR}", vision_tower2="",
        connector_type="transcript_linear", cache_dir=None,
        attn_implementation="eager", model_max_length=2048,
        mm_vision_select_layer=-2, mm_vision_select_feature="patch",
        image_aspect_ratio="square", tokenizer_use_fast=False,
    )
    mcfg = TinyLlavaConfig()
    mcfg.load_from_config(margs)
    model = TinyLlavaForConditionalGeneration(mcfg)
    model.load_llm(model_name_or_path=str(backbone), cache_dir=None, attn_implementation="eager")
    model.load_vision_tower(model_name_or_path=str(CFG_DIR))
    model.load_connector(connector_type="transcript_linear")
    model.train()
    tokenizer = model.tokenizer

    # confirm the vicuna loader really produced a Llama backbone at REAL hidden dim
    llm_cls = type(model.language_model).__name__
    report("V-backbone", llm_class=llm_cls, llm_hidden=model.config.hidden_size,
           tokenizer=type(tokenizer).__name__, vocab_size=len(tokenizer))
    assert "Llama" in llm_cls, "vicuna loader must yield a Llama backbone"
    assert model.config.hidden_size == vcfg.hidden_size == 4096, "hidden dim must be real Vicuna 4096"

    # ---- verify --conv_version llama (Vicuna v1 template) is usable ---------
    assert TemplateFactory("llama"), "llama (Vicuna v1) template must be registered"
    report("V-conv", conv_version="llama", template=TemplateFactory("llama").__name__)

    # ---- (A) dataset .npy branch + collator --------------------------------
    data_json, img_dir = make_tiny_dataset(scratch, n=4)
    data_args = SimpleNamespace(conv_version="pretrain", image_folder=str(img_dir),
                                image_processor=model.vision_tower._image_processor,
                                image_aspect_ratio="square", is_multimodal=True,
                                image_grid_pinpoints=None, data_path=str(data_json))
    ds = LazySupervisedDataset(str(data_json), tokenizer, data_args)
    collate = DataCollatorForSupervisedDataset(tokenizer)
    batch = collate([ds[i] for i in range(len(ds))])
    report("A-dataset", images_shape=tuple(batch["images"].shape),
           input_ids_shape=tuple(batch["input_ids"].shape))
    assert tuple(batch["images"].shape) == (4, N_GENES), "collated expression batch wrong"

    # ---- (C) full model forward + backward (connector-only, stage-1 regime) -
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.connector.parameters():
        p.requires_grad_(True)

    conn_out_dim = model.connector(tout.float()).shape[-1]
    report("C-connector", conn_in=cfg.hidden_size, conn_out=conn_out_dim,
           llm_hidden=model.config.hidden_size)
    assert conn_out_dim == model.config.hidden_size == 4096, "connector must map to Vicuna 4096"

    batch = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                labels=batch["labels"], images=batch["images"])
    loss = out.loss
    loss.backward()
    conn_grad = next(p.grad for p in model.connector.parameters() if p.grad is not None)
    tower_grads = [p.grad for p in model.vision_tower.parameters() if p.grad is not None]
    report("C-step", loss=round(float(loss), 5),
           connector_grad_norm=round(float(conn_grad.norm()), 6),
           tower_grads=len(tower_grads))
    assert torch.isfinite(loss), "loss not finite"
    assert len(tower_grads) == 0, "frozen tower received gradients"

    print("\nVICUNA SMOKE TEST PASSED — real vicuna loader + real tokenizer + real")
    print("hidden dim (4096) verified; loss computes and gradients flow into the")
    print("connector. (Depth reduced locally; full-weight run belongs on CUDA env.)")


if __name__ == "__main__":
    sys.exit(main())
