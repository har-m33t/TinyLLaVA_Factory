# Vicuna-7B backbone swap — smoke-test status

## Status: **PASSED (wiring, real Vicuna backbone) — swap complete. Full-depth / real-weight run and compute check remain for the CUDA training env.**

`scripts/train/train_bulkformer.sh` now uses the real Vicuna-7B backbone
(`LLM_VERSION=lmsys/vicuna-7b-v1.5`, `CONV_VERSION=llama`). The swap was verified
end-to-end with the **real Vicuna loader path, real Vicuna tokenizer, and real
embedding dimension (4096)** substituted for the tiny-random-Llama used in the
original wiring test. Reproduce: `python -m integration.vicuna_smoke_test`.

The original tower/connector wiring result (`integration/smoke_test_result.md`,
stub encoder) was **not** overwritten — this is its own configuration and its own
result file.

---

## What had to be resolved first (two gaps this fork had)

The upstream-TinyLLaVA assumption that Vicuna is a zero-code, already-registered
path did **not** hold in this fork. Both gaps were confirmed empirically against
the live factories, then fixed with minimal, idiomatic registration:

**Gap 1 — conversation template.** `--conv_version vicuna_v1` is **not** a valid
name here (`TemplateFactory('vicuna_v1') -> AssertionError: vicuna_v1 is not
implmentation`). This repo's Vicuna v1 format is registered under the name
**`llama`** — `tinyllava/data/template/llama_template.py` is verbatim the Vicuna
v1.5 template (system "A chat between a curious user…", `USER:`/`ASSISTANT:` turns,
`</s>` separator). **Fix:** use `--conv_version llama` (no code change; the
template already existed under this name). `repo_findings.md` §9 should record this
name→format mapping, which it currently omits.

**Gap 2 — LLM loader.** `LLMFactory` selects a loader by substring-matching the
checkpoint path against registered keys; before this work the keys were
`gemma/openelm/phi/qwen2/stablelm/tinyllama`, none of which is a substring of
`lmsys/vicuna-7b-v1.5`, so it raised `... is not registered`. The Llama
*architecture* was already available (the `tinyllama` loader returns
`LlamaForCausalLM`) — only name-based selection was missing. **Fix:** added
`tinyllava/model/llm/vicuna.py` — `@register_llm('vicuna')` returning
`LlamaForCausalLM, (AutoTokenizer, tokenizer_and_post_load)`, ~5 lines mirroring
`tinyllama.py`. Any path containing "vicuna" now selects it. Confirmed:
`LLMFactory('lmsys/vicuna-7b-v1.5') -> LlamaForCausalLM`.

## Smoke-test result (real Vicuna backbone substituted)

Same tiny synthetic dataset and connector-only (stage-1) regime as the original
wiring test, run through the **real vicuna loader + real Vicuna tokenizer + real
Vicuna config dimensions**. The BulkFormer encoder stays a shape-correct STUB
(encoder-forward correctness is separately tracked — see the encoder gate in
`smoke_test_result.md`). Observed:

| Probe | Boundary | Observed |
|---|---|---|
| **V-config** | real Vicuna-7B config | `num_layers=32` (real), `hidden=4096`, `vocab=32000`, `intermediate=11008` |
| **V-backbone** | vicuna loader → model | class **`LlamaForCausalLM`**, `hidden=4096`, tokenizer **`LlamaTokenizer`**, `vocab=32000` |
| **V-conv** | `--conv_version llama` | resolves to `LlamaTemplate` (Vicuna v1) |
| **B. Tower** | `BulkFormerVisionTower.forward` | `(3, 20010)` → **`(3, 1, 643)`**, finite; **0** encoder params require grad |
| **A. Data path** | `.npy` branch → collator | `images` **`(4, 20010)`**, `input_ids` `(4, 5)` |
| **C. Connector** | `TranscriptLinearConnector` | `643` → **`4096`** (matches real Vicuna hidden) |
| **C. Train step** | tower → connector → LLM → loss → backward | **loss ≈ 10.724**, connector grad-norm ≈ **11.20**, **0** gradients on the frozen tower |

Assertions passed: backbone is a Llama class; hidden dim is the real Vicuna 4096;
connector output == 4096; `llama` template registered; tower output exactly
`(B, 1, 643)` with the encoder frozen; loss finite; `backward()` succeeds; the
frozen tower receives **no** gradients.

### Scope note (honest boundary of this local pass)

The host has **24 GB RAM**, so the full 32-layer 7B weights (~14–28 GB) cannot be
loaded locally. `num_hidden_layers` was reduced (32→2) **for this local run only**;
`hidden_size`, `vocab_size`, `intermediate_size`, and the tokenizer are the **real
Vicuna-7B values**. Depth is not what changes when swapping the backbone — the
loader path, tokenizer, and embedding dimension are, and all three are real here.
A **full-depth, real-weight** forward/backward should be run on the CUDA training
environment (same place the real-encoder gate from `smoke_test_result.md` runs)
before Stage 1 pretraining — this closes the wiring question, not the
full-weight-load question.

---

## Compute-budget finding (task 3)

**Reference config** (`scripts/train/pretrain.sh`, `finetune.sh`, `scripts/zero3.json`):
DeepSpeed **ZeRO-3, 4 GPUs** (`--include localhost:4,5,6,7`), **fp16**,
flash_attention_2, **no CPU offload**. Frozen BulkFormer tower is CPU-only (~0 GB
on-device); the `transcript_linear` connector (643→4096) is a few M params
(negligible).

Estimates for **Vicuna-7B** (7e9 params), fp16, ZeRO-3 across 4 GPUs, no offload:

| Stage | Trainable | GPU model-state /GPU (÷4) | Batch (per-dev × accum × 4) | Assessment |
|---|---|---|---|---|
| **1 pretrain** | connector only (LLM+tower frozen) | ~3.5 GB (frozen 7B fp16 ÷4) + tiny connector opt states | 32 × 2 × 4 = 256 | Dominated by activations at per-device batch **32**; OK on 40 GB-class, tight on 24 GB. |
| **2 finetune** | **full 7B** + connector | **~28 GB** (16 B/param model states = 112 GB ÷4) **before activations** | 8 × 4 × 4 = 128 | Needs **~40 GB-class GPUs ×4** (A100-40GB / A6000). **Won't fit 24 GB** at batch 8 without gradient checkpointing and/or ZeRO-3 offload. |

Stage-2 model-state figure uses standard mixed-precision full-FT accounting
(~16 B/param: fp32 master + fp16 param + fp16 grad + Adam m/v), sharded ÷4 by
ZeRO-3; activations add ~10–30 GB/GPU at batch 8 × seq 2048 with no activation
checkpointing flag present in the reference scripts.

**Still to check on the target env (stated, not assumed):** the actual GPU count
and **per-GPU VRAM** are unknown to me. Before queuing Stage 2, confirm GPUs are
**≥40 GB/GPU ×4**; if they are 24 GB, enable gradient checkpointing and/or ZeRO-3
offload (`offload_optimizer`/`offload_param` in `zero3.json`) — flag that config
change rather than silently reducing batch size. This is also gated behind the
separate still-open **real-encoder** gate (torch_geometric + torch_sparse +
`BulkFormer-127M.pt`), which must pass on that same environment.

---

## Files changed by this task

- `tinyllava/model/llm/vicuna.py` (new) — `vicuna` LLM loader → `LlamaForCausalLM`.
- `scripts/train/train_bulkformer.sh` — `LLM_VERSION=lmsys/vicuna-7b-v1.5`,
  `CONV_VERSION=llama`, with the compute-budget caveat inline.
- `integration/vicuna_smoke_test.py` (new) — the real-backbone wiring test.
- `integration/vicuna_smoke_test_result.md` (this file). `smoke_test_result.md`
  left untouched.

## Remaining before Stage 1 pretraining (not part of the backbone swap)

1. On the CUDA training env, run `python -m integration.vicuna_smoke_test` with
   full depth / real weights (remove the local layer reduction), and the
   real-encoder gate (`python -m integration.smoke_test --real-encoder`).
2. Confirm GPU VRAM per the compute-budget section above.
