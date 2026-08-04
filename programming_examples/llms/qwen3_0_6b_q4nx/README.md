# Qwen3-0.6B Q4NX Inference on AMD NPU2 (MLIR-AIR)

A full **prefill + decode** MLIR-AIR inference for Qwen3-0.6B using **Q4NX**
(4-bit) weights on NPU2 (AIE2P). Same on-device pipeline as the sibling bf16
[`qwen3_0_6b`](../qwen3_0_6b) example — only the **weight source** differs:
FastFlowLM's single `model.q4nx` bundle, dequantized to bf16 on the host at load.

- **Prefill** (reuses `qwen3_0_6b_prefill`) — RMSNorm / QKV / QK-norm / RoPE /
  head-first causal-GQA flash attention / SwiGLU on the NPU with resident weight
  BOs; on-device LM-head GEMV. Fills a per-layer KV cache.
- **Autoregressive decode** (`qwen3_0_6b_decode`) — NPU projections + RoPE +
  GEMVs, CPU attention over the KV cache, NPU LM-head GEMV.
- **Host orchestration** — embedding, final RMSNorm, argmax, chat template, EOS,
  streaming output between tokens.

The gate is the standard `make verify` **top-k token-set inclusion vs HF
transformers bf16** — the apples-to-bf16 signal that AIR faithfully runs the
4-bit model.

## Model Config

28 layers, emb_dim=1024, n_heads=16, head_dim=128 (decoupled: n_heads·head_dim
= 2048 ≠ emb_dim), n_kv_heads=8, hidden_dim=3072, vocab_size=151936,
rope_theta=1e6, **QK-norm** (per-head RMSNorm on Q and K), **no QKV bias**, tied
embeddings (lm_head = embed_tokens).

## The Q4NX codec (I8 / GGUF-Q4_1-derived)

FastFlowLM ships Qwen3 in a different codec than the Llama-3.2 `Q4NX` bundles.
Each projection is stored as dtype `I8` with a **packed** header shape
`(n_chunks, 5120)` (not the real `(out, in)`). Each 32×256 chunk packs 512 B
scale (bf16) + 512 B min (bf16) + 4096 B of 4-bit quants (group_size=32,
parallel=16). Dequant is **`w = scale·q + min`** (additive min), with the row
de-interleave `row_in_block = g·16 + r·2 + b`. `load_q4nx_weights` reads this
into the bf16 `qwen3_0_6b` weight container (validated: dequant vs HF
`Qwen/Qwen3-0.6B` cosine ≈ 0.997 — the expected 4-bit fidelity). Writing this
loader is the only new component; the rest is the bf16 `qwen3_0_6b` driver.

## Prerequisites

1. **MLIR-AIR base environment** — AMD NPU2, Peano (`PEANO_INSTALL_DIR`),
   `source utils/env_setup.sh ...`.
2. **Extra Python packages**: `pip install -r requirements.txt` (`safetensors`,
   `huggingface_hub`, `transformers`, `torch`).
3. **HuggingFace access** — `Qwen/Qwen3-0.6B` (tokenizer + `make verify` bf16
   reference) and `FastFlowLM/Qwen3-0.6B-NPU2` (the `model.q4nx` bundle) are
   ungated and auto-download on first use.

## Data

**One weight source — one HuggingFace download.** `MODEL_SOURCE` /
`Q4NX_MODEL_SOURCE` (default `FastFlowLM/Qwen3-0.6B-NPU2`) is a `model.q4nx`
safetensors file with per-layer Q4NX projections + bf16 norms/QK-norm/embed,
dequantized on the host at load. May also be a local dir/file. `tie_word_embeddings=true`,
so the LM head is the full-precision `embed_tokens` (the bundle's separate I8
lm_head is ignored). Nothing compiled is committed — `make compile` reproduces
every ELF from source (see `.gitignore`).

## Quick Start

```bash
# One-time: compile all prefill + decode kernels (no weights needed)
make compile

# Run inference (streams up to 1000 tokens)
make run PROMPT="How does photosynthesis work?"

# Profiling breakdown / interactive chat
make profile
make chat

# Top-k token-set correctness gate: NPU q4nx vs HF bf16
make verify

# Full sweep / per-layer cosine diagnosis
make verify-full
make diagnosis
```

Override the NPU weight source (e.g. a local bundle):

```bash
make run MODEL_SOURCE=/path/to/Qwen3-0.6B-NPU2/model.q4nx
```

## Correctness

`make verify` greedily decodes 32 tokens on the NPU (Q4NX) and on HF transformers
(bf16) per prompt and checks that every NPU token is in HF's top-5 set at that
position (2-prompt CI gate; `make verify-full` for the full sweep). A lone
near-miss on the most quant-sensitive prompt is expected 4-bit drift, not a bug
(HF's token stays within the NPU top-k).

## Key Files

| Path | Purpose |
|---|---|
| `qwen3_0_6b_q4nx_weights.py` | I8/GGUF-Q4_1 dequant (`w = scale·q + min`) → bf16 into the `qwen3_0_6b` LlamaWeights container |
| `qwen3_0_6b_q4nx_inference.py` | Thin driver: reuses the `qwen3_0_6b` runtime + generation loop; swaps the weight source + tokenizer |
| `verify_adapter.py` | Hooks into the shared `../verify/` subsystem; NPU q4nx vs HF bf16 gate |
| `Makefile` | compile / run / profile / chat / verify / verify-full / diagnosis / clean |

Cross-directory reuse: the sibling [`qwen3_0_6b`](../qwen3_0_6b) prefill + decode +
inference drivers, and the shared `llms/` infra + `verify/` subsystem.
