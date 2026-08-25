# LFM2-1.2B Q4_0 chatbot on AMD NPU2

This implementation reimplements the corresponding AMD NPU LLM design,
originally developed by the [FastFlowLM](https://github.com/ROCm/FastFlowLM)
team, using the higher-level abstractions of the MLIR-AIR dialect.

A full **prefill + decode** MLIR-AIR inference for
[`LiquidAI/LFM2-1.2B`](https://huggingface.co/LiquidAI/LFM2-1.2B) using **Q4_0**
weights (per-block symmetric 4-bit, `w = q * scale`), exposed as an interactive
chatbot on NPU2 (AIE2P).

**LFM2 is the first non-pure-transformer model in `llms/`.** Only 6 of its 16
layers are attention; the other 10 are `Lfm2ShortConv`, a gated causal depthwise
convolution — and the schedule is irregular (attention at 2, 5, 8, 10, 12, 14),
so nothing here may be driven off a modulo.

- **Batched prefill** (`lfm2_1_2b_q4nx_prefill.py`) — dequant Q4_0→bf16 on the
  host, then op-by-op bf16 GEMM / RMSNorm / QK-norm / RoPE / SwiGLU /
  causal-GQA flash attention **and** the ShortConv leaves (gate, depthwise
  conv, gate) all on the NPU with resident weight BOs.
- **Fused per-token decode** (the standalone [`fused_decode`](../../fused_decode)
  example) — one dispatch = **all 16 layers, both types** + LM head. The layer
  type is chosen per wave by a runtime arm, and the kernels branch on it
  internally, so the whole hybrid is ONE xclbin.
- **Host orchestration** (`lfm2_1_2b_q4nx_inference.py`) — embedding, final
  RMSNorm, sampler, chat template, EOS and streaming between tokens.

## Architecture

```
                        ATTENTION (6)              SHORTCONV (10)
                    ---------------------    --------------------------
  RMSNorm(operator_norm) for both
                        Q/K/V proj               in_proj -> [B | C | x]
                        QK-norm (per head)       h = B * x
                        RoPE (theta 1e6)         h = causal_dwconv(h, 3 taps)
                        FlashAttention           y = C * h
                    ------------- both then: -------------
        out_proj -> +residual -> RMSNorm(ffn_norm) -> SwiGLU FFN -> +residual
```

`out_proj` is 2048×2048 for **both** block types, which is why one `o_ffn` ELF
serves both: the ShortConv result lands where attention output would.

### The prefill hands the decode TWO regions

A pure transformer passes one thing across the prefill/decode boundary: a KV
cache. LFM2 passes two, because half its layers have no KV at all.

```
  arg4 = [ 16 KV-cache slabs | 16 ShortConv-state slabs ]
```

- attention layers seed a **region-major K/V** slab;
- ShortConv layers seed a **carried state**: the last `conv_L_cache - 1` rows of
  the *pre-convolution* gated signal `h`. Causality here is a left **pad**, not
  a mask, so that state *is* the pad the next token's convolution consumes.

Every layer gets a slot in both regions even though it uses exactly one. That is
deliberate: it keeps the decode's per-layer address a plain `iv * SLAB` and
costs a few hundred KB, where the alternative is an irregular per-layer offset
table threaded through every DMA in the launch.

## Performance (NPU2, AMD Strix, 16 layers, warm)

- **Prefill TTFT** @ 2048 ≈ **1.42 s** warm (`make profile`; the cold number
  includes the one-time host weight load and Q4_0 quantization and is not the
  steady-state figure).
- **Decode**: **53.2 tok/s** end to end (`make profile`, 200 tokens) — the whole
  model in ONE dispatch, streaming 0.755 GB of weights per token.
- Decode-only, synthetic weights, 64 iters: 18.1 ms/token at ATTN_MAXL 2048
  (55.2 tok/s) and 15.6 ms at 16. Context is nearly free here: 2032 extra tokens
  of KV cost 2.5 ms, i.e. **0.12 ms per 1k**.

Measure them yourself with `make profile` (end to end) and
`make decode-gate` (decode only). Check `uptime` first — a busy box has produced
phantom regressions here before.

## Why Q4_0 from the fp checkpoint

The device kernels are built `-DQ4_0`: symmetric signed int4, `w = q * scale`,
per 32-element group along the reduction dim, with no offset term. There is no
pre-quantized LFM2 bundle, and re-quantizing an *affine* 4-bit bundle into a
symmetric grid would quantize twice and lose more than starting from full
precision. So the weights are quantized once, from the bf16 checkpoint —
simpler and strictly more accurate. Round-trip cosine is **0.9953**
(`make gate`); that is the correct bar for Q4_0, not the ~0.997 an affine Q4_1
codec reaches.

## Prerequisites

```bash
source utils/env_setup.sh
pip install -r requirements.txt        # huggingface_hub, safetensors, transformers
export HF_TOKEN=...                    # checkpoint + tokenizer download
```

## Quick Start

```bash
# Build 1 — Prefill ELFs (7 of them; no weights/NPU needed) — ~2 min
make compile

# Build 2 — Fused hybrid decode templates (decode_L2048 + decode_L2047 slope ref)
make compile-decode

# Run inference (NPU prefill + fused NPU decode)
make run

# With the decode-throughput / TTFT profiling summary
make profile

# Interactive multi-turn chatbot (streams tokens)
make chat

# Correctness gate: top-k token-set inclusion, NPU q4_0 vs HF bf16
make verify

# Fast weight-integrity smoke: prefill first token -> *** PARIS ***
make verify-paris
```

## Correctness

| gate | what it covers | result |
|---|---|---|
| `make gate` | Q4_0 round-trip on a real tensor | cosine **0.9953** |
| `make verify-paris` | prefill only: weights, schedule, every GEMM's operand order | first token **' Paris'** |
| `make decode-gate` | decode only, both layer types interleaved, no prefill | see below |
| `make verify` | **end to end**, top-k token-set inclusion vs HF bf16 | **PASS**, 2 prompts × 32 tokens, k=5 |

`make verify` is the gate that matters, and the one CI runs. The other three
exist to localise a failure: they split "the weights are wrong" from "the decode
is wrong" from "the prefill is wrong", which the end-to-end gate cannot.

### Reading the decode gate

`lfm2_1_2b_q4nx_decode_gate.py` scores logits **and** the read-back ShortConv
state per layer. Two things about it are worth knowing before reading a number:

- It uploads **zero q-norm weights** on purpose, which makes every attention
  score 0 and the softmax uniform, so the attention output is exactly
  `V / ATTN_L`. That removes the unrecoverable score scale from the comparison —
  at the cost of not covering RoPE, QK-norm or the score path. `make verify`
  covers those.
- bf16-on-device against an f32 reference **compounds down the residual
  stream**, so cosine falls with depth for reasons that are not defects. Judge
  exactness at one layer (`make compile-decode LAYERS=1`, expect ≥ 0.996) and
  the whole model on cosine plus top-1 agreement. The carried-state check is
  depth-independent by construction: it bounds what one *model layer* adds, not
  the total.

## How it works

The decode is ONE xclbin whose per-token context length `L` is set by patching a
few instruction words on the host (`decode_insts_gen.py`). Two same-`ATTN_MAXL`
builds (`decode_L2048` + `decode_L2047`) calibrate the L-slope, so the generator
can synthesize the instruction stream for any `L` in `[1, 2048]` — byte-identical
to a native per-L build. The attention kernel skips fully-masked blocks, so the
single template is correct at every context length.

> **Trap.** `make compile-decode LAYERS=N` writes a short bisect build under the
> *same* `decode_L<ATTN_MAXL>.*` name, and the template is selected by context
> reach, not layer count — so a leftover 1-layer build gets picked for a short
> prompt and silently runs one layer of a sixteen-layer model. The driver checks
> the template's own compiled signature and refuses, but `make clean` after a
> bisect is the habit to keep.

## Key Files

| file | role |
|---|---|
| `lfm2_1_2b_q4nx_weights.py` | Q4_0 loader: HF bf16 → Q4_0 → bf16, plus the config |
| `lfm2_1_2b_q4nx_prefill.py` | batched NPU prefill; fills the KV cache **and** the conv state |
| `lfm2_1_2b_q4nx_inference.py` | driver: prefill + fused decode + host sampling/chat |
| `lfm2_1_2b_q4nx_decode_gate.py` | decode-only numerics, per layer |
| `verify_adapter.py` | plugs into `llms/verify/` for `make verify` |
| `../../fused_decode/` | the fused hybrid decode builder and its kernels |
