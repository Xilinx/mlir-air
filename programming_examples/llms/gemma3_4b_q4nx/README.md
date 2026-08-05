# GEMMA3-4B Q4NX decode on AMD NPU2

Fused per-token decode for Gemma3-4B (text) in MLIR-AIR, reproducing
FastFlowLM's decode mechanism: **one NPU dispatch = 34 decoder layers + the
tied LM head**, reading and appending a shared per-layer KV cache. Weights are
Q4NX, dequantized on device.

This is the Gemma sibling of [`llama32_1b_q4nx`](../llama32_1b_q4nx/); both
drive the same superkernel engine in
[`programming_examples/fused_decode`](../../fused_decode/), selected with
`DECODE_MODEL=gemma3-4b`.

## Architecture

Verified against FastFlowLM's `Gemma3-4B-NPU2` packaging and the HF config:

| | |
|---|---|
| model dim / head dim | 2560 / 256 |
| attention | 8 query heads, 4 KV heads (GQA 2), scale 0.0625 |
| MLP | intermediate 10240, GELU-tanh |
| layers / vocab | 34 / 262208 (tied embeddings) |
| norms | 4 per layer (input, post-attention, pre-ffn, post-ffn), `1+w` fold, eps 1e-6 |
| RoPE | dual theta — local 1e4, global 1e6 (linear x8), pattern 5 local : 1 global |
| extras | qk-norm, sliding window 1024 on local layers, embedding scale sqrt(2560) |

Divergences from Llama are parametric, host-side, or compile flags — see
[How it works](#how-it-works).

## Performance (NPU2, AMD Strix, 34 layers, warm)

| | |
|---|---|
| decode | **12.5 tok/s** (80 ms/token), ATTN_MAXL=2048, 64 tokens |

Decode is data-movement bound: at Q4 the 4B weights are ~3.2x the bytes/token
of the 1B Llama example, which runs 40 tok/s on the same engine — i.e. both sit
at the same effective shim bandwidth.

**There is no NPU prefill.** The KV cache and the first token come from a numpy
reference forward (`gemma3_4b_q4nx_weights.forward_prompt`), which takes tens of
seconds and dominates wall clock. Only the decode loop runs on the NPU. (The
Llama example, by contrast, has a batched NPU prefill.)

## Prerequisites

- NPU2 (Strix / AIE2P) + XRT, and a Peano (llvm-aie) install
- `source utils/env_setup.sh` (see the repo README)
- Weights: `FastFlowLM/Gemma3-4B-NPU2` from the Hub (single `model.q4nx`
  safetensors bundle, ~3.7 GB). Override the source with
  `MODEL_SOURCE=<repo id | local dir | model.q4nx path>`.

## Quick Start

```bash
# Build the decode templates (weight-free). Production ATTN_MAXL=2048, ~15 min.
make compile

# ...or the small templates the Paris gate needs (much faster).
make compile LBUILD=16

# Paris gate: first generated token must be 9079 (" Paris").
make run          # or: make verify   (same gate, the name CI uses)

# Decode throughput over NTOK tokens (needs the production templates).
make profile NTOK=64

# Point at a local bundle instead of the Hub.
make run MODEL_SOURCE=/path/to/Gemma3-4B-NPU2
```

## Correctness

Gated on FastFlowLM's Gemma3 NPU output rather than HF bf16 top-k: greedy
first-token argmax for "The capital of France is" must be **9079** (" Paris"),
and generation continues coherently:

> ' Paris.\n\nParis is a global center for art, fashion, gastronomy and
> culture.\n\nIt is located on the River Seine. ...'

`make verify` is that gate. CI runs it via `run_npu2_verify.lit`, skipping
cleanly when the Q4NX weights are absent from the runner's HF cache.

## How it works

The decode engine is shared with the Llama example; Gemma differs in five
places, none of which required a new dataflow:

1. **4-norm sandwich.** Gemma norms around both the attention and the MLP. The
   two extra norm weights are packed two-per-channel and read by
   `rms_norm_lo_aie` / `rms_norm_hi_aie`, so the rms tile stays within 4 packet
   ids per S2MM port.
2. **GELU-tanh** instead of SiLU — a lookup-table selection in
   `kernels/lut_based_ops.h`.
3. **qk-norm + dual-theta RoPE**, fed as a packed `rope_w` slab (cos/sin,
   q_norm, k_norm) with the per-layer LUT chosen by the 5:1 local/global
   pattern.
4. **DH=256 GQA2 attention** — the attention accumulators size by
   `Q_HEADS_PADDED_PER_CU` rather than a fixed head count.
5. **Weight prep**: `1+w` norm fold, embedding scale, vocab-262208 chunking.

The rms tile's DMA layout depends on a compute-tile allocator rule
(`rebalanceAperiodicPacketChains`): the 4-norm structure makes the tile consume
the sublayer result twice per dispatch beside once-per-dispatch norm weights,
and such unequal-multiplicity flows must not share one S2MM BD chain.

## Key Files

| file | role |
|---|---|
| `gemma3_4b_q4nx_inference.py` | driver: numpy prefill (KV seed) + fused NPU decode loop |
| `gemma3_4b_q4nx_weights.py` | `model.q4nx` loader, Q4NX dequant, RoPE LUTs, numpy reference |
| `gemma3_4b_q4nx_requant.py` | Q4NX re-quantization helper |
| `Makefile` | build / run / verify / profile |
| `../../fused_decode/models/gemma3-4b.h` | kernel-side model config |
| `../../fused_decode/fused_decode.py` | the shared superkernel IR builder |
