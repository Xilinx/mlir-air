# GEMMA3-4B Q4NX prefill + decode on AMD NPU2

Gemma3-4B (text) in MLIR-AIR, reproducing FastFlowLM's mechanism end to end on
the NPU:

- **prefill** — a batched pass over the padded context: RMSNorm, Q/K/V GEMMs,
  per-head QK-norm, dual-theta RoPE, GQA flash attention (alternating
  sliding-window / global), GELU-tanh GLU and the tied LM head, all on device
  with resident weight BOs.
- **decode** — **one NPU dispatch = 34 decoder layers + the tied LM head**,
  reading and appending a shared per-layer KV cache seeded by the prefill.

Weights are Q4NX throughout (dequantized on device for decode, on the host at
load for prefill).

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
| prefill (TTFT) | **4.6 s** at CTX=2048 (444 tok/s); 4.58 s on-device, 29 ms host |
| decode | **12.5 tok/s** (80 ms/token), ATTN_MAXL=2048, 64 tokens |

Prefill runs at a fixed padded context (`CTX`), so its cost is roughly constant
in prompt length rather than proportional to it. Measure it with
`make profile-prefill`. Per-op on-device split at CTX=2048:

| attn | gate | up | down | rms_qkv | o_norm | gelu | lm_head |
|---|---|---|---|---|---|---|---|
| 1151 ms | 712 | 710 | 635 | 599 | 394 | 335 | 45 |

Of the 4.58 s on device, 3.90 s is kernel time, 132 ms host->device and 70 ms
back; the rest is host-side marshalling (the flash-attention layout transposes
are ~4 ms/layer of it).

For reference, FastFlowLM's own Gemma3-4B prefill measures **3.21 s** (637
tok/s) at the same context on this machine, producing the same first token —
i.e. this example is currently **1.4x slower** than the implementation it
reproduces. (Contrast the Llama-3.2-1B sibling, where the AIR prefill is at
parity.)

What remains is spread across the FFN GEMMs rather than concentrated in one
stage. `gate` and `up` run at ~5.3 TFLOP/s where the same K against a narrower
N reaches ~7: their `tile_k_l2` is capped at 64 because the DMA stride limit is
`tile_k_l2 * N <= 1048576` and N is 10240. Lifting it means either splitting
the projection into N bands — which changes the layout of everything
downstream of it in the FFN chain — or feeding the weights K-major so the BD
stride is 2560 rather than 10240.

Attention still *streams* every out-of-window K block and masks it; only the
arithmetic is skipped. Skipping the streaming as well needs the per-round
variable-size puts of `attn_npu2_temporal_causal.py`, which is a different
dataflow from the head-first kernel's.

Separately, loading the model — Q4NX host dequant of ~6 GB of weights plus the
one-time write into resident BOs — takes ~67 s per process. That is a startup
cost, not part of TTFT; the driver reports the two separately.

Decode is data-movement bound: at Q4 the 4B weights are ~3.2x the bytes/token
of the 1B Llama example, which runs 40 tok/s on the same engine — i.e. both sit
at the same effective shim bandwidth.

## Prerequisites

- NPU2 (Strix / AIE2P) + XRT, and a Peano (llvm-aie) install
- `source utils/env_setup.sh` (see the repo README)
- Weights: `FastFlowLM/Gemma3-4B-NPU2` from the Hub (single `model.q4nx`
  safetensors bundle, ~3.7 GB). Override the source with
  `MODEL_SOURCE=<repo id | local dir | model.q4nx path>`.

## Quick Start

```bash
# Build everything, weight-free: the 8 prefill ELFs + the decode templates.
# Production decode templates are ATTN_MAXL=2048 (~15 min).
make compile

# ...or just one half:
make compile-prefill            # the 8 prefill ELFs (CTX=2048)
make compile-decode LBUILD=16   # small decode templates (much faster)

# Paris gate: first generated token must be 9079 (" Paris").
make run          # or: make verify   (same gate, the name CI uses)

# Prefill only (no decode loop) -- same 9079 gate, useful when bringing up the
# prefill on its own.
make prefill

# Decode throughput over NTOK tokens (needs the production templates).
make profile NTOK=64

# Shorter prefill context, and a local bundle instead of the Hub.
make run CTX=2048 MODEL_SOURCE=/path/to/Gemma3-4B-NPU2
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

### Prefill

Eight ELFs, each driven through `KernelCache.load_and_run` with per-layer
resident weight BOs (weights written once, skipped on every later call):

| ELF | contents |
|---|---|
| `rms_qkv_qknorm_rope` | input norm + Q/K/V GEMMs + per-head QK-norm + RoPE (8 launches) |
| `flash_attn_global` | GQA flash attention, plain causal — the global layers |
| `flash_attn_local` | same kernel with the 1024 sliding-window mask — the local layers |
| `o_norm_res_norm` | O proj + post-attention norm + residual + pre-FFN norm |
| `gate`, `up` | the two FFN projections (2560 -> 10240) |
| `gelu_mul` | GELU-tanh GLU: `gelu(gate) * up` |
| `down_norm_add` | Down proj + post-FFN norm + residual |
| `lm_head_gemv` | tied LM head, 17 partitions of 16384 (vocab 262208) |

Two structural notes:

- **Gemma norms the sublayer *output*, before the residual add** (`O -> norm ->
  +residual`, `down -> norm -> +residual`), unlike the Llama/Qwen shape. That is
  why the O and Down tails carry an extra `weighted_rms_norm` slice. All Gemma
  norms use eps 1e-6.
- **Attention alternates.** Five of every six layers use a 1024-token sliding
  window; the sixth is global. Both come from the same flash-attention kernel —
  the window is a compile-time mask (`apply_window_mask` in `attn_npu2.cc`,
  selected by `build_module(window=...)`), so the two ELFs differ only in that
  mask. RoPE theta switches with them (local 1e4 / global 1e6 with linear x8),
  which is purely a choice of host LUT.

At head_dim=256 the head-first flash attention tiles at `lkp=32` rather than the
head_dim=128 path's 64: the resident Q tile is `head_dim * lkp * 2B`, so `lkp=64`
needs 72 KB of L1 and aiecc rejects it.

What actually sets the runtime is L3 traffic, not L1 or FLOPs. The kernel
re-streams all of K once per launch iteration, so

    K bytes = (seq / lqp) * n_heads * (head_dim / dv_tile) * seq * head_dim * 2

and the two free knobs both belong in the numerator's denominators: `lqp` is
maximised by spending the 32-core budget on Q tiles rather than on unrolled
heads (`num_heads_per_unroll=1`, `num_q_tiles=8`), and `dv_tile` is widened past
`lkp` so the `dv_chunks` launch axis — which re-streams the whole of K and Q per
chunk — shrinks to 2. Together that is 5.9x less traffic and 2.8x less
attention time. See `_FA_TILING` in `llms/shared/infra/fa_headfirst.py`.

### Decode

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
| `gemma3_4b_q4nx_inference.py` | driver: AIR prefill (KV seed + first token) + fused NPU decode loop |
| `gemma3_4b_q4nx_prefill.py` | the 8-ELF batched prefill and its `causal_lm` interface |
| `gemma3_4b_q4nx_weights.py` | `model.q4nx` loader, Q4NX dequant, RoPE LUTs, numpy reference |
| `gemma3_4b_q4nx_requant.py` | Q4NX re-quantization helper |
| `Makefile` | build / run / verify / profile |
| `../../fused_decode/models/gemma3-4b.h` | kernel-side model config |
| `../../fused_decode/fused_decode.py` | the shared superkernel IR builder |
