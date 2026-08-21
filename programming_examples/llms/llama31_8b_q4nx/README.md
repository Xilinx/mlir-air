# LLAMA-3.1-8B Q4NX Inference on AMD NPU2 (MLIR-AIR)

This implementation reimplements the corresponding AMD NPU LLM design,
originally developed by the [FastFlowLM](https://github.com/ROCm/FastFlowLM)
team, using the higher-level abstractions of the MLIR-AIR dialect.

A full **prefill + decode** MLIR-AIR inference for Llama-3.1-8B using **Q4NX**
weights (per-block 4-bit affine quantization, `w = q*scale + min`) on NPU2
(AIE2P). The decoder layer runs **entirely on the AIE array** — attention
included, over a device-resident KV cache — the same mapping FastFlowLM uses.

- **Prefill** (`llama31_8b_q4nx_prefill.py`) — op-by-op bf16 GEMM / RMSNorm /
  RoPE / SwiGLU / head-first causal-GQA flash attention on the NPU with resident
  weight BOs; on-device 8-partition LM-head GEMV. Its per-layer roped-K / raw-V
  seed the decode's device KV cache, so a long prompt is not replayed token-wise.
  The stitcher/FA builders are dimension-driven and 8B shares 3B's `head_dim=128`
  head-first FA path, so they are reused rather than re-authored.
- **Autoregressive decode** (`q4nx_decode_8b.py`) — one dispatch per token through
  the [`fused_decode`](../../fused_decode) superkernel at `MODEL_TYPE=LLAMA_3_1_8B`:
  proj → RoPE → flash attention over the on-device KV → O → FFN, ×32 layers, then
  the LM head. No CPU attention, no host KV.
- **Host orchestration** — embedding, argmax, chat template, EOS, and streaming
  output between tokens.

NPU weights are FastFlowLM's `model.q4nx` packing of `meta-llama/Llama-3.1-8B-Instruct`
(`FastFlowLM/Llama-3.1-8B-NPU2`), so this example gates with the standard
`make verify` **top-k token-set inclusion vs HF transformers bf16**.

## Model Config

32 layers, emb_dim=4096, n_heads=32, head_dim=128, n_kv_heads=8 (GQA group=4),
q_dim=4096, kv_dim=1024, hidden_dim=14336, vocab_size=128256, rope_theta=500000
with **llama3 rope scaling factor 8.0**, eps=1e-5, **untied** embeddings
(`lm_head` ≠ `embed_tokens`). Q4NX codec: 32×256 blocks, 32-column groups.
Pure Llama (no QK-norm, no bias).

Two of those differ from the Llama-3.2 siblings and are both load-bearing:

**Untied LM head.** Llama-3.2-1B/3B set `tie_word_embeddings=true`, so their LM
head *is* the embedding matrix, and the shared requant path hard-coded that.
Llama-3.1-8B sets it false and the bundle ships a real quantized `lm_head`.
Measured `cosine(bundle lm_head, embed_tokens) = 0.017` — essentially orthogonal,
so the tied default would have produced a silently wrong model rather than a
build error. `build_requant_cache(..., tie_lm_head=False)` selects the real one.

**RoPE scaling factor 8.0.** Llama-3.1 scales by 8.0, Llama-3.2 by 32.0. This is
also exactly the curve behind FastFlowLM's shared `llama_3b_8b_rope` table:
factor 8.0 reproduces that table to 4e-05 (its print precision), 32.0 is off by
75% and unscaled by 7x. So FLM-faithful and HF-correct coincide here.

## Weight fidelity (Q4NX dequant vs HF bf16, layer 0)

`python3 llama31_8b_q4nx_weights.py` — measured on this bundle:

| proj | q | k | v | o | up | gate | down | lm_head |
|---|---|---|---|---|---|---|---|---|
| cosine | 0.99687 | 0.99585 | 0.99523 | 0.99775 | 0.99810 | 0.99860 | 0.99792 | 0.99964 |

Worst 0.99523 — the expected range for this codec.

## Performance (NPU2, AMD Strix, 32 layers, warm)

| | |
|---|---|
| prefill (TTFT) | **6.22 s** at CTX=2048 (330 tok/s); 6.19 s on-device, 21 ms host |
| decode | **10.7 tok/s** (93.1 ms/token), ATTN_MAXL=2048, 128 tokens |

FastFlowLM does not publish a figure for this model that I could source, so
there is no reproduction ratio here — unlike the Qwen3-8B sibling.

Prefill runs at a fixed padded context (`CTX`), so its cost is roughly constant
in prompt length rather than proportional to it. Measure it with
`make profile-prefill`. On-device time per op at CTX=2048 (sum over 32 layers):

| o_ffn | flash_attn | rms_gemms_rope | lm_head_gemv |
|---|---|---|---|
| 3900 ms | 1452 ms | 575 ms | 28 ms (x1) |

Of the 5.93 s of measured XRT time, 96% is NPU execution, 3% host->DDR writes of
the dynamic inputs and <1% readback. The fused O+FFN is the dominant term at this
hidden size (14336), ahead of attention.

Separately, loading the model — Q4NX host dequant plus the one-time write into
resident BOs (13.3 GB across 32 layers) — is a startup cost, not part of TTFT.

## Prerequisites

- NPU2 (Strix / AIE2P) + XRT, and a Peano (llvm-aie) install
- `source utils/env_setup.sh` (see the repo README)
- An `llvm-link` from LLVM <23 for `make compile-decode` (the fused_decode
  preflight enforces it). Where the AIR env script puts an LLVM 24 `llvm-link`
  first on `PATH`, prepend a compatible one:
  `export PATH=/usr/lib/llvm-20/bin:$PATH`.
- Weights: `FastFlowLM/Llama-3.1-8B-NPU2` from the Hub (single `model.q4nx`
  safetensors bundle). Override with
  `MODEL_SOURCE=<repo id | local dir | model.q4nx path>`.
- Python deps on top of the base environment: `pip install -r requirements.txt`
- ~13 GB of host memory for the prefill's resident per-layer BOs, and 4.4 GB for
  the decode weight BOs (five buffers, see below).

## Quick Start

```bash
make compile          # prefill ELFs (~4 min, one-time; no weights)
make compile-decode   # fused decode templates into this dir (~15 min; no weights)
make run              # 32 layers + LM head on-device, one dispatch per token
make verify           # top-k token-set gate: NPU q4nx vs HF bf16
```

## Correctness

`make verify` is the shared gate every `llms/` example uses: top-k token-set
inclusion (k=5, first divergence over 32 greedy tokens) of the NPU q4nx run
against an HF **bf16** reference, driven through
[`verify_adapter.py`](verify_adapter.py) and `llms/verify/`.

**The reference is `NousResearch/Meta-Llama-3.1-8B-Instruct`**, a faithful
ungated re-upload of the `meta-llama` checkpoint FastFlowLM's bundle names as its
`base_model`. Its config matches the FLM bundle's exactly, including the 3-way
`eos_token_id` list, so CI needs no license grant — the same pattern
`gemma3_4b_q4nx` uses for google's gated Gemma. Point `MODEL_CHOICES` in
`verify_adapter.py` at `meta-llama` instead if you have accepted Meta's license.

Measured on NPU2 (`LBUILD=2048`):

| | result |
|---|---|
| `make verify` (2-prompt CI slice) | **2 / 2 PASS** |
| `make verify-full` (8 prompts) | **8 / 8 PASS** |
| `make verify-paris` (prefill-only smoke) | **PASS** (argmax 12366) |

`make verify-paris` is a prefill-only first-token smoke (no decode templates, no
reference) for bringing the prefill up on its own. It gates on **"The capital
city of France is called"**, not the bare "The capital of France is" the 1B/3B
siblings use: on the bare phrasing this model ranks `" a"` (16.000) just above
`" Paris"` (15.938), and the HF bf16 reference ranks them the same way, so that
prompt fails a correct build. The "...is called" phrasing puts `" Paris"` 3.4
logits clear.

## How it works

### Prefill

A thin driver over the config-parameterized `llama32_3b` builders — 8B shares
3B's `head_dim=128` head-first FA path, so the stitcher/FA builders are reused
rather than re-authored. `llama31_8b_q4nx_weights.py` supplies the config and
builds that example's `LlamaWeights`/`LayerWeights` from the `model.q4nx` bundle.
Per-layer roped-K / raw-V seed the decode's device KV cache, so a long prompt is
not replayed token-wise.

Two things at this size are not just bigger constants:

- **Two FFN GEMM shapes had to be swept into the kernel registry**
  (2048x4096x14336 gate/up and 2048x14336x4096 down; the attention shapes came
  free from the Qwen3-8B sibling, which shares K=4096). Note that the Qwen3-8B
  N=12288 entry concludes fused-cast is legal only at `tile_k_l2=64` because the
  f32-out B-tile stride must stay under 1048576 — that is **not** what binds at
  N=14336, where `tile_k_l2=128` places and is 8% faster at identical accuracy.
  `tile_k_l2=256` is uncharacterized: aiecc ran over an hour without finishing.
- **The LM-head GEMV needs three non-default knobs at K=4096**, exactly as the
  Qwen3-8B sibling does. L2 needs `herd_m * tile_m * (K+1) * 2 <= 512 KiB`, which
  the default 8x8 misses by 128 B, and `m_input=4` puts the ping-pong L1 A tiles
  at all of L1. `herd_m` **must stay 8** (`herd_m=4` compiles then hangs the
  dispatch), so `herd_m=8, tile_m=4, m_input=2`, plus `compile_mv(tile_m=4)`
  because `mv.o` is keyed to `tile_m` through `DIM_M_OUTPUT`.

### Decode

One dispatch per token through the [`fused_decode`](../../fused_decode)
superkernel at `MODEL_TYPE=LLAMA_3_1_8B`: proj -> RoPE -> flash attention over
the on-device KV -> O -> FFN, x32 layers, then the LM head. No CPU attention, no
host KV. The attention topology is identical to Llama-3.2-1B/3B
(`ATTN_IMPL_2x4x1`, 8 kv heads, DH=128), so the per-CU KV geometry carries over
unchanged; only the proj/FFN widths grow.

## Build notes

The decode-side knobs this geometry needs; see Prerequisites for the toolchain
requirement and How it works for the prefill-side tiling.

- **The weights do not fit one buffer (`DECODE_WGROUP=8`).** A shim BD's byte
  offset is a `uint32` end to end, so a single BO is addressable over at most
  4 GiB. This model's 32 layer slabs plus the lm-head are **4.375 GiB**: the
  offsets wrapped past the boundary and every logit came back NaN. `DECODE_WGROUP`
  splits the weights into four 1.02 GiB layer groups plus a 0.31 GiB lm-head
  buffer, each addressed from its own base, and the host slices its BOs to match
  (read off the engine, not the environment, so the two cannot disagree). The
  build and the run must use the same value; the Makefile's `DECODE_WGROUP` and
  `q4nx_decode_8b.py`'s constant are both 8, and the template stamp is keyed on it.
- **The rms/residual core runs a trimmed stack at this size (`DECODE_STACK=8064`).**
  It holds 7 K-sized bf16 buffers; at K=4096 that is 56 KB of the 64 KB L1, so the
  engine's default 10240-byte stack does not fit and buffer allocation fails
  outright. 8064 fits in banks (bank0 = stack + 1 buffer + the 4-byte per-herd RTP)
  and still leaves >2x headroom over the measured worst frame (2112 B). Any model
  at K=4096 hits this same wall.
- **Vocab chunking is `VOCAB_CHUNK_I2=16`, `UNI_LM=8`.** The vocab relay drains
  whole-K blocks, so `K/PAYLOAD = 8` must divide `VOCAB_I2*PAIR_ROWS`; with
  `VOCAB_FULL_ROWBLKS = 4096` the legal set is {4,8,16} and 16 gives the fewest
  host-armed waves. Another value deadlocks the vocab wave.

## Decode staircase (opt-in, ~1.1-1.2x)

The decode template streams `ATTN_MAXL` KV positions per token regardless of the real
context length, so a 2048-window build wastes most of that readback at short context.
Building one template pair per window and dispatching each token on the smallest covering
window recovers it, with a token stream identical to the single-window baseline.

```bash
make compile-decode-windows      # once; WINDOWS="64 512 2048" to override the set
DECODE_STAIRCASE=1 make run
```

Off by default. See
[`programming_examples/fused_decode/README.md`](../../fused_decode/README.md) for the
mechanism, the measurements and the `.decode_windows` manifest guard.

## Key Files

| file | role |
|---|---|
| `llama31_8b_q4nx_inference.py` | driver: AIR prefill (KV seed + first token) + fused NPU decode loop |
| `llama31_8b_q4nx_prefill.py` | thin driver over the `llama32_3b` builders + its `causal_lm` interface |
| `llama31_8b_q4nx_weights.py` | `model.q4nx` loader, Q4NX dequant, llama3-scaled RoPE LUTs |
| `q4nx_decode_8b.py` | fused decode wrapper: template load, split weight BOs, KV seeding, per-token dispatch |
| `decode_bringup_8b.py` | standalone on-device decode bring-up (no prefill), for isolating the decode |
| `Makefile` | build / run / verify / verify-full / verify-paris / diagnosis / profile |
| `verify_adapter.py` | binds the Q4NX prefill + fused decode to the shared `verify/` runner contract |
| `../llama32_1b_q4nx/q4nx_requant.py` | shared Q4NX -> q4k-cascade re-quantization (`tie_lm_head=False` here) |
| `../../fused_decode/models/llama3.1-8b.h` | kernel-side model config |
| `../../fused_decode/fused_decode.py` | the shared superkernel IR builder |
