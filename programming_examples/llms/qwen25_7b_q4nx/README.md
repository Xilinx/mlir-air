# QWEN2.5-7B Q4NX prefill + decode on AMD NPU2

Qwen2.5-7B-Instruct in MLIR-AIR, reproducing FastFlowLM's mechanism end to end on
the NPU:

- **prefill** — a batched pass over the padded context: RMSNorm, Q/K/V GEMMs, the
  q/k/v projection bias, RoPE, GQA flash attention, SwiGLU and the LM head, all on
  device with resident weight BOs.
- **decode** — **one NPU dispatch = 28 decoder layers + the LM head**, reading and
  appending a shared per-layer KV cache seeded by the prefill.

Weights are Q4NX throughout (dequantized on device for decode, on the host at load
for prefill).

This is the first **`HAS_QKV_BIAS`** model on the shared superkernel engine in
[`programming_examples/fused_decode`](../../fused_decode/), selected with
`DECODE_MODEL=qwen2.5-7b`.

## Weights

Unlike the [Qwen3-8B sibling](../qwen3_8b_q4nx/), there is no bundle to
download: **FastFlowLM publishes no Qwen2.5-7B NPU2 model** — their Qwen2.5 line
stops at 3B. So this example quantizes the upstream bf16 checkpoint itself, the
way the [Qwen2.5-3B Q4_0 sibling](../qwen25_3b_q4/) does for the analogous
reason, and needs nothing beyond an ungated HF repo:

```bash
make run                                   # MODEL_SOURCE=Qwen/Qwen2.5-7B-Instruct
make run MODEL_SOURCE=/path/to/checkpoint  # or a local checkpoint dir
```

Every projection is rounded onto the Q4NX grid at load by
`quantize_dequantize_q4nx` — the **same** quantizer the decode cascade cache
uses — so the prefill and the fused decode see bit-identical weights, and the
prefill models the 4-bit datapath the hardware actually runs rather than
flattering it with bf16. Norms and the q/k/v biases stay bf16, as in the
reference design.

A `model.q4nx` bundle is still accepted (`MODEL_SOURCE=<dir with model.q4nx>`)
for anyone who has one. Note that FastFlowLM's converter writes its Qwen2.5
output through a custom nibble interleave
(`q4nx/gguf_tensor.py: transform_nibble_layout`) that their Llama/Qwen3 bundles
do not use, so a Qwen2.5 bundle is **not** byte-interchangeable with those — the
loader here implements the Llama/Qwen3 layout.

## Architecture

Verified against the HF `config.json` and FastFlowLM's own
`generic_decoding_layer/models/qwen2.5-7b.h`:

| | |
|---|---|
| model dim / head dim | 3584 / 128 |
| attention | 28 query heads, 4 KV heads (GQA 7), scale 1/sqrt(128) |
| MLP | intermediate 18944, SwiGLU |
| layers / vocab | 28 / 152064 (**untied** — a separate Q4NX `lm_head`) |
| norms | 2 per layer (input, post-attention), plain `w`, eps 1e-6 |
| RoPE | single theta 1e6, half-split |
| extras | **q/k/v projection bias**; no qk-norm, no sliding window, no embed scale |

## Performance (NPU2, 28 layers, warm)

Measured on an **AMD Ryzen AI 9 HX 370** (Strix Point, XDNA2 / NPU2), 96 GB,
Ubuntu 25.04, XRT 2.23.0 (engine built, weights resident). Every prefill figure
below is from a single `make profile-prefill` run, so they reconcile:

| | |
|---|---|
| prefill (TTFT) | **9.59 s** at CTX=2048 (214 tok/s); 9.58 s on-device, 8 ms host |
| decode | **11.46 tok/s** (87.2 ms/token), ATTN_MAXL=2048, 64 tokens |

FastFlowLM publishes no Qwen2.5-7B bundle, so there is no vendor decode number to
compare against here — unlike the Qwen3-8B and Gemma3-4B siblings.

Prefill runs at a fixed padded context (`CTX`), so its cost is roughly constant in
prompt length rather than proportional to it. On-device time per op at CTX=2048
(NPU-execution only, summed over 28 layers):

| up | gate | flash_attn | down_add | rms_qkv_bias_rope | o_res_norm | swiglu | lm_head_gemv |
|---|---|---|---|---|---|---|---|
| 2368 ms | 2358 | 1135 | 837 | 451 | 311 | 286 | 31 (x1) |

Those sum to 7776 ms. The three prefill totals are different scopes, which is why
they differ: **7.78 s** of NPU execution, **8.50 s** of instrumented XRT call time
(adds 577 ms of host→DDR writes of the dynamic inputs and 142 ms of readback), and
**9.59 s** of wall TTFT (adds ~1.1 s of host-side Python between dispatches).

Unlike the Gemma3-4B sibling the lever here is not attention: `up` + `gate` alone
are 61% of NPU time, because `INTERMEDIATE_SIZE=18944` is 5.3x hidden (2.7-4.0x
for every other model in `llms/`).

Separately, loading the model — the HF bf16 read, the Q4NX host dequant, and the
write into resident BOs — takes ~130 s per process with a warm HF cache and peaks
at **~33 GB RSS** (measured: 32.6 GB for the prefill path alone). That is a
startup cost, not part of TTFT; the driver reports the two separately.

## Prerequisites

- NPU2 (Strix / AIE2P) + XRT, and a Peano (llvm-aie) install
- `source utils/env_setup.sh` (see the repo README)
- Weights: nothing to fetch beyond the ungated `Qwen/Qwen2.5-7B-Instruct`
  checkpoint (see [Weights](#weights))
- Python deps on top of the base environment: `pip install -r requirements.txt`
- **~33 GB of host RAM.** Peak is the load phase, not the run: the bf16
  checkpoint, the Q4NX dequant and the resident BOs (five decode weight
  buffers, 4x0.95 GB + 0.32 GB) are live together. Measured 32.6 GB peak RSS.

## Quick Start

```bash
# Build everything, weight-free: the prefill ELFs + the decode templates.
# Production decode templates are ATTN_MAXL=2048.
make compile

# ...or just one half:
make compile-prefill            # the prefill ELFs (CTX=2048)
make compile-decode LBUILD=64   # small decode templates (much faster)

# Correctness gate: top-k token-set inclusion vs HF bf16 (k=5, 32 tokens).
make verify       # 2-prompt CI slice;  make verify-full  sweeps every prompt

# Demo run: greedy decode, prints a *** PARIS *** verdict.
make run

# Prefill only (no decode loop, no reference) -- first-token 12095 smoke.
make prefill      # or: make verify-paris

# A single Q&A turn on your own prompt.
make ask PROMPT="What is the capital of France?" N_TOKENS=32

# Interactive chat, streaming. The engine is built once and held across turns.
make chat N_TOKENS=64

# Decode throughput over N_TOKENS tokens (needs the production templates).
make profile N_TOKENS=64

# A local bundle instead of the default repo id.
make run MODEL_SOURCE=/path/to/qwen25_7b_q4nx_bundle
```

## Correctness

`make verify` is the shared gate every `llms/` example uses: top-k token-set
inclusion (k=5, first divergence over 32 greedy tokens) of the NPU q4nx run
against an HF **bf16** reference, driven through
[`verify_adapter.py`](verify_adapter.py) and `llms/verify/`.

**The reference is `Qwen/Qwen2.5-7B-Instruct`** — the very checkpoint the NPU
weights are quantized from, so the gate measures the 4-bit loss and nothing else.
Ungated, so CI needs no license grant. Not `Qwen2.5-7B-Instruct-1M`: same
geometry, but it ropes at theta 1e7 instead of 1e6.

## How it works

### Prefill

The prefill is a **thin driver over the config-parameterized builders in
[`qwen25_3b`](../qwen25_3b/)** — same Qwen2.5 block shape, different constants —
rather than a self-contained clone. `qwen25_7b_q4nx_weights.py` supplies the
config and builds that example's `LlamaWeights`/`LayerWeights` from the
quantized checkpoint; those containers already carry the `bq`/`bk`/`bv` fields the
`rms_qkv_bias_rope` kernel needs.

Two things at 7B are not just bigger constants:

- **`q_dim == emb_dim`.** Both are 3584, so the O-projection is a square
  2048x3584x3584 GEMM.
- **A separate `lm_head`.** Qwen2.5-7B sets `tie_word_embeddings=false` (the 3B
  ties), so the requant emits a real vocab weight slab rather than reusing the
  embedding.
- **SwiGLU has to be split.** At hidden=18944 a single launch needs 1024 DMA
  iterations and exhausts the shim's 16 buffer descriptors; `herd_x` caps at 8
  and `tile_n` at ~5120 (L1), so the floor is 1024. `swiglu_plan` in
  `qwen25_3b/qwen25_3b_prefill.py` splits it into 2 row chunks of 512 iters
  (SwiGLU is elementwise, so a row split is exact) and returns a single chunk —
  the unchanged path — for every smaller hidden_dim.

Unlike Qwen3-8B at K=4096, the LM-head GEMV needs no unusual tile knobs: the L2 A
stage needs `herd_m * tile_m * (K+1) * 2 <= 512 KiB`, which at K=3584 allows
`herd_m * tile_m <= 73`, so the default 64 fits. Only `m_input` comes down to 2,
because `m_input=4` would put the ping-pong L1 A tiles at 2x28 KiB of a 64 KiB L1.
`tile_m` stays 8, so the default `mv.o` is reused as-is.

All four GEMM shapes (2048x3584x{3584,512,18944} and 2048x18944x3584) are in
`kernel_registry/details/GEMM_bf16_in_bf16_out.json`; `gemm_config` raises on an
unregistered shape, so they must be there before the prefill will build.

### Decode

The decode engine is shared with the other q4nx examples. Qwen2.5-7B differs in
two model-level ways and one engine-level way:

**The q/k/v projection bias (`HAS_QKV_BIAS`).** Qwen2.5 is the only family here
whose Q/K/V projections carry a bias. The kernel side already existed
(`fused_decode/kernels/rope.cc::add_q_k_v_bias`, added for the Qwen2.5-3B Q4_0
example) and adds the bias in place *before* RoPE, reading it at `rope_w + DH`.
What was missing was the engine allocating that slab: `ROPE_W_LEN` was
`(3*DH) if HAS_QK_NORM else ROPE_DIM`, with no bias branch, so this model adds
`DH + (DQ+DK+DV) = 128 + 4608 = 4736`. The driver rewrites the whole slab each
position (the cos/sin half changes; the bias half is constant).

**Non-paired proj egress (`PAIR_ROWS=1`).** D=3584 is 7 col-blocks wide, which is
odd in units of the paired egress: `PAIR_ROWS=2` would need
`I2P[0] = 4608/1024 = 4.5`. So this model takes the gemma-style non-paired path,
where each compute tile emits one block into a 4-way memtile gather.

**LM-head vocab chunking is forced, not chosen.** `VOCAB_SIZE_PADDED_FULL =
ceil(152064/3584)*3584 = 154112` → 4816 rowblocks, and `VOCAB_ROWBLKS =
16*VOCAB_I2` at `PAIR_ROWS=1` must divide it, so `UNI_LM * VOCAB_I2 = 301 = 7*43`
and `VOCAB_I2 ∈ {1,7,43,301}`. `K/PAYLOAD = 3584/512 = 7` must divide
`VOCAB_RNDS = VOCAB_I2*PAIR_ROWS`, which drops 1; the tested `2*VOCAB_I2 <= 63`
envelope drops 43 and 301. **`VOCAB_I2=7` is the only legal chunk** → 43 waves of
112 rowblocks. Getting this wrong deadlocks the vocab wave rather than failing
loudly, so the driver must set `VOCAB_CHUNK_I2=7` to match `UNI_LM=43`.

**The weight buffer is split (`DECODE_WGROUP`).** A shim BD's byte offset is a
`uint32` all the way down, so **one BO is only addressable over a 4 GiB span**.
28 layers of Q4NX weights are 3.80 GiB and the lm-head adds 0.32 GiB, so the total
crosses the line. `DECODE_WGROUP=7` splits the layers over four 0.95 GiB buffers
plus a dedicated lm-head buffer — nine buffer args, still **one dispatch**. See
the [Qwen3-8B README](../qwen3_8b_q4nx/README.md#decode) for the mechanism and its
two load-bearing constraints (the selector must fold, and the switch must wrap the
whole phase fan).

## Decode staircase (opt-in)

The decode template streams `ATTN_MAXL` KV positions per token regardless of the
real context length, so a 2048-window build wastes most of that readback at short
context. Building one template pair per window and dispatching each token on the
smallest covering window recovers it, with a token stream identical to the
single-window baseline.

```bash
make compile-decode-windows      # once; WINDOWS="64 512 2048" to override the set
DECODE_STAIRCASE=1 make run
```

Off by default, and not benchmarked for this model — the numbers in
[Performance](#performance-npu2-28-layers-warm) are single-window. See
[`programming_examples/fused_decode/README.md`](../../fused_decode/README.md) for
the mechanism, the measurements on its sibling models and the `.decode_windows`
manifest guard.

## Key Files

| file | role |
|---|---|
| `qwen25_7b_q4nx_inference.py` | driver: AIR prefill (KV seed + first token) + fused NPU decode loop; owns `DECODE_WGROUP` |
| `qwen25_7b_q4nx_prefill.py` | thin driver over the `qwen25_3b` builders + its `causal_lm` interface |
| `qwen25_7b_q4nx_weights.py` | weight sources (HF quantize-on-load, or a `model.q4nx` bundle), RoPE LUTs, numpy reference |
| `qwen25_7b_q4nx_requant.py` | the Q4NX quantizer + the decode cascade cache (2 norm stacks + the untied lm_head) |
| `Makefile` | build / run / verify / verify-full / verify-paris / diagnosis / profile |
| `verify_adapter.py` | Binds the Q4NX prefill + fused decode to the shared `verify/` runner contract |
| `../../fused_decode/models/qwen2.5-7b.h` | kernel-side model config |
| `../../fused_decode/fused_decode.py` | the shared superkernel IR builder |
