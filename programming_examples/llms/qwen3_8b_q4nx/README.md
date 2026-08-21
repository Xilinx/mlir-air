# QWEN3-8B Q4NX prefill + decode on AMD NPU2

This implementation is an MLIR-AIR reimplementation, based on the corresponding
AMD NPU LLM design originally developed by the
[FastFlowLM](https://github.com/ROCm/FastFlowLM) team.

Qwen3-8B in MLIR-AIR, reproducing FastFlowLM's mechanism end to end on the NPU:

- **prefill** — a batched pass over the padded context: RMSNorm, Q/K/V GEMMs,
  per-head QK-norm, RoPE, GQA flash attention, SwiGLU and the LM head, all on
  device with resident weight BOs.
- **decode** — **one NPU dispatch = 36 decoder layers + the LM head**, reading
  and appending a shared per-layer KV cache seeded by the prefill.

Weights are Q4NX throughout (dequantized on device for decode, on the host at
load for prefill).

This is the largest model on the shared superkernel engine in
[`programming_examples/fused_decode`](../../fused_decode/), selected with
`DECODE_MODEL=qwen3-8b`. It is also the model that forced the weight-buffer
split described under [Decode](#decode).

## Architecture

Verified against FastFlowLM's `Qwen3-8B-NPU2` packaging and the HF config:

| | |
|---|---|
| model dim / head dim | 4096 / 128 |
| attention | 32 query heads, 8 KV heads (GQA 4), scale 1/sqrt(128) |
| MLP | intermediate 12288, SwiGLU |
| layers / vocab | 36 / 151936 (**untied** — a separate Q4NX `lm_head`) |
| norms | 2 per layer (input, post-attention), plain `w`, eps 1e-6 |
| RoPE | single theta 1e6, half-split |
| extras | qk-norm (RMSNorm over head_dim, before RoPE); no sliding window, no embed scale |

## Performance (NPU2, AMD Strix, 36 layers, warm)

| | |
|---|---|
| prefill (TTFT) | **7.72 s** at CTX=2048 (265 tok/s); 7.70 s on-device, 20 ms host |
| decode | **10.22 tok/s** (97.8 ms/token), ATTN_MAXL=2048, 64 tokens |

FastFlowLM publishes **11.9 tok/s** for this model at 1k context, so decode is
at **0.86x** of the implementation it reproduces.

Prefill runs at a fixed padded context (`CTX`), so its cost is roughly constant
in prompt length rather than proportional to it. Measure it with
`make profile-prefill`. On-device time per op at CTX=2048 (sum over 36 layers):

| flash_attn | up | gate | down_add | rms_qkv_qknorm_rope | o_res_norm | swiglu | lm_head_gemv |
|---|---|---|---|---|---|---|---|
| 1649 ms | 1243 | 1218 | 807 | 684 | 392 | 235 | 35 (x1) |

Of the 6.84 s of measured XRT time, 89% is NPU execution, 8% host→DDR writes of
the dynamic inputs and 2% readback. Attention is the single largest term and the
main open lever, as in the Gemma3-4B sibling.

Separately, loading the model — Q4NX host dequant of ~6 GB of weights plus the
one-time write into resident BOs — takes ~80 s per process. That is a startup
cost, not part of TTFT; the driver reports the two separately.

## Prerequisites

- NPU2 (Strix / AIE2P) + XRT, and a Peano (llvm-aie) install
- `source utils/env_setup.sh` (see the repo README)
- Weights: `FastFlowLM/Qwen3-8B-NPU2` from the Hub (single `model.q4nx`
  safetensors bundle, ~5.6 GB). Override the source with
  `MODEL_SOURCE=<repo id | local dir | model.q4nx path>`.
- Python deps on top of the base environment: `pip install -r requirements.txt`
  (`huggingface_hub` for the bundle, `transformers` for the tokenizer).
- ~6 GB of host memory for the decode weight BOs (five buffers, see below).

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

# Demo run: greedy decode, prints a *** PARIS *** verdict against a recorded
# 10-token continuation.
make run

# Prefill only (no decode loop, no reference) -- first-token 12095 smoke, useful
# when bringing up the prefill on its own.
make prefill      # or: make verify-paris

# A single Q&A turn on your own prompt.
make ask PROMPT="What is the capital of France?" N_TOKENS=32

# Interactive chat, streaming. The engine is built once and held across turns,
# so only the first turn pays the model load. Turns are independent.
make chat N_TOKENS=64

# Decode throughput over N_TOKENS tokens (needs the production templates).
make profile N_TOKENS=64

# Shorter prefill context, and a local bundle instead of the Hub.
make run CTX=2048 MODEL_SOURCE=/path/to/Qwen3-8B-NPU2
```

## Correctness

`make verify` is the shared gate every `llms/` example uses: top-k token-set
inclusion (k=5, first divergence over 32 greedy tokens) of the NPU q4nx run
against an HF **bf16** reference, driven through
[`verify_adapter.py`](verify_adapter.py) and `llms/verify/`.

**The reference is `Qwen/Qwen3-8B`.** FastFlowLM's model card for
`FastFlowLM/Qwen3-8B-NPU2` declares that as `base_model`, and Qwen3 unifies the
base and instruct variants in one checkpoint, so both `MODEL_CHOICES` map to it.
It is ungated, so CI needs no license grant.

Measured on NPU2 (`LBUILD=2048`, `Qwen/Qwen3-8B` reference):

| | result |
|---|---|
| `make verify` (2-prompt CI slice) | **2 / 2 PASS** |
| `make verify-full` (8 prompts) | **8 / 8 PASS** |

Two cheaper checks sit beside the gate. `make verify-paris` is a prefill-only
first-token smoke (no decode templates, no reference) for bringing the prefill up
on its own, and `make run` prints a `*** PARIS ***` verdict against a recorded
10-token greedy continuation — a demo self-check, not the gate. At 64 tokens the
demo continues coherently and factually (`" Paris. The capital of Italy is Rome.
The capital of Germany is Berlin. ..."`).

## How it works

### Prefill

The prefill is a **thin driver over the config-parameterized builders in
[`qwen3_4b`](../qwen3_4b/)** — same Qwen3 shape, different constants —
rather than a self-contained clone. `qwen3_8b_q4nx_weights.py` supplies the
config and builds that example's `LlamaWeights`/`LayerWeights` from the
`model.q4nx` bundle.

Three things at 8B are not just bigger constants:

- **`q_dim == emb_dim`.** Both are 4096, so the decoupled O-projection builder
  degenerates to a square 2048x4096x4096 GEMM. It is parameterized on
  `(q_dim, emb_dim)` and needs no special case.
- **The LM head GEMV needs three non-default knobs at K=4096.** The defaults
  blow two budgets: L2 needs `herd_m * tile_m * (K+1) * 2 <= 512 KiB` (so
  `herd_m * tile_m <= 63`, which the default 64 misses by 128 B), and
  `m_input=4` puts the ping-pong L1 A tiles at 2x32 KiB, all of L1. `herd_m`
  **must stay 8** — `herd_m=4` compiles and then hangs the dispatch. So:
  `herd_m=8, tile_m=4, m_input=2`, plus `compile_mv(tile_m=4)` because `mv.o` is
  keyed to `tile_m` through `DIM_M_OUTPUT`.
- **A separate Q4NX `lm_head`.** Qwen3-8B does not tie embeddings, so the
  requant emits a real vocab weight slab rather than reusing the embedding.

All four GEMM shapes (2048x4096x{4096,1024,12288} and 2048x12288x4096) are in
`kernel_registry/details/GEMM_bf16_in_bf16_out.json`; `gemm_config` raises on an
unregistered shape, so they must be there before the prefill will build.

### Decode

The decode engine is shared with the other q4nx examples. Qwen3-8B differs in
two model-level ways — 2 norms rather than Gemma's 4-norm sandwich, and qk-norm
folded into the packed `rope_w` slab — and in one engine-level way that is
specific to its size:

**The weight buffer had to be split (`DECODE_WGROUP`).** A shim BD's byte offset
is a `uint32` all the way down (`aie.dma_bd $offset` -> `aiex.npu.address_patch
$arg_plus` -> `uint32_t patchedArgPlus` in `AIETargetNPU`), so **one BO is only
addressable over a 4 GiB span**. Qwen3-8B's 36 layers are 4.04 GiB of Q4NX
weights and the lm-head adds 0.37 GiB more; past layer 33 the offsets wrap to
near the buffer start, the model reads layer-0 weights, and every logit comes
back NaN. This is the first model on this engine large enough to hit it —
llama-3.2-3b is 1.87 GiB and gemma3-4b 2.26 GiB.

`DECODE_WGROUP=G` splits the weights over `ceil(36/G)` layer buffers plus a
dedicated lm-head buffer, each addressed from its own base. This example uses
**G=9**: four 1.01 GiB groups (4x under the limit) and a 0.37 GiB lm-head, nine
buffer args, still **one dispatch**. That is the same layer-invariant addressing
FastFlowLM gets by binding one BO per layer, except FLM needs 36 dispatches (and
an XRT runlist) to do it, because their runtime sequence is one reusable program
with one weight arg. Ours is fully unrolled before BD assignment, so each wave's
BDs can name a different arg inline.

The selection is an `scf.index_switch` on the wave induction variable, one arm
per group, wrapped around the weight feeds. Once
`aie-unroll-runtime-sequence-loops` makes the IV constant, `AIRRtToNpuPass`
folds the switch to its single taken arm — the split costs **zero** extra BDs in
the final sequence (verified: 1372 `dma_bd` split and unsplit alike, with the
weight BDs simply re-attributed across args).

Two constraints on that mechanism, both load-bearing:

1. The selector may only use ops in the post-unroll fold set
   (cmpi/select/index_cast/ext/addi/subi/muli). `iv / G` and `iv % G` are the
   obvious spelling and do **not** fold — `divui`/`remui` are not in the set, so
   the switch survives and an `index_switch` cannot parent
   `aiex.dma_configure_task_for`. A nested `select` chain over the group
   boundaries folds fine.
2. The switch must wrap the whole phase fan, not each column. The cross-channel
   phase barrier opens a new phase group at every block boundary, and each
   switch arm is its own block — a per-column switch shrinks the phase group
   from 2*NCX channels to 2 and deadlocks the dispatch.

`DECODE_WGROUP=0` (the default) is a no-op: the emitted IR is byte-identical to
before the knob existed, for every model.

**The decode core stack also had to shrink.** At K=4096 the seven K-wide L1
activation buffers are 7x8 KiB, so the engine's default `stack_size=10240`
overflows the 64 KiB core. The Makefile passes `DECODE_STACK=6144`.

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
| `qwen3_8b_q4nx_inference.py` | driver: AIR prefill (KV seed + first token) + fused NPU decode loop; owns `DECODE_WGROUP` |
| `qwen3_8b_q4nx_prefill.py` | thin driver over the `qwen3_4b` builders + its `causal_lm` interface |
| `qwen3_8b_q4nx_weights.py` | `model.q4nx` loader, Q4NX dequant, RoPE LUTs, numpy reference |
| `qwen3_8b_q4nx_requant.py` | Q4NX re-quantization helper (2 norm stacks + the untied lm_head) |
| `Makefile` | build / run / verify / verify-full / verify-paris / diagnosis / profile |
| `verify_adapter.py` | Binds the Q4NX prefill + fused decode to the shared `verify/` runner contract |
| `../../fused_decode/models/qwen3-8b.h` | kernel-side model config |
| `../../fused_decode/fused_decode.py` | the shared superkernel IR builder |
