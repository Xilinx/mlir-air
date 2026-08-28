# QWEN3-4B Q4NX prefill + decode on AMD NPU2

Qwen3-4B in MLIR-AIR, reproducing FastFlowLM's mechanism end to end on the NPU:

- **prefill** — a batched pass over the padded context: RMSNorm, Q/K/V GEMMs,
  per-head QK-norm, RoPE, GQA flash attention, SwiGLU and the LM head, all on
  device with resident weight BOs.
- **decode** — **one NPU dispatch = 36 decoder layers + the LM head**, reading
  and appending a shared per-layer KV cache seeded by the prefill.

Weights are Q4NX throughout (dequantized on device for decode, on the host at
load for prefill).

Qwen3-4B is the DFlash speculative-decoding **target** model (see
[`docs/DFlashFeasibility.md`](../../../docs/DFlashFeasibility.md)); this is the
driver that makes it a real, runnable NPU target rather than a stand-in. It is
the same shared superkernel engine in
[`programming_examples/fused_decode`](../../fused_decode/) the other Q4NX
examples use, selected with `DECODE_MODEL=qwen3-4b`.

## Architecture

Same topology as [`qwen3_8b_q4nx`](../qwen3_8b_q4nx/) (DH=128, GQA4, per-head
qk-norm, single-theta RoPE, standard 2-norm pre-norm) at smaller dims, plus one
structural difference:

| | |
|---|---|
| model dim / head dim | 2560 / 128 |
| attention | 32 query heads, 8 KV heads (GQA 4), scale 1/sqrt(128) |
| MLP | intermediate 9728, SwiGLU |
| layers / vocab | 36 / 151936 (**tied** — the head is the embedding table, re-quantized) |
| norms | 2 per layer (input, post-attention), plain `w`, eps 1e-6 |
| RoPE | single theta 1e6, half-split |
| extras | qk-norm (RMSNorm over head_dim, before RoPE); the q-projection is decoupled from the hidden dim (32*128=4096 != 2560), so o-proj is a non-square 2560x4096 contraction |

## Correctness

`make verify` is the shared gate every `llms/` example uses: top-k token-set
inclusion (k=5, first divergence over 32 greedy tokens) of the NPU q4nx run
against an HF **bf16** reference, driven through
[`verify_adapter.py`](verify_adapter.py) and `llms/verify/`.

**The reference is `Qwen/Qwen3-4B`.** `FastFlowLM/Qwen3-4B-NPU2`'s model card
declares that as `base_model`, and Qwen3 unifies the base and instruct variants
in one checkpoint, so both `MODEL_CHOICES` map to it. It is ungated, so CI
needs no license grant.

The Paris gate (`make run`, greedy, prompt "The capital of France is") decodes
to `" Paris. The capital of Germany is Berlin. The"` — fluent and factually
correct, recorded as `PARIS_GREEDY`.

## Prerequisites

- NPU2 (Strix / AIE2P) + XRT, and a Peano (llvm-aie) install
- `source utils/env_setup.sh` (see the repo README)
- Weights: `FastFlowLM/Qwen3-4B-NPU2` from the Hub (single `model.q4nx`
  safetensors bundle). Override the source with
  `MODEL_SOURCE=<repo id | local dir | model.q4nx path>`.
- Python deps on top of the base environment: `pip install -r requirements.txt`
  (`huggingface_hub` for the bundle, `transformers` for the tokenizer).

## Quick Start

```bash
# Build everything, weight-free: the prefill ELFs + the decode templates.
make compile

# ...or just one half:
make compile-prefill            # the prefill ELFs (CTX=2048)
make compile-decode LBUILD=16   # small decode templates (Paris gate only)

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

# Interactive chat, streaming.
make chat N_TOKENS=64

# Decode throughput over N_TOKENS tokens (needs the production templates).
make profile N_TOKENS=64
```

## How it works

### Prefill

A thin driver over the config-parameterized builders in
[`qwen3_4b`](../qwen3_4b/) — same Qwen3 shape, different constants — rather
than a self-contained clone. `qwen3_4b_q4nx_weights.py` supplies the config and
builds that example's `LlamaWeights`/`LayerWeights` from the `model.q4nx`
bundle.

### Decode

The decode engine is shared with the other q4nx examples. Two deltas from
[`qwen3_8b_q4nx`](../qwen3_8b_q4nx/), both load-bearing:

- **Non-paired egress (`PAIR_ROWS=1`).** K=2560 is odd in paired-row units
  (`ROW_BLOCK*NCX*NCY*PAIR_ROWS`), so every phase uses the non-paired divisor,
  same as `gemma3_4b_q4nx` and `qwen25_7b_q4nx`.
- **Tied LM head.** `tie_word_embeddings=true`: the head is
  `model.embed_tokens.weight`, re-quantized, not a separate Q4NX `lm_head`
  tensor (`qwen3_4b_q4nx_requant.py`'s tied-head branch, mirroring
  `llama32_1b_q4nx`'s pattern rather than `qwen3_8b_q4nx`'s untied one).

No weight-BO split is needed (unlike `qwen3_8b_q4nx`'s `DECODE_WGROUP=9`): the
total decode weight is ~2.1 GiB (36 layers at K=2560), under the 4 GiB one-BO
shim-BD-offset limit.

**`fused_decode/models/qwen3-4b.h`'s `GLU_SLICE` must be 512, not 1024.** The
Python builder computes its own `GLU_SLICE` from this model's *own* egress
round parity (`GLU_PKTS = 2 if (ROUNDS_PER_DEST[GLU_DEST]//2) % 2 == 0 else 1`)
— for qwen3-4b that evaluates `GLU_PKTS=1`, i.e. `GLU_SLICE=512`, the same odd-
parity case `qwen2.5-7b.h` already documents and sets `GLU_SLICE=512` for
("18944/1024 = 37 slices is odd"). A kernel built at the wrong `GLU_SLICE`
still compiles and dispatches — it just consumes/produces GLU slices at double
the width the IR is actually feeding it, so the FFN/down-projection's
contribution to the residual comes out silently zero, every layer. There is no
compile-time or runtime error; the only symptom is wrong decode output from
the very first generated token, with prefill (which does not go through this
kernel) unaffected. Confirmed via `DECODE_ACC_STOP` (an existing debug knob
that drops the accumulate-add from a chosen pipeline stage without changing
control flow): the broken build's full-layer output was byte-identical to a
build with the FFN's contribution explicitly zeroed for all 36 layers.

## Key Files

| file | role |
|---|---|
| `qwen3_4b_q4nx_inference.py` | driver: AIR prefill (KV seed + first token) + fused NPU decode loop |
| `qwen3_4b_q4nx_prefill.py` | thin driver over the `qwen3_4b` builders + its `causal_lm` interface |
| `qwen3_4b_q4nx_weights.py` | `model.q4nx` loader, Q4NX dequant, tied-head accessor, RoPE LUTs |
| `qwen3_4b_q4nx_requant.py` | Q4NX re-quantization helper (2 norm stacks + the tied lm-head) |
| `Makefile` | build / run / verify / verify-full / verify-paris / diagnosis / profile |
| `verify_adapter.py` | Binds the Q4NX prefill + fused decode to the shared `verify/` runner contract |
| `../../fused_decode/models/qwen3-4b.h` | kernel-side model config |
| `../../fused_decode/fused_decode.py` | the shared superkernel IR builder |
