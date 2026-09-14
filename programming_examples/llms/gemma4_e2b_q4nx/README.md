# Gemma4-E2B (text) Q4NX on NPU2

FastFlowLM's 4-bit `model.q4nx` bundle for Gemma4-E2B, running on the NPU
through the [`fused_decode_ple`](../../fused_decode_ple) engine.

Both halves run on the NPU: the prefill end to end, and the decode token by
token through a single 35-layer dispatch.

```bash
export PEANO_INSTALL_DIR=/path/to/llvm-aie   # must be >= 22.0.0, see Reproducibility
make compile               # prefill ELFs + both decode templates
make run                   # generate on device; gated on the Paris continuation
make ask PROMPT="..."      # a single Q&A turn
make prefill               # prefill only, gated vs the CPU reference
make profile               # TTFT + decode tok/s into the nightly's perf.json row
make verify                # the on-device decode gate: every layer class
make layer-gate            # just the own-KV classes
make layer-gate-shared     # just the KV-shared classes
make verify-topk           # the shared top-k gate vs the bf16 HF reference
make help                  # everything else
```

**Two decode templates, and they are not interchangeable.** `compile-decode`
builds the layer gate's -- ONE decoder layer, left in the engine directory.
`compile-decode-full` builds the token driver's -- all 35 layers, left here as
`decode_L<N>.{xclbin,insts.bin}`. Pointing the driver at the gate's template does
not fail; it dispatches a 1/35-scale model.

`make verify` here scores the decoder layer per class against a numpy reference
rather than the siblings' top-k token-set check. That check is available as
`make verify-topk` / `make verify-full`, but it is not what `verify` runs: the
layer gate needs no second checkpoint, and the top-k one downloads a bf16
Gemma4-E2B. Two gates over different things, neither subsuming the other.

## What makes this model need its own engine

Two things, both of which `fused_decode` has no room for.

**Per-layer embeddings (PLE).** Gemma4 carries a second per-token embedding
table and folds a slice of it into every layer, after the FFN residual. The
per-layer input is itself computed on device from the token embedding rather
than the hidden state — a 1536→256 projection with a per-layer weight slice,
RMS-normed and added to that layer's row of the table. Three extra herds.

**A per-layer class map.** Layer type crosses with the KV/FFN regime:

| | own KV, FFN 6144 | shared KV, FFN 12288 |
|---|---|---|
| **sliding** (dh 256, θ 1e4) | 0–3, 5–8, 10–13 | 15–18, 20–23, … (KV from 13) |
| **full** (dh 512, θ 1e6, partial rotary 0.25) | 4, 9, 14 | 19, 24, 29, 34 (KV from 14) |

The last 20 layers carry no k/v projection at all: they attend the cache of the
last layer of their *own type* below the sharing boundary (35 − 20 = 15). The
wide FFN spends exactly what the skipped KV projection saved.

## The gate

`make check` is the CPU-only per-layer gate against FastFlowLM's own golden
activations. It is the reference's gate, and it needs an FLM source tree
(`FLM_REFERENCE`) which is not published — so it is a local development tool.

`make layer-gate` / `make layer-gate-shared` are the on-device gates, and the
ones CI runs. They feed a chosen hidden state through one dispatch and compare
the layer output element for element against the numpy reference. One dispatch
exercises the whole device path at once: q4 projections, the 5-norm sandwich,
both attention types, the GLU, and the PLE branch.

Coverage is **by class, not by sample** — a layer from one class says nothing
about the others:

| layer | class | cos |
|---|---|---|
| 0 | sliding, own kv, 6144 | 0.999385 |
| 4 | full, own kv, 6144 | 0.998385 |
| 14 | full, own kv, 6144 | 0.992663 |
| 15 | sliding, kv←13, 12288 | 0.997008 |
| 19 | full, kv←14, 12288 | 0.991631 |
| 34 | full, kv←14, 12288 | 0.999172 |

Two things worth knowing about how those numbers are produced.

**A shared layer needs a two-layer build.** It has no k/v of its own, so slab 0
runs the layer that owns the cache and slab 1 runs the shared layer with
`DECODE_KV_SRC=0,0` pointing its readback at slab 0.

**A chain must not be scored against a pristine reference.** End-to-end, the
`[14,19]` pair reads 0.975 and looks like a failure. It isn't: layer 19's
12288-wide FFN roughly doubles its contribution to the residual stream, so it
*amplifies* the 0.992663 it is handed to 0.986433 before making any error of its
own. `--prefix-dump` starts the reference chain from the device's own upstream
output, which is what isolates the layer under test.

## Reproducibility

**Peano must be at least 22.x.** The repo's pinned llvm-aie (21.0.0) compiles
this model and then returns all-NaN from every layer — a silent wrong answer,
not a build failure, and one that looks exactly like a numerics bug in the
design. The engine's `preflight-peano` refuses, and the lit tests require the
`peano_ge22` feature so they skip rather than red on the pin.

**Weights.** `MODEL_SOURCE` defaults to the HF repo id
[`FastFlowLM/Gemma4-E2B-IT-NPU2`](https://huggingface.co/FastFlowLM/Gemma4-E2B-IT-NPU2)
and accepts a local directory or file. The repo is ungated, and it also ships
`tokenizer.json` / `config.json` / `chat_template.jinja`, so the CPU-reference
targets need no second checkpoint.

The bundle carries **two** codecs — Codec B (I8, packed) for the projections and
lm_head, int8 group-32 with an f32 per-group scale for the two embedding tables
— and `model.per_layer_token_embd.weight` is already pre-scaled by `sqrt(256)`.

## Prefill

`gemma4_e2b_q4nx_prefill.py` builds 22 ELFs and runs the whole prompt on the
NPU: RMSNorm, Q/K/V, per-head QK-norm, the weightless value-norm, RoPE, MQA
flash attention, the GELU-tanh GLU, the per-layer-embedding branch and the LM
head. `make prefill` scores the full 262144-wide logit vector against
`gemma4_e2b_q4nx_weights.forward_prompt`; measured **cosine 0.998582**, argmax
9079 `' Paris'`.

Twenty-two rather than Gemma3-4B's eight, because the per-layer class map
reaches the ELF shapes: every attention-shaped ELF is built at head_dim 256 and
512, every FFN-shaped one at 6144 and 12288, plus the three PLE stages.

Two device constraints are worth knowing before changing the tiling:

- **One GEMM per ELF.** Two GEMM slices stitched into one ELF returned partial
  NaN nondeterministically -- the same binary and inputs, clean on 2 of 4
  repeats. Q, K and V therefore each get their own ELF rather than sharing the
  siblings' fused `rms_qkv_qknorm_rope`.
- **Narrow GEMMs take fewer herd columns, not a smaller tile.** At N=256 the
  4-column/64-wide shape was nondeterministic the same way; 2 columns x 128 is
  stable over repeats and more accurate. See `gemm_herd_n`.

The 28 sliding layers and the 7 full layers differ in head_dim (256 / 512), and
512 is a head dim no other model here builds -- `shared/infra/fa_headfirst`
gained a `_FA_TILING[512]` entry for it, verified on device at cos 0.999700
against a float64 SDPA reference.

## Decode

`gemma4_e2b_q4nx_inference.py` runs the prefill, seeds the device KV cache from
it, and then dispatches one token at a time: all 35 decoder layers, the PLE
branch and the 4-bit LM head in a single dispatch per token. One template built
at `RUN_LBUILD` serves every context length in `[1, ATTN_MAXL]`; the
L-dependent instruction words are patched per token.

`make run` is the gate. Greedy from `<bos> The capital of France is`, it must
produce `[9079, 236761]` -- `' Paris.'` -- and then stop on `<end_of_turn>`.
That sequence is not merely recorded from a device run: the CPU oracle
(`forward_prompt`, re-run on the growing prompt) emits exactly
`[9079, 236761, 106]`.

**The gate covers the stop, not just the tokens,** because two of them is a
short sequence and the failure mode here is a correct first token followed by
plausible garbage. The first token comes from the *prefill*, so it passes even
when the decode is broken -- seeding the KV cache without the padded-head
interleave produced `' Parisнии est le Humदा is a het de'`.

**The KV hand-off is the part with no sibling.** Every device KV row is
`REGION_W = 1024 = 2 CUs x 512`, and the single MQA head is replicated into both
halves. A sliding layer's head is 256 wide and does *not* sit contiguously in
its 512-wide slot: it is scattered as `[real_lo | zeros | real_hi | zeros]` so
that the rope kernel's fixed `(i, i+256)` pairing lands on the real `(i, i+128)`
pairs. `seed_kv` is the only place that knows this, and getting it wrong seeds
plausible garbage rather than raising.

Measured here end to end (`make profile`, 64 tokens from a 6-token prompt):
TTFT **4.505 s**, decode **19.72 tok/s**. That sits just above the synthetic
sweep's 18.21 tok/s at 1k context, which is what a 70-slot cache against a
1024-slot one should look like.

## Benchmarks

Both curves the [LLM benchmark page](https://xilinx.github.io/mlir-air/llms/)
carries are wired up:

- `run_npu2_prefill_sweep.lit` -- TTFT vs padded prompt length. Measured here:
  512 -> 1184 ms (432 tok/s), 1024 -> 2177 ms (470 tok/s), 2048 -> 4719 ms
  (434 tok/s), first-token gate passing at every length.
- `run_npu2_sweep.lit` -- decode tok/s vs KV depth, all 35 layers. Measured
  here: 18.21 / 16.27 / 13.39 / 9.91 / 6.52 / 3.87 / 2.14 tok/s at
  1k / 2k / 4k / 8k / 16k / 32k / 64k; 128k is out of reach.

The decode sweep's points are all marked expected-fail, which is temporary and
is the subject of the section below -- it publishes those numbers on a host that
can produce them, and does not redden a nightly on one that cannot.

`run_npu2_profile.lit` runs `profile-prefill`, so the published row carries a
real TTFT and a null decode tok/s. That is the same #1984 constraint, NOT a
missing driver: `make profile` measures both halves, and does so with
`--ignore-eos`, because this model answers the Paris prompt in two tokens and a
run that honours the stop reports per-call setup rather than decode.

## The decode dispatch hang (#1984)

`_compile_decode_build` builds exactly what the sweep wants (the full 35-wave
decode, not the layer gate's single wave), and on a development box every
context dispatches cleanly. On the benchmark runner the same binaries hang
nondeterministically: ERT_CMD_STATE_TIMEOUT on ~40% of attempts at 4-5 waves,
which compounds to near-certain failure at 35.

The only measured difference between the two hosts is amdxdna/XRT 2.21.0 on the
runner against 2.23.0 on the development box; firmware, Peano and power mode
match. That is why `run_npu2_sweep.lit` marks every context expected-fail rather
than being held out of the tree: the build is still exercised and the numbers
are still published wherever the dispatch works. Drop `--expect-fail` once this
is fixed.

**This, not a missing driver, is why the token gate has no lit.** `make run` is
a real gate and passes locally, but every one of its dispatches is a 35-wave
decode. A lit around it would not go red on this runner, it would TIME OUT and
block the nightly, which `--expect-fail` cannot express. So the decode gate is
local-only, `run_npu2_profile.lit` measures TTFT through `profile-prefill`, and
CI's on-device decode coverage stays the per-layer gate (one wave, reliable).
When #1984 is fixed, three things land together: `--expect-fail` comes off the
sweep, the profile lit goes back to `make profile` with `--n-tokens 64`, and a
`run_npu2_verify_decode.lit` around `make run` becomes possible.

Reproduce with:

    make -C ../../fused_decode_ple compile-decode LBUILD=1024 UNI_DEC=35
