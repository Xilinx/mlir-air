# Gemma4-E2B (text) Q4NX on NPU2

FastFlowLM's 4-bit `model.q4nx` bundle for Gemma4-E2B, running on the NPU
through the [`fused_decode_ple`](../../fused_decode_ple) engine.

**This example is partial, and the target list says which half is which.**
Prefill runs end to end on the NPU and is gated against the CPU reference. The
decoder layer runs on the NPU too, but is gated per LAYER rather than per token:
there is no token-level generation driver yet, so the `run` / `ask` / `profile`
targets the other `llms/` examples carry are deliberately absent rather than
present and broken -- each of them generates through a decode loop.

```bash
export PEANO_INSTALL_DIR=/path/to/llvm-aie   # must be >= 22.0.0, see Reproducibility
make compile               # build the prefill ELFs + the decode template
make compile-prefill       # just the 22 prefill ELFs (weight-free)
make prefill-paris         # prefill on device, gated vs the CPU reference
make verify                # the on-device decode gate: every layer class
make layer-gate            # just the own-KV classes
make layer-gate-shared     # just the KV-shared classes
make help                  # everything else
```

`make verify` here scores the decoder layer per class against a numpy reference
rather than running the siblings' top-k token-set check, which needs a prefill.

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
head. `make prefill-paris` scores the full 262144-wide logit vector against
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

## Not here yet

Token-level generation on device, and the top-k verify against an HF bf16
reference that needs it.

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
is fixed. Reproduce with:

    make -C ../../fused_decode_ple compile-decode LBUILD=1024 UNI_DEC=35
