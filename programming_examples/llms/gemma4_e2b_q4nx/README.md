# Gemma4-E2B (text) Q4NX on NPU2

FastFlowLM's 4-bit `model.q4nx` bundle for Gemma4-E2B, running on the NPU
through the [`fused_decode_ple`](../../fused_decode_ple) engine.

**This example is partial, and the target list says which half is which.** The
decoder layer runs on the NPU and is gated there per layer. There is no prefill
engine and no token-level generation on device yet, so the `run` / `ask` /
`verify` / `profile` targets the other `llms/` examples carry are deliberately
absent rather than present and broken.

```bash
export PEANO_INSTALL_DIR=/path/to/llvm-aie   # must be >= 22.0.0, see Reproducibility
make compile-decode        # build the decode template (weight-free)
make layer-gate            # score the own-KV layer classes on device
make layer-gate-shared     # score the KV-shared layer classes
make help                  # everything else
```

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

## Not here yet

Prefill, token-level generation on device, a top-k verify against an HF bf16
reference, and the perf/sweep lits the mature examples publish to the [LLM
benchmark page](https://xilinx.github.io/mlir-air/llms/). Those need a prefill
engine this example does not have.
