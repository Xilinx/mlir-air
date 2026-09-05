# Fused Superkernel Decode + Per-Layer Embeddings (PLE)

A fork of [`fused_decode`](../fused_decode) that adds the branch Gemma4-E2B needs
and the base engine has no room for. One dispatch is still the whole model —
every decoder layer plus the LM head, reading and appending a shared per-layer KV
cache — with three extra herds on top.

Consumed by [`llms/gemma4_e2b_q4nx`](../llms/gemma4_e2b_q4nx).

## What PLE adds

Gemma4 carries a second per-token embedding table and folds a slice of it into
every layer, after the FFN residual:

```
gate = gelu_tanh(x @ W_inp_gate.T) * per_layer_input[L]
out  = x + rmsnorm(gate @ W_per_layer_projection.T, post_layernorm)
```

`per_layer_input[L]` is itself computed on device, from the token embedding
rather than the hidden state: a 1536→256 projection with a per-layer weight
slice, RMS-normed and added to that layer's row of the per-layer token embedding
table. That is three herds — `proj`, `gate`, `up` — placed in column 3, fed by
their own `@pleW` packet channel plus a `@pleX` channel for the singletons, and
handing the residual to the RMS core over shared L1.

## Build

```bash
export PEANO_INSTALL_DIR=/path/to/llvm-aie   # must be >= 22.0.0, see below
make compile-decode                          # kernels + one template, ATTN_MAXL=16
make compile-decode LBUILD=2048              # production-sized
```

Weight-free: weights are runtime BOs, so no bundle is read at build time.

Unlike `fused_decode`'s `compile-decode`, this one is **not** idempotent. The
template is keyed on three things a consumer varies per run (`LBUILD`,
`UNI_DEC`, `KV_SRC`), and the layer gate does not build — it dispatches whatever
`decode.xclbin` is on disk. A skipped rebuild therefore scores a different design
and reads as a numerics regression.

## Reproducibility

**Peano must be at least 22.x.** The repo's pinned llvm-aie (21.0.0) miscompiles
this model to all-NaN on every layer — a silent wrong answer, not a build
failure, and one that looks exactly like a numerics bug in the design.
`make preflight-peano` checks it and refuses.

It does **not** inherit `fused_decode`'s `preflight-llvm-link`. That target
aborts unless `PATH` carries an `llvm-link` older than LLVM 23, and this build
does not use one: instrumenting `PATH` with a logging `llvm-link` shim across a
full build recorded zero calls, and the build succeeds with an LLVM 23
`llvm-link` first on `PATH`. The inline-attn merge goes through mlir-aie's own
path.

## CI

There is deliberately **no lit test in this directory**. The compile gate lives
at [`llms/gemma4_e2b_q4nx/run_npu2_compile.lit`](../llms/gemma4_e2b_q4nx/run_npu2_compile.lit),
which drives this Makefile and is picked up by `check-programming-examples-llms-compile`
(`-j1`). A second lit here would build into the same directory as that one and
race it under the unfiltered `check-programming-examples` suite, which has no
`-j1`.

## Fork fidelity

For every model *other* than gemma4-e2b this builder must emit IR
byte-identical to its parent's. That check is the only thing keeping the two
from drifting, so run it after any edit:

```bash
cd ../fused_decode     && FUSED_DECODE_EMIT_ONLY=1 python3 fused_decode.py     > /tmp/a.mlir
cd ../fused_decode_ple && FUSED_DECODE_EMIT_ONLY=1 python3 fused_decode_ple.py > /tmp/b.mlir
diff /tmp/a.mlir /tmp/b.mlir
```

The one difference the fork carries deliberately: it imports `proj_qmm_pack`
from `../fused_decode`, because the q4nx codec is a bundle format rather than a
per-model one.

## Knobs worth knowing

| Variable | Default | Meaning |
|---|---|---|
| `LBUILD` | 16 | `ATTN_MAXL` the template is built for |
| `UNI_DEC` | 1 | decode waves in the unified sequence |
| `KV_SRC` | *(unset)* | per-wave KV **source** slab, e.g. `0,0` |

`KV_SRC` exists for Gemma4's KV sharing: its last 20 layers carry no k/v
projection and attend the cache of the last layer of their own type below the
boundary. It moves only the readback — the append stays on the wave's own slab,
because a per-wave append geometry that varies would split the channel into more
than one BD task, and a multi-task channel in AIR runs *once per PDI load*
rather than once per wave.
