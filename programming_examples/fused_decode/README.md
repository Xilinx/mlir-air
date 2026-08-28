# Fused superkernel decode (Llama-3.2-1B) on AMD NPU2

A **fused per-token decode** for Llama-3.2-1B in MLIR-AIR: **one dispatch = 16 decoder
layers + LM head**, reading and appending a shared per-layer KV cache. This is the decode
path consumed by the [`llama32_1b_q4nx`](../llms/llama32_1b_q4nx) LLM end-to-end example
(its prefill fills the KV cache; this decode generates autoregressively).

## Design

The whole transformer decode step for one token is built as a single AIR module
(`build_module()` in `fused_decode.py`): 16 layers of Q4NX projections (`proj_qmm`),
RMSNorm + residual, RoPE, SwiGLU (`glu`), and GQA attention (`attn_qk` / `attn_kv`),
followed by the LM-head vocab projection — all in one xclbin dispatch.

**One template serves every context length.** The xclbin is built once at
`ATTN_MAXL=2048`. The attention block loop runs a compile-time 128-block schedule and
**skips fully-masked far blocks**, so a single build is correct for every `L` in
`[1, 2048]`. The per-token, `L`-dependent instruction words (RTP-L bound + KV-append
offset) are patched on the **host** by `decode_insts_gen.py` (`DecodeInstsGen`) — no
per-length recompile, no per-window xclbin. Two same-`ATTN_MAXL` builds
(`decode_L2048` + `decode_L2047`) calibrate the L-slope so the generator can synthesize
the instruction stream for any `L`, byte-identical to a native per-L build.

## Files

| File | Purpose |
|---|---|
| `fused_decode.py` | Builds the fused decode AIR module (16 layers + LM head) and emits the xclbin + insts |
| `decode_insts_gen.py` | Host per-token instruction patcher (`DecodeInstsGen`); serves any L from one template |
| `proj_qmm_pack.py` | numpy Q4NX block packer + dequant reference |
| `kernels/` | Peano decode kernel sources (`proj_qmm`, `rms_residual`, `glu`, `rope`, `attn_qk`, `attn_kv`, headers) |
| `models/` | model spec headers |
| `q4_0_codec.py` | HF safetensors reader, Q4_0 quantizer and vectorized cascade packer, shared by the Q4_0 models |

### Two on-device weight formats, one builder

`fused_decode.py` covers every model, selected by `DECODE_MODEL`. What differs
between them is the **on-device weight format**, and the split is by codec, not
by model family: Qwen2.5-7B is Q4NX and sits with Llama and Gemma, while
Qwen2.5-3B and LFM2-1.2B are Q4_0.

| | Q4NX (default) | Q4_0 |
|---|---|---|
| device format | unsigned int4 **+ per-group min** | signed int4, **scale only** |
| dequant | `w = scale*q + min` | `w = q*scale` |
| kernel build | (default) | `-DQ4_0` |

Both come from `kernels/q4_k.h`, which carries the two forms under `#ifndef
Q4_0`; in-tree the switch is set per model, e.g.
[`models/qwen2.5-3b.h`](models/qwen2.5-3b.h). This follows the shipped bundles:
FastFlowLM's Qwen2.5-3B `model.q4nx` has an all-zero `mins` field in every
tensor and signed nibbles, and its own design sets `#define Q4_0` to match,
while its Llama and Gemma designs leave it undefined.

### The Q4_0 branch splits again, on the toolchain

Only the Q4_0 side has a *signed* nibble to decode, and only chess can do it
directly: `aie::to_float(vector<int4>)` on peano folds to zero and the whole
matmul is dead-code-eliminated. So under `#if defined(__chess__)` the two
backends compile **different source** from this one file -- worth knowing before
concluding that a peano and a chess build of "the same kernel" should cost the
same.

The peano side loads the nibbles as `uint4` and sign-corrects them itself, under
`Q4_SFIX_MODE`:

| | how | inner-loop body |
|---|---|---|
| `1` (default) | xor bias: `(u ^ 8) - 8 == q`, the xor done on the packed bytes since `0x88` flips both nibbles | 140 bundles |
| `0` | compare/select: `select(f, f - 16, f >= 8)` | 355 bundles |
| `2` | no sign fix at all -- **numerically wrong**, for attributing static cost only | 121 bundles |
| | (chess, for reference) | 142 bundles |

Mode 0 is only 64 bundles of compare and select; the rest of its cost is spill
traffic, because keeping four `vector<float,128>` and two broadcast constants
live at once is well past the register file. That body runs 8x2 per 32x256
weight block, and an AIE2P core issues about one bundle per cycle, so the
difference is most of the projection: it moved the qwen2.5-3B decode intercept
from 81.1 to 32.6 ms/token. `aie::bit_xor` on a `vector<uint4>` segfaults
peano's frontend, which is why the xor is applied a byte at a time.

The weight *bundles* are uniform: every FastFlowLM `model.q4nx` is the same
per-block affine encoding regardless of model. That is why `llms/qwen25_3b_q4`
quantizes the fp checkpoint directly rather than reading a bundle -- an affine
bundle re-quantized into the symmetric device form would quantize twice.

## Decode staircase: one template per KV window (opt-in)

The compiled KV readback streams `ATTN_MAXL` positions **whatever the real context length
is**. `RB_ROUNDS` (default `ceil(ATTN_L/16)`) feeds the outer size of the readback nd-DMA
and is a build constant, so a template built at `ATTN_MAXL=2048` moves the whole padded
cache on every token -- 224 MiB/token on the 3B -- to use 7 MiB of it at L=50. The core
does skip fully-masked far blocks, but only *after* the DMA has hauled them from DDR:
masking saves compute, not bandwidth.

`DecodeInstsGen` cannot patch this away. It recovers per-word slopes by diffing two builds,
and `ceil(L/16)` is a step function whose calibration points (e.g. 2047 and 2048) sit in
the same step, so the readback count never appears among its L-dependent words.

The staircase builds a template pair **per window** and dispatches each token on the
smallest one covering the current L. At the same L, an `ATTN_MAXL=64` build and a 2048
build differ in only 500 of 70,648 instruction words -- all `ATTN_MAXL`-scaled shim-NOC BD
sizes, strides and offsets.

### Using it

```bash
# once: builds a pair per window (default 64 128 256 512 1024 2048)
make -C ../llms/llama32_1b_q4nx compile-decode-windows

DECODE_STAIRCASE=1 make -C ../llms/llama32_1b_q4nx chat     # 1B, gemma3_4b_q4nx
make -C ../llms/llama32_3b_q4nx chat ... --staircase        # 3B uses a CLI flag
```

`WINDOWS="64 512 2048"` overrides the set. Off by default: with a single template the
behaviour and the code path are unchanged, and `staircase=True` against a one-window
directory is identical to off.

Measured over 300 greedy tokens from an empty cache, token stream **identical** to the
single-window baseline in every case:

| model | baseline | staircase | |
|---|---|---|---|
| `llama32_1b_q4nx` | 53.00 tok/s | 58.61 | **1.106x** |
| `llama32_3b_q4nx` | 22.20 tok/s | 24.45 | **1.101x** |
| `gemma3_4b_q4nx` | 15.41 tok/s | 18.55 | **1.204x** |

The saving is `(1 - L/ATTN_MAXL) x` the full readback cost -- ~10% at short context, ~5%
typical, and 0 once the context fills the window. Gemma gains most: 34 layers at head_dim
256 is the largest padded cache.

### How a window switch stays cheap

Host-only BOs stay valid across an xclbin/hw_context swap, so every window's kernel is
created once up front and a switch is kernel selection plus a KV re-space -- **no weight
re-upload**. The cache is region-major on `ATTN_MAXL*REGION_W`, so changing window re-lays
each region's live prefix (`respace_kv` in `decode_staircase.py`); the cost is proportional
to the live positions, and a crossing happens exactly when that prefix is small (~33 ms,
three crossings per 300 tokens).

Templates are discovered by scanning a directory, and the staircase makes multi-window
directories normal rather than a mistake, so the build writes a `.decode_windows` manifest
and `DecodeInstsGen` rejects a directory whose calibrated set does not match it. A stray
pair fails loudly instead of silently changing which window is selected.

## Dual-MM2S weight feed (`W_DUAL_CHAN`, on by default)

Batch-1 decode is a weight-streaming problem: every token reads every weight once,
with no reuse. Each proj column's weights are therefore fed on **both** of that
column's shim MM2S channels rather than ch0 alone, following FastFlowLM. Set
`W_DUAL_CHAN=0` to fall back to the single-channel feed; the flag reaches both the
build and the run (the DDR weight layout changes with it), and every consumer keys
its requant cache and its `.decode_flags` template stamp on it, so the two can
never silently disagree.

Three things make it work, all mirroring FLM's `mem_C_1`:

| | what | why |
|---|---|---|
| split axis | **spatial**, by cascade pair -- ch0 feeds rows 2/3, ch1 rows 4/5, on two independent lock cycles | a *temporal* split (alternating fan steps) gives every core one MM2S chain alternating between both channels' buffers, couples the two shim channels at every step, and deadlocks |
| shim placement | per-column channels `@inW{0,1}c{cx}`, columns derived | a `[NCX]` bundle cannot express a per-index column, so the channels stay separate; the column itself is not stated -- each feeds an L2 buffer pinned with `air.memtile_col`, and the tile placer puts the shim in the column of the memtile it feeds |
| X memtile | on the hub column (`XMT_PCOL = MAIN_PCOL`) | FLM has **no** memtile in column 2. Leaving the 16-way X broadcast there alongside the shim feeds and both cores makes the pathfinder fail outright once the weight flows double |

The DDR side is a pure permutation of the packed cascade
(`pack_q4k_cascade(dual_chan=True)`), so each channel still reads one contiguous
1-D run -- a strided feed cannot, because a 10240-element fan step exceeds the AIE2
per-dim wrap limit and only a contiguous BD gets the wide `buffer_length` register.

Measured on Strix (quiet box, `make profile N_TOKENS=64`, 3 runs each; the nightly
dashboard numbers come from the Krackan Point runner instead):

| model | single-channel | dual-channel | |
|---|---|---|---|
| Llama-3.2-1B | 46.2 tok/s | 52.3 tok/s | **1.13x** |
| Llama-3.2-3B | 18.8 tok/s | 22.5 tok/s | **1.20x** |
| Gemma3-4B | 15.2 tok/s | 17.1 tok/s | **1.12x** |

The retired Qwen-only builder gained nothing from the second channel (105.1 vs
105.5 ms/tok), and that was expected: it streamed 1.91 GB/token at ~105 ms =
**18.2 GB/s**, while the Llama-1B design already sustains 35.7 GB/s on four
channels. Its layers did not pipeline, so it was not weight-bandwidth-bound and
doubling the channels could not help. Check GB/s before assuming the feed is the
bottleneck. Qwen2.5-3B now runs on this builder and has not been re-measured
single-channel.

## Reproducibility (toolchain)

`make compile-decode` merges an inline attention kernel into the core via an **external
`llvm-link`** (resolved from `PATH`), which must come from **LLVM < 23** — the last
release series before the `llvm.lifetime` change. A ≥ 23 one rewrites that intrinsic to
the no-size form, which Peano `opt` then rejects (`Broken module found`). Keep any such
LLVM's `bin` off `PATH` for this build; the Makefile preflight aborts on it.

This is the same rule the lit harness gates on (`llvm_link_pre23` in
`programming_examples/lit.cfg.py`), and the two must stay in sync: every decode e2e test
is `REQUIRES`-gated on that feature, so a preflight stricter than lit turns an
UNSUPPORTED into a hard test failure. Verify:

```bash
which llvm-link && llvm-link --version   # major must be < 23
```

The Peano wheel (`llvm-aie`) does **not** ship `llvm-link`, and the MLIR distro wheel's
is too new (LLVM 24 today), so on a machine with neither you need to fetch one. Any
`llvm-link` from an LLVM < 23 release works; without root, extracting the Debian
packages is enough:

```bash
LLVM_V=21   # any < 23
BASE=https://apt.llvm.org/$(lsb_release -cs)/pool/main/l/llvm-toolchain-$LLVM_V
# grab the llvm-$LLVM_V and libllvm$LLVM_V .deb names from that index, then:
dpkg-deb -x llvm-$LLVM_V*.deb   unpack/
dpkg-deb -x libllvm$LLVM_V*.deb unpack/
export LD_LIBRARY_PATH=$PWD/unpack/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PATH=$PWD/unpack/usr/lib/llvm-$LLVM_V/bin:$PATH
```

Prepend only a directory holding `llvm-link` itself — putting a full system
`bin` ahead of the AIE toolchain shadows Peano's `clang`/`opt`/`llc`. (`lit.cfg.py`
does exactly this, via a shim dir containing a single symlink.)

Nothing compiled is committed — `make compile-decode` reproduces every kernel object,
xclbin, and instruction stream from source (see `.gitignore`).

## Build

```bash
# Weight-free build (weights are runtime BOs); reproduces both templates from source.
make compile-decode
```

Produces `decode_L2048.{xclbin,insts.bin}` + `decode_L2047.{xclbin,insts.bin}`.

## Use

The decode is exercised end-to-end from the LLM example:

```bash
cd ../llms/llama32_1b_q4nx
make chat        # interactive chatbot (prefill fills KV, this decode generates)
make gen         # prefill+decode Paris gate -> *** PARIS ***
```

Per-kernel `-O` is load-bearing (encoded in the Makefile): `proj_qmm` / `rms_residual` /
`glu` / `rope` at `-O2`; `attn_qk` / `attn_kv` at `-O1` (a `-O2` do-while deadlock; and
`rope` at `-O1` miscompiles). The rolled 128-block decode loop is single-buffered:
`air-label-scf-for-to-ping-pong` declines it, because the running max and accumulator are
live across blocks and the score buffer is shared with the kv tile.
Enable turbo for ~50 tok/s:
`xrt-smi configure -d <BDF> --pmode turbo`.
