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
| `fused_decode_qwen.py` | The Qwen2.5-3B variant of the builder (see below) |
| `qwen25_3b_requant.py` | Q4_0 quantizer + weight cache for the Qwen decode |
| `qwen_prefill_to_decode.py` | Hands `llms/qwen25_3b_q4`'s prefill KV to the Qwen decode |

### Why Qwen has its own builder

`fused_decode.py` covers Llama-3.2-1B/3B and Gemma3-4B, selected by
`DECODE_MODEL`. Qwen2.5-3B is a separate file because its **on-device weight
format differs**, not because its weights are packaged differently:

| | Llama / Gemma | Qwen2.5 |
|---|---|---|
| device format | unsigned int4 **+ per-group min** | signed int4, **scale only** |
| dequant | `w = scale*q + min` | `w = q*scale` |
| kernel build | (default) | `-DQ4_0` |

Both come from `kernels/q4_k.h`, which carries the two forms under `#ifndef
Q4_0`; in-tree the switch is set by [`models/qwen2.5-3b.h`](models/qwen2.5-3b.h).
This mirrors FastFlowLM, whose own Qwen design sets `#define Q4_0` in its
`Qwen2_5/decoding_3b/models/qwen2_3b.h` while its Llama and Gemma designs leave
it undefined.

The weight *bundles* are uniform: every FastFlowLM `model.q4nx` is the same
per-block affine encoding regardless of model. That is why `llms/qwen25_3b_q4`
quantizes the fp checkpoint directly rather than reading a bundle -- an affine
bundle re-quantized into the symmetric device form would quantize twice.

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
| shim placement | per-column channels `@inW{0,1}c{cx}`, each pinned with `air.shim_col` | a `[NCX]` bundle cannot express a per-index column; unpinned, the extra flows steal the rms/rope column and silently violate *its* `air.shim_col` |
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
| Qwen2.5-3B | 105.1 ms/tok | 105.5 ms/tok | none |

Qwen gains nothing, and that is expected: it streams 1.91 GB/token at ~105 ms =
**18.2 GB/s**, while the Llama-1B design already sustains 35.7 GB/s on four
channels. Qwen's decode is not weight-bandwidth-bound (its layers do not
pipeline), so doubling the channels cannot help. Check GB/s before assuming the
feed is the bottleneck.

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
`rope` at `-O1` miscompiles). The rolled 128-block decode loop is always single-buffered
(`air.disable_ping_pong`, unconditional in `fused_decode.py`) to keep the KV ring aligned.
Enable turbo for ~50 tok/s:
`xrt-smi configure -d <BDF> --pmode turbo`.
