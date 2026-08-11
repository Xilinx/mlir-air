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
Q4_0`. This mirrors FastFlowLM, whose own Qwen design sets `#define Q4_0` in
`models/qwen2_3b.h` while its Llama and Gemma designs leave it undefined.

The weight *bundles* are uniform: every FastFlowLM `model.q4nx` is the same
per-block affine encoding regardless of model. That is why `llms/qwen25_3b_q4`
quantizes the fp checkpoint directly rather than reading a bundle -- an affine
bundle re-quantized into the symmetric device form would quantize twice.

## Reproducibility (toolchain)

`make compile-decode` merges an inline attention kernel into the core via an **external
`llvm-link`** (resolved from `PATH`) whose **LLVM major must equal Peano's** (21 today).
A newer (≥ 23) one rewrites the `llvm.lifetime` intrinsic to the no-size form, which
Peano `opt` then rejects (`Broken module found`); an older one (e.g. the system LLVM 20)
links without complaint and *silently miscompiles* the attention kernels — the decode
runs at full speed and emits fluent garbage. Keep any mismatched LLVM's `bin` off `PATH`
for this build; the Makefile preflight aborts early on either. Verify:

```bash
which llvm-link && llvm-link --version
$PEANO_INSTALL_DIR/bin/clang --version   # majors must match
```

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
