# Implementation Guide: LLAMA-3.2-1B on MLIR-AIR

This document explains how the LLAMA inference pipeline is implemented, how
kernels are compiled, and how multi-launch ELF stitching works.

---

## Architecture Overview

The pipeline compiles and runs LLAMA-3.2-1B inference entirely on the AMD NPU:

```
Python IR Builders          MLIR-AIR Compiler          NPU Hardware
──────────────────          ──────────────────          ────────────
build_module()          →   aircc (AIR passes)     →   Load ELF
  ↓ generates MLIR            ↓ dependency,            ↓
Multi-launch stitcher          placement,            xrt.run()
  ↓ combines N modules        air-to-aie               ↓
Module.parse()          →   aiecc (AIE passes)     →   Execute on
  ↓ validates MLIR             ↓ vectorize,            AIE tiles
XRTBackend.compile()           routing, link           (8×4 array)
  ↓ calls aircc                ↓
  ↓                         per-tile ELFs
  ↓                            ↓
kernel_cache/*.elf      ←   Package into
                            single ELF
```

Each transformer block operation (GEMM, RMSNorm, RoPE, etc.) is first built as
an independent MLIR module in Python, then multiple modules are stitched into a
single multi-launch ELF for reduced dispatch overhead.

---

## Kernel Compilation Pipeline

### Step 1: Python IR Generation

Each kernel has a Python builder function that generates MLIR using the AIR dialect API:

```python
# Example: RMSNorm from weighted_rms_norm/weighted_rms_norm.py
@module_builder
def build_module(seq_len, emb_dim, np_dtype, vector_size, herd_x):
    @FuncOp.from_py_func(input_type, weight_type, output_type)
    def weighted_rms_norm(x, weight, out):
        @herd(name="herd_0", sizes=[herd_x, 1], operands=[x, weight, out])
        def herd_body(tx, ty, sx, sy, h_x, h_w, h_out):
            # DMA input from DDR → L1
            # Compute RMSNorm
            # DMA output from L1 → DDR
```

This produces an MLIR module like:
```mlir
module {
  func.func @weighted_rms_norm(%arg0: memref<2048x2048xbf16>,
                                %arg1: memref<2048xbf16>,
                                %arg2: memref<2048x2048xbf16>) {
    air.herd @herd_0 tile(%tx, %ty) in (%sx=8, %sy=1) ... {
      // DMA + compute + DMA
    }
    return
  }
}
```

### Step 2: Multi-Launch Stitching

Multiple kernel modules are combined into a single MLIR function with multiple
`air.launch` operations. This is done via **text-based MLIR stitching** — no
MLIR API manipulation, just string processing:

```python
# In kernel_builder/stitching.py:

# 1. Extract function body (between func signature and return)
body = _extract_between_func_and_return(mlir_text)

# 2. Rename all SSA values with unique prefix to avoid collisions
body = _rename_all(body, prefix="q")  # %arg0 → %q_arg0, %0 → %q_n0

# 3. Remap func arg references to combined function's arg indices
body = _fix_launch_func_args(body, "q", {0: 2, 1: 3, 2: 4})

# 4. Assemble into combined module
combined = f"""
module {{
  func.func @combined(%arg0: ..., %arg1: ..., ...) {{
    {body_kernel_1}   // air.launch 1
    {body_kernel_2}   // air.launch 2
    ...
    return
  }}
}}
"""
module = Module.parse(combined)  # Validate MLIR
```

**Why text stitching?** The MLIR Python API doesn't support moving operations
between modules. Text manipulation is simpler and proven reliable for this use case.

### Step 3: Compilation (aircc → aiecc → ELF)

```python
backend = XRTBackend(output_format="elf", instance_name="kernel_name")
backend.compile(mlir_module)
# Internally:
#   1. Write MLIR to air.mlir
#   2. Run aircc: AIR passes (dependency, broadcast, placement, air-to-aie)
#   3. Run aiecc: AIE passes (vectorize, route, generate per-tile ELFs, package)
#   4. Output: kernel_name.elf
```

The `aircc` pipeline has ~47 passes including:
- `air-dependency`: analyze data dependencies between operations
- `air-broadcast-detection`: identify broadcast DMA patterns
- `air-place-herds`: map herds to physical tile positions
- `air-to-aie`: lower AIR dialect to AIE dialect
- `air-opt-shim-dma-bds`: optimize shim DMA buffer descriptors

### Step 4: Caching

Compiled ELFs are saved to `prefill_kernel_cache/` or `decode_kernel_cache/`
with a `manifest.json` mapping kernel names to ELF paths. Subsequent runs with
`--run-only` load from cache without recompilation.

---

## Multi-Launch Stitching Details

### The Stitching Utilities (`kernel_builder/stitching.py`)

| Function | Purpose |
|----------|---------|
| `_extract_between_func_and_return()` | Extract the body of a func (between signature and return) |
| `_extract_affine_maps()` | Extract `#map` declarations from module header |
| `_extract_private_funcs()` | Extract `func.func private` declarations (external kernels) |
| `_rename_all(text, prefix)` | Rename all SSA values (`%arg0`→`%q_arg0`), symbols (`@herd`→`@q_herd`), and maps (`#map0`→`#q_map0`) with prefix. Preserves external kernel function names. |
| `_rename_all_with_externs(text, prefix, extern_funcs)` | Like `_rename_all` but with configurable set of preserved external names |
| `_fix_launch_func_args(text, prefix, arg_map)` | Remap `air.launch args(=%q_arg0)` references to combined func's `%argN` |
| `_wrap_ir_in_launch(mlir_text)` | Wrap a bare `air.herd` in `air.launch { air.segment { ... } }` (required for multi-launch ELFs) |

### Example: How `rms_gemms_rope` Is Built

The 6-launch prefill kernel merges RMSNorm + Q/K/V GEMMs + RoPE Q + RoPE K:

```python
# In multi_launch_builder/rms_gemms_rope_multi.py:

# Build 6 sub-kernels independently
rms_ir  = _wrap_ir_in_launch(str(build_rms(seq_len, emb_dim, ...)))
q_ir    = str(_build_gemm_module(seq_len, emb_dim, emb_dim, ...))
k_ir    = str(_build_gemm_module(seq_len, emb_dim, kv_dim, ...))
v_ir    = str(_build_gemm_module(seq_len, emb_dim, kv_dim, ...))
rope_q  = str(_build_rope_2d(seq_len, emb_dim, head_dim, ...))
rope_k  = str(_build_rope_2d(seq_len, kv_dim, head_dim, ...))

# Stitch with arg mappings
stitch_specs = [
    (rms_ir,   "r",  {0:0, 1:1, 2:2}),      # x_in, norm_w, normed
    (q_ir,     "q",  {0:2, 1:3, 2:4}),       # normed→wq→q
    (k_ir,     "k",  {0:2, 1:5, 2:6}),       # normed→wk→k
    (v_ir,     "v",  {0:2, 1:7, 2:8}),       # normed→wv→v
    (rope_q,   "rq", {0:4, 1:9, 2:11}),      # q→lut_q→q_roped
    (rope_k,   "rk", {0:6, 1:10, 2:12}),     # k→lut_k→k_roped
]

# Each spec: (ir_text, prefix, {orig_arg: combined_arg})
# The arg_map connects sub-kernel outputs to subsequent inputs:
#   e.g., RMSNorm output (arg2=normed) → Q GEMM input (arg0→2)
```

The arg mapping creates the data flow between launches:
```
arg0 (x_in) ──→ L1 RMSNorm ──→ arg2 (normed) ──→ L2 Q GEMM ──→ arg4 (q)
                                     │                              │
                                     ├──→ L3 K GEMM ──→ arg6 (k)   │
                                     │                              │
                                     └──→ L4 V GEMM ──→ arg8 (v)   │
                                                                    │
arg9 (lut_q) ──────────────────→ L5 RoPE Q ←───────────────────────┘
arg10 (lut_k) ─────────────────→ L6 RoPE K ←── arg6 (k)
```

### Key Technique: `collapse_shape` for Type Aliasing

Some merged kernels need the same DDR buffer with different memref types.
For example, GEMM outputs `memref<2048x2048xbf16>` (2D) but RoPE reads
`memref<4194304xbf16>` (1D). These are the same bytes in DDR, just different views.

Solution: use `memref.collapse_shape` inside the `air.launch` body:

```python
@module_builder
def _build_rope_2d(outer_rows, outer_cols, embed_dim, ...):
    # Func arg is 2D (matches GEMM output type)
    @FuncOp.from_py_func(l3_2d_ty, l3_1d_ty, l3_2d_ty)
    def rope_2d(in_2d, lut_1d, out_2d):
        @launch(operands=[in_2d, lut_1d, out_2d])
        def rope_launch(l_in_2d, l_lut, l_out_2d):
            # Collapse 2D → 1D inside the launch (before segment)
            in_flat = collapse_shape(l3_1d_ty, l_in_2d, [[0, 1]])
            out_flat = collapse_shape(l3_1d_ty, l_out_2d, [[0, 1]])
            @segment(operands=[in_flat, l_lut, out_flat])
            def rope_seg(s_in, s_lut, s_out):
                # RoPE herd operates on 1D flat arrays
                ...
```

### Key Technique: External Kernel Rename (`-D` Preprocessor)

The decode `o_gemv_ffn` kernel merges K=2048 GEMVs with the K=8192 Down GEMV.
Both call the same C++ function `@matvec_vectorized_bf16_bf16`, but with different
memref type signatures — MLIR requires one declaration per name.

Solution: compile `mv.cc` twice with different `-D` defines:

```bash
# Standard GEMV (K=2048): original function name
clang++ -c mv.cc -o mv.o

# Down GEMV (K=8192): renamed function
clang++ -c mv.cc -o mv_k8192.o \
    -Dmatvec_vectorized_bf16_bf16=dg_matvec_vectorized_bf16_bf16 \
    -Dlinalg_fill_bf16=dg_linalg_fill_bf16
```

In the MLIR module, the Down GEMV's `@matvec` references are renamed during
stitching (by not preserving them in `extern_funcs`), and its `link_with`
attribute points to `"mv_k8192.o"` instead of `"mv.o"`.

### Key Technique: Half-Split RoPE Kernel

HuggingFace Llama uses **half-split** RoPE rotation: pairs `(x[i], x[i+32])` within
each head's 64 dimensions. The upstream `rope.cc` kernel uses a different
**interleaved** convention pairing adjacent elements `(x[2i], x[2i+1])`.

We provide our own `rope_halfsplit.cc` (`kernel_builder/rope_halfsplit.cc`) that
implements the half-split convention directly, matching HuggingFace exactly:

```
LUT layout:  [cos_0, cos_1, ..., cos_31, sin_0, sin_1, ..., sin_31]
Rotation:    out[i]      = x[i] * cos[i]      - x[i+32] * sin[i]
             out[i+32]   = x[i] * sin[i]      + x[i+32] * cos[i]
```

The kernel exports the same `@rope` function name and signature as upstream,
so no MLIR or multi-launch builder changes are needed. It is compiled to `rope.o`
in `external_kernels.py:compile_rope()`.

The NPU output is then gated against HuggingFace transformers in bf16
(`make verify` — see [`VERIFICATION.html`](detail/VERIFICATION.html)),
which exercises the same half-split RoPE convention end-to-end.

---

## Kernel Directory Map

### Sub-Kernel Builders (individual operations)

These generate MLIR for a single operation. They live in their respective
`programming_examples/` directories:

| Kernel | Builder Location | Output |
|--------|-----------------|--------|
| GEMM (prefill) | `matrix_multiplication/bf16/run.py` + `kernel_builder/gemm_builder.py` (transform IR) | `air.launch` with [8,4] herd |
| GEMV (decode) | `matrix_vector_multiplication/bf16/matvec.py` | `air.launch` with [8,1] herd |
| RMSNorm | `weighted_rms_norm/weighted_rms_norm.py` | bare `air.herd` (needs `_wrap_ir_in_launch`) |
| RoPE | `rope_lut/rope_lut.py` | bare `air.herd` (needs wrapping) |
| FlashAttention | `flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py` | `air.launch` with channels + cascade |
| SiLU×mul | `llama32_1b/ffn_swiglu/silu_and_mul.py` | bare `air.herd` (needs wrapping) |
| Eltwise Add | `eltwise_add/eltwise_add.py` | bare `air.herd` (needs wrapping) |

### Multi-Launch Builders (stitched kernels)

These combine multiple sub-kernels into single ELFs. They live in
`llama32_1b/multi_launch_builder/`:

| Builder | Launches | Sub-kernels Combined |
|---------|----------|---------------------|
| **Prefill** | | |
| `rms_gemms_rope_multi.py` | 6 | RMSNorm + Q/K/V GEMM + RoPE Q + RoPE K |
| `o_ffn_multi.py` | 8 | O GEMM + Add + RMSNorm + Gate/Up GEMM + SiLU + Down GEMM + Add |
| `lm_head_multi.py` | 8 | 8-partition GEMM (vocab projection) |
| **Decode** | | |
| `rms_gemv_rope_multi.py` | 6 | RMSNorm(1D) + Q/K/V GEMV + RoPE Q + RoPE K |
| `o_gemv_ffn_multi.py` | 8 | O GEMV + Add + RMSNorm(1D) + Gate/Up GEMV + SiLU + Down GEMV + Add |
| `lm_head_gemv_multi.py` | 8 | 8-partition GEMV (vocab projection) |

### External C++ Kernels

These are compiled from C++ source by `kernel_builder/external_kernels.py`:

| .o File | Source | Function | Used By |
|---------|--------|----------|---------|
| `rope.o` | `aie_kernels/aie2p/rope.cc` | `@rope` | RoPE launches |
| `silu_and_mul.o` | `ffn_swiglu/silu_and_mul.cc` | `@silu_and_mul_bf16` | SiLU×mul launch |
| `attn_npu2.o` | `flash_attention/.../attn_npu2.cc` | 16 attention functions | FlashAttention |
| `mv.o` | `matrix_vector_multiplication/bf16/mv.cc` | `@matvec_vectorized_bf16_bf16` | K=2048 GEMVs |
| `mv_k8192.o` | same `mv.cc` with `-D` renames | `@dg_matvec_vectorized_bf16_bf16` | K=8192 Down GEMV |

All `.o` files are compiled fresh from source via `compile_all_external_kernels()`.
No pre-compiled artifacts are copied.

### Shared Infrastructure (`kernel_builder/`)

| Module | Contents |
|--------|----------|
| `cache.py` | `KernelCache` (compile/load/run + per-layer BO management), `Profiler`, `prepare_air_project()` |
| `backend_presets.py` | All `*_BACKEND` kwarg dicts (`SIMPLE_BACKEND`, `RMS_GEMMS_ROPE_BACKEND`, `O_FFN_BACKEND`, `LM_HEAD_BACKEND`, `GEMV_K2048_BACKEND`, `RGR_BACKEND`, `OGF_BACKEND`, `LM_GEMV_BACKEND`) |
| `gemm_builder.py` | `_build_gemm_module()` + GEMM transform IR (vectorization) |
| `stitching.py` | MLIR text manipulation functions for multi-launch stitching (`_extract_*`, `_rename_all`, `_rename_all_with_externs`, `_wrap_ir_in_launch`, ...) |
| `external_kernels.py` | `compile_all_external_kernels()` and per-kernel compile functions |

---

## Runtime Execution

### KernelCache: Compile, Cache, Load, Run

The `KernelCache` class manages the full lifecycle:

```python
cache = KernelCache("prefill_kernel_cache")

# Compile: MLIR → aircc → aiecc → .elf (saved to disk)
cache.compile_and_cache("rms_gemms_rope", mlir_module, backend_kwargs)

# Load: .elf → XRT context + kernel handle (cached in memory)
# Run: write BOs → xrt.run() → read BOs (cached BOs reused)
results = cache.load_and_run("rms_gemms_rope", backend_kwargs,
    *inputs,
    output_indices=[8, 11, 12],          # which BOs to read back
    static_input_indices={1, 3, 5, 7},   # weights: skip write after first call
    intermediate_indices={2, 4, 6, ...}, # kernel-overwritten: skip write
    bo_key="rms_gemms_rope_L0",          # per-layer BO isolation
)
```

### Weight Pre-loading

Before timed inference, all weights are written to per-layer BOs:

```python
# In prepare_runtime():
preload_prefill_weights(weights, config, cache, seq_len, rope_lut)
#   For each of 16 layers:
#     Allocate BOs for rms_gemms_rope_L{i} (13 BOs)
#     Write weights: wq, wk, wv, norm_w, lut_q, lut_k
#     Allocate BOs for o_ffn_L{i} (15 BOs)
#     Write weights: wo, ffn_norm_w, w_gate, w_up, w_down
```

During inference, `static_input_indices` skips weight writes (only activations
are written per call). This reduces BO write overhead from 14% to 4% of total time.

---

## Compiler Limitations Discovered

| Limitation | Impact | Workaround |
|-----------|--------|-----------|
| FlashAttention channels block multi-launch merging | Prefill stuck at 3 invocations/layer | Keep FlashAttention as separate kernel |
| `omit_pingpong` is global per-module | ~19ms/token decode penalty for K=2048 GEMVs | Accept tradeoff (overall still faster) |
| `air-opt-shim-dma-bds` scales super-linearly with FlashAttention | 9+ launches with FlashAttention: >10min compile | Don't merge FlashAttention |
| External kernel type mismatch (different K dims) | Can't merge K=2048 and K=8192 GEMVs naively | `-D` preprocessor rename of function symbols |
| Stack overflow for 9+ launches | `AIRDependencyCanonicalize` deep recursion | Not an issue for current 6-8 launch modules (no `ulimit` needed) |
