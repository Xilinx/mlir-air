<!---//===- LayerNorm_bf16.md ---------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# Affine LayerNorm (BF16) — Kernel Detail

> Affine layer normalization `y = (x − mean) / sqrt(var + eps) · weight + bias`, per row, for the vision-encoder (SigLIP) norms of a vision-language model. BF16 inputs/output, **FP32 reduction** (GPU/HF standard).
> Shapes are written **`M×N`**: input `x[M, N]`, weight `[N]`, bias `[N]`, output `y[M, N]` (M = rows / patches, N = hidden dim, the reduction axis).
>
> Companion: [`../supported_kernels.md`](../supported_kernels.md) · [`../README.md`](../README.md)
> **Scope: NPU2 (Strix / AIE2P) only.** Measured on real NPU2 (RyzenAI-npu4), July 2026. Reproduce commands in "How to reproduce" below.

---

## What LayerNorm is (and how it differs from RMSNorm)

RMSNorm and LayerNorm share a datapath tier but are **different ops**:

| | RMSNorm | **LayerNorm (this kernel)** |
|---|---|---|
| Centering | none | **subtracts the mean** `(x − mean)` |
| Reductions | 1 (`sum(x²)`) | **2** (`sum(x)` for mean, then `sum((x−mean)²)` for var) |
| Affine | weight only (γ) | **weight *and* bias** (γ and β) |
| Formula | `x / sqrt(mean(x²)+eps) · w` | `(x−mean) / sqrt(var+eps) · w + b` |

Both accumulate their reductions in FP32 and run the elementwise epilogue in the bf16 vector unit. LayerNorm has one more reduction pass and one more affine term (the bias add) than RMSNorm.

---

## Builder

```
programming_examples/layer_norm/layer_norm.py
  build_module(M, N, np_dtype, vector_size=16, herd_x=1)
```

Driven by `layer_norm.py`'s CLI; the example also has a `Makefile`. Single code-generation path: the compute is direct-codegen MLIR (vectorized `vector.transfer_read/write` + `arith` + `math.rsqrt`), no external `.cc`. Two herd modes share the same math:

| Mode | `herd_x` | Layout | Note |
|---|---|---|---|
| single-tile | 1 | `[1,1]` herd, one tile loops all M rows | baseline |
| multi-tile | 8 | `[herd_x,1]` herd, each column owns `M/herd_x` rows | full-chip-width, recommended |

The herd is **1-D** (`sizes=[herd_x, 1]`) — LayerNorm is row-independent, so the M rows are split across `herd_x` AIE columns. The affine params `weight[N]` and `bias[N]` are broadcast to every column.

**Packed weight||bias param buffer.** weight (γ) and bias (β) are concatenated host-side into one flat `[2N]` buffer (`[0:N]` = weight, `[N:2N]` = bias) and DMA'd in a **single** L3→L1 transfer. This keeps the kernel at **3 L3↔L1 DMA streams per tile** (packed-param in, row in, row out). Four separate streams (in + weight + bias + out) overflow the AIE per-tile routing capacity — `aie.connect ... targets same dst as another connect op` at compile — which is why the params are packed. (Same 3-DMA ceiling the registry's elementwise kernels hit.)

---

## Numerical datapath (what "BF16 LayerNorm" means here)

```
x[m,:] bf16 → sum(x) in FP32 → mean = sum/N (f32)
            → sum((x−mean)²) in FP32 → var = sum/N (f32)
            → rstd = rsqrt(var + eps) in FP32
            → y = (x − mean) · rstd · weight + bias → cast to bf16
```

This follows the **GPU / HuggingFace standard** (PyTorch / HF `nn.LayerNorm`): the bf16 input is upcast and **both reductions run in FP32**, casting back to bf16 only for the output.

- **Both reductions are accumulated in FP32.** The `sum(x)` (mean) and `sum((x−mean)²)` (variance) each accumulate into an f32 buffer — each `vector_size`-wide block is upcast to f32 and added into an f32 running sum, so the reductions over N do not lose low-order bits, matching the GPU/HF standard.
- **`mean`, `var`, `rstd = rsqrt(var + eps)` are computed in FP32.** `eps = 1e-5`.
- **The affine epilogue `(x−mean)·rstd·weight + bias` is done per-element and cast to bf16 at the write.** The per-element ops use the bf16 vector path (the aie vector unit does bf16 elementwise; f32 *vector* elementwise is not legalized here). This is one more bf16 rounding step than RMSNorm — the trailing bias add — but the accuracy-critical part (the reductions) is fully f32.

The deviations from a pure-f32 GPU kernel: the mean subtraction, the `·rstd·weight`, and the `+bias` elementwise ops run in bf16 rather than f32 (an aie vector-unit constraint), contributing only the standard bf16 output-rounding error.

---

## Numerical accuracy

Verified element-wise over the full output against an f32 reference (the same methodology as GEMM/GEMV/RMSNorm):

| Metric (M=1024, N=768, randn inputs) | Measured |
|---|---|
| `mean_rel_L1 = mean|y−ref| / mean|ref|` | **4.4e-3** |
| `abs_err max` | 1.9e-1 |

- **`mean_rel_L1 = 4.4e-3` is in line with RMSNorm (4.2e-3) and this registry's GEMM (~9.3e-3)** — the FP32 reductions put LayerNorm in the standard bf16 tier.
- **`abs_err max = 1.9e-1`** comes from a handful of large-magnitude elements where the bf16-rounded output differs from the bf16-rounded reference by a few bf16 ULP (one bf16 ULP at magnitude ~10 is `0.0625`). This is the bf16 *output* granularity, not a reduction error, which is why `atol = 6e-2` is needed alongside the standard `rtol = 1.6e-2`.
- **Do not use a `rel_err max` gate.** Because LayerNorm has an additive bias, some reference outputs are legitimately near zero, and `|y−ref| / |ref|` explodes on those denominators (the harness prints `rel_err max ≈ 1.3e3` for exactly this reason). The meaningful metrics are `mean_rel_L1` and `abs_err max` against the element-wise `np.isclose(|a−e| ≤ atol + rtol·|e|)` gate — never a relative-only or cosine check.
- **Accuracy is independent of `herd_x`** — bit-identical error at `herd_x` ∈ {1,2,4,8} — so the reduction precision, not the tiling, sets the number.

---

## Tunable parameters

LayerNorm is **memory-bound** (it streams the whole `M×N` matrix in and out for an O(M·N) elementwise op), so the only performance knob that matters is `herd_x` — and it should always be 8.

| Knob | Recommended | Hard constraint | Note |
|---|---|---|---|
| `herd_x` | **8 (fixed)** | `M % herd_x == 0` | AIE columns; **always 8** = full chip width. Near-linear speedup with column count (see below); not a tuning target |
| `vector_size` | 16 | `N % vector_size == 0` | SIMD width of the inner loops; 16 is the AIE2/AIE2P bf16 vector width |

`herd_x = 8` is the right choice for any reasonably tall input; `vector_size = 16` is the natural bf16 lane count and does not need tuning. There is no `tile_m`-style knob — each column simply loops over its row chunk.

> **`Makefile` default**: `make run AIE_TARGET=aie2p` defaults to `HERD_X=8` (multi-tile) and `M=1024 N=768`; `make run_single_tile` forces `herd_x=1` for the baseline. The builder default is `herd_x=1`. **Always pass `AIE_TARGET=aie2p`** — the default `aie2` (NPU1) silently produces zeros on this NPU2 machine.

---

## Tolerances & reference

The example verifies correctness element-wise over the **full output** against an f32 reference: every element must pass `np.isclose(|a−e| ≤ atol + rtol·|e|)`.

| Output dtype | rtol | atol |
|---|---|---|
| bf16 | 1.6e-2 | 6e-2 |

- **Reference** = CPU FP32 LayerNorm (`x.astype(f32)`, mean in f32, var-of-centered in f32, `rsqrt`, `·weight + bias` in f32, cast to bf16), matching PyTorch / HF `nn.LayerNorm`. Inputs are `randn` (seed 0); weight and bias are `randn[N]`.
- `rtol = 1.6e-2` is PyTorch / vLLM's canonical bf16 tolerance. `atol = 6e-2` covers the few large-magnitude elements where bf16 *output* rounding lands ~4 ULP off — one more ULP than RMSNorm's `5e-2` because LayerNorm has the extra trailing bias-add rounding; it is not a reduction-precision relaxation. With the FP32 reductions, `mean_rel_L1 = 4.4e-3` sits well inside `rtol`.

---

## Tested shapes

The LayerNorm shape used by the SmolVLA vision encoder (SigLIP): `M = num_patches = 1024`, `N = hidden = 768`. Each vision transformer layer uses two LayerNorms (`layer_norm1`, `layer_norm2`) plus a final `post_layernorm`, all at this shape and all affine.

| (M, N) | herd_x | latency | bandwidth | mean_rel_L1 | abs_err max | Used by | Status |
|---|---|---|---|---|---|---|---|
| 1024×768 | 8 | 324 µs | 9.7 GB/s | 4.4e-3 | 1.9e-1 | SmolVLA vision (SigLIP) layer_norm1/2 + post_layernorm (hidden=768) | ✅ |

**Reading the table**:
- **Memory-bound**: latency is gated by DMA, not compute. At `herd_x=8` the kernel moves ~3.15 MB (in + out matrix + packed weight||bias) in 324 µs ≈ 9.7 GB/s. Throughput is reported as bandwidth, not GFLOP/s (the op is O(M·N) elementwise). The bandwidth is below the taller RMSNorm rows because M=1024 (half the 2048 rows) is closer to the ~80–150 µs launch/DMA-setup floor — the kernel is small, not slower per byte.
- **Accuracy in the bf16 standard tier** (`mean_rel_L1 = 4.4e-3`) thanks to the FP32 reductions — see [Numerical accuracy](#numerical-accuracy).
- **Accuracy is independent of `herd_x`** — set only by the reduction precision; `herd_x` is a pure performance knob.

---

## herd_x choice vs performance

LayerNorm is memory-bound, so spreading the M rows across more AIE columns scales throughput near-linearly. Full sweep of the legal `herd_x` (must divide M=1024), all on the single direct-codegen path, at 1024×768:

| herd_x | latency | speedup vs herd_x=1 |
|---|---|---|
| 1 | 1974 µs | 1.0× |
| 2 | 1007 µs | 2.0× |
| 4 | 555 µs | 3.6× |
| 8 | **324 µs** | **6.1×** |

`herd_x = 8` (full NPU2 chip width) is the clear best — near-linear scaling because each column independently streams its row chunk through its own DMA. (The 8-column speedup is 6.1× rather than the RMSNorm's 7.5× because at only M=1024 the fixed launch/DMA-setup floor is a larger fraction of the 324 µs total.) Accuracy is identical (bit-for-bit) across all four — `herd_x` is purely a performance knob. The FP32 reductions cost essentially nothing here — the kernel is memory-bound, so the extra arithmetic is hidden behind DMA. Latencies are the median of 3 runs.

---

## How to reproduce (correctness + performance, one command)

`layer_norm.py` (compile-and-run mode, the default) does **both** in a single invocation, via `XRTRunner`:
- **correctness** — full-output element-wise check against the f32 reference; prints `[precision] mean_rel_L1=... | rel_err max=... | abs_err max=... | rtol=... atol=...` and `PASS!` / `failed.`
- **performance** — add `--perf-iters N` to time the kernel over `N` iterations (after 10 warmup runs, kernel-only) and print `Latency (us): ...` (memory-bound op, so latency/bandwidth rather than GFLOP/s).

The tested-shapes row reproduces with:

```bash
cd programming_examples/layer_norm

# multi-tile (herd_x=8, recommended) — compiles and runs correctness + perf
make run AIE_TARGET=aie2p PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR

# to also print latency, run the script directly with --perf-iters:
mkdir -p build_peano && cd build_peano
python3 ../layer_norm.py --M 1024 --N 768 --herd-x 8 --perf-iters 20
```

For another `herd_x`, change `--herd-x` (must divide M). The single-tile baseline is `make run_single_tile` (`herd_x=1`).

Notes:
- **Always compile for `aie2p` (NPU2).** The default `aie2` (NPU1) silently produces all-zero output on this NPU2 machine.
- If the NPU is shared with other jobs, serialize on-device runs (e.g. with `flock -x -w 1800 /tmp/mlir-air-npu.lock`) so timing measurements aren't perturbed.
