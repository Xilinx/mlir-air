<!---//===- RMSNorm_bf16.md -----------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# Weighted RMSNorm (BF16) — Kernel Detail

> Weighted RMS normalization `y = x / sqrt(mean(x²) + eps) · weight`, per row, for the pre-attention / pre-FFN norms of a decoder-only LLM. BF16 inputs/output, **FP32 reduction** (GPU/HF standard).
> Shapes are written **`M×N`**: input `x[M, N]`, weight `[N]`, output `y[M, N]` (M = rows / seq, N = embedding dim, the reduction axis).
>
> Companion: [`../supported_kernels.md`](../supported_kernels.md) · [`../README.md`](../README.md)
> **Scope: NPU2 (Strix / AIE2P) only.** Measured on real NPU2 (RyzenAI-npu4), June 2026. Reproduce commands in "How to reproduce" below.

---

## Builder

```
programming_examples/weighted_rms_norm/weighted_rms_norm.py
  build_module(M, N, np_dtype, vector_size=16, herd_x=1)
```

Driven by `weighted_rms_norm.py`'s CLI; the example also has a `Makefile`. Single code-generation path: the compute is direct-codegen MLIR (vectorized `vector.transfer_read/write` + `arith` + `math.rsqrt`), no external `.cc`. Two herd modes share the same math:

| Mode | `herd_x` | Layout | Note |
|---|---|---|---|
| single-tile | 1 | `[1,1]` herd, one tile loops all M rows | baseline |
| multi-tile | 8 | `[herd_x,1]` herd, each column owns `M/herd_x` rows | full-chip-width, recommended |

The herd is **1-D** (`sizes=[herd_x, 1]`) — RMSNorm is row-independent, so the M rows are split across `herd_x` AIE columns. The weight `[N]` is broadcast to every column.

---

## Numerical datapath (what "BF16 RMSNorm" means here)

```
x[m,:] bf16 → sum(x²) accumulated in FP32 → mean = sum/N (f32) → rstd = rsqrt(mean + eps) in FP32 → y = x · rstd · weight → cast to bf16
```

This follows the **GPU / HuggingFace standard** (PyTorch `rms_norm_composite`, HF `LlamaRMSNorm`, vLLM `RMSNorm`): the bf16 input is upcast and **the whole reduction runs in FP32**, casting back to bf16 only for the output.

- **The reduction `sum(x²)` is accumulated in FP32.** Each `vector_size`-wide block is squared, upcast to f32, and added into an f32 accumulator buffer; the running sum stays f32 across the whole N reduction, so it does not lose low-order bits — matching the GPU/HF standard of accumulating the reduction in f32.
- **`mean`, `rstd = rsqrt(mean + eps)` are computed in FP32.** `eps = 1e-5`.
- **The output scaling `x · rstd · weight` is done per-element and cast to bf16 at the write.** The per-element multiply uses the bf16 vector path (the aie vector unit does bf16 elementwise; f32 *vector* elementwise is not legalized here), so the final scale + bf16 output rounding is the only remaining quantization step — the same single-epilogue-rounding a standard GPU RMSNorm has.

The one deviation from a pure-f32 GPU kernel: the final `x·rstd·weight` elementwise multiply runs in bf16 rather than f32 (an aie vector-unit constraint), contributing only the standard bf16 output-rounding error. The accuracy-critical part — the reduction — is fully f32.

---

## Numerical accuracy

Verified element-wise over the full output against an f32 reference (the same methodology as GEMM/GEMV):

| Metric (M=2048, N=2048, randn inputs) | Measured |
|---|---|
| `mean_rel_L1 = mean|y−ref| / mean|ref|` | **4.2e-3** |
| `rel_err max` | 2.3e-2 |
| `abs_err max` | 1.9e-1 |

- **`mean_rel_L1 = 4.2e-3` is in line with this registry's GEMM (~9.3e-3)** — the FP32 reduction puts RMSNorm in the standard bf16 tier.
- **`rel_err max = 2.3e-2`** comes from a handful of elements (≈14 of 4.2M) where the bf16-rounded output differs from the bf16-rounded reference by 2–3 bf16 ULP — e.g. `9.75` vs `9.5625` at a large-magnitude element (one bf16 ULP at that scale is `0.0625`). This is the bf16 *output* granularity, not a reduction error, which is why `atol = 5e-2` (covering ~3 ULP at the largest magnitudes) is needed alongside the standard `rtol = 1.6e-2`.
- **Accuracy is independent of `herd_x`** — bit-identical error at `herd_x` ∈ {1,2,4,8} — so the reduction precision, not the tiling, sets the number.

---

## Tunable parameters

RMSNorm is **memory-bound** (it streams the whole `M×N` matrix in and out for an O(M·N) elementwise op), so the only performance knob that matters is `herd_x` — and it should always be 8.

| Knob | Recommended | Hard constraint | Note |
|---|---|---|---|
| `herd_x` | **8 (fixed)** | `M % herd_x == 0` | AIE columns; **always 8** = full chip width. Near-linear speedup with column count (see below); not a tuning target |
| `vector_size` | 16 | `N % vector_size == 0` | SIMD width of the inner loops; 16 is the AIE2 bf16 vector width |

`herd_x = 8` is the right choice for any reasonably tall input; `vector_size = 16` is the natural bf16 lane count and does not need tuning. There is no `tile_m`/`m_input`-style knob — each column simply loops over its row chunk.

> **`Makefile` default**: `make run` defaults to `HERD_X=8` (multi-tile); `make run_single_tile` forces `herd_x=1` for the baseline. The builder default is `herd_x=1`.

---

## Tolerances & reference

The example verifies correctness element-wise over the **full output** against an f32 reference: every element must pass `np.isclose(|a−e| ≤ atol + rtol·|e|)`.

| Output dtype | rtol | atol |
|---|---|---|
| bf16 | 1.6e-2 | 5e-2 |

- **Reference** = CPU FP32 RMSNorm (`x.astype(f32)`, mean-of-squares in f32, `rsqrt`, `× weight` in f32, cast to bf16), matching PyTorch `rms_norm_composite` / HF `LlamaRMSNorm`. Inputs are `randn` (seed 0).
- `rtol = 1.6e-2` is PyTorch / vLLM's canonical bf16 tolerance (`torch.testing` default; PyTorch's own `nn.functional.rms_norm` OpInfo test uses the default bf16 rtol). `atol = 5e-2` covers the few large-magnitude elements where bf16 *output* rounding lands 2–3 ULP off (one bf16 ULP at magnitude ~10 is `0.0625`); it is not a reduction-precision relaxation. With the FP32 reduction, `mean_rel_L1 = 4.2e-3` sits well inside `rtol`.

---

## Tested shapes

The RMSNorm shape used by llama-3.2-1B **prefill** (`M = seq_len = 2048`, `N = emb_dim = 2048`), which directly uses this 2-D `build_module`. (Decode uses a 1-D `M=1` wrapper around the same math — see "Used by" note.)

| (M, N) | herd_x | latency | bandwidth | mean_rel_L1 | rel_err max | abs_err max | Used by | Status |
|---|---|---|---|---|---|---|---|---|
| 2048×2048 | 8 | 911 µs | 18.4 GB/s | 4.2e-3 | 2.3e-2 | 1.9e-1 | llama-3.2-1B + Qwen3-1.7B + Qwen2.5-3B prefill RMSNorm | ✅ |
| 2048×1024 | 8 | 407 µs | 20.6 GB/s | 4.3e-3 | 2.3e-2 | 1.25e-1 | Qwen3-0.6B prefill RMSNorm | ✅ |
| 2048×128 | 8 | 155 µs | 6.8 GB/s | 4.6e-3 | 2.3e-2 | 1.25e-1 | Qwen3-0.6B + Qwen3-1.7B QK-norm (per-head, N=head_dim) | ✅ |
| 2048×896 | 8 | 398 µs | 18.4 GB/s | 4.2e-3 | 2.3e-2 | 1.25e-1 | Qwen2.5-0.5B prefill RMSNorm | ✅ |
| 2048×1536 | 8 | 570 µs | 22.1 GB/s | 4.3e-3 | 2.3e-2 | 1.25e-1 | Qwen2.5-1.5B prefill RMSNorm | ✅ |
| 2048×2560 | 8 | 867 µs | 24.2 GB/s | 4.2e-3 | 2.3e-2 | 1.25e-1 | Qwen3-4B prefill RMSNorm (emb=2560) | ✅ |
| 2048×3072 | 8 | 1012 µs | 24.9 GB/s | 4.2e-3 | 2.3e-2 | 1.25e-1 | Llama-3.2-3B prefill RMSNorm (emb=3072) | ✅ |

> **Qwen2.5-0.5B (2048×896)** is the per-layer RMSNorm at emb=896. Qwen2.5 has no QK-norm (unlike Qwen3). Verified PASS at 4.2e-3.

> **Qwen2.5-1.5B (2048×1536)** is the per-layer RMSNorm at emb=1536. No QK-norm. Verified PASS at 4.3e-3.

> **Qwen3-0.6B QK-norm (2048×128)** is Qwen3's per-head q_norm/k_norm: weighted RMSNorm with the reduction axis = `head_dim = 128` (the smallest `N` in the registry). Verified PASS at 4.6e-3, confirming the kernel handles a 128-wide reduction. (Harness `EPS = 1e-5`; Qwen3 uses `eps = 1e-6` — negligible vs the bf16 datapath error. Run via `python3 weighted_rms_norm.py --M 2048 --N <N> --herd-x 8`; the Makefile `run` target does not forward `--M/--N`.)

> **What "Used by" means here.** This kernel (`weighted_rms_norm.py`, 2-D `build_module`) is the exact kernel llama-3.2-1B **prefill** uses for the per-layer RMSNorm (in the `rms_gemms_rope` ELF, `herd_x=8`). **Decode** uses a 1-D (`M=1`) wrapper (`_build_rms_1d` in `rms_gemv_rope_multi.py`) built around the same vectorized math — same datapath, same FP32 reduction, just a single-row layout.

**Reading the table**:
- **Memory-bound**: latency is gated by DMA, not compute. At `herd_x=8` the kernel moves ~16.8 MB (in + out matrix + weight) in 911 µs ≈ 18 GB/s. Throughput is reported as bandwidth, not GFLOP/s (the op is O(M·N) elementwise, not a matmul).
- **Accuracy in the bf16 standard tier** (`mean_rel_L1 = 4.2e-3`) thanks to the FP32 reduction — see [Numerical accuracy](#numerical-accuracy).
- **Accuracy is independent of `herd_x`** — set only by the reduction precision; `herd_x` is a pure performance knob.

---

## herd_x choice vs performance

RMSNorm is memory-bound, so spreading the M rows across more AIE columns scales throughput near-linearly. Full sweep of the legal `herd_x` (must divide M=2048), all on the single direct-codegen path:

| herd_x | latency | speedup vs herd_x=1 |
|---|---|---|
| 1 | 6801 µs | 1.0× |
| 2 | 3426 µs | 2.0× |
| 4 | 1788 µs | 3.8× |
| 8 | **911 µs** | **7.5×** |

`herd_x = 8` (full NPU2 chip width) is the clear best — near-linear scaling because each column independently streams its row chunk through its own DMA. There is no reason to use fewer columns. Accuracy is identical (bit-for-bit) across all four — `herd_x` is purely a performance knob. The FP32 reduction (f32 accumulator buffer + per-block upcast) costs essentially nothing here — the kernel is memory-bound, so the extra arithmetic is hidden behind DMA.

---

## How to reproduce (correctness + performance, one command)

`weighted_rms_norm.py` (compile-and-run mode, the default) does **both** in a single invocation, via `XRTRunner`:
- **correctness** — full-output element-wise check against the f32 reference; prints `[precision] mean_rel_L1=... | rel_err max=... | abs_err max=... | rtol=... atol=...` and `PASS!` / `failed.`
- **performance** — add `--perf-iters N` to time the kernel over `N` iterations (after 10 warmup runs, kernel-only) and print `Latency (us): ...` (memory-bound op, so latency/bandwidth rather than GFLOP/s).

The tested-shapes row reproduces with:

```bash
cd programming_examples/weighted_rms_norm

# multi-tile (herd_x=8, recommended) — compiles and runs correctness + perf
make run HERD_X=8 PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR

# to also print latency, run the script directly with --perf-iters:
mkdir -p build_peano && cd build_peano
python3 ../weighted_rms_norm.py --M 2048 --N 2048 --herd-x 8 --perf-iters 20
```

For another `herd_x`, change `--herd-x` (must divide M). The single-tile baseline is `make run_single_tile` (`herd_x=1`).

Notes:
- If the NPU is shared with other jobs, serialize on-device runs (e.g. with `flock`) so timing measurements aren't perturbed.
