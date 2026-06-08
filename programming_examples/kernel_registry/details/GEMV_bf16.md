<!---//===- GEMV_bf16.md --------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# GEMV (BF16) — Kernel Detail

> Matrix–vector product `C = A @ B` for the **decode-time** (batch = 1) weight projections of a decoder-only LLM. BF16 inputs, FP32 accumulation, BF16 output. In llama-3.2-1B decode this is the exact kernel for **Q / K / V projections and the LM-head**; the O / Gate / Up / Down projections use *fused* cascade variants (separate registry entries) — see [Tested shapes](#tested-shapes).
> Shapes are written **`M×K`**: `C[M] = A[M,K] @ B[K]` (M = output size / matrix rows, K = reduction / vector length).
>
> Companion: [`../supported_kernels.md`](../supported_kernels.md) · [`../README.md`](../README.md)
> **Scope: NPU2 (Strix / AIE2P) only.** All numbers, tile configs, and tolerances here are for the aie2p target, measured on real NPU2 (RyzenAI-npu4), June 2026. Reproduce commands in "How to reproduce" below.

---

## Builder

```
programming_examples/matrix_vector_multiplication/bf16/matvec.py
  build_module(m, k,
               tile_m, m_input, herd_m,
               np_dtype_in, np_dtype_out)
```

Driven by `matvec.py`'s CLI; the example also has a `Makefile`. Unlike GEMM, GEMV has a **single code-generation path**: the compute kernel is a hand-written vector microkernel `mv.cc` → `mv.o`, linked into the herd via `link_with`. There is no direct-codegen variant and the output dtype is always **bf16**.

The herd is **1-D** — `sizes=[herd_m, 1]` — because the output is a length-`M` vector, not a 2-D tile. The `herd_m` AIE columns each own an independent chunk of the `M` output rows; there is no `herd_n` (N = 1).

---

## Numerical datapath (what "BF16 GEMV" means here)

```
A,B stored bf16 → per-row dot product, FP32 vector accumulate → cast to bf16 once at the end
```

- **Accumulation is FP32.** Each output row is `sum_k A[m,k]·B[k]` computed with `aie::reduce_add` over an FP32 accumulator (`mv.cc`: `reduce_add(acc.to_vector<float>())`), then cast to bf16 a single time per row.
- **No BFP16-emulated MMA.** This is the key difference from BF16 GEMM. GEMM routes through the 8×8×8 matrix unit, which on aie2p casts operands to block-floating-point `v64bfp16ebs8` (8 elements share one exponent) and adds a block-quantization error. GEMV uses a plain FP32 vector MAC, so there is **no block quantization** — the on-device FP32 partial sums match a CPU FP32 reference essentially bit-for-bit, and the only error is the final per-row bf16 rounding (which the f32 reference incurs identically when cast to bf16). Measured `mean_rel_L1` is `0`–`2.7e-8` (see [tested shapes](#tested-shapes)), i.e. effectively exact.

---

## Tunable parameters

**`herd_m` is effectively not a tuning target — use 8 (the full NPU2 chip width) whenever the shape allows.** GEMV is memory-bound, so spreading the work across all 8 AIE columns is what gets the DMA bandwidth; 8 columns is ≈ 2.6× faster than 4. The only reason to use fewer is the legality constraint `M % (tile_m × herd_m) == 0` (and `M ≥ tile_m × herd_m`): for the llama-3.2-1B decode shapes (M ∈ {512, 2048, 8192, 16384}) `herd_m = 8` is always legal, but a model with an M not divisible by `8 × tile_m` would need a smaller `herd_m`. Otherwise treat `herd_m = 8` as fixed and only tune `tile_m` / `m_input` below.

The two genuine knobs to set when deploying a **new** model are `tile_m` and `m_input`. The **Recommended** column is what the tile sweep found best across the llama-3.2-1B GEMV shapes; it is **not** the `Makefile`'s default (`TILE_M=4 M_INPUT=1 HERD_M=4`) — see the note below.

| Knob | Recommended | Hard constraint | Note |
|---|---|---|---|
| `herd_m` | **8 when legal** | `M % (tile_m × herd_m) == 0`; `M ≥ tile_m × herd_m` | number of AIE columns; 8 = full chip width. Use the largest legal value (8 for all llama decode shapes) — fewer columns leaves bandwidth on the table (8 ≈ 2.6× over 4) |
| `tile_m` | largest legal | `M % (tile_m × herd_m) == 0`; `tile_m × 2` byte-aligned ⇒ `tile_m` even | output rows per column per launch; bounded by the L2 budget below |
| `m_input` | `= tile_m` | `tile_m % m_input == 0` | rows per kernel call; larger = fewer calls, less overhead (≈ 1.2× from `m_input=1` → `tile_m`) |

`K` must be a multiple of 64 (vector width). The best config for nearly every shape is **`herd_m=8` (when legal), `tile_m` = the largest the L2 budget allows, `m_input = tile_m`**.

> **`Makefile` default ≠ Recommended.** The example's `Makefile` defaults to `TILE_M=4 M_INPUT=1 HERD_M=4` (half the columns, smallest call granularity) — a conservative config that is ~2–3× slower than the recommended `herd_m=8` settings. Pass the recommended values explicitly for performance.

**L2 budget** (512 KiB / MemTile): `herd_m · tile_m · K · 2` bytes (the A panel) `+ herd_m · tile_m · 2` bytes (the C panel) must be `≤ 524288`. This is the binding constraint on `tile_m`: at `K=2048, herd_m=8` it caps `tile_m ≤ 8`; at `K=8192, herd_m=8` it caps `tile_m ≤ 2`. The builder asserts this — an over-budget config raises a clear error rather than silently corrupting.

⚠️ **4-byte DMA alignment**: `tile_m = 1` fails (`aie.dma_bd` transfer length must be a multiple of 4 bytes; one bf16 row = 2 bytes). Use `tile_m ≥ 2`.

---

## Tolerances & reference

The example verifies correctness element-wise over the **full output** vector against an f32 reference: every element must pass `np.isclose(|a−e| ≤ atol + rtol·|e|)`.

| Output dtype | rtol | atol |
|---|---|---|
| bf16 | 1.6e-2 | 1e-3 |

- **Reference** = CPU FP32 dot product (`a.astype(f32) @ b.astype(f32)`), cast to bf16 before compare. Inputs are `randn × 4` (seed 42).
- `rtol = 1.6e-2` is PyTorch / vLLM's canonical bf16 tolerance (`torch.testing` default, also PyTorch's `test_addmv` matvec test). `atol = 1e-3` is *tighter* than PyTorch's matvec test (which sets bf16 `atol = 1.0` and relies on rtol alone). The thresholds are still **very loose relative to the measured error**: because GEMV does not use the BFP16-emulated MMA, the actual `mean_rel_L1` is `0`–`2.7e-8` (orders of magnitude below the bf16 GEMM floor). All tested shapes pass with this much-tighter-than-standard tolerance, several bit-identically.

---

## Tested shapes

The five GEMV shapes used by llama-3.2-1B decode. Tile for each shape is the **fastest legal config found by a full sweep** of `herd_m × tile_m × m_input` (99 configs total; exact search space and legality rules in [tile choice vs performance](#tile-choice-vs-performance)).

The precision columns are exactly what the reproduce command's `[precision]` line prints: `mean_rel_L1 = mean|out−ref| / mean|ref|` (the robust headline metric), `rel_err max`, and `abs_err max`.

| (M, K) | tile (herd_m/tile_m/m_input) | latency | GFLOPS | mean_rel_L1 | rel_err max | abs_err max | Used by | Status |
|---|---|---|---|---|---|---|---|---|
| 2048×2048 | 8/8/8 | 330 µs | 25.5 | 1.6e-9 | 4.4e-3 | 2.0e-3 | llama-3.2-1B Q proj | ✅ |
| 512×2048 | 8/8/8 | 136 µs | 15.5 | 0.0 | 0.0 | 0.0 | llama-3.2-1B K / V proj | ✅ |
| 8192×2048 | 8/8/8 | 1067 µs | 31.5 | 2.7e-8 | 5.9e-3 | 1.3e-1 | coverage (see note) | ✅ |
| 2048×8192 | 8/2/2 | 1082 µs | 31.0 | 0.0 | 0.0 | 0.0 | coverage (see note) | ✅ |
| 16384×2048 | 8/8/8 | 2193 µs | 30.6 | 0.0 | 0.0 | 0.0 | llama-3.2-1B LM-head (per-partition) | ✅ |

> **What "Used by" means here.** This kernel (`matvec.py` + `mv.cc`, extern `@matvec_vectorized_bf16_bf16`) is the **exact** kernel llama-3.2-1B decode uses for **Q / K / V projections** (in the `rms_gemv_rope` ELF) and the **LM-head** (in the `lm_head_gemv` ELF). The **O / Gate / Up / Down** projections in decode do **not** use this plain GEMV — they run through *fused* cascade kernels (GEMV + residual add, and GEMV + SwiGLU + RMSNorm) in the `o_gemv_ffn` ELF, which are separate kernels (`bf16_cascade/mv_bf16.cc`, `decode_ffn_swiglu/matvec_swiglu_rms.py`) and get their own registry entries. The 8192×2048 and 2048×8192 rows above are **coverage shapes** — they exercise this kernel at the Gate/Up and Down dimensions for completeness, but those projections are served by the fused kernels in the actual model.

**Reading the table**:
- **Accuracy is effectively exact** (`mean_rel_L1 ≤ 2.7e-8`, several shapes bit-identical to the f32 reference) — because GEMV uses an FP32 vector accumulate, not the BFP16-emulated MMA. This is fundamentally more accurate than BF16 GEMM (whose `mean_rel_L1 ≈ 9e-3`). The lone larger figure, `abs_err max = 1.3e-1` on the 8192×2048 coverage shape, is one output element where the reference magnitude is large; its relative error is still `5.9e-3`.
- **GEMV is memory-bound**, so throughput is ~15–32 GFLOPS — far below GEMM's ~9000. Each kernel reads the entire `M×K` matrix to produce one length-`M` vector (arithmetic intensity ≈ 0.5 FLOP/byte), so performance is gated by DMA bandwidth, not the MAC array. Larger M (LM-head, the 8192-row coverage shape) amortizes overhead better and gets closer to the bandwidth ceiling; small-M K/V is overhead-dominated.
- **Accuracy is independent of tile / herd** — it is set only by the FP32-accumulate datapath, so `tile_m` / `m_input` / `herd_m` change only performance, never the numbers (which is why `herd_m` can be fixed at 8 with no accuracy cost).

---

## tile choice vs performance

### What the sweep covered

The tiles above were picked by sweeping, per shape, **every legal combination** of the three knobs on the (only) external path and taking the fastest. The search space:

| Knob | Values swept | Notes |
|---|---|---|
| `herd_m` | `{4, 8}` | both swept only to confirm 8 always wins (see below); deploy fixed at 8 |
| `tile_m` | `{2, 4, 8, 16}` | even values from 2 up to whatever the L2 budget allows |
| `m_input` | all divisors of `tile_m` (`{1, 2, 4, 8, 16}`) | per-call row granularity |

A combination is **"legal"** when it satisfies all the [Tunable parameters](#tunable-parameters) hard constraints: `M % (tile_m·herd_m) == 0`, `tile_m % m_input == 0`, `K % 64 == 0`, the L2 budget `herd_m·tile_m·K·2 + herd_m·tile_m·2 ≤ 512 KiB`, and `tile_m ≥ 2` even (4-byte DMA alignment). This yields **99 legal configs across the five shapes** (e.g. 23 each for the K=2048 shapes, 7 for Down). The legality rules prune the larger `tile_m` values automatically: at `K=2048, herd_m=8` the L2 budget caps `tile_m ≤ 8`; at `K=8192` (Down) it caps `tile_m ≤ 2`; `tile_m=16` is only reachable at `herd_m=4`. Every legal config **passes correctness** — illegal ones fail at build time with a clear assertion (L2 over-budget) or DMA-alignment error (`tile_m=1`), never silently.

Two knobs dominate performance:

**`herd_m = 8` is the biggest lever** — using all 8 columns instead of 4 is ~2.6× faster (more DMA engines moving the matrix in parallel, which is what a memory-bound kernel needs). Example (2048×2048): `herd_m=4` best = 9.4 GFLOPS vs `herd_m=8` best = 25.5 GFLOPS.

**`m_input = tile_m` is a secondary lever** — processing the whole tile in one kernel call instead of row-by-row is ~1.2× (2048×2048, `tile_m=8`: `m_input=1` → 20.9, `m_input=8` → 25.5 GFLOPS), by cutting per-call overhead.

`tile_m` itself mostly matters via the L2 budget: pick the largest legal value (8 at K=2048, 2 at K=8192). The Down shape (K=8192) is forced to `tile_m=2` by L2 yet still reaches 31 GFLOPS, because at large K the A-panel DMA dominates regardless of `tile_m`.

---

## How to reproduce (correctness + performance, one command)

`matvec.py` (compile-and-run mode, the default) does **both** in a single invocation, via `XRTRunner`:
- **correctness** — full-output element-wise check against the f32 reference; prints `[precision] mean_rel_L1=... | rel_err max=... | abs_err max=... | rtol=... atol=...` and `PASS!` / `failed.`
- **performance** — add `--perf-iters N` to time the kernel over `N` iterations (after 10 warmup runs, kernel-only) and print `Latency (us): ... | Throughput: ... GFLOP/s`.

Every entry in the [tested-shapes table](#tested-shapes) reproduces with the commands below. The external kernel links a precompiled `mv.o`, so compile it for THIS `tile_m` first (`mv.o` only depends on `tile_m`, via `-DDIM_M_OUTPUT`).

Example: the 2048×8192 (Down) row, best tile `herd_m=8 / tile_m=2 / m_input=2`.

```bash
cd programming_examples/matrix_vector_multiplication/bf16

# external kernel mv.o depends only on tile_m (DIM_M_OUTPUT)
make compile-kernel AIE_TARGET=aie2p TILE_M=2      # → build_peano/mv.o
cd build_peano
python3 ../matvec.py \
  --m 2048 --k 8192 \
  --tile-m 2 --m-input 2 --herd-m 8 \
  --perf-iters 20
```

For another shape/tile from the table, change `--m/--k` and `--tile-m/--m-input/--herd-m` to that row (and recompile `mv.o` with matching `TILE_M`).

Notes:
- If the NPU is shared with other jobs, serialize on-device runs (e.g. with `flock`) so timing measurements aren't perturbed.
