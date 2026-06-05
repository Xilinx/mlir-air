<!---//===- GEMM_bf16.md --------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# GEMM (BF16) — Kernel Detail

> Generic `C = A @ B` for the Q/K/V/O/Gate/Up/Down weight projections of a decoder-only LLM. BF16 inputs, FP32 accumulation, BF16 or FP32 output.
> Shapes are written **`M×K×N`**: `C[M,N] = A[M,K] @ B[K,N]` (M = output rows / seq, K = reduction, N = output cols).
>
> Companion: [`../supported_kernels.md`](../supported_kernels.md) · [`../README.md`](../README.md)
> **Scope: NPU2 (Strix / AIE2P) only.** All numbers, tile configs, and tolerances here are for the aie2p target, measured on real NPU2 (RyzenAI-npu4), full-chip herd 8×4, June 2026. Reproduce commands in "How to test" below.

---

## Builder

```
programming_examples/matrix_multiplication/bf16/run.py
  build_module(m, k, n,
               tile_m, tile_k_l2, tile_k_l1, tile_n,
               herd_m, herd_n,
               np_dtype_in, np_dtype_out,
               arch="aie2p",           # NPU2 (this registry's target)
               direct_codegen=False)
```

Driven by `run.py`'s CLI; the example also has a `Makefile`. Two distinct **code-generation paths** on aie2p produce the same math:

| Path | Flag | Compute kernel | Note |
|---|---|---|---|
| **external** (non-direct) | `--direct-codegen` **off** (default) | hand-tuned `mm_aie2p.cc` → `mm.o`, linked via `link_with` | needs `mm.o` pre-compiled |
| **direct codegen** | `--direct-codegen` **on** | matmul lowered by the compiler from a transform script (no `.cc`) | self-contained |

**Both paths are numerically bit-identical** (verified: per-element residuals match to the last digit across all tested shapes). At the swept-best tiles the external path is **~1.5–1.65× faster** on large shapes (its `mm.o` is a hand-tuned, Peano `-O2` vectorized microkernel with tighter instruction scheduling; the gap narrows to ~1.1× on tiny shapes). Choose external for performance, direct for a self-contained / inspectable IR.

---

## Numerical datapath (what "BF16 GEMM" means here)

```
A,B stored bf16 → 8×8×8 MMA (BFP16-emulated) → FP32 accumulator → output bf16 or f32
```

- **MMA accumulator is always FP32** (`accfloat` / ACC2048) — there is no bf16-accumulate path on the hardware for bf16 inputs.
- **aie2p uses BFP16 emulation** (`-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`): bf16 operands are cast to block-floating-point `v64bfp16ebs8` (8 elements share one exponent) for the 8×8×8 MMA. This adds a small block-quantization error a true bf16 GEMM does not have; `conv_even` rounding keeps it unbiased.
- **K-dimension partial sums stay in FP32** across the whole K loop; the cast to bf16 (when output is bf16) happens **once at the end**, not per K-tile. This matches a standard GPU BF16 GEMM (FP32 partials, single epilogue cast).

---

## Tunable parameters

These are the knobs to set when deploying a **new** model. The **Recommended** column is what the tile sweep found best for large GEMMs (full-chip, `tile_n=128`); it is **not** `run.py`'s argparse default — see the note below. For a new shape, start from the nearest [tested shape](#tested-shapes) and tune from there.

| Knob | Recommended | Hard constraint | Note |
|---|---|---|---|
| `tile_m` | 64 | `M % (tile_m × herd_m) == 0` | larger → fewer M iters, more L1/tile (use 32 when M is small, e.g. 512) |
| `tile_k_l2` | 256–512 | `K % tile_k_l2 == 0` | larger → fewer A-DMA reloads; secondary perf lever (512 helps at moderate K, saturates by K=4096) |
| `tile_k_l1` | 32 | `tile_k_l2 % tile_k_l1 == 0` | inner-accumulation chunk; only this enters L1 for K |
| `tile_n` | **128** | **`N % (tile_n × herd_n) == 0`** ⚠️ | **dominant perf knob** — 128 beats 64 by ~1.3–1.4× at large N |
| `herd_m` | 8 | `M ≥ tile_m × herd_m` | row-parallel; 8 = full chip rows |
| `herd_n` | 4 | `N ≥ tile_n × herd_n` | col-parallel; 4 = full chip cols → 8×4 = 32-tile herd |

`herd_m=8, herd_n=4` uses all 32 NPU2 compute tiles and is right for any large GEMM. Among the tile knobs, **`tile_n=128` is the biggest lever**, `tile_k_l2` is secondary (see [tile choice vs performance](#tile-choice-vs-performance)).

> **`run.py` argparse default ≠ Recommended.** The example's argparse defaults (`--tile-m 128 --tile-k-l2 128 --tile-k-l1 32 --tile-n 64 --herd-m 4 --herd-n 4`) are a half-chip (4×4), `tile_n=64` config that is **sub-optimal** (~half the peak GFLOPS). Always pass the Recommended values explicitly for performance.

⚠️ **Silent-corruption trap**: `N % (tile_n × herd_n) != 0` is **not asserted** by the builder — always verify it before running, or output is silently wrong.

**L1 budget** (64 KB / tile): `tile_m·tile_k_l1 + tile_k_l1·tile_n + tile_m·tile_n` BF16 elements (×2 for ping-pong) must fit. Note only `tile_k_l1` (not `tile_k_l2`) lands in L1 — the two-level K-tiling is why `tile_k_l2` can be large (256) without overflowing L1.

---

## Tolerances & reference

The example verifies correctness element-wise over the **full output** against an f32 reference: **every element must pass** `np.isclose(|a−e| ≤ atol + rtol·|e|)`.

| Output dtype | rtol | atol |
|---|---|---|
| f32 (default) | 2e-3 | 2e-3 |
| bf16 | 1.6e-2 | 4e-3 |

- **Reference** = CPU FP32 matmul (`a.astype(f32) @ b.astype(f32)`), cast to the output dtype before compare. Inputs are **variance-normalized** `randn / sqrt(K)` so output magnitude is O(1) and the relative tolerance behaves consistently across K.
- `rtol = 1.6e-2` for bf16 is PyTorch / vLLM's canonical bf16 tolerance (≈ bf16's 7-bit-mantissa floor 2⁻⁸ ≈ 0.4 %). `atol = 4e-3` (bf16) is sized to the BFP16-emulated path's measured worst-case absolute error (~3e-3, from block quantization + bf16 output rounding) — set per measured kernel behavior, not by copying a single GPU number.

---

## Tested shapes

Shapes cover both LLM weight-projection shapes (the four 2048-row entries) and a square K-sweep (512³→4096³) to show how error and throughput scale with K. The tile for each shape is the **fastest legal tile found by an external-path sweep** over `tile_m × tile_k_l2 × tile_n` (full-chip herd 8×4); all three code-paths then use that tile.

The precision columns are exactly what the reproduce command's `[precision]` line prints: `mean_rel_L1 = mean|out−ref| / mean|ref|` (the robust headline metric), `rel_err max`, and `abs_err max`. Note `rel_err max` is large where the reference is ≈0 (a single near-zero output element) — read `mean_rel_L1` and `abs_err max` for real accuracy.

### EXTERNAL (non-direct), f32 out — **fastest path**

| (M, K, N) | tile (m/kl2/kl1/n) | latency | GFLOPS | mean_rel_L1 | rel_err max | abs_err max | Used by | Status |
|---|---|---|---|---|---|---|---|---|
| 2048×2048×2048 | 64/512/32/128 | 2.01 ms | 8540 | 9.3e-3 | 4.6e+5 | 1.1e-3 | llama-3.2-1B Q/O proj | ✅ |
| 2048×2048×512 | 64/256/32/128 | 0.58 ms | 7384 | 9.3e-3 | 1.3e+4 | 1.0e-3 | llama-3.2-1B K/V proj | ✅ |
| 2048×2048×8192 | 64/256/32/128 | 8.37 ms | 8210 | 9.3e-3 | 3.7e+5 | 1.1e-3 | llama-3.2-1B Gate/Up proj | ✅ |
| 2048×8192×2048 | 64/256/32/128 | 7.24 ms | **9492** | 9.3e-3 | 1.3e+6 | 5.8e-4 | llama-3.2-1B Down proj | ✅ |
| 512×512×512 | 32/256/32/128 | 0.14 ms | 1870 | 9.3e-3 | 1.1e+3 | 2.0e-3 | K-sweep | ✅ |
| 1024×1024×1024 | 64/256/32/128 | 0.34 ms | 6337 | 9.5e-3 | 2.6e+3 | 1.5e-3 | K-sweep | ✅ |
| 4096×4096×4096 | 64/512/32/128 | 14.87 ms | 9243 | 9.4e-3 | 3.9e+4 | 8.3e-4 | K-sweep | ✅ |

### DIRECT codegen, f32 out

| (M, K, N) | tile (m/kl2/kl1/n) | latency | GFLOPS | mean_rel_L1 | rel_err max | abs_err max | Used by | Status |
|---|---|---|---|---|---|---|---|---|
| 2048×2048×2048 | 64/512/32/128 | 3.14 ms | 5470 | 9.3e-3 | 4.6e+5 | 1.1e-3 | llama-3.2-1B Q/O proj | ✅ |
| 2048×2048×512 | 64/256/32/128 | 0.87 ms | 4910 | 9.3e-3 | 1.3e+4 | 1.0e-3 | llama-3.2-1B K/V proj | ✅ |
| 2048×2048×8192 | 64/256/32/128 | 12.84 ms | 5351 | 9.3e-3 | 3.7e+5 | 1.1e-3 | llama-3.2-1B Gate/Up proj | ✅ |
| 2048×8192×2048 | 64/256/32/128 | 12.00 ms | 5726 | 9.3e-3 | 1.3e+6 | 5.8e-4 | llama-3.2-1B Down proj | ✅ |
| 512×512×512 | 32/256/32/128 | 0.16 ms | 1716 | 9.3e-3 | 1.1e+3 | 2.0e-3 | K-sweep | ✅ |
| 1024×1024×1024 | 64/256/32/128 | 0.49 ms | 4407 | 9.5e-3 | 2.6e+3 | 1.5e-3 | K-sweep | ✅ |
| 4096×4096×4096 | 64/512/32/128 | 23.88 ms | 5756 | 9.4e-3 | 3.9e+4 | 8.3e-4 | K-sweep | ✅ |

### DIRECT codegen, bf16 out

| (M, K, N) | tile (m/kl2/kl1/n) | latency | GFLOPS | mean_rel_L1 | rel_err max | abs_err max | Used by | Status |
|---|---|---|---|---|---|---|---|---|
| 2048×2048×2048 | 64/512/32/128 | 3.29 ms | 5224 | 1.3e-2 | 5.4e+5 | 2.9e-3 | llama-3.2-1B Q/O proj | ✅ |
| 2048×2048×512 | 64/256/32/128 | 0.97 ms | 4444 | 1.3e-2 | 2.6e+4 | 2.9e-3 | llama-3.2-1B K/V proj | ✅ |
| 2048×2048×8192 | 64/256/32/128 | 13.42 ms | 5122 | 1.3e-2 | 4.1e+5 | 3.2e-3 | llama-3.2-1B Gate/Up proj | ✅ |
| 2048×8192×2048 | 64/256/32/128 | 12.75 ms | 5388 | 1.9e-2 | 3.5e+6 | 2.4e-3 | llama-3.2-1B Down proj | ✅ |
| 512×512×512 | 32/256/32/128 | 0.15 ms | 1837 | 1.0e-2 | 1.2e+3 | 2.9e-3 | K-sweep | ✅ |
| 1024×1024×1024 | 64/256/32/128 | 0.50 ms | 4289 | 1.1e-2 | 3.1e+3 | 2.4e-3 | K-sweep | ✅ |
| 4096×4096×4096 | 64/512/32/128 | 25.22 ms | 5450 | 1.5e-2 | 5.6e+4 | 2.9e-3 | K-sweep | ✅ |

**Reading the tables**:
- **Performance**: external > direct-f32 > direct-bf16. External peaks at ~9.5K GFLOPS (Down shape); direct codegen is slower but self-contained. Throughput climbs with problem size — tiny 512³ only reaches ~1.7–1.9K (can't fill the 32-tile fabric).
- **Accuracy**: external and direct-f32 are **bit-identical** (same mean_rel_L1 / rel_err / abs_err to the digit). bf16 output is slightly less accurate (one extra output-quantization step, abs_err max ~2.4–3.2e-3) but still within the bf16 tolerance (atol 4e-3).
- **Error grows with K**: at the K=8192 Down shape the bf16-out mean_rel_L1 reaches ~1.9e-2 — measure at the real K, not just a small smoke shape.
- **Accuracy is independent of tile / herd**: precision is set only by dtype + accumulation; tile/herd are pure performance knobs.

---

## tile choice vs performance

The tiles in the tables above were picked by sweeping, per shape, every legal combination of `tile_m ∈ {32, 64, 128}` × `tile_k_l2 ∈ {64, 128, 256, 512}` × `tile_n ∈ {32, 64, 128}` (with `tile_k_l1 = 32` fixed, herd 8×4), on the external path, and taking the fastest. A combination is "legal" when it satisfies the hard constraints in [Tunable parameters](#tunable-parameters) (`M % (tile_m·herd_m) == 0`, `N % (tile_n·herd_n) == 0`, `K % tile_k_l2 == 0`, `tile_k_l2 % tile_k_l1 == 0`) and fits the L1 budget; `tile_n = 256` was excluded as it overflowed L1 / failed to compile at these tile_m. Two knobs dominate:

**`tile_n` is the biggest lever** — `tile_n=128` beats `tile_n=64` by ~1.3–1.4× at every large shape (same `tile_m`, same `tile_k_l2`):

| shape | tile_k_l2 | tile_n=64 | tile_n=128 | speedup |
|---|---|---|---|---|
| 2048³ | 512 | 6367 GFLOPS | 8600 | 1.35× |
| Down (K=8192) | 256 | 6816 | 9321 | 1.37× |
| 4096³ | 256 | 6623 | 9161 | 1.38× |

**`tile_k_l2` is a secondary lever** — larger reduces A-DMA reloads, helps most at moderate K (2048³: 256→512 gives +13%; at K=4096 it's already saturated, ~+0%).

- **accuracy is identical** across all tiles (bit-for-bit `mean_rel_L1`) — tile is a pure performance knob.
- the earlier registry tiles (and the example's `run.py` argparse default) used `tile_n=64`, leaving ~30–40% on the table; the swept tiles here use `tile_n=128`.

---

## How to reproduce (correctness + performance, one command)

`run.py --compile-mode compile-and-run` does **both** in a single invocation, via `XRTRunner`:
- **correctness** — full-output element-wise check against the f32 reference; prints `[precision] mean_rel_L1=... | rel_err max=... | abs_err max=... | rtol=... atol=...` and `PASS!` / `failed.`
- **performance** — add `--perf-iters N` to time the kernel over `N` iterations (after 10 warmup runs, kernel-only — buffer sync excluded) and print `Latency (us): ... | Throughput: ... GFLOP/s`.

Every entry in the [tested-shapes tables](#tested-shapes) reproduces with the commands below. Pass the shape via `--m/--k/--n` and the tile config from that row via `--tile-m/--tile-k-l2/--tile-k-l1/--tile-n`. `--herd-m 8 --herd-n 4` is the full-chip herd.

Example: the 2048×8192×2048 (Down) row, best tile `64/256/32/128`.

```bash
cd programming_examples/matrix_multiplication/bf16

# ---- DIRECT codegen, f32 out (self-contained) ----
python3 run.py --arch aie2p --direct-codegen \
  --m 2048 --k 8192 --n 2048 \
  --tile-m 64 --tile-k-l2 256 --tile-k-l1 32 --tile-n 128 \
  --herd-m 8 --herd-n 4 --compile-mode compile-and-run --perf-iters 20

# ---- DIRECT codegen, bf16 out ----  (add --output-dtype bf16)
python3 run.py --arch aie2p --direct-codegen --output-dtype bf16 \
  --m 2048 --k 8192 --n 2048 \
  --tile-m 64 --tile-k-l2 256 --tile-k-l1 32 --tile-n 128 \
  --herd-m 8 --herd-n 4 --compile-mode compile-and-run --perf-iters 20

# ---- EXTERNAL kernel, f32 out (fastest) ----
# external links a precompiled mm.o, so compile it for THIS tile first.
# DIM_K = tile_k_l1; DIM_M/DIM_N = tile_m/tile_n.
make compile-kernel AIE_TARGET=aie2p TILE_M=64 TILE_N=128 TILE_K_L1=32   # → build_peano/mm.o
cd build_peano
python3 ../run.py --arch aie2p \
  --m 2048 --k 8192 --n 2048 \
  --tile-m 64 --tile-k-l2 256 --tile-k-l1 32 --tile-n 128 \
  --herd-m 8 --herd-n 4 --compile-mode compile-and-run --perf-iters 20
```

For another shape/tile from the tables, change `--m/--k/--n` and the four `--tile-*` flags to that row (and, for external, recompile mm.o with matching `TILE_M`/`TILE_N`/`TILE_K_L1`).

Notes:
- ⚠️ **`make profile` hardcodes M=K=N=1024** — its `M=`/`K=`/`N=` args only select which kernel `.o` compiles, not the profiled shape. Use `run.py --perf-iters N` (above) to time a specific shape. (An alternative C++ path: `run.py --compile-mode compile-and-xclbin` then `test.exe -M -K -N`.)
- If the NPU is shared with other jobs, serialize on-device runs (e.g. with `flock`) so timing measurements aren't perturbed.
