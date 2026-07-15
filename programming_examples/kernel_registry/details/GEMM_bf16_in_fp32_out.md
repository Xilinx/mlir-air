<!---//===- GEMM_bf16_in_fp32_out.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# GEMM (BF16 in, FP32 out) — Kernel Detail

> Generic `C = A @ B` for the Q/K/V/O/Gate/Up/Down weight projections of a decoder-only LLM. **BF16 inputs, FP32 accumulation, FP32 output.**
> Shapes are written **`M×K×N`**: `C[M,N] = A[M,K] @ B[K,N]` (M = output rows / seq, K = reduction, N = output cols).
>
> Companion: [`GEMM_bf16_in_bf16_out.md`](GEMM_bf16_in_bf16_out.md) (the bf16-output sibling) · [`../supported_kernels.md`](../supported_kernels.md) · [`../README.md`](../README.md)
> **Scope: NPU2 (Strix / AIE2P) only.** All numbers, tile configs, and tolerances here are for the aie2p target, measured on real NPU2 (RyzenAI-npu4), full-chip herd 8×4, June 2026. Reproduce commands in "How to test" below.

---

## Why this page is "always high precision"

Because the output is **f32**, the L1 C **accumulator IS the output** — the K reduction stays FP32 the entire way and there is **no intermediate bf16 truncation anywhere**. This matches the GPU standard (FP32 accumulate, single cast / no cast at the epilogue). So there is no precision knob here: f32 output is unconditionally the high-precision tier (`mean_rel_L1 ≈ 9.3e-3`, the BFP16-emulated MMA's block-quantization floor + bf16 *input* rounding — both of which a GPU bf16 GEMM also has). If you want a **bf16** output, see the sibling [`GEMM_bf16_in_bf16_out.md`](GEMM_bf16_in_bf16_out.md), where keeping this precision costs an extra epilogue cast.

---

## Builder

```
programming_examples/matrix_multiplication/bf16_in_fp32_out/run.py
  build_module(m, k, n,
               tile_m, tile_k_l2, tile_k_l1, tile_n,
               herd_m, herd_n,
               np_dtype_in, np_dtype_out,    # bfloat16, float32
               arch="aie2p",                 # NPU2 (this registry's target)
               direct_codegen=False,         # external (default) vs direct
               emit_external_call=True)      # external path
```

The example is self-contained (`run.py` + `Makefile` + `mm_aie2p.cc` + lit + README). The one contract knob is **`--codegen {external,direct}`**; both compute identical math:

| `--codegen` | Compute kernel | Note |
|---|---|---|
| **external** (default) | hand-tuned `mm_aie2p.cc` → `mm.o`, linked via `link_with` | needs `mm.o` pre-compiled (`make compile-kernel`) |
| **direct** | matmul lowered by the compiler from a transform script (no `.cc`) | self-contained, PEANO only |

**Both paths are numerically bit-identical** (verified: per-element residuals match to the last digit across all tested shapes). At the swept-best tiles the external path is **~1.5–1.7× faster** on large shapes (its `mm.o` is a hand-tuned, Peano `-O2` vectorized microkernel with tighter instruction scheduling). Choose external for performance, direct for a self-contained / inspectable IR.

> The legacy `matrix_multiplication/bf16/` example carries the same `build_module` base behind low-level flags and also targets NPU1 (aie2); this registry tracks the NPU2 contract-split `bf16_in_fp32_out/` example.

---

## Numerical datapath (what "BF16-in / FP32-out GEMM" means here)

```
A,B stored bf16 → 8×8×8 MMA (BFP16-emulated) → FP32 accumulator → FP32 output
```

- **MMA accumulator is always FP32** (`accfloat` / ACC2048) — there is no bf16-accumulate path on the hardware for bf16 inputs. Within a single matmul invocation, the K-reduction accumulates in FP32 registers.
- **aie2p uses BFP16 emulation** (`-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`): bf16 operands are cast to block-floating-point `v64bfp16ebs8` (8 elements share one exponent) for the 8×8×8 MMA. This adds a small block-quantization error a true bf16 GEMM does not have; `conv_even` rounding keeps it unbiased.
- **The L1 C buffer dtype = output dtype = f32**, so there is **no inter-L2-tile truncation**: K is tiled twice (an outer L2 loop `K / tile_k_l2` wrapping the herd, and an inner L1 loop `tile_k_l2 / tile_k_l1`), and because the partial sum is written back to an **f32** L1 C buffer at each L2-tile boundary, the partials stay FP32 across the whole K loop. This is the GPU single-/no-epilogue-cast standard.
- **Consequence (measured):** the f32-out path sits flat at `mean_rel_L1 ≈ 9.3e-3` regardless of shape or tile — the accuracy is set by the dtype + accumulation path, not by K or the tile count. (Contrast: direct-codegen *bf16*-out truncates once per L2 tile and degrades with K — see the [bf16-out page](GEMM_bf16_in_bf16_out.md).)

---

## Tunable parameters

These are the knobs to set when deploying a **new** model. The **Recommended** column is what the tile sweep found best for large GEMMs (full-chip, `tile_n=128`). For a new shape, start from the nearest [tested shape](#tested-shapes) and tune from there.

| Knob | Recommended | Hard constraint | Note |
|---|---|---|---|
| `tile_m` | 64 | `M % (tile_m × herd_m) == 0` | larger → fewer M iters, more L1/tile (use 32 when M is small, e.g. 512) |
| `tile_k_l2` | 256–512 | `K % tile_k_l2 == 0` | larger → fewer A-DMA reloads; secondary perf lever (512 helps at moderate K, saturates by K=4096) |
| `tile_k_l1` | 32 | `tile_k_l2 % tile_k_l1 == 0` | inner-accumulation chunk; only this enters L1 for K |
| `tile_n` | **128** | **`N % (tile_n × herd_n) == 0`** ⚠️ | **dominant perf knob** — 128 beats 64 by ~1.3–1.4× at large N |
| `herd_m` | 8 | `M ≥ tile_m × herd_m` | row-parallel; 8 = full chip rows |
| `herd_n` | 4 | `N ≥ tile_n × herd_n` | col-parallel; 4 = full chip cols → 8×4 = 32-tile herd |

`herd_m=8, herd_n=4` uses all 32 NPU2 compute tiles and is right for any large GEMM. Among the tile knobs, **`tile_n=128` is the biggest lever**, `tile_k_l2` is secondary (see [tile choice vs performance](#tile-choice-vs-performance)).

⚠️ **Silent-corruption trap**: `N % (tile_n × herd_n) != 0` is **not asserted** by the builder — always verify it before running, or output is silently wrong.

**L1 budget** (64 KB / tile): the f32 C accumulator at `tile_m=64`, `tile_n=128` is 32 KB; with bf16 A/B ping-pong it totals ~57 KB < 64 KB, so it fits (no separate drain buffer, unlike the bf16-out path). Note only `tile_k_l1` (not `tile_k_l2`) lands in L1.

---

## Tolerances & reference

The example verifies correctness element-wise over the **full output** against an f32 reference: **every element must pass** `np.isclose(|a−e| ≤ atol + rtol·|e|)` (0% mismatch allowed).

| Output dtype | rtol | atol |
|---|---|---|
| f32 | 1.6e-2 | 1.5e-3 |

- **Reference** = CPU FP32 matmul (`a.astype(f32) @ b.astype(f32)`). Inputs are **variance-normalized** `randn / sqrt(K)` so output magnitude is O(1) and the relative tolerance behaves consistently across K.
- **`rtol = 1.6e-2` anchors to PyTorch's bf16 standard** (`torch.testing.assert_close`: bf16 `rtol=1.6e-2`, ≈ bf16's 7-bit-mantissa floor 2⁻⁸ ≈ 0.4 %). Even though the output is *stored* as f32, the GEMM **computes** in bf16 (bf16 inputs + BFP16-emulated MMA), so the per-element error floor is bf16's, not f32's `1.3e-6` — applying PyTorch's f32 rtol here would wrongly fail a correct kernel.
- **`atol = 1.5e-3` encodes the FP32-accumulate tier**: measured worst-case `abs_err ≈ 5.8e-4` (Down 2048×8192×2048, ~2.5× margin), tight enough to reject the bf16-truncation tier (`abs_err ≈ 2.4e-3`, see the bf16-out page). `mean_rel_L1` (printed, ~9.3e-3) is diagnostic only — it does not gate.

---

## Tested shapes

> **Machine-readable source of truth: [`GEMM_bf16_in_fp32_out.json`](GEMM_bf16_in_fp32_out.json).**
> The tables below mirror that JSON. Model code reads it via
> `kernel_registry/registry_lookup.py` `gemm_config(M,K,N,"f32")` to pick the tile
> sizes per shape — never hand-copied. Update the JSON first when re-measuring.

Shapes cover both LLM weight-projection shapes (the four 2048-row entries) and a square K-sweep (512³→4096³) to show how throughput scales with K. The tile for each shape is the **fastest legal tile found by an external-path sweep** over `tile_m × tile_k_l2 × tile_n` (full-chip herd 8×4); both code-paths use that tile. Accuracy is **tile-independent** at this tier.

Measured on NPU2 (RyzenAI-npu4), June 2026, with this example's `run.py`, all PASS, `mean_rel_L1` 9.28–9.46e-3.

### EXTERNAL (`--codegen external`), f32 out — **fastest path**

| (M, K, N) | tile (m/kl2/kl1/n) | GFLOPS | mean_rel_L1 | Used by | Status |
|---|---|---|---|---|---|
| 2048×2048×2048 | 64/512/32/128 | 8508 | 9.3e-3 | llama-3.2-1B Q/O proj | ✅ |
| 2048×2048×512 | 64/256/32/128 | 7342 | 9.3e-3 | llama-3.2-1B K/V proj | ✅ |
| 2048×2048×8192 | 64/256/32/128 | 8278 | 9.3e-3 | llama-3.2-1B Gate/Up proj | ✅ |
| 2048×8192×2048 | 64/256/32/128 | **9797** | 9.3e-3 | llama-3.2-1B Down proj | ✅ |
| 512×512×512 | 32/256/32/128 | 1791 | 9.3e-3 | K-sweep | ✅ |
| 1024×1024×1024 | 64/256/32/128 | 6256 | 9.5e-3 | K-sweep | ✅ |
| 4096×4096×4096 | 64/512/32/128 | 9329 | 9.4e-3 | K-sweep | ✅ |

### DIRECT (`--codegen direct`), f32 out — self-contained

| (M, K, N) | tile (m/kl2/kl1/n) | GFLOPS | mean_rel_L1 | Used by | Status |
|---|---|---|---|---|---|
| 2048×2048×2048 | 64/512/32/128 | 5516 | 9.3e-3 | llama-3.2-1B Q/O proj | ✅ |
| 2048×2048×512 | 64/256/32/128 | 4896 | 9.3e-3 | llama-3.2-1B K/V proj | ✅ |
| 2048×2048×8192 | 64/256/32/128 | 5582 | 9.3e-3 | llama-3.2-1B Gate/Up proj | ✅ |
| 2048×8192×2048 | 64/256/32/128 | 6010 | 9.3e-3 | llama-3.2-1B Down proj | ✅ |
| 512×512×512 | 32/256/32/128 | 1536 | 9.3e-3 | K-sweep | ✅ |
| 1024×1024×1024 | 64/256/32/128 | 4413 | 9.5e-3 | K-sweep | ✅ |
| 4096×4096×4096 | 64/512/32/128 | 5791 | 9.4e-3 | K-sweep | ✅ |

**Reading the tables**:
- **Performance**: external > direct everywhere (~1.5–1.7× on large shapes). External peaks at ~9.8K GFLOPS (Down); direct codegen is slower but self-contained. Throughput climbs with problem size — tiny 512³ only reaches ~1.5–1.8K (can't fill the 32-tile fabric).
- **Accuracy**: external and direct are **bit-identical** (true FP32 partials across K) at `mean_rel_L1 = 9.3e-3` for every shape and tile. Tile and herd are pure performance knobs.

---

## tile choice vs performance

The tiles in the tables above were picked by sweeping, per shape, every legal combination of `tile_m ∈ {32, 64, 128}` × `tile_k_l2 ∈ {64, 128, 256, 512}` × `tile_n ∈ {32, 64, 128}` (with `tile_k_l1 = 32` fixed, herd 8×4), on the external path, and taking the fastest. A combination is "legal" when it satisfies the hard constraints in [Tunable parameters](#tunable-parameters) and fits the L1 budget; `tile_n = 256` was excluded as it overflowed L1 at these tile_m. Two knobs dominate:

**`tile_n` is the biggest lever** — `tile_n=128` beats `tile_n=64` by ~1.3–1.4× at every large shape (same `tile_m`, same `tile_k_l2`):

| shape | tile_k_l2 | tile_n=64 | tile_n=128 | speedup |
|---|---|---|---|---|
| 2048³ | 512 | 6367 GFLOPS | 8600 | 1.35× |
| Down (K=8192) | 256 | 6816 | 9321 | 1.37× |
| 4096³ | 256 | 6623 | 9161 | 1.38× |

**`tile_k_l2` is a secondary lever** — larger reduces A-DMA reloads, helps most at moderate K (2048³: 256→512 gives +13%; at K=4096 it's already saturated, ~+0%).

- **accuracy is identical** across all tiles (bit-for-bit `mean_rel_L1`) — tile is a pure performance knob.

---

## How to reproduce (correctness + performance, one command)

`make run` drives `run.py --compile-mode compile-and-run`, which does **both** in a single invocation via `XRTRunner`:
- **correctness** — full-output element-wise check against the f32 reference; prints `[precision] mean_rel_L1=... | rel_err max=... | abs_err max=... | rtol=... atol=...` and `PASS!` / `failed.`
- **performance** — `--perf-iters N` (Makefile `PERF_ITERS`) times the kernel over `N` iterations (after warmup, kernel-only) and prints `Throughput: ... GFLOP/s`.

Every entry in the [tested-shapes tables](#tested-shapes) reproduces with the commands below. Example: the 2048×8192×2048 (Down) row, best tile `64/256/32/128`.

```bash
cd programming_examples/matrix_multiplication/bf16_in_fp32_out

# ---- EXTERNAL, f32 out (fastest; make compiles mm.o for this tile first) ----
make run CODEGEN=external M=2048 K=8192 N=2048 \
  TILE_M=64 TILE_K_L2=256 TILE_K_L1=32 TILE_N=128 \
  AIE_TARGET=aie2p PERF_ITERS=20

# ---- DIRECT codegen, f32 out (self-contained, no mm.o) ----
make run CODEGEN=direct M=2048 K=8192 N=2048 \
  TILE_M=64 TILE_K_L2=256 TILE_K_L1=32 TILE_N=128 \
  AIE_TARGET=aie2p PERF_ITERS=20
```

For another shape/tile from the tables, change `M/K/N` and the four `TILE_*` vars to that row.

Notes:
- The lit tests `run_npu2_external_peano.lit` / `run_npu2_direct_peano.lit` pin the Down shape and assert `PASS!`.
- If the NPU is shared with other jobs, serialize on-device runs (e.g. with `flock`) so timing measurements aren't perturbed.
