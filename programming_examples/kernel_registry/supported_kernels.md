<!---//===- supported_kernels.md ------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# Supported Kernels Registry — LLM Deployment on NPU2

High-level index of the leaf kernels validated for decoder-only LLM deployment on AMD NPU2 (Strix, AIE2P): which kernels are covered, which shapes have been tested, and the best measured performance. Per-kernel detail (datapath, tunable parameters, tolerances, how to reproduce) lives in `details/<KERNEL>.md`.

This is **documentation, not executable code** — it records results produced by the `programming_examples/` kernels, run on real NPU2. See [`README.md`](README.md) for scope and methodology.

**Status legend**: ✅ verified on real NPU2, accuracy in line with the bf16 standard · ⚠️ verified on real NPU2 but with a documented precision/coverage caveat · ❌ broken/missing

> **Scope**: currently **GEMM**, **GEMV**, **RMSNorm**, **FlashAttention**, **Element-wise Add**, **SiLU-and-Mul**, and **RoPE** — the registry is built up one verified kernel at a time. The core LLM leaf kernels are now covered; see [`README.md`](README.md) for the roadmap.

---

## Kernels

| Kernel | Detail | Best measured throughput (NPU2, units per entry) | Status |
|---|---|---|---|
| GEMM (BF16 in, FP32 out) | [`details/GEMM_bf16_in_fp32_out.md`](details/GEMM_bf16_in_fp32_out.md) | **9797 GFLOP/s** (external, 2048×8192×2048, full-chip 8×4) | ✅ |
| GEMM (BF16 in, BF16 out) | [`details/GEMM_bf16_in_bf16_out.md`](details/GEMM_bf16_in_bf16_out.md) | **8898 GFLOP/s** (fused-cast incl. cast, 2048×8192×2048, full-chip 8×4) | ✅ |
| GEMV (BF16) | [`details/GEMV_bf16.md`](details/GEMV_bf16.md) | **32 GFLOP/s** (memory-bound, 16384×2048, herd 8) | ✅ |
| RMSNorm (BF16) | [`details/RMSNorm_bf16.md`](details/RMSNorm_bf16.md) | **18.4 GB/s** (memory-bound, 2048×2048, herd 8) | ✅ |
| FlashAttention (BF16, GQA) | [`details/FlashAttention_bf16.md`](details/FlashAttention_bf16.md) | **1065–1131 GFLOP/s** (2048×2048, dk=64, 32q/8kv causal, full-chip 32 tiles) | ✅ |
| Element-wise Add (BF16) | [`details/EltwiseAdd_bf16.md`](details/EltwiseAdd_bf16.md) | **57.7 GB/s** (memory-bound, N=4194304, herd 8×1) | ✅ |
| SiLU-and-Mul (BF16) | [`details/SiLU_Mul_bf16.md`](details/SiLU_Mul_bf16.md) | **25.1 GB/s** (memory-bound, N=16777216, herd 8×1) | ✅ |
| RoPE (BF16, half-split) | [`details/RoPE_bf16.md`](details/RoPE_bf16.md) | **43.4 GB/s** (memory-bound, 65536×64, herd 8×1) | ✅ |

---

## GEMM (f32 out) — tested shapes

`C[M,N] = A[M,K] @ B[K,N]`, shapes written `M×K×N`. **BF16 in, FP32 out** — always FP32-accumulate (no precision knob). GFLOPS is the fastest (external) path; `mean_rel_L1` = `mean|out−ref| / mean|ref|` vs an FP32 reference. Full per-path data, tolerances, and reproduce commands are in [`details/GEMM_bf16_in_fp32_out.md`](details/GEMM_bf16_in_fp32_out.md).

| (M×K×N) | best tile (m/kl2/kl1/n) | external GFLOPS | direct GFLOPS | mean_rel_L1 | Used by | Status |
|---|---|---|---|---|---|---|
| 2048×2048×2048 | 64/512/32/128 | 8508 | 5516 | 9.3e-3 | llama-3.2-1B Q/O proj | ✅ |
| 2048×2048×512 | 64/256/32/128 | 7342 | 4896 | 9.3e-3 | llama-3.2-1B K/V proj | ✅ |
| 2048×2048×8192 | 64/256/32/128 | 8278 | 5582 | 9.3e-3 | llama-3.2-1B Gate/Up proj | ✅ |
| 2048×8192×2048 | 64/256/32/128 | **9797** | 6010 | 9.3e-3 | llama-3.2-1B Down proj | ✅ |
| 512×512×512 | 32/256/32/128 | 1791 | 1536 | 9.3e-3 | K-sweep | ✅ |
| 1024×1024×1024 | 64/256/32/128 | 6256 | 4413 | 9.5e-3 | K-sweep | ✅ |
| 4096×4096×4096 | 64/512/32/128 | 9329 | 5791 | 9.4e-3 | K-sweep | ✅ |

> Measured on NPU2 (RyzenAI-npu4), June 2026. Two code-paths (external / direct-codegen); external is ~1.5–1.7× faster and bit-identical in accuracy to direct — see [`details/GEMM_bf16_in_fp32_out.md`](details/GEMM_bf16_in_fp32_out.md).

---

## GEMM (bf16 out) — tested shapes

`C[M,N] = A[M,K] @ B[K,N]`, **BF16 in, BF16 out** (half the DDR bytes of f32-out). `--high-precision true` (default) keeps FP32-accumulate + a single epilogue cast (`mean_rel_L1 ≈ 9.7e-3`, GPU standard); `false` is direct-codegen with per-L2-tile bf16 truncation (faster, 1.3e-2–1.9e-2). Within high-precision, `--method auto` picks **fused-cast** (`M*K*N ≥ 4e9`) or **drain** (else). GFLOPS for fused-cast includes the cast launch. Full data in [`details/GEMM_bf16_in_bf16_out.md`](details/GEMM_bf16_in_bf16_out.md).

| (M×K×N) | high-prec fused-cast | high-prec drain | low-prec direct | mean_rel_L1 (high / low) | Used by | Status |
|---|---|---|---|---|---|---|
| 2048×2048×2048 | **6215** | 6025 | 5230 | 9.7e-3 / 1.3e-2 | llama-3.2-1B Q/O proj + Qwen3-1.7B Q/O proj (square) + Qwen2.5-3B Q/O proj (square) | ✅ |
| 2048×2048×512 | 4083 | **5626** | 4765 | 9.7e-3 / 1.3e-2 | llama-3.2-1B K/V proj | ✅ |
| 2048×2048×8192 | **6893** | 5784 | 5287 | 9.7e-3 / 1.3e-2 | llama-3.2-1B Gate/Up proj | ✅ |
| 2048×8192×2048 | **8898** | 7234 | 5592 | 9.7e-3 / 1.9e-2 | llama-3.2-1B Down proj | ✅ |
| 512×512×512 | 482 | **1703** | 1750 | 9.7e-3 / 1.0e-2 | K-sweep | ✅ |
| 1024×1024×1024 | 2502 | **4637** | 4456 | 9.9e-3 / 1.1e-2 | K-sweep | ✅ |
| 4096×4096×4096 | **8423** | 7002 | 5509 | 9.9e-3 / 1.5e-2 | K-sweep | ✅ |
| 2048×1024×2048 | **4425** | — | — | 9.9e-3 / — | Qwen3-0.6B Q proj | ✅ |
| 2048×1024×1024 | — | **4980** | — | 9.4e-3 / — | Qwen3-0.6B K/V proj | ✅ |
| 2048×2048×1024 | **5392** | — | — | 9.7e-3 / — | Qwen3-0.6B O proj + Qwen3-1.7B K/V proj | ✅ |
| 2048×1024×3072 | ⚠️ | ⚠️ | **5006** | 9.4e-3 / 1.1e-2 | Qwen3-0.6B Gate/Up proj | ⚠️ |
| 2048×3072×1024 | **6461** | — | — | 9.9e-3 / — | Qwen3-0.6B Down proj | ✅ |
| 2048×896×896 | — (drain m32/n32) | **2516** | — | 9.4e-3 / — | Qwen2.5-0.5B Q/O proj | ✅ |
| 2048×896×128 | — (drain m32/n32) | **1890** | — | 9.4e-3 / — | Qwen2.5-0.5B K/V proj | ✅ |
| 2048×896×4864 | ⚠️ | — | **4320** | — / 1.11e-2 | Qwen2.5-0.5B Gate/Up proj | ⚠️ |
| 2048×4864×896 | **3640** (n32) | — | — | 9.8e-3 / — | Qwen2.5-0.5B Down proj | ✅ |
| 2048×1536×1536 | **4821** | — | — | 9.7e-3 / — | Qwen2.5-1.5B Q/O proj | ✅ |
| 2048×1536×256 | — (drain n64) | **3770** | — | 9.3e-3 / — | Qwen2.5-1.5B K/V proj | ✅ |
| 2048×1536×8960 | ⚠️ | — | **4165** (n64) | — / 1.2e-2 | Qwen2.5-1.5B Gate/Up proj | ⚠️ |
| 2048×8960×1536 | **8804** | — | — | 9.7e-3 / — | Qwen2.5-1.5B Down proj | ✅ |
| 2048×2048×6144 | **6729** | — | — | 9.7e-3 / — | Qwen3-1.7B Gate/Up proj | ✅ |
| 2048×6144×2048 | **8536** | — | — | 9.7e-3 / — | Qwen3-1.7B Down proj | ✅ |
| 2048×2048×256 | — | **4112** (drain m32/n64) | — | 9.3e-3 / — | Qwen2.5-3B K/V proj | ✅ |
| 2048×2048×11008 | ⚠️ | — | **4276** (n64, tile_k_l2=128) | — / 1.28e-2 | Qwen2.5-3B Gate/Up proj | ⚠️ |
| 2048×11008×2048 | **9447** | — | — | 9.8e-3 / — | Qwen2.5-3B Down proj | ✅ |
| 2048×2560×4096 | **fused-cast (m64/k256/n128)** | — | — | max_abs 1.22e-3 | Qwen3-4B Q proj | ✅ |
| 2048×2560×1024 | **fused-cast (m64/k256/n128)** | — | — | max_abs 9.77e-4 | Qwen3-4B K/V proj | ✅ |
| 2048×4096×2560 | **fused-cast (m64/k256/n128)** | — | — | max_abs 9.77e-4 | Qwen3-4B O proj (decoupled) | ✅ |
| 2048×2560×9728 | ⚠️ | — | **direct (m64/k128/n64)** | — / max_abs 2.93e-3 | Qwen3-4B Gate/Up proj | ⚠️ |
| 2048×9728×2560 | **fused-cast (m64/k256/n128)** | — | — | max_abs 4.88e-4 | Qwen3-4B Down proj | ✅ |
| 2048×3072×3072 | **7513** | — | — | 9.9e-3 / — | Llama-3.2-3B Q/O proj (square) | ✅ |
| 2048×3072×8192 | **7601** | — | — | 9.9e-3 / — | Llama-3.2-3B Gate/Up proj | ✅ |
| 2048×8192×3072 | **9092** | — | — | 9.7e-3 / — | Llama-3.2-3B Down proj | ✅ |

> **Qwen3-4B rows — emb=2560 (512-aligned, NOT 1024-aligned), q_dim=4096 decoupled (≠emb), kv_dim=1024, hidden=9728=512·19.** All proj N divisible by 4·TILE_N=512, so stock TILE_N=128 HERD_N=4 places. Q/K/V/O/Down PASS high-precision fused-cast directly (max_abs ≤ 1.22e-3, well within high-prec tolerance), K=2560/4096/9728 all use tile_k_l2=256. O proj is **decoupled** (K=q_dim=4096, N=emb=2560), the largest non-square O in the registry. **2048×2560×9728 (Gate/Up) ⚠️**: high-precision fused-cast FAILS at compile (`aie.dma_bd op Stride exceeds [1:1048576] range` on the f32-out B-tile DMA at N=9728 — same large-N class as Qwen2.5-3B's N=11008); the low-precision `direct` path (tile_k_l2=128, TILE_N=64) PASSES at max_abs 2.93e-3 — same Gate/Up low-prec tier-down as every Qwen sibling. Large-K Down (K=9728) does NOT trigger the bug (only large-N does). Qwen3-4B uses the qwen25_3b 5-ELF un-merge (o_res_norm / gate / up / HOST SwiGLU / down_add).

> **Qwen2.5-3B rows — emb=q_dim=2048 (1024-aligned, square O), hidden=11008=256·43 (NOT 512-aligned).** Q/O proj is square 2048×2048×2048 (reuses the llama Q/O row). K/V is **2048×2048×256** — thin N=256→TILE_N=64, drain TILE_M=32, K=2048 tile_k_l2=256 (differs from Qwen2.5-1.5B K/V only in K=2048 vs 1536). Down is **2048×11008×2048** — N=2048 stock TILE_N=128, K=11008 tile_k_l2=256, fused-cast PASSES high-precision at 9.8e-3. **2048×2048×11008 (Gate/Up) ⚠️**: both high-precision fused-cast AND low-prec direct at `tile_k_l2=256` fail aiecc with `aie.dma_bd op Stride 2818048 exceeds the [1:1048576] range` (stride = tile_k_l2·N); the low-precision `direct` path with **`tile_k_l2=128`** PASSES at 1.28e-2 (`atol=4e-3`) — same Gate/Up low-prec tier-down as every Qwen sibling, root cause here being the DMA stride range (not L1 over-allocation as in 1.5B).

> **Qwen3-1.7B rows — all dims 1024-aligned, square O.** emb=q_dim=2048 → O proj is square 2048×2048×2048 (reuses the llama Q/O row); K/V 2048×2048×1024 reuses the Qwen3-0.6B O-proj row. The two new shapes are Gate/Up **2048×2048×6144** (N=6144=512·12, stock TILE_N=128 HERD_N=4; `tile_k_l2=256` — `512` BD-exhausts at this N) and Down **2048×6144×2048** (K=6144, `tile_k_l2=256`). Both MKN=2.6e10 ≥ 4e9 → fused-cast, and both PASS high-precision directly at 9.7e-3 (no near-zero atol artifact, unlike the smaller-Qwen Gate/Up shapes) — no low-precision tier needed.

> **Qwen2.5-1.5B rows — 1536 is 512-aligned.** emb=q_dim=1536=512·3 is divisible by the default `4·TILE_N=512`, so **Q/O/Down (N=1536) place at the stock `TILE_N=128 HERD_N=4`** — no TILE_N shrink (contrast Qwen2.5-0.5B's 896). Only thin **K/V (N=256 → TILE_N=64, drain TILE_M=32)** and wide **Gate/Up (N=8960 → TILE_N=64)** drop below 128. K=1536 uses `tile_k_l2=256` (1536/256=6), K=8960 uses `tile_k_l2=256` (8960/256=35). **2048×1536×8960 (Gate/Up) ⚠️**: high-precision fused-cast (TILE_M=64 TILE_N=64) over-allocates L1 → compile fail; the low-precision `direct` path PASSES but needs `tile_k_l2=128` (tile_k_l2=256 also compile-fails at this N), at 1.2e-2 — same Gate/Up tier-down as the smaller Qwen siblings.

> **Qwen2.5-0.5B rows — non-512-aligned N.** Qwen2.5's projection widths (896, 128, 4864) are not divisible by the default `4·TILE_N=512`, and `HERD_N=1` (e.g. `TILE_N=128` for N=896) **fails at runtime** (`qds_device::wait() unexpected command state` — the fused-cast/drain paths assume the 8×4 array). The working recipe keeps `HERD_N=4` and shrinks `TILE_N` so `4·TILE_N | N`: **N=896/128 → TILE_N=32**, **N=4864 → TILE_N=64**. K=896 uses `tile_k_l2=128` (896/128=7), K=4864 uses `tile_k_l2=256`. The thin shapes need `METHOD=drain` (`tile_m=32`; `tile_m=64` over-allocates L1). No padding was required — every real shape placed and PASSED. **2048×896×4864 (Gate/Up) ⚠️**: high-precision fused-cast computes the in-tier result (9.4e-3) but the harness gate trips on 2 near-zero-reference elements (abs_err ≈ 1.6–1.9e-3 > high-prec `atol=1.5e-3`); PASSES on the low-precision `direct` path (`atol=4e-3`, 1.11e-2) — same artifact as Qwen3-0.6B Gate/Up.

> GFLOPS, all PASS. **Bold** = faster high-precision method (what `auto` picks); the `M*K*N ≥ 4e9` threshold matches the bold winner for all 7 shapes.
> Qwen3-0.6B rows: only the `auto`-selected high-precision method was swept (`—` = the other method not measured for that shape); all `auto` picks PASS at 9.4–9.9e-3. **2048×1024×3072 (Gate/Up) ⚠️**: both high-precision methods compute the in-tier result (mean_rel_L1 = 9.4e-3) but the harness element-wise gate trips on a single near-zero-reference output element (abs_err ≈ 1.7e-3 > the high-precision `atol = 1.5e-3`, `rtol·|ref|≈0`); the shape PASSES on the low-precision `direct` path (`atol = 4e-3`, 1.1e-2). Harness tolerance edge, not a datapath failure — see [`details/GEMM_bf16_in_bf16_out.md`](details/GEMM_bf16_in_bf16_out.md). fused-cast is tile_m=64, drain is tile_m=32. The high-precision tier preserves f32-out accuracy (9.3–9.9e-3) via a single cast; low-precision direct degrades with the L2-tile count (`K / tile_k_l2`). See [`details/GEMM_bf16_in_bf16_out.md`](details/GEMM_bf16_in_bf16_out.md).

---

## GEMV — tested shapes

`C[M] = A[M,K] @ B[K]`, shapes written `M×K`. The decode-time (batch = 1) projections of llama-3.2-1B. GEMV is **memory-bound** (reads the whole `M×K` matrix for one length-`M` output), so GFLOPS is far below GEMM; the fastest config is `herd_m=8` (all columns) with the largest L2-legal `tile_m`. Full data, tunables, and reproduce commands are in [`details/GEMV_bf16.md`](details/GEMV_bf16.md).

| (M×K) | best tile (herd_m/tile_m/m_input) | GFLOPS | mean_rel_L1 | Used by | Status |
|---|---|---|---|---|---|
| 2048×2048 | 8/8/8 | 25.5 | 1.6e-9 | llama-3.2-1B Q proj + Qwen3-1.7B decode Q/O proj + Qwen2.5-3B decode Q/O proj | ✅ |
| 512×2048 | 8/8/8 | 15.5 | 0.0 | llama-3.2-1B K/V proj | ✅ |
| 8192×2048 | 8/8/8 | 31.5 | 2.7e-8 | coverage | ✅ |
| 2048×8192 | 8/2/2 | 31.0 | 0.0 | coverage | ✅ |
| 16384×2048 | 8/8/8 | **30.6** | 0.0 | llama-3.2-1B LM-head + Qwen3-1.7B LM-head + Qwen2.5-3B LM-head (K=2048 partition datapath) | ✅ |
| 49152×2048 | 8/8/8 | 32.5 | 5.9e-8 | SmolLM2-1.7B LM-head | ✅ |
| 2048×1024 | 8/8/8 | 18.2 | 1.2e-6 | Qwen3-0.6B decode Q proj | ✅ |
| 1024×1024 | 8/8/8 | 14.3 | 0.0 | Qwen3-0.6B decode K/V proj | ✅ |
| 16384×1024 | 8/16/16 | 31.4 | 2.0e-8 | Qwen3-0.6B LM-head (per-partition) | ✅ |
| 896×896 | 8/8/8 | (mem-bound) | 0.0 | Qwen2.5-0.5B decode Q/O proj | ✅ |
| 128×896 | 8/8/8 | (mem-bound) | 0.0 | Qwen2.5-0.5B decode K/V proj | ✅ |
| 4864×896 | 8/8/8 | (mem-bound) | 0.0 | Qwen2.5-0.5B decode Gate/Up proj | ✅ |
| 896×4864 | 8/2/2 | (mem-bound) | 0.0 | Qwen2.5-0.5B decode Down proj | ✅ |
| 16384×896 | 8/16/16 | (mem-bound) | 7.2e-12 | Qwen2.5-0.5B LM-head (per-partition) | ✅ |
| 1536×1536 | 8/8/8 | (mem-bound) | 0.0 | Qwen2.5-1.5B decode Q/O proj | ✅ |
| 256×1536 | 8/8/8 | (mem-bound) | 0.0 | Qwen2.5-1.5B decode K/V proj | ✅ |
| 8960×1536 | 8/8/8 | (mem-bound) | 1.7e-9 | Qwen2.5-1.5B decode Gate/Up proj | ✅ |
| 1536×8960 | 8/2/2 | (mem-bound) | 2.2e-6 | Qwen2.5-1.5B decode Down proj | ✅ |
| 16384×1536 | 8/16/16 | (mem-bound) | 2.3e-8 | Qwen2.5-1.5B LM-head (per-partition) | ✅ |
| 1024×2048 | 8/8/8 | (mem-bound) | 0.0 | Qwen3-1.7B decode K/V proj | ✅ |
| 6144×2048 | 8/8/8 | (mem-bound) | 0.0 | Qwen3-1.7B decode Gate/Up proj | ✅ |
| 2048×6144 | 8/2/2 | (mem-bound) | 0.0 | Qwen3-1.7B decode Down proj | ✅ |
| 256×2048 | 8/8/8 | (mem-bound) | 0.0 | Qwen2.5-3B decode K/V proj | ✅ |
| 11008×2048 | 8/8/8 | (mem-bound) | 7.9e-8 | Qwen2.5-3B decode Gate/Up proj | ✅ |
| 2048×11008 | 8/2/1 | (mem-bound) | 0.0 | Qwen2.5-3B decode Down proj (K=11008 L1-bound → m_input=1) | ✅ |
| 4096×2560 | 8/8/8 | (mem-bound) | 0.0 | Qwen3-4B decode Q proj | ✅ |
| 1024×2560 | 8/8/8 | (mem-bound) | 0.0 | Qwen3-4B decode K/V proj | ✅ |
| 2560×4096 | 8/4/4 | (mem-bound) | 0.0 | Qwen3-4B decode O proj (decoupled K=4096 → tile_m=4 m_input=4 to fit L2) | ✅ |
| 9728×2560 | 8/8/8 | (mem-bound) | 0.0 | Qwen3-4B decode Gate/Up proj | ✅ |
| 2048×9728 | 8/2/1 | (mem-bound) | 0.0 | Qwen3-4B decode Down proj (HOST — K=9728 stitched-ELF L1 overflow) | ✅ |
| 8192×2560 | 8/16/16 | (mem-bound) | 0.0 | Qwen3-4B LM-head (19 partitions × n_part=8192, K=2560) | ✅ |

> **Qwen3-4B GEMV.** Decode projections bit-identical (0.0) to the f32 ref. emb=2560 K, q_dim=4096 decoupled. O proj is **decoupled** (M=emb=2560, K=q_dim=4096) — at K=4096 the full `[m_input, K]` A tile constrains L2, so `tile_m=4, m_input=4` (vs the stock 8/8) keeps A=tile_m·herd_m·K·2 ≤ 512 KiB. Down proj (K=9728) runs on **HOST** (stitched-ELF L1 overflow, same as Qwen2.5-3B's K=11008). LM-head reuses the shared 19-partition vocab=151936 datapath at K=2560 per partition.

> **Qwen2.5-3B GEMV.** Decode projections bit-identical (0.0) or ≤7.9e-8 to the f32 ref. Q/O proj is 2048×2048 (reuses the llama Q row); LM-head is K=2048 per-partition (reuses the 16384×2048 datapath row). K=11008 (Down proj) is the most L1-constrained GEMV in the registry — the harness loads the full `[m_input, K]` A tile + `[K]` B vector into L1 (no K-tiling), so at K=11008 even `tile_m=2, m_input=2` (44 KB A-tile) overflows the 64 KB L1; **`tile_m=2, m_input=1` (22 KB A-tile) PASSES**. (`tile_m=1` is rejected by the 4-byte transfer-length check.)

> **Qwen3-1.7B GEMV.** Decode projections all bit-identical (0.0) to the f32 ref. Q/O proj is 2048×2048 (reuses the llama Q row). K=6144 (Down proj) is the L2-constrained shape — `8·tile_m·6144·2 ≤ 256KB` forces `tile_m=2`. **LM-head is 151936×2048** — too tall single-shot (outer > 255 BD repeat limit, same as all siblings); run per-partition (n_part=8192, 19 partitions), and the K=2048 LM-head datapath is verified at partition scale by the 16384×2048 row above (8/8/8, mean_rel_L1=0.0).

> **Qwen2.5-1.5B GEMV.** Decode projections (Q/O/K/V/Gate-Up) bit-identical or ≤1.7e-9 to the f32 ref. K=8960 (Down proj) is the L2-constrained shape — `tile_m=2` places (`tile_m=1` fails the placement pass, not L2). **LM-head is 151936×1536** — too tall single-shot (outer > 255 BD repeat limit, same as all siblings); run per-partition, K=1536 datapath verified at partition scale by the 16384×1536 row (outer=128, mean_rel_L1=2.3e-8).

> **Qwen2.5-0.5B GEMV.** Decode projections (Q/O/Gate-Up/Down) all bit-identical to the f32 ref. K=4864 (Down proj) is the only L2-constrained shape — `8·tile_m·4864·2 ≤ 256KB` forces `tile_m=2`. **LM-head is 151936×896** — too tall single-shot (outer loop > 255 BD repeat limit, same as Qwen3/llama); run per-partition, the K=896 datapath verified at partition scale by the 16384×896 row (outer=128, mean_rel_L1=7.2e-12).

> **Qwen3-0.6B LM-head is 151936×1024** — too tall to run single-shot: the outer launch loop = `M/(tile_m·herd_m)` exceeds the 255 buffer-descriptor repeat-count limit at every legal tile (151936 = 8·16·1187 has no `tile_m` divisor between 16 and 1187), so it is run **per-partition** like llama-3.2-1B's LM-head. The 16384×1024 row above verifies the K=1024 LM-head datapath at partition scale (128 launches, PASS, mean_rel_L1 = 2.0e-8).

> This plain GEMV is the exact kernel for llama-3.2-1B decode's **Q / K / V projections and LM-head**. The **O / Gate / Up / Down** projections use *fused* cascade variants (GEMV+residual, GEMV+SwiGLU+RMSNorm) — separate kernels, separate registry entries; the 8192×2048 / 2048×8192 rows here are coverage shapes. See [`details/GEMV_bf16.md`](details/GEMV_bf16.md).
> GEMV uses an **FP32 vector accumulate** (not the BFP16-emulated MMA that GEMM uses), so accuracy is effectively exact — `mean_rel_L1 ≤ 2.7e-8`, several shapes bit-identical to the f32 reference, orders of magnitude tighter than BF16 GEMM's ~9e-3.

---

## RMSNorm — tested shapes

`y = x / sqrt(mean(x²) + eps) · weight`, per row; shapes written `M×N` (M = rows / seq, N = emb_dim = reduction axis). The per-layer norm of llama-3.2-1B. **Memory-bound** (streams the whole matrix for an elementwise op), so throughput is reported as bandwidth; the fastest config is `herd_x=8` (all columns, near-linear scaling). Full data, the precision caveat, and reproduce commands are in [`details/RMSNorm_bf16.md`](details/RMSNorm_bf16.md).

| (M×N) | herd_x | latency | bandwidth | mean_rel_L1 | Used by | Status |
|---|---|---|---|---|---|---|
| 2048×2048 | 8 | 911 µs | 18.4 GB/s | 4.2e-3 | llama-3.2-1B + Qwen3-1.7B + Qwen2.5-3B prefill RMSNorm | ✅ |
| 2048×1024 | 8 | — | (mem-bound) | 4.3e-3 | Qwen3-0.6B prefill RMSNorm | ✅ |
| 2048×128 | 8 | — | (mem-bound) | 4.6e-3 | Qwen3-0.6B + Qwen3-1.7B QK-norm (per-head, N=head_dim) | ✅ |
| 2048×896 | 8 | — | (mem-bound) | 4.2e-3 | Qwen2.5-0.5B prefill RMSNorm | ✅ |
| 2048×1536 | 8 | — | (mem-bound) | 4.3e-3 | Qwen2.5-1.5B prefill RMSNorm | ✅ |

> **Qwen3-0.6B QK-norm (2048×128)** is per-head RMSNorm over `head_dim=128` (Qwen3-specific q_norm/k_norm) — the same weighted-RMSNorm kernel with a small `N=128` reduction axis; verified PASS at 4.6e-3, confirming the kernel handles a 128-wide reduction. (Harness `eps = 1e-5`; Qwen3 `eps = 1e-6` — the difference is negligible vs the bf16 datapath error.)

> Follows the **GPU / HuggingFace standard**: the `sum(x²)` reduction is accumulated in **FP32** (matching PyTorch `rms_norm_composite` / HF `LlamaRMSNorm`), giving `mean_rel_L1 = 4.2e-3` — in line with the GEMM tier and passing the canonical bf16 `rtol = 1.6e-2`. (`atol = 5e-2` covers a few large-magnitude bf16 *output*-rounding ULPs, not a reduction relaxation.) The FP32 reduction costs essentially nothing on this memory-bound kernel. See [`details/RMSNorm_bf16.md`](details/RMSNorm_bf16.md).

---

## FlashAttention — tested shapes

Fused scaled-dot-product attention (online-softmax FlashAttention) with grouped-query attention and optional causal masking. **Compute-bound** (two matmuls Q@Kᵀ and P@V), so throughput is GFLOP/s. Kernel = `attn_npu2.o`, driven by the **heads-first** harness `attn_npu2.py`; verified on NPU2 across head dim 64/128, MHA & GQA, short & long sequences, causal & non-causal. (A **seq-first** variant `attn_npu2_seqfirst.py` drives the same `.o` for llama-3.2-1B prefill — bit-identical.) **All rows use the one near-unique full-chip config** `lqp=256, num_q_tiles=4, num_heads_per_unroll=2, num_cascade_stages=4` (FA's tile config is determined by the constraints, not tuned — see detail page). Full datapath, tunables, and reproduce commands in [`details/FlashAttention_bf16.md`](details/FlashAttention_bf16.md).

| lq×lk | dk/dv | heads q/kv | causal | dv_chunks | latency | GFLOP/s | mean_rel_L1 | Status |
|---|---|---|---|---|---|---|---|---|
| 2048×2048 | 64/64 | 32/8 | ✓ | 1 | 15.4–16.1 ms | **1065–1116** | 3.9e-2 | ✅ |
| 2048×2048 | 64/64 | 32/32 | ✓ | 1 | 16.9 ms | 2031 | 3.9e-2 | ✅ |
| 512×512 | 64/64 | 2/2 | ✗ | 1 | 0.73 ms | 184 | 4.4e-2 | ✅ |
| 512×512 | 64/64 | 12/6 | ✗ | 1 | 1.22 ms | 661 | 4.6e-2 | ✅ |
| 512×512 | 64/64 | 64/8 | ✗ | 1 | 3.79 ms | 1135 | 4.6e-2 | ✅ |
| 512×512 | 128/128 | 32/8 | ✗ | 2 | 4.38 ms | 980 | 4.4e-2 | ✅ |
| 512×512 | 128/128 | 28/4 | ✗ | 2 | 4.05 ms | 928 | 4.4e-2 | ✅ |
| 16384×16384 | 64/64 | 2/2 | ✓ | 1 | 39.6 ms | 1734 | 4.5e-2 | ✅ |
| 16384×16384 | 64/64 | 2/2 | ✗ | 1 | 40.1 ms | **3427** | 5.5e-2 | ✅ |
| 2048×2048 | 128/128 | 16/8 | ✓ | 2 | — | — | 3.8e-2 | ✅ |
| 2048×2048 | 64/64 | 14/2 | ✓ | 1 | — | — | 3.8e-2 | ✅ |

> **Qwen3-0.6B prefill attention** (`head_dim = 128`, 16q/8kv GQA, causal, lq=lk=2048): verified PASS at mean_rel_L1 = 3.8e-2 (full-output check, rtol 1.6e-2 / atol 1e-1) with the default full-chip config (`lqp=256, num_q_tiles=4, num_heads_per_unroll=2, num_cascade_stages=4`, `dv_chunks=2` for head_dim=128). Note: head_dim=128 FA has been flaky (hang/NaN) on some NPU2 setups; this run completed cleanly, and Qwen3-0.6B prefill can also fall back to CPU attention (`cpu_attn`) if a deployment hits the hang.

> **Qwen2.5-1.5B prefill attention** (`head_dim = 128`, 12q/2kv GQA, causal, lq=lk=2048): verified PASS at mean_rel_L1 = 3.83e-2 (full-output check, rtol 1.6e-2 / atol 1e-1) with the default full-chip config (`lqp=256, num_q_tiles=4, num_heads_per_unroll=2, num_cascade_stages=4`, `dv_chunks=2` for head_dim=128). head_dim=128 FA has been flaky (hang/NaN) on some NPU2 setups; this run completed cleanly, and prefill can fall back to CPU attention (`cpu_attn`) if a deployment hits the hang.

> **Qwen2.5-0.5B prefill attention** (`head_dim = 64`, 14q/2kv GQA, causal, lq=lk=2048): verified PASS at mean_rel_L1 = 3.83e-2 with the default full-chip config (`lqp=256, lkp=64, num_q_tiles=4, num_heads_per_unroll=2, num_cascade_stages=4`, `dv_chunks=1` for head_dim=64). head_dim=64 has no hang risk. Prefill can also fall back to CPU attention (`cpu_attn`).

> All rows measured on NPU2 with the heads-first harness at the default tiling (`lqp=256, num_q_tiles=4, num_heads_per_unroll=2, num_cascade_stages=4` = 32 tiles, full 8×4 array). Accuracy `mean_rel_L1 ≈ 3.9e-2` is ~4× the GEMM tier: FA chains **two BFP16-emulated MMAs** plus a **bf16 online-softmax**, so it is looser than a single matmul (looser than GPU FA's `5e-2` only by the `atol`, not the standard `rtol = 1.6e-2`); accuracy is set by the datapath, not the shape. The **2048, 32q/8kv causal** row is llama-3.2-1B prefill's config (seq-first harness, bit-identical to heads-first — verified `max abs diff = 0`); its GFLOP/s range is run-to-run timing variation. `head_dim=128` rows use `dv_chunks=2`. A separate tunable sweep found only 2 of 8 candidate 32-tile configs place (constraints: columns `num_heads_per_unroll × num_q_tiles ≤ 8`, rows `num_cascade_stages ≤ 4`, `num_heads_per_unroll ≤ 2`). See [`details/FlashAttention_bf16.md`](details/FlashAttention_bf16.md).

---

## Element-wise Add — tested shapes

`c = a + b`, per-element, BF16. The residual adds of llama-3.2-1B (the prefill residual is the fused `o_ffn` inline 2-D variant — same math; this entry measures the **standalone** `eltwise_add`). **Memory-bound** (O(N) streaming, zero arithmetic intensity), so throughput is bandwidth. The **cleanest** kernel in the registry — a single bf16 rounding, no accumulation. Full datapath, herd sweep, and reproduce commands in [`details/EltwiseAdd_bf16.md`](details/EltwiseAdd_bf16.md).

| N | best config (hx/hy/tile_n) | latency | bandwidth | mean_rel_L1 | Status |
|---|---|---|---|---|---|
| 1048576 | 8/1/2048 | 175 µs | 36.0 GB/s | 1.9e-3 | ✅ |
| 2097152 | 8/1/2048 | 277 µs | 45.4 GB/s | 1.9e-3 | ✅ |
| 4194304 (2048×2048) | 8/1/2048 | 437 µs | 57.7 GB/s | 1.9e-3 | ✅ (llama-3.2-1B + Qwen3-1.7B + Qwen2.5-3B residual, seq·emb) |
| 8388608 | 8/1/2048 | 798 µs | **63.0 GB/s** | 1.9e-3 | ✅ |
| 1835008 (2048×896) | 8/1/2048 | — | (mem-bound) | 1.9e-3 | ✅ (Qwen2.5-0.5B residual, seq·emb) |
| 3145728 (2048×1536) | 8/1/2048 | — | (mem-bound) | 1.9e-3 | ✅ (Qwen2.5-1.5B residual, seq·emb) |

> `mean_rel_L1 = 1.9e-3` is the lowest in the registry — `c=a+b` rounds each output once (matching `torch.add` bf16: f32 sum, single round, no accumulation), bit-identical across all configs and `N`. Best config `herd_x=8, herd_y=1` for every shape: the 3-DMA-per-tile shim-channel limit caps the herd at one 8-column row (**cannot fill 32 tiles** — `herd_y>1` fails to place), but within that `herd_x` scales near-linearly (9→57.7 GB/s as herd_x 1→8). Highest bandwidth in the registry (pure streaming). See [`details/EltwiseAdd_bf16.md`](details/EltwiseAdd_bf16.md).

---

## SiLU-and-Mul — tested shapes

`out = SiLU(gate) · up`, `SiLU(x) = x·sigmoid(x)`, per-element, BF16. The SwiGLU activation of llama-3.2-1B prefill FFN (the standalone `silu_and_mul` is measured; llama runs the bit-identical 2-D `build_module_2d` variant). **Memory-bound** (O(N) streaming, ~1 op/byte), so throughput is bandwidth. sigmoid is computed via the hardware `aie::tanh` (`0.5·(1+tanh(g/2))`); the precision is the "bf16 + one transcendental" tier. Full datapath, sweep, and reproduce commands in [`details/SiLU_Mul_bf16.md`](details/SiLU_Mul_bf16.md).

| N | (as 2-D) | best config (hx/hy/tile_n) | latency | bandwidth | mean_rel_L1 | abs_err max | Status |
|---|---|---|---|---|---|---|---|
| 2097152 | — | 8/1/4096 | 569 µs | 22.1 GB/s | 1.0e-2 | 0.125 | ✅ |
| 4194304 | 2048×2048 | 8/1/4096 | 1052 µs | 23.9 GB/s | 1.0e-2 | 0.125 | ✅ |
| 8388608 | — | 8/1/4096 | 2247 µs | 22.4 GB/s | 1.0e-2 | 0.125 | ✅ |
| 16777216 | 2048×8192 | 8/1/4096 | 4016 µs | **25.1 GB/s** | 1.0e-2 | 0.125 | ✅ |
| 6291456 | 2048×3072 (seq·hidden) | 8/1/4096 | — | (mem-bound) | 1.0e-2 | 0.125 | ✅ |
| 9961472 | 2048×4864 (seq·hidden) | 8/1/4096 | — | (mem-bound) | 1.0e-2 | 0.125 | ✅ |
| 18350080 | 2048×8960 (seq·hidden) | 8/1/4096 | — | (mem-bound) | 1.0e-2 | 0.188 | ✅ |
| 12582912 | 2048×6144 (seq·hidden) | 8/1/4096 | — | (mem-bound) | 1.0e-2 | 0.125 | ✅ (Qwen3-1.7B SwiGLU) |
| 22544384 | 2048×11008 (seq·hidden) | 8/1/4096 | — | (mem-bound) | 1.0e-2 | 0.188 | ✅ (Qwen2.5-3B SwiGLU) |

> **Qwen2.5-1.5B SwiGLU**: `N = 18350080 = seq·hidden = 2048·8960` (intermediate size 8960), verified PASS at 1.0e-2 with the default best config.

> **Qwen3-0.6B SwiGLU**: `N = 6291456 = seq·hidden = 2048·3072` (intermediate size 3072), verified PASS at 1.0e-2 with the default best config.

> **Qwen2.5-0.5B SwiGLU**: `N = 9961472 = seq·hidden = 2048·4864` (intermediate size 4864), verified PASS at 1.0e-2 with the default best config.

> `mean_rel_L1 = 1.0e-2` is an order of magnitude above Element-wise Add (1.9e-3): the hardware `aie::tanh<bf16>` LUT approximation plus a chain of bf16 roundings (vs a single rounding for a plain add). Verified element-wise over the full output (no cosine) at `rtol = 1.6e-2, atol = 8e-2` — `atol` covers the worst-case `tanh`-LUT element (`abs_err max = 0.125`); the mean error sits inside `rtol`. Best config `herd_x=8, herd_y=1, tile_n=4096` for every shape (= llama's default): `herd_y>1` fails the shim-channel limit and some `tile_n`/`herd_x` fail a non-monotonic buffer-descriptor limit, so the best config is the fastest one that places. `herd_x` scales 7.6× (1→8). See [`details/SiLU_Mul_bf16.md`](details/SiLU_Mul_bf16.md).

---

## RoPE — tested shapes

Rotary Position Embedding applied to Q/K, **half-split** convention (HuggingFace Llama `rotate_half`), per row; shapes written `rows × head_dim` (rows = n_heads·seq for prefill, n_heads for decode). BF16 in/out, per-element rotation (no reduction, no non-linearity — cos/sin come from a precomputed LUT). **Memory-bound** (streams input + LUT in, output out, ~1 flop/byte), so throughput is bandwidth; the fastest config is `herd_x=8` (all columns, near-linear). The kernel links the **same `rope_halfsplit.cc` (`rope.o`) llama uses** — not the interleaved `rope_lut/`/`rope_sincos/` decoys. Full data, the decoy/provenance note, and reproduce commands are in [`details/RoPE_bf16.md`](details/RoPE_bf16.md).

| (rows×head_dim) | herd (hx/hy) | latency | bandwidth | mean_rel_L1 | Used by | Status |
|---|---|---|---|---|---|---|
| 8×64 | 8/1 | 83 µs | 0.04 GB/s | 2.4e-3 | llama-3.2-1B decode RoPE-K | ✅ |
| 32×64 | 8/1 | 82 µs | 0.15 GB/s | 2.7e-3 | llama-3.2-1B decode RoPE-Q | ✅ |
| 2048×64 | 8/1 | 105 µs | 7.5 GB/s | 2.8e-3 | coverage | ✅ |
| 4096×64 | 8/1 | 118 µs | 13.3 GB/s | 2.8e-3 | coverage / Qwen2.5-0.5B prefill RoPE-K (rows=n_kv·seq=2·2048) | ✅ |
| 28672×64 | 8/1 | — | (mem-bound) | 2.8e-3 | Qwen2.5-0.5B prefill RoPE-Q (rows=n_heads·seq=14·2048) | ✅ |
| 16384×64 | 8/1 | 210 µs | 30.0 GB/s | 2.8e-3 | llama-3.2-1B prefill RoPE-K | ✅ |
| 65536×64 | 8/1 | 579 µs | **43.4 GB/s** | 2.8e-3 | llama-3.2-1B prefill RoPE-Q | ✅ |
| 32768×128 | 8/1 | — | (mem-bound) | 2.8e-3 | Qwen3-0.6B + Qwen3-1.7B + Qwen2.5-3B prefill RoPE-Q (rows=n_heads·seq=16·2048) | ✅ |
| 16384×128 | 8/1 | — | (mem-bound) | 2.8e-3 | Qwen3-0.6B + Qwen3-1.7B prefill RoPE-K (rows=n_kv_heads·seq=8·2048) | ✅ |
| 24576×128 | 8/1 | — | (mem-bound) | 2.8e-3 | Qwen2.5-1.5B prefill RoPE-Q (rows=n_heads·seq=12·2048) | ✅ |
| 4096×128 | 8/1 | — | (mem-bound) | 2.8e-3 | Qwen2.5-1.5B + Qwen2.5-3B prefill RoPE-K (rows=n_kv_heads·seq=2·2048) | ✅ |

> **Qwen3-0.6B uses `head_dim = 128`** (vs llama's 64) — the two rows above are the first registry coverage of `head_dim = 128`; same half-split `rope_halfsplit.cc` kernel, verified PASS at 2.8e-3 (accuracy unchanged, set by the datapath not the head dim).

> `mean_rel_L1 = 2.8e-3` is the second-cleanest in the registry (above Element-wise Add 1.9e-3, below RMSNorm 4.2e-3): a rotation is a few bf16 multiplies and one add/sub per element with **no accumulation** — nothing to amplify error, and `|out| ≈ |x|` so no near-zero blowup. Verified element-wise over the full output (no cosine) at `rtol = 1.6e-2, atol = 5e-2`; bit-identical across all herd configs and shapes (decode rows 8/32 read slightly lower from smaller rotation angles). Best config `herd_x=8, herd_y=1` for every shape: each tile uses 3 shim DMAs (input/LUT in, output out), so `herd_x·herd_y>8` exhausts the shim channels (the herd **cannot fill 32 tiles**, same limit as Element-wise Add / SiLU); within 8 tiles `herd_x` scales 7.4× (1→8). Small shapes are latency-bound by a ~80 µs launch floor. See [`details/RoPE_bf16.md`](details/RoPE_bf16.md).
