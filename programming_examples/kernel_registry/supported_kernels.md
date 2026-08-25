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

> **Scope**: currently **GEMM**, **GEMV**, **RMSNorm**, **LayerNorm**, **FlashAttention**, **Element-wise Add**, **SiLU-and-Mul**, **RoPE**, and **GELU** — the registry is built up one verified kernel at a time. The core LLM leaf kernels are now covered; see [`README.md`](README.md) for the roadmap.

---

## Kernels

| Kernel | Detail | Best measured throughput (NPU2, units per entry) | Status |
|---|---|---|---|
| GEMM (BF16 in, FP32 out) | [`details/GEMM_bf16_in_fp32_out.md`](details/GEMM_bf16_in_fp32_out.md) | **9797 GFLOP/s** (external, 2048×8192×2048, full-chip 8×4) | ✅ |
| GEMM (BF16 in, BF16 out) | [`details/GEMM_bf16_in_bf16_out.md`](details/GEMM_bf16_in_bf16_out.md) | **8898 GFLOP/s** (fused-cast incl. cast, 2048×8192×2048, full-chip 8×4) | ✅ |
| GEMV (BF16) | [`details/GEMV_bf16.md`](details/GEMV_bf16.md) | **32.7 GFLOP/s** (memory-bound, 16384×3072, herd 8/8/8) | ✅ |
| RMSNorm (BF16) | [`details/RMSNorm_bf16.md`](details/RMSNorm_bf16.md) | **24.9 GB/s** (memory-bound, 2048×3072, herd 8) | ✅ |
| LayerNorm (BF16, affine) | [`details/LayerNorm_bf16.md`](details/LayerNorm_bf16.md) | **9.7 GB/s** (memory-bound, 1024×768, herd 8) | ✅ |
| FlashAttention (BF16, GQA) | [`details/FlashAttention_bf16.md`](details/FlashAttention_bf16.md) | **1065–1131 GFLOP/s** (2048×2048, dk=64, 32q/8kv causal, full-chip 32 tiles) | ✅ |
| Element-wise Add (BF16) | [`details/EltwiseAdd_bf16.md`](details/EltwiseAdd_bf16.md) | **57.7 GB/s** (memory-bound, N=4194304, herd 8×1) | ✅ |
| SiLU-and-Mul (BF16) | [`details/SiLU_Mul_bf16.md`](details/SiLU_Mul_bf16.md) | **25.1 GB/s** (memory-bound, N=16777216, herd 8×1) | ✅ |
| RoPE (BF16, half-split) | [`details/RoPE_bf16.md`](details/RoPE_bf16.md) | **56.6 GB/s** (memory-bound, 49152×128, herd 8×1) | ✅ |
| GELU (BF16, tanh approx) | [`details/GELU_bf16.md`](details/GELU_bf16.md) | **27.0 GB/s** (memory-bound, N=8388608, herd 8×2) | ✅ |
| Conv1D (BF16, causal depthwise, k=3) | [`details/Conv1D_bf16.md`](details/Conv1D_bf16.md) | **28.2 GB/s** (memory-bound, 2048×2048, herd 8×1) | ✅ |
| Element-wise Multiply (BF16) | [`details/EltwiseMul_bf16.md`](details/EltwiseMul_bf16.md) | **55.7 GB/s** (memory-bound, N=4194304, herd 8×1) | ✅ |

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
| 256×960×960 | — | **1896** (n80/k320) | — | 9.5e-3 / — | SmolVLA Q/O proj (M=256 pad; emb=960) | ✅ |
| 256×960×320 | — | **1228** (n80/k320) | — | 9.4e-3 / — | SmolVLA K/V proj | ✅ |
| 256×960×2560 | — | **2838** (n128/k320) | — | 9.4e-3 / — | SmolVLA Gate/Up proj | ✅ |
| 256×2560×960 | — | **3425** (n80/k320) | — | 9.4e-3 / — | SmolVLA Down proj | ✅ |
| 1024×768×768 | — | **3798** (n96/k384) | — | 9.5e-3 / — | SmolVLA vision q/k/v/o proj + patch-embed (SigLIP hidden=768) | ✅ |
| 1024×768×3072 | — | **4195** (n128/k256) | 3650 | 9.4e-3 / 1.1e-2 | SmolVLA vision mlp fc1 (deploy drain: faster + more accurate than direct) | ✅ |
| 1024×3072×768 | — | **5790** (n96/k384) | — | 9.4e-3 / — | SmolVLA vision mlp fc2 | ✅ |
| 64×12288×960 | — | **2246** (m16/**n240**/k384, herd 4×4) | — | 9.5e-3 / — | SmolVLA connector proj (vision→LLM) | ✅ |
| 2048×2560×2048 | **fused-cast (m64/k256/n128)** | — | — | 1.0e-2 / — | Gemma3-4B Q proj | ✅ |
| 2048×2048×2560 | **fused-cast (m64/k256/n128)** | — | — | 1.0e-2 / — | Gemma3-4B O proj (decoupled) | ✅ |
| 2048×2560×10240 | **fused-cast (m64/k128/n128)** | — | 4849 | 1.0e-2 / — | Gemma3-4B Gate/Up proj | ✅ |
| 2048×10240×2560 | **fused-cast (m64/k256/n128)** | — | — | 1.0e-2 / — | Gemma3-4B Down proj | ✅ |

> **SmolVLA rows — SmolLM2-360M backbone, emb=960, kv_dim=320, hidden=2560, seq padded 241→256.** All four projections are **small** (M·K·N < 4e9) so `auto`→**drain** (tile_m=32; M=256=32·8 fills exactly one 8-column herd row). emb=960 is not 512-aligned, so N=960/320 use **TILE_N=80** (4·80=320 | 960 and | 320) and N=2560 uses stock TILE_N=128 (4·128·5=2560); K=960/2560 use **tile_k_l2=320** (960/320=3, 2560/320=8). ⚠️ **K=960 with tile_k_l2=160 silently corrupts** (mean_rel_L1≈0.77, not a compile error) — tile_k_l2 ∈ {64,192,320,960} are clean; use 320. All four PASS drain high-precision at 9.4–9.5e-3 (`atol=1.5e-3` near-zero elements are the same harness edge as the Qwen Gate/Up rows, not a datapath failure). bf16-out chosen because the backbone projections feed further matmuls (mirrors every llama/qwen sibling's q/k/v/gate/up/down). GFLOPS are lower than the 2048-row shapes because M=256 is latency-bound, not a datapath difference.

> **SmolVLA vision rows — SigLIP vision encoder + connector, seq=1024, hidden=768.** Five GEMM shapes, all `auto`→**drain** (M·K·N < 4e9). **q/k/v/out_proj (1024×768×768)** and **patch-embedding-as-GEMM** (im2col, identical shape) share one row: hidden=768 not 512-aligned → **TILE_N=96** (4·96=384 | 768), K=768 → tile_k_l2=384. M=1024 with drain tile_m=32 runs **4 M-launch iterations** — the standalone drain harness handles multi-iteration M correctly (PASS 9.465e-3); the mm.o multi-iteration garbage bug is a fused-ELF external-CallOp concern, not this single-launch path. **mlp fc2 (1024×3072×768)** is the fast down-proj-class shape (5790 GFLOPS, clean 9.423e-3). **mlp fc1 (1024×768×3072)**: wide-N up-proj. **Deploy high-prec drain (4195 GFLOPS, 9.4e-3) — both faster AND more accurate than low-prec direct (3650, 1.1e-2)**; it is what `gemm_config()`'s default returns. The standalone harness reports FAIL only on one near-zero-reference element (abs_err 1.953e-3 > high-prec atol=1.5e-3), a tolerance artifact identical to every Qwen Gate/Up, not a datapath error. The direct row is recorded only because it clears the strict atol; do not select it for deployment. **connector proj (64×12288×960)**: M=64 forces tile_m=16 + **herd_m=4** (tile_m·herd_m=64=M), K=12288→tile_k_l2=384. **K=12288 is NOT a placement blocker** — it places and PASSES cleanly at 9.450e-3, **no CPU fallback**. ⚠️ **Use TILE_N=240, not 80**: the tile_n sweep is strongly non-flat (16:3352 µs, 48:1347, 80:890, **240:667–713**) — 240 is **1.33× faster at bit-identical accuracy**, and the old 80 came from the generic "N not 512-aligned → shrink TILE_N" habit, which is wrong for tiny-M/huge-K. **16 tiles (4×4) is the hard ceiling and also the right choice**: the full tile/herd grid admits no 32-tile config, and padding M 64→128 to fill 8×4 was measured **slower** in wall-clock (734 vs 667–713 µs). Root cause is that this GEMM is **weight-DMA-bound** — B is 23.6 MB (93% of traffic, independent of M) moved at ~28 GB/s — so extra tiles buy nothing; halving K halves latency (892→518 µs) and the herd sweep saturates at 16 tiles. See [`details/GEMM_bf16_in_bf16_out.md`](details/GEMM_bf16_in_bf16_out.md). bf16-out because these feed further matmuls.

> **Qwen3-4B rows — emb=2560 (512-aligned, NOT 1024-aligned), q_dim=4096 decoupled (≠emb), kv_dim=1024, hidden=9728=512·19.** All proj N divisible by 4·TILE_N=512, so stock TILE_N=128 HERD_N=4 places. Q/K/V/O/Down PASS high-precision fused-cast directly (max_abs ≤ 1.22e-3, well within high-prec tolerance), K=2560/4096/9728 all use tile_k_l2=256. O proj is **decoupled** (K=q_dim=4096, N=emb=2560), the largest non-square O in the registry. **2048×2560×9728 (Gate/Up) ⚠️**: high-precision fused-cast FAILS at compile (`aie.dma_bd op Stride exceeds [1:1048576] range` on the f32-out B-tile DMA at N=9728 — same large-N class as Qwen2.5-3B's N=11008); the low-precision `direct` path (tile_k_l2=128, TILE_N=64) PASSES at max_abs 2.93e-3 — same Gate/Up low-prec tier-down as every Qwen sibling. Large-K Down (K=9728) does NOT trigger the bug (only large-N does). Qwen3-4B uses the qwen25_3b 5-ELF un-merge (o_res_norm / gate / up / HOST SwiGLU / down_add).

> **Gemma3-4B rows — emb=2560, q_dim=8·256=2048 decoupled, kv_dim=1024 (shares the Qwen3-4B 2048×2560×1024 K/V row), hidden=10240=512·20.** Every N is divisible by 4·TILE_N=512, so stock TILE_N=128 HERD_N=4 places, and all four shapes keep the high-precision fused-cast tier at 1.0e-2. Tiles are **inherited** from the nearest same-K / same-N Qwen3-4B neighbour rather than independently swept, EXCEPT 2048×2560×10240 which was swept and moved to `tile_k_l2=128` (4849 vs 4516 GF/s at 64, same precision) — the `tile_k_l2·N ≤ 1048576` guideline is one step too conservative, 128·10240 compiles and 256 does not. The other three are already at their optimum. Unlike the Qwen2.5 siblings it does **not** have to drop to the low-precision direct path. Gemma3-4B's prefill uses an 8-ELF split (rms_qkv_qknorm_rope / global + sliding-window flash attention / o_norm_res_norm / gate / up / gelu_mul / down_norm_add / lm_head_gemv).

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
| 896×896 | 8/16/16 | 9.5 | 0.0 | Qwen2.5-0.5B decode Q/O proj | ✅ |
| 128×896 | 8/16/16 | 2.7 | 0.0 | Qwen2.5-0.5B decode K/V proj | ✅ |
| 4864×896 | 8/16/16 | 20.3 | 0.0 | Qwen2.5-0.5B decode Gate/Up proj | ✅ |
| 896×4864 | 8/4/4 | 26.3 | 0.0 | Qwen2.5-0.5B decode Down proj | ✅ |
| 16384×896 | 8/16/16 | 28.5 | 7.2e-12 | Qwen2.5-0.5B LM-head (per-partition) | ✅ |
| 1536×1536 | 8/16/16 | 22.5 | 0.0 | Qwen2.5-1.5B decode Q/O proj | ✅ |
| 256×1536 | 8/16/16 | 7.5 | 0.0 | Qwen2.5-1.5B decode K/V proj | ✅ |
| 8960×1536 | 8/16/16 | 25.0 | 1.7e-9 | Qwen2.5-1.5B decode Gate/Up proj | ✅ |
| 1536×8960 | 8/2/2 | 30.6 | 2.2e-6 | Qwen2.5-1.5B decode Down proj | ✅ |
| 16384×1536 | 8/16/16 | 32.6 | 2.3e-8 | Qwen2.5-1.5B LM-head (per-partition) | ✅ |
| 1024×2048 | 8/8/8 | 21.0 | 0.0 | Qwen3-1.7B decode K/V proj | ✅ |
| 6144×2048 | 8/8/8 | 30.8 | 0.0 | Qwen3-1.7B decode Gate/Up proj + LFM2-1.2B decode conv in_proj (3·conv_dim) | ✅ |
| 2048×6144 | 8/4/4 | 31.4 | 0.0 | Qwen3-1.7B decode Down proj | ✅ |
| 256×2048 | 8/8/8 | 10.1 | 0.0 | Qwen2.5-3B decode K/V proj | ✅ |
| 11008×2048 | 8/8/8 | 31.9 | 7.9e-8 | Qwen2.5-3B decode Gate/Up proj | ✅ |
| 2048×11008 | 8/2/1 | 27.6 | 0.0 | Qwen2.5-3B decode Down proj (K=11008 L1-bound → m_input=1) | ✅ |
| 4096×2560 | 8/8/8 | 30.1 | 7.3e-7 | Qwen3-4B decode Q proj | ✅ |
| 1024×2560 | 8/8/8 | 22.6 | 0.0 | Qwen3-4B decode K/V proj | ✅ |
| 2560×4096 | 8/4/4 | 29.4 | 0.0 | Qwen3-4B decode O proj (decoupled K=4096 → tile_m=4 m_input=4 to fit L2) | ✅ |
| 9728×2560 | 8/8/8 | 32.6 | 0.0 | Qwen3-4B decode Gate/Up proj | ✅ |
| 2560×9728 | 8/2/2 | 31.0 | 2.3e-10 | Qwen3-4B decode Down proj — standalone (model runs this on HOST: stitched-ELF L1 overflow) | ✅ |
| 16384×2560 | 8/8/8 | 30.2 | 4.2e-7 | Qwen3-4B LM-head (per-partition, K=2560) | ✅ |
| 1024×3072 | 8/8/8 | 24.5 | 0.0 | coverage (K=3072) | ✅ |
| 3072×1024 | 8/16/16 | 22.7 | 4.9e-10 | coverage (M=3072, K=1024) | ✅ |
| 3072×3072 | 8/8/8 | 30.4 | 1.8e-9 | coverage (K=3072) | ✅ |
| 3072×8192 | 8/2/2 | 29.4 | 0.0 | coverage (K=8192) | ✅ |
| 8192×3072 | 8/8/8 | 32.2 | 1.1e-7 | coverage (M=8192, K=3072) | ✅ |
| 16384×3072 | 8/8/8 | 32.6 | 3.4e-7 | LM-head coverage (K=3072) | ✅ |

> **Qwen3-4B GEMV.** Decode projections bit-identical (0.0) to the f32 ref. emb=2560 K, q_dim=4096 decoupled. O proj is **decoupled** (M=emb=2560, K=q_dim=4096) — at K=4096 the full `[m_input, K]` A tile constrains L2, so `tile_m=4, m_input=4` (vs the stock 8/8) keeps A=tile_m·herd_m·K·2 ≤ 512 KiB. Down proj is **2560×9728** (M=emb=2560, K=intermediate=9728); the standalone harness places at `8/2/2` (31.0 GFLOPS, 2.3e-10), but in the model it runs on **HOST** (stitched-ELF L1 overflow, same as Qwen2.5-3B's K=11008). LM-head reuses the shared 19-partition vocab=151936 datapath at K=2560 per partition (16384×2560 row, 30.2 GFLOPS).

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
| 2048×1024 | 8 | 407 µs | 20.6 GB/s | 4.3e-3 | Qwen3-0.6B prefill RMSNorm | ✅ |
| 2048×128 | 8 | 155 µs | 6.8 GB/s | 4.6e-3 | Qwen3-0.6B + Qwen3-1.7B QK-norm (per-head, N=head_dim) | ✅ |
| 2048×64 | 8 | 137 µs | 3.8 GB/s | 4.7e-3 | LFM2-1.2B QK-norm (per-head, N=head_dim=64) | ✅ |
| 2048×896 | 8 | 398 µs | 18.4 GB/s | 4.2e-3 | Qwen2.5-0.5B prefill RMSNorm | ✅ |
| 2048×1536 | 8 | 570 µs | 22.1 GB/s | 4.3e-3 | Qwen2.5-1.5B prefill RMSNorm | ✅ |
| 2048×2560 | 8 | 867 µs | 24.2 GB/s | 4.2e-3 | Qwen3-4B prefill RMSNorm | ✅ |
| 2048×3072 | 8 | 1012 µs | **24.9 GB/s** | 4.2e-3 | Llama-3.2-3B prefill RMSNorm | ✅ |
| 256×960 | 8 | 147 µs | 6.7 GB/s | 4.2e-3 | SmolVLA prefill RMSNorm (emb=960) | ✅ |

> **Qwen3-0.6B QK-norm (2048×128)** is per-head RMSNorm over `head_dim=128` (Qwen3-specific q_norm/k_norm) — the same weighted-RMSNorm kernel with a small `N=128` reduction axis; verified PASS at 4.6e-3, confirming the kernel handles a 128-wide reduction. (Harness `eps = 1e-5`; Qwen3 `eps = 1e-6` — the difference is negligible vs the bf16 datapath error.)

> **LFM2-1.2B QK-norm (2048×64)** is the same per-head construct at `head_dim = 64` — the first head_dim=64 QK-norm coverage (LFM2's attention layers carry `q_layernorm`/`k_layernorm` like Qwen3, but at llama's head dim). Verified PASS at 4.7e-3, marginally above the 2048×2048 row's 4.2e-3: a shorter reduction axis gives each rounding proportionally more weight. At 137 µs this shape is on the ~80–155 µs small-shape launch floor, so the 3.8 GB/s figure is launch-bound and not a bandwidth result — compare latency, not bandwidth, against the wider rows.

> Follows the **GPU / HuggingFace standard**: the `sum(x²)` reduction is accumulated in **FP32** (matching PyTorch `rms_norm_composite` / HF `LlamaRMSNorm`), giving `mean_rel_L1 = 4.2e-3` — in line with the GEMM tier and passing the canonical bf16 `rtol = 1.6e-2`. (`atol = 5e-2` covers a few large-magnitude bf16 *output*-rounding ULPs, not a reduction relaxation.) The FP32 reduction costs essentially nothing on this memory-bound kernel. See [`details/RMSNorm_bf16.md`](details/RMSNorm_bf16.md).

---

## LayerNorm — tested shapes

`y = (x − mean) / sqrt(var + eps) · weight + bias`, per row; shapes written `M×N` (M = rows / patches, N = hidden = reduction axis). Unlike RMSNorm, LayerNorm **subtracts the mean** and has an **affine bias** (γ *and* β). The vision-encoder (SigLIP) norms of SmolVLA. **Memory-bound** (streams the whole matrix for an elementwise op), so throughput is reported as bandwidth; the fastest config is `herd_x=8` (all columns, near-linear scaling). Full data, the precision note, and reproduce commands are in [`details/LayerNorm_bf16.md`](details/LayerNorm_bf16.md).

| (M×N) | herd_x | latency | bandwidth | mean_rel_L1 | abs_err max | Used by | Status |
|---|---|---|---|---|---|---|---|
| 1024×768 | 8 | 324 µs | 9.7 GB/s | 4.4e-3 | 1.9e-1 | SmolVLA vision (SigLIP) layer_norm1/2 + post_layernorm (hidden=768) | ✅ |

> Follows the **GPU / HuggingFace standard** (PyTorch / HF `nn.LayerNorm`): **both** reductions — `sum(x)` for the mean *and* `sum((x−mean)²)` for the variance — are accumulated in **FP32**, giving `mean_rel_L1 = 4.4e-3`, in the same bf16 tier as RMSNorm (4.2e-3) and GEMM (~9e-3), and passing the canonical bf16 `rtol = 1.6e-2`. The `atol = 6e-2` (vs RMSNorm's 5e-2) covers LayerNorm's one extra bf16 *output*-rounding step — the bias add in `(x−mean)·rstd·weight + bias` — not a reduction relaxation. eps = 1e-5 (SigLIP uses 1e-6; the difference is negligible vs the bf16 datapath error). See [`details/LayerNorm_bf16.md`](details/LayerNorm_bf16.md).

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
| 1024×1024 | 64/64 | 12/12 | ✗ | 1 | 2.36–2.61 ms | 1235–1366 | 4.8e-2 | ✅ SmolVLA vision self-attn (SigLIP) |
| 512×512 | 128/128 | 32/8 | ✗ | 2 | 4.38 ms | 980 | 4.4e-2 | ✅ |
| 512×512 | 128/128 | 28/4 | ✗ | 2 | 4.05 ms | 928 | 4.4e-2 | ✅ |
| 16384×16384 | 64/64 | 2/2 | ✓ | 1 | 39.6 ms | 1734 | 4.5e-2 | ✅ |
| 16384×16384 | 64/64 | 2/2 | ✗ | 1 | 40.1 ms | **3427** | 5.5e-2 | ✅ |
| 2048×2048 | 128/128 | 16/8 | ✓ | 2 | 17.6 ms | 979 | 3.8e-2 | ✅ |
| 2048×2048 | 64/64 | 14/2 | ✓ | 1 | 7.27 ms | 1035 | 3.8e-2 | ✅ |
| 2048×2048 | 128/128 | 12/2 | ✓ | 2 | 14.5 ms | 891 | 3.8e-2 | ✅ |
| 2048×2048 | 128/128 | 24/8 | ✓ | 2 | 25.9 ms | 995 | 3.8e-2 | ✅ |
| 2048×2048 | 128/128 | 32/8 | ✓ | 2 | 35.0 ms | 983 | 3.8e-2 | ✅ |

> **Qwen3-0.6B prefill attention** (`head_dim = 128`, 16q/8kv GQA, causal, lq=lk=2048): verified PASS at mean_rel_L1 = 3.8e-2 (full-output check, rtol 1.6e-2 / atol 1e-1) with the default full-chip config (`lqp=256, num_q_tiles=4, num_heads_per_unroll=2, num_cascade_stages=4`, `dv_chunks=2` for head_dim=128). Note: head_dim=128 FA has been flaky (hang/NaN) on some NPU2 setups; this run completed cleanly, and Qwen3-0.6B prefill can also fall back to CPU attention (`cpu_attn`) if a deployment hits the hang.

> **Qwen2.5-1.5B prefill attention** (`head_dim = 128`, 12q/2kv GQA, causal, lq=lk=2048): verified PASS at mean_rel_L1 = 3.83e-2 (full-output check, rtol 1.6e-2 / atol 1e-1) with the default full-chip config (`lqp=256, num_q_tiles=4, num_heads_per_unroll=2, num_cascade_stages=4`, `dv_chunks=2` for head_dim=128). head_dim=128 FA has been flaky (hang/NaN) on some NPU2 setups; this run completed cleanly, and prefill can fall back to CPU attention (`cpu_attn`) if a deployment hits the hang.

> **Qwen2.5-0.5B prefill attention** (`head_dim = 64`, 14q/2kv GQA, causal, lq=lk=2048): verified PASS at mean_rel_L1 = 3.83e-2 with the default full-chip config (`lqp=256, lkp=64, num_q_tiles=4, num_heads_per_unroll=2, num_cascade_stages=4`, `dv_chunks=1` for head_dim=64). head_dim=64 has no hang risk. Prefill can also fall back to CPU attention (`cpu_attn`).

> **SmolVLA vision encoder (SigLIP) self-attention** (`head_dim = 64`, 12q/12kv **MHA**, **non-causal / bidirectional**, lq=lk=1024): verified PASS at mean_rel_L1 = 4.8e-2 (full-output check, rtol 1.6e-2 / atol 1e-1) with the default full-chip config (`lqp=256, lkp=64, num_q_tiles=4, num_heads_per_unroll=2, num_cascade_stages=4`, `dv_chunks=1`). Being **12 heads (even)** lets `num_heads_per_unroll=2` fill the full 8×4 array; forcing `num_heads_per_unroll=1` (as the odd-15-head decoder backbone must) runs the identical numerics on the half array at **864 GFLOP/s vs 1235–1366** — a measured ~1.58× throughput from filling the chip. No mask (bidirectional), head_dim=64 → no hang risk.

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
| 1835008 (2048×896) | 8/1/2048 | 243 µs | 45.3 GB/s | 1.9e-3 | ✅ (Qwen2.5-0.5B residual, seq·emb) |
| 3145728 (2048×1536) | 8/1/2048 | 364 µs | 51.9 GB/s | 1.9e-3 | ✅ (Qwen2.5-1.5B residual, seq·emb) |
| 5242880 (2048×2560) | 8/1/2048 | 516 µs | 61.0 GB/s | 1.9e-3 | ✅ (Qwen3-4B residual, seq·emb) |
| 6291456 (2048×3072) | 8/1/2048 | 614 µs | 61.4 GB/s | 1.9e-3 | ✅ (Llama-3.2-3B residual, seq·emb) |
| 245760 (256×960) | 8/1/1920 | 112 µs | 13.2 GB/s | 1.9e-3 | ✅ (SmolVLA residual, seq·emb; tile_n=2048 corrupts → use 1920) |

> `mean_rel_L1 = 1.9e-3` is the lowest in the registry — `c=a+b` rounds each output once (matching `torch.add` bf16: f32 sum, single round, no accumulation), bit-identical across all configs and `N`. Best config `herd_x=8, herd_y=1` for every shape: the 3-DMA-per-tile shim-channel limit caps the herd at one 8-column row (**cannot fill 32 tiles** — `herd_y>1` fails to place), but within that `herd_x` scales near-linearly (9→57.7 GB/s as herd_x 1→8). Highest bandwidth in the registry (pure streaming). See [`details/EltwiseAdd_bf16.md`](details/EltwiseAdd_bf16.md).

---

## Element-wise Multiply — tested shapes

`c = a · b`, per-element, BF16. The two gates of LFM2's `Lfm2ShortConv` (`h = B·x`, then `y = C·h`). **Memory-bound** (O(N) streaming, zero arithmetic intensity), so throughput is bandwidth. Distinct from SiLU-and-Mul, which applies a transcendental to one operand before multiplying, and from Element-wise Add — a product carries more relative error than a sum. Harness `programming_examples/eltwise_mul` (the `air.api` DSL, mirroring `eltwise_add`). Full datapath and reproduce commands in [`details/EltwiseMul_bf16.md`](details/EltwiseMul_bf16.md).

| N | (as 2-D) | config (hx/hy/tile_n) | latency | bandwidth | mean_rel_L1 | Used by | Status |
|---|---|---|---|---|---|---|---|
| 1048576 | 1024×1024 | default | 180.5 µs | 34.9 GB/s | 2.73e-3 | coverage | ✅ |
| 4194304 | 2048×2048 (seq·conv_dim) | default | **451.6 µs** | **55.7 GB/s** | 2.74e-3 | LFM2-1.2B ShortConv gates (×2 per conv layer, 10 of 16 layers) | ✅ |

> Measured on an **idle** NPU2. Element-wise Multiply lands within **2%** of Element-wise Add at the same N (451.6 vs 441.8 µs) — expected, since the two share a DSL body and differ only in the arithmetic op, and both are DMA-bound. An earlier attempt measured 17 ms for this kernel while an unrelated LLM server held the device; that number was discarded, not recorded. The `eltwise_add` control (441.8 µs vs its recorded 437 µs) is what certifies the box was clean.

> `mean_rel_L1 = 2.7e-3` vs Element-wise Add's `1.9e-3` is the expected ordering: both are single-rounding ops in the cleanest tier, but a bf16 product carries more relative error than a bf16 sum. Verified element-wise over the full output (no cosine) at `rtol = 1.6e-2, atol = 5e-2`.

> **Provenance check**: this kernel was derived from `eltwise_add`, so the emitted module was grepped to confirm it contains **1 `arith.mulf` and 0 `arith.addf`** — i.e. it is genuinely a multiply, not an add that survived the copy. Worth repeating for any kernel cloned from a sibling; a same-shaped decoy passes every shape test.

---


## SiLU-and-Mul — tested shapes

`out = SiLU(gate) · up`, `SiLU(x) = x·sigmoid(x)`, per-element, BF16. The SwiGLU activation of llama-3.2-1B prefill FFN (the standalone `silu_and_mul` is measured; llama runs the bit-identical 2-D `build_module_2d` variant). **Memory-bound** (O(N) streaming, ~1 op/byte), so throughput is bandwidth. sigmoid is computed via the hardware `aie::tanh` (`0.5·(1+tanh(g/2))`); the precision is the "bf16 + one transcendental" tier. Full datapath, sweep, and reproduce commands in [`details/SiLU_Mul_bf16.md`](details/SiLU_Mul_bf16.md).

| N | (as 2-D) | best config (hx/hy/tile_n) | latency | bandwidth | mean_rel_L1 | abs_err max | Status |
|---|---|---|---|---|---|---|---|
| 2097152 | — | 8/1/4096 | 569 µs | 22.1 GB/s | 1.0e-2 | 0.125 | ✅ |
| 4194304 | 2048×2048 | 8/1/4096 | 1052 µs | 23.9 GB/s | 1.0e-2 | 0.125 | ✅ |
| 8388608 | — | 8/1/4096 | 2247 µs | 22.4 GB/s | 1.0e-2 | 0.125 | ✅ |
| 16777216 | 2048×8192 | 8/1/4096 | 4016 µs | **25.1 GB/s** | 1.0e-2 | 0.125 | ✅ |
| 6291456 | 2048×3072 (seq·hidden) | 8/1/4096 | 1771 µs | 21.3 GB/s | 1.0e-2 | 0.125 | ✅ |
| 9961472 | 2048×4864 (seq·hidden) | 8/1/4096 | 2489 µs | 24.0 GB/s | 1.0e-2 | 0.125 | ✅ |
| 18350080 | 2048×8960 (seq·hidden) | 8/1/4096 | 4933 µs | 22.3 GB/s | 1.0e-2 | 0.188 | ✅ |
| 12582912 | 2048×6144 (seq·hidden) | 8/1/4096 | 3041 µs | 24.8 GB/s | 1.0e-2 | 0.125 | ✅ (Qwen3-1.7B SwiGLU) |
| 19922944 | 2048×9728 (seq·hidden) | 8/1/4096 | 5077 µs | 23.5 GB/s | 1.0e-2 | 0.125 | ✅ (Qwen3-4B SwiGLU) |
| 22544384 | 2048×11008 (seq·hidden) | 8/1/4096 | 5694 µs | 23.8 GB/s | 1.0e-2 | 0.188 | ✅ (Qwen2.5-3B SwiGLU) |
| 655360 | 256×2560 (seq·hidden) | 8/1/4096 | 247 µs | 15.9 GB/s | 1.0e-2 | 0.125 | ✅ (SmolVLA SwiGLU) |

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
| 28672×64 | 8/1 | 303 µs | 36.4 GB/s | 2.8e-3 | Qwen2.5-0.5B prefill RoPE-Q (rows=n_heads·seq=14·2048) | ✅ |
| 16384×64 | 8/1 | 210 µs | 30.0 GB/s | 2.8e-3 | llama-3.2-1B prefill RoPE-K | ✅ |
| 65536×64 | 8/1 | 579 µs | 43.4 GB/s | 2.8e-3 | llama-3.2-1B prefill RoPE-Q | ✅ |
| 32768×128 | 8/1 | 477 µs | 52.8 GB/s | 2.8e-3 | Qwen3-0.6B + Qwen3-1.7B + Qwen2.5-3B prefill RoPE-Q (rows=n_heads·seq=16·2048) | ✅ |
| 16384×128 | 8/1 | 285 µs | 44.2 GB/s | 2.8e-3 | Qwen3-0.6B + Qwen3-1.7B prefill RoPE-K (rows=n_kv_heads·seq=8·2048) | ✅ |
| 49152×128 | 8/1 | 667 µs | **56.6 GB/s** | 2.8e-3 | Llama-3.2-3B prefill RoPE-Q (rows=n_heads·seq=24·2048) | ✅ |
| 24576×128 | 8/1 | 380 µs | 49.7 GB/s | 2.8e-3 | Qwen2.5-1.5B prefill RoPE-Q (rows=n_heads·seq=12·2048) | ✅ |
| 4096×128 | 8/1 | 149 µs | 21.1 GB/s | 2.8e-3 | Qwen2.5-1.5B + Qwen2.5-3B prefill RoPE-K (rows=n_kv_heads·seq=2·2048) | ✅ |
| 256×64 | 8/1 | 80 µs | 1.2 GB/s | 2.8e-3 | SmolVLA RoPE-Q/K datapath (head_dim=64, seq 241→256; θ via host LUT) | ✅ |

> **Qwen3-0.6B uses `head_dim = 128`** (vs llama's 64) — the two rows above are the first registry coverage of `head_dim = 128`; same half-split `rope_halfsplit.cc` kernel, verified PASS at 2.8e-3 (accuracy unchanged, set by the datapath not the head dim).

> `mean_rel_L1 = 2.8e-3` is the second-cleanest in the registry (above Element-wise Add 1.9e-3, below RMSNorm 4.2e-3): a rotation is a few bf16 multiplies and one add/sub per element with **no accumulation** — nothing to amplify error, and `|out| ≈ |x|` so no near-zero blowup. Verified element-wise over the full output (no cosine) at `rtol = 1.6e-2, atol = 5e-2`; bit-identical across all herd configs and shapes (decode rows 8/32 read slightly lower from smaller rotation angles). Best config `herd_x=8, herd_y=1` for every shape: each tile uses 3 shim DMAs (input/LUT in, output out), so `herd_x·herd_y>8` exhausts the shim channels (the herd **cannot fill 32 tiles**, same limit as Element-wise Add / SiLU); within 8 tiles `herd_x` scales 7.4× (1→8). Small shapes are latency-bound by a ~80 µs launch floor. See [`details/RoPE_bf16.md`](details/RoPE_bf16.md).

---

## GELU-and-tanh — tested shapes

`out = GELU(x)` in the **tanh approximation** (`out = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`), per-element, BF16. The SmolVLA SigLIP vision encoder MLP activation (applied to the `fc1` output, between `fc1` and `fc2`). **Memory-bound** (O(N) streaming, ~1.5 op/byte), so throughput is bandwidth. tanh is the hardware `__builtin_aie2p_tanh` (direct-codegen, no external `.o`); the precision is the "bf16 + one transcendental" tier. **Unlike SiLU-and-Mul / Element-wise Add, GELU is single-input (2 DMAs/tile), so `herd_y>1` places — the best config is `herd_x=8, herd_y=2` (16 tiles), twice the 8-tile ceiling of the 3-DMA elementwise kernels.** Full datapath, sweep, and reproduce commands in [`details/GELU_bf16.md`](details/GELU_bf16.md).

| N | (as 2-D) | best config (hx/hy/tile_n) | latency | bandwidth | mean_rel_L1 | abs_err max | Used by | Status |
|---|---|---|---|---|---|---|---|---|
| 1048576 | — | 8/2/4096 | 235 µs | 17.8 GB/s | 8.4e-3 | 1.56e-2 | coverage | ✅ |
| 2097152 | — | 8/2/4096 | 379 µs | 22.1 GB/s | 8.4e-3 | 1.56e-2 | coverage | ✅ |
| 3145728 | 1024×3072 | 8/2/4096 | 522 µs | 24.1 GB/s | 8.4e-3 | 1.56e-2 | SmolVLA vision MLP GELU (seq 1024 · intermediate 3072) | ✅ |
| 4194304 | 2048×2048 | 8/2/4096 | 672 µs | 25.0 GB/s | 8.4e-3 | 1.56e-2 | coverage | ✅ |
| 8388608 | — | 8/2/4096 | 1244 µs | **27.0 GB/s** | 8.4e-3 | 1.56e-2 | coverage | ✅ |

> The 3145728 row is SmolVLA's SigLIP vision MLP activation scale (seq 1024 · intermediate 3072); the `fc1` GEMM that produces its input is the `1024×768×3072` GEMM row. The reference is the **tanh** approximation (GELUTanh), matching SmolVLA / HF SigLIP — not the exact erf-GELU (which would add a ~1e-3 systematic bias). All shapes use the same best config.

> `mean_rel_L1 = 8.4e-3` is the "bf16 + one transcendental" tier — below SiLU-and-Mul (1.0e-2), above RMSNorm (4.2e-3): the hardware `tanh` LUT approximation plus a chain of bf16 roundings. Slightly cleaner than SiLU because the tanh argument is scaled down and there is no `0.5·g·u` amplification (GELU's `abs_err max = 0.0156` vs SiLU's 0.125). Verified element-wise over the full output (no cosine) at `rtol = 1.6e-2, atol = 5e-2` (tighter than SiLU's 8e-2). **Best config `herd_x=8, herd_y=2, tile_n=4096` (16 tiles) for every shape**: GELU's single input (2 DMAs/tile) lets the herd use a second row where the 3-DMA elementwise kernels (Element-wise Add, SiLU) cannot; `herd_x·herd_y=32` (8×4) still does not place. `herd_x` scales 7.6× (1→8), plus ~1.8× for the second row. See [`details/GELU_bf16.md`](details/GELU_bf16.md).

---

## Conv1D — tested shapes

`y[t,c] = Σ_j w[j,c] · x[t+j, c]`, causal depthwise 1-D convolution with kernel size **k=3** — the convolution inside LFM2's `Lfm2ShortConv` operator. Depthwise (per-channel, no cross-channel mixing), so the channel axis is the vectorization axis. Shapes written `seq × channels`. **Memory-bound** (O(N) streaming, ~3 FMA per element), so throughput is bandwidth. Kernel = `conv1d_depthwise.o`, harness `programming_examples/conv1d_depthwise`. Full datapath, the pre-padding convention, the placement sweep, and reproduce commands in [`details/Conv1D_bf16.md`](details/Conv1D_bf16.md).

| (seq×channels) | k | config (hx/hy/tile_s) | latency | bandwidth | mean_rel_L1 | Used by | Status |
|---|---|---|---|---|---|---|---|
| 2048×2048 | 3 | **8/1/8** | **594.6 µs** | **28.2 GB/s** | 2.8e-3 | LFM2-1.2B ShortConv (10 of 16 layers) | ✅ |
| 2048×2048 | 3 | 8/1/4 | 599.5 µs | 28.0 GB/s | 2.8e-3 | tile_s sweep | ✅ |
| 2048×2048 | 3 | 8/1/16 | 599.4 µs | 28.0 GB/s | 2.8e-3 | tile_s sweep | ✅ |
| 2048×2048 | 3 | 8/1/32 | 621.4 µs | 27.0 GB/s | 2.8e-3 | tile_s sweep | ✅ |
| 2048×2048 | 3 | 4/1/16 | 1355 µs | 12.4 GB/s | 2.8e-3 | herd_x scaling point | ✅ |
| 2048×2048 | 3 | 1/1/4 | 4367 µs | 3.8 GB/s | 2.8e-3 | single-tile datapath | ✅ |
| 8×2048 | 3 | 8/1/8 | — | — | 2.8e-3 | LFM2-1.2B short-sequence / decode-scale | ✅ |

> Measured on an **idle** NPU2, validated by a control: `eltwise_add` at N=4194304 reproduced **441.8 µs** against its recorded 437 µs (1%). An earlier set of figures taken while an unrelated LLM server held the device was discarded rather than recorded — see "Measurement hazard" in the detail page.

> **`tile_s` barely matters (594–621 µs, 4% spread); `herd_x` is everything.** `herd_x` scales **7.3×** from 1→8 (4367 → 597 µs), the signature of a memory-bound streaming kernel. Bandwidth at the best config: read `(2050+3)·2048` + write `2048·2048` elements × 2 B = 16.8 MB in 594.6 µs.

> ⚠️ **`herd_x = 2` is MEASURED-BROKEN — and silently so.** At `tile_s=8` it compiles cleanly and returns **wrong results** (`mean_rel_L1 = 5.0e-1` vs the correct 2.8e-3); at `tile_s=4` it fails inside aircc. This is **not** the L1 budget: `2/1/8` needs only 42 KB of the 64 KB L1, while `herd_x=1` (52 KB) and `herd_x=4` (37 KB) are both correct — so the bad axis is `herd_x=2` itself, non-monotonically. `conv1d_depthwise.py` asserts `herd_x != 2`. Use `herd_x ∈ {1, 4, 8}`.

> ⚠️ **`herd_y > 1` is SINGLE-SHOT ONLY — it passes `make run` and then deadlocks.** `8×2` places and a *single* invocation is numerically correct (2.813e-3, identical to `herd_y=1`), but the **second invocation times out** (`ERT_CMD_STATE_TIMEOUT`), reproduced with as few as 2 iterations. Any real deployment calls this kernel in a loop (10× per prefill for LFM2-1.2B), so `herd_y>1` is unusable. **A one-shot `make run` sweep would have concluded `herd_y=2` is fine and shipped a deadlock** — gate herd changes with a *repeated*-invocation run. Suspected cause is the loop-invariant weight DMA hoisted outside the sequence loop, whose producer does not re-fire for the extra herd row on re-invocation; that mechanism is **unproven**.

> 📝 **History**: an earlier revision of this page claimed `herd_y=2` *hangs immediately* and inferred a 3-shim-DMA / one-8-column-row ceiling shared with Element-wise Add. That was wrong on both counts — it was measured under NPU contention, and `herd_y=2` in fact places and computes correctly once. The correct statement is the re-invocation deadlock above. Kept as a caution: a hardware-sounding ceiling is exactly the kind of claim that gets copied forward without re-measurement.

> **Causality is expressed by pre-padding, not masking.** The input carries `seq + 2` rows and the output `seq`: input row `t` is the sample at original position `t − 2`, so it is the oldest sample feeding `y[t]` and pairs with tap 0 (oldest-first, matching `nn.Conv1d` cross-correlation over a left-padded input). The two leading rows are the **conv state** — zeros at sequence start (prefill), or the carried tail of the previous chunk (decode). **Prefill and decode are the same kernel with a different pad**; there is no separate decode variant. Weights are passed **tap-major `(3, C)`** so each tap's channel slice is contiguous for unit-stride vector loads (HF stores them channel-major as `(C, 1, 3)`; the host transposes once at load).

> `mean_rel_L1 = 2.8e-3` is the **cleanest tier**, tied with RoPE and just above Element-wise Add (1.9e-3): three bf16 products accumulated in **FP32** (`aie::mul` + 2× `aie::mac` into an `accfloat` accumulator) with a single bf16 rounding on store — no transcendental, and a 3-term reduction too short to accumulate meaningful error. Verified element-wise over the full output (no cosine) at `rtol = 1.6e-2, atol = 5e-2`. Accuracy is bit-identical across every placeable config, as expected.

> ⚠️ **`herd_y > 1` is UNTESTED — an earlier claim that it hangs has been RETRACTED.** A previous revision of this page stated, as a *measured* result, that `herd_y=2` hangs and that Conv1D therefore sits under the same 3-shim-DMA / one-8-column-row ceiling as Element-wise Add and SiLU-and-Mul. That is withdrawn: the `ERT_CMD_STATE_TIMEOUT` behind it occurred while an unrelated **LLM server was holding the NPU**, and contention explains a submission timeout at least as well as the herd shape does. The two cannot be separated without an idle device. Conv1D *does* issue 3 shim DMAs per tile, so the ceiling is plausible — but plausible is not measured, and the registry's own methodology warns specifically against inheriting a sibling kernel's herd cap. **Sweep `herd_y` on a quiet box before recording any ceiling.** All shipped configs are `herd_y=1`, which is verified. `tile_s ∈ {4, 8, 16, 32}` all PASS at `herd_x=8`.

> ⚠️ **L1 over-allocation is SILENT at `herd_x=4`.** The three live L1 buffers are `(tile_s+2)×tile_c` (halo window) + `3×tile_c` (weights) + `tile_s×tile_c` (output). `herd_x=4, tile_s=32` needs 70.6 KB against a 64 KB compute-tile L1 and **compiles cleanly, then returns wrong results**, while the larger `herd_x∈{1,2}` overflows fail inside aircc — so the failure mode is *non-monotonic* in overflow size. `conv1d_depthwise.py` carries an explicit `L1_BYTES` assert to reject these up front; do not remove it. This is the same "silent-corruption tile config the builder does not assert" class as the GEMM `N % (tile_n × herd_n)` trap.

> ⚠️ **A hanging config can wedge the whole device, not just its own context.** During the placement sweep a config hit `ERT_CMD_STATE_TIMEOUT`, after which *every* NPU submission — including previously-passing configs of this kernel, untouched upstream examples, and `xrt-smi validate --run latency` itself — failed with `DRM_IOCTL_AMDXDNA_EXEC_CMD IOCTL failed (err=-5)`. A `modprobe -r amdxdna` reload did **not** recover it; the kernel log showed the failure was one layer lower (`aie2_smu_start: Access power failed` → `amdxdna_probe: Hardware init failed`), and only a reboot cleared it. **When sweeping a new kernel, re-run a known-good control after any TIMEOUT before trusting subsequent results**, or a wedged device will be misread as a long run of genuine placement failures — that is exactly what happened here.

> ⚠️ **Before recording any perf number or diagnosing any hang, check who else holds the NPU.** This kernel's sweep was run against a device already owned by an unrelated LLM server, which produced both a spurious "herd_y hangs" conclusion and a full set of meaningless latencies. `for p in $(ls /proc | grep -E '^[0-9]+$'); do ls -l /proc/$p/fd 2>/dev/null | grep -q accel && echo "$p $(cat /proc/$p/comm)"; done`
