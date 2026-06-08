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

> **Scope**: currently **GEMM**, **GEMV**, and **RMSNorm** — the registry is built up one verified kernel at a time. The remaining LLM leaf kernels (RoPE, FlashAttention, SiLU+Mul, Eltwise Add) are on the roadmap in [`README.md`](README.md) and **not yet** included.

---

## Kernels

| Kernel | Detail | Best measured (NPU2) | Status |
|---|---|---|---|
| GEMM (BF16) | [`details/GEMM_bf16.md`](details/GEMM_bf16.md) | **9492 GFLOP/s** (external, 2048×8192×2048, full-chip 8×4) | ✅ |
| GEMV (BF16) | [`details/GEMV_bf16.md`](details/GEMV_bf16.md) | **32 GFLOP/s** (memory-bound, 16384×2048, herd 8) | ✅ |
| RMSNorm (BF16) | [`details/RMSNorm_bf16.md`](details/RMSNorm_bf16.md) | **18.4 GB/s** (memory-bound, 2048×2048, herd 8) | ✅ |

---

## GEMM — tested shapes

`C[M,N] = A[M,K] @ B[K,N]`, shapes written `M×K×N`. GFLOPS is the fastest (external) path at the tile found by sweep; `mean_rel_L1` = `mean|out−ref| / mean|ref|` vs an FP32 reference. Full per-path data, tolerances, and reproduce commands are in [`details/GEMM_bf16.md`](details/GEMM_bf16.md).

| (M×K×N) | best tile (m/kl2/kl1/n) | GFLOPS | mean_rel_L1 | Used by | Status |
|---|---|---|---|---|---|
| 2048×2048×2048 | 64/512/32/128 | 8540 | 9.3e-3 | llama-3.2-1B Q/O proj | ✅ |
| 2048×2048×512 | 64/256/32/128 | 7384 | 9.3e-3 | llama-3.2-1B K/V proj | ✅ |
| 2048×2048×8192 | 64/256/32/128 | 8210 | 9.3e-3 | llama-3.2-1B Gate/Up proj | ✅ |
| 2048×8192×2048 | 64/256/32/128 | **9492** | 9.3e-3 | llama-3.2-1B Down proj | ✅ |
| 512×512×512 | 32/256/32/128 | 1870 | 9.3e-3 | K-sweep | ✅ |
| 1024×1024×1024 | 64/256/32/128 | 6337 | 9.5e-3 | K-sweep | ✅ |
| 4096×4096×4096 | 64/512/32/128 | 9243 | 9.4e-3 | K-sweep | ✅ |

> Measured on NPU2 (RyzenAI-npu4), June 2026, at the fastest tile from an external-path sweep. There are three code-paths (external / direct-codegen f32 / direct-codegen bf16); external is fastest, and external vs direct-f32 are bit-identical in accuracy — see [`details/GEMM_bf16.md`](details/GEMM_bf16.md) for all three.

---

## GEMV — tested shapes

`C[M] = A[M,K] @ B[K]`, shapes written `M×K`. The decode-time (batch = 1) projections of llama-3.2-1B. GEMV is **memory-bound** (reads the whole `M×K` matrix for one length-`M` output), so GFLOPS is far below GEMM; the fastest config is `herd_m=8` (all columns) with the largest L2-legal `tile_m`. Full data, tunables, and reproduce commands are in [`details/GEMV_bf16.md`](details/GEMV_bf16.md).

| (M×K) | best tile (herd_m/tile_m/m_input) | GFLOPS | mean_rel_L1 | Used by | Status |
|---|---|---|---|---|---|
| 2048×2048 | 8/8/8 | 25.5 | 1.6e-9 | llama-3.2-1B Q proj | ✅ |
| 512×2048 | 8/8/8 | 15.5 | 0.0 | llama-3.2-1B K/V proj | ✅ |
| 8192×2048 | 8/8/8 | 31.5 | 2.7e-8 | coverage | ✅ |
| 2048×8192 | 8/2/2 | 31.0 | 0.0 | coverage | ✅ |
| 16384×2048 | 8/8/8 | **30.6** | 0.0 | llama-3.2-1B LM-head | ✅ |

> This plain GEMV is the exact kernel for llama-3.2-1B decode's **Q / K / V projections and LM-head**. The **O / Gate / Up / Down** projections use *fused* cascade variants (GEMV+residual, GEMV+SwiGLU+RMSNorm) — separate kernels, separate registry entries; the 8192×2048 / 2048×8192 rows here are coverage shapes. See [`details/GEMV_bf16.md`](details/GEMV_bf16.md).
> GEMV uses an **FP32 vector accumulate** (not the BFP16-emulated MMA that GEMM uses), so accuracy is effectively exact — `mean_rel_L1 ≤ 2.7e-8`, several shapes bit-identical to the f32 reference, orders of magnitude tighter than BF16 GEMM's ~9e-3.

---

## RMSNorm — tested shapes

`y = x / sqrt(mean(x²) + eps) · weight`, per row; shapes written `M×N` (M = rows / seq, N = emb_dim = reduction axis). The per-layer norm of llama-3.2-1B. **Memory-bound** (streams the whole matrix for an elementwise op), so throughput is reported as bandwidth; the fastest config is `herd_x=8` (all columns, near-linear scaling). Full data, the precision caveat, and reproduce commands are in [`details/RMSNorm_bf16.md`](details/RMSNorm_bf16.md).

| (M×N) | herd_x | latency | bandwidth | mean_rel_L1 | Used by | Status |
|---|---|---|---|---|---|---|
| 2048×2048 | 8 | 911 µs | 18.4 GB/s | 4.2e-3 | llama-3.2-1B prefill RMSNorm | ✅ |

> Follows the **GPU / HuggingFace standard**: the `sum(x²)` reduction is accumulated in **FP32** (matching PyTorch `rms_norm_composite` / HF `LlamaRMSNorm`), giving `mean_rel_L1 = 4.2e-3` — in line with the GEMM tier and passing the canonical bf16 `rtol = 1.6e-2`. (`atol = 5e-2` covers a few large-magnitude bf16 *output*-rounding ULPs, not a reduction relaxation.) The FP32 reduction costs essentially nothing on this memory-bound kernel. See [`details/RMSNorm_bf16.md`](details/RMSNorm_bf16.md).
