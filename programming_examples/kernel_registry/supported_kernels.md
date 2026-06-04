<!---//===- supported_kernels.md ------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# Supported Kernels Registry — LLM Deployment on NPU2

High-level index of the leaf kernels validated for decoder-only LLM deployment on AMD NPU2 (Strix, AIE2P): which kernels are covered, which shapes have been tested, and the best measured performance. Per-kernel detail (datapath, tunable parameters, tolerances, how to reproduce) lives in `details/<KERNEL>.md`.

This is **documentation, not executable code** — it records results produced by the `programming_examples/` kernels, run on real NPU2. See [`README.md`](README.md) for scope and methodology.

**Status legend**: ✅ verified on real NPU2 · ⚠️ pending standalone verification · ❌ broken/missing

> **Scope**: currently **GEMM only** — the registry is built up one verified kernel at a time. Other LLM leaf kernels (GEMV, RMSNorm, RoPE, FlashAttention, SiLU+Mul, Eltwise Add) are on the roadmap in [`README.md`](README.md) and **not yet** included.

---

## Kernels

| Kernel | Detail | Best measured (NPU2, full-chip 8×4) | Status |
|---|---|---|---|
| GEMM (BF16) | [`details/GEMM_bf16.md`](details/GEMM_bf16.md) | **9492 GFLOP/s** (external, 2048×8192×2048) | ✅ |

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
