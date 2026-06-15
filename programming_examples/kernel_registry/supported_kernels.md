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
| GEMM (BF16) | [`details/GEMM_bf16.md`](details/GEMM_bf16.md) | **9492 GFLOP/s** (external, 2048×8192×2048, full-chip 8×4) | ✅ |
| GEMV (BF16) | [`details/GEMV_bf16.md`](details/GEMV_bf16.md) | **32 GFLOP/s** (memory-bound, 16384×2048, herd 8) | ✅ |
| RMSNorm (BF16) | [`details/RMSNorm_bf16.md`](details/RMSNorm_bf16.md) | **18.4 GB/s** (memory-bound, 2048×2048, herd 8) | ✅ |
| FlashAttention (BF16, GQA) | [`details/FlashAttention_bf16.md`](details/FlashAttention_bf16.md) | **1065–1131 GFLOP/s** (2048×2048, dk=64, 32q/8kv causal, full-chip 32 tiles) | ✅ |
| Element-wise Add (BF16) | [`details/EltwiseAdd_bf16.md`](details/EltwiseAdd_bf16.md) | **57.7 GB/s** (memory-bound, N=4194304, herd 8×1) | ✅ |
| SiLU-and-Mul (BF16) | [`details/SiLU_Mul_bf16.md`](details/SiLU_Mul_bf16.md) | **25.1 GB/s** (memory-bound, N=16777216, herd 8×1) | ✅ |
| RoPE (BF16, half-split) | [`details/RoPE_bf16.md`](details/RoPE_bf16.md) | **43.4 GB/s** (memory-bound, 65536×64, herd 8×1) | ✅ |

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

---

## FlashAttention — tested shapes

Fused scaled-dot-product attention (online-softmax FlashAttention) with grouped-query attention and optional causal masking. **Compute-bound** (two matmuls Q@Kᵀ and P@V), so throughput is GFLOP/s. Kernel = `attn_npu2.o`, driven by the **heads-first** harness `attn_npu2.py`; verified on NPU2 across head dim 64/128, MHA & GQA, short & long sequences, causal & non-causal. (A **seq-first** variant `attn_npu2_seqfirst.py` drives the same `.o` for llama-3.2-1B prefill — bit-identical.) **All rows use the one near-unique full-chip config** `lqp=256, num_q_tiles=4, num_heads_per_unroll=2, num_cascade_stages=4` (FA's tile config is determined by the constraints, not tuned — see detail page). Full datapath, tunables, and reproduce commands in [`details/FlashAttention_bf16.md`](details/FlashAttention_bf16.md).

| lq×lk | dk/dv | heads q/kv | causal | dv_chunks | latency | GFLOP/s | mean_rel_L1 | Status |
|---|---|---|---|---|---|---|---|---|
| 2048×2048 | 64/64 | 32/8 | ✓ | 1 | 15.4–16.1 ms | **1065–1116** | 3.9e-2 | ✅ |
| 512×512 | 64/64 | 2/2 | ✗ | 1 | 0.73 ms | 184 | 4.4e-2 | ✅ |
| 512×512 | 64/64 | 12/6 | ✗ | 1 | 1.22 ms | 661 | 4.6e-2 | ✅ |
| 512×512 | 64/64 | 64/8 | ✗ | 1 | 3.79 ms | 1135 | 4.6e-2 | ✅ |
| 512×512 | 128/128 | 32/8 | ✗ | 2 | 4.38 ms | 980 | 4.4e-2 | ✅ |
| 512×512 | 128/128 | 28/4 | ✗ | 2 | 4.05 ms | 928 | 4.4e-2 | ✅ |
| 16384×16384 | 64/64 | 2/2 | ✓ | 1 | 39.6 ms | 1734 | 4.5e-2 | ✅ |
| 16384×16384 | 64/64 | 2/2 | ✗ | 1 | 40.1 ms | **3427** | 5.5e-2 | ✅ |

> All rows measured on NPU2 with the heads-first harness at the default tiling (`lqp=256, num_q_tiles=4, num_heads_per_unroll=2, num_cascade_stages=4` = 32 tiles, full 8×4 array). Accuracy `mean_rel_L1 ≈ 3.9e-2` is ~4× the GEMM tier: FA chains **two BFP16-emulated MMAs** plus a **bf16 online-softmax**, so it is looser than a single matmul (looser than GPU FA's `5e-2` only by the `atol`, not the standard `rtol = 1.6e-2`); accuracy is set by the datapath, not the shape. The **2048, 32q/8kv causal** row is llama-3.2-1B prefill's config (seq-first harness, bit-identical to heads-first — verified `max abs diff = 0`); its GFLOP/s range is run-to-run timing variation. `head_dim=128` rows use `dv_chunks=2`. A separate tunable sweep found only 2 of 8 candidate 32-tile configs place (constraints: columns `num_heads_per_unroll × num_q_tiles ≤ 8`, rows `num_cascade_stages ≤ 4`, `num_heads_per_unroll ≤ 2`). See [`details/FlashAttention_bf16.md`](details/FlashAttention_bf16.md).

---

## Element-wise Add — tested shapes

`c = a + b`, per-element, BF16. The residual adds of llama-3.2-1B (the prefill residual is the fused `o_ffn` inline 2-D variant — same math; this entry measures the **standalone** `eltwise_add`). **Memory-bound** (O(N) streaming, zero arithmetic intensity), so throughput is bandwidth. The **cleanest** kernel in the registry — a single bf16 rounding, no accumulation. Full datapath, herd sweep, and reproduce commands in [`details/EltwiseAdd_bf16.md`](details/EltwiseAdd_bf16.md).

| N | best config (hx/hy/tile_n) | latency | bandwidth | mean_rel_L1 | Status |
|---|---|---|---|---|---|
| 1048576 | 8/1/2048 | 175 µs | 36.0 GB/s | 1.9e-3 | ✅ |
| 2097152 | 8/1/2048 | 277 µs | 45.4 GB/s | 1.9e-3 | ✅ |
| 4194304 (2048×2048) | 8/1/2048 | 437 µs | 57.7 GB/s | 1.9e-3 | ✅ |
| 8388608 | 8/1/2048 | 798 µs | **63.0 GB/s** | 1.9e-3 | ✅ |

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

> `mean_rel_L1 = 1.0e-2` is an order of magnitude above Element-wise Add (1.9e-3): the hardware `aie::tanh<bf16>` LUT approximation plus a chain of bf16 roundings (vs a single rounding for a plain add). Verified element-wise over the full output (no cosine) at `rtol = 1.6e-2, atol = 8e-2` — `atol` covers the worst-case `tanh`-LUT element (`abs_err max = 0.125`); the mean error sits inside `rtol`. Best config `herd_x=8, herd_y=1, tile_n=4096` for every shape (= llama's default): `herd_y>1` fails the shim-channel limit and some `tile_n`/`herd_x` fail a non-monotonic buffer-descriptor limit, so the best config is the fastest one that places. `herd_x` scales 7.6× (1→8). See [`details/SiLU_Mul_bf16.md`](details/SiLU_Mul_bf16.md).

---

## RoPE — tested shapes

Rotary Position Embedding applied to Q/K, **half-split** convention (HuggingFace Llama `rotate_half`), per row; shapes written `rows × head_dim` (rows = n_heads·seq for prefill, n_heads for decode). BF16 in/out, per-element rotation (no reduction, no non-linearity — cos/sin come from a precomputed LUT). **Memory-bound** (streams input + LUT in, output out, ~1 flop/byte), so throughput is bandwidth; the fastest config is `herd_x=8` (all columns, near-linear). The kernel links the **same `rope_halfsplit.cc` (`rope.o`) llama uses** — not the interleaved `rope_lut/`/`rope_sincos/` decoys. Full data, the decoy/provenance note, and reproduce commands are in [`details/RoPE_bf16.md`](details/RoPE_bf16.md).

| (rows×head_dim) | herd (hx/hy) | latency | bandwidth | mean_rel_L1 | Used by | Status |
|---|---|---|---|---|---|---|
| 8×64 | 8/1 | 83 µs | 0.04 GB/s | 2.4e-3 | llama-3.2-1B decode RoPE-K | ✅ |
| 32×64 | 8/1 | 82 µs | 0.15 GB/s | 2.7e-3 | llama-3.2-1B decode RoPE-Q | ✅ |
| 2048×64 | 8/1 | 105 µs | 7.5 GB/s | 2.8e-3 | coverage | ✅ |
| 4096×64 | 8/1 | 118 µs | 13.3 GB/s | 2.8e-3 | coverage | ✅ |
| 16384×64 | 8/1 | 210 µs | 30.0 GB/s | 2.8e-3 | llama-3.2-1B prefill RoPE-K | ✅ |
| 65536×64 | 8/1 | 579 µs | **43.4 GB/s** | 2.8e-3 | llama-3.2-1B prefill RoPE-Q | ✅ |

> `mean_rel_L1 = 2.8e-3` is the second-cleanest in the registry (above Element-wise Add 1.9e-3, below RMSNorm 4.2e-3): a rotation is a few bf16 multiplies and one add/sub per element with **no accumulation** — nothing to amplify error, and `|out| ≈ |x|` so no near-zero blowup. Verified element-wise over the full output (no cosine) at `rtol = 1.6e-2, atol = 5e-2`; bit-identical across all herd configs and shapes (decode rows 8/32 read slightly lower from smaller rotation angles). Best config `herd_x=8, herd_y=1` for every shape: each tile uses 3 shim DMAs (input/LUT in, output out), so `herd_x·herd_y>8` exhausts the shim channels (the herd **cannot fill 32 tiles**, same limit as Element-wise Add / SiLU); within 8 tiles `herd_x` scales 7.4× (1→8). Small shapes are latency-bound by a ~80 µs launch floor. See [`details/RoPE_bf16.md`](details/RoPE_bf16.md).
