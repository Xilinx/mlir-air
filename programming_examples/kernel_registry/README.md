<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# Kernel Registry — LLM Deployment on NPU2

A documentation database of the leaf kernels used to deploy decoder-only LLMs on AMD NPU2 (Strix, AIE2P): what kernels exist, what shapes have been validated, their tile configs, and measured accuracy + performance.

This is **documentation, not executable code** — nothing here compiles or runs. It records results produced by the kernels under `programming_examples/`, measured on real NPU2 hardware.

## Scope

This registry targets **NPU2 (Strix / AIE2P) only** — all shapes, tile configs, tolerances, and measured numbers are for the aie2p target. It is built up **one kernel at a time** — only kernels we have independently understood and verified are included, so every number here can be trusted and reproduced. Currently the registry covers **GEMM only**.

| File | What it is |
|---|---|
| [`supported_kernels.md`](supported_kernels.md) | High-level index: which kernels are covered, tested shapes, and best measured performance. |
| [`details/GEMM_bf16.md`](details/GEMM_bf16.md) | Full detail for the BF16 GEMM kernel: numerical datapath, tunable parameters, tolerances, per-path data, reproduce commands. |

## Where the data comes from

Each kernel × shape row is filled from a **standalone harness** — a self-contained run that compiles the kernel to an ELF, runs it on NPU2, and compares against a CPU reference. GEMM uses the top-level `programming_examples/matrix_multiplication/bf16` example.

**Reference precision.** The GEMM reference is a **CPU FP32-accumulate** dot product (bf16 inputs upcast to f32, summed in f32, cast back to the output dtype), with `randn/sqrt(K)`-normalized inputs. This matches how a standard GPU BF16 GEMM is verified: the accumulator is FP32 on CPU, GPU, and NPU alike, so comparing against an FP32 reference isolates each device's quantization error rather than penalizing bf16-vs-fp32 noise.

## Methodology notes

- **Accuracy is independent of tile / herd choice** — it is set only by the data type and accumulation precision. Tile and herd are pure performance knobs. (The detail page shows the same shape at different tiles giving bit-identical accuracy.)
- **BF16 GEMM error grows with K** — measure at the real reduction dimension, not just a small smoke shape.
- The robust accuracy metric is `mean_rel_L1 = mean|out−ref| / mean|ref|`; per-element relative error blows up where the reference is near zero and is not a meaningful failure signal on its own.

## Roadmap (kernels not yet in this registry)

The other LLM leaf kernels are being verified and will be added in subsequent contributions, each as a `details/<KERNEL>.md` page plus a section in `supported_kernels.md`:

| Kernel | Role | Status |
|---|---|---|
| GEMV | decode-time projections (batch=1) | not yet |
| RMSNorm | per-row normalization | not yet |
| RoPE | rotary positional encoding | not yet |
| FlashAttention | causal attention | not yet |
| SiLU + Mul | SwiGLU activation | not yet |
| Eltwise Add | residual adds | not yet |
