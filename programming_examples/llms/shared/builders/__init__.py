# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Architecture-orthogonal assembly toolkit: multi-launch ELF builders that
# stitch GEMM / RMSNorm / RoPE / attention sub-kernels into a single ELF for one
# XRT invocation. Parameterized purely by shapes (seq_len, emb_dim, kv_dim, ...);
# any model (llama32_1b, smollm2, future qwen3/MoE) assembles its own transformer
# block from these. The low-level text-stitching primitives live in
# shared.infra.stitching; gemm_builder.py is the per-shape GEMM adapter
# over matrix_multiplication + kernel_registry.
