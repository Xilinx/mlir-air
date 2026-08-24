// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
///@file qwen3-4b.h
///@brief Define the model parameters for the qwen3-4b model
///
/// The DFlash target model (docs/DFlashFeasibility.md). Qwen3-8B's header with
/// two numbers changed: MODEL_DIM 4096 -> 2560 and INTERMEDIATE_SIZE
/// 12288 -> 9728. Head counts, head dim, vocab and the attention implementation
/// are identical, which is why the attention kernels cost the same on both --
/// attn_qk_blk depends on DH and the head counts and never on MODEL_DIM.
///
/// Taken from llms/qwen3_4b/qwen3_4b_weights.py's LlamaConfig, the repo's
/// existing bf16 Qwen3-4B, rather than from a datasheet. The Python side of
/// this model already exists as fused_decode.py's `qwen3-4b` _MODELS entry;
/// this is the kernel-side half, without which nothing compiles for it.
///
/// NOTE the decoupled q dim: NUM_ATTN_HEADS*DH = 4096 != MODEL_DIM, so o_proj
/// contracts 4096 -> 2560. model_spec.h derives DQ from the head counts and
/// gets this right on its own; it is called out because 8B is square here and
/// 4B is not.
#ifndef __QWEN3_4B_H__
#define __QWEN3_4B_H__

#if MODEL_TYPE == QWEN3_4B
constexpr int MODEL_DIM = 2560;
constexpr int NUM_ATTN_HEADS = 32;
constexpr int NUM_KV_HEADS = 8;
constexpr int INTERMEDIATE_SIZE = 9728;
constexpr int GLU_SLICE = 1024;
constexpr int DH = 128;
constexpr int VOCAB_SIZE = 151936;
constexpr float ATTN_SCALE = 0.08838834764831843f; // 1/sqrt(128)

#define ATTN_IMPL ATTN_IMPL_2x4x1
#define A_FUNC A_SILU
#define HAS_QK_NORM

// Q4NX (unsigned int4), as Qwen3-8B -- not the signed Q4_0 path qwen2.5-3b
// uses.
constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif

#endif // __QWEN3_4B_H__
