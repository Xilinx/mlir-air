// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
///@file qwen3-coder-30b-a3b.h
///@brief Qwen3-Coder-30B-A3B: 48 GQA attention layers, hidden 2048, MoE FFN.
/// Its decode is llms/qwen3_coder_30b_a3b_q4k; this header serves the
/// attention and rope kernels it shares with this engine.
#ifndef __QWEN3_CODER_30B_A3B_H__
#define __QWEN3_CODER_30B_A3B_H__

#if MODEL_TYPE == QWEN3_CODER_30B_A3B
constexpr int MODEL_DIM = 2048;
// 32 query heads, 4 KV heads, head dim 128 (1 KV head and 8 query heads per
// CU), q/k RMSNorm, RoPE over the whole head
constexpr int NUM_ATTN_HEADS = 32;
constexpr int NUM_KV_HEADS = 4;
constexpr int DH = 128;
#define HAS_QK_NORM
#define ATTN_IMPL ATTN_IMPL_1x8x1
#define ATTN_NO_CONV_O_PASS
constexpr float ATTN_SCALE = 0.08838834764831845f; // 1/sqrt(128)
constexpr int INTERMEDIATE_SIZE = 768;
#define A_FUNC A_SILU
constexpr int VOCAB_SIZE = 151936;

constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif

#endif // __QWEN3_CODER_30B_A3B_H__
