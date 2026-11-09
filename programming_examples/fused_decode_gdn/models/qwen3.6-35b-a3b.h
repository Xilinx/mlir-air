// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
///@file qwen3.6-35b-a3b.h
///@brief Qwen3.6-35B-A3B (text): 30 Gated DeltaNet + 10 gated full-attention
/// layers (layer % 4 == 3), hidden 2048, MoE FFN. Its decode is
/// llms/qwen36_35b_a3b_q4nx; this header serves the attention and rope
/// kernels it shares with this engine.
#ifndef __QWEN3_6_35B_A3B_H__
#define __QWEN3_6_35B_A3B_H__

#if MODEL_TYPE == QWEN3_6_35B_A3B
constexpr int MODEL_DIM = 2048;
// full attention: 16 query heads, 2 KV heads, head dim 256 (1 KV head per CU,
// 8 query heads per CU), q/k RMSNorm, RoPE over the leading 64 dims
constexpr int NUM_ATTN_HEADS = 16;
constexpr int NUM_KV_HEADS = 2;
constexpr int DH = 256;
#define PARTIAL_ROPE_DIM 64
#define HAS_QK_NORM
#define ATTN_IMPL ATTN_IMPL_1x8x1
// the gate multiplies o in its own core, not in attn_kv
#define ATTN_NO_CONV_O_PASS
constexpr float ATTN_SCALE = 0.0625f; // 1/sqrt(256)
// the shared expert's width
constexpr int INTERMEDIATE_SIZE = 512;
constexpr int GLU_SLICE = 1024;
#define A_FUNC A_SILU
constexpr int VOCAB_SIZE = 248320;
// Gated DeltaNet: 16 key heads, 32 value heads, key / value dim 128
constexpr int GDN_HEADS = 16;
constexpr int GDN_DK = 128;
constexpr int GDN_DV = 128;

constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif

#endif // __QWEN3_6_35B_A3B_H__
