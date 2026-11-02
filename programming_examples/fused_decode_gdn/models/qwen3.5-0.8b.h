// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
///@file qwen3.5-0.8b.h
///@brief Qwen3.5-0.8B (text): 18 Gated DeltaNet + 6 gated full-attention
/// layers (layer % 4 == 3), hidden 1024. Must agree with the builder's
/// "qwen3.5-0.8b" entry in fused_decode_gdn.py.
#ifndef __QWEN3_5_08B_H__
#define __QWEN3_5_08B_H__

#if MODEL_TYPE == QWEN3_5_08B
constexpr int MODEL_DIM = 1024;
// full attention: 8 query heads, 2 KV heads, head dim 256 (1 KV head per CU,
// 4 query heads per CU), q/k RMSNorm, RoPE over the leading 64 dims
constexpr int NUM_ATTN_HEADS = 8;
constexpr int NUM_KV_HEADS = 2;
constexpr int DH = 256;
#define PARTIAL_ROPE_DIM 64
#define HAS_QK_NORM
#define ATTN_IMPL ATTN_IMPL_1x4x1
#define ATTN_LAZY_RESCALE 1
constexpr float ATTN_SCALE = 0.0625f; // 1/sqrt(256)
constexpr int INTERMEDIATE_SIZE = 3584;
constexpr int GLU_SLICE = 512;
#define A_FUNC A_SILU
// the Q8 LM head as hi / lo affine-Q4 row pairs: 2 x 248320 device rows
constexpr int VOCAB_SIZE = 497664; // padded to 486 x MODEL_DIM
// Gated DeltaNet: 16 heads, key / value dim 128, conv kernel 4
constexpr int GDN_HEADS = 16;
constexpr int GDN_DK = 128;
constexpr int GDN_DV = 128;

constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif

#endif // __QWEN3_5_08B_H__
