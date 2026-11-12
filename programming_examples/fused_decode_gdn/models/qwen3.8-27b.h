// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
///@file qwen3.8-27b.h
///@brief Qwen3.8-27B (text): 48 Gated DeltaNet + 16 gated full-attention
/// layers (layer % 4 == 3), hidden 5120, dense FFN. Its decode is
/// llms/qwen38_27b_q4nx; this header serves the attention and rope kernels it
/// shares with this engine.
#ifndef __QWEN3_8_27B_H__
#define __QWEN3_8_27B_H__

#if MODEL_TYPE == QWEN3_8_27B
constexpr int MODEL_DIM = 5120;
// full attention: 24 query heads, 4 KV heads, head dim 256. The kernels take
// one KV head and 8 query heads at a time (1x8x1): each KV head's 6 query
// heads are padded with 2 zero heads, and an attention unit runs two KV heads
// in turn. q/k RMSNorm, RoPE over the leading 64 dims
constexpr int NUM_ATTN_HEADS = 24;
constexpr int NUM_KV_HEADS = 4;
constexpr int DH = 256;
#define PARTIAL_ROPE_DIM 64
#define HAS_QK_NORM
#define ATTN_IMPL ATTN_IMPL_1x8x1
// the gate multiplies o in its own core, not in attn_kv
#define ATTN_NO_CONV_O_PASS
constexpr float ATTN_SCALE = 0.0625f; // 1/sqrt(256)
constexpr int INTERMEDIATE_SIZE = 17408;
constexpr int GLU_SLICE = 1024;
#define A_FUNC A_SILU
constexpr int VOCAB_SIZE = 248320;
// Gated DeltaNet: 16 key heads, 48 value heads, key / value dim 128
constexpr int GDN_HEADS = 16;
constexpr int GDN_DK = 128;
constexpr int GDN_DV = 128;

constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif

#endif // __QWEN3_8_27B_H__
