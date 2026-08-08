// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
///@file qwen2.5-3b.h
///@brief Define the model parameters for the qwen2.5-3b model
#ifndef __QWEN2_5_3B_H__
#define __QWEN2_5_3B_H__

#if MODEL_TYPE == QWEN2_5_3B
constexpr int MODEL_DIM = 2048;
constexpr int NUM_ATTN_HEADS = 16;
constexpr int NUM_KV_HEADS = 2;
constexpr int INTERMEDIATE_SIZE = 11008 + 256; // pad to a multiple of 512
constexpr int GLU_SLICE = 1024;
constexpr int DH = 128;
constexpr int VOCAB_SIZE = 151936;
constexpr float ATTN_SCALE = 0.08838834764831843f; // 1/sqrt(128)

#define ATTN_IMPL ATTN_IMPL_1x8x1
#define A_FUNC A_SILU
#define HAS_QKV_BIAS

// Qwen2.5 weights are signed-int4 (Q4_0) blocks (scale + min + int4), unlike
// the Llama unsigned-int4 (Q4NX) path.
#define Q4_0

constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif

#endif // __QWEN2_5_3B_H__
