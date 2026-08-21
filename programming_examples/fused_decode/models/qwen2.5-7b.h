// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
///@file qwen2.5-7b.h
///@brief Define the model parameters for the qwen2.5-7b model
#ifndef __QWEN2_5_7B_H__
#define __QWEN2_5_7B_H__

#if MODEL_TYPE == QWEN2_5_7B
constexpr int MODEL_DIM = 3584;
constexpr int NUM_ATTN_HEADS = 28;
constexpr int NUM_KV_HEADS = 4;
constexpr int INTERMEDIATE_SIZE = 18944; // already a multiple of 512
constexpr int GLU_SLICE =
    512; // one demux packet: 18944/1024 = 37 slices is odd
constexpr int DH = 128;
constexpr int VOCAB_SIZE = 152064;
constexpr float ATTN_SCALE = 0.08838834764831843f; // 1/sqrt(128)

#define ATTN_IMPL ATTN_IMPL_1x8x1
#define A_FUNC A_SILU
#define HAS_QKV_BIAS

constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif

#endif // __QWEN2_5_7B_H__
