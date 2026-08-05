// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
///@file gemma3-4b.h
///@brief Define the model parameters for the gemma3-4b (text) model
#ifndef __GEMMA3_4B_H__
#define __GEMMA3_4B_H__

#if MODEL_TYPE == GEMMA3_4B
constexpr int MODEL_DIM = 2560;
constexpr int NUM_ATTN_HEADS = 8;
constexpr int NUM_KV_HEADS = 4;
constexpr int INTERMEDIATE_SIZE = 10240;
constexpr int GLU_SLICE = 1024;
constexpr int DH = 256;
constexpr int VOCAB_SIZE = 262208;
constexpr float ATTN_SCALE =
    0.0625f; // query_pre_attn_scalar=256 -> 1/sqrt(256)

// 4 kv heads -> KV_HEADS_PER_CU=1 (NUM_MHA_CU=4); Q_HEADS_PER_GROUP=2 pads
// to 4.
#define ATTN_IMPL ATTN_IMPL_1x4x1
#define A_FUNC A_GELU
// qk-norm: RMSNorm(DH) on Q,K per head before RoPE (rope_w carries q/k norm w).
#define HAS_QK_NORM

constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif

#endif // __GEMMA3_4B_H__
