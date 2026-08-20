// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
///@file qwen3-8b.h
///@brief Define the model parameters for the qwen3-8b model
#ifndef __QWEN3_8B_H__
#define __QWEN3_8B_H__

#if MODEL_TYPE == QWEN3_8B
constexpr int MODEL_DIM = 4096;
constexpr int NUM_ATTN_HEADS = 32;
constexpr int NUM_KV_HEADS = 8;
constexpr int INTERMEDIATE_SIZE = 12288;
constexpr int GLU_SLICE = 1024;
constexpr int DH = 128;
constexpr int VOCAB_SIZE = 151936;
constexpr float ATTN_SCALE = 0.08838834764831843f; // 1/sqrt(128)

#define ATTN_IMPL ATTN_IMPL_2x4x1
#define A_FUNC A_SILU
#define HAS_QK_NORM

// Qwen3 weights are unsigned-int4 (Q4NX), like Llama/Gemma -- NOT the signed
// Q4_0 path qwen2.5-3b uses.

constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif

#endif // __QWEN3_8B_H__
