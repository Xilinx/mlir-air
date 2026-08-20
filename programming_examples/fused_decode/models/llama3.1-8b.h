// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
///@file llama3.1-8b.h
///@brief Model parameters for llama3.1-8b, matching FastFlowLM's generic
///       decoding layer for LLAMA_3_1_8B (hidden=4096, 32 heads / 8 kv,
///       head_dim=128, intermediate=14336, vocab=128256). Same attention
///       topology as 3B (8 CUs, 2x4x1, 8 kv heads, DH=128); only the proj/FFN
///       widths grow, so the per-CU KV geometry is unchanged (2*128 = 256).
///       Unlike 1B/3B this model does NOT tie the LM head to the embedding.
#ifndef __LLAMA3_1_8B_H__
#define __LLAMA3_1_8B_H__

#if MODEL_TYPE == LLAMA_3_1_8B
constexpr int MODEL_DIM = 4096;
constexpr int NUM_ATTN_HEADS = 32;
constexpr int NUM_KV_HEADS = 8;
constexpr int INTERMEDIATE_SIZE = 14336;
constexpr int GLU_SLICE = 1024;
constexpr int DH = 128;
constexpr int VOCAB_SIZE = 128256;
constexpr float ATTN_SCALE = 0.08838834764831843f; // 1/sqrt(128)

#define ATTN_IMPL ATTN_IMPL_2x4x1 // 8 kv heads (same as 1B/3B)
#define A_FUNC A_SILU

constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif

#endif // __LLAMA3_1_8B_H__
