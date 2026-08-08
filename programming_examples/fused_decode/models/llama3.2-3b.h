// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
///@file llama3.2-3b.h
///@brief Model parameters for llama3.2-3b, matching FastFlowLM's generic
///       decoding layer for LLAMA_3_2_3B (hidden=3072, 24 heads / 8 kv,
///       head_dim=128, intermediate=8192, vocab=128256). Same attention
///       topology as 1B (8 CUs, 2x4x1, 8 kv heads); the head_dim-dependent
///       widths double vs 1B (per-CU KV = 2*128 = 256, cf. FLM attn_memtile
///       k_cu_offset:256, d:512).
#ifndef __LLAMA3_2_3B_H__
#define __LLAMA3_2_3B_H__

#if MODEL_TYPE == LLAMA_3_2_3B
constexpr int MODEL_DIM = 3072;
constexpr int NUM_ATTN_HEADS = 24;
constexpr int NUM_KV_HEADS = 8;
constexpr int INTERMEDIATE_SIZE = 8192;
constexpr int GLU_SLICE = 1024;
constexpr int DH = 128;
constexpr int VOCAB_SIZE = 128256;
constexpr float ATTN_SCALE = 0.08838834764831843f; // 1/sqrt(128)

#define ATTN_IMPL ATTN_IMPL_2x4x1 // 8 kv heads (same as 1B)
#define A_FUNC A_SILU

constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif

#endif // __LLAMA3_2_3B_H__
