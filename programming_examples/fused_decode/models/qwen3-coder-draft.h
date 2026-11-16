// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
///@file qwen3-coder-draft.h
///@brief The z-lab Qwen3-Coder-30B-A3B DFlash drafter: dense Qwen3, q width
/// 4096 over a model dim of 2048; its 4 KV heads each stored twice.
#ifndef __QWEN3_CODER_DRAFT_H__
#define __QWEN3_CODER_DRAFT_H__

#if MODEL_TYPE == QWEN3_CODER_DRAFT
constexpr int MODEL_DIM = 2048;
constexpr int NUM_ATTN_HEADS = 32;
constexpr int NUM_KV_HEADS = 8;
constexpr int INTERMEDIATE_SIZE = 6144;
constexpr int GLU_SLICE = 1024;
constexpr int DH = 128;
constexpr int VOCAB_SIZE = 151936;
constexpr float ATTN_SCALE = 0.08838834764831843f; // 1/sqrt(128)

#define ATTN_IMPL ATTN_IMPL_2x4x1
#define A_FUNC A_SILU
#define HAS_QK_NORM

constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif

#endif // __QWEN3_CODER_DRAFT_H__
