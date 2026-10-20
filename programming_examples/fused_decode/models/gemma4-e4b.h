// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
///@file gemma4-e4b.h
///@brief Define the model parameters for the gemma4-e4b (text) model
#ifndef __GEMMA4_E4B_H__
#define __GEMMA4_E4B_H__

#if MODEL_TYPE == GEMMA4_E4B
constexpr int MODEL_DIM = 2560;
constexpr int NUM_ATTN_HEADS = 8;
// One kv head per attention CU.
constexpr int NUM_KV_HEADS = 2;
constexpr int INTERMEDIATE_SIZE = 10240;
constexpr int GLU_SLICE = 1024;
// Full-attention layers use head dim 512, sliding ones 256 in a 512-wide slot.
constexpr int DH = 512;
constexpr int SWA_DH = 256;
#define HAS_SWA_HEADS
constexpr int SLIDING_WINDOW = 512;
constexpr int VOCAB_SIZE = 262144;
// HF config scaling is 1.0, as on E2B.
constexpr float ATTN_SCALE = 1.0f;
constexpr float SWA_ATTN_SCALE = 1.0f;
#define ATTN_SCALE_NOT_INV_SQRT_DH

// --- per-layer embeddings; names follow gemma4-e2b.h ---
constexpr int PLI_D = 256;
constexpr float PER_LAYER_INPUT_SCALE = 0.01976423537605237f; // 2560**-0.5
constexpr float PER_LAYER_MODEL_PROJECTION_SCALE = 0.7071067811865476f;

constexpr int BF16_PROJ_M_BLOCK = 32;
constexpr int BF16_PROJ_K_BLOCK = 256;

#define ATTN_IMPL ATTN_IMPL_1x4x1
#define ATTN_LAZY_RESCALE 1
#define A_FUNC A_GELU
#define HAS_QK_NORM
#define HAS_V_NORM
#define HAS_PER_LAYER_EMBEDDING
constexpr int NUM_LAYERS = 42;
constexpr int FIRST_KV_SHARED_LAYER = 24;

constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif

#endif // __GEMMA4_E4B_H__
