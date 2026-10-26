// SmolLM3 geometry, using the existing AIR Q4NX mapping.
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#ifndef SMOLLM3_3B_H
#define SMOLLM3_3B_H
#if MODEL_TYPE == SMOLLM3_3B
constexpr int MODEL_DIM = 2048;
constexpr int NUM_ATTN_HEADS = 16;
constexpr int NUM_KV_HEADS = 4;
constexpr int INTERMEDIATE_SIZE = 11264; // logical 11008 plus 256 zero lanes
constexpr int GLU_SLICE = 1024;
constexpr int DH = 128;
constexpr int VOCAB_SIZE = 128256;
constexpr float ATTN_SCALE = 0.08838834764831843f;
#define RMS_NORM_EPS 1e-6f
#define ATTN_IMPL ATTN_IMPL_1x4x1
#define A_FUNC A_SILU
constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif
#endif
