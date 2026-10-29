// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#ifndef OLMO2_1B_H
#define OLMO2_1B_H
#if MODEL_TYPE == OLMO2_1B
constexpr int MODEL_DIM = 2048;
constexpr int NUM_ATTN_HEADS = 16;
constexpr int NUM_KV_HEADS = 16;
constexpr int INTERMEDIATE_SIZE = 8192;
constexpr int GLU_SLICE = 1024;
constexpr int DH = 128;
constexpr int VOCAB_SIZE = 100352;
constexpr float ATTN_SCALE = 0.08838834764831843f;
#define RMS_NORM_EPS 1e-6f
#define HAS_QK_NORM
#define FULL_QK_NORM
#define ATTN_IMPL ATTN_IMPL_4x1x1
#define A_FUNC A_SILU
constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif
#endif
