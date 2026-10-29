// Granite geometry; adapted from the Llama-3.2-1B mapping.
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#ifndef GRANITE_3_3_2B_H
#define GRANITE_3_3_2B_H
#if MODEL_TYPE == GRANITE_3_3_2B
constexpr int MODEL_DIM = 2048;
constexpr int NUM_ATTN_HEADS = 32;
constexpr int NUM_KV_HEADS = 8;
constexpr int INTERMEDIATE_SIZE = 8192;
constexpr int GLU_SLICE = 1024;
constexpr int DH = 64;
constexpr int VOCAB_SIZE = 49159;
constexpr float ATTN_SCALE = 0.015625f;
#define ATTN_SCALE_NOT_INV_SQRT_DH
#define RMS_NORM_EPS 1e-5f
#define RESIDUAL_MULTIPLIER 0.22f
#define ATTN_IMPL ATTN_IMPL_2x4x1
#define A_FUNC A_SILU
constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif
#endif
