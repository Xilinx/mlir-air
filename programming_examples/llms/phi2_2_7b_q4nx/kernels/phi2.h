// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once
// Select local geometry without changing the frozen shared model catalog.
#define __ALL_MODELS_H__
#define ATTN_IMPL_2x4x1 0
#define ATTN_IMPL_1x4x1 1
#define ATTN_IMPL_1x8x1 2
#define ATTN_IMPL_4x1x1 3
#define ATTN_IMPL ATTN_IMPL_4x1x1
#define A_SILU 0
#define A_GELU 1
#define A_FUNC A_GELU
#define PARTIAL_ROPE_DIM 32
#define ATTN_SCALE_NOT_INV_SQRT_DH
#define RMS_NORM_EPS 1e-5f
constexpr int MODEL_DIM = 3072;
constexpr int NUM_ATTN_HEADS = 32;
constexpr int NUM_KV_HEADS = 32;
constexpr int DH = 128;
constexpr int INTERMEDIATE_SIZE = 10752;
constexpr int GLU_SLICE = 512;
constexpr int VOCAB_SIZE = 61440;
constexpr float ATTN_SCALE = 0.11180339887498948f;
constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
