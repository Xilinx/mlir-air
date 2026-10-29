// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once
// Target-local constants; keep the shared source catalog unchanged.
#define __ALL_MODELS_H__
#define ATTN_IMPL_2x4x1 0
#define ATTN_IMPL_1x4x1 1
#define ATTN_IMPL_1x8x1 2
#define ATTN_IMPL_4x1x1 3
#define ATTN_IMPL ATTN_IMPL_2x4x1
#define A_SILU 0
#define A_GELU 1
#define A_FUNC A_SILU
#define RMS_NORM_EPS 1e-5f
constexpr int MODEL_DIM = 4096;
constexpr int NUM_ATTN_HEADS = 32;
constexpr int NUM_KV_HEADS = 8;
constexpr int DH = 128;
constexpr int INTERMEDIATE_SIZE = 14336;
constexpr int GLU_SLICE = 1024;
constexpr int VOCAB_SIZE = 32768;
constexpr float ATTN_SCALE = 0.08838834764831843f;
constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
