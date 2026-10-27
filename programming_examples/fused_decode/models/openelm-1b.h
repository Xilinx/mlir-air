// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// OpenELM padded spatial geometry; logical geometry is retained by packing.
#ifndef OPENELM_1B_H
#define OPENELM_1B_H
#if MODEL_TYPE == OPENELM_1B
constexpr int MODEL_DIM = 2048;
constexpr int NUM_ATTN_HEADS = 32;
constexpr int NUM_KV_HEADS = 8;
constexpr int INTERMEDIATE_SIZE = 8192;
constexpr int GLU_SLICE = 1024;
constexpr int DH = 64;
constexpr int VOCAB_SIZE = 32000;
constexpr float ATTN_SCALE = 0.125f;
#define ATTN_IMPL ATTN_IMPL_2x4x1
#define A_FUNC A_SILU
#define HAS_QK_NORM
constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif
#endif
