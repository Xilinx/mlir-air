// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
///@file gpt-oss-20b.h
///@brief Define the model parameters for the gpt-oss-20b model. Only the
/// dense kernels (proj_qmm, rope, rms_residual) are built from these.
#ifndef __GPT_OSS_20B_H__
#define __GPT_OSS_20B_H__

#if MODEL_TYPE == GPT_OSS_20B
constexpr int MODEL_DIM = 2880;
constexpr int NUM_ATTN_HEADS = 64;
constexpr int NUM_KV_HEADS = 8;
constexpr int INTERMEDIATE_SIZE = 2880;
constexpr int GLU_SLICE = 1024;
constexpr int DH = 64;
constexpr int VOCAB_SIZE = 201088;
constexpr float ATTN_SCALE = 0.125f;
#define RMS_NORM_EPS 1e-5f

#define ATTN_IMPL ATTN_IMPL_2x4x1
#define A_FUNC A_SILU
#define HAS_QKV_BIAS

// FLM's GPT-OSS bundle packs the dense projections in 32x128 blocks.
constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 128;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif

#endif // __GPT_OSS_20B_H__
