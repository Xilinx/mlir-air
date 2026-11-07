// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
///@file all_models.h
///@brief The fork's model list; the ATTN_IMPL and activation ids are those of
/// ../fused_decode/models.
#ifndef __ALL_MODELS_H__
#define __ALL_MODELS_H__

#define ATTN_IMPL_2x4x1 0
#define ATTN_IMPL_1x4x1 1
#define ATTN_IMPL_1x8x1 2
#define ATTN_IMPL_4x1x1 3

#define A_SILU 0
#define A_GELU 1

#define QWEN3_5_08B 16
#define QWEN3_6_35B_A3B 17

#ifndef MODEL_TYPE
#define MODEL_TYPE QWEN3_5_08B
#endif

#include "qwen3.5-0.8b.h"
#include "qwen3.6-35b-a3b.h"

/// @brief RMSNorm epsilon, added to the mean square (Qwen3.5: 1e-6).
#ifndef RMS_NORM_EPS
#define RMS_NORM_EPS 1e-6f
#endif

#endif // __ALL_MODELS_H__
