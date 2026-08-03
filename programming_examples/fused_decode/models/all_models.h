// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
#ifndef __ALL_MODELS_H__
#define __ALL_MODELS_H__

/// @brief Attention implementation
/// @note 2x4x1: for 8 kv heads
#define ATTN_IMPL_2x4x1 0
/// @brief Attention implementation
/// @note 2x4x1: for 4 kv heads
#define ATTN_IMPL_1x4x1 1
/// @brief Attention implementation
/// @note 1x8x1: for 4 kv heads
#define ATTN_IMPL_1x8x1 2

/// @brief Activation function
#define A_SILU 0
/// @brief Activation function
#define A_GELU 1

#define LLAMA_3_2_1B 0

#ifndef MODEL_TYPE
#define MODEL_TYPE LLAMA_3_2_1B
#endif

#include "../models/llama3.2-1b.h"

#endif // __ALL_MODELS_H__
