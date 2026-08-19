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
#define GEMMA3_4B 1
#define QWEN2_5_3B 2
#define LLAMA_3_2_3B 3
#define QWEN3_8B 4

#ifndef MODEL_TYPE
#define MODEL_TYPE LLAMA_3_2_1B
#endif

#include "../models/gemma3-4b.h"
#include "../models/llama3.2-1b.h"
#include "../models/llama3.2-3b.h"
#include "../models/qwen2.5-3b.h"
#include "../models/qwen3-8b.h"

#endif // __ALL_MODELS_H__
