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
#define LLAMA_3_2_3B 1
#define LLAMA_3_1_8B 2
#define DEEPSEEK_R1_8B 3
#define QWEN3_0_6B 4
#define QWEN3_1_7B 5
#define QWEN3_4B 6
#define QWEN3_8B 7
#define GEMMA3_4B 8
#define PHI4_4B 9
#define QWEN2_5_7B 10
#define NANBEIGE_3B 11

#ifndef MODEL_TYPE
#define MODEL_TYPE NANBEIGE_3B
#endif

#include "../models/llama3.2-1b.h"
#include "../models/llama3.2-3b.h"
#include "../models/nanbeige-3b.h"
#include "../models/phi4-4b.h"
#include "../models/qwen2.5-7b.h"
#include "../models/qwen3-0.6b.h"
#include "../models/qwen3-1.7b.h"
#include "../models/qwen3-4b.h"
#include "../models/qwen3-8b.h"

#endif // __ALL_MODELS_H__