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
#define PHI4_4B 4
#define QWEN3_8B 5
#define LLAMA_3_1_8B 6
#define QWEN2_5_7B 7
#define LFM2_1_2B 8

#ifndef MODEL_TYPE
#define MODEL_TYPE LLAMA_3_2_1B
#endif

#include "../models/gemma3-4b.h"
#include "../models/lfm2-1.2b.h"
#include "../models/llama3.1-8b.h"
#include "../models/llama3.2-1b.h"
#include "../models/llama3.2-3b.h"
#include "../models/phi4-4b.h"
#include "../models/qwen2.5-3b.h"
#include "../models/qwen2.5-7b.h"
#include "../models/qwen3-8b.h"

/// @brief RMSNorm epsilon, added to the MEAN square.
/// @note Every model shipped before this was introduced was validated against
///       1e-6, so that stays the default and their numerics are unchanged. A
///       model only needs to set it when its activations are small enough for
///       eps to matter -- see the note in kernels/rms_residual.cc.
#ifndef RMS_NORM_EPS
#define RMS_NORM_EPS 1e-6f
#endif

#endif // __ALL_MODELS_H__
