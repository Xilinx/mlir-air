// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
///@file gemma4-e2b.h
///@brief Define the model parameters for the gemma4-e2b (text) model
#ifndef __GEMMA4_E2B_H__
#define __GEMMA4_E2B_H__

#if MODEL_TYPE == GEMMA4_E2B
constexpr int MODEL_DIM = 1536;
constexpr int NUM_ATTN_HEADS = 8;
constexpr int NUM_KV_HEADS = 1;
// Widest of the two per-layer FFNs. Layers <15 are half this (6144); the
// builder zero-pads them, so the kernel only ever sees the wide shape.
constexpr int INTERMEDIATE_SIZE = 12288;
constexpr int GLU_SLICE = 1024;
// TWO head dims in one model: full-attention layers (4,9,...,34) use 512,
// the sliding-window layers use 256. DH is the wide one for the same
// pad-to-max reason as INTERMEDIATE_SIZE; SWA_DH is what the sliding path
// actually rotates and attends over.
constexpr int DH = 512;
constexpr int SWA_DH = 256;
constexpr int SLIDING_WINDOW = 512;
constexpr int VOCAB_SIZE = 262144;
// 1.0, NOT 1/sqrt(DH): Gemma4 folds query_pre_attn_scalar into the q weights.
// Verified against FLM's own golden activations (their ATTN_SCALE is also 1.0).
constexpr float ATTN_SCALE = 1.0f;

// 1 kv head -> KV_HEADS_PER_CU=1 with a single CU; 8 q heads per group, so no
// GQA padding is needed (Q_HEADS_PER_GROUP=8 == GQA_SEG).
#define ATTN_IMPL ATTN_IMPL_1x8x1
#define A_FUNC A_GELU
// qk-norm: RMSNorm(DH) on Q,K per head before RoPE (rope_w carries q/k norm w).
#define HAS_QK_NORM
// Gemma4 keeps Gemma3's 4-norm sandwich and adds a FIFTH norm on the per-layer
// embedding branch (post_layernorm), applied to the PLE projection before its
// residual add.
#define HAS_PER_LAYER_EMBEDDING
constexpr int PER_LAYER_INPUT_DIM = 256;
constexpr int NUM_LAYERS = 35;
// Layers >= this index reuse a lower layer's KV cache and carry the double-wide
// FFN. The two switches are the same bit -- see the builder's LAYER_CLASS note.
constexpr int FIRST_KV_SHARED_LAYER = 15;

constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif

#endif // __GEMMA4_E2B_H__
