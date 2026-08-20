// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
///@file phi4-4b.h
///@brief Model parameters for Phi-4-mini-instruct, matching FastFlowLM's
///       generic decoding layer for PHI4_4B (hidden=3072, 24 heads / 8 kv,
///       head_dim=128, intermediate=8192, vocab=200064). Attention topology and
///       proj/FFN block counts are identical to llama3.2-3b; what differs is
///       the vocab and the PARTIAL rotary.
///
///       PARTIAL ROTARY: Phi-4 sets partial_rotary_factor=0.75, so RoPE covers
///       only the leading 96 of the 128 head dims and the trailing 32 pass
///       through untouched (FastFlowLM ships a whole separate kernel,
///       kernels/rope_phi4.cc, for this). Here it is PARTIAL_ROPE_DIM, which
///       model_spec.h turns into ROPE_DIM and rope.cc's apply_rope honours; the
///       rope weight buffer is correspondingly ROPE_DIM (96), not DH.
///
///       LongRoPE: the bundle carries rope.short/long.weight[48] factor tables.
///       short_factor is all 1.0, so at contexts within
///       original_max_position_embeddings=4096 the frequencies reduce to plain
///       inv_freq(theta=1e4, rope_dim=96) -- which is exactly FLM's phi4_rope
///       short half. The host builds the LUT; the kernel is factor-agnostic.
#ifndef __PHI4_4B_H__
#define __PHI4_4B_H__

#if MODEL_TYPE == PHI4_4B
constexpr int MODEL_DIM = 3072;
constexpr int NUM_ATTN_HEADS = 24;
constexpr int NUM_KV_HEADS = 8;
constexpr int INTERMEDIATE_SIZE = 8192;
constexpr int GLU_SLICE = 1024;
constexpr int DH = 128;
constexpr int VOCAB_SIZE = 200064;
constexpr float ATTN_SCALE = 0.08838834764831843f; // 1/sqrt(128)

#define ATTN_IMPL ATTN_IMPL_2x4x1 // 8 kv heads (same as llama 1B/3B/8B)
#define A_FUNC A_SILU

// RoPE covers dims [0,96); [96,128) is copied through (partial_rotary_factor).
#define PARTIAL_ROPE_DIM 96

constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif

#endif // __PHI4_4B_H__
