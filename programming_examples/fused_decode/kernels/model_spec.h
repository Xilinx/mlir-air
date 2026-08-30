// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
#ifndef __MODEL_SPEC__
#define __MODEL_SPEC__

#include "../models/all_models.h"
#include "typedef.hpp"

constexpr int Q_HEADS_PER_GROUP = NUM_ATTN_HEADS / NUM_KV_HEADS;

// Rotary width. Every Llama/Qwen/Gemma here ropes the whole head, so ROPE_DIM
// defaults to DH. Phi-4 ropes only the leading 3/4 (config
// partial_rotary_factor=0.75 -> 96 of 128) and passes the tail through; its
// model header sets ROPE_DIM and the rope kernel takes the partial path.
#ifdef PARTIAL_ROPE_DIM
constexpr int ROPE_DIM = PARTIAL_ROPE_DIM;
#else
constexpr int ROPE_DIM = DH;
#endif
static_assert(ROPE_DIM % 32 == 0 && ROPE_DIM <= DH,
              "ROPE_DIM must be a multiple of 32 (half must vectorize by 16) "
              "and cannot exceed the head dim");
// The qk-norm / qkv-bias slabs are addressed at rope_w+DH, which only coincides
// with "just past the cos/sin LUT" while the LUT is DH wide. No model needs
// both today; this fires if one ever does, instead of silently reading the LUT.
#if defined(HAS_QK_NORM) || defined(HAS_QKV_BIAS)
static_assert(
    ROPE_DIM == DH,
    "partial rotary + qk-norm/qkv-bias: the rope_w+DH slab offsets "
    "assume a DH-wide cos/sin LUT; make them ROPE_DIM-relative first");
#endif

// Attention scale. Every model here sets ATTN_SCALE to 1/sqrt(DH), but the two
// are different quantities: Gemma scales by 1/sqrt(query_pre_attn_scalar), which
// equals the head dim in the configs carried here and need not in general
// (gemma-3-27b: 168 against a head dim of 128). So the model header stays
// authoritative and this only catches a value copied from a neighbour with a
// different head dim -- which has no runtime symptom, because a wrong scale
// still produces plausible logits. Written squared to keep it a constant
// expression; sqrt is not constexpr. A model that genuinely scales by something
// else defines ATTN_SCALE_NOT_INV_SQRT_DH.
#ifndef ATTN_SCALE_NOT_INV_SQRT_DH
static_assert(ATTN_SCALE * ATTN_SCALE * DH > 0.999f &&
                  ATTN_SCALE * ATTN_SCALE * DH < 1.001f,
              "ATTN_SCALE is not 1/sqrt(DH) for this model; if that is "
              "deliberate (a Gemma-style query_pre_attn_scalar differing from "
              "the head dim), define ATTN_SCALE_NOT_INV_SQRT_DH");
#endif

constexpr int DQ = NUM_ATTN_HEADS * DH;
constexpr int DK = NUM_KV_HEADS * DH;
constexpr int DV = DK;

// Analyze the GQA patterns. 2x4x1 packs 2 kv heads per CU; 1x8x1 / 1x4x1 use 1
// kv head per CU (so the CU count equals the kv-head count).
#if ATTN_IMPL == ATTN_IMPL_2x4x1
constexpr int NUM_MHA_CU = NUM_KV_HEADS / 2;
#else
constexpr int NUM_MHA_CU = NUM_KV_HEADS;
#endif
constexpr int KV_HEADS_PER_CU = NUM_KV_HEADS / NUM_MHA_CU;
constexpr int Q_HEADS_PER_CU = NUM_ATTN_HEADS / NUM_MHA_CU;

#if ATTN_IMPL == ATTN_IMPL_2x4x1
const int GQA_R = 8;
const int GQA_S = 8;
const int GQA_T = 8;
constexpr int GQA_SEGMENT_SIZE = 4; // 8x8x8 or 4x8x8, must be multiple of 4
constexpr int ATTN_GROUPS_PADDING =
    ((Q_HEADS_PER_GROUP + GQA_SEGMENT_SIZE - 1) / GQA_SEGMENT_SIZE) *
        GQA_SEGMENT_SIZE -
    Q_HEADS_PER_GROUP;
#elif ATTN_IMPL == ATTN_IMPL_1x4x1
const int GQA_R = 4;
const int GQA_S = 8;
const int GQA_T = 8;
constexpr int GQA_SEGMENT_SIZE = 4; // 4x8x8, must be multiple of 4
constexpr int ATTN_GROUPS_PADDING =
    ((Q_HEADS_PER_GROUP + GQA_SEGMENT_SIZE - 1) / GQA_SEGMENT_SIZE) *
        GQA_SEGMENT_SIZE -
    Q_HEADS_PER_GROUP;
#elif ATTN_IMPL == ATTN_IMPL_1x8x1
const int GQA_R = 8;
const int GQA_S = 8;
const int GQA_T = 8;
constexpr int GQA_SEGMENT_SIZE = 8; // 8x8x8, must be multiple of 8
constexpr int ATTN_GROUPS_PADDING =
    ((Q_HEADS_PER_GROUP + GQA_SEGMENT_SIZE - 1) / GQA_SEGMENT_SIZE) *
        GQA_SEGMENT_SIZE -
    Q_HEADS_PER_GROUP;
#endif

constexpr int Q_HEADS_PER_GROUP_PADDED =
    Q_HEADS_PER_GROUP + ATTN_GROUPS_PADDING;

constexpr int Q_HEADS_PADDED_PER_CU =
    KV_HEADS_PER_CU * Q_HEADS_PER_GROUP_PADDED;

constexpr int VOCAB_SIZE_PADDED =
    (VOCAB_SIZE + MODEL_DIM - 1) / MODEL_DIM * MODEL_DIM;

///@brief Q4K block
///@param scales: scales and mins, bf16
///@param mins: scales and mins, bf16
///@param qs: 4--bit quants (unsigned for Q4NX, signed for Q4_0)
#ifndef Q4_0
typedef struct {
  bf16 scales[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE /
              Q4NX_GROUP_SIZE]; // scales and mins, quantized with 6 bits
  bf16 mins[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE /
            Q4NX_GROUP_SIZE]; // scales and mins, quantized with 6 bits
  uint4 qs[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE]; // 4--bit quants
} q4k_block_t;
#else
typedef struct {
  bf16 scales[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE / Q4NX_GROUP_SIZE];
  bf16 mins[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE / Q4NX_GROUP_SIZE];
  int4 qs[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE]; // signed 4-bit quants
} q4k_block_t;
#endif

#endif
