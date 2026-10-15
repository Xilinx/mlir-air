// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "fused_decode/kernels/attn_qk.cc"

extern "C" __attribute__((always_inline))
void attn_qk_blk_row(bf16 *__restrict q, bf16 *__restrict k,
                     bf16 *__restrict maximum, float *__restrict correction,
                     bf16 *__restrict scores, int block, int length, int row) {
  attn_qk_blk(q + row * Q_HEADS_PADDED_PER_CU * DH, k, maximum, correction,
               scores, block, length);
}

#include "attn_pair_qk.h"

// One KV tile serves several queries. Each query keeps its own online-softmax
// state, and visits KV blocks in precisely the original order.
#if defined(ATTN_QUERY_TILE) && ATTN_QUERY_TILE > 1
extern "C" __attribute__((always_inline))
void attn_qk_blk_tile(bf16 *__restrict q, bf16 *__restrict k,
                      bf16 *__restrict maximum, float *__restrict correction,
                      bf16 *__restrict scores, int block, int length,
                      int first_row, int causal) {
#if defined(ATTN_PAIR_QK) && ATTN_PAIR_QK
  dflash_qk_tile(q, k, maximum, correction, scores, block, length, first_row, causal);
  return;
#endif
  #pragma clang loop unroll(disable)
  for (int local = 0; local < ATTN_QUERY_TILE; ++local)
    attn_qk_blk(q + (first_row + local) * Q_HEADS_PADDED_PER_CU * DH, k,
                maximum + local * 16, correction + local * 8,
                scores + local * SSZ_BLK, block, length + local * causal);
}
#endif
