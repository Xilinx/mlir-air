// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "fused_decode/kernels/attn_qk.cc"

#include "attn_pair_qk.h"

// One KV tile serves several queries. Each query keeps its own online-softmax
// state, and visits KV blocks in precisely the original order.
extern "C" __attribute__((always_inline)) void
attn_qk_blk_tile(bf16 *__restrict q, bf16 *__restrict k,
                 bf16 *__restrict maximum, float *__restrict correction,
                 bf16 *__restrict scores, int block, int length, int first_row,
                 int causal) {
  dflash_qk_tile(q, k, maximum, correction, scores, block, length, first_row,
                 causal);
}
