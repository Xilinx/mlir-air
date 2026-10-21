// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Included after the masked-V wrapper and the PV implementation it wraps.
static_assert(KV_HEADS_PER_CU == 2 && Q_HEADS_PER_GROUP == 4 &&
              Q_HEADS_PADDED_PER_CU == 8 && DH == 128);
static_assert(ATTN_QUERY_TILE >= 2 && ATTN_QUERY_TILE % 2 == 0);

#include "attn_reuse_pv.h"

static inline __attribute__((always_inline)) void
dflash_pv_full_tile(bf16 *__restrict scores, bf16 *__restrict values,
                    float *__restrict output, float *__restrict denominator,
                    int block) {
  aie_round_nearest_even();
  dflash_pv_reuse_v(scores, values, output, denominator, block);
}
