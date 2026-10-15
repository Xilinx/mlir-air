// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// BFP16 uses shared exponents across the reduction dimension of S*V.
// Zeroing scores alone does not isolate masked V: a large future value can
// change the quantization of a live value in the same eight-key group.
#define attn_kv_blk attn_kv_blk_unmasked
#include "fused_decode/kernels/attn_kv.cc"
#undef attn_kv_blk

extern "C" __attribute__((always_inline))
void attn_kv_blk(bf16 *__restrict scores, bf16 *__restrict values,
                 float *__restrict output, float *__restrict denominator,
                 int block, int length) {
  const int valid = length - block * 16;
  if (valid > 0 && valid < 16) {
    // The DMA layout is [two 8-key halves, KV dimension / 8, key, 8].
    constexpr int width = KV_HEADS_PER_CU * DH;
    const auto zero = aie::zeros<bf16, 64>();
    for (int half = 0; half < 2; ++half) {
      const int live = valid - half * 8;
      const uint64_t bits = live >= 8 ? ~uint64_t{0}
                            : live <= 0 ? uint64_t{0}
                                        : (uint64_t{1} << (live * 8)) - 1;
      const auto keep = aie::mask<64>::from_uint64(bits);
      for (int group = 0; group < width / 8; ++group) {
        bf16 *tile = values + half * width * 8 + group * 64;
        aie::store_v(tile, aie::select(zero, aie::load_v<64>(tile), keep));
      }
    }
  }
  attn_kv_blk_unmasked(scores, values, output, denominator, block, length);
}

extern "C" __attribute__((always_inline))
void attn_kv_fin_row(float *__restrict y, float *__restrict denominator,
                     bf16 *__restrict output, int row) {
  attn_kv_fin(y, denominator, output + row * Q_HEADS_PADDED_PER_CU * DH);
}

#include "attn_pair_pv.h"

#if defined(ATTN_QUERY_TILE) && ATTN_QUERY_TILE > 1
extern "C" __attribute__((always_inline))
void attn_kv_blk_tile(bf16 *__restrict scores, bf16 *__restrict values,
                      float *__restrict output, float *__restrict denominator,
                      int block, int length, int causal) {
#if defined(ATTN_PAIR_PV) && ATTN_PAIR_PV
  // A shared V operand is safe only when all queries use the full key tile.
  // Causal boundary tiles retain separate in-place masks in descending order.
  if (length - block * 16 >= 16) {
    dflash_pv_full_tile(scores, values, output, denominator, block);
    return;
  }
#endif
  // attn_kv_blk clears a query's masked values IN PLACE before BFP16 packing.
  // Visit the least restrictive query first. Descending masks then only clear
  // additional values, so every query sees exactly its original operand words.
  #pragma clang loop unroll(disable)
  for (int local = ATTN_QUERY_TILE - 1; local >= 0; --local)
    attn_kv_blk(scores + local * SSZ_BLK, values,
                 output + local * Q_HEADS_PADDED_PER_CU * DH,
                 denominator + local * 16, block, length + local * causal);
}

extern "C" __attribute__((always_inline))
void attn_kv_fin_tile(float *__restrict y, float *__restrict denominator,
                      bf16 *__restrict output, int first_row) {
  #pragma clang loop unroll(disable)
  for (int local = 0; local < ATTN_QUERY_TILE; ++local)
    attn_kv_fin(y + local * Q_HEADS_PADDED_PER_CU * DH,
                denominator + local * 16,
                output + (first_row + local) * Q_HEADS_PADDED_PER_CU * DH);
}
#endif
