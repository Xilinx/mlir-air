// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Included after the safe masked-V wrapper and current PV implementation.
#if defined(ATTN_PAIR_PV) && ATTN_PAIR_PV
static_assert(KV_HEADS_PER_CU == 2 && Q_HEADS_PER_GROUP == 4 &&
              Q_HEADS_PADDED_PER_CU == 8 && DH == 128);
static_assert(ATTN_QUERY_TILE >= 2 && ATTN_QUERY_TILE % 2 == 0);

#ifdef DFLASH_FUSED_PV_RESCALE
#include "attn_optimized_pv.h"
#endif
#ifdef DFLASH_PV_REUSE
#include "attn_reuse_pv.h"
#endif

static inline __attribute__((always_inline))
void dflash_pv_full_tile(bf16 *__restrict scores, bf16 *__restrict values,
                         float *__restrict output, float *__restrict denominator,
                         int block) {
  aie_round_nearest_even();
#ifdef DFLASH_PV_REUSE
  dflash_pv_reuse_v(scores,values,output,denominator,block);
  return;
#endif
#ifdef DFLASH_FUSED_PV_RESCALE
  dflash_pv_full_tile_fused_rescale(scores,values,output,denominator,block);
  return;
#endif
  using MM = aie::mmul<8, 8, 8, bf16, bf16, accfloat>;
  constexpr int query_width = Q_HEADS_PADDED_PER_CU * DH;
  #pragma clang loop unroll(disable)
  for (int local = 0; local < ATTN_QUERY_TILE; ++local) {
    float *y = output + local * query_width;
    float *den = denominator + local * 16;
    bf16 *s = scores + local * SSZ_BLK;
    float *c = reinterpret_cast<float *>(s + 128);
    if (block == 0) {
      zero_vectorized<float, query_width>(y);
      aie::store_v(den, aie::zeros<float, 16>());
    }
    calculate_l(s, c, den);
    // Retain the original separate FP32 online-softmax rescaling pass.
    auto correction = aie::concat(
        aie::broadcast<float,8>(c[0]), aie::broadcast<float,8>(c[1]),
        aie::broadcast<float,8>(c[2]), aie::broadcast<float,8>(c[3]),
        aie::broadcast<float,8>(c[4]), aie::broadcast<float,8>(c[5]),
        aie::broadcast<float,8>(c[6]), aie::broadcast<float,8>(c[7]));
    #pragma clang loop unroll(disable)
    for (int d = 0; d < DH / 8; ++d)
      aie::store_v(y + d * 64,
          aie::mul(correction, aie::load_v<64>(y + d * 64)).template to_vector<float>());
  }
  #pragma clang loop unroll(disable)
  for (int local = 0; local < ATTN_QUERY_TILE; local += 2) {
    #pragma clang loop unroll(disable)
    for (int group = 0; group < 2; ++group) {
      auto q0 = aie::load_v<64>(scores + local * SSZ_BLK + group * 64);
      auto q1 = aie::load_v<64>(scores + (local + 1) * SSZ_BLK + group * 64);
      auto first = aie::concat(aie::filter_even(q0, 8), aie::filter_even(q1, 8));
      auto second = aie::concat(aie::filter_odd(q0, 8), aie::filter_odd(q1, 8));
      #pragma clang loop unroll(disable)
      for (int d = 0; d < DH / 8; ++d) {
        bf16 *v = values + (group * (DH / 8) + d) * 64;
        MM product(aie::zeros<bf16, 64>());
        product.mac(first, aie::load_v<64>(v));
        product.mac(second, aie::load_v<64>(v + KV_HEADS_PER_CU * DH * 8));
        float *y0 = output + local * query_width + d * 64 + group * 32;
        float *y1 = output + (local + 1) * query_width + d * 64 + group * 32;
        auto previous = aie::concat(aie::load_v<32>(y0), aie::load_v<32>(y1));
        auto sum = aie::add(product.template to_vector<float>(), previous);
        aie::store_v(y0, sum.template extract<32>(0));
        aie::store_v(y1, sum.template extract<32>(1));
      }
    }
  }
}
#endif
