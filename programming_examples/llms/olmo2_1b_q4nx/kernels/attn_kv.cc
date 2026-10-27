// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
extern "C" __attribute__((always_inline)) void
attn_kv_blk(bf16 *scores, bf16 *v, float *y, float *denom, int blk,
            int length) {
  aie_round_nearest_even();
  if (blk == 0) {
    for (int i = 0; i < 512; i += 16)
      aie::store_v(y + i, aie::zeros<float, 16>());
    aie::store_v(denom, aie::zeros<float, 16>());
  }
  int rem = length - blk * 16;
  if (rem <= 0)
    return;
  if (rem > 16)
    rem = 16;
  float *correction = reinterpret_cast<float *>(scores + 64);
  for (int h = 0; h < 4; ++h) {
    auto sf = aie::mul(aie::load_v<16>(scores + h * 16), (bf16)1.0f)
                  .to_vector<float>();
    denom[h] = denom[h] * correction[h] + aie::reduce_add(sf);
    for (int d = 0; d < 128; d += 16) {
      aie::accum<accfloat, 16> acc =
          aie::mul(aie::load_v<16>(y + h * 128 + d), correction[h]);
      for (int t = 0; t < rem; ++t)
        acc = aie::mac(acc, aie::load_v<16>(v + t * 512 + h * 128 + d),
                       scores[h * 16 + t]);
      aie::store_v(y + h * 128 + d, acc.to_vector<float>());
    }
  }
}
extern "C" void attn_kv_fin(float *y, float *denom, bf16 *out) {
  aie_round_nearest_even();
  for (int h = 0; h < 4; ++h) {
    float inv = 1.0f / denom[h];
    for (int d = 0; d < 128; d += 16) {
      auto val =
          aie::mul(aie::load_v<16>(y + h * 128 + d), inv).to_vector<bf16>();
      aie::store_v(out + h * 128 + d, val);
    }
  }
}
