// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Eight independent MHA heads per CU. Natural [token,head,dimension] KV.
#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "lut_based_ops.h"
extern "C" __attribute__((always_inline)) void
attn_qk_blk(bf16 *q, bf16 *k, bf16 *m, float *scratch, bf16 *scores, int blk,
            int length) {
  aie_round_nearest_even();
  int rem = length - blk * 16;
  if (rem <= 0)
    return;
  if (rem > 16)
    rem = 16;
  float *correction = reinterpret_cast<float *>(scores + 128);
  const auto indices = aie::vector<int, 16>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                            12, 13, 14, 15, 16);
  const auto mask = aie::le(indices, rem);
  for (int h = 0; h < 8; ++h) {
    alignas(32) bf16 logits[16];
    for (int t = 0; t < 16; ++t) {
      aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();
      for (int d = 0; d < 128; d += 32)
        acc = aie::mac(acc, aie::load_v<32>(q + h * 128 + d),
                       aie::load_v<32>(k + t * 1024 + h * 128 + d));
      logits[t] = (bf16)(aie::reduce_add(acc.to_vector<float>()) * ATTN_SCALE);
    }
    auto z = aie::load_v<16>(logits);
    auto masked = aie::select((bf16)-0x1.FEp127f, z, mask);
    bf16 maximum = aie::reduce_max(masked);
    if (blk != 0)
      maximum = aie::max(maximum, m[h]);
    auto shift = aie::clamp(aie::sub(z, maximum), (bf16)-87.0f, (bf16)88.0f);
    aie::accum<accfloat, 16> exp_acc = getExpBf16(shift);
    auto exponent = exp_acc.to_vector<bf16>();
    aie::store_v(scores + h * 16, aie::select((bf16)0, exponent, mask));
    if (blk == 0)
      correction[h] = 0.0f;
    else {
      auto delta =
          aie::broadcast<bf16, 16>((bf16)((float)m[h] - (float)maximum));
      delta = aie::clamp(delta, (bf16)-87.0f, (bf16)88.0f);
      aie::accum<accfloat, 16> corr_acc = getExpBf16(delta);
      correction[h] = corr_acc.to_vector<float>().get(0);
    }
    m[h] = maximum;
  }
}
