// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
static void phi_rope(bf16 *y, const bf16 *x, const bf16 *lut) {
  for (int i = 0; i < 4096; i += 16)
    aie::store_v(y + i, aie::load_v<16>(x + i));
  for (int h = 0; h < 32; h++) {
    int off = h * 128 + (h % 2) * 16;
    auto a = aie::load_v<16>(x + off), b = aie::load_v<16>(x + off + 16);
    auto c = aie::load_v<16>(lut), s = aie::load_v<16>(lut + 16);
    auto ac = aie::mul(a, c).to_vector<bf16>(),
         bs = aie::mul(b, s).to_vector<bf16>();
    auto bc = aie::mul(b, c).to_vector<bf16>(),
         as = aie::mul(a, s).to_vector<bf16>();
    aie::store_v(y + off, aie::sub(ac, bs));
    aie::store_v(y + off + 16, aie::add(bc, as));
  }
}
extern "C" void rope_compute(bf16 *q, bf16 *k, bf16 *v, bf16 *qkv, bf16 *lut,
                             int arm) {
  aie_round_nearest_even();
  phi_rope(q, qkv, lut);
  phi_rope(k, qkv + 4096, lut);
  for (int i = 0; i < 4096; i += 16)
    aie::store_v(v + i, aie::load_v<16>(qkv + 8192 + i));
}
extern "C" void phi_rope_row(bf16 *y, bf16 *x, bf16 *lut) {
  aie_round_nearest_even();
  phi_rope(y, x, lut);
}
