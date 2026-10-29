// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "gelu_lut.h"
static bf16 gelu_new(bf16 x) {
  union {
    bf16 f;
    unsigned short u;
  } v;
  v.f = x;
  unsigned m = v.u & 0x7fff;
  if (m < 0x3a80)
    return (bf16)((float)x * 0.5f);
  if (m > 0x4180)
    return (v.u & 0x8000) ? (bf16)0.0f : x;
  v.u = phi_gelu_lut[m - 0x3a80 + ((v.u & 0x8000) ? 1793 : 0)];
  return v.f;
}
extern "C" void glu_aie(bf16 *y, bf16 *x, int arm, int slice) {
  aie_round_nearest_even();
  for (int i = 0; i < 256; i++) {
    int index = slice * 256 + i;
    y[i] = index < 10240 ? gelu_new((bf16)((float)x[i] + (float)x[i + 256]))
                         : (bf16)(index == 10240);
  }
}
extern "C" void phi_gelu_row(bf16 *y, bf16 *x) {
  aie_round_nearest_even();
  for (int i = 0; i < 10240; i++)
    y[i] = gelu_new(x[i]);
  for (int i = 10240; i < 10752; i++)
    y[i] = (bf16)(i == 10240);
}
extern "C" void phi_attn_affine(bf16 *y, bf16 *x) {
  aie_round_nearest_even();
  for (int i = 0; i < 4096; i += 16)
    aie::store_v(y + i, aie::load_v<16>(x + i));
  y[96] = (bf16)1.0f;
}
