// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Reuses candidate head-wise norm/RoPE implementations in full-input prefill.
#include "rope.cc"
extern "C" void openelm_qk_row(bf16 *out, bf16 *in, bf16 *params, int heads) {
  aie_round_nearest_even();
  bf16 *lut = params;
  bf16 *weight = params + 64;
  for (int h = 0; h < heads; ++h) {
    qk_norm<64>(in + h * 64, weight);
    apply_rope(out + h * 64, in + h * 64, lut, lut + 32);
  }
}
