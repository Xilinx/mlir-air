// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rope.cc"
extern "C" void olmo_rope_row(bf16 *out, bf16 *in, bf16 *lut) {
  aie_round_nearest_even();
  for (int h = 0; h < 16; ++h)
    apply_rope(out + h * 128, in + h * 128, lut, lut + 64);
}
