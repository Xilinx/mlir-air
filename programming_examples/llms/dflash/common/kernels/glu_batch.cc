// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "fused_decode/kernels/glu.cc"

extern "C" {
void glu_row_aie(bf16 *__restrict y, bf16 *__restrict x, int row, int arm) {
  (void)arm;
  aie_round_nearest_even();
  pseduo_glu<GLU_SLICE>(y + row * (GLU_SLICE / 2), x + row * GLU_SLICE);
}

void glu_copy_aie(bf16 *__restrict y, bf16 *__restrict x, int offset, int size,
                  int arm) {
  (void)arm;
  aie_round_nearest_even();
  for (int j = 0; j < size; j += 16)
    aie::store_v(y + j, aie::load_v<16>(x + offset + j));
}
}
