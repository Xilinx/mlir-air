// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fused_decode_gdn's attention value kernels, and the unit output back in
// natural head order.

#include "attn_kv.cc"

extern "C" {

// o [DH / 8][unit's heads][8] -> on [unit's heads][DH]
void o_natural(bf16 *__restrict on, bf16 *__restrict o) {
  constexpr int QH = Q_HEADS_PER_GROUP;
  for (int h = 0; h < QH; h++)
    for (int dc = 0; dc < DH / 8; dc++)
      aie::store_v(on + h * DH + dc * 8,
                   aie::load_v<8>(o + dc * QH * 8 + h * 8));
}

} // extern "C"
