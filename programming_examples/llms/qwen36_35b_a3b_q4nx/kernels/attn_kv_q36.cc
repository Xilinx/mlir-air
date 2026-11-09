// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fused_decode_gdn's attention value kernels, and the unit output back in
// natural head order.

#include "attn_kv.cc"

extern "C" {

// o [256 / 8][8 heads][8] -> on [8 heads][256]
void o_natural(bf16 *__restrict on, bf16 *__restrict o) {
  for (int h = 0; h < 8; h++)
    for (int dc = 0; dc < 32; dc++)
      aie::store_v(on + h * 256 + dc * 8, aie::load_v<8>(o + dc * 64 + h * 8));
}

} // extern "C"
