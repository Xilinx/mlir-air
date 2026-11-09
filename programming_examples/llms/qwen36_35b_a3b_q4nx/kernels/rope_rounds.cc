// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fused_decode_gdn's rope kernel, and the copy that assembles its input from
// the GEMV array's rounds of 512 rows.

#include "rope.cc"

extern "C" {

// qkv[512 j ..] = rnd
void rope_round(bf16 *__restrict qkv, bf16 *__restrict rnd, int j) {
  for (int i = 0; i < 512; i += 32)
    aie::store_v(qkv + 512 * j + i, aie::load_v<32>(rnd + i));
}

// q [16 heads][256] -> the attention units' layout: [unit][256 / 8][8 heads][8]
void q_pack(bf16 *__restrict qp, bf16 *__restrict q) {
  for (int u = 0; u < 2; u++)
    for (int h = 0; h < 8; h++)
      for (int dc = 0; dc < 32; dc++)
        aie::store_v(qp + u * 2048 + dc * 64 + h * 8,
                     aie::load_v<8>(q + (u * 8 + h) * 256 + dc * 8));
}

} // extern "C"
