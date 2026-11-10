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

// q [KV head][8 query heads (padded)][256], as rope_compute writes it -> the
// attention kernels' layout per KV head: [KV head][256 / 8][8 heads][8]
void q_pack(bf16 *__restrict qp, bf16 *__restrict q) {
  constexpr int QH = Q_HEADS_PER_GROUP_PADDED;
  static_assert(QH == 8, "the attention kernels take 8 query heads");
  for (int g = 0; g < NUM_KV_HEADS; g++)
    for (int h = 0; h < QH; h++)
      for (int dc = 0; dc < DH / 8; dc++)
        aie::store_v(qp + g * QH * DH + dc * QH * 8 + h * 8,
                     aie::load_v<8>(q + (g * QH + h) * DH + dc * 8));
}

} // extern "C"
