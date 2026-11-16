// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fused_decode_gdn's attention value kernels, attn_kv_blk with the masked
// keys' values zeroed first, and the unit output back in natural head order.

#include "attn_kv.cc"

extern "C" {

// attn_kv_blk, the masked keys' values zeroed first. The probabilities x V
// product runs in BFP16 (AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16), whose
// blocks of 8 keys share an exponent: a masked key's probability is 0, but
// its value still sets that exponent and so the rounding of the others.
// Zeroed, keys past L (a cache's stale rows, or a two-token verify's token 1
// for its token 0) and before a window change nothing. v_block is
// [key half][DH / 8][8 keys][8], as these models' memtiles lay it out.
void attn_kv_blk_z(bf16 *__restrict s_block, bf16 *__restrict v_block,
                   float *__restrict y_state, float *__restrict l_state,
                   int blk, int L) {
  const int lo = attn_window_lo(L) - blk * 16;
  const int hi = (L & ATTN_RTP_L_MASK) - blk * 16;
  if (hi > 0 && (hi < 16 || lo > 0)) {
    const aie::vector<bf16, 8> zero = aie::zeros<bf16, 8>();
    for (int k = 0; k < 16; k++) {
      if (k >= lo && k < hi)
        continue;
      bf16 *p = v_block + (k / 8) * (DH / 8) * 64 + (k % 8) * 8;
      for (int d8 = 0; d8 < DH / 8; d8++)
        aie::store_v(p + d8 * 64, zero);
    }
  }
  attn_kv_blk(s_block, v_block, y_state, l_state, blk, L);
}

// o [DH / 8][unit's heads][8] -> on [unit's heads][DH]
void o_natural(bf16 *__restrict on, bf16 *__restrict o) {
  constexpr int QH = Q_HEADS_PER_GROUP;
  for (int h = 0; h < QH; h++)
    for (int dc = 0; dc < DH / 8; dc++)
      aie::store_v(on + h * DH + dc * 8,
                   aie::load_v<8>(o + dc * QH * 8 + h * 8));
}

} // extern "C"
