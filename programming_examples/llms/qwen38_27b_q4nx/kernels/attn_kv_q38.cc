// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Qwen3.6's attention value kernels (attn_kv_q36.cc), and attn_kv_blk with
// the masked keys' values zeroed first. The probabilities x V product runs
// in BFP16 (AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16), whose blocks of 8 keys
// share an exponent: a masked key's probability is 0, but its value still
// sets that exponent and so the rounding of the others. Zeroed, the keys
// past L (a cache's stale rows, or the verify pass's token 1 for token 0)
// change nothing.

#include "attn_kv_q36.cc"

extern "C" {

// v_block [key half][DH / 8][8 keys][8]: keys rem.. of block blk zeroed, then
// attn_kv_blk
void attn_kv_blk_z(bf16 *__restrict s_block, bf16 *__restrict v_block,
                   float *__restrict y_state, float *__restrict l_state,
                   int blk, int L) {
  const int rem = (L & ATTN_RTP_L_MASK) - blk * 16;
  if (rem > 0 && rem < 16) {
    const aie::vector<bf16, 8> zero = aie::zeros<bf16, 8>();
    for (int k = rem; k < 16; k++) {
      bf16 *p = v_block + (k / 8) * (DH / 8) * 64 + (k % 8) * 8;
      for (int d8 = 0; d8 < DH / 8; d8++)
        aie::store_v(p + d8 * 64, zero);
    }
  }
  attn_kv_blk(s_block, v_block, y_state, l_state, blk, L);
}

} // extern "C"
