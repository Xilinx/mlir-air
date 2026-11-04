// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The RoPE core's kernels: AIR's rope_compute, and the q hand-off to the four
// attention units -- each unit's 16 heads in the attention kernel's layout
// with their 16 sinks behind them, as FLM's RoPE core sends q (its q carries
// the sinks too). Build with -DMODEL_TYPE=GPT_OSS_20B.

#include "rope.cc"

extern "C" {

// q: 64 heads x 64, natural. rw: the RoPE table and q/k/v bias, then the 64
// sinks at ROPE_SINKS. out: per unit [half][dchunk][head][d] (1024), then 16
// sinks.
void pack_q_sinks(bf16 *restrict q, bf16 *restrict rw, bf16 *restrict out) {
  constexpr int ROPE_SINKS = DH + DQ + DK + DV;
  constexpr int UNIT = 16 * DH + 16;
  for (int c = 0; c < 4; c++) {
    bf16 *o = out + c * UNIT;
    for (int hh = 0; hh < 2; hh++)
      for (int dc = 0; dc < 8; dc++)
        for (int h = 0; h < 8; h++) {
          const bf16 *src = q + (c * 16 + hh * 8 + h) * DH + dc * 8;
          aie::store_v(o, aie::load_v<8>(src));
          o += 8;
        }
    for (int j = 0; j < 16; j++)
      o[j] = rw[ROPE_SINKS + c * 16 + j];
  }
}

} // extern "C"
