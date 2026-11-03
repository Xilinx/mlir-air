// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Gemma4 q/k norm, rope and v norm for one token of the eight-token decoder.
// Every layer is computed at the full-attention width. A sliding layer's
// q/k/v heads arrive compact (its weights stream only their real rows) and are
// spread here into the full-width slots, [lo | 0 | hi | 0] for q and k and
// [v | 0] for v; its rope table rotates only the real pairs. The table row is
// [cos | sin | q_norm | k_norm | v width], the last word being the width
// v_norm averages over (256 or 512) as a float.

#include "fused_decode/kernels/rope.cc"

static_assert(ATTN_GROUPS_PADDING == 0 && Q_HEADS_PER_GROUP == 4 &&
              NUM_KV_HEADS == 2 && DH == 512);

extern "C" void rope_b8(bf16 *restrict q, bf16 *restrict k, bf16 *restrict v,
                        bf16 *restrict qkv, bf16 *restrict rope_w, int arm) {
  aie_round_nearest_even();
  (void)arm;
  const int v_width = (int)(float)rope_w[3 * DH];
  if (v_width == SWA_DH)
    spread_swa_heads(qkv);
  pseduo_rope(q, k, v, qkv, rope_w, v_width);
}
