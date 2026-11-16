// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The RoPE core's kernels (Qwen3.6's rope_rounds.cc) for two tokens: [2][n]
// buffers, token 1 after token 0; a two-token round is [column][core][token]
// [32 rows].

#include "rope_rounds.cc"

extern "C" {

// qkv[t][512 j ..] = round j's rows of token t, both tokens; qkv is [2][n]
void rope_round2(bf16 *__restrict qkv, bf16 *__restrict rnd, int j, int n) {
  for (int t = 0; t < 2; t++)
    for (int c = 0; c < 4; c++)
      for (int r = 0; r < 4; r++)
        aie::store_v(qkv + t * n + 512 * j + 128 * c + 32 * r,
                     aie::load_v<32>(rnd + ((c * 4 + r) * 2 + t) * 32));
}

// a round that is not kept (the attention layer's zero padding rows)
void round_drop(bf16 *__restrict rnd) { (void)rnd; }

// rope_compute of token t: k, v [2][NUM_KV_HEADS * DH], qkv [2][n], rope_w
// [2][3 DH]; q one token's
void rope_compute_t(bf16 *__restrict q, bf16 *__restrict k, bf16 *__restrict v,
                    bf16 *__restrict qkv, bf16 *__restrict rope_w, int n,
                    int t) {
  constexpr int DKV = NUM_KV_HEADS * DH;
  rope_compute(q, k + t * DKV, v + t * DKV, qkv + t * n, rope_w + t * 3 * DH,
               0);
}

// q_pack into token t's part of qp [2][n]
void q_pack_t(bf16 *__restrict qp, bf16 *__restrict q, int n, int t) {
  q_pack(qp + t * n, q);
}

} // extern "C"
