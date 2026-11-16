// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Qwen3.6's gate kernels (gate.cc) for two tokens: [2][n] buffers, token 1
// after token 0; a two-token round is [column][core][token][32 rows].

#include "gate.cc"

extern "C" {

// z[t][512 j ..] = round j of the z projection, both tokens; z is [2][n]
void z_round2(bf16 *__restrict z, bf16 *__restrict rnd, int j, int n) {
  for (int t = 0; t < 2; t++)
    for (int c = 0; c < 4; c++)
      for (int r = 0; r < 4; r++)
        for (int i = 0; i < 32; i += 16)
          aie::store_v(z + t * n + 512 * j + 128 * c + 32 * r + i,
                       aie::load_v<16>(rnd + ((c * 4 + r) * 2 + t) * 32 + i));
}

// dn_norm_rhv of token t: o is [2][n], each [4 cores][NH heads][32]
void dn_norm_rhv_t(bf16 *__restrict y, bf16 *__restrict o, bf16 *__restrict w,
                   int n, int t) {
  dn_norm_rhv(y + t * n, o + t * n, w);
}

void gate_silu_t(bf16 *__restrict y, bf16 *__restrict z, bf16 *__restrict o,
                 int n, int t) {
  gate_silu(y + t * n, z + t * n, o + t * n, n);
}

void gate_sigmoid_t(bf16 *__restrict y, bf16 *__restrict gate,
                    bf16 *__restrict o, int n, int t) {
  gate_sigmoid(y + t * n, gate + t * n, o + t * n, n);
}

// z[t] = SiLU(z[t]) * o, in place in token t of z [2][n]; o one token's
void gate_silu_tz(bf16 *z, bf16 *__restrict o, int n, int t) {
  z += t * n;
  for (int i = 0; i < n; i += 16)
    aie::store_v(z + i,
                 aie::mul(silu(aie::load_v<16>(z + i)), aie::load_v<16>(o + i))
                     .template to_vector<bf16>());
}

// z[t] = sigmoid(z[t]) * o, in place in token t of z [2][n]; o one token's
void gate_sigmoid_tz(bf16 *z, bf16 *__restrict o, int n, int t) {
  z += t * n;
  for (int i = 0; i < n; i += 16)
    aie::store_v(
        z + i, aie::mul(sigmoid(aie::load_v<16>(z + i)), aie::load_v<16>(o + i))
                   .template to_vector<bf16>());
}

// The gate core's per-token input wo = [the DeltaNet norm weight HD | the
// mixer output DQ], so a token's comes in one transfer.

// y = dn_norm_rhv(wo's output, wo's weight)
void dn_norm_wo(bf16 *__restrict y, bf16 *__restrict wo) {
  dn_norm_rhv(y, wo + HD, wo);
}

// z[t] = sigmoid(z[t]) * wo's output, in place in token t of z [2][n]
void gate_sigmoid_wo(bf16 *z, bf16 *__restrict wo, int n, int t) {
  gate_sigmoid_tz(z, wo + HD, n, t);
}

} // extern "C"
